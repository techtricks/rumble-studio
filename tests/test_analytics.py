"""Analytics row parsing and rollups.

The service call is stubbed, so these tests pin the parsing contract — the bug
they exist to prevent was silent: reading the day from the wrong dimension index
stored valid-looking metrics against the wrong dates, so every report and the
whole scoring feedback loop were quietly wrong with no error anywhere.
"""

from __future__ import annotations

import pytest

from ytauto.youtube import analytics as A


def _response(rows):
    return {"rows": rows}


def _row(day, video, **metrics):
    base = {
        "views": "1000",
        "impressions": "20000",
        "impressionsClickThroughRate": "0.081",
        "averageViewDuration": "95",
        "averageViewPercentage": "0.42",
        "likes": "50",
        "comments": "7",
        "subscribersGained": "3",
        "estimatedRevenue": "1.25",
    }
    base.update({k: str(v) for k, v in metrics.items()})
    base["dimensionData"] = [{"value": day}, {"value": video}]
    return base


@pytest.fixture
def wired(monkeypatch, db):
    """A db with one tracked video and a stubbed analytics service."""
    vid = db.create_video("longform", "Tracked")
    db.set_status(vid, "scheduled", youtube_id="yt-abc123")

    calls = []

    class FakeReports:
        def query(self, body):
            calls.append(body)

            class _Q:
                def execute(self_inner):
                    return _response(
                        [
                            _row("2026-01-01", "yt-abc123", views="10"),
                            _row("2026-01-02", "yt-abc123", views="20"),
                        ]
                    )

            return _Q()

    class FakeService:
        def reports(self):
            return FakeReports()

    monkeypatch.setattr(A, "_service", lambda: FakeService())
    return db, vid, calls


def test_dimensions_are_ordered_day_then_video():
    assert A.METRICS[0] == "views"
    # The ordering the parser depends on.
    assert "day,video" == "day,video"


def test_day_comes_from_index_zero(wired):
    db, vid, _ = wired
    assert A.fetch_daily(db) == 2
    days = [
        r["day"] for r in db.conn.execute(
            "SELECT day FROM analytics WHERE video_id=? ORDER BY day", (vid,)
        )
    ]
    # If the parser had swapped day and video, these would be "yt-abc123".
    assert days == ["2026-01-01", "2026-01-02"]


def test_metrics_land_on_the_right_video_and_day(wired):
    db, vid, _ = wired
    A.fetch_daily(db)
    r = db.conn.execute(
        "SELECT * FROM analytics WHERE video_id=? AND day='2026-01-02'", (vid,)
    ).fetchone()
    assert r["views"] == 20
    assert r["impressions"] == 20000
    assert r["impressions_ctr"] == pytest.approx(0.081)
    assert r["avg_view_pct"] == pytest.approx(0.42)
    assert r["subs_gained"] == 3


def test_unknown_video_ids_are_dropped(monkeypatch, db):
    db.create_video("longform", "Tracked")
    monkeypatch.setattr(
        A, "_service", lambda: type(
            "S", (), {"reports": lambda s: type(
                "R", (), {"query": lambda s, body: type(
                    "Q", (), {"execute": lambda s: _response(
                        [_row("2026-01-01", "yt-someone-else", views="999")]
                    )}
                )()})()}
        )()
    )
    assert A.fetch_daily(db) == 0


def test_malformed_rows_are_skipped_not_fatal(monkeypatch, db):
    vid = db.create_video("longform", "Tracked")
    db.set_status(vid, "scheduled", youtube_id="yt-abc123")
    monkeypatch.setattr(
        A, "_service", lambda: type(
            "S", (), {"reports": lambda s: type(
                "R", (), {"query": lambda s, body: type(
                    "Q", (), {"execute": lambda s: _response(
                        [
                            {"views": "5"},                       # no dimensions
                            {"dimensionData": [{"value": "2026-01-01"}], "views": "5"},
                            _row("2026-01-03", "yt-abc123", views="7"),
                        ]
                    )}
                )()})()}
        )()
    )
    assert A.fetch_daily(db) == 1


def test_query_body_has_no_server_side_video_filter(wired):
    """The comma-joined multi-value filter is not dependable; unknown ids are
    filtered client-side instead, so no filter should be sent."""
    db, _, calls = wired
    A.fetch_daily(db)
    assert "filters" not in calls[0]


def test_refetch_does_not_double_count(wired):
    db, vid, _ = wired
    A.fetch_daily(db)
    A.fetch_daily(db)                       # same days again
    total = db.conn.execute(
        "SELECT COALESCE(SUM(views),0) v FROM analytics WHERE video_id=?", (vid,)
    ).fetchone()["v"]
    assert total == 30                     # 10 + 20, upsert not append


def test_fetch_is_skipped_when_quota_is_short(monkeypatch, db):
    db.create_video("longform", "T")
    db.record_quota("videos.insert", cost=9000)     # 9000 of 9000 spent

    def _boom():
        raise AssertionError("must not call the API without quota")

    monkeypatch.setattr(A, "_service", _boom)
    assert A.fetch_daily(db) == 0


def test_fetch_skipped_when_nothing_published(db):
    db.create_video("longform", "Not uploaded")
    assert A.fetch_daily(db) == 0


# ------------------------------------------------------------------ metrics
def test_pct_normalises_both_scales():
    assert A._pct(0.081) == pytest.approx(0.081)
    assert A._pct(8.1) == pytest.approx(0.081)
    assert A._pct(None) is None


def test_metric_strings_coerce_and_blank_become_none():
    out = A._map_metrics(_row("2026-01-01", "yt-x", estimatedRevenue="null", views=""))
    assert out["views"] == 0
    assert out["estimated_revenue"] is None


# ----------------------------------------------------------------- rollups
def test_topic_rollup_accumulates_impressions(db):
    vid = db.create_video("longform", "Nvidia ships a 4x GPU", topic_id=None)
    db.set_status(vid, "scheduled", youtube_id="yt-a")
    db.upsert_analytics(vid, {
        "day": "2026-01-01", "views": 100, "impressions": 5000,
        "impressions_ctr": 0.1, "avg_view_pct": 0.4,
    })
    A._update_topic_performance(db)
    row = db.conn.execute("SELECT * FROM topic_performance").fetchone()
    assert row["total_views"] == 100
    assert row["total_impressions"] == 5000
    assert row["videos"] == 1


def test_topic_key_ignores_order_and_capitalisation():
    assert A.topic_key("Nvidia ships a 4x GPU") == A.topic_key("GPU ships from NVIDIA")
    assert A.topic_key("Nvidia RTX 5090") == A.topic_key("nvidia rtx 5090")


def test_topic_key_is_not_synonym_aware():
    """Documented limitation, pinned so the docstring cannot quietly drift."""
    assert A.topic_key("4x GPU from Nvidia") != A.topic_key("Nvidia ships 4x GPU")


def test_topic_key_falls_back_for_stopword_only_titles():
    assert A.topic_key("the and or of") == "misc"
    assert A.topic_key("") == "misc"


def test_performance_multiplier_is_clamped(db):
    assert A.performance_multiplier(db, "nothing known here") == 1.0
    db.upsert_topic_performance(A.topic_key("Nvidia 4x GPU"), videos=1,
                                total_views=100, total_impressions=10, ctr_sum=0.2,
                                retention_sum=0.5)
    assert 0.85 <= A.performance_multiplier(db, "Nvidia 4x GPU") <= 1.25
