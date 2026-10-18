"""Database invariants: quota ledger, state machine, leasing, scoring."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from ytauto.db import Database, SCHEMA_VERSION, iso, utcnow


# --------------------------------------------------------------- schema
def test_schema_version_recorded(db):
    assert db.get_meta("schema_version") == str(SCHEMA_VERSION)


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "m.db"
    first = Database(path)
    assert first.get_meta("schema_version") == str(SCHEMA_VERSION)
    first.close()
    # Re-opening must not re-run migrations or blow up.
    second = Database(path)
    assert second.get_meta("schema_version") == str(SCHEMA_VERSION)
    cols = [r["name"] for r in second.conn.execute("PRAGMA table_info(topics)")]
    assert "published_at" in cols


def test_future_schema_is_refused(tmp_path):
    path = tmp_path / "future.db"
    d = Database(path)
    d.set_meta("schema_version", str(SCHEMA_VERSION + 5))
    d.close()
    with pytest.raises(RuntimeError, match="Downgrade the code"):
        Database(path)


# ---------------------------------------------------------------- quota
def test_quota_ledger_tracks_spend(db, settings):
    db.record_quota("videos.list")
    db.record_quota("videos.list")
    assert db.quota_used_today() == 2
    db.record_quota("videos.insert")           # 1600
    assert db.quota_used_today() == 1602


def test_quota_respects_safety_margin(db, settings):
    # daily_quota=10000, margin 0.9 => 9000 spendable.
    assert db.quota_remaining() == 9000
    db.record_quota("videos.insert")
    assert db.quota_remaining() == 9000 - 1600


def test_quota_never_negative(db):
    db.record_quota("videos.insert", cost=99_999)
    assert db.quota_remaining() == 0


def test_can_afford_blocks_expensive_call(db):
    db.conn.execute(
        "INSERT INTO quota_ledger(day,cost,endpoint,note,at) VALUES(?,?,?,?,?)",
        (utcnow().strftime("%Y-%m-%d"), 9_000, "videos.insert", "prior", iso()),
    )
    # 9000 of the 9000-unit spendable budget is gone, so even a 1-unit call must
    # be refused. The 10% margin is a hard reserve, not headroom to spend.
    assert db.quota_remaining() == 0
    assert not db.can_afford("videos.list")
    assert not db.can_afford("videos.insert")


def test_can_afford_allows_call_that_fits(db):
    db.conn.execute(
        "INSERT INTO quota_ledger(day,cost,endpoint,note,at) VALUES(?,?,?,?,?)",
        (utcnow().strftime("%Y-%m-%d"), 1, "videos.list", "prior", iso()),
    )
    assert db.can_afford("videos.list")          # 1 + 1 <= 9000
    assert db.can_afford("videos.insert")        # 1 + 1600 <= 9000
    db.conn.execute(
        "INSERT INTO quota_ledger(day,cost,endpoint,note,at) VALUES(?,?,?,?,?)",
        (utcnow().strftime("%Y-%m-%d"), 7_400, "videos.insert", "prior", iso()),
    )
    # 7401 used leaves 1599, one unit short of the 1600-unit insert.
    assert db.quota_remaining() == 1599
    assert not db.can_afford("videos.insert")


def test_record_quota_survives_same_second_collisions(db):
    """The bug that made this suite interesting: (day, at, endpoint) was a key,
    and `at` only has second resolution, so two calls in one second crashed."""
    for _ in range(5):
        db.record_quota("videos.list")
    assert db.quota_used_today() == 5


# --------------------------------------------------------- state machine
def test_status_constraint_rejects_illegal_value(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO videos(id,kind,status,created_at,updated_at) "
            "VALUES('x','longform','teleported','now','now')"
        )


def test_kind_constraint_rejects_unknown_kind(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO videos(id,kind,status,created_at,updated_at) "
            "VALUES('x','podcast','discovered','now','now')"
        )


def test_create_video_sets_status_by_script_presence(db):
    assert db.create_video("longform", "No script") and True
    a = db.create_video("longform", "Empty")
    b = db.create_video("longform", "Written", script="words here")
    assert db.get_video(a)["status"] == "discovered"
    assert db.get_video(b)["status"] == "scripted"


def test_create_video_rejects_bad_kind(db):
    with pytest.raises(ValueError):
        db.create_video("podcast", "x")


def test_set_status_rejects_unknown_column(db):
    """Column names reach SQL as text, so an unlisted key must be refused
    outright rather than interpolated."""
    vid = db.create_video("longform", "T")
    with pytest.raises(ValueError, match="unknown column"):
        db.set_status(vid, "scripted", youtube_id="ok-but-unlisted", evil="'; DROP TABLE videos; --")


def test_set_status_allows_known_column_with_any_value(db):
    """Injection through a *permitted* column is impossible because values are
    bound parameters; this locks that in."""
    vid = db.create_video("longform", "T")
    db.set_status(vid, "scripted", youtube_id="'; DROP TABLE videos; --")
    assert db.get_video(vid)["youtube_id"] == "'; DROP TABLE videos; --"
    assert db.conn.execute("SELECT COUNT(*) c FROM videos").fetchone()["c"] == 1


def test_set_status_rejects_invalid_status(db):
    vid = db.create_video("longform", "T")
    with pytest.raises(ValueError, match="invalid status"):
        db.set_status(vid, "nonsense")


def test_tags_are_stored_as_json(db):
    vid = db.create_video("longform", "T", tags=["ai", "gpu"])
    db.set_status(vid, "scripted", tags=["ai", "gpu", "nvidia"])
    assert json.loads(db.get_video(vid)["tags"]) == ["ai", "gpu", "nvidia"]


# ---------------------------------------------------------------- leases
def test_claim_is_exclusive(db):
    vid = db.create_video("longform", "T")
    assert db.claim(vid, "worker-a") is True
    assert db.claim(vid, "worker-b") is False   # a holds the lease


def test_expired_lease_can_be_reclaimed(db):
    vid = db.create_video("longform", "T")
    db.conn.execute(
        "UPDATE videos SET lease_owner='dead', lease_until=? WHERE id=?",
        (iso(utcnow() - timedelta(minutes=5)), vid),
    )
    assert db.claim(vid, "worker-b") is True


def test_release_only_affects_owner(db):
    vid = db.create_video("longform", "T")
    db.claim(vid, "worker-a")
    db.release(vid, "worker-b")                  # wrong owner
    assert db.get_video(vid)["lease_owner"] == "worker-a"
    db.release(vid, "worker-a")
    assert db.get_video(vid)["lease_owner"] is None


# ---------------------------------------------------------------- topics
def test_add_topic_dedupes_by_url(db):
    assert db.add_topic("A", "rss", url="http://x/1") is not None
    assert db.add_topic("A", "rss", url="http://x/1") is None


def test_add_topic_dedupes_by_title_without_url(db):
    assert db.add_topic("Same Title", "rss") is not None
    assert db.add_topic("same   title", "rss") is None


def test_ingest_topics_counts_duplicates(db, sample_topic):
    from ytauto.sources.base import Topic

    t = Topic(title=sample_topic["title"], url=sample_topic["url"], source="rss")
    out = db.ingest_topics([t, t])
    assert out == {"added": 1, "duplicates": 1}


def test_freshness_uses_published_at_not_created_at(db):
    """A 3-day-old story discovered today must NOT look brand new."""
    old = (utcnow() - timedelta(days=3)).isoformat()
    tid = db.add_topic("Old news about ai chips", "rss", published_at=old)
    db.score_topics(["ai", "tech"], half_life_h=24.0)
    row = db.get_topic(tid)
    assert row["freshness"] < 0.2          # ~2^-72
    fresh_tid = db.add_topic(
        "New news about ai chips", "rss", published_at=utcnow().isoformat()
    )
    db.score_topics(["ai", "tech"], half_life_h=24.0)
    assert db.get_topic(fresh_tid)["freshness"] > 0.9


def test_niche_score_uses_whole_words_not_substrings(db):
    """The bug that made ranking useless: `t in hay` matched "ai" inside
    "explained" and "tech" inside "techniques", so an off-topic story scored a
    perfect 1.0 and outranked real ones."""
    off = db.add_topic(
        "Medieval pottery techniques explained by a historian", "rss"
    )
    on = db.add_topic("Nvidia ships a new AI gpu for datacentre racks", "rss")
    db.score_topics(["ai", "tech", "gadgets"], half_life_h=24.0)
    assert db.get_topic(off)["niche_score"] == 0.0
    assert db.get_topic(on)["niche_score"] > 0.0


def test_off_topic_story_cannot_outrank_an_on_topic_one(db):
    db.add_topic("Medieval pottery techniques explained by a historian", "rss")
    db.add_topic("Nvidia ships a new AI gpu for datacentre racks", "rss")
    db.score_topics(["ai", "tech", "gadgets"], half_life_h=24.0)
    top = db.top_topics(limit=1)[0]
    assert "nvidia" in top["title"].lower()


def test_niche_score_zero_when_off_topic(db):
    tid = db.add_topic("A story about medieval pottery", "rss")
    db.score_topics(["ai", "gadgets"], half_life_h=24.0)
    assert db.get_topic(tid)["niche_score"] == 0.0


def test_top_topics_respects_min_niche(db):
    db.add_topic("Fresh but irrelevant pottery news", "rss", published_at=utcnow().isoformat())
    db.add_topic("Fresh relevant ai gadget news", "rss", published_at=utcnow().isoformat())
    db.score_topics(["ai", "gadgets"], half_life_h=24.0)
    got = db.top_topics(limit=10, min_niche=0.1)
    assert all("ai" in r["title"] or "gadget" in r["title"] for r in got)


def test_score_topic_only_touches_one_row(db):
    a = db.add_topic("ai chip one", "rss")
    b = db.add_topic("ai chip two", "rss")
    db.score_topics(["ai"], half_life_h=24.0)
    before = db.get_topic(b)["niche_score"]
    db.score_topic(a, ["gadgets"], half_life_h=24.0)   # wrong niche on purpose
    assert db.get_topic(b)["niche_score"] == before
    assert db.score_topic("nope", ["ai"]) is False


# ------------------------------------------------------------------ runs
def test_run_lifecycle(db):
    rid = db.start_run("daily")
    db.finish_run(rid, "ok", "did a thing")
    row = db.conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
    assert row["status"] == "ok" and row["ended_at"]


def test_mark_topic_used_links_both_tables(db):
    tid = db.add_topic("A topic", "rss")
    vid = db.create_video("longform", "T", topic_id=tid)
    db.mark_topic_used(tid, vid)
    assert db.get_topic(tid)["status"] == "used"
    assert db.get_video(vid)["topic_id"] == tid


def test_analytics_upsert_is_idempotent(db):
    vid = db.create_video("longform", "T")
    row = {"day": "2099-01-01", "views": 10, "impressions": 100, "impressions_ctr": 0.1}
    db.upsert_analytics(vid, row)
    db.upsert_analytics(vid, dict(row, views=20))
    got = db.conn.execute("SELECT * FROM analytics WHERE video_id=?", (vid,)).fetchone()
    assert got["views"] == 20


def test_the_same_headline_from_two_urls_is_one_topic(db):
    """Same story, different link, two feeds.

    A URL-only dedup check let both through, so discover offered the channel
    two videos on identical news.
    """
    title = "Insurers claim AI is already increasing healthcare costs"
    first = db.add_topic(title, "reddit", url="https://reddit.com/r/x/1")
    second = db.add_topic(title, "rss", url="https://news.example.com/story")
    assert first is not None
    assert second is None
    rows = db.conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    assert rows == 1


def test_url_match_still_dedupes(db):
    assert db.add_topic("Title A", "rss", url="https://x.com/a") is not None
    assert db.add_topic("Totally different headline", "rss", url="https://x.com/a") is None


def test_different_headlines_are_both_kept(db):
    assert db.add_topic("Title A", "rss", url="https://x.com/a") is not None
    assert db.add_topic("Title B", "rss", url="https://x.com/b") is not None


def test_rejecting_a_topic_never_touches_its_source_text(db):
    """`summary` is the text the research step is given.

    The reason used to be appended to it, so a re-run researched a rejection
    notice instead of the article: "OpenAI ships GPT-6" came back with zero
    facts because the source it was shown was the string
    "AI news [rejected: research has no angle]".
    """
    topic_id = db.add_topic("OpenAI ships GPT-6", "rss", summary="AI news")
    db.mark_topic_rejected(topic_id, "research has no angle")
    after = db.conn.execute(
        "SELECT summary, status, reject_reason FROM topics WHERE id=?", (topic_id,)
    ).fetchone()
    assert after["summary"] == "AI news"
    assert after["status"] == "rejected"
    assert after["reject_reason"] == "research has no angle"


def test_rejecting_twice_does_not_grow_the_source_text(db):
    """The old append grew the field without bound on repeated retries."""
    topic_id = db.add_topic("GPT-6", "rss", summary="AI news")
    db.mark_topic_rejected(topic_id, "thin")
    db.mark_topic_rejected(topic_id, "still thin")
    summary = db.conn.execute(
        "SELECT summary FROM topics WHERE id=?", (topic_id,)
    ).fetchone()["summary"]
    assert summary == "AI news"
