"""End-to-end pipeline, fully offline.

No network, no ffmpeg, no credentials: the point is to prove the lifecycle
wiring — discover -> script -> approve -> render -> upload -> analytics — holds
together, and that every safety stop actually fires.
"""

from __future__ import annotations

import pytest

from ytauto.pipeline import discover as discover_stage
from ytauto.pipeline import render as render_stage
from ytauto.pipeline import script as script_stage
from ytauto.pipeline import upload as upload_stage
from ytauto.sources.base import Topic


@pytest.fixture(autouse=True)
def _offline_ok(monkeypatch):
    """Let degraded template content through so the chain can be walked."""
    monkeypatch.setenv("ALLOW_DEGRADED", "1")


def _seed(db, n=3):
    for i in range(n):
        db.add_topic(
            f"AI chip story number {i} about new GPUs",
            "rss",
            url=f"https://example.com/{i}",
            summary=(
                f"Nvidia announced a new GPU number {i} on Tuesday. The company "
                "says it runs inference four times faster than the last part. "
                "Analysts argue power draw, not compute, is the real limit. "
                "Shipments begin next quarter and pricing is not disclosed."
            ),
        )
    db.score_topics(["ai", "gpu", "nvidia"], half_life_h=24.0)


def _script_one(db, index=0):
    topic_id = db.top_topics(limit=index + 1)[index]["id"]
    return script_stage.script_topic(topic_id, db=db)


def _fake_media(monkeypatch, tmp_path=None):
    """Replace TTS + ffmpeg with cheap fakes.

    Patches `render_stage.voice.*` / `.vid.*` / `.ff.*` — the names
    render_video actually calls. Patching `render_stage.narrate` silently does
    nothing and the suite then runs real `say` for a minute per test.
    """
    def _narrate(text, out, **k):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"RIFF" + b"\0" * 64)
        return out

    def _captions(text, audio, kind):
        return [{"start": 0.0, "end": 1.0, "text": "hello"}]

    def _write_srt(captions, out):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")

    def _render_video(**k):
        out = k["out_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\0" * 4096)
        return out

    def _render_thumbnail(out, **k):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\xff" * 512)
        return out

    monkeypatch.setattr(render_stage.voice, "narrate", _narrate)
    monkeypatch.setattr(render_stage.voice, "build_captions", _captions)
    monkeypatch.setattr(render_stage.voice, "write_srt", _write_srt)
    monkeypatch.setattr(render_stage.vid, "render_video", _render_video)
    monkeypatch.setattr(render_stage.vid, "render_thumbnail", _render_thumbnail)
    monkeypatch.setattr(render_stage.ff, "probe_duration", lambda p: 42.0)


# ------------------------------------------------------------------ discover
def test_discover_adds_and_scores(db, monkeypatch):
    class FakeSource:
        name = "fake"

        def harvest(self, limit=20):
            return [
                Topic(title="Fresh AI gpu launch coverage", source="fake",
                      url="https://example.com/new", summary="A new GPU is faster."),
                Topic(title="Irrelevant pottery history", source="fake",
                      url="https://example.com/pottery", summary="Old pots."),
            ]

    monkeypatch.setattr(discover_stage, "harvest",
                        lambda limit_each=20, only=None: FakeSource().harvest())
    result = discover_stage.run(db=db, limit_each=5)

    assert result.fetched == 2
    assert result.added == 2
    # The off-topic story is now filtered out entirely by the niche gate.
    assert len(result.kept) == 1
    assert result.kept[0]["title"] == "Fresh AI gpu launch coverage"


def test_discover_is_deduplicated(db, monkeypatch):
    class FakeSource:
        name = "fake"

        def harvest(self, limit=20):
            return [Topic(title="Same story", source="fake", url="https://e.com/1")]

    monkeypatch.setattr(discover_stage, "harvest",
                        lambda limit_each=20, only=None: FakeSource().harvest())
    discover_stage.run(db=db, limit_each=5)
    again = discover_stage.run(db=db, limit_each=5)
    assert again.added == 0
    assert again.duplicates == 1


def test_manual_topic_added(db):
    tid = discover_stage.add_manual(db, "Manual AI story", url="https://e.com/m",
                                    summary="Something happened with a GPU.")
    assert tid
    assert db.get_topic(tid)["status"] == "new"


def test_manual_topic_needs_a_title(db):
    with pytest.raises(ValueError):
        discover_stage.add_manual(db, "   ")


# -------------------------------------------------------------------- script
def test_script_links_video_to_topic(db):
    _seed(db, 1)
    topic_id = db.top_topics(limit=1)[0]["id"]
    result = script_stage.script_topic(topic_id, db=db)

    assert result.status == "scripted", result.errors
    assert db.get_topic(topic_id)["topic_video_id"] == result.video_id
    assert db.get_video(result.video_id)["topic_id"] == topic_id


def test_script_is_idempotent_without_force(db):
    _seed(db, 1)
    topic_id = db.top_topics(limit=1)[0]["id"]
    first = script_stage.script_topic(topic_id, db=db)
    second = script_stage.script_topic(topic_id, db=db)
    assert second.video_id == first.video_id
    assert len(db.videos_in_status("scripted", "archived", "failed")) == 1


def test_force_rescripts_and_archives_the_old_video(db):
    _seed(db, 1)
    topic_id = db.top_topics(limit=1)[0]["id"]
    first = script_stage.script_topic(topic_id, db=db)
    second = script_stage.script_topic(topic_id, db=db, force=True)
    assert second.video_id != first.video_id
    assert db.get_video(first.video_id)["status"] == "archived"


def test_degraded_content_is_flagged(db):
    _seed(db, 1)
    result = _script_one(db)
    assert result.degraded is True                    # no ANTHROPIC_API_KEY
    assert "DEGRADED" in db.get_video(result.video_id)["source_note"]


def test_degraded_content_fails_without_the_escape_hatch(db, monkeypatch):
    monkeypatch.setenv("ALLOW_DEGRADED", "0")
    from ytauto.config import get_settings

    _seed(db, 1)
    topic_id = db.top_topics(limit=1)[0]["id"]
    result = script_stage.script_topic(topic_id, db=db, settings=get_settings())
    assert result.status == "failed"
    assert any("ANTHROPIC_API_KEY" in e for e in result.errors)


def test_a_failed_script_rejects_the_topic_rather_than_using_it(db, monkeypatch):
    """A topic whose script failed validation must not be recorded as `used`.

    `used` claims a video was built from it, which would be a lie, and it also
    hides the reason. It must be `rejected`, stay out of the selection queue,
    and keep the failing video linked so `script --show` can trace it back.
    """
    monkeypatch.setenv("ALLOW_DEGRADED", "0")
    _seed(db, 1)
    topic_id = db.top_topics(limit=1)[0]["id"]
    result = script_stage.script_topic(topic_id, db=db)

    assert result.status == "failed"
    row = db.get_topic(topic_id)
    assert row["status"] == "rejected"
    assert row["topic_video_id"] == result.video_id
    # The reason lives in its own column. It used to be appended to `summary`,
    # which is the source text research reads, so a retry researched a status
    # message instead of the article.
    assert row["reject_reason"]
    assert "rejected" not in (row["summary"] or "")
    # Out of the queue, so `run` cannot loop on the same unusable source.
    assert topic_id not in [t["id"] for t in db.top_topics(limit=5)]


def test_a_successful_script_marks_the_topic_used(db, monkeypatch):
    monkeypatch.setenv("ALLOW_DEGRADED", "1")
    _seed(db, 1)
    topic_id = db.top_topics(limit=1)[0]["id"]
    result = script_stage.script_topic(topic_id, db=db)
    assert result.status == "scripted"
    assert db.get_topic(topic_id)["status"] == "used"


def test_rescore_reapplies_the_current_niche(db):
    """Changing NICHE in .env must actually change stored scores."""
    _seed(db, 1)
    topic_id = db.top_topics(limit=1)[0]["id"]
    db.score_topics(["completely", "unrelated"], half_life_h=24.0)
    assert db.get_topic(topic_id)["niche_score"] == 0.0

    n = db.score_topics(["ai", "gpu", "nvidia"], half_life_h=24.0)
    assert n == 1
    assert db.get_topic(topic_id)["niche_score"] > 0.0


def test_offline_title_b_is_a_real_variant(db):
    """The A/B variant must not be a synonym swap, or the split tests nothing."""
    _seed(db, 1)
    result = _script_one(db)
    row = db.get_video(result.video_id)
    assert row["title_variant_b"]
    from ytauto.validate import check_title_b

    assert check_title_b(row["title_variant_b"], row["title"]) == []


def test_approve_moves_scripted_to_approved(db):
    _seed(db, 1)
    result = _script_one(db)
    assert result.status == "scripted"
    assert script_stage.approve(result.video_id, db=db) is True
    assert db.get_video(result.video_id)["status"] == "approved"


def test_approve_refuses_a_failed_video(db):
    _seed(db, 1)
    result = _script_one(db)
    db.mark_failed(result.video_id, "nope")
    assert script_stage.approve(result.video_id, db=db) is False


def test_pending_review_lists_only_awaiting(db):
    _seed(db, 2)
    # Pick both topic ids up front: scripting the first marks it used, which
    # would otherwise shrink top_topics() and shift the second index.
    topic_ids = [t["id"] for t in db.top_topics(limit=2)]
    a = script_stage.script_topic(topic_ids[0], db=db).video_id
    b = script_stage.script_topic(topic_ids[1], db=db).video_id
    script_stage.approve(a, db=db)
    pending = {r["id"] for r in script_stage.pending_review(db)}
    assert b in pending and a not in pending


# -------------------------------------------------------------------- render
def test_render_refuses_unapproved_video(db):
    _seed(db, 1)
    result = _script_one(db)
    r = render_stage.render_video(result.video_id, db=db)
    assert not r.ok
    assert "scripted" in (r.error or "")


def test_render_refuses_a_video_with_no_script(db):
    vid = db.create_video("longform", "Bare")
    db.set_status(vid, "approved")
    r = render_stage.render_video(vid, db=db)
    assert not r.ok
    assert "no script" in (r.error or "")


def test_render_writes_media_and_marks_rendered(db, monkeypatch):
    _seed(db, 1)
    result = _script_one(db)
    vid = result.video_id
    script_stage.approve(vid, db=db)

    _fake_media(monkeypatch)
    p = render_stage.paths_for(vid)
    r = render_stage.render_video(vid, db=db)
    assert r.ok, r.error
    assert db.get_video(vid)["status"] == "rendered"
    assert db.get_video(vid)["video_path"] == str(p["video"])


def test_render_is_idempotent(db, monkeypatch):
    _seed(db, 1)
    result = _script_one(db)
    vid = result.video_id
    script_stage.approve(vid, db=db)
    _fake_media(monkeypatch)
    p = render_stage.paths_for(vid)
    p["video"].parent.mkdir(parents=True, exist_ok=True)
    p["video"].write_bytes(b"already done")

    monkeypatch.setattr(render_stage.vid, "render_video",
                        lambda **k: pytest.fail("must not re-render"))
    r = render_stage.render_video(vid, db=db)
    assert r.ok
    assert db.get_video(vid)["status"] == "rendered"


def test_render_failure_marks_failed(db, monkeypatch):
    _seed(db, 1)
    result = _script_one(db)
    vid = result.video_id
    script_stage.approve(vid, db=db)

    def _boom(*a, **k):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(render_stage.voice, "narrate", _boom)
    r = render_stage.render_video(vid, db=db)
    assert not r.ok
    assert db.get_video(vid)["status"] == "failed"
    assert "ffmpeg exploded" in (db.get_video(vid)["last_error"] or "")


# -------------------------------------------------------------------- upload
# These tests exercise upload mechanics, not editorial judgement, so the fixture
# uses a script the publish gate would actually accept. The stubbed LLM returns
# degraded template content, which the gate correctly refuses, so seeding with it
# would make every upload test fail for a reason that has nothing to do with what
# they are checking.
POLICY_CLEAN_SCRIPT = (
    "Nvidia's new GPU is faster, but the interesting number is not the compute. "
    "The reason is that inference, not training, is what most buyers actually run, "
    "and inference is limited by memory bandwidth and power draw rather than raw "
    "flops. Compared with the previous part, the gain is real but it is far "
    "smaller than the headline suggests, which is a trade-off rather than a free "
    "win for anyone budgeting for throughput. In practice the four-times claim "
    "holds for one batch size and not for the next, so what to watch is the "
    "per-watt figure rather than the peak number. However, pricing is still "
    "undisclosed, so the real question is what it costs per delivered token. "
    "That suggests buyers should wait for independent benchmarks, because the "
    "company benchmarks are the ones most likely to be flattering. So the "
    "practical advice is to compare throughput per watt against what you already "
    "run, and to treat a four-times claim as a starting point for a conversation "
    "rather than as a decision."
)


def _rendered(db, tmp_path):
    _seed(db, 1)
    result = _script_one(db)
    vid = result.video_id
    script_stage.approve(vid, db=db)
    media = tmp_path / f"{vid}.mp4"
    media.write_bytes(b"\0" * (2 * 1024 * 1024))
    db.update_video(vid, script=POLICY_CLEAN_SCRIPT)
    db.set_status(vid, "rendered", video_path=str(media))
    return vid, media


def test_upload_in_dry_run_does_not_call_youtube(db, tmp_path, monkeypatch):
    vid, _ = _rendered(db, tmp_path)
    monkeypatch.setattr(
        "ytauto.youtube.client.YouTubeClient.upload",
        lambda *a, **k: pytest.fail("dry run must not call the API"),
        raising=False,
    )
    outcome = upload_stage.upload_video(vid, db=db, dry_run=True)
    assert outcome.ok, outcome.error
    assert db.get_video(vid)["status"] == "rendered"     # not "scheduled"
    assert db.get_video(vid)["youtube_id"] is None
    assert db.quota_used_today() == 0     # no quota burned on a simulation
    assert outcome.dry_run is True


def test_dry_run_does_not_reserve_a_calendar_slot(db, tmp_path):
    """A preview must not consume real capacity; otherwise repeated dry runs
    fill the horizon and block the genuine upload."""
    vid, _ = _rendered(db, tmp_path)
    assert db.get_video(vid)["publish_at"] is None
    outcome = upload_stage.upload_video(vid, db=db, dry_run=True)
    assert outcome.publish_at                       # it reports the plan...
    assert db.get_video(vid)["publish_at"] is None  # ...without saving it


def test_a_real_upload_would_persist_the_slot(db, tmp_path, monkeypatch):
    """Dry runs deliberately do not save a slot; a real upload must."""
    vid, _ = _rendered(db, tmp_path)

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def upload(self, **k):
            from ytauto.youtube.client import UploadResult

            return UploadResult(
                video_id="yt-new", title="t", privacy="private",
                publish_at=k.get("publish_at"),
            )

    monkeypatch.setattr(upload_stage, "YouTubeClient", FakeClient)
    outcome = upload_stage.upload_video(vid, db=db, dry_run=False)
    assert outcome.ok, outcome.error
    assert db.get_video(vid)["status"] == "scheduled"
    assert db.get_video(vid)["youtube_id"] == "yt-new"
    assert db.get_video(vid)["publish_at"]


def test_upload_is_blocked_when_quota_is_exhausted(db, tmp_path):
    vid, _ = _rendered(db, tmp_path)
    db.record_quota("videos.insert", cost=9000)   # budget gone
    outcome = upload_stage.upload_video(vid, db=db, dry_run=False)
    assert not outcome.ok
    assert "quota" in (outcome.error or "").lower()

def test_upload_of_an_unrendered_video_is_refused(db):
    _seed(db, 1)
    result = _script_one(db)
    outcome = upload_stage.upload_video(result.video_id, db=db, dry_run=True)
    assert not outcome.ok
    assert "render" in (outcome.error or "")


def test_upload_refuses_missing_media(db, tmp_path):
    vid, _ = _rendered(db, tmp_path)
    db.conn.execute("UPDATE videos SET video_path=? WHERE id=?", ("/tmp/gone.mp4", vid))
    outcome = upload_stage.upload_video(vid, db=db, dry_run=True)
    assert not outcome.ok
    assert "no media" in (outcome.error or "")


def test_reupload_of_a_scheduled_video_is_a_noop(db, tmp_path):
    """Re-running the uploader must not spend a second 1600 units."""
    vid, _ = _rendered(db, tmp_path)
    db.set_status(vid, "scheduled", youtube_id="yt-existing",
                  publish_at="2099-01-01T13:00:00+00:00")
    monkey = None
    outcome = upload_stage.upload_video(vid, db=db, dry_run=False)
    assert outcome.ok
    assert outcome.youtube_id == "yt-existing"
    assert "already uploaded" in " ".join(outcome.notes)
    assert db.quota_used_today() == 0


# ----------------------------------------------------------------- analytics
def test_analytics_before_any_upload_is_a_noop(db):
    from ytauto.youtube import analytics

    assert analytics.fetch_daily(db) == 0
    assert analytics.summary(db)["totals"]["views"] == 0
