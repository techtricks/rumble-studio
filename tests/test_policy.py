"""Tests for the publish gate.

These encode YouTube's monetisation rules as executable checks. The rules
themselves come from the Partner Program policy, and the reason they are hard
failures rather than warnings is that "inauthentic content" is enforced at
channel level: one flagged video can cost the whole channel its revenue.
"""

from __future__ import annotations

import pytest

from ytauto import policy
from ytauto.config import Settings
from ytauto.db import Database
from ytauto.pipeline import upload as up


ANALYTICAL = (
    "Google is testing buying from Flipkart inside Gemini, which is a bigger deal "
    "than it looks. The reason is that checkout is where commerce actually "
    "happens, and until now the assistant could recommend a product but not "
    "complete the purchase. That means a merchant who optimised for the app "
    "suddenly loses the click, which is why several of them are already "
    "lobbying for the same feature. However, the pilot is limited to India, so "
    "the effect on any other market is unclear for now. Compared with the last "
    "attempt at instant checkout, this one keeps the payment step in the browser, "
    "where it used to live. The risk for users is that an assistant choosing what "
    "to buy is a very different thing from the user choosing, and the trade-off "
    "is convenience against control. In practice the feature only fires when the "
    "model is confident. So what to watch is whether the pilot widens, and "
    "whether regulators in India ask questions about who chose the item."
)

FACT_DUMP = (
    "Google is testing buying from Flipkart inside Gemini. The pilot is in India. "
    "Flipkart is owned by Walmart. Walmart bought Flipkart in 2018. The deal was "
    "worth sixteen billion dollars. Gemini is Google's assistant. Google launched "
    "Gemini in 2023. The feature is called AI Mode. AI Mode is a new search "
    "interface. It is available in the United States. It rolls out gradually. "
    "Users can also use the app. The app is on Android. It is also on iOS. "
    "Shoppers in India use it daily. Orders arrive in two days. Refunds take five. "
    "The company said nothing about pricing. Analysts expect a change. The "
    "feature is limited for now. More details are expected soon."
)


# --------------------------------------------------------------------------
# originality
# --------------------------------------------------------------------------
def test_analytical_script_passes():
    res = policy.check_originality(ANALYTICAL, [])
    assert res.ok, res.violations


def test_fact_dump_warns_about_missing_reasoning():
    """A weak signal warns rather than refuses; see policy.check_originality."""
    res = policy.check_originality(FACT_DUMP, [])
    assert res.ok, res.violations
    assert any("reasoning" in w for w in res.warnings)


def test_analysis_ratio_separates_restating_from_reasoning():
    assert policy.analysis_ratio(ANALYTICAL) > 0.15
    assert policy.analysis_ratio(FACT_DUMP) < 0.15


def test_empty_script_is_not_flagged_as_inauthentic():
    # An empty script is a different failure; do not disguise it as a policy one.
    res = policy.check_originality("", [])
    assert res.ok, res.violations


def test_a_tiny_script_is_not_judged_at_all():
    # A Short can legitimately be three sentences. Judging those on a ratio
    # would flag every Short, which is noise rather than signal.
    tiny = "Google is testing Flipkart in Gemini. The pilot is India-only. Watch the rollout."
    res = policy.check_originality(tiny, [])
    assert res.ok
    assert not res.warnings


def test_near_duplicate_of_a_published_video_is_blocked():
    published = [{"id": "old1", "title": "Older video", "script": ANALYTICAL}]
    res = policy.check_originality(ANALYTICAL, published)
    assert not res.ok
    assert any("interchangeable" in v for v in res.violations)


def test_different_topics_still_pass_against_each_other():
    other = (
        "A foldable phone finally has a hinge rated for its full advertised life, "
        "which removes the only real objection people had. The reason this matters "
        "is that the previous generation bent at around two hundred thousand folds. "
        "Compared with a normal slab phone, the foldable costs more and lasts "
        "longer, which is a trade-off rather than a free win. In practice the "
        "rating is still a laboratory number, so what to watch is independent "
        "testing. However, the warranty is now longer than the part's rated life, "
        "which suggests the manufacturer expects it to hold. That suggests the "
        "hinge, not the screen, was always the weak point. So the practical advice "
        "is to keep the warranty, and to wait for a repair price to be published."
    )
    published = [{"id": "old1", "title": "Older video", "script": other}]
    assert policy.check_originality(ANALYTICAL, published).ok


def test_a_moderate_overlap_warns_but_does_not_block():
    # Half-shared wording is a flag for a human, not an automatic failure; the
    # hard block sits above the threshold where the videos really are the same.
    base = ANALYTICAL
    near = (
        base[: int(len(base) * 0.25)]
        + " Separately, a different subject entirely: a foldable phone hinge was "
        "rated for its full advertised life, which removes the usual objection. "
        "The reason it matters is that the previous part bent far sooner. "
        "Compared with a slab phone, it costs more and lasts longer. "
        "In practice the rating is still a laboratory number, so what to watch "
        "is independent testing. However, the warranty now outlasts the rating. "
        "So the practical advice is to keep the warranty and wait for repair prices."
    )
    res = policy.check_originality(near, [{"id": "old", "title": "t", "script": base}])
    assert not res.violations, res.violations
    assert res.warnings, "moderate overlap should be flagged for a human"


def test_short_scripts_are_not_compared():
    assert policy.script_similarity("too short", "also short") == 0.0


def test_similarity_is_symmetric_and_bounded():
    assert policy.script_similarity(ANALYTICAL, FACT_DUMP) == policy.script_similarity(FACT_DUMP, ANALYTICAL)
    v = policy.script_similarity(ANALYTICAL, FACT_DUMP)
    assert 0.0 <= v <= 1.0


# --------------------------------------------------------------------------
# rewrite versus analysis, measured against the real source
# --------------------------------------------------------------------------
SOURCE_ARTICLE = (
    "Waymo has roughly four thousand vehicles in operation. Eighty percent of "
    "them are in California and Texas. The company has not published a national "
    "rollout timetable. Waymo said in a statement that expansion requires "
    "regulatory approval in each state where it intends to operate. Industry "
    "analysts said the concentration reflects permitting timelines rather than "
    "demand. Waymo has not said when it will operate outside those two states. "
    "The company has not said how much the fleet has cost to build. Waymo has "
    "not said how much each vehicle costs to operate. Waymo said earlier this "
    "year that it would add vehicles in Phoenix, and the city now has more of "
    "them than it did. Waymo said that the vehicles in California and Texas are "
    "used by paying customers, and the ones elsewhere are used for testing."
)

# What the same story sounds like when the only job is to restate it: the
# article's own sentences, stitched together with no added reasoning.
RESTATEMENT = (
    "Waymo has roughly four thousand vehicles in operation. Eighty percent of "
    "them are in California and Texas. The company has not published a national "
    "rollout timetable. Waymo said in a statement that expansion requires "
    "regulatory approval in each state where it intends to operate. Industry "
    "analysts said the concentration reflects permitting timelines rather than "
    "demand. Waymo has not said when it will operate outside those two states. "
    "The company has not said how much the fleet has cost to build. Waymo has "
    "not said how much each vehicle costs to operate. Waymo said earlier this "
    "year that it would add vehicles in Phoenix, and the city now has more of "
    "them than it did."
)


def test_verbatim_rewrite_is_detected_as_a_rewrite():
    ngram, novel = policy.source_overlap(RESTATEMENT, SOURCE_ARTICLE)
    assert ngram > policy._MAX_SOURCE_NGRAM_SHARE
    assert novel < policy._MIN_NOVEL_WORD_SHARE
    res = policy.check_originality(RESTATEMENT, [], source_text=SOURCE_ARTICLE)
    assert not res.ok
    assert any("rewrite rather than commentary" in v for v in res.violations)


def test_original_commentary_about_the_same_facts_passes():
    ngram, novel = policy.source_overlap(ANALYTICAL, SOURCE_ARTICLE)
    assert ngram <= policy._MAX_SOURCE_NGRAM_SHARE
    assert novel >= policy._MIN_NOVEL_WORD_SHARE
    assert policy.check_originality(ANALYTICAL, [], source_text=SOURCE_ARTICLE).ok


def test_source_check_is_skipped_when_no_source_is_available():
    # An article that has moved must not block a good video on its own.
    assert policy.check_originality(ANALYTICAL, [], source_text="").ok


def test_source_overlap_ignores_short_words_and_digits():
    """'a' and '80' match anything, so they must not carry the measurement."""
    ngram, novel = policy.source_overlap(ANALYTICAL, SOURCE_ARTICLE)
    assert 0.0 <= novel <= 1.0
    assert "80" not in policy._content_words(ANALYTICAL)


def test_source_overlap_of_a_tiny_script_is_not_judged():
    assert policy.source_overlap("too short", SOURCE_ARTICLE) == (0.0, 1.0)


# --------------------------------------------------------------------------
# assets
# --------------------------------------------------------------------------
GOOD_CREDIT = {
    "url": "https://commons.wikimedia.org/a.jpg",
    "title": "Googleplex HQ",
    "creator": "Asoundd",
    "license_name": "CC BY-SA 4.0",
    "credit": "Asoundd / CC BY-SA 4.0",
}


def test_clean_assets_pass():
    res = policy.check_assets(
        image_credits=[GOOD_CREDIT], has_thumbnail=True, has_caption_file=True
    )
    assert res.ok, res.violations
    assert not res.warnings


def test_nc_licence_is_refused():
    """Non-commercial cannot be monetised; ND cannot be cropped to 16:9."""
    for lic in ("CC BY-NC 2.0", "CC BY-ND 3.0", "BY-NC-ND"):
        bad = dict(GOOD_CREDIT, license_name=lic)
        res = policy.check_assets(
            image_credits=[bad], has_thumbnail=True, has_caption_file=True
        )
        assert not res.ok, lic
        assert any("commercial use or derivatives" in v for v in res.violations)


def test_unlicensed_image_is_refused():
    res = policy.check_assets(
        image_credits=[dict(GOOD_CREDIT, license_name="")],
        has_thumbnail=True, has_caption_file=True,
    )
    assert not res.ok
    assert any("no licence" in v for v in res.violations)


def test_image_without_attribution_is_refused():
    """CC-BY requires the credit to travel with the work; that means here."""
    res = policy.check_assets(
        image_credits=[dict(GOOD_CREDIT, credit="")],
        has_thumbnail=True, has_caption_file=True,
    )
    assert not res.ok
    assert any("attribution" in v for v in res.violations)


def test_missing_thumbnail_and_captions_warn_only():
    res = policy.check_assets(
        image_credits=[GOOD_CREDIT], has_thumbnail=False, has_caption_file=False
    )
    assert res.ok
    assert len(res.warnings) == 2


def test_no_images_at_all_warns():
    res = policy.check_assets(
        image_credits=[], has_thumbnail=True, has_caption_file=True
    )
    assert res.ok
    assert res.warnings


# --------------------------------------------------------------------------
# description
# --------------------------------------------------------------------------
def test_description_carries_credits_and_disclosure():
    d = policy.build_description(
        base="Summary paragraph.",
        image_credits=[GOOD_CREDIT],
        source_url="https://example.com/story",
        brand="Tech Tricks",
        voice="Tara",
    )
    assert "Summary paragraph." in d
    # The credit has to reach YouTube, not a file on this machine.
    assert "Asoundd" in d and "CC BY-SA 4.0" in d
    assert "https://commons.wikimedia.org/a.jpg" in d
    assert "https://example.com/story" in d
    # Synthetic narration must be disclosed.
    assert "synthesised" in d.lower()
    assert "Tara" in d


def test_description_disclosure_present_even_with_nothing_else():
    d = policy.build_description(base="")
    assert "automation" in d.lower()


def test_description_fits_the_api_limit():
    d = policy.build_description(
        base="x" * 20_000,
        image_credits=[dict(GOOD_CREDIT, title="t" * 500) for _ in range(50)],
    )
    assert len(d) <= 5000


def test_manual_steps_mention_the_studio_disclosure():
    steps = " ".join(policy.manual_steps()).lower()
    # No API surface for the altered-content flag, so it has to be a human step.
    assert "studio" in steps
    assert "disclos" in steps


# --------------------------------------------------------------------------
# config defaults that back the policy
# --------------------------------------------------------------------------
def test_article_images_are_off_by_default():
    """The outlet's copyrighted photo is the biggest single monetisation risk."""
    from ytauto.config import Settings

    assert Settings().allow_article_images is False


def test_voice_is_not_pinned_by_default():
    from ytauto.config import Settings

    assert Settings().tts_voice == ""


def test_voice_pool_is_not_a_single_voice():
    from ytauto.render.voice import DEFAULT_VOICE_POOL

    voices = [v for v in DEFAULT_VOICE_POOL.split(",") if v]
    assert len(set(voices)) > 1
    # Novelty voices are not appropriate for a technology explainer.
    for novelty in ("Bubbles", "Cellos", "Zarvox", "Wobble", "Bad News"):
        assert novelty not in voices


def test_pick_voice_varies_across_videos_and_is_stable():
    from ytauto.config import Settings
    from ytauto.render.voice import pick_voice

    s = Settings()
    picks = {pick_voice(f"vid{i:03d}", settings=s) for i in range(24)}
    assert len(picks) > 1
    # Re-rendering the same video must not change its voice, or published audio
    # could never be matched to its row.
    assert pick_voice("vid001", settings=s) == pick_voice("vid001", settings=s)


def test_explicit_voice_overrides_rotation():
    from ytauto.config import Settings
    from ytauto.render.voice import pick_voice

    s = Settings()
    s.tts_voice = "Daniel"
    assert all(pick_voice(f"v{i}", settings=s) == "Daniel" for i in range(10))


# --------------------------------------------------------------------------
# the gate as wired into upload
# --------------------------------------------------------------------------
def _published_video(db, tmp_path, *, vid, topic, script, credits="[]"):
    media = tmp_path / f"{vid}.mp4"
    media.write_bytes(b"\x00" * 4000)
    db.conn.execute(
        "INSERT INTO topics(id,title,url,source,created_at) VALUES (?,?,?,?,?)",
        (topic, "Story", "https://example.com/x", "rss", "2026-01-01T00:00:00+00:00"),
    )
    db.conn.execute(
        """INSERT INTO videos(id,kind,status,title,description,tags,script,
           video_path,thumb_path,topic_id,image_credits,narrator_voice,
           created_at,updated_at)
           VALUES (?,'longform','rendered','T','D','[]',?,?,?,?,?,?,?,?)""",
        (
            vid, script, str(media), str(tmp_path / f"{vid}.jpg"), topic,
            credits, "Tara", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00",
        ),
    )
    return media


def test_upload_is_blocked_when_the_script_rewrites_the_article(tmp_path, monkeypatch):
    """The hard gate: a script that tracks the source's own wording."""
    from ytauto.pipeline import upload as up

    db = Database(tmp_path / "rewrite.db")
    _published_video(db, tmp_path, vid="v1", topic="t1", script=RESTATEMENT)
    monkeypatch.setattr(up, "_source_text", lambda db, row: SOURCE_ARTICLE)

    s = Settings()
    s.dry_run = True
    outcome = up.upload_video("v1", db=db, settings=s, dry_run=True, schedule=False)
    assert not outcome.ok
    assert "policy gate" in (outcome.error or "")
    assert "rewrite rather than commentary" in (outcome.error or "")
    # The reason is persisted on the row, not just in a log line.
    assert db.get_video("v1")["policy_error"]


def test_upload_is_blocked_for_a_nc_licensed_picture(tmp_path):
    """Non-commercial imagery cannot be monetised, so it never gets uploaded."""
    from ytauto.pipeline import upload as up

    db = Database(tmp_path / "lic.db")
    _published_video(
        db, tmp_path, vid="v2", topic="t2", script=ANALYTICAL,
        credits='[{"title": "A photo", "url": "u", "license_name": "CC BY-NC 2.0", "credit": "x / CC BY-NC 2.0"}]',
    )
    s = Settings()
    s.dry_run = True
    outcome = up.upload_video("v2", db=db, settings=s, dry_run=True, schedule=False)
    assert not outcome.ok
    assert "commercial use or derivatives" in (outcome.error or "")


def test_upload_is_blocked_by_a_duplicate_of_a_live_video(tmp_path):
    """Channel-level inauthenticity is judged across the whole catalogue."""
    from ytauto.pipeline import upload as up

    db = Database(tmp_path / "dup.db")
    _published_video(db, tmp_path, vid="old", topic="t1", script=ANALYTICAL)
    db.conn.execute(
        "UPDATE videos SET status='published', youtube_id='yt123' WHERE id='old'"
    )
    _published_video(db, tmp_path, vid="new", topic="t2", script=ANALYTICAL)

    s = Settings()
    s.dry_run = True
    outcome = up.upload_video("new", db=db, settings=s, dry_run=True, schedule=False)
    assert not outcome.ok
    assert "interchangeable" in (outcome.error or "")


def test_upload_passes_the_gate_for_a_good_script(tmp_path):
    """The gate must not be a blanket block; an original script gets through."""
    from ytauto.pipeline import upload as up

    db = Database(tmp_path / "ok.db")
    _published_video(
        db, tmp_path, vid="v3", topic="t3", script=ANALYTICAL,
        credits='[{"title": "HQ", "url": "u", "license_name": "CC BY 4.0", "credit": "A / CC BY 4.0"}]',
    )
    s = Settings()
    s.dry_run = True
    outcome = up.upload_video("v3", db=db, settings=s, dry_run=True, schedule=False)
    assert outcome.ok, outcome.error
    assert outcome.dry_run
    assert db.get_video("v3")["policy_error"] is None


def test_manual_steps_are_documented_in_readme():
    from pathlib import Path

    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8").lower()
    assert "monetis" in text or "monetiz" in text
    assert "inauthentic" in text or "repetitive" in text
