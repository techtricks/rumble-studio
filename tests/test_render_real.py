"""Real render smoke tests.

These shell out to the real ffmpeg binary and the real macOS `say`, because the
fake-based pipeline tests cannot catch the failures that actually happen in
production: a filter expression the build does not support, a missing font, an
encoder that rejects the pixel format.

Marked `slow` — run just them with `pytest -m slow`.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from ytauto.pipeline import render as render_stage
from ytauto.render import ffmpeg as ff
from ytauto.render import video as vid
from ytauto.render import voice

pytestmark = pytest.mark.slow

needs_ffmpeg = pytest.mark.skipif(
    ff.find_ffmpeg() is None, reason="no ffmpeg binary available"
)
needs_say = pytest.mark.skipif(
    voice.say_available() is False, reason="macOS `say` not available"
)


# ------------------------------------------------------------------ ffmpeg
@needs_ffmpeg
def test_ffmpeg_binary_runs():
    out = subprocess.run(
        [str(ff.find_ffmpeg()), "-version"], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0
    assert "ffmpeg version" in out.stdout


@needs_ffmpeg
def test_version_of_reports_a_version():
    assert "ffmpeg version" in ff.version_of()


# -------------------------------------------------------------------- tts
@needs_say
def test_say_produces_audible_wav(tmp_path):
    out = tmp_path / "a.wav"
    voice.narrate("Testing one two three.", out)
    assert out.exists()
    assert out.stat().st_size > 1000
    assert ff.probe_duration(out) > 0.5


@needs_say
def test_caption_timing_covers_the_audio(tmp_path):
    out = tmp_path / "a.wav"
    voice.narrate(
        "This is the first sentence. And this is the second one, which is a "
        "little longer so the timing has to adapt to it.",
        out,
    )
    dur = ff.probe_duration(out)
    caps = voice.build_captions(
        "This is the first sentence. And this is the second one, which is a "
        "little longer so the timing has to adapt to it.",
        out,
        "longform",
    )
    assert caps
    assert caps[0].start == 0.0
    assert caps[-1].end <= dur + 1.0
    assert all(c.duration > 0 for c in caps)
    # Monotonic, non-overlapping: the SRT would be broken otherwise.
    for a, b in zip(caps, caps[1:]):
        assert b.start >= a.end - 0.01


@needs_say
def test_srt_is_written_in_srt_format(tmp_path):
    out = tmp_path / "a.wav"
    voice.narrate("Hello there.", out)
    caps = voice.build_captions("Hello there.", out, "longform")
    srt = tmp_path / "a.srt"
    voice.write_srt(caps, srt)
    text = srt.read_text(encoding="utf-8")
    assert text.startswith("1\n")
    assert "-->" in text


# ------------------------------------------------------------------ video
@needs_ffmpeg
@needs_say
@pytest.mark.parametrize("kind,expect_w,expect_h", [
    ("longform", 1920, 1080),
    ("shorts", 1080, 1920),
])
def test_rendered_video_has_the_right_dimensions(tmp_path, kind, expect_w, expect_h):
    audio = tmp_path / "a.wav"
    voice.narrate("A short test of the render pipeline.", audio)
    caps = voice.build_captions("A short test of the render pipeline.", audio, kind)
    out = tmp_path / f"{kind}.mp4"

    vid.render_video(
        audio_path=audio,
        captions=caps,
        out_path=out,
        kind=kind,
        title="Rendering pipeline smoke test",
        thumbnail_plan={"text": "It works", "emotion": "confident",
                        "subject": "a GPU on a desk"},
        brand="testbrand",
    )
    assert out.exists()
    assert out.stat().st_size > 10_000

    probe = subprocess.run(
        [str(ff.find_ffmpeg()), "-i", str(out)],
        capture_output=True, text=True, timeout=60,
    )
    # ffmpeg writes the stream summary to stderr and exits non-zero without -t.
    summary = probe.stderr
    assert f"{expect_w}x{expect_h}" in summary, summary[-800:]
    assert "Video: h264" in summary


@needs_ffmpeg
@needs_say
def test_rendered_video_is_playable_and_silent_free(tmp_path):
    """Decode the whole file: a truncated/garbled mp4 passes a size check but
    fails here."""
    audio = tmp_path / "a.wav"
    line = (
        "Decoding check. This line is deliberately long enough that the render "
        "guard accepts it, because a sub-second clip is refused on purpose."
    )
    voice.narrate(line, audio)
    caps = voice.build_captions(line, audio, "longform")
    out = tmp_path / "x.mp4"
    vid.render_video(
        audio_path=audio, captions=caps, out_path=out, kind="longform",
        title="Decode check", thumbnail_plan={}, brand="b",
    )
    probe = subprocess.run(
        [str(ff.find_ffmpeg()), "-v", "error", "-i", str(out), "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
    )
    assert probe.returncode == 0, probe.stderr[-2000:]
    assert not probe.stderr.strip(), f"decode errors: {probe.stderr[-500:]}"


@needs_ffmpeg
def test_thumbnail_is_a_valid_jpeg(tmp_path):
    out = tmp_path / "t.jpg"
    vid.render_thumbnail(
        out, text="Big news", subtitle="brand", kind="longform",
        thumbnail_plan={"text": "Big news", "emotion": "shocked",
                        "subject": "a chip close up"},
    )
    assert out.exists()
    assert out.read_bytes()[:3] == b"\xff\xd8\xff"   # JPEG SOI marker


# ---------------------------------------------------------------- pipeline
@needs_ffmpeg
@needs_say
def test_pipeline_render_stage_produces_playable_media(db, monkeypatch):
    """The real thing: a scripted video goes through the render stage and comes
    out with a file the uploader would accept."""
    from ytauto.sources.base import Topic

    db.add_topic(
        "Nvidia ships a GPU four times faster than before", "rss",
        url="https://example.com/a",
        summary=(
            "Nvidia announced a new datacenter GPU on Tuesday. The company says "
            "it runs inference four times faster than the last part. Analysts "
            "argue power draw is the real limit. Pricing was not disclosed."
        ),
    )
    db.score_topics(["ai", "gpu", "nvidia"], half_life_h=24.0)
    monkeypatch.setenv("ALLOW_DEGRADED", "1")

    from ytauto.config import get_settings
    from ytauto.pipeline import script as script_stage

    s = get_settings()
    topic_id = db.top_topics(limit=1)[0]["id"]
    result = script_stage.script_topic(topic_id, db=db, settings=s)
    assert result.status == "scripted", result.errors
    assert script_stage.approve(result.video_id, db=db, settings=s)

    r = render_stage.render_video(result.video_id, db=db, settings=s)
    assert r.ok, r.error
    row = db.get_video(result.video_id)
    assert row["status"] == "rendered"
    media = row["video_path"]
    assert media and shutil.os.path.getsize(media) > 10_000
    assert int(row["duration_s"]) > 0
