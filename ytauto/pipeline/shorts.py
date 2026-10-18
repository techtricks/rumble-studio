"""Stage 3.5 — shorts: derive a vertical cut from every finished longform.

Each rendered longform automatically gains a short vertical clip of its
opening hook, so one story reaches both the browse page (longform) and the
feed (shorts) without an extra scripting pass. The short reuses the longform's
own footage — voice, captions, title card — re-framed into 9:16 with the
blurred-background composition vertical surfaces demand, plus a slow push so
the frame keeps moving.

The videos table has no parent column, and `topic_id` belongs to the script
pipeline, so the parent link lives in `source_note` as `short-of:<id>`. A
short is status `rendered` from birth: the Rumble upload stage treats it like
any other rendered video, which is exactly the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import Settings, get_settings
from ..config import media_dir as _media_dir
from ..db import Database, get_db
from ..logging_setup import get_logger
from ..render import ffmpeg as ff

log = get_logger("pipeline.shorts")

SHORT_SIZE = (1080, 1920)
SHORT_LEN_S = 50.0     # feed sweet spot: 30-60 s
MIN_SOURCE_S = 25.0    # anything shorter has no hook worth cutting


@dataclass
class ShortResult:
    video_id: str           # the longform this belongs to
    short_id: Optional[str] = None
    video_path: Optional[Path] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.short_id is not None


def short_paths(video_id: str) -> dict:
    d = _media_dir()
    d.mkdir(parents=True, exist_ok=True)
    return {
        "video": d / f"{video_id}-short.mp4",
        "thumb": d / f"{video_id}-short.jpg",
    }


def _existing_short(db: Database, parent_id: str) -> Optional["object"]:
    return db.conn.execute(
        "select * from videos where kind='shorts' and source_note=?",
        (f"short-of:{parent_id}",),
    ).fetchone()


def make_short(
    video_id: str,
    *,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
    force: bool = False,
) -> ShortResult:
    """Cut the hook of a rendered longform into a finished vertical short."""
    s = settings or get_settings()
    db = db or get_db()
    ff.require()

    row = db.get_video(video_id)
    if row is None:
        return ShortResult(video_id=video_id, error=f"unknown video {video_id}")
    if row["kind"] != "longform":
        return ShortResult(video_id=video_id, error="only longform videos get shorts")
    # `published` is allowed on purpose: shorts for already-live videos are as
    # legitimate as shorts for freshly rendered ones — they just upload later.
    if row["status"] not in ("rendered", "published"):
        return ShortResult(video_id=video_id, error=f"video is {row['status']}")

    src = Path(row["video_path"] or "")
    if not src.exists():
        return ShortResult(video_id=video_id, error=f"media missing: {src}")

    out = short_paths(video_id)
    if out["video"].exists() and not force:
        short = _existing_short(db, video_id)
        dur = int(ff.probe_duration(out["video"]) or 0)
        if short is not None:
            if short["status"] == "rendered":
                db.set_status(short["id"], "rendered", video_path=str(out["video"]),
                              thumb_path=str(out["thumb"]) if out["thumb"].exists() else None,
                              duration_s=dur)
            return ShortResult(video_id=video_id, short_id=short["id"], video_path=out["video"])
        # media without a row (a crash between file and INSERT) — recover below
    if out["video"].exists() and force:
        out["video"].unlink(missing_ok=True)

    duration = ff.probe_duration(src) or row["duration_s"] or 0
    if duration < MIN_SOURCE_S:
        return ShortResult(video_id=video_id, error=f"source too short ({duration:.0f}s) for a hook cut")

    length = min(SHORT_LEN_S, duration - 0.5)
    W, H = SHORT_SIZE
    # Foreground: the original 16:9 frame, slightly inset so the blurred
    # background reads around it, with a slow push-in of its own. Captions
    # burned near the bottom of the source frame stay well inside this window.
    fg_w = int(W * 0.92) // 2 * 2          # 992
    fg_h = int(fg_w * 9 / 16) // 2 * 2     # 558
    seg = length * 30  # frames for the push-in expression
    graph = (
        f"[0:v]trim=start=0:duration={length:.3f},setpts=PTS-STARTPTS,split[a][b];"
        # blurred plate fills the 9:16 canvas
        f"[b]scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},gblur=sigma=26,eq=brightness=-0.10:saturation=1.25[bg];"
        # sharp foreground with a gentle cinematic push
        f"[a]scale={fg_w}:{fg_h},setsar=1,"
        f"zoompan=z='1.0+0.14*on/{seg:.0f}':d=1:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"s={fg_w}x{fg_h}:fps=30,"
        f"vignette=PI/4.6[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )
    cmd = [
        "-i", str(src),
        "-filter_complex", graph,
        "-map", "[v]", "-map", "0:a",
        "-t", f"{length:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "22",
        "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        "-movflags", "+faststart",
        str(out["video"]),
    ]
    try:
        ff.run(cmd, timeout=1800)
    except Exception as exc:  # noqa: BLE001 - a short must never kill the run
        return ShortResult(video_id=video_id, error=f"ffmpeg: {exc}")

    if not out["video"].exists() or out["video"].stat().st_size < 10_000:
        return ShortResult(video_id=video_id, error="ffmpeg produced no usable short")

    # vertical poster straight out of the finished cut
    try:
        ff.run([
            "-y", "-ss", "3", "-i", str(out["video"]), "-frames:v", "1",
            "-q:v", "3", str(out["thumb"]),
        ], timeout=120)
    except Exception as exc:  # noqa: BLE001 - thumb optional
        log.warning("short thumbnail failed for %s: %s", video_id, exc)

    short = _existing_short(db, video_id)
    dur = int(ff.probe_duration(out["video"]) or length)
    if short is None:
        short_id = db.create_video(
            "shorts",
            str(row["title"] or "")[:70],
            description="",
            tags=[],
            source_note=f"short-of:{video_id}",
            duration_s=dur,
        )
    else:
        short_id = short["id"]
    db.set_status(
        short_id, "rendered",
        video_path=str(out["video"]),
        thumb_path=str(out["thumb"]) if out["thumb"].exists() else None,
        duration_s=dur,
    )
    log.info("short ready: %s (%.0fs) from %s", short_id, dur, video_id)
    return ShortResult(video_id=video_id, short_id=short_id, video_path=out["video"])


def build_missing(db: Optional[Database] = None, *, limit: int = 3) -> list[ShortResult]:
    """Cut shorts for every rendered longform that does not have one yet."""
    db = db or get_db()
    out: list[ShortResult] = []
    rows = db.videos_in_status("rendered", "published")
    done = {
        r["source_note"].split(":", 1)[1]
        for r in db.conn.execute(
            "select source_note from videos where kind='shorts' and source_note like 'short-of:%'"
        ).fetchall()
    }
    for row in rows:
        if row["kind"] != "longform" or row["id"] in done:
            continue
        if len(out) >= limit:
            break
        out.append(make_short(row["id"], db=db))
    return out
