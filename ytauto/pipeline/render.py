"""Stage 3 — render: approved -> rendered (video + thumbnail on disk).

Rendered media is keyed by video id under `data/media/`, so a re-render
overwrites in place and a crashed run leaves no half-named files that a later
run would mistake for finished work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Settings, get_settings
from ..config import media_dir as _media_dir
from ..db import Database, get_db
from ..logging_setup import get_logger
from ..render import ffmpeg as ff
from ..render import video as vid
from ..render import voice
from ..sources.images import ImageAsset, images_for_video
from ..util import clip, humanize_duration

log = get_logger("pipeline.render")


@dataclass
class RenderResult:
    video_id: str
    video_path: Optional[Path] = None
    thumb_path: Optional[Path] = None
    duration_s: Optional[float] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.video_path is not None


def media_dir() -> Path:
    d = _media_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def paths_for(video_id: str) -> Dict[str, Path]:
    d = media_dir()
    return {
        "video": d / f"{video_id}.mp4",
        "thumb": d / f"{video_id}.jpg",
        "audio": d / f"{video_id}.wav",
        "srt": d / f"{video_id}.srt",
        "images": d / video_id / "images",
        "credits": d / video_id / "credits.txt",
    }


def _image_assets(
    row: Any,
    db: Database,
    video_id: str,
    p: Dict[str, Path],
    limit: int,
    *,
    allow_article: bool = True,
) -> List[ImageAsset]:
    """Gather on-topic pictures for the story, degrading to none on any failure.

    A picture problem must never cost a rendered video, so every upstream call
    here is already best-effort. The credits file is not optional though: several
    of these sources are CC-licensed and require attribution, so whatever ends up
    on screen gets recorded next to the media.
    """
    # With the article picture disallowed there is no point paying for its page
    # fetch: every candidate would come from the free stock APIs instead.
    url = ""
    if allow_article and row["topic_id"]:
        topic = db.get_topic(row["topic_id"])
        if topic is not None:
            url = topic["url"] or ""
    try:
        tags = json.loads(row["tags"] or "[]")
        if not isinstance(tags, list):
            tags = []
    except Exception:  # noqa: BLE001
        tags = []

    try:
        assets = images_for_video(
            url=url,
            title=row["title"] or "",
            tags=[str(t) for t in tags],
            dest_dir=p["images"],
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001 - pictures are optional
        log.warning("image collection failed for %s (gradient fallback): %s", video_id, exc)
        return []

    if assets:
        lines = ["Image credits for %s" % (row["title"] or video_id), ""]
        for a in assets:
            lines.append(f"- {a.title or 'image'}: {a.credit() or a.url}")
        try:
            p["credits"].parent.mkdir(parents=True, exist_ok=True)
            p["credits"].write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            log.warning("could not write credits file: %s", exc)
    return assets


def render_video(
    video_id: str,
    *,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
    force: bool = False,
    with_thumbnail: bool = True,
) -> RenderResult:
    """Narrate + composite a video. Idempotent unless `force`."""
    s = settings or get_settings()
    db = db or get_db()

    row = db.get_video(video_id)
    if row is None:
        return RenderResult(video_id=video_id, error=f"unknown video {video_id}")
    if row["status"] not in ("approved", "rendered", "scheduled"):
        return RenderResult(
            video_id=video_id,
            error=f"video is {row['status']}; only approved/rendered videos can render",
        )
    if not row["script"]:
        return RenderResult(video_id=video_id, error="video has no script")

    p = paths_for(video_id)
    if p["video"].exists() and not force:
        log.info("video %s already rendered -> %s (use force to redo)", video_id, p["video"])
        # Sync the status on the way out. Without this, a video whose media
        # exists but whose row still says "approved" — a crash between writing
        # the file and the UPDATE, or a manual status edit — would be reported
        # as rendered successfully while the row disagreed, and render_queue
        # would keep re-claiming it on every run, forever.
        if row["status"] != "rendered":
            db.set_status(
                video_id, "rendered",
                video_path=str(p["video"]),
                thumb_path=str(p["thumb"]) if p["thumb"].exists() else None,
                duration_s=int(ff.probe_duration(p["video"]) or 0),
            )
        return RenderResult(
            video_id=video_id, video_path=p["video"],
            thumb_path=p["thumb"] if p["thumb"].exists() else None,
            duration_s=ff.probe_duration(p["video"]),
        )

    try:
        plan = _plan(row)
        narrator = voice.pick_voice(video_id, settings=s)
        audio = voice.narrate(row["script"], p["audio"], settings=s, voice=narrator)
        duration = ff.probe_duration(audio)
        captions = voice.build_captions(row["script"], audio, row["kind"])
        if not captions:
            return RenderResult(video_id=video_id, error="caption timing produced nothing")
        voice.write_srt(captions, p["srt"])
        # The narration is one audio track; the extra languages travel as
        # sidecar subtitle files the platforms can auto-select per viewer.
        # Best-effort: a rate-limited LLM must never fail a render.
        try:
            from ..render.langs import translate_sidecars
            translate_sidecars(captions, p["srt"], settings=s)
        except Exception as exc:  # noqa: BLE001 - subtitles are optional
            log.warning("sidecar translation skipped for %s: %s", video_id, exc)

        assets = _image_assets(
            row, db, video_id, p, limit=s.images_per_video,
            allow_article=s.allow_article_images,
        )

        out = vid.render_video(
            audio_path=audio,
            captions=captions,
            out_path=p["video"],
            kind=row["kind"],
            title=row["title"] or "",
            thumbnail_plan=plan,
            brand=s.brand,
            settings=s,
            images=[Path(a.path) for a in assets],
        )

        thumb: Optional[Path] = None
        if with_thumbnail:
            try:
                thumb = vid.render_thumbnail(
                    p["thumb"],
                    text=str(plan.get("text") or clip(row["title"] or "", 60)),
                    subtitle=s.brand,
                    kind=row["kind"],
                    thumbnail_plan=plan,
                    settings=s,
                )
            except Exception as exc:  # noqa: BLE001 - video is the deliverable
                log.error("thumbnail failed for %s (video is still fine): %s", video_id, exc)

        final_s = int(ff.probe_duration(out) or duration or 0)
        # Licence and voice travel with the row so the upload step can build a
        # compliant description and re-check the policy without re-rendering.
        credits = [
            {
                "url": a.url, "title": a.title, "creator": a.creator,
                "license_name": a.license_name, "source_page": a.source_page,
                "credit": a.credit(),
            }
            for a in assets
        ]
        db.update_video(
            video_id,
            image_credits=json.dumps(credits),
            narrator_voice=narrator,
        )
        db.set_status(
            video_id, "rendered",
            video_path=str(out),
            thumb_path=str(thumb) if thumb else None,
            duration_s=final_s,
        )
        log.info(
            "rendered %s: %s (%s, %d captions)",
            video_id, out.name, humanize_duration(final_s), len(captions),
        )
        return RenderResult(
            video_id=video_id, video_path=out, thumb_path=thumb, duration_s=final_s,
        )
    except Exception as exc:  # noqa: BLE001 - one bad video must not stop the run
        db.mark_failed(video_id, f"render: {exc}")
        log.error("render failed for %s: %s", video_id, exc)
        return RenderResult(video_id=video_id, error=str(exc))


def _plan(row: Any) -> Dict[str, Any]:
    try:
        plan = json.loads(row["thumbnail_plan"] or "{}")
        return plan if isinstance(plan, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def render_queue(
    *,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
    limit: int = 5,
    kind: Optional[str] = None,
) -> List[RenderResult]:
    """Render up to `limit` approved videos, oldest first.

    Deliberately excludes `failed`: the render guard rejects failed rows
    outright, so letting them into the queue would just burn the `limit`
    slots (and lease churn) every cycle while genuinely approved videos
    starve behind them. Re-rendering a failed video is a `render --force`
    operation, not a queue job.
    """
    db = db or get_db()
    s = settings or get_settings()
    statuses = ("approved", "rendered")
    rows = db.videos_in_status(*statuses)
    if kind:
        rows = [r for r in rows if r["kind"] == kind]
    rows = sorted(rows, key=lambda r: r["created_at"])[:limit]

    out: List[RenderResult] = []
    for row in rows:
        owner = db.owner_token()
        if not db.claim(row["id"], owner, minutes=60):
            log.info("skipping %s: leased by another worker", row["id"])
            continue
        try:
            out.append(render_video(row["id"], db=db, settings=s))
        finally:
            db.release(row["id"], owner)
    return out


def cleanup(video_id: str) -> int:
    """Delete rendered artefacts for a video (keeps the db row)."""
    removed = 0
    for key, path in paths_for(video_id).items():
        if key == "srt" or path.exists():
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    log.info("cleaned %d file(s) for %s", removed, video_id)
    return removed
