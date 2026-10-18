"""Stage 4 — upload + Stage 5 — publish confirmation.

`upload` is the only stage that spends YouTube quota, so it is the most
defensive: a lease is taken first, quota is checked, DRY_RUN short-circuits, and
a failed upload records the quota *and* the error before releasing the lease.

Publishing is confirmed, not assumed. YouTube flips a scheduled video to public
on its own; we just detect it and update our row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Settings, get_settings
from ..db import Database, get_db, iso
from ..logging_setup import get_logger
from ..policy import PolicyResult, build_description, check_assets, check_originality
from ..schedule import assign
from ..util import clip, humanize_duration, parse_dt
from ..youtube.client import QuotaExhausted, UploadFailed, YouTubeClient, YouTubeError

log = get_logger("pipeline.upload")


@dataclass
class UploadOutcome:
    video_id: str
    ok: bool
    youtube_id: Optional[str] = None
    url: Optional[str] = None
    publish_at: Optional[str] = None
    dry_run: bool = False
    error: str = ""
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.dry_run:
            return f"[dry-run] would upload {self.video_id}"
        if self.ok:
            return f"uploaded {self.youtube_id} -> {self.url} (publish {self.publish_at})"
        return f"FAILED {self.video_id}: {self.error}"


def upload_video(
    video_id: str,
    *,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
    client: Optional[YouTubeClient] = None,
    schedule: bool = True,
    dry_run: Optional[bool] = None,
    privacy: str = "private",
) -> UploadOutcome:
    """Upload one rendered video and schedule it.

    `dry_run` defaults to the configured DRY_RUN, which is True out of the box:
    the pipeline will plan, render and schedule, but never touch the channel
    until a human sets DRY_RUN=0.
    """
    s = settings or get_settings()
    db = db or get_db()
    dry = s.dry_run if dry_run is None else dry_run

    row = db.get_video(video_id)
    if row is None:
        return UploadOutcome(video_id=video_id, ok=False, error="unknown video")
    if row["status"] not in ("rendered", "scheduled", "failed"):
        return UploadOutcome(
            video_id=video_id, ok=False,
            error=f"video is {row['status']}; render it before uploading",
        )
    if row["youtube_id"] and row["status"] == "scheduled":
        return UploadOutcome(
            video_id=video_id, ok=True, youtube_id=row["youtube_id"],
            url=f"https://youtu.be/{row['youtube_id']}",
            publish_at=row["publish_at"],
            error="", notes=["already uploaded"],
        )
    video_path = Path(row["video_path"] or "")
    if not video_path.exists():
        return UploadOutcome(
            video_id=video_id, ok=False, error=f"no media at {video_path or '(unset)'}",
        )

    # The publish gate. YouTube can pull monetisation from a whole channel over
    # inauthentic content, so this refuses rather than warns: a video that
    # narrates someone else's article, reads like an upload already on the
    # channel, or puts an NC/ND photograph on screen never leaves the machine.
    gate = _policy_gate(row, db=db, settings=s)
    if not gate.ok:
        db.update_video(video_id, policy_error=gate.reason()[:2000])
        return UploadOutcome(
            video_id=video_id, ok=False,
            error=f"blocked by the publish policy gate: {gate.reason()}",
        )
    for w in gate.warnings:
        log.warning("policy warning for %s: %s", video_id, w)
    db.update_video(video_id, policy_error=None)

    # Pick the publish time first: if the calendar is full there is no point
    # spending 1600 quota units on an upload we cannot schedule.
    publish_at: Optional[datetime] = None
    if schedule:
        slot = assign(db, video_id, row["kind"], settings=s, persist=not dry)
        if slot is None:
            return UploadOutcome(
                video_id=video_id, ok=False,
                error="no free publish slot inside the horizon; not spending quota",
            )
        publish_at = slot.when
    elif row["publish_at"]:
        publish_at = parse_dt(row["publish_at"])

    tags = _tags(row)
    duration = humanize_duration(row["duration_s"])

    if dry:
        log.info(
            "[dry-run] %s %s -> %s | %s | %d tags | publish %s | thumb %s",
            row["kind"], row["title"], video_path.name, duration, len(tags),
            publish_at.isoformat() if publish_at else "immediately",
            "yes" if row["thumb_path"] else "no",
        )
        return UploadOutcome(
            video_id=video_id, ok=True, dry_run=True,
            publish_at=publish_at.isoformat() if publish_at else None,
            notes=[f"would upload {video_path.name}", f"title={row['title']!r}"],
        )

    try:
        client = client or YouTubeClient(db=db, settings=s)
    except Exception as exc:  # noqa: BLE001 - auth problems
        return UploadOutcome(video_id=video_id, ok=False, error=f"auth: {exc}")

    owner = db.owner_token()
    if not db.claim(video_id, owner, minutes=120):
        return UploadOutcome(
            video_id=video_id, ok=False, error="another worker holds the lease",
        )
    try:
        result = client.upload(
            video_path=video_path,
            title=row["title"] or "",
            description=_description(row, db=db, settings=s),
            tags=tags,
            publish_at=publish_at,
            privacy=privacy,
            thumbnail_path=Path(row["thumb_path"]) if row["thumb_path"] else None,
        )
        db.set_status(
            video_id, "scheduled",
            youtube_id=result.video_id,
            scheduled_at=iso(),
            publish_at=result.publish_at or (iso(publish_at) if publish_at else None),
        )
        log.info(
            "video %s uploaded: %s (privacy=%s, publish_at=%s)",
            video_id, result.url, result.privacy, result.publish_at,
        )
        return UploadOutcome(
            video_id=video_id, ok=True, youtube_id=result.video_id,
            url=result.url, publish_at=result.publish_at,
        )
    except QuotaExhausted as exc:
        # Not a video problem: the daily cap is spent. Leave it approved so the
        # next run retries without re-rendering.
        db.set_status(video_id, "rendered", error=f"quota: {exc}")
        log.error("quota exhausted, %s left rendered for a later run: %s", video_id, exc)
        return UploadOutcome(video_id=video_id, ok=False, error=f"quota: {exc}")
    except (UploadFailed, YouTubeError) as exc:
        db.mark_failed(video_id, f"upload: {exc}")
        return UploadOutcome(video_id=video_id, ok=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        db.mark_failed(video_id, f"upload: {exc}")
        log.exception("unexpected upload error for %s", video_id)
        return UploadOutcome(video_id=video_id, ok=False, error=str(exc))
    finally:
        db.release(video_id, owner)


def _credits(row: Any) -> List[Dict[str, str]]:
    import json

    try:
        raw = json.loads(row["image_credits"] or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(raw, list):
        return []
    return [c for c in raw if isinstance(c, dict)]


def _published_scripts(db: Database, exclude_id: str = "") -> List[Dict[str, str]]:
    """Scripts already on the channel, for the interchangeability check.

    Only real uploads count. Drafts that never went out cannot make the channel
    look repetitive to a viewer, and comparing against them would block the
    first few videos of a batch for no reason.
    """
    rows = db.conn.execute(
        "SELECT id, title, script FROM videos "
        "WHERE status IN ('scheduled','published') AND youtube_id IS NOT NULL"
    ).fetchall()
    out = []
    for r in rows:
        if exclude_id and r["id"] == exclude_id:
            continue
        if r["script"]:
            out.append({"id": r["id"], "title": r["title"] or "", "script": r["script"]})
    return out


def _source_text(db: Database, row: Any) -> str:
    """The article the research came from, for the rewrite-versus-analysis check.

    One extra HTTP request per upload, and an upload is rare, so it is not worth
    caching. A failure here is not a reason to block: the originality, duplicate
    and licence checks all still apply without it, and refusing a video because an
    article moved would be worse than the risk being missed.
    """
    if not row["topic_id"]:
        return ""
    topic = db.get_topic(row["topic_id"])
    url = (topic["url"] if topic is not None else "") or ""
    if not url:
        return ""
    try:
        from ..sources.article import fetch_article_text

        return fetch_article_text(url) or ""
    except Exception as exc:  # noqa: BLE001
        log.warning("could not re-read the source article for the policy check: %s", exc)
        return ""


def _policy_gate(
    row: Any, *, db: Database, settings: Settings
) -> "PolicyResult":
    """Everything that must be true before this video may be published."""
    result = check_originality(
        row["script"] or "",
        _published_scripts(db, exclude_id=row["id"]),
        source_text=_source_text(db, row),
    )
    # The caption file is not a column; it sits next to the video, named by id.
    from ..config import media_dir

    srt = media_dir() / f"{row['id']}.srt"
    assets = check_assets(
        image_credits=_credits(row),
        has_thumbnail=bool(row["thumb_path"]),
        has_caption_file=srt.exists(),
    )
    result.violations.extend(assets.violations)
    result.warnings.extend(assets.warnings)
    return result


def _description(row: Any, *, db: Database, settings: Settings) -> str:
    """The description that actually reaches YouTube, credits and disclosure included."""
    source_url = ""
    if row["topic_id"]:
        topic = db.get_topic(row["topic_id"])
        if topic is not None:
            source_url = topic["url"] or ""
    return build_description(
        base=row["description"] or "",
        image_credits=_credits(row),
        source_url=source_url,
        brand=settings.brand,
        voice=(row["narrator_voice"] if "narrator_voice" in row.keys() else "") or "",
    )


def _tags(row: Any) -> List[str]:
    import json

    try:
        tags = json.loads(row["tags"] or "[]")
        return [str(t) for t in tags if str(t).strip()]
    except (json.JSONDecodeError, TypeError):
        return []


def upload_queue(
    *,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
    client: Optional[YouTubeClient] = None,
    limit: int = 3,
    kind: Optional[str] = None,
    dry_run: Optional[bool] = None,
) -> List[UploadOutcome]:
    """Upload rendered videos, cheapest-risk first (oldest publish time)."""
    db = db or get_db()
    s = settings or get_settings()
    rows = db.videos_in_status("rendered")
    if kind:
        rows = [r for r in rows if r["kind"] == kind]
    rows = sorted(rows, key=lambda r: (r["publish_at"] or "9999", r["created_at"]))[:limit]
    return [
        upload_video(r["id"], db=db, settings=s, client=client, dry_run=dry_run)
        for r in rows
    ]


# --------------------------------------------------------------- publish
def sync_published(
    *,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
    client: Optional[YouTubeClient] = None,
) -> int:
    """Confirm which scheduled videos are now live. Cheap (1 unit each)."""
    db = db or get_db()
    s = settings or get_settings()
    if s.dry_run:
        log.info("[dry-run] would verify scheduled videos against YouTube")
        return 0
    try:
        client = client or YouTubeClient(db=db, settings=s)
        return client.sync_published()
    except Exception as exc:  # noqa: BLE001
        log.error("publish sync failed: %s", exc)
        return 0


def go_live_now(
    video_id: str,
    *,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
    client: Optional[YouTubeClient] = None,
    dry_run: Optional[bool] = None,
) -> UploadOutcome:
    """Publish immediately instead of at the scheduled time (50 units).

    This is the escape hatch for "this must go out now" — it costs quota and is
    therefore never done implicitly.
    """
    db = db or get_db()
    s = settings or get_settings()
    dry = s.dry_run if dry_run is None else dry_run
    row = db.get_video(video_id)
    if row is None or not row["youtube_id"]:
        return UploadOutcome(video_id=video_id, ok=False, error="video is not uploaded yet")
    if dry:
        return UploadOutcome(
            video_id=video_id, ok=True, dry_run=True,
            notes=[f"would make {row['youtube_id']} public now"],
        )
    try:
        client = client or YouTubeClient(db=db, settings=s)
        client.service.videos().update(
            part="status",
            body={"status": {"privacyStatus": "public", "publishAt": ""}},
            id=row["youtube_id"],
        ).execute()
        db.record_quota("videos.update", note=f"publish now {video_id}")
        db.set_status(video_id, "published", published_at=iso())
        log.info("video %s is public now", video_id)
        return UploadOutcome(
            video_id=video_id, ok=True, youtube_id=row["youtube_id"],
            url=f"https://youtu.be/{row['youtube_id']}",
        )
    except Exception as exc:  # noqa: BLE001
        db.mark_failed(video_id, f"publish: {exc}")
        return UploadOutcome(video_id=video_id, ok=False, error=str(exc))
