"""YouTube Data API v3 client.

The hard constraint here is quota. A single `videos.insert` costs **1600
units** and the daily cap is 10,000 — six uploads and the key is dead until
00:00 Pacific, and an overage returns 403 with `quotaExceeded` for a full day.
So *every* call goes through `_call()`, which checks the ledger first and
records the spend after. A client that talks to YouTube without that ledger
will eventually brick the project for a day.

Scheduling is done the only way the API allows: upload as `private` with
`status.publishAt` in the future. YouTube then publishes it itself, which means
we do not need a live process at the publish moment.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import COST, Settings, get_settings
from ..db import Database, get_db, iso, utcnow
from ..logging_setup import get_logger
from ..util import parse_dt, retry
from .auth import AuthError, load_credentials

log = get_logger("youtube.client")

API_NAME = "youtube"
API_VERSION = "v3"

# Uploads are a single request but can run for many minutes; never let the
# default HTTP timeout abort a half-finished upload.
UPLOAD_TIMEOUT_S = 3600

# `status` reasons that mean "stop, do not retry the same request".
_FATAL = ("quotaExceeded", "forbidden", "invalidCredentials", "insufficientPermissions")


class YouTubeError(RuntimeError):
    pass


class QuotaExhausted(YouTubeError):
    pass


class UploadFailed(YouTubeError):
    pass


@dataclass
class UploadResult:
    video_id: str
    title: str
    privacy: str
    publish_at: Optional[str] = None

    @property
    def url(self) -> str:
        return f"https://youtu.be/{self.video_id}"


def get_service(credentials: Any = None, *, cache_discovery: bool = False) -> Any:
    """Build an authenticated googleapiclient service."""
    from googleapiclient.discovery import build

    creds = credentials or load_credentials(refresh=True)
    if creds is None or not getattr(creds, "valid", False):
        raise AuthError(
            "no valid YouTube credentials. Run `python -m ytauto auth` to grant access."
        )
    # cache_discovery writes a .py cache next to site-packages; on a read-only
    # venv that raises, so default it off.
    return build(API_NAME, API_VERSION, credentials=creds, cache_discovery=cache_discovery)


class YouTubeClient:
    """Quota-aware wrapper. One instance per run; reuses the HTTP session."""

    def __init__(self, db: Optional[Database] = None, settings: Optional[Settings] = None) -> None:
        self.s = settings or get_settings()
        self.db = db or get_db()
        self._service: Any = None
        self._channel_id: Optional[str] = None

    # ------------------------------------------------------------ plumbing
    @property
    def service(self) -> Any:
        if self._service is None:
            self._service = get_service()
        return self._service

    def _call(self, endpoint: str, fn, *, note: str = "", cost: Optional[int] = None, retries: int = 3):
        """Run an API call, guarded by the quota ledger.

        Raises QuotaExhausted *before* spending anything if the ledger says we
        cannot afford it, so a misconfigured cron cannot burn the day's budget
        on partial work.
        """
        cost = COST.get(endpoint, 100) if cost is None else cost
        if not self.db.can_afford(endpoint):
            raise QuotaExhausted(
                f"{endpoint} costs {cost} units but only "
                f"{self.db.quota_remaining()} remain today "
                f"(cap {self.s.daily_quota}, safety margin applied). Refusing to call."
            )

        def _invoke():
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                text = str(exc)
                if any(f in text for f in _FATAL):
                    raise
                raise

        result = retry(
            _invoke,
            attempts=retries,
            base_delay=2.0,
            on_error=lambda n, e: log.warning("%s attempt %d failed: %s", endpoint, n, e),
        )
        self.db.record_quota(endpoint, note=note or endpoint, cost=cost)
        return result

    # -------------------------------------------------------------- channel
    def channel_id(self, *, discover: bool = True) -> str:
        """The authenticated user's channel id, cached in the db."""
        if self.s.channel_id:
            return self.s.channel_id
        if self._channel_id:
            return self._channel_id
        cached = self.db.get_meta("channel_id")
        if cached:
            self._channel_id = cached
            return cached
        if not discover:
            raise YouTubeError(
                "YOUTUBE_CHANNEL_ID is not set and none is cached. "
                "Run `python -m ytauto auth` to detect it."
            )
        cid = self._call(
            "channels.list",
            lambda: self.service.channels()
            .list(part="id,snippet", mine=True)
            .execute(),
            note="discover channel id",
        )
        items = cid.get("items") or []
        if not items:
            raise YouTubeError(
                "no channel found for these credentials. Confirm the OAuth "
                "account owns a channel (a Brand Account must be selected as the "
                "'YouTube' account on the consent screen)."
            )
        self._channel_id = items[0]["id"]
        self.db.set_meta("channel_id", self._channel_id)
        log.info("channel id: %s (%s)", self._channel_id, items[0].get("snippet", {}).get("title", ""))
        return self._channel_id

    # --------------------------------------------------------------- upload
    def upload(
        self,
        *,
        video_path: Path,
        title: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        publish_at: Optional[datetime] = None,
        privacy: str = "private",
        made_for_kids: bool = False,
        category_id: str = "27",  # Science & Technology
        thumbnail_path: Optional[Path] = None,
    ) -> UploadResult:
        """Upload a video. Scheduled when `publish_at` is in the future.

        `privacy` is forced to `private` whenever `publish_at` is set: YouTube
        only honours `publishAt` for a private video, and uploading a
        public/unlisted one with a future timestamp silently publishes it now.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise UploadFailed(f"video file not found: {video_path}")
        if video_path.stat().st_size < 10_000:
            raise UploadFailed(f"video file suspiciously small: {video_path}")

        if publish_at is not None:
            privacy = "private"
        if privacy not in ("private", "unlisted", "public"):
            raise UploadFailed(f"invalid privacyStatus {privacy!r}")

        body: Dict[str, Any] = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "tags": [t for t in (tags or []) if t][:500],
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": made_for_kids,
                "embeddable": True,
            },
        }
        if publish_at is not None:
            body["status"]["publishAt"] = publish_at.astimezone(timezone.utc).isoformat()
            log.info("scheduling for %s (%s)", body["status"]["publishAt"], privacy)

        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)

        def _do() -> Any:
            request = self.service.videos().insert(
                part="snippet,status", body=body, media_body=media
            )
            response = None
            while response is None:
                status, response = request.next_chunk(num_retries=3)
                if status is not None:
                    log.info("  uploaded %.0f%%", status.progress() * 100)
            return response

        result = self._call("videos.insert", _do, note=f"upload {title[:40]}")
        video_id = result.get("id")
        if not video_id:
            raise UploadFailed(f"upload returned no id: {json.dumps(result)[:300]}")

        # The status block we sent is echoed back; trust the server, not us.
        actual = result.get("status", {}).get("privacyStatus", privacy)
        echoed_publish = result.get("status", {}).get("publishAt")
        log.info("uploaded %s -> %s", video_id, result.get("snippet", {}).get("title", "")[:60])

        if thumbnail_path and Path(thumbnail_path).exists():
            try:
                self.set_thumbnail(video_id, Path(thumbnail_path))
            except YouTubeError as exc:
                # A missing thumbnail must not invalidate a completed upload.
                log.error("thumbnail upload failed for %s: %s", video_id, exc)

        return UploadResult(
            video_id=video_id,
            title=result.get("snippet", {}).get("title", title),
            privacy=actual,
            publish_at=echoed_publish or (iso(publish_at) if publish_at else None),
        )

    # ------------------------------------------------------------ thumbnail
    def set_thumbnail(self, video_id: str, path: Path) -> None:
        """Attach a custom thumbnail (50 units)."""
        from googleapiclient.http import MediaFileUpload

        if not Path(path).exists():
            raise YouTubeError(f"thumbnail not found: {path}")
        self._call(
            "thumbnails.set",
            lambda: self.service.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(str(path), resumable=False),
            ).execute(),
            note=f"thumbnail {video_id}",
        )
        log.info("thumbnail set on %s", video_id)

    # -------------------------------------------------------------- queries
    def video_status(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Cheap (1 unit) status read. Used to confirm a scheduled video went live."""

        def _q() -> Dict[str, Any]:
            resp = self.service.videos().list(
                part="status,snippet,contentDetails", id=video_id
            ).execute()
            items = resp.get("items") or []
            return items[0] if items else {}

        return self._call("videos.list", _q, note=f"status {video_id}")

    def is_live(self, video_id: str) -> Optional[bool]:
        """True/False if the video's privacy is public, None if unknown."""
        try:
            info = self.video_status(video_id)
        except YouTubeError as exc:
            log.warning("status check failed for %s: %s", video_id, exc)
            return None
        if not info:
            return None
        privacy = (info.get("status") or {}).get("privacyStatus")
        if privacy is None:
            return None
        return privacy == "public"

    def confirm_published(self, video_id: str, *, published_at: Optional[str] = None) -> bool:
        """Mark a video published once YouTube reports it public.

        Returns True if the state changed. False is normal and expected for a
        video whose time has not come yet.
        """
        if self.is_live(video_id):
            self.db.conn.execute(
                "UPDATE videos SET status='published', published_at=COALESCE(published_at,?), "
                "updated_at=? WHERE id=?",
                (published_at or iso(), iso(), video_id),
            )
            log.info("video %s confirmed live", video_id)
            return True
        return False

    def sync_published(self, *, only_scheduled: bool = True) -> int:
        """Reconcile every scheduled video against YouTube. Returns the count
        of videos that went live since the last run."""
        sql = "SELECT id, youtube_id, publish_at FROM videos WHERE youtube_id IS NOT NULL"
        args: List[Any] = []
        if only_scheduled:
            sql += " AND status='scheduled'"
        rows = self.db.conn.execute(sql, args).fetchall()
        newly_live = 0
        for row in rows:
            expected = parse_dt(row["publish_at"])
            # Don't spend 1 unit on a video that cannot be live yet.
            if expected and expected > utcnow():
                continue
            if self.confirm_published(row["youtube_id"], published_at=iso(expected)):
                newly_live += 1
        if rows:
            log.info("checked %d scheduled video(s), %d newly live", len(rows), newly_live)
        return newly_live

    # -------------------------------------------------------------- playlist
    def add_to_playlist(self, playlist_id: str, video_id: str) -> None:
        self._call(
            "playlistItems.insert",
            lambda: self.service.playlistItems().insert(
                part="snippet",
                body={"snippet": {"playlistId": playlist_id, "resourceId": video_id}},
            ).execute(),
            note=f"playlist {playlist_id}",
        )
        log.info("added %s to playlist %s", video_id, playlist_id)

    def update_metadata(
        self,
        video_id: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Patch an existing upload (50 units). Used by the A/B title swap."""
        body: Dict[str, Any] = {"snippet": {}}
        if title is not None:
            body["snippet"]["title"] = title[:100]
        if description is not None:
            body["snippet"]["description"] = description[:5000]
        if tags is not None:
            body["snippet"]["tags"] = list(tags)[:500]
        if not body["snippet"]:
            return
        self._call(
            "videos.update",
            lambda: self.service.videos().update(part="snippet", body=body, id=video_id).execute(),
            note=f"update {video_id}",
        )
        log.info("metadata updated on %s", video_id)

    # ---------------------------------------------------------------- quota
    def quota_snapshot(self) -> Dict[str, Any]:
        day = utcnow().strftime("%Y-%m-%d")
        rows = self.db.conn.execute(
            "SELECT endpoint, SUM(cost) total, COUNT(*) n FROM quota_ledger "
            "WHERE day=? GROUP BY endpoint ORDER BY total DESC",
            (day,),
        ).fetchall()
        return {
            "day": day,
            "used": self.db.quota_used_today(day),
            "remaining": self.db.quota_remaining(day),
            "cap": self.s.daily_quota,
            "by_endpoint": [
                {"endpoint": r["endpoint"], "units": r["total"], "calls": r["n"]} for r in rows
            ],
        }


_client: Optional[YouTubeClient] = None


def get_client() -> YouTubeClient:
    global _client
    if _client is None:
        _client = YouTubeClient()
    return _client
