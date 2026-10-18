"""Publish-slot calculation.

YouTube scheduling works by uploading as `private` with a `publishAt` timestamp
in the future. Picking that timestamp is where most of the pain lives:

* slots must be in the creator's timezone, not the server's;
* YouTube rejects `publishAt` in the past, and rejects anything more than ~50
  days out;
* two videos must never land on top of each other;
* re-running the planner must be idempotent, otherwise a second `plan` call
  moves every video to a new time.

So: candidate slots come from config, collisions are resolved by pushing to the
next free slot, and an already-assigned `publish_at` is never changed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .config import Settings, get_settings
from .logging_setup import get_logger
from .util import humanize_delta, parse_dt

log = get_logger("schedule")

# YouTube's own limit: publishAt more than ~50 days out is rejected.
MAX_HORIZON_DAYS = 45
# Leave room for the upload itself to finish before the publish time.
MIN_LEAD_MINUTES = 30

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


class ScheduleError(RuntimeError):
    pass


def tz(s: Settings) -> timezone:
    from zoneinfo import ZoneInfo

    try:
        return ZoneInfo(s.timezone)  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        raise ScheduleError(
            f"invalid TIMEZONE {s.timezone!r}: {exc}. Use an IANA name like 'Asia/Kolkata'."
        ) from exc


def parse_hhmm(value: str) -> Tuple[int, int]:
    m = _TIME_RE.match(str(value).strip())
    if not m:
        raise ScheduleError(f"bad time {value!r}; expected HH:MM in 24h form")
    hh, mm = int(m.group(1)), int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ScheduleError(f"time out of range: {value!r}")
    return hh, mm


def local_slots(kind: str, day: datetime, s: Settings) -> List[datetime]:
    """Every configured publish time for `kind`, as aware datetimes on `day`.

    `day` is interpreted in the channel timezone.
    """
    times = s.shorts_times if kind == "shorts" else s.longform_times
    zone = tz(s)
    out: List[datetime] = []
    for raw in times:
        hh, mm = parse_hhmm(raw)
        local = day.astimezone(zone).replace(hour=hh, minute=mm, second=0, microsecond=0)
        out.append(local)
    return sorted(out)


def _occupied(db: Any, exclude: Optional[str] = None) -> List[datetime]:
    """Publish times already taken by videos we manage."""
    rows = db.conn.execute(
        "SELECT id, publish_at FROM videos WHERE publish_at IS NOT NULL"
    ).fetchall()
    out: List[datetime] = []
    for r in rows:
        if exclude and r["id"] == exclude:
            continue
        dt = parse_dt(r["publish_at"])
        if dt:
            out.append(dt)
    return out


@dataclass
class Slot:
    when: datetime
    moved_minutes: int = 0
    reason: str = ""

    @property
    def iso(self) -> str:
        return self.when.astimezone(timezone.utc).isoformat(timespec="seconds")


def next_slot(
    kind: str,
    db: Any,
    *,
    settings: Optional[Settings] = None,
    exclude: Optional[str] = None,
    now: Optional[datetime] = None,
    horizon_days: int = MAX_HORIZON_DAYS,
) -> Optional[Slot]:
    """Earliest free, valid publish time for a video of `kind`.

    Returns None when the configured cadence is fully booked inside the horizon,
    which the caller must treat as "do not upload yet" rather than an error.
    """
    s = settings or get_settings()
    now = now or datetime.now(timezone.utc)
    zone = tz(s)
    taken = _occupied(db, exclude=exclude)
    deadline = now + timedelta(days=horizon_days)

    times = s.shorts_times if kind == "shorts" else s.longform_times
    if not times:
        log.warning("no %s publish times configured", kind)
        return None

    # Walk day by day; a full calendar of slots can legitimately be booked out.
    day = now.astimezone(zone).replace(hour=0, minute=0, second=0, microsecond=0)
    for day_offset in range(horizon_days + 1):
        for candidate in local_slots(kind, day + timedelta(days=day_offset), s):
            if candidate > deadline:
                return None
            # Must clear MIN_LEAD_MINUTES for the upload to finish first.
            if candidate - now < timedelta(minutes=MIN_LEAD_MINUTES):
                continue
            if any(abs((candidate - t).total_seconds()) < s.min_gap_minutes * 60 for t in taken):
                continue
            moved = round((candidate - now).total_seconds() / 60)
            reason = "next configured slot"
            if day_offset > 0:
                reason = f"today's slots full, {day_offset}d out"
            return Slot(when=candidate.astimezone(timezone.utc), moved_minutes=moved, reason=reason)

    log.info("no free %s slot in the next %d days", kind, horizon_days)
    return None


def assign(
    db: Any,
    video_id: str,
    kind: str,
    *,
    settings: Optional[Settings] = None,
    now: Optional[datetime] = None,
    persist: bool = True,
) -> Optional[Slot]:
    """Give a video a publish time, unless it already has a valid one.

    Idempotent: re-running the planner never moves a scheduled video, which is
    what stops a second cron run from reshuffling the calendar.

    `persist=False` plans without writing. A dry run needs this: it must be able
    to report the slot it *would* take without reserving real capacity, or
    repeated previews fill the horizon and then block the genuine upload with
    "no free publish slot".
    """
    s = settings or get_settings()
    now = now or datetime.now(timezone.utc)
    row = db.get_video(video_id)
    if row is None:
        raise ScheduleError(f"unknown video {video_id}")

    existing = parse_dt(row["publish_at"])
    if existing is not None:
        if existing > now:
            return Slot(when=existing, reason="already scheduled")
        log.info("video %s publish_at %s is in the past; replanning", video_id, existing)

    slot = next_slot(kind, db, settings=s, exclude=video_id, now=now)
    if slot is None:
        return None
    if persist:
        db.set_status(video_id, row["status"], publish_at=slot.iso)
        log.info("video %s -> publish at %s (%s)", video_id, slot.iso, slot.reason)
    else:
        log.info("video %s would publish at %s (%s) [not saved]", video_id, slot.iso, slot.reason)
    return slot


def upcoming(db: Any, *, limit: int = 20, now: Optional[datetime] = None) -> List[Dict[str, object]]:
    """The calendar, for `ytauto status`."""
    now = now or datetime.now(timezone.utc)
    rows = db.conn.execute(
        "SELECT id, kind, title, status, publish_at, youtube_id FROM videos "
        "WHERE publish_at IS NOT NULL ORDER BY publish_at LIMIT ?",
        (limit,),
    ).fetchall()
    out: List[Dict[str, object]] = []
    for r in rows:
        when = parse_dt(r["publish_at"])
        out.append(
            {
                "id": r["id"],
                "kind": r["kind"],
                "title": r["title"],
                "status": r["status"],
                "publish_at": when,
                "in": humanize_delta(when - now) if when else "?",
                "youtube_id": r["youtube_id"],
            }
        )
    return out
