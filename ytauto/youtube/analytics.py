"""YouTube Analytics + Reporting API.

This is the feedback half of the system: performance per video, per day, fed
back into topic scoring so the pipeline learns what this channel's audience
actually watches instead of only what is trending.

Two things differ from the Data API and both matter:

* Analytics data is delayed 24-72h, so a "today" row is always empty. Asking
  for it wastes a call and returns nothing.
* `reports.query` costs **1000 units** — 10% of the daily budget in one
  request. It is therefore batched: one query per day for all videos, not one
  per video.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..config import Settings, get_settings
from ..db import Database, get_db, iso
from ..logging_setup import get_logger
from ..util import parse_dt
from .client import YouTubeClient, YouTubeError

log = get_logger("youtube.analytics")

ANALYTICS_COST = 1000  # not in config.COST: one batched query per day

# Metrics we store, in the order they are requested.
METRICS = (
    "views",
    "impressions",
    "impressionsClickThroughRate",
    "averageViewDuration",
    "averageViewPercentage",
    "likes",
    "comments",
    "subscribersGained",
    "estimatedRevenue",
)

# Analytics lags reality; never query a day closer than this.
LAG_DAYS = 2


class AnalyticsError(RuntimeError):
    pass


def _service() -> Any:
    """The Analytics API uses a different discovery doc than Data v3."""
    from googleapiclient.discovery import build

    from .auth import load_credentials

    creds = load_credentials(refresh=True)
    if creds is None or not getattr(creds, "valid", False):
        raise AnalyticsError("no valid credentials for the Analytics API; run `python -m ytauto auth`")
    return build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)


def fetch_daily(
    db: Database,
    *,
    days: int = 28,
    settings: Optional[Settings] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> int:
    """Pull per-video per-day metrics for the whole channel in ONE query.

    Returns the number of rows written. Skips (returns 0) when the query would
    exceed today's remaining quota, or when there is nothing to ask about.
    """
    s = settings or get_settings()
    # reports.query is not in the standard Data v3 cost table but is charged
    # 1000 units; register it so can_afford/record_quota stay accurate.
    if db.quota_remaining() < ANALYTICS_COST:
        log.warning(
            "skipping analytics: a reports.query costs %d units, only %d remain",
            ANALYTICS_COST, db.quota_remaining(),
        )
        return 0

    rows = db.conn.execute(
        "SELECT id, youtube_id, kind FROM videos WHERE youtube_id IS NOT NULL"
    ).fetchall()
    if not rows:
        log.info("no uploaded videos yet; nothing to analyse")
        return 0

    youtube_ids = [r["youtube_id"] for r in rows if r["youtube_id"]]
    if not youtube_ids:
        return 0

    today = datetime.now(timezone.utc).date()
    end_date = end or (today - timedelta(days=LAG_DAYS)).isoformat()
    start_date = start or (today - timedelta(days=days + LAG_DAYS)).isoformat()

    body = {
        "startDate": start_date,
        "endDate": end_date,
        "metrics": ",".join(METRICS),
        "dimensions": "day,video",
        "sort": "day",
    }
    # No server-side `video==` filter. The comma-joined multi-value filter syntax
    # is not dependable, this query is already one-per-channel, and unknown ids
    # are dropped client-side by `by_youtube` below — so a filter would add a
    # fragile dependency (and a 200-id ceiling) for no benefit.

    service = _service()

    def _q() -> Dict[str, Any]:
        return service.reports().query(body=body).execute()

    try:
        resp = _q()
    except Exception as exc:  # noqa: BLE001
        log.error("analytics query failed: %s", exc)
        return 0

    db.record_quota("reports.query", note=f"analytics {start_date}..{end_date}", cost=ANALYTICS_COST)

    by_youtube = {r["youtube_id"]: r["id"] for r in rows}
    written = 0
    for row in resp.get("rows", []) or []:
        # dimensions are "day,video" in that order, so day is element 0 and the
        # video id element 1. Getting this backwards files every metric under the
        # wrong date, which then silently skews the whole feedback loop.
        dims = row.get("dimensionData") or []
        if len(dims) < 2:
            log.debug("skipping analytics row with %d dimensions", len(dims))
            continue
        day = dims[0].get("value")
        vid_key = dims[1].get("value")
        video_id = by_youtube.get(vid_key) if vid_key else None
        if not day or not video_id:
            continue
        metrics = _map_metrics(row)
        metrics["day"] = day  # single source of truth for the date
        db.upsert_analytics(video_id, metrics)
        written += 1

    log.info("analytics: %d rows for %s..%s (%d tracked videos)", written, start_date, end_date, len(youtube_ids))
    if written:
        _update_topic_performance(db)
    return written


def _map_metrics(row: Dict[str]) -> Dict[str, Any]:
    """Analytics returns every metric as a string; coerce and rename."""
    def g(name: str) -> Any:
        v = row.get(name)
        if v in (None, "", "null"):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "day": row.get("dimensionData", [{}])[0].get("value"),
        "views": g("views") or 0,
        "impressions": g("impressions") or 0,
        "impressions_ctr": _pct(g("impressionsClickThroughRate")),
        "avg_view_duration_s": g("averageViewDuration"),
        "avg_view_pct": _pct(g("averageViewPercentage")),
        "likes": g("likes") or 0,
        "comments": g("comments") or 0,
        "subs_gained": g("subscribersGained") or 0,
        "estimated_revenue": g("estimatedRevenue"),
    }


def _pct(value: Optional[float]) -> Optional[float]:
    """YouTube returns CTR/view-% as a 0-1 ratio already; normalise anyway."""
    if value is None:
        return None
    return value if value <= 1.0 else value / 100.0


# ------------------------------------------------- feedback into scoring
def _update_topic_performance(db: Database) -> None:
    """Aggregate per-video performance into per-topic-key rollups.

    The topic key is a normalised keyword signature, so every video about the
    same subject accumulates into one row and `topic_performance` becomes a
    lookup the scorer can consult before choosing what to make next.
    """
    rows = db.conn.execute(
        "SELECT v.id, v.kind, v.title, t.title AS topic_title, t.summary "
        "FROM videos v LEFT JOIN topics t ON t.id = v.topic_id "
        "WHERE v.youtube_id IS NOT NULL"
    ).fetchall()

    buckets: Dict[str, Dict[str, Any]] = {}
    for v in rows:
        key = topic_key(v["topic_title"] or v["title"] or v["id"])
        agg = db.conn.execute(
            "SELECT AVG(impressions_ctr) ctr, AVG(avg_view_pct) ret, SUM(views) views, "
            "SUM(impressions) impressions FROM analytics WHERE video_id=?",
            (v["id"],),
        ).fetchone()
        if not agg or not agg["views"]:
            continue
        b = buckets.setdefault(
            key,
            {"videos": 0, "views": 0, "impressions": 0, "ctr": 0.0, "ret": 0.0},
        )
        b["videos"] += 1
        b["views"] += int(agg["views"] or 0)
        b["impressions"] += int(agg["impressions"] or 0)
        b["ctr"] += float(agg["ctr"] or 0.0)
        b["ret"] += float(agg["ret"] or 0.0)

    for key, b in buckets.items():
        db.upsert_topic_performance(
            key,
            videos=b["videos"],
            total_views=b["views"],
            total_impressions=b["impressions"],
            ctr_sum=round(b["ctr"], 4),
            retention_sum=round(b["ret"], 4),
        )
    if buckets:
        log.info("updated %d topic performance rollups", len(buckets))


_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "is", "are", "was", "were", "be", "as", "it", "its", "this",
    "that", "new", "says", "said", "how", "why", "what", "will", "has", "have",
}


def topic_key(title: str) -> str:
    """Stable signature for "what is this video about".

    The up-to-six longest content words, sorted, so word order and capitalisation
    stop mattering: "Nvidia ships a 4x GPU" and "GPU 4x from nvidia" share a key.

    It is deliberately *not* synonym-aware — "4x GPU from Nvidia" and "Nvidia
    ships 4x GPU" differ (from vs ships) and land on different keys, so their
    views do not roll up together. That errs towards splitting history, never
    towards wrongly merging two different subjects.
    """
    import re

    from ..util import norm_text

    words = [w for w in re.findall(r"[a-z0-9+]{3,}", norm_text(title)) if w not in _STOP]
    return "-".join(sorted(set(words))[:6]) or "misc"


# ------------------------------------------------------------- reporting
def summary(db: Database, *, days: int = 28) -> Dict[str, Any]:
    """Aggregate performance for `status` / the analytics report."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    row = db.conn.execute(
        "SELECT COALESCE(SUM(views),0) views, COALESCE(SUM(impressions),0) imp, "
        "AVG(impressions_ctr) ctr, AVG(avg_view_pct) ret, COALESCE(SUM(subs_gained),0) subs, "
        "COALESCE(SUM(likes),0) likes, COALESCE(SUM(comments),0) comments, "
        "COALESCE(SUM(estimated_revenue),0) rev "
        "FROM analytics WHERE day>=?",
        (since,),
    ).fetchone()

    per_video = db.conn.execute(
        "SELECT v.id, v.kind, v.title, v.youtube_id, v.publish_at, "
        "SUM(a.views) views, SUM(a.impressions) impressions, AVG(a.impressions_ctr) ctr, "
        "AVG(a.avg_view_pct) retention "
        "FROM analytics a JOIN videos v ON v.id=a.video_id "
        "WHERE a.day>=? GROUP BY v.id ORDER BY views DESC LIMIT 20",
        (since,),
    ).fetchall()

    best = db.conn.execute(
        "SELECT topic_key, videos, total_views, ROUND(ctr_sum/videos,4) ctr, "
        "ROUND(retention_sum/videos,4) retention FROM topic_performance "
        "WHERE videos>0 ORDER BY total_views DESC LIMIT 10"
    ).fetchall()

    return {
        "days": days,
        "totals": {
            "views": int(row["views"] or 0),
            "impressions": int(row["imp"] or 0),
            "ctr": round(float(row["ctr"] or 0.0), 4),
            "retention": round(float(row["ret"] or 0.0), 4),
            "subs_gained": int(row["subs"] or 0),
            "likes": int(row["likes"] or 0),
            "comments": int(row["comments"] or 0),
            "estimated_revenue": round(float(row["rev"] or 0.0), 2),
        },
        "top_videos": [dict(r) for r in per_video],
        "top_topics": [dict(r) for r in best],
    }


def performance_multiplier(db: Database, title: str) -> float:
    """Score boost for topics matching a previously successful subject.

    Clamped to [0.85, 1.25] — enough to reorder close candidates, never enough
    to make the pipeline ignore a much better story.
    """
    key = topic_key(title)
    row = db.conn.execute(
        "SELECT videos, total_views, ctr_sum FROM topic_performance WHERE topic_key=?", (key,)
    ).fetchone()
    if not row or not row["videos"]:
        return 1.0
    avg_views = (row["total_views"] or 0) / max(1, row["videos"])
    ctr = (row["ctr_sum"] or 0.0) / max(1, row["videos"])
    boost = 1.0
    if avg_views >= 1000:
        boost += 0.15
    if ctr >= 0.06:
        boost += 0.10
    return round(max(0.85, min(1.25, boost)), 3)
