"""Stage 1 — discover: harvest sources, score, keep the best of them.

Scoring is where most of the quality lives, so it is explicit rather than
hidden in a SQL ORDER BY: niche fit, freshness decay, source popularity, and a
performance multiplier learned from what this channel already did well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import Settings, get_settings
from ..db import Database, get_db
from ..logging_setup import get_logger
from ..sources import harvest
from ..util import clip, norm_text

log = get_logger("pipeline.discover")


@dataclass
class Discovered:
    fetched: int = 0
    added: int = 0
    duplicates: int = 0
    scored: int = 0
    kept: List[Dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.kept is None:
            self.kept = []


def run(
    *,
    db: Optional[Database] = None,
    settings: Optional[Settings] = None,
    limit_each: int = 45,
    only: Optional[List[str]] = None,
    apply_performance_boost: bool = True,
) -> Discovered:
    """Harvest -> persist -> score -> return the best candidates.

    Nothing is auto-published here. This stage only fills the topic pool.
    """
    s = settings or get_settings()
    db = db or get_db()

    topics = harvest(limit_each=limit_each, only=only)
    if not topics:
        log.warning("harvest returned nothing (sources down, or all filtered?)")
        return Discovered()

    result = db.ingest_topics(topics)
    scored = db.score_topics(s.niche_terms, half_life_h=s.topic_half_life_h)
    result = Discovered(
        fetched=len(topics),
        added=result["added"],
        duplicates=result["duplicates"],
        scored=scored,
    )

    rows = db.top_topics(
        limit=s.max_topics_per_day,
        min_score=s.min_total_score,
        min_niche=s.min_niche_score,
    )

    from ..youtube.analytics import performance_multiplier

    kept: List[Dict[str, Any]] = []
    for r in rows:
        title = r["title"]
        mult = performance_multiplier(db, title) if apply_performance_boost else 1.0
        kept.append(
            {
                "id": r["id"],
                "title": title,
                "url": r["url"],
                "source": r["source"],
                "niche": round(r["niche_score"] or 0.0, 3),
                "fresh": round(r["freshness"] or 0.0, 3),
                "total": round((r["total_score"] or 0.0) * mult, 4),
                "boost": mult,
                "published_at": r["published_at"],
            }
        )
    kept.sort(key=lambda t: t["total"], reverse=True)
    result.kept = kept

    log.info(
        "discover: %d fetched, %d new (%d dupes), %d scored, %d above threshold",
        result.fetched, result.added, result.duplicates, scored, len(kept),
    )
    return result


def preview(db: Database, limit: int = 15) -> List[Dict[str, Any]]:
    """What `discover` would pick, without touching the network."""
    rows = db.top_topics(limit=limit)
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "source": r["source"],
            "niche": round(r["niche_score"] or 0.0, 3),
            "fresh": round(r["freshness"] or 0.0, 3),
            "total": round(r["total_score"] or 0.0, 3),
        }
        for r in rows
    ]


def add_manual(
    db: Database,
    title: str,
    url: Optional[str] = None,
    summary: str = "",
) -> str:
    """Seed a topic by hand. Returns the topic id.

    Scored with the real niche terms (not a wildcard) so a hand-seeded topic
    competes on equal footing with crawled ones instead of being auto-promoted
    to a perfect niche score.
    """
    title = (title or "").strip()
    if not title:
        # norm_text() turns whitespace into "", which would otherwise sail
        # through as a topic with a blank title and be unscoreable forever.
        raise ValueError("a topic needs a non-empty title")
    tid = db.add_topic(
        title=title, source="manual", url=url, source_score=1.0, summary=summary
    )
    if tid is None:
        raise ValueError(f"already tracking this topic: {title}")
    s = get_settings()
    db.score_topic(tid, s.niche_terms, half_life_h=s.topic_half_life_h)
    log.info("manual topic added: %s (%s)", clip(title, 60), tid)
    return tid
