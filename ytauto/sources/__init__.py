"""Topic sources registry.

`build_sources()` turns Settings into the live source list so a run is fully
described by configuration, and `harvest()` is the one call the pipeline makes.
"""

from __future__ import annotations

from typing import List, Optional

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from .base import Source, SourceError, Topic, dedupe, fetch, fetch_all
from .reddit import RedditSource
from .rss import RssSource

log = get_logger("sources")

__all__ = [
    "Source",
    "SourceError",
    "Topic",
    "RssSource",
    "RedditSource",
    "build_sources",
    "harvest",
    "dedupe",
]


def build_sources(settings: Optional[Settings] = None, only: Optional[List[str]] = None) -> List[Source]:
    s = settings or get_settings()
    wanted = {o.lower() for o in (only or [])}
    out: List[Source] = []
    if s.rss_feeds and (not wanted or "rss" in wanted):
        out.append(RssSource(s.rss_feeds))
    if s.subreddits and (not wanted or "reddit" in wanted):
        out.append(RedditSource(s.subreddits))
    return out


def harvest(limit_each: int = 25, only: Optional[List[str]] = None) -> List[Topic]:
    sources = build_sources(only=only)
    if not sources:
        log.warning("no sources configured; set RSS_FEEDS and/or SUBREDDITS")
        return []
    return fetch_all(sources, limit_each=limit_each)
