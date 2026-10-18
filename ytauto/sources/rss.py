"""RSS / Atom topic mining via feedparser (already a project dependency).

feedparser normalises RSS 2.0, Atom and RDF to one dict shape, so this module
is thin: fetch bytes -> hand to feedparser -> map entries to `Topic`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..logging_setup import get_logger
from ..util import parse_dt, strip_html
from .base import Source, SourceError, Topic, dedupe, fetch

log = get_logger("sources.rss")


class RssSource(Source):
    name = "rss"

    def __init__(self, feeds: List[str], per_feed: int = 15) -> None:
        self.feeds = [f for f in feeds if f]
        self.per_feed = per_feed

    def harvest(self, limit: int = 25) -> List[Topic]:
        """Collect topics across every feed, capped at `limit` in total.

        `limit` caps the combined result, but the budget must be shared: the
        loop used to break as soon as ONE feed had filled the quota, so the
        later feeds — usually the India-specific ones — silently returned
        nothing cycle after cycle. Now every feed is visited up to its own
        `per_feed` depth, and the total cap is applied ROUND-ROBIN across
        feeds, so a chatty wire feed can no longer crowd out a quiet museum
        blog.
        """
        import feedparser

        per_feed_lists: List[List[Topic]] = []
        for url in self.feeds:
            try:
                raw = fetch(url, as_json=False)
            except Exception as exc:  # noqa: BLE001 - a dead feed is not fatal
                log.warning("feed skipped %s: %s", url, exc)
                continue
            parsed = feedparser.parse(raw)
            if parsed.bozo and not parsed.entries:
                log.warning("feed unparseable %s: %s", url, parsed.get("bozo_exception", ""))
                continue
            host = _host(url)
            got = []
            for entry in parsed.entries[: self.per_feed]:
                topic = self._to_topic(entry, host, url)
                if topic:
                    got.append(topic)
            if got:
                per_feed_lists.append(got)
        # Round-robin merge: one topic from each feed in turn keeps every
        # source represented once the total is capped.
        merged: List[Topic] = []
        idx = 0
        longest = max((len(l) for l in per_feed_lists), default=0)
        while idx < longest and len(merged) < limit:
            for feed_list in per_feed_lists:
                if idx < len(feed_list) and len(merged) < limit:
                    merged.append(feed_list[idx])
            idx += 1
        return dedupe(merged)

    def _to_topic(self, entry: Dict[str, Any], host: str, feed_url: str) -> Optional[Topic]:
        title = strip_html(entry.get("title", ""))
        if not title:
            return None
        link = (entry.get("link") or "").strip()
        if not link:
            for link_obj in entry.get("links", []) or []:
                if link_obj.get("rel") == "alternate" and link_obj.get("href"):
                    link = link_obj["href"].strip()
                    break
        summary = _summary(entry)
        published = _published_at(entry)
        return Topic(
            title=title,
            url=link or None,
            source="rss",
            # No universal popularity signal in RSS. Use position in the feed:
            # editors put what matters most at the top.
            source_score=0.5,
            summary=summary,
            published_at=published,
            raw={
                "feed": feed_url,
                "host": host,
                "author": entry.get("author", ""),
                "tags": [t.get("term", "") for t in (entry.get("tags") or [])][:8],
                "feed_rank": entry.get("_feed_rank", 0),
            },
        )


def _host(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return (urlparse(url).hostname or "").replace("www.", "")
    except Exception:  # pragma: no cover - malformed URL
        return ""


def _summary(entry: Dict[str, Any]) -> str:
    # Take the RICHER body, not the first one seen: a <description> teaser of
    # 120 characters loses against a full content:encoded body of two
    # paragraphs, and research grounded on the teaser comes back thin. Prefer
    # whichever candidate text is genuinely longer.
    candidates: List[str] = []
    for key in ("summary", "description", "subtitle"):
        val = entry.get(key)
        if val:
            text = strip_html(val)
            if text:
                candidates.append(text)
    content = entry.get("content") or []
    if content and isinstance(content, list):
        text = strip_html(content[0].get("value", ""))
        if text:
            candidates.append(text)
    return max(candidates, key=len) if candidates else ""


def _published_at(entry: Dict[str, Any]) -> Optional[str]:
    """Normalise a feed's publication date to ISO-8601 UTC.

    The parsed struct wins over the raw string. Preferring the raw text (as this
    used to) stored whatever format the feed happened to use — typically
    RFC-2822 like "Sat, 25 Sep 2026 09:00:00 +0000" — and `parse_dt` cannot read
    that, so scoring silently fell back to `created_at`. Every story then looked
    hours old, and recency ranking did nothing at all on real feeds.
    """
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        if entry.get(key):
            iso = _from_struct(entry[key])
            if iso:
                return iso
    for key in ("published", "updated", "created", "dc:date"):
        raw = entry.get(key)
        if not raw:
            continue
        iso = parse_dt(raw)
        if iso:
            return iso
        # RFC-2822 / RFC-822, the classic RSS date format.
        try:
            from email.utils import parsedate_to_datetime

            return parsedate_to_datetime(str(raw)).astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError):
            continue
    return None


def _from_struct(struct: Any) -> Optional[str]:
    import calendar
    from datetime import datetime, timezone

    try:
        return datetime.fromtimestamp(calendar.timegm(struct), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        return None
