"""Reddit topic mining via each subreddit's official Atom feed.

Reddit's JSON API (`/r/<sub>/hot.json`) now 403s most non-browser clients from
datacenter IPs, and `old.reddit.com` answers the same path with an HTML page
rather than JSON. The endpoint that *is* officially supported and returns clean
`application/atom+xml` to any client is `https://www.reddit.com/r/<sub>/.rss`,
so that is what we read — via the same feedparser path as RSS sources.

Cost: we lose the upvote `score` (it is not in the feed). Position in the feed
stands in for it. Rate limits are aggressive, so requests are spaced and 429 is
treated as a long, quiet backoff instead of an error.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..logging_setup import get_logger
from ..util import strip_html
from .base import Source, Topic, fetch

log = get_logger("sources.reddit")

FEED = "https://www.reddit.com/r/{sub}/.rss"
# Anonymous Reddit allows very little. Measured: 10s still trips 429 during a
# harvest burst, so 15s is the default. Slower, but a 429 costs the whole sub.
MIN_GAP_SECONDS = 15.0

_BLOCKED = ("[removed]", "[deleted]", "removed by moderators")

# A subreddit's .rss feed mixes link posts (an external article) with self posts
# (text written in Reddit, whose <link> points back at reddit.com). For a
# news-style channel the self posts are not topics at all — they are AMAs,
# "look at my project" showcases and advice threads. Unfiltered, they routinely
# outrank real stories: a live run picked "Hi Reddit, I'm Katie Paul, a Reuters
# journalist who..." as the single best available topic.
#
# The reliable signal is the link host rather than the title. Title patterns are
# a weak backstop ("I built a thing" is sometimes genuinely newsworthy), so they
# only reject the unambiguous cases.
_SELF_POST_HOSTS = ("reddit.com", "redd.it")
_SELF_POST_TITLES = (
    "hi reddit",
    "ama",
    "i am",
    "i'm ",
    "ask me anything",
    "upvote",
    "karma",
    "how do i",
    "why do i",
    "help me",
    "anyone else",
    "what is the",
    "is it just me",
    "my experience",
    "tips for",
    "need advice",
    "should i",
)


def _is_reddit_host(url: str) -> bool:
    from urllib.parse import urlparse

    host = (urlparse(url or "").netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return any(host == h or host.endswith("." + h) for h in _SELF_POST_HOSTS)


def _outbound_link(entry: Dict[str, Any], link: str) -> str:
    """The external article a link post points at, if there is one.

    Reddit's Atom feed is not consistent about which field carries it: some
    entries put the outbound URL in <link>, others only expose it as an
    alternate link, and self posts put the outbound article inside the summary
    HTML. Checking `link` alone therefore rejects a mix of real stories along
    with the chatter, which is how an earlier version silently lost the entire
    subreddit.
    """
    if link and not _is_reddit_host(link):
        return link
    for alt in entry.get("links") or []:
        href = (alt.get("href") or "").strip()
        if href and not _is_reddit_host(href) and href.startswith("http"):
            return href
    body = (entry.get("summary") or entry.get("description") or "")
    if body:
        import re

        m = re.search(r'href="(https?://[^"]+)"', body)
        if m and not _is_reddit_host(m.group(1)):
            return m.group(1)
    return ""


def _is_self_post(title: str, body: str) -> bool:
    """True when this entry is Reddit chatter rather than a linked story.

    Uses the structural marker first: Reddit renders a text post's body inside
    `<div class="md">`, while a link post's body is a `<table>` holding a
    thumbnail. That is far more dependable than the title.

    The title patterns are the backstop for entries whose body is missing.

    Note what is deliberately *not* a rejection: an entry with no extractable
    outbound URL. Reddit's .rss no longer publishes the article link at all
    (verified: the only hrefs in a link post's body point back at reddit.com),
    so that rule rejected the entire subreddit rather than the chatter. A
    headline with no source is weaker material, not useless material.
    """
    if 'class="md"' in (body or ""):
        return True
    low = (title or "").strip().lower()
    if low in _SELF_POST_TITLES:
        return True
    # Every entry in the list is chatter on sight. Note what is deliberately
    # absent: a bare "i " prefix, because "I built a thing that..." is often the
    # actual news, so a first-person title survives.
    return low.startswith(_SELF_POST_TITLES)


class RedditSource(Source):
    name = "reddit"

    def __init__(self, subreddits: List[str], limit_each: int = 12) -> None:
        self.subreddits = [s.strip().strip("/").replace("r/", "", 1) for s in subreddits if s.strip()]
        self.per_sub = limit_each
        self._last_call = 0.0

    def harvest(self, limit: int = 25) -> List[Topic]:
        topics: List[Topic] = []
        for sub in self.subreddits:
            self._throttle()
            try:
                topics.extend(self._harvest_one(sub))
            except Exception as exc:  # noqa: BLE001 - one blocked sub must not stop the rest
                log.warning("subreddit r/%s failed: %s", sub, exc)
        return topics

    def _throttle(self) -> None:
        gap = time.time() - self._last_call
        if gap < MIN_GAP_SECONDS:
            time.sleep(MIN_GAP_SECONDS - gap)
        self._last_call = time.time()

    def _harvest_one(self, sub: str) -> List[Topic]:
        import feedparser

        url = FEED.format(sub=sub)
        raw = fetch(url, as_json=False, attempts=2)
        parsed = feedparser.parse(raw)
        if parsed.bozo and not parsed.entries:
            log.warning("r/%s feed unparseable: %s", sub, parsed.get("bozo_exception", ""))
            return []

        out: List[Topic] = []
        for rank, entry in enumerate(parsed.entries[: self.per_sub]):
            topic = self._to_topic(entry, sub, rank)
            if topic:
                out.append(topic)
        return out

    def _to_topic(self, entry: Dict[str, Any], sub: str, rank: int) -> Optional[Topic]:
        title = strip_html(entry.get("title", ""))
        if not title or title.strip().lower() in _BLOCKED:
            return None
        link = (entry.get("link") or "").strip()
        body = self._body_raw(entry)
        if _is_self_post(title, body):
            log.debug("r/%s skipping self post: %s", sub, title[:60])
            return None
        outbound = _outbound_link(entry, link)
        summary = strip_html(body)
        return Topic(
            title=title,
            url=outbound or None,
            source="reddit",
            # Feed order is Reddit's own ranking, newest+hotest first.
            source_score=round(max(0.0, 1.0 - rank / max(1, self.per_sub)), 4),
            summary=summary,
            published_at=entry.get("updated") or entry.get("published") or None,
            raw={"subreddit": sub, "feed_rank": rank, "author": _author(entry)},
        )

    @staticmethod
    def _body_raw(entry: Dict[str, Any]) -> str:
        """The entry body with its HTML intact, for structural checks."""
        for key in ("summary", "description"):
            text = entry.get(key, "") or ""
            if text:
                return text
        content = entry.get("content") or []
        if content and isinstance(content, list):
            return content[0].get("value", "") or ""
        return ""

    @staticmethod
    def _body(entry: Dict[str, Any]) -> str:
        for key in ("summary", "description"):
            text = strip_html(entry.get(key, "") or "")
            if text:
                return text
        content = entry.get("content") or []
        if content and isinstance(content, list):
            return strip_html(content[0].get("value", ""))
        return ""


def _author(entry: Dict[str, Any]) -> str:
    author = entry.get("author_detail") or {}
    return author.get("name", "") if isinstance(author, dict) else ""
