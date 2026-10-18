"""Shared plumbing for topic sources.

Every source is a network call to a server we do not control, so all of them
route through `fetch` (timeout + UA + bounded retry) and return `Topic`
candidates instead of touching the database. Persistence is the caller's job —
that keeps sources trivially testable offline.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..logging_setup import get_logger
from ..util import digest, norm_text, retry, strip_html

log = get_logger("sources")

# Feeds and Reddit throttle or 403 hard on default python-requests UA.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36 ytauto/1.0 (+topic-research)"
)

MAX_SUMMARY = 1200
MAX_BODY = 400_000


def decode_body(body: bytes, resp: Any = None) -> str:
    """Decode a response body to str, preferring the declared charset.

    News pages are not reliably UTF-8. Guessing wrong turns a curly quote into
    mojibake, which then lands in the research prompt as unquotable evidence.
    """
    if isinstance(body, str):
        return body
    enc = None
    if resp is not None:
        enc = getattr(resp, "encoding", None)
        if not enc or enc.lower() == "iso-8859-1":
            # requests falls back to ISO-8859-1 for text/* with no charset,
            # which mangles UTF-8 pages. The HTML meta tag is more honest.
            enc = None
            head = body[:2048].decode("ascii", "ignore")
            m = re.search(r"charset\s*=\s*[\"']?([\w-]+)", head, re.I)
            if m:
                enc = m.group(1)
    for candidate in (enc, "utf-8", "cp1252", "latin-1"):
        if not candidate:
            continue
        try:
            return body.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", "replace")


class SourceError(RuntimeError):
    """A source failed in a way the pipeline should log and skip."""


@dataclass
class Topic:
    """One candidate story. Mirrors the columns of the `topics` table."""

    title: str
    url: Optional[str] = None
    source: str = "manual"
    source_score: float = 0.0
    summary: str = ""
    published_at: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Identity for dedupe. URL when we have one, else normalised title."""
        return self.url or digest(self.title)

    @property
    def clean_title(self) -> str:
        return strip_html(self.title)

    def normalised_title(self) -> str:
        return norm_text(self.clean_title)

    def to_row(self) -> Dict[str, Any]:
        return {
            "title": self.clean_title[:300],
            "url": self.url,
            "source": self.source,
            "source_score": round(float(self.source_score), 4),
            "summary": self.summary[:MAX_SUMMARY],
            "published_at": self.published_at,
            "raw": self.raw,
        }


def fetch(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 20,
    attempts: int = 3,
    as_json: bool = False,
    as_text: bool = False,
) -> Any:
    """GET a URL with a real UA, a hard timeout and bounded retries.

    `as_json` returns the parsed body. `as_text` decodes to str, honouring the
    charset the server declared. Without it callers get bytes and have to guess
    an encoding, which is how the article extractor silently returned None on
    every single request: `isinstance(body, str)` was false for all of them, so
    the enrichment it was built for never ran and the pipeline fell back to
    whatever the feed teaser happened to contain.
    """
    import requests

    hdrs = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json" if as_json else "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    hdrs.update(headers or {})

    def _get() -> Any:
        resp = requests.get(url, headers=hdrs, timeout=timeout)
        if resp.status_code == 429:
            raise SourceError(f"429 rate limited: {url}")
        # 403/404 on a feed is normal (feed went away); do not burn retries.
        if resp.status_code in (401, 403, 404, 410):
            raise SourceError(f"HTTP {resp.status_code}: {url}")
        # Wrap transport/HTTP failures so callers only ever have to handle
        # SourceError. Letting a raw requests.HTTPError out for a 503 while a
        # 404 arrives as SourceError means a caller cannot write one
        # `except SourceError` clause and be right.
        try:
            resp.raise_for_status()
            body = resp.content[:MAX_BODY]
            if as_json:
                return resp.json()
            if as_text:
                return decode_body(body, resp)
            return body
        except requests.RequestException as exc:
            raise SourceError(f"HTTP {resp.status_code}: {url} ({exc})") from exc

    def _note(attempt: int, exc: Exception) -> None:
        log.warning("fetch %s (attempt %d/%d) failed: %s", url, attempt, attempts, exc)

    return retry(_get, attempts=attempts, on_error=_note)


class Source:
    """Base class. Subclasses implement `name` and `harvest`."""

    name = "base"

    def harvest(self, limit: int = 25) -> List[Topic]:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<{type(self).__name__} {self.name}>"


def dedupe(topics: List[Topic]) -> List[Topic]:
    """Collapse repeats within one harvest, newest-looking first preserved."""
    seen: set = set()
    out: List[Topic] = []
    for t in topics:
        if not t.clean_title:
            continue
        k = t.key
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def recency_hours(published_at: Optional[str]) -> float:
    from ..util import parse_dt

    dt = parse_dt(published_at)
    if dt is None:
        return 999.0
    from datetime import datetime, timezone

    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)


def fetch_all(sources: List[Source], limit_each: int = 25) -> List[Topic]:
    """Run every source, never letting one bad feed kill the harvest."""
    out: List[Topic] = []
    for src in sources:
        t0 = time.time()
        try:
            got = src.harvest(limit_each)
            log.info("source %-10s -> %3d topics (%.1fs)", src.name, len(got), time.time() - t0)
            out.extend(got)
        except Exception as exc:  # noqa: BLE001 - one dead feed must not stop the rest
            log.error("source %s failed: %s", src.name, exc)
    return dedupe(out)
