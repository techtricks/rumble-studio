"""Small shared helpers. No third-party imports, no side effects.

Python 3.9 cannot parse a trailing "Z" with `datetime.fromisoformat`, and the
whole codebase round-trips UTC timestamps through SQLite as text. Everything
that reads a timestamp back goes through `parse_dt` so a "Z" ever written by an
older run (or by hand in a test) can never blow up a cron job.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional, TypeVar

T = TypeVar("T")

_WS = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_NONWORD = re.compile(r"[^a-z0-9 ]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# ----------------------------------------------------------------- time
def parse_dt(value: Any) -> Optional[datetime]:
    """Parse anything we may have stored: ISO string, epoch seconds, datetime.

    Always returns an aware datetime in UTC, or None when unparseable.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def humanize_delta(delta: timedelta) -> str:
    total = int(abs(delta).total_seconds())
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    if total < 86400:
        return f"{total // 3600}h{(total % 3600) // 60:02d}m"
    return f"{total // 86400}d{(total % 86400) // 3600:02d}h"


def humanize_duration(seconds: Optional[int]) -> str:
    if not seconds or seconds < 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ----------------------------------------------------------------- text
def strip_html(text: str) -> str:
    """Feeds hand us HTML inside <description>. Flatten it to plain text."""
    if not text:
        return ""
    out = _TAG_RE.sub(" ", str(text))
    for entity, char in (
        ("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&quot;", '"'), ("&#39;", "'"), ("&apos;", "'"), ("&#8217;", "'"),
        ("&mdash;", "-"), ("&ndash;", "-"), ("&hellip;", "..."),
    ):
        out = out.replace(entity, char)
    out = re.sub(r"&#\d+;", " ", out)
    return _WS.sub(" ", out).strip()


def norm_text(text: str) -> str:
    """Lowercase, de-accented, single-spaced. Used for dedupe + keyword match."""
    out = unicodedata.normalize("NFKD", str(text or ""))
    out = "".join(c for c in out if not unicodedata.combining(c))
    return _WS.sub(" ", out).lower().strip()


def slugify(text: str, max_len: int = 60) -> str:
    out = _NONWORD.sub("", norm_text(text).replace(" ", "-"))
    out = re.sub(r"-{2,}", "-", out).strip("-")
    return out[:max_len].strip("-")


def clip(text: str, limit: int, ellipsis: str = "…") -> str:
    text = _WS.sub(" ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - len(ellipsis))]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:-") + ellipsis


def sentences(text: str) -> list:
    return [s.strip() for s in _SENTENCE_END.split(_WS.sub(" ", str(text or "")).strip()) if s.strip()]


def words(text: str) -> int:
    return len(norm_text(text).split())


_TOKENS = re.compile(r"[a-z0-9+]+")


def tokens(text: str) -> set:
    """Split text into comparable word tokens.

    Use this instead of `term in text` for keyword matching. Raw substring
    tests are quietly wrong for short niche keywords: in a "tech, ai, gadgets"
    channel, "ai" matches expl-ai-ned, "email" and "detail", and "tech" matches
    "techniques" — so a story about medieval pottery scored a perfect 1.0
    relevance and jumped to the top of the queue. Word tokens cannot do that.
    """
    return set(_TOKENS.findall(norm_text(text)))


def has_token(text: str, term: str) -> bool:
    """Whole-word (or whole-phrase) presence test for one keyword."""
    t = (term or "").strip().lower()
    if not t:
        return False
    if " " in t:
        return t in norm_text(text)
    return t in tokens(text)


def digest(*parts: str) -> str:
    """Stable short hash, used to dedupe topics without storing the full text."""
    h = hashlib.sha256()
    for p in parts:
        h.update(norm_text(p).encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def title_case(text: str) -> str:
    small = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on",
             "or", "the", "to", "up", "vs", "with", "from", "is", "are", "it"}
    words_ = str(text or "").split()
    out = []
    for i, w in enumerate(words_):
        out.append(w.lower() if (i and w.lower() in small) else w)
    return " ".join(out)


# --------------------------------------------------------------- retries
def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    factor: float = 2.0,
    jitter: float = 0.25,
    retry_on: Iterable[type] = (Exception,),
    give_up: Optional[Callable[[Exception], bool]] = None,
    on_error: Optional[Callable[[int, Exception], None]] = None,
) -> T:
    """Bounded exponential backoff with jitter.

    Every outbound HTTP call in this project goes through here: a flaky RSS
    server or a 429 from the Anthropic API should delay a cron run, never
    abort it.

    `give_up` marks errors that retrying cannot fix — a 404 for a model that no
    longer exists, a 400 for a malformed request. Retrying those just multiplies
    the wall-clock cost by `attempts` before reporting the same thing: a Gemini
    call that 404s in 1s can otherwise take four attempts and minutes.
    """
    retry_on = tuple(retry_on)
    last: Optional[BaseException] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return fn()
        except retry_on as exc:
            last = exc
            if give_up is not None and give_up(exc):
                raise
            if attempt >= max(1, attempts):
                break
            if on_error:
                on_error(attempt, exc)
            delay = base_delay * (factor ** (attempt - 1))
            delay *= 1.0 + random.uniform(-jitter, jitter)
            time.sleep(max(0.0, delay))
    assert last is not None
    raise last


def chunked(items: Iterable[T], size: int) -> Iterable[list]:
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
