"""Fetch the article behind a headline when the feed gave us too little.

An RSS `<description>` is meant to be a teaser. Measured across the five feeds
this project follows, real entries carried 118-367 characters of plain text.
That is not enough to ground a 1200-word script: research came back with zero
key facts, the confidence gate rejected the topic, and the pipeline produced
nothing at all. Fetching the article itself costs nothing and hands the
research step real prose to quote from.

This is deliberately a stdlib-only extractor rather than a readability
library. It is not trying to be the best article parser on the web; it needs to
turn a news page into a few thousand characters of body text, and it has to do
that with no new dependency to keep the install honest.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import List, Optional, Tuple

from ..logging_setup import get_logger
from .base import fetch

log = get_logger(__name__)

# Below this the feed gave us a teaser, not something to script from. Measured
# against real entries this sat between 118 and 367 characters.
MIN_USEFUL_CHARS = 400

# Hard ceiling on how much text we keep. The research prompt clips to 6000
# characters anyway, and a bounded field keeps the topics table readable.
MAX_ARTICLE_CHARS = 12000

# Chrome that is never the story. Dropped whole rather than filtered, so
# "Subscribe" and "Share this article" do not become evidence.
_DROP_TAGS = frozenset(
    """script style noscript nav header footer aside form figure figcaption
    svg iframe button select input template head title""".split()
)

# Block-level tags become paragraph breaks; without this, HTML without
# whitespace between <p> elements collapses into one unreadable run-on string.
_BLOCK_TAGS = frozenset(
    """p div br li tr section article h1 h2 h3 h4 h5 h6 blockquote pre
    dd dt""".split()
)

_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")


# Void elements never have a closing tag. Counting <input> as "entering" chrome
# without an </input> to leave it meant the skip depth stayed raised for the rest
# of the document and the extractor returned only the first fifteen characters
# of the page. It is a one-character bug that silently gutted every article.
_VOID_TAGS = frozenset(
    """area base br col embed hr img input link meta source track wbr param""".split()
)


class _ArticleText(HTMLParser):
    """Collect visible text, dropping navigation and keeping paragraph breaks.

    Chrome is tracked as a stack rather than a counter. Real-world markup
    leaves `<p>` and `<div>` unclosed and never closes void elements at all, so
    a counter drifts upwards and swallows the entire article. Popping to the
    matching open tag makes the parse tolerant of that.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._drop_stack: List[str] = []
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in _VOID_TAGS:
            return
        if tag in _DROP_TAGS:
            self._drop_stack.append(tag)
        elif not self._drop_stack and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:  # noqa: ANN001
        # Self-closing form: <input ... />, <br />
        if not self._drop_stack and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:  # noqa: ANN001
        if tag in self._drop_stack:
            # Unwind to this tag, discarding anything left unclosed inside it.
            while self._drop_stack:
                if self._drop_stack.pop() == tag:
                    break
            return
        if not self._drop_stack and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._drop_stack and data.strip():
            self.parts.append(data)

    def text(self) -> str:
        joined = "".join(self.parts)
        out = "\n".join(_WS.sub(" ", ln).strip() for ln in joined.split("\n"))
        return _BLANKS.sub("\n\n", out).strip()


def extract_text(html: str) -> str:
    """Visible body text from an HTML document."""
    if not html or not html.strip():
        return ""
    p = _ArticleText()
    try:
        p.feed(html)
        p.close()
    except Exception as exc:  # noqa: BLE001 - malformed markup is routine
        log.debug("article parse stopped early: %s", exc)
    return p.text()[:MAX_ARTICLE_CHARS]


def needs_enrichment(summary: str, threshold: int = MIN_USEFUL_CHARS) -> bool:
    """True when the feed text is too thin to ground a script."""
    return len((summary or "").strip()) < threshold


def fetch_article_text(
    url: str, *, timeout: int = 20, attempts: int = 2
) -> Optional[str]:
    """Best-effort article text. Returns None rather than raising.

    A dead link, a paywall, or a 403 must not stop the run: the topic simply
    stays thin and the existing confidence gate decides its fate on the little
    text there is.
    """
    if not url or not url.startswith(("http://", "https://")):
        return None
    try:
        body = fetch(url, timeout=timeout, attempts=attempts, as_text=True)
    except Exception as exc:  # noqa: BLE001
        log.info("article fetch failed for %s: %s", url[:60], exc)
        return None
    if not isinstance(body, str):
        return None
    text = extract_text(body)
    if not text:
        return None
    return text


def enrich(summary: str, url: str, *, threshold: int = MIN_USEFUL_CHARS) -> str:
    """Return article text if the feed's was too thin, else the feed's own.

    Never raises and never returns less than it was given, so a caller can use
    it unconditionally.
    """
    if not needs_enrichment(summary, threshold):
        return summary or ""
    text = fetch_article_text(url)
    if not text:
        return summary or ""
    # Keep the teaser too: it is often the outlet's own framing of the story.
    combined = f"{summary.strip()}\n\n{text}".strip() if summary else text
    log.info(
        "enriched %s: %d -> %d chars", (url or "")[:60], len(summary or ""),
        len(combined),
    )
    return combined[:MAX_ARTICLE_CHARS]
