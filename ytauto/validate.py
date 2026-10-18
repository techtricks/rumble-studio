"""Pre-publish quality gates.

An LLM will happily produce a 94-character title, a description full of invented
links, or a script that rambles past the render target. These checks are cheap,
deterministic, and the last thing between a hallucination and a public upload —
so they fail closed and explain themselves.

Each function returns a list of human-readable problems. Empty list = pass.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Sentence boundary, keeping the terminator so openers stay intact.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

from .util import norm_text, words

# --- title spam heuristics -------------------------------------------------
_CLICKS = (
    "you won't believe", "you wont believe", "shocking", "shockingly", "insane",
    "crazy", "mind blowing", "mindblowing", "this changes everything",
    "everything changes", "nobody is talking", "no one is talking", "secret",
    "the truth", "bombshell", "game changer", "life changing", "literally",
    "omg", "wtf", "hilarious", "awesome", "amazing", "incredible", "unbelievable",
    "literally insane", "craziest", "biggest ever", "never seen before",
)
_BAIT = ("?", "!", "...", "→", "->", "|", "🔥", "💥", "😱", "🚀")
_CLICKS_YOUTUBE = ("click here", "link in description", "subscribe now", "smash that")
# ALL-CAPS words of 4+ letters read as shouting in a title.
_SHOUT = re.compile(r"\b[A-Z]{4,}\b")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)

# --- description -----------------------------------------------------------
_SPAM_PHRASES = (
    "click here", "subscribe now", "smash that", "like and subscribe",
    "follow me on", "buy now", "limited time", "act now", "100% free",
    "guaranteed", "miracle", "crypto giveaway",
)
_URL = re.compile(r"https?://[^\s<>\"')]+", re.I)

# --- tags ------------------------------------------------------------------
_TAG_BAD = re.compile(r"[^a-z0-9\s+#]")

MAX_TAGS = 30
MAX_TAG_LEN = 30
MIN_DESC = 80
MAX_DESC = 5000


def check_title(
    title: str,
    *,
    max_len: int = 70,
    min_len: int = 8,
    niche_terms: Optional[List[str]] = None,
) -> List[str]:
    problems: List[str] = []
    t = (title or "").strip()
    if not t:
        return ["title is empty"]
    if len(t) < min_len:
        problems.append(f"title too short ({len(t)} chars, min {min_len})")
    if len(t) > max_len:
        problems.append(f"title too long ({len(t)} chars, max {max_len})")
    low = norm_text(t)
    hits = [c for c in _CLICKS if c in low]
    if hits:
        problems.append(f"clickbait phrasing: {', '.join(sorted(set(hits))[:3])}")
    if any(b in t for b in _BAIT):
        problems.append("title contains an emoji or arrow/pipe gimmick")
    if _SHOUT.search(t):
        problems.append("title contains an ALL-CAPS word")
    if _EMOJI.search(t):
        problems.append("title contains an emoji")
    # No aggregate uppercase-ratio check here. `_SHOUT` already catches real
    # shouting at 4+ letters, and a ratio test cannot tell "COMPLETELY SHOUTING"
    # from the "AI"/"GPU"/"TPU" acronyms that a tech channel title is made of —
    # in this niche it rejected titles like "Nvidia's New GPU Runs AI Four Times
    # Faster", which is exactly the kind of title we want.
    if t.count("?") > 1:
        problems.append("more than one question mark")
    if t != t.strip(" -–—|:•"):
        problems.append("title has dangling punctuation")
    if len(norm_text(t).split()) > 14:
        problems.append("title has too many words to display fully")
    return problems


def check_title_b(title_b: str, title_a: str) -> List[str]:
    """A/B variant must be a real alternative, not a synonym swap."""
    t = (title_b or "").strip()
    if not t:
        return ["title_b is empty"]
    a, b = set(norm_text(title_a).split()), set(norm_text(t).split())
    if not b:
        return ["title_b is empty"]
    if a and len(a & b) / max(1, len(a | b)) > 0.7:
        return ["title_b is too similar to title (Jaccard > 0.7) — not a real variant"]
    return []


def check_description(
    description: str,
    *,
    min_len: int = MIN_DESC,
    max_len: int = MAX_DESC,
    expected_urls: Optional[List[str]] = None,
) -> List[str]:
    problems: List[str] = []
    d = (description or "").strip()
    if len(d) < min_len:
        problems.append(f"description too short ({len(d)} chars, min {min_len})")
    if len(d) > max_len:
        problems.append(f"description too long ({len(d)} chars, max {max_len})")
    low = norm_text(d)
    hits = [p for p in _SPAM_PHRASES if p in low]
    if hits:
        problems.append(f"spam phrase: {', '.join(hits[:3])}")
    urls = _URL.findall(d)
    if len(urls) > 8:
        problems.append(f"too many links in description ({len(urls)})")
    if expected_urls is not None:
        # A description that cites a source the video did not use is a red flag
        # for hallucinated attribution.
        for u in expected_urls:
            if u and u not in d:
                problems.append(f"source URL missing from description: {u[:60]}")
    # hashtag stuffing
    if d.count("#") > 8:
        problems.append("more than 8 hashtags")
    return problems


def check_tags(tags: List[str], *, max_tags: int = MAX_TAGS) -> List[str]:
    problems: List[str] = []
    if not tags:
        return ["no tags"]
    if len(tags) > max_tags:
        problems.append(f"too many tags ({len(tags)}, max {max_tags})")
    cleaned: List[str] = []
    for t in tags:
        tag = _TAG_BAD.sub(" ", (t or "").lower())
        tag = " ".join(tag.split())
        if not tag:
            continue
        if len(tag) > MAX_TAG_LEN:
            problems.append(f"tag too long: {tag[:20]}...")
        cleaned.append(tag)
    if len(set(cleaned)) != len(cleaned):
        problems.append("duplicate tags after normalisation")
    spam = [t for t in cleaned if any(b in t for b in ("best", "top ", "must have", "#1"))]
    if spam:
        problems.append(f"spam tags: {', '.join(spam[:3])}")
    return problems


def check_script(
    script_text: str,
    *,
    target_s: int,
    wpm: int,
    tolerance: float = 0.25,
    kind: str = "longform",
) -> List[str]:
    """Flag a script whose length will not match the render target.

    Both directions matter. A script that overruns the target gets truncated or
    runs long; one that undershoots produces a video that is not the format it
    was scheduled as — a 74-second clip labelled longform against a 480s target
    is a defect, not a short video. The under-length case used to go unflagged
    entirely, so a model that ignored the word-count instruction produced a
    silently wrong video that passed every gate.
    """
    problems: List[str] = []
    n = words(script_text)
    if n < 20:
        return [f"script far too short ({n} words) — nothing to narrate"]
    est_s = n / max(1, wpm) * 60.0
    lo, hi = target_s * (1 - tolerance), target_s * (1 + tolerance)
    if est_s > hi:
        problems.append(
            f"script too long: ~{est_s:.0f}s estimated vs {target_s}s target "
            f"({n} words at {wpm} wpm)"
        )
    elif est_s < lo:
        need = int(lo / max(1, wpm) * 60.0)
        problems.append(
            f"script too short for {kind}: ~{est_s:.0f}s estimated vs "
            f"{target_s}s target ({n} words at {wpm} wpm) — needs about "
            f"{need} words. Shorten the render target or write a fuller script."
        )
    return problems


def check_repetition(
    script_text: str,
    *,
    max_opener_share: float = 0.07,
    max_opener_run: int = 3,
    min_distinct_ratio: float = 0.55,
) -> List[str]:
    """Flag a script padded to length by repeating the same sentence frame.

    A live 1232-word script was factually solid and passed every other gate
    while ending in 141 sentences of which twelve began "The test" and eleven
    began "It is" -- "The test is the proof. The rollout is the expansion. The
    partnership is the foundation." No sentence was duplicated verbatim, so an
    exact-match check sees a clean script, but a viewer hears the ending as
    padding and the retention cost is real.

    Counting two-word openers catches the frame without false-flagging the
    ordinary reuse of "the user" or "the company" that good prose does.
    """
    text = (script_text or "").strip()
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.split()) >= 3]
    if len(sents) < 12:
        # Too few sentences for a share of anything to mean.
        return []

    openers = [" ".join(s.split()[:2]).lower() for s in sents]
    counts: Dict[str, int] = {}
    for o in openers:
        counts[o] = counts.get(o, 0) + 1

    problems: List[str] = []
    total = len(openers)
    worst, worst_n = max(counts.items(), key=lambda kv: kv[1])
    if worst_n / total > max_opener_share:
        problems.append(
            f"padded script: {worst_n}/{total} sentences open with "
            f"'{worst}' ({worst_n / total:.0%})"
        )

    run = 1
    longest = 1
    for a, b in zip(openers, openers[1:]):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    if longest >= max_opener_run + 1:
        problems.append(
            f"padded script: {longest} consecutive sentences open with "
            f"'{openers[0]}'"
        )

    if len(counts) / total < min_distinct_ratio:
        problems.append(
            f"padded script: only {len(counts)} distinct sentence openers "
            f"across {total} sentences"
        )
    return problems


def check_research(research: Dict[str, Any], *, min_confidence: float = 0.4) -> List[str]:
    problems: List[str] = []
    if not research.get("angle"):
        problems.append("research has no angle")
    facts = research.get("key_facts") or []
    if len(facts) < 2:
        problems.append(f"only {len(facts)} key fact(s) — too thin to script")
    try:
        conf = float(research.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    if conf < min_confidence:
        problems.append(f"research confidence {conf:.2f} below {min_confidence}")
    return problems


def check_niche(
    title: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    niche_terms: Optional[List[str]] = None,
) -> List[str]:
    """The package must be recognisably about the channel's niche.

    Topic selection already scores relevance, but the title, description and
    tags are written by the model afterwards and can drift somewhere else
    entirely — a "why AI will save us" angle on a story about pottery, say.
    This is the last gate before words reach a public channel.

    Scoped to title + tags + description together on purpose: requiring the term
    in the title alone would reject perfectly good headlines, since plenty of
    good ones never name the subject.

    Matching is on whole words, never raw substrings. A substring test on short
    terms is worse than useless in this niche: "ai" hits expl-ai-ned, "detail",
    "email" and "available", so a video about medieval pottery sailed past the
    gate and the check reported success while protecting nothing.
    """
    terms = [t.strip().lower() for t in (niche_terms or []) if t and t.strip()]
    if not terms:
        return []
    # Two views of the same text. A set is right for single words and the only
    # safe way to match "ai" without hitting "available", but a set throws away
    # word order, which is the one thing a phrase like "electric vehicle" is
    # made of. Joining a set also never reliably contains the phrase in order,
    # so a phrase term was silently unmatchable, and in the rare case the set
    # happened to hash in order it would match the words reversed. Phrases are
    # therefore checked against the text itself.
    #
    # Hyphens become spaces on both sides. The token pattern does not span a
    # hyphen, so "self-driving" would otherwise be compared as one term against
    # a text that had already split it into two words.
    def _flatten(value: str) -> str:
        return value.lower().replace("-", " ").replace("_", " ")

    parts = [title or "", description or ""] + [str(t) for t in (tags or [])]
    text = " ".join(re.findall(r"[a-z0-9+]+", " ".join(_flatten(p) for p in parts)))
    words = set(text.split())
    terms = [_flatten(t) for t in terms]

    def _present(term: str) -> bool:
        if " " in term:                      # multi-word phrase, order matters
            return f" {term} " in f" {text} "
        return term in words

    if any(_present(t) for t in terms):
        return []
    return [
        "nothing in the title, tags or description mentions the niche "
        f"({', '.join(terms[:4])}) — this looks off-topic for this channel"
    ]


def check_chapters(chapters: List[Dict[str, Any]], kind: str) -> List[str]:
    problems: List[str] = []
    if kind == "shorts":
        if chapters:
            problems.append("Shorts must not have chapters")
        return problems
    if not chapters:
        return problems  # chapters are optional
    last = -1
    for i, ch in enumerate(chapters):
        try:
            t = int(ch.get("time_s", -1))
        except (TypeError, ValueError):
            problems.append(f"chapter {i + 1}: non-integer time_s")
            continue
        if t <= last:
            problems.append(f"chapter {i + 1}: time_s not increasing ({t} after {last})")
        last = t
        if not str(ch.get("title", "")).strip():
            problems.append(f"chapter {i + 1}: empty title")
    if chapters and chapters[0].get("time_s") not in (0, "0", 0):
        problems.append("first chapter must start at 0")
    if len(chapters) > 20:
        problems.append("too many chapters (max 20)")
    return problems


def check_thumbnail_plan(plan: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    if not plan:
        return ["no thumbnail plan"]
    text = str(plan.get("text", "")).strip()
    if not text:
        problems.append("thumbnail has no text")
    elif len(text.split()) > 5:
        problems.append(f"thumbnail text too long ({len(text.split())} words, max 5)")
    elif len(text) > 32:
        problems.append("thumbnail text longer than 32 characters")
    if not str(plan.get("emotion", "")).strip():
        problems.append("thumbnail has no emotion")
    if not str(plan.get("subject", "")).strip():
        problems.append("thumbnail has no subject")
    return problems


# ----------------------------------------------------------------- bundle
def validate_package(
    *,
    kind: str,
    title: str,
    title_b: str,
    description: str,
    tags: List[str],
    script: str,
    thumbnail_plan: Dict[str, Any],
    research: Dict[str, Any],
    chapters: Optional[List[Dict[str, Any]]] = None,
    max_title_len: int = 70,
    niche_terms: Optional[List[str]] = None,
    target_s: int = 480,
    wpm: int = 150,
    source_url: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Run every gate. Returns (errors, warnings).

    Errors block upload. Warnings are logged and stored, never fatal.
    """
    errors: List[str] = []
    warnings: List[str] = []

    errors += check_title(title, max_len=max_title_len)
    errors += check_title_b(title_b, title)
    errors += check_description(
        description, expected_urls=[source_url] if source_url else None
    )
    errors += check_tags(tags)
    errors += check_research(research)
    errors += check_thumbnail_plan(thumbnail_plan)
    errors += check_chapters(chapters or [], kind)

    errors += check_niche(title, description, tags, niche_terms)

    for problem in check_script(script, target_s=target_s, wpm=wpm):
        # Too long is worth fixing but not worth discarding a good video.
        warnings.append(problem)

    for problem in check_repetition(script):
        # Padded narration reads as filler to a viewer. A warning, not an error:
        # the facts underneath may still be good enough for a human to approve.
        warnings.append(problem)

    if kind == "shorts" and title and len(title) > 55:
        warnings.append(f"Short title is {len(title)} chars — will truncate in the Shorts feed")

    if description and "#" in description and description.count("#") > 5:
        warnings.append("description looks hashtag-heavy")

    return errors, warnings
