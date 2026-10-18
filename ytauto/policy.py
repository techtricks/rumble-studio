"""What has to be true before a video is allowed to be published.

This module is the difference between a channel that can be monetised and one
that cannot. YouTube's Partner Program pays for "original and authentic" content
and has spent years pulling monetisation from channels that look automated. The
15 July 2025 clarification renamed the old "repetitious content" rule to
"inauthentic content" and made the examples concrete. Four of them describe this
project's original design almost word for word:

* "Reading materials you didn't create (e.g., articles from websites)" — a
  pipeline that narrates a tech-news feed is the textbook case.
* "Image slideshows, templated storylines, or scrolling text with minimal or no
  narrative, commentary, or educational value."
* "channels where all slideshows use the same voiceover across videos".
* "AI-generated content made with generic or unoriginal templates giving the
  impression of mass production without adding the creator's original,
  authentic insights or perspective."

And this is enforced at channel level: one flagged video can cost the whole
channel its monetisation, not just that video's revenue. So the checks here are
hard failures, not warnings. A video that cannot clear them is left unrendered
for a human to fix.

Two rules are not optional and are handled by the platform rather than by us:

* Synthetic content must be disclosed. Our narration is a synthesised voice and
  some visuals are sourced photographs, so `build_description` states this in
  the video's own description. The matching "altered content" toggle lives in
  YouTube Studio and has no API surface, so it is surfaced as a manual step.
* Reused content is judged on the whole. `check_originality` compares a script
  against what the channel has already published, because a channel whose
  videos read as interchangeable fails even when every single video is fine.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .logging_setup import get_logger

log = get_logger(__name__)

# Below this similarity against an already-published script, the two videos are
# close enough that a viewer scrolling the channel would call them the same
# video. There is no official number; this is deliberately strict, because the
# cost of being wrong is the channel, not one upload.
MAX_SCRIPT_SIMILARITY = 0.45

# A script has to contain some reasoning, not just restated facts. Counted in
# clauses that add interpretation rather than in word count, so padding cannot
# satisfy it.
_ANALYSIS_MARKERS = (
    "why", "because", "which means", "that suggests", "the risk", "the catch",
    "compared", "in practice", "what this means", "the takeaway", "however",
    "but that", "the interesting", "worth noting", "the trade-off", "the tradeoff",
    "so what", "implication", "unlike", "whereas", "the real question",
    "does not mean", "does not necessarily",
)

_WORDS = re.compile(r"[a-z0-9']+")

# --- how original is this, against the article it came from -----------------
#
# Counting "does this sentence contain the word however" measures our prompt, not
# the script: it can be satisfied by a script that is mostly restatement, and it
# fails a script that reasons well in different words. Comparing against the
# source text is harder to argue with. A restatement can only re-use the
# source's own phrasing, so it scores high on shared 4-grams and low on vocabulary
# the source never used. Analysis has to introduce words, because connecting two
# facts is not something the article did.
_SHARE_NGRAM = 4
_MAX_SOURCE_NGRAM_SHARE = 0.25
_MIN_NOVEL_WORD_SHARE = 0.25


@dataclass
class PolicyResult:
    """A compliance verdict for one video.

    `ok` is derived, never stored. It was a plain field first, and a check that
    appended a violation to a result that had been created with `ok=True` still
    reported `ok=True` — which turns a hard gate into a suggestion at exactly
    the moment a violation is found. Deriving it makes that impossible.
    """

    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def fail(self, reason: str) -> "PolicyResult":
        self.violations.append(reason)
        return self

    def warn(self, reason: str) -> "PolicyResult":
        self.warnings.append(reason)
        return self

    def reason(self) -> str:
        return "; ".join(self.violations)


def _tokens(text: str) -> List[str]:
    return _WORDS.findall((text or "").lower())


def _content_words(text: str) -> List[str]:
    """Lowercased words of three letters or more, for overlap comparison.

    Short words and digits are dropped: "a", "of" and "80" match everything, so
    including them would push a genuine restatement's score down and make the
    check look like it passed.
    """
    return [w for w in re.findall(r"[a-z]{3,}", (text or "").lower())]


def script_similarity(a: str, b: str) -> float:
    """How interchangeable two scripts are, 0.0 to 1.0."""
    ta, tb = _tokens(a), _tokens(b)
    if len(ta) < 20 or len(tb) < 20:
        # Too short to judge; do not invent a reason to block.
        return 0.0
    return difflib.SequenceMatcher(None, ta, tb).ratio()


def analysis_ratio(script: str) -> float:
    """Share of sentences that add interpretation rather than restate facts."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", script or "") if s.strip()]
    if not sentences:
        return 0.0
    hits = 0
    for s in sentences:
        low = s.lower()
        if any(m in low for m in _ANALYSIS_MARKERS):
            hits += 1
    return hits / float(len(sentences))


def check_originality(
    script: str, published: Sequence[Dict[str, str]], source_text: str = ""
) -> PolicyResult:
    """Compare a script against what the channel has already put out.

    `published` is a sequence of {"id", "title", "script"} for videos that
    already exist on the channel. `source_text` is the article the research came
    from, when it is available; the closer the script tracks the source's own
    wording, the more it reads as reading material rather than commentary.
    """
    res = PolicyResult()
    # A three-sentence Short is not the target of this rule. Judging one on a
    # ratio of clauses would block every Short, so the checks only apply once
    # there is enough narration to actually contain analysis.
    if len(_tokens(script)) < 40:
        return res

    ratio = analysis_ratio(script)
    if ratio < 0.15:
        # A warning, not a refusal. The discourse-marker count is the weakest
        # signal in this module: it flagged a script that reasoned well but said
        # "but the physical assets tell a different story" instead of "however",
        # and a gate that blocks good scripts stops being read. The source
        # comparison below is the check that is actually well founded, so it
        # carries the decision and this one asks for a human to look.
        res.warn(
            f"only {ratio:.0%} of sentences carry explicit contrast or reasoning; "
            "worth reading before you approve it"
        )

    if source_text:
        ngram_share, novel_share = source_overlap(script, source_text)
        if ngram_share > _MAX_SOURCE_NGRAM_SHARE:
            res.fail(
                f"{ngram_share:.0%} of the script's word sequences are lifted from "
                "the source article, which is a rewrite rather than commentary"
            )
        elif novel_share < _MIN_NOVEL_WORD_SHARE:
            res.fail(
                f"only {novel_share:.0%} of the script's vocabulary is absent from "
                "the source; it restates the article instead of analysing it"
            )

    worst = 0.0
    worst_id = ""
    for p in published or []:
        sim = script_similarity(script, p.get("script") or "")
        if sim > worst:
            worst, worst_id = sim, p.get("id") or "?"
    if worst > MAX_SCRIPT_SIMILARITY:
        res.fail(
            f"script is {worst:.0%} similar to already-published video {worst_id}; "
            "videos must not read as interchangeable"
        )
    elif worst > MAX_SCRIPT_SIMILARITY * 0.75:
        res.warn(f"script is {worst:.0%} similar to published video {worst_id}")
    return res


def source_overlap(script: str, source: str) -> tuple:
    """(shared n-gram share, novel word share) of a script against its source.

    Both numbers point the same way and are kept together because either alone
    misleads: a script can parrot the article's exact sentences with new
    vocabulary, or discuss the article without reusing any of its phrasing.
    A genuine analysis is low on the first and high on the second.
    """
    sw = _content_words(script)
    aw = _content_words(source)
    # A different floor from the analysis check above. Judging clauses needs
    # enough sentences to mean anything, but a 4-gram comparison only needs
    # enough words to have n-grams at all, and a real restatement is often not
    # much longer than the article it is copying.
    if len(sw) < 20 or len(aw) < 20:
        return 0.0, 1.0
    n = _SHARE_NGRAM
    sgrams = set(zip(*(sw[i:] for i in range(n))))
    agrams = set(zip(*(aw[i:] for i in range(n))))
    if not sgrams:
        return 0.0, 1.0
    ngram_share = len(sgrams & agrams) / float(len(sgrams))
    aset = set(aw)
    novel_share = sum(1 for w in sw if w not in aset) / float(len(sw))
    return ngram_share, novel_share


def check_assets(
    *,
    image_credits: Sequence[Dict[str, str]],
    has_thumbnail: bool,
    has_caption_file: bool,
) -> PolicyResult:
    """Every picture must be licensed for monetised, modified use and credited."""
    res = PolicyResult()
    if not image_credits:
        res.warn("no image credits recorded; video has no sourced pictures")
    for c in image_credits:
        lic = (c.get("license_name") or "").strip()
        label = c.get("title") or c.get("url")
        if not lic:
            res.fail(f"image {label} has no licence")
        elif re.search(r"\b(nc|nd|non-?commercial|no-?deriv)", lic, re.I):
            res.fail(
                f"image {label} is {lic}, which forbids commercial use or derivatives"
            )
        if not c.get("credit"):
            res.fail(f"image {c.get('url')} is missing its attribution")
    if not has_thumbnail:
        res.warn("no custom thumbnail; click-through will be poor")
    if not has_caption_file:
        # Not a policy rule, but captions are how a technical video stays usable
        # and they are effectively expected of a serious channel.
        res.warn("no caption file uploaded")
    return res


AI_DISCLOSURE = (
    "This video was produced with automation: the script was drafted with AI "
    "assistance and reviewed, the narration is a synthesised computer voice, and "
    "some on-screen images are openly licensed photographs credited below. The "
    "analysis and the editing are the channel's own."
)


def build_description(
    *,
    base: str,
    image_credits: Sequence[Dict[str, str]] = (),
    source_url: str = "",
    brand: str = "",
    voice: str = "",
) -> str:
    """Assemble a description that carries the credits and the disclosure.

    Attribution is not optional for CC-BY or CC-BY-SA content: the licence
    requires the credit to travel with the work. Keeping it in a local file, as
    this project originally did, satisfies nobody, so the credits are part of
    the description that actually reaches YouTube.
    """
    parts: List[str] = []
    if (base or "").strip():
        parts.append(base.strip())

    if source_url:
        parts.append(f"Source: {source_url}")

    if image_credits:
        lines = ["Image credits:"]
        for c in image_credits:
            who = (c.get("creator") or "").strip()
            lic = (c.get("license_name") or "").strip()
            link = (c.get("source_page") or c.get("url") or "").strip()
            name = (c.get("title") or "").strip()
            bits = [b for b in (name, who, lic) if b]
            lines.append(f"- {' — '.join(bits) if bits else 'image'}{f' ({link})' if link else ''}")
        parts.append("\n".join(lines))

    if voice:
        parts.append(f"Narration voice: {voice}")

    parts.append(AI_DISCLOSURE)

    if brand:
        parts.append(brand)

    text = "\n\n".join(p for p in parts if p.strip())
    return text[:4990]  # leave room; the API truncates at 5000 regardless


def manual_steps() -> List[str]:
    """The compliance steps that have no API surface, stated plainly."""
    return [
        "In YouTube Studio, open the video and set the 'altered or synthetic "
        "content' disclosure under 'Show more'. The API cannot set this.",
        "Confirm the channel's AdSense account is linked and the channel is in "
        "an AdSense-supported country before the first monetised upload.",
        "Keep a human in the loop: the pipeline refuses to upload a script that "
        "fails originality, so approving a video is a real editorial decision.",
    ]
