"""Prompt text, kept apart from the call machinery so it can be reviewed and
tuned without touching retry logic.

House style for the channel: concrete over general, no hype words, no "in this
video we'll cover", numbers over adjectives. Every prompt states the format
constraint (length in words, Shorts vs longform) because the renderer times the
voiceover against the script and a miss means a truncated or padded video.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .config import Settings
from .util import clip, words


def system_prompt(s: Settings, kind: str) -> str:
    base = (
        f"You are the head writer for the YouTube channel '{s.brand}' "
        f"({s.brand_tagline}). You write in {s.language}. "
        f"Audience: {s.target_audience or 'curious, non-specialist tech users who want the short version'}.\n"
        "Rules you never break:\n"
        "- Lead with the payoff, never with a greeting or a promise to explain.\n"
        "- Concrete nouns, numbers, version numbers, prices, dates. No adjectives doing a noun's job.\n"
        "- Never say 'in this video', 'let's dive in', 'buckle up', 'without further ado'.\n"
        "- Never invent a statistic, quote, person, or product. If the source text does not say it, leave it out.\n"
        "- If the material is thin, say so by returning very few key_facts and a low confidence. "
        "A boring honest video beats a confident wrong one.\n"
        "- Plain text only: no markdown headings, no emoji, no bullet dashes, no stage directions."
    )
    if kind == "shorts":
        return base + (
            "\nYou are writing a YouTube Short (vertical, under 60 seconds). "
            "One idea only. First sentence is the whole hook — a viewer must not be able to swipe away in the first 2 seconds."
        )
    return base + (
        "\nYou are writing a long-form explainer. Open cold, escalate, land the point, close in two sentences."
    )


# --------------------------------------------------------------- research
def research_prompt(topic: Dict[str, Any], s: Settings) -> str:
    raw = topic.get("summary") or ""
    body = clip(raw, 6000) if raw else "(no article body was provided)"
    url = topic.get("url") or "(no url)"
    return f"""Topic headline: {topic.get('title')}

Source: {url}
Published: {topic.get('published_at') or 'unknown'}
Source type: {topic.get('source')}

Source text:
---
{body}
---

Assess this as a video idea for a {s.niche} channel.

1. `angle`: the ONE specific thing a viewer would learn. Not the topic restated
   — the part that is actually interesting. If you cannot find one, return no
   facts: the topic will be skipped.
2. `key_facts`: 3-6 hard facts from the text above. Attribute nothing you were
   not given. If the text is thin, return fewer.
3. `risks`: anything a careful editor would want flagged — unverified claims,
   single-sourced numbers, legal/medical/financial advice, ongoing litigation.
Return only the JSON object described by the schema. Do not return a
confidence field: confidence is measured against the source text, not asked
of you."""


# ----------------------------------------------------------------- script
def script_prompt(
    kind: str,
    topic: Dict[str, Any],
    research: Dict[str, Any],
    s: Settings,
) -> str:
    if kind == "shorts":
        target_s = s.shorts_target_s
    else:
        target_s = s.longform_target_s
    target_words = int(target_s / 60.0 * s.words_per_minute)

    facts = "\n".join(f"- {f}" for f in research.get("key_facts", []))
    risks = "\n".join(f"- {r}" for r in research.get("risks", [])) or "- (none flagged)"

    if kind == "shorts":
        shape = f"""Write a Short. Target {target_words} words ({target_s}s at {s.words_per_minute} wpm).

Format exactly:
hook: 1-2 sentences, max 30 words, states the payoff outright.
body: 2-4 sentences delivering it, no filler.
outro: one short sentence, or leave it empty.

Hard limit: {int(target_s * 1.2)} words total. Aim for {target_words}; a Short
under half the target length is a failed Short, not a tight one."""
    else:
        shape = f"""Write a long-form script. Target {target_words} words ({target_s}s at {s.words_per_minute} wpm).

Format exactly:
hook: 2-3 sentences. The specific claim or question, cold. No throat-clearing.
body: the explanation, broken into beats separated by a blank line. Escalate —
each beat must earn the next. Include the concrete numbers.
outro: 2 sentences. What the viewer should do or think differently, then a
       one-clause subscribe ask.

Hard limit: {int(target_s * 1.15)} words total, aim for {target_words}. Getting
there means more beats with real detail, not filler: if the research only
supports a short video, say so in the angle rather than writing thin."""

    return f"""Video topic: {topic.get('title')}
Angle: {research.get('angle')}
Why now: {research.get('why_now')}
Viewer question: {research.get('audience_pull')}

Established facts (use these; do not go beyond them):
{facts}

Caveats the editor flagged:
{risks}

    {shape}

    This has to be the channel's own analysis, not a read-out of a news article.
    YouTube's monetisation policy names "reading materials you didn't create
    (e.g., articles from websites)" as ineligible, and enforces it at channel
    level, so a viewer should hear reasoning they would not get from the source.
    Every beat must do at least one of these:
      - say what the news actually means for a person using the product,
      - compare it to what existed before and what changes,
      - name the catch, the limit, or the thing the announcement left out,
      - connect it to a pattern in how this industry keeps behaving.
    Restating the fact is never the point on its own. Do not reuse the
    wording, sentence shapes or metaphors of the source.

    Source URL for the description: {topic.get('url') or '(none)'}


Also return, in the same response, the packaging for the script you just wrote:
title (under {s.title_max_len} chars, specific, no clickbait), title_b (a
genuinely different alternative using different words), description (two short
paragraphs plus the source URL), tags (5-8 lowercase, no # in the values), and
thumbnail_plan (text of at most 4 words, plus emotion, subject and palette).
This is one request on purpose: the free tier allows 20 generateContent calls a
day, so metadata travels with the script rather than costing another call.

Return only the JSON object described by the schema."""


# --------------------------------------------------------------- metadata
def metadata_prompt(
    kind: str,
    topic: Dict[str, Any],
    research: Dict[str, Any],
    script: Dict[str, Any],
    s: Settings,
) -> str:
    if kind == "shorts":
        title_rule = (
            f"Title: max 55 characters (Shorts titles truncate hard). "
            f"Absolute max {s.title_max_len}. No ALL CAPS words, no clickbait lie."
        )
    else:
        title_rule = (
            f"Title: max {s.title_max_len} characters total. Front-load the keyword. "
            "No clickbait the video does not pay off."
        )

    return f"""Channel: {s.brand} — {s.brand_tagline}
Format: {"YouTube Short (vertical)" if kind == "shorts" else "long-form video"}

Topic: {topic.get('title')}
Angle: {research.get('angle')}
Facts covered: {"; ".join(research.get('key_facts', [])[:4])}

Narration opening:
{clip(script.get('hook') or script.get('body', '')[:300], 400)}

Write the upload metadata.

{title_rule}
`title_b`: a second title with a different angle for the same video — for A/B
testing it must not be a synonym swap of `title`.

`description`: 2 short paragraphs, then a Sources line listing the real source
URL/name, then a natural 3-5 hashtag line. Under 450 characters. No keyword
stuffing, no links other than the source and {s.brand_url}.

`tags`: 8-15 tags. Mix broad and specific. Lowercase. No competitor brand names,
no "best/top/must" spam tags.

`thumbnail_plan`: `text` is at most 4 words burned on the image. `emotion` is
one of curiosity/shock/relief/greed/surprise. `subject` is the literal visual.
`palette` is a two-colour description.

`chapters`: leave EMPTY for a Short. For long-form, give 3-6 chapters with
`time_s` in seconds, strictly increasing, first at 0.

Return only the JSON object described by the schema."""


BEAT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "beat": {"type": "string"},
    },
    "required": ["beat"],
}


def beat_prompt(
    topic: Dict[str, Any],
    research: Dict[str, Any],
    s: Settings,
    *,
    index: int,
    total: int,
    words: int,
    used_facts: Sequence[str],
) -> str:
    """One additional body beat, sized explicitly.

    A single request for a 1200-word script reliably comes back at 180-230
    words: the model treats the whole thing as one answer and lands on a summary
    length. Asking for one beat of a stated size, repeatedly, gets the volume
    without the filler, because each beat is still bound to the researched
    facts rather than to whatever the previous beat happened to say.
    """
    facts = "\n".join(f"- {f}" for f in research.get("key_facts", []))
    remaining = [f for f in research.get("key_facts", []) if f not in used_facts]
    fresh = "\n".join(f"- {f}" for f in remaining) or "- (all facts used; go deeper on those above)"
    risks = "\n".join(f"- {r}" for r in research.get("risks", [])) or "- (none flagged)"

    return f"""Video topic: {topic.get('title')}
Angle: {research.get('angle')}
Viewer question: {research.get('audience_pull')}

All researched facts:
{facts}

Facts not yet covered by the script, prefer these:
{fresh}

Caveats:
{risks}

Write body beat {index} of {total} for a long-form video. This is one section of
a larger script, not a summary and not an introduction.

Rules:
- Write {words} words. This is a hard floor of {int(words * 0.85)} words; a shorter
  answer is a failed answer, not a tight one.
- Open with a concrete detail, number or claim. No recap of the previous beat.
- Cover only the facts listed above, plus the reasoning that connects them. If a
  fact will not bear {words} words on its own, combine facts, do not speculate.
- End on a consequence or an open question that sets up the next beat.
- No filler, no "in today's fast-moving world", no restating the title.

Return only the JSON object described by the schema."""
