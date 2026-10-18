"""Content production: topic -> research -> script -> metadata.

Each stage tries the LLM first and falls back to a deterministic template when
no API key is configured. The fallback is not good writing — it exists so the
pipeline is exercisable end to end (and CI-testable) without a key, and so a
key outage degrades instead of stopping. `used_fallback` is carried on the
result so the pipeline can refuse to publish degraded content when
`REQUIRE_LLM=1`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from .config import Settings, get_settings
from .llm import (
    LLM,
    LLMError,
    LLMUnavailable,
    METADATA_SCHEMA,
    RESEARCH_SCHEMA,
    SCRIPT_SCHEMA,
    SCRIPT_WITH_META_SCHEMA,
    Metadata,
    Research,
    Script,
    get_llm,
)
from .logging_setup import get_logger
from .prompts import (
    BEAT_SCHEMA,
    beat_prompt,
    metadata_prompt,
    research_prompt,
    script_prompt,
    system_prompt,
)
from .util import clip, norm_text, sentences, slugify, words

log = get_logger("content")

DEFAULT_EMOJI_FREE = True


def _require_llm() -> bool:
    """True when a missing key should be a hard failure instead of a fallback."""
    import os

    return os.getenv("REQUIRE_LLM", "0").strip().lower() in {"1", "true", "yes"}


def _topic_dict(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return row
    keys = row.keys()
    return {k: row[k] for k in keys}


# ============================================================== research
def research_topic(
    topic: Any,
    *,
    settings: Optional[Settings] = None,
    llm: Optional[LLM] = None,
) -> Research:
    s = settings or get_settings()
    llm = llm or get_llm()
    data = _topic_dict(topic)

    if llm.available:
        try:
            raw = llm.complete(
                research_prompt(data, s),
                RESEARCH_SCHEMA,
                system=system_prompt(s, "longform"),
            )
            r = _to_research(raw)
            r.confidence = grounded_confidence(
                r.key_facts, str(data.get("summary") or ""), r.sources
            )
            log.info(
                "research ok for %s: %d facts, grounded confidence %.2f",
                clip(data.get("title", ""), 50), len(r.key_facts), r.confidence,
            )
            return r
        except (LLMError, Exception) as exc:  # noqa: BLE001

            log.error("research failed for %s: %s", clip(data.get("title", ""), 50), exc)
            if _require_llm():
                raise

    log.warning("using offline research fallback (quality degraded)")
    return _offline_research(data)


# Words carrying no topical signal. Without these, "the company said it would"
# overlaps the source perfectly while asserting nothing checkable.
_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those of in on at to for from by
    with without into over under again further is are was were be been being am do does
    did doing have has had having i we you he she it they them his her its their our your
    as not no nor so such only own same too very can will just should now also about
    after before during while where which who whom what when how why all any both each
    few more most other some out up down off there here""".split()
)


def _content_words(text: str) -> Set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOPWORDS}


def grounded_confidence(
    facts: List[str], source_text: str, sources: Optional[List[str]] = None
) -> float:
    """How much of the research is actually supported by the source.

    This replaces asking the model to rate its own confidence. Self-reported
    confidence is not calibrated: told to "be pessimistic", qwen3.8 returned
    0.10 for a well-sourced story while other providers returned 0.9 for the
    same input, so the publish gate moved whenever the provider failover
    changed model.

    Grounding is measurable instead. A fact the model actually read shares most
    of its content words with the source; a hallucinated one largely does not.
    That gives a number that means the same thing on every provider.
    """
    facts = [f for f in (facts or []) if str(f).strip()]
    if not facts or not source_text.strip():
        # Nothing to check against: an RSS headline with no body cannot support
        # a checkable claim, whatever the model said.
        return 0.2

    src = _content_words(source_text)
    if not src:
        return 0.2

    ratios = []
    for f in facts:
        fw = _content_words(str(f))
        if not fw:
            continue
        ratios.append(len(fw & src) / len(fw))
    if not ratios:
        return 0.2

    grounding = sum(ratios) / len(ratios)
    # Four facts is the prompt's target; reward reaching it, then flatten.
    coverage = min(1.0, len(facts) / 4.0)
    # Extra sources corroborate but cannot rescue invented facts, so they only
    # break ties between two otherwise equally grounded fact sets.
    corroboration = 0.05 if sources and len(sources) > 1 else 0.0
    return round(min(1.0, 0.65 * grounding + 0.35 * coverage + corroboration), 2)


def _to_research(raw: Dict[str, Any]) -> Research:
    def _str(v: Any) -> str:
        return str(v).strip() if v is not None else ""

    def _list(v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if str(x).strip()]

    try:
        conf = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return Research(
        angle=_str(raw.get("angle")),
        summary=_str(raw.get("summary")),
        key_facts=_list(raw.get("key_facts")),
        why_now=_str(raw.get("why_now")),
        audience_pull=_str(raw.get("audience_pull")),
        risks=_list(raw.get("risks")),
        sources=_list(raw.get("sources")),
        confidence=max(0.0, min(1.0, conf)),
    )


def _offline_research(data: Dict[str, Any]) -> Research:
    """Extract whatever structure the source text already has."""
    title = str(data.get("title") or "")
    body = str(data.get("summary") or "")
    sents = sentences(body)
    # Lead sentences of a news post are almost always the load-bearing facts.
    facts = [s for s in sents[:5] if 40 <= len(s) <= 280]
    angle = clip(facts[0], 180) if facts else clip(title, 180)
    return Research(
        angle=angle,
        summary=clip(" ".join(sents[:2]), 400),
        key_facts=facts,
        why_now="",
        audience_pull=clip(title, 140),
        risks=["generated without an LLM: no fact-checking was performed"],
        sources=[data["url"]] if data.get("url") else [],
        confidence=0.2,
        degraded=True,
    )


# ================================================================ script
def write_script(
    kind: str,
    topic: Any,
    research: Research,
    *,
    settings: Optional[Settings] = None,
    llm: Optional[LLM] = None,
    meta_out: Optional[Dict[str, Any]] = None,
) -> Script:
    s = settings or get_settings()
    llm = llm or get_llm()
    data = _topic_dict(topic)
    rdict = _research_dict(research)

    if llm.available:
        try:
            raw = llm.complete(
                script_prompt(kind, data, rdict, s),
                SCRIPT_WITH_META_SCHEMA,
                system=system_prompt(s, kind),
                max_tokens=s.llm_max_tokens,
            )
            script = Script(
                hook=str(raw.get("hook", "")).strip(),
                body=str(raw.get("body", "")).strip(),
                outro=str(raw.get("outro", "")).strip(),
            )
            if meta_out is not None:
                # The model already saw the script, so its title, description
                # and thumbnail plan come back in the same response. Sourcing
                # them here costs no extra request against a 20-per-day quota.
                meta_out.update(
                    {k: v for k, v in raw.items() if k not in ("hook", "body", "outro")}
                )
            if kind != "shorts":
                # Growth runs before the thin-script guard, because a too-thin
                # script is precisely the case growth exists to fix. Checking
                # first turned every short answer into an error instead.
                script = _grow_longform(script, data, rdict, s, llm)
            if script.word_count < 20:
                raise LLMError(f"script too thin ({script.word_count} words)")
            log.info(
                "script ok (%s): %d words", kind, script.word_count
            )
            return script
        except (LLMError, Exception) as exc:  # noqa: BLE001
            log.error("script failed (%s): %s", kind, exc)
            if _require_llm():
                raise

    log.warning("using offline script fallback for %s (quality degraded)", kind)
    return _offline_script(kind, data, research, s)


def _grow_longform(
    script: Script,
    topic: Dict[str, Any],
    research: Dict[str, Any],
    s: Settings,
    llm: LLM,
) -> Script:
    """Add real beats until a long-form script reaches its target length.

    Gemini answers a "write a 1200-word script" request with 180-230 words: one
    big request lands on summary length no matter how the target is phrased, and
    the validated result was an 8-minute video with two minutes of content. Each
    pass asks for one beat of a stated size instead, so the volume comes from
    more researched material rather than from padding.

    Stops at `LONGFORM_MAX_BEATS` and returns whatever it has: validation still
    rejects a script that stayed short, so a topic whose research genuinely
    cannot fill the runtime fails loudly instead of shipping thin.
    """
    target_words = int(s.longform_target_s / 60.0 * s.words_per_minute)
    if script.word_count >= target_words:
        return script

    have = script.word_count
    log.info(
        "script is %d words, target %d; adding beats", have, target_words
    )
    used_facts: List[str] = []
    for i, fact in enumerate(research.get("key_facts", []), start=1):
        if fact.lower() in script.body.lower() or fact.lower() in script.hook.lower():
            used_facts.append(fact)
        del i

    rounds = 0
    while have < target_words and rounds < s.longform_max_beats:
        rounds += 1
        missing = target_words - have
        # Ask for the whole remainder on the last pass rather than overshooting,
        # but never ask for more than one beat's worth. The free tiers here cap
        # output tokens per minute at roughly a thousand, so a single request
        # for half a script cannot complete: it stalls until the window resets
        # and then the call still wants more than the window allows. Small beats
        # fit several to a window, and the length comes from more researched
        # material rather than one oversized answer.
        beat_words = max(60, int(min(missing, s.beat_words)))
        idx = rounds
        try:
            raw = llm.complete(
                beat_prompt(
                    topic, research, s,
                    index=idx, total=s.longform_max_beats,
                    words=beat_words, used_facts=used_facts,
                ),
                BEAT_SCHEMA,
                system=system_prompt(s, "longform"),
                max_tokens=s.llm_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("beat %d failed: %s", idx, exc)
            break
        beat = str(raw.get("beat", "")).strip()
        if len(beat.split()) < 40:
            # A stub beat means the model had nothing left to say. Stopping here
            # leaves the script short on purpose, so validation can reject it.
            log.info("beat %d was %d words; treating research as exhausted",
                     idx, len(beat.split()))
            break
        script.body = (script.body + "\n\n" + beat).strip()
        have = script.word_count
        log.info("beat %d added: now %d words", idx, have)
        for fact in research.get("key_facts", []):
            if fact.lower() in beat.lower() and fact not in used_facts:
                used_facts.append(fact)

    return script


def _research_dict(r: Research) -> Dict[str, Any]:
    return {
        "angle": r.angle,
        "summary": r.summary,
        "key_facts": r.key_facts,
        "why_now": r.why_now,
        "audience_pull": r.audience_pull,
        "risks": r.risks,
        "sources": r.sources,
        "confidence": r.confidence,
    }


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _offline_script(kind: str, data: Dict[str, Any], research: Research, s: Settings) -> Script:
    title = str(data.get("title") or "this story")
    facts = research.key_facts or [research.summary or title]
    target_words = int(
        (s.shorts_target_s if kind == "shorts" else s.longform_target_s)
        / 60.0
        * s.words_per_minute
    )

    if kind == "shorts":
        hook = _fit_words(
            f"{_strip_title(title)} — here's what actually changed and why it matters.",
            30,
        )
        body = " ".join(_fit_words(f, 40) for f in facts[:3])
        outro = f"More like this on {s.brand}."
        return Script(hook=hook, body=body, outro=outro)

    # Long-form: alternate fact / commentary so it does not read as a list.
    beats: List[str] = []
    for i, fact in enumerate(facts):
        beats.append(fact)
        beats.append(_commentary(fact, i))
    body = "\n\n".join(beats)
    hook = _fit_words(
        f"{_strip_title(title)}. Here's the part that actually matters.",
        45,
    )
    outro = (
        f"That's {title}. If you want this kind of breakdown daily, "
        f"subscribe to {s.brand}."
    )
    script = Script(hook=hook, body=body, outro=outro)

    if script.word_count > target_words * 1.2:
        script.body = _fit_words_to(script.body, target_words)
    return script


def _strip_title(title: str) -> str:
    t = re.sub(r"\s+[—–-]\s+.*$", "", str(title or "").strip())
    return t.rstrip(".") or str(title or "")


def _commentary(fact: str, i: int) -> str:
    stems = [
        "That detail is the one most summaries skip.",
        "Worth pausing on, because it changes the practical takeaway.",
        "This is the part that affects what you actually do about it.",
        "And that is where most of the confusion comes from.",
    ]
    return stems[i % len(stems)]


def _fit_words(text: str, limit: int) -> str:
    w = str(text or "").split()
    return " ".join(w[:limit]).rstrip(" ,;:")


def _fit_words_to(text: str, target_words: int) -> str:
    sents = _SENTENCE_SPLIT.split(str(text or ""))
    out: List[str] = []
    used = 0
    for sent in sents:
        n = len(sent.split())
        if used + n > target_words:
            break
        out.append(sent)
        used += n
    return " ".join(out)


# ============================================================== metadata
def write_metadata(
    kind: str,
    topic: Any,
    research: Research,
    script: Optional[Script] = None,
    *,
    settings: Optional[Settings] = None,
    llm: Optional[LLM] = None,
    prefetched: Optional[Dict[str, Any]] = None,
) -> Metadata:
    s = settings or get_settings()
    llm = llm or get_llm()
    data = _topic_dict(topic)
    # A caller re-generating metadata (e.g. to fix a bad title) may not have the
    # script on hand; fall back to whatever source text exists.
    sdict = {
        "hook": script.hook if script else "",
        "body": script.body if script else str(data.get("summary") or ""),
        "outro": script.outro if script else "",
    }

    # Already produced alongside the script, at no extra request. Only missing
    # fields fall through to a dedicated call, so a partial response still works.
    raw = dict(prefetched or {})
    if raw and "title" in raw and "description" in raw and "tags" in raw:
        try:
            return _to_metadata(raw, kind, s)
        except Exception as exc:  # noqa: BLE001
            log.warning("prefetched metadata unusable (%s); regenerating", exc)

    if llm.available:
        try:
            raw = llm.complete(
                metadata_prompt(kind, data, _research_dict(research), sdict, s),
                METADATA_SCHEMA,
                system=system_prompt(s, kind),
            )
            meta = _to_metadata(raw, kind, s)
            log.info("metadata ok (%s): %r", kind, clip(meta.title, 60))
            return meta
        except (LLMError, Exception) as exc:  # noqa: BLE001
            log.error("metadata failed (%s): %s", kind, exc)
            if _require_llm():
                raise

    log.warning("using offline metadata fallback for %s", kind)
    return _offline_metadata(kind, data, research, s)


def _to_metadata(raw: Dict[str, Any], kind: str, s: Settings) -> Metadata:
    tags = raw.get("tags")
    plan = raw.get("thumbnail_plan")
    chapters = raw.get("chapters")
    return Metadata(
        title=str(raw.get("title", "")).strip(),
        title_b=str(raw.get("title_b", "")).strip(),
        description=str(raw.get("description", "")).strip(),
        tags=[_clean_tag(t) for t in tags if str(t).strip()] if isinstance(tags, list) else [],
        thumbnail_plan=dict(plan) if isinstance(plan, dict) else {},
        chapters=[c for c in chapters if isinstance(c, dict)] if isinstance(chapters, list) else [],
    )


def _clean_tag(tag: str) -> str:
    t = re.sub(r"[^a-z0-9\s+#]", " ", str(tag).lower())
    return " ".join(t.split())[:30].strip()


def _offline_title_b(title: str, research: Research, s: Settings) -> str:
    """A second title that shares few words with the first.

    The A/B variant is worthless if it is a synonym swap, so derive it from a
    different part of the material (the audience's question) rather than by
    appending a word to the original headline.

    Candidates are checked with `validate.check_title_b` itself, not a private
    similarity guess. That matters: an earlier version screened with a lenient
    word-overlap measure while the gate used Jaccard, so a candidate scoring 0.67
    locally sailed through to be rejected at 0.73 by the real rule. One metric,
    one source of truth.
    """
    from .validate import check_title_b

    limit = min(55, s.title_max_len) if s.shorts_target_s and len(title) > 55 else s.title_max_len
    question = (research.audience_pull or "").strip().rstrip("?")
    headline = _strip_title(title)

    candidates: List[str] = []
    if len(question.split()) >= 3:
        # rstrip the terminal punctuation so we never emit "Tuesday.?" or "GPU!!".
        candidates.append(f"{question.rstrip(' .!?,;:')}?")
    if " nobody " in headline:
        candidates.append(headline.replace(" nobody ", " everyone "))
    candidates.append(f"What {headline} actually means")
    candidates.append(f"{headline}: the real story")
    # A lead clause of a fact is a different sentence from the headline, so it
    # makes the best last resort before giving up on being different.
    for fact in research.key_facts[:3]:
        clause = fact.split(",")[0].split(". ")[0].strip()
        if clause:
            candidates.append(clause)

    for cand in candidates:
        cand = clip(cand, limit)
        if cand and not check_title_b(cand, title):
            return cand

    # Nothing scored as a real variant. Empty is honest: the validator reports
    # "title_b is empty", which is a true statement about this material, and is
    # far better than shipping a variant that silently reuses the headline.
    return ""


def _word_overlap(a: str, b: str) -> float:
    sa = {w for w in norm_text(a).split() if len(w) > 2}
    sb = {w for w in norm_text(b).split() if len(w) > 2}
    if not sa or not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _offline_metadata(kind: str, data: Dict[str, Any], research: Research, s: Settings) -> Metadata:
    title = _strip_title(str(data.get("title") or ""))
    if kind == "shorts":
        title = clip(title, 55)
    else:
        title = clip(title, s.title_max_len)

    url = str(data.get("url") or "")
    source_line = f"Source: {url}" if url else ""
    desc_parts = [clip(research.summary or research.angle, 300)]
    if research.key_facts:
        top = "; ".join(clip(f, 90) for f in research.key_facts[:3])
        desc_parts.append(f"In this one: {top}.")
    if source_line:
        desc_parts.append(source_line)
    desc_parts.append(f"{s.brand} — {s.brand_tagline}. {s.brand_url}")
    desc_parts.append("#tech #gadgets #news")
    description = "\n\n".join(p for p in desc_parts if p)

    tags = _offline_tags(data, s)

    return Metadata(
        title=title,
        title_b=_offline_title_b(title, research, s),
        description=description,
        tags=tags,
        thumbnail_plan={
            "text": " ".join(title.split()[:3]),
            "emotion": "curiosity",
            "subject": clip(title, 60),
            "palette": "dark navy background, cyan accent",
        },
        chapters=[],
    )


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "for", "with", "from", "that", "this",
    "is", "are", "was", "were", "has", "have", "will", "its", "it", "in", "on",
    "at", "to", "of", "new", "says", "said", "how", "why", "what", "you", "your",
}


def _offline_tags(data: Dict[str, Any], s: Settings) -> List[str]:
    from .util import norm_text

    text = norm_text(f"{data.get('title', '')} {data.get('summary', '')}")
    words_ = [w for w in re.findall(r"[a-z0-9][a-z0-9+.-]{2,}", text) if w not in _STOPWORDS]
    seen: List[str] = []
    for w in words_:
        if w not in seen:
            seen.append(w)
        if len(seen) >= 8:
            break
    base = [norm_text(n) for n in s.niche.split(",") if n.strip()]
    tags = seen[:6] + base[:4] + [norm_text(s.brand), "tech news"]
    out: List[str] = []
    for t in tags:
        t = t.replace(" ", "")
        if t and t not in out:
            out.append(t)
    return out[:15]
