"""Stage 3.7 — language tracks: sidecar translated subtitle files.

The narration is ONE audio track. "Wherever the video opens, it plays in the
local language" is a subtitle contract: platforms with caption systems
(YouTube first; others as they expose the API) auto-select a caption track
that matches the viewer's app language. So each video also ships every
language configured in SUBTITLE_LANGUAGES as a sidecar
`{id}.{lang-code}.srt`, translated from the base caption track — batched
through the same free LLM the scripts use, degrading gracefully (a failed
language simply doesn't ship that cycle; timings are never shifted).

Rumble has no caption-track API today, so there the burned-in captions carry
the base language; the sidecars still travel with the media and attach the
moment an upload target supports them.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from ..config import Settings, get_settings
from ..llm import LLMError, LLMUnavailable, get_llm
from ..logging_setup import get_logger
from .voice import Caption, write_srt

log = get_logger("render.langs")

# BCP-47 codes for the platform caption tracks.
LANG_CODES = {
    "hindi": "hi", "english": "en", "english-us": "en", "english-uk": "en",
    "urdu": "ur", "bengali": "bn", "bangla": "bn", "tamil": "ta",
    "telugu": "te", "marathi": "mr", "gujarati": "gu", "punjabi": "pa",
    "kannada": "kn", "malayalam": "ml", "odia": "or", "assamese": "as",
    "nepali": "ne", "spanish": "es", "español": "es", "portuguese": "pt",
    "arabic": "ar", "french": "fr", "german": "de", "japanese": "ja",
    "korean": "ko", "chinese": "zh", "mandarin": "zh", "indonesian": "id",
    "russian": "ru", "turkish": "tr", "italian": "it", "thai": "th",
}

_BATCH = 40  # captions per LLM call — keeps requests inside free-tier limits


def lang_code(name: str) -> str:
    clean = str(name or "").strip().lower()
    if clean in LANG_CODES:
        return LANG_CODES[clean]
    return clean[:3] or "xx"


def _translate_batch(texts: Sequence[str], target: str, source: str) -> List[str]:
    """One LLM call translating a caption batch, line-count preserved."""
    llm = get_llm()
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = (
        f"Translate each numbered subtitle line from {source} into {target}.\n"
        "Rules: keep history terms correct; keep numbers and years exact; "
        "do NOT merge or split lines; do NOT add notes. "
        f"Return exactly {len(texts)} translated lines.\n\n" + numbered
    )
    schema = {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exactly one translated line per input line, same order.",
            },
        },
        "required": ["lines"],
        "additionalProperties": False,
    }
    raw = llm.complete(
        prompt, schema,
        max_tokens=max(800, 60 * len(texts)),
    )
    lines = [str(x).strip() for x in raw.get("lines", [])]
    # The model must return one line per input: pad with the original text so
    # the track stays syncable instead of shifting every line after a miss.
    while len(lines) < len(texts):
        lines.append(texts[len(lines)])
    return lines[: len(texts)]


def translate_sidecars(
    captions: Sequence[Caption],
    base_srt: Path,
    *,
    settings: Optional[Settings] = None,
) -> List[Path]:
    """Write `{stem}.{code}.srt` for every SUBTITLE_LANGUAGES entry.

    Timing comes from the measured base audio, so every sidecar lines up
    with the narration without re-measuring anything.
    """
    s = settings or get_settings()
    langs = [
        lng.strip() for lng in (s.subtitle_languages or "").split(",")
        if lng.strip()
    ]
    if not langs or not captions:
        return []
    base_lang = str(s.language or "English").strip() or "English"
    written: List[Path] = []
    for lng in langs:
        code = lang_code(lng)
        target = base_srt.with_name(f"{base_srt.stem}.{code}.srt")
        if target.exists():
            written.append(target)
            continue
        translated: List[Optional[str]] = [None] * len(captions)
        failed_batches = 0
        total_batches = (len(captions) + _BATCH - 1) // _BATCH
        for start in range(0, len(captions), _BATCH):
            slice_ = captions[start: start + _BATCH]
            try:
                got = _translate_batch([c.text for c in slice_], lng, base_lang)
            except (LLMError, LLMUnavailable) as exc:
                failed_batches += 1
                log.warning("translation to %s failed for captions %d-%d: %s",
                            lng, start, start + len(slice_), exc)
                continue
            for i, text in enumerate(got):
                translated[start + i] = text
        if failed_batches >= total_batches:
            # Nothing usable for this language — skip without writing a file
            # that would look translated but isn't.
            log.warning("skipping sidecar %s: all %d batches failed", lng, total_batches)
            continue
        versioned = [
            t if t is not None else captions[i].text
            for i, t in enumerate(translated)
        ]
        caps = [
            Caption(text=v, start=captions[i].start, end=captions[i].end)
            for i, v in enumerate(versioned)
            if v
        ]
        if caps:
            written.append(write_srt(caps, target))
            log.info("sidecar captions: %s (%s)", target.name, lng)
    return written
