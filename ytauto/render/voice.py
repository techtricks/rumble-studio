"""Voiceover synthesis + caption timing.

Default engine is macOS `say`, which is offline, free and needs no key. An
API-backed engine can be dropped in behind the same `synthesize()` contract.

Caption timing deserves a note. Rendering one `say` call for the whole script
gives natural prosody, but then nothing knows where each sentence lands in the
audio. Rendering one call per sentence gives exact timing but chops the
intonation at every boundary. We take the first option and estimate boundaries
proportionally by syllable weight, then rescale the estimates so the final
caption ends exactly on the measured audio duration. Captions drift by a few
tens of milliseconds, which is invisible; a chopped-up voice is obvious.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from ..config import MEDIA_DIR, Settings, get_settings
from ..logging_setup import get_logger
from ..util import digest
from ..util import words as count_words
from . import ffmpeg as ff

log = get_logger("render.voice")

# Caption shaping. Shorts need fewer, larger words than long-form.
CHUNK_WORDS = {"shorts": 4, "longform": 7}
# Silence inserted after each caption, as a fraction of total audio.
TAIL_PAUSE = 0.02


class VoiceError(RuntimeError):
    pass


@dataclass
class Caption:
    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def say_available() -> bool:
    return platform.system() == "Darwin" and shutil.which("say") is not None


# ------------------------------------------------------------- synthesis
# Voices that suit a technology explainer. macOS also ships novelty voices
# ("Bubbles", "Cellos", "Zarvox", "Wobble") which read as a novelty rather than
# as a person explaining something, so they are deliberately not in the pool.
# The list is a default, not a lock: TTS_VOICE_POOL overrides it.
DEFAULT_VOICE_POOL = (
    "Samantha,Daniel,Moira,Karen,Tessa,Aman,Rishi,Tara"
)
# Narration language decides the pool, not the user's keyboard: a Hindi
# channel read by an English `say` voice once spelled out Devanagari with
# seconds-long dead pauses per glyph and turned 1211 words into 45 minutes
# of audio. macOS ships Lekha (hi_IN) for Hindi; keep her as the default
# base and let TTS_VOICE_POOL override both pools.
DEFAULT_VOICE_POOL_HI = "Lekha"

# The Linux deploy (Oracle VPS) has no macOS `say`; Edge-TTS neural voices
# stand in. Rotating a *region* mix (US/UK/AU/IE/IN) keeps the channel's
# same anti-fingerprint property: one speaker per video, not per channel.
EDGE_VOICE_POOL = (
    "en-US-GuyNeural,en-GB-SoniaNeural,en-AU-NatashaNeural,"
    "en-US-JennyNeural,en-IE-ConnorNeural,en-IN-NeerjaNeural"
)
EDGE_VOICE_POOL_HI = "hi-IN-SwaraNeural,hi-IN-MadhurNeural"


def _pool_for_language(s: Settings) -> str:
    """The default macOS `say` pool for the narration language.

    Edge-TTS (Linux deploy) has its own pools — see `_voice_pool_from_engine`.
    """
    if str(s.language or "").strip().lower() in ("hindi", "hi", "हिंदी"):
        return DEFAULT_VOICE_POOL_HI
    return DEFAULT_VOICE_POOL


def _engine_pool_for_language(s: Settings) -> str:
    """Edge-TTS voices matching the narration language."""
    if str(s.language or "").strip().lower() in ("hindi", "hi", "हिंदी"):
        return EDGE_VOICE_POOL_HI
    return EDGE_VOICE_POOL


def available_voices(locale_prefix: str = "en") -> List[str]:
    """Installed voice names, optionally filtered to a language.

    `say -v ?` prints one line per voice *per language*, so a voice that speaks
    several languages appears several times. Names are de-duplicated keeping
    first-seen order, which is alphabetical from the tool.
    """
    if not say_available():
        return []
    import subprocess

    try:
        out = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, timeout=20
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    pattern = re.compile(r"^(?P<name>.+?)\s{2,}(?P<loc>[a-z]{2}_[A-Z]{2})\s+#")
    seen: List[str] = []
    for line in out.splitlines():
        m = pattern.match(line.rstrip())
        if not m:
            continue
        if not m.group("loc").lower().startswith(locale_prefix.lower()):
            continue
        name = m.group("name").strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def pick_voice(video_id: str, settings: Optional[Settings] = None) -> str:
    """Choose a narration voice deterministically for a video.

    YouTube lists "channels where all slideshows use the same voiceover across
    videos" as an example of inauthentic content, and that is enforced per
    channel, not per video. Rotating the voice keeps the narration from becoming
    the channel's fingerprint.

    The choice is derived from the video id rather than randomised so that
    re-rendering the same video produces the same audio; a render that changed
    its own voice every time would make the output untestable and would leave
    published videos unmatchable.
    """
    s = settings or get_settings()
    if s.tts_voice:
        # An explicit voice always wins; rotation is opt-out.
        pool = [v.strip() for v in s.tts_voice.split(",") if v.strip()]
    else:
        pool = [v.strip() for v in (s.tts_voice_pool or "").split(",") if v.strip()]
    if not pool:
        engine = (s.tts_engine or "auto").lower()
        pool = [
            v.strip()
            for v in (
                _engine_pool_for_language(s) if engine == "edge"
                else _pool_for_language(s)
            ).split(",")
            if v.strip()
        ]
    if len(pool) == 1:
        return pool[0]
    return pool[int(digest(video_id)[:8], 16) % len(pool)]


def narrate(
    script_text: str,
    out_path: Path,
    *,
    settings: Optional[Settings] = None,
    voice: Optional[str] = None,
    rate: Optional[int] = None,
) -> Path:
    """Render narration to a normalised WAV. Returns the path."""
    s = settings or get_settings()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = _clean_for_speech(script_text, language=s.language)
    if not text:
        raise VoiceError("nothing to narrate: script is empty")

    engine = (s.tts_engine or "auto").lower()
    # The intermediate must not be the same path we normalise into (ffmpeg
    # refuses in-place edits), and the extension hints the container: `say`
    # derives WAV from it, edge-tts writes MP3.
    raw_ext = "mp3" if engine == "edge" else "wav"
    raw_path = out_path.with_name(f"{out_path.stem}.raw.{raw_ext}")
    if engine in ("auto", "say"):
        if not say_available():
            if engine == "say":
                raise VoiceError("TTS_ENGINE=say but macOS `say` was not found")
            raise VoiceError(
                "no TTS engine available. macOS `say` is required for the default "
                "engine; on other systems set TTS_ENGINE=edge (Edge-TTS neural "
                "voices, needs the edge-tts pip package)."
            )
        raw = _say(text, raw_path, voice or s.tts_voice, rate or s.tts_rate_wpm)
    elif engine == "edge":
        raw = _edge_tts(text, raw_path, voice or s.tts_voice, rate or s.tts_rate_wpm)
    elif engine == "none":
        raise VoiceError("TTS_ENGINE=none: cannot synthesise a voiceover")
    else:
        raise VoiceError(f"unknown TTS_ENGINE {engine!r} (expected auto|say|edge|none)")

    if not raw.exists() or raw.stat().st_size < 1024:
        raise VoiceError(f"the TTS engine produced no usable audio ({raw})")

    _normalise(raw, out_path)
    return out_path


def _clean_for_speech(text: str, *, language: str = "") -> str:
    """Strip things that make a TTS engine stumble or spell out characters.

    The Devanagari strip only applies when the narration voice cannot read
    the script: an English voice reads Devanagari with seconds-long dead
    pauses per glyph (the 45-minute-audio bug), but a Hindi voice such as
    Lekha reads it natively, so a Hindi channel keeps its words intact.
    """
    out = str(text or "").strip()
    out = re.sub(r"https?://\S+", "the link in the description", out)
    out = re.sub(r"[*_`#>|]+", " ", out)
    is_hindi = str(language or "").strip().lower() in ("hindi", "hi", "हिंदी")
    devanagari = len(re.findall(r"[\u0900-\u097F]", out))
    if devanagari and not is_hindi:
        log.warning(
            "stripping %d non-Latin characters from the narration "
            "(script had mixed-language content)", devanagari,
        )
        out = re.sub(r"[\u0900-\u097F]+", " ", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _say(text: str, out_path: Path, voice: str, rate: int) -> Path:
    cmd = ["say", "-o", str(out_path), "--data-format=LEI16@24000"]
    if voice:
        cmd += ["-v", voice]
    if rate:
        cmd += ["-r", str(int(rate))]
    cmd += ["--", text]
    log.info("synthesising %d words at %d wpm (%s)", count_words(text), rate, voice or "default voice")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        raise VoiceError(f"`say` failed to run: {exc}") from exc
    if proc.returncode != 0:
        raise VoiceError(f"`say` exited {proc.returncode}: {(proc.stderr or '').strip()[:300]}")
    return out_path


def _edge_tts(text: str, out_path: Path, voice: str, rate: int) -> Path:
    """Synthesise with Microsoft Edge neural voices (free, no API key).

    Runs the edge-tts CLI so the Linux deploy needs only the pip package:
    Microsoft's public demo endpoint serves the voices, no account or card.
    The emitted MP3 (24 kHz mono) is lifted to the broadcast WAV by
    `_normalise` afterwards, same as the `say` path.
    """
    if not voice:
        raise VoiceError(
            "Edge-TTS needs a voice name (e.g. en-US-GuyNeural). "
            "Set TTS_VOICE or leave TTS_VOICE_POOL empty for the language pool."
        )
    cmd = [
        sys.executable, "-m", "edge_tts",
        "--voice", voice,
        "--text", text,
        "--write-media", str(out_path),
    ]
    # `rate` is a wpm target tuned for `say`'s ~150wpm baseline; edge voices'
    # default pace is close to it, and happy pace swings beyond ±50% produce
    # robotic speed, so clamp.
    pct = max(-50, min(50, round((int(rate or 150) / 150 - 1) * 100)))
    if pct:
        cmd += ["--rate", f"{pct:+d}%"]
    log.info("synthesising %d words (%s, edge-tts)", count_words(text), voice)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except (OSError, subprocess.SubprocessError) as exc:
        raise VoiceError(f"edge-tts failed to run: {exc}") from exc
    if proc.returncode != 0:
        raise VoiceError(f"edge-tts exited {proc.returncode}: {(proc.stderr or '').strip()[:300]}")
    return out_path


def _normalise(src: Path, dst: Path) -> None:
    """Convert to 44.1k stereo and level-match to broadcast-ish loudness.

    `say` emits a mono 24k file at an inconsistent gain; YouTube's player
    normalises anyway, but a -20 dB video reads as broken.
    """
    ff.run(
        [
            "-i", str(src),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=44100",
            "-ac", "2", "-ar", "44100",
            "-c:a", "pcm_s16le",
            str(dst),
        ],
        timeout=300,
    )
    try:
        src.unlink()
    except OSError:
        pass


# -------------------------------------------------------------- captions
def chunk_script(text: str, kind: str) -> List[str]:
    """Split narration into caption-sized pieces, preferring sentence ends.

    Two rules matter for readability: never end a caption on a fragment ("not
    compute." on its own line reads as a glitch), and never let a caption run
    long enough to wrap into three lines on a phone.
    """
    max_words = CHUNK_WORDS.get(kind, 6)
    min_words = max(2, int(max_words * 0.6))
    flat = re.sub(r"\s+", " ", str(text or "")).strip()
    if not flat:
        return []

    chunks: List[str] = []
    # The danda (।) ends Hindi sentences where a full stop ends English ones;
    # splitting without it would glue two Hindi sentences into one caption.
    for sentence in re.split(r"(?<=[.!?।॥])\s+", flat):
        words_ = sentence.split()
        if not words_:
            continue
        if len(words_) <= max_words:
            chunks.append(sentence)
            continue
        # Oversized sentence: cut at the comma/conjunction nearest to max_words.
        for group in _split_long(words_, max_words, min_words):
            chunks.append(group)

    # Merge runt chunks forward, so a short trailing clause joins its
    # neighbour — but never past the caption cap, which is what would cause a
    # three-line caption on a phone.
    merged: List[str] = []
    for chunk in chunks:
        n = len(chunk.split())
        if (
            merged
            and n < min_words
            and len(merged[-1].split()) + n <= max_words
        ):
            merged[-1] = f"{merged[-1]} {chunk}".strip(" ,;:")
        else:
            merged.append(chunk)
    return [c.strip() for c in merged if c.strip()]


def _split_long(words_: Sequence[str], max_words: int, min_words: int) -> List[str]:
    """Greedy split of an oversized sentence into readable caption groups.

    Prefers a break just after a comma (natural breath) but only when the
    resulting caption is not a stub; otherwise it breaks at the conjunction
    before the limit; otherwise it hard-splits.
    """
    out: List[str] = []
    i = 0
    total = len(words_)
    while i < total:
        remaining = total - i
        if remaining <= max_words:
            out.append(" ".join(words_[i:]))
            break
        window = words_[i : i + max_words]
        # Look for a comma or conjunction in the back half of the window.
        cut = None
        for j in range(len(window) - 1, max(0, min_words) - 1, -1):
            token = window[j].rstrip(",;:")
            if token.endswith(",") or token.lower() in _BREAK_WORDS:
                if j + 1 >= min_words:  # do not leave a stub after the break
                    cut = j + 1
                break
        if cut is None:
            cut = max_words
        out.append(" ".join(window[:cut]).strip(" ,;:"))
        i += cut
    return [o for o in out if o]


# Conjunctions that make a natural caption boundary.
_BREAK_WORDS = {
    "and", "but", "so", "because", "which", "while", "whereas", "since",
    "though", "although", "however", "then", "now", "also", "yet",
    # Hindi word-final connectives — without them an oversized Hindi sentence
    # hard-splits mid-phrase ("...पुराना पत्थर / का ढाँचा है।"), which reads
    # like a glitch on screen.
    "और", "लेकिन", "पर", "परंतु", "क्योंकि", "जब", "तब", "कि", "जो", "जहाँ",
    "यदि", "अगर", "फिर", "इसलिए", "हालाँकि", "जबकि", "तथा", "इसके", "बाद",
}


_VOWEL_GROUP = re.compile(r"[aeiouy]+")


def _weight(text: str) -> float:
    """Relative spoken length of a chunk: words + syllable-ish count.

    English speech time tracks syllables far better than character count, so
    "fire" weighs more than "fine" even though the words are the same length.
    """
    words_ = re.findall(r"[A-Za-z0-9']+", text)
    deva = re.findall(r"[\u0900-\u097F]+", text)
    if not words_ and not deva:
        return 1.0
    syllables = 0
    for w in words_:
        groups = _VOWEL_GROUP.findall(w.lower())
        n = max(1, len(groups))
        # Silent trailing "e" ("the" -> 1 syllable, not 2).
        if w.lower().endswith("e") and n > 1 and not w.lower().endswith(("le", "ee", "ye")):
            n -= 1
        syllables += n
    # Devanagari is used only for relative timing weights, so a character
    # count/2.5 (≈ akshara density for typical Hindi words) is accurate
    # enough and can't critically fail: "स्तूप" (5 chars) ≈ 2 beats.
    for w in deva:
        syllables += max(1, round(len(w) / 2.5))
    return max(1.0, syllables + 0.35 * (len(words_) + len(deva)))


def build_captions(
    text: str,
    audio_path: Path,
    kind: str,
    *,
    tail_pause: float = TAIL_PAUSE,
) -> List[Caption]:
    """Time caption chunks against the real, measured audio length."""
    chunks = chunk_script(text, kind)
    if not chunks:
        return []
    duration = ff.probe_duration(audio_path)
    if not duration or duration <= 0:
        log.warning("could not measure %s; captions will not be timed", audio_path.name)
        return []

    weights = [_weight(c) for c in chunks]
    total_weight = sum(weights) or 1.0
    # Leave a beat of silence at the end rather than captioning over nothing.
    usable = max(0.5, duration * (1.0 - tail_pause))

    captions: List[Caption] = []
    cursor = 0.0
    for chunk, weight in zip(chunks, weights):
        span = usable * (weight / total_weight)
        captions.append(Caption(text=chunk, start=round(cursor, 3), end=round(cursor + span, 3)))
        cursor += span
    if captions:
        captions[-1].end = round(min(duration, cursor + duration * tail_pause), 3)
    log.info("%d captions timed over %.1fs of audio", len(captions), duration)
    return captions


def write_srt(captions: Sequence[Caption], path: Path) -> Path:
    """Sidecar .srt — also lets a human check timing without rendering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for i, c in enumerate(captions, 1):
        lines.append(str(i))
        lines.append(f"{_srt_ts(c.start)} --> {_srt_ts(c.end)}")
        lines.append(c.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _srt_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:  # rounding can tip over a second
        ms, s = 0, s + 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
