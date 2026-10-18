from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Deliberately stdlib logging rather than logging_setup.get_logger: that helper
# calls get_settings(), and config is what it imports. Handlers are attached
# later by setup_logging(), so a plain getLogger here is the same object.
log = logging.getLogger("ytauto.config")

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# DATA_DIR is the single knob that moves everything mutable (db, media, logs).
# Overriding only DB_PATH used to leave logs and renders in the real tree, which
# is surprising in production and unusable for tests. Setting DATA_DIR relocates
# all three together; the individual vars below still win if set.
DATA_DIR = Path(os.getenv("DATA_DIR") or (ROOT / "data"))
MEDIA_DIR = Path(os.getenv("MEDIA_DIR") or (DATA_DIR / "media"))
LOG_DIR = Path(os.getenv("LOG_DIR") or (DATA_DIR / "logs"))
SECRETS_DIR = Path(os.getenv("SECRETS_DIR") or (ROOT / "secrets"))
DB_PATH = Path(os.getenv("DB_PATH") or (DATA_DIR / "ytauto.db"))
TOKEN_PATH = Path(os.getenv("TOKEN_PATH") or (SECRETS_DIR / "token.json"))
CLIENT_SECRET_PATH = Path(
    os.getenv("CLIENT_SECRET_PATH") or (SECRETS_DIR / "client_secret.json")
)


def log_dir() -> Path:
    """Resolve the log directory at call time.

    The module constants above are resolved at import, which is fine for
    defaults but means anything that changes LOG_DIR later (a test, a second
    channel) is silently ignored. Anything that actually writes or reads logs
    goes through here instead of trusting the constant.
    """
    return Path(os.getenv("LOG_DIR") or LOG_DIR)


def media_dir() -> Path:
    """Same idea as `log_dir`, for rendered media."""
    return Path(os.getenv("MEDIA_DIR") or MEDIA_DIR)

# YouTube Data API v3. Full detail:
# https://developers.google.com/youtube/v3/determine_quota_cost
COST = {
    "videos.insert": 1600,
    "videos.list": 1,
    "videos.update": 50,
    "videos.delete": 50,
    "search.list": 100,
    "channels.list": 1,
    "thumbnails.set": 50,
    "playlistItems.insert": 50,
    "reports.query": 1000,
}
DEFAULT_DAILY_QUOTA = 10_000
SAFETY_MARGIN = 0.9  # never plan beyond 90% of the daily cap

# Vocabulary the on-topic gate matches against, not just for the description.
# Three words were not enough to describe a channel that covers all of
# technology: a robotaxi fleet, a chip export ruling and a battery breakthrough
# are unmistakably tech stories, and the gate rejected all three while it
# existed to catch a video about medieval pottery. It still does, because none
# of these words appear in a subject that is not technology.
#
# Order matters. The first four are the ones that get turned into tags, so the
# broadest and most searched-for come first and the long tail is only used to
# recognise a subject. Multi-word entries are matched as phrases, in order.
DEFAULT_NICHE = (
    "tech,technology,ai,gadgets,"
    "artificial intelligence,machine learning,chatbot,"
    "smartphone,phone,laptop,tablet,headphone,earbuds,wearable,smartwatch,"
    "gaming,console,handheld,gpu,cpu,chip,semiconductor,processor,foundry,"
    "software,app,apps,operating system,update,firmware,driver,open source,"
    "coding,programming,developer,cloud,data centre,data center,server,"
    "cybersecurity,hack,hacker,data breach,encryption,malware,privacy,"
    "robot,robotics,drone,autonomous,self-driving,driverless,robotaxi,"
    "electric vehicle,ev,battery,tesla,space,satellite,nasa,rocket,launch,"
    "quantum,blockchain,crypto,bitcoin,streaming,social media,startup,"
    "internet,browser,search engine,5g,6g,algorithm"
)


def _env_list(key: str, default: Optional[List[str]] = None) -> List[str]:
    raw = os.getenv(key)
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime configuration. Every value is overridable via .env."""

    # --- YouTube / Google ---
    client_secret_path: Path = field(default_factory=lambda: CLIENT_SECRET_PATH)
    token_path: Path = field(default_factory=lambda: TOKEN_PATH)
    # The channel id to post to. Filled in automatically after `auth`.
    channel_id: Optional[str] = field(
        default_factory=lambda: os.getenv("YOUTUBE_CHANNEL_ID") or None
    )
    daily_quota: int = field(
        default_factory=lambda: _env_int("YOUTUBE_DAILY_QUOTA", DEFAULT_DAILY_QUOTA)
    )

    # --- Content direction ---
    niche: str = field(
        default_factory=lambda: os.getenv("NICHE") or DEFAULT_NICHE
    )
    language: str = field(
        default_factory=lambda: os.getenv("LANGUAGE") or "English"
    )
    # Extra caption tracks translated from the base language at render time,
    # written as sidecar {id}.{code}.srt next to the media. CSV of names from
    # langs.LANG_CODES; empty means base language only.
    subtitle_languages: str = field(
        default_factory=lambda: os.getenv("SUBTITLE_LANGUAGES") or ""
    )
    target_audience: str = field(
        default_factory=lambda: os.getenv("TARGET_AUDIENCE") or ""
    )

    # --- Cadence ---
    # Times are local HH:MM. One long-form + one Short per day by default.
    longform_times: List[str] = field(
        default_factory=lambda: _env_list("LONGFORM_TIMES", ["18:30"])
    )
    shorts_times: List[str] = field(
        default_factory=lambda: _env_list("SHORTS_TIMES", ["12:30", "20:30"])
    )
    timezone: str = field(
        default_factory=lambda: os.getenv("TIMEZONE") or "Asia/Kolkata"
    )
    # Minimum gap between two publishes, guards against duplicate/thrash posts.
    min_gap_minutes: int = field(
        default_factory=lambda: _env_int("MIN_GAP_MINUTES", 90)
    )

    # --- LLM (Anthropic) ---
    # Which backend `ytauto.llm` talks to. `auto` prefers Gemini when its key is
    # present, then Anthropic, so a free-tier setup needs no extra config beyond
    # the key itself. Pin it to "anthropic" or "gemini" to be explicit.
    llm_provider: str = field(
        default_factory=lambda: (os.getenv("LLM_PROVIDER") or "auto").strip().lower()
    )
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None
    )
    # Gemini free tier: a key from Google AI Studio, no billing required.
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or None
    )
    # Groq free tier: a key from console.groq.com, no credit card. Preferred
    # over Gemini for this pipeline because its free tier allows tens of
    # thousands of requests a day, against Gemini's 20.
    #
    # The default model is qwen3.8-27b because Groq's catalogue turns over
    # fast: llama-3.3-70b-versatile, the previous default, now answers 404
    # model_not_found. qwen3.8 is also not a reasoning model, so it spends no
    # budget on thinking before emitting the JSON we asked for. Verify what
    # your key can actually reach with `ytauto llm --list-models`.
    groq_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY") or None
    )
    # NVIDIA NIM free tier: build.nvidia.com, part of the free developer
    # program. 40 requests a minute. A second free provider so a Groq outage or
    # quota is not a stop.
    nvidia_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY") or None
    )
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL") or "qwen/qwen3.8-27b"
    )
    nvidia_model: str = field(
        default_factory=lambda: os.getenv("NVIDIA_MODEL") or "meta/llama-3.3-70b-instruct"
    )
    groq_base_url: str = field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
    )
    nvidia_base_url: str = field(
        default_factory=lambda: os.getenv("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
    )
    # Dateless ID = a moving alias. See
    # https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL") or "claude-sonnet-5"
    )
    # Haiku is ~3x cheaper; used for the cheap, high-volume classification work.
    llm_model_fast: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL_FAST") or "claude-haiku-4-5-20251001"
    )
    # Used when LLM_PROVIDER resolves to gemini. The free tier serves these
    # without a billing account.
    #
    # Default to the `-latest` aliases, not a dated id. Google retires point
    # versions for new keys without warning — `gemini-2.5-flash` is listed by
    # the models endpoint but 404s for anyone who created a key recently — and
    # a hardcoded version is exactly how that reaches production. The alias
    # always points at the current release. Pin a dated id (e.g.
    # `gemini-3.8-flash`) in .env if you want reproducible output instead.
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL") or "gemini-flash-latest"
    )
    gemini_model_fast: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL_FAST") or "gemini-flash-lite-latest"
    )
    llm_max_tokens: int = field(
        default_factory=lambda: _env_int("LLM_MAX_TOKENS", 8000)
    )
    longform_max_beats: int = field(
        default_factory=lambda: _env_int("LONGFORM_MAX_BEATS", 10)
    )
    # Words per beat request. Kept small on purpose: the free tiers that work
    # here cap output tokens per minute at around a thousand, and one request
    # for half a script exceeds that on its own. A 160-word beat is about 210
    # output tokens, so several fit in one window and the length comes from more
    # researched material rather than from one big answer.
    beat_words: int = field(
        default_factory=lambda: _env_int("BEAT_WORDS", 160)
    )
    llm_timeout_s: int = field(
        default_factory=lambda: _env_int("LLM_TIMEOUT_S", 180)
    )
    llm_attempts: int = field(
        default_factory=lambda: _env_int("LLM_ATTEMPTS", 4)
    )
    # Appended to every system prompt so the channel sounds like one channel.
    brand: str = field(
        default_factory=lambda: os.getenv("BRAND") or "Tech Tricks Hub"
    )
    brand_tagline: str = field(
        default_factory=lambda: os.getenv("BRAND_TAGLINE") or "Daily tech explained fast"
    )
    brand_url: str = field(
        default_factory=lambda: os.getenv("BRAND_URL") or "https://www.tech-tricks-hub.com"
    )

    # --- Content sources ---
    rss_feeds: List[str] = field(
        default_factory=lambda: _env_list(
            "RSS_FEEDS",
            [
                "https://techcrunch.com/feed/",
                "https://www.theverge.com/rss/index.xml",
                "https://arstechnica.com/feed/",
                "https://news.ycombinator.com/rss",
                "https://www.androidauthority.com/feed/",
            ],
        )
    )
    # Off by default. Reddit's .rss publishes neither the article URL nor the
    # article text any more -- verified against the live feed, every href in a
    # link post's body points back at reddit.com -- so a Reddit topic reaches
    # research as a bare headline. The RSS feeds above already carry the same
    # stories with a real source and a real summary, so enabling this buys
    # duplicate headlines with less material, and it costs a 15s throttle plus
    # aggressive anonymous rate limits (HTTP 429).
    # Self posts and AMAs are filtered out if you do enable it; see
    # ytauto/sources/reddit.py.
    subreddits: List[str] = field(
        default_factory=lambda: _env_list("SUBREDDITS", [])
    )

    # --- Safeguards ---
    # When False, the pipeline plans and writes files but never calls upload.
    dry_run: bool = field(
        default_factory=lambda: os.getenv("DRY_RUN", "1").strip().lower()
        in {"1", "true", "yes"}
    )
    max_topics_per_day: int = field(
        default_factory=lambda: _env_int("MAX_TOPICS_PER_DAY", 12)
    )
    # Reject generated titles that trip spam heuristics.
    title_max_len: int = field(
        default_factory=lambda: _env_int("TITLE_MAX_LEN", 70)
    )
    # How many pictures to put behind a video. Each one is a slideshow slide
    # with a slow push and a cross-fade, so even four reads as a produced video
    # rather than a caption track over a colour wash. Zero disables pictures
    # entirely and renders the original gradient.
    images_per_video: int = field(
        default_factory=lambda: _env_int("IMAGES_PER_VIDEO", 6)
    )
    # The source article's own picture is the most on-topic image available, but
    # it is the outlet's copyrighted photo rather than a licensed one, and
    # YouTube's reused-content policy treats "reading materials you didn't
    # create (e.g., articles from websites)" as ineligible for monetisation.
    # Off by default: only CC0/PDM/BY/BY-SA pictures from Wikimedia Commons and
    # Openverse are used, which is the defensible position for a channel that is
    # meant to earn money. Set to 1 only if you accept that risk.
    allow_article_images: bool = field(
        default_factory=lambda: os.getenv("ALLOW_ARTICLE_IMAGES", "0").strip().lower()
        in {"1", "true", "yes"}
    )
    # Voices are rotated per video. YouTube names "channels where all slideshows
    # use the same voiceover across videos" as inauthentic, so a single fixed
    # voice is a channel-level risk, not a stylistic choice.
    tts_voice_pool: str = field(
        default_factory=lambda: os.getenv("TTS_VOICE_POOL", "") or ""
    )

    # --- Topic selection ---
    # A topic below this niche-match is off-brand and is not scripted.
    min_niche_score: float = field(
        default_factory=lambda: _env_float("MIN_NICHE_SCORE", 0.3)
    )
    min_total_score: float = field(
        default_factory=lambda: _env_float("MIN_TOTAL_SCORE", 0.35)
    )
    # Freshness half-life in hours. 24h => a 3-day-old story scores ~0.09.
    topic_half_life_h: float = field(
        default_factory=lambda: _env_float("TOPIC_HALF_LIFE_H", 24.0)
    )

    # --- Format targets (drives render length and script length) ---
    longform_target_s: int = field(
        default_factory=lambda: _env_int("LONGFORM_TARGET_S", 480)
    )
    shorts_target_s: int = field(
        default_factory=lambda: _env_int("SHORTS_TARGET_S", 45)
    )
    words_per_minute: int = field(
        default_factory=lambda: _env_int("WORDS_PER_MINUTE", 150)
    )

    # --- Render ---
    ffmpeg_path: str = field(
        default_factory=lambda: os.getenv("FFMPEG_PATH") or ""
    )
    # A single macOS `say` voice, e.g. "Daniel". Empty means rotate per video
    # from tts_voice_pool, which is the default: one fixed voice across every
    # upload is named in YouTube's inauthentic-content examples.
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE") or "")
    tts_rate_wpm: int = field(default_factory=lambda: _env_int("TTS_RATE_WPM", 175))
    tts_engine: str = field(
        default_factory=lambda: os.getenv("TTS_ENGINE") or "auto"  # auto | say | none
    )
    render_workers: int = field(
        default_factory=lambda: _env_int("RENDER_WORKERS", 2)
    )

    # --- Auto-advance ---
    # When True the pipeline renders and uploads a scripted video without a
    # human pressing `approve`. Off by default: it spends real quota and puts
    # unreviewed words on a public channel.
    auto_approve: bool = field(
        default_factory=lambda: os.getenv("AUTO_APPROVE", "0").strip().lower()
        in {"1", "true", "yes"}
    )

    # Lets a script pass validation when it was built by the offline template
    # because ANTHROPIC_API_KEY is missing. This exists so the pipeline can be
    # exercised end-to-end (and dry runs reviewed) without a key — it does NOT
    # make template content good enough to publish, so it is off by default and
    # never satisfies auto_approve on its own.
    allow_degraded: bool = field(
        default_factory=lambda: os.getenv("ALLOW_DEGRADED", "0").strip().lower()
        in {"1", "true", "yes"}
    )

    def llm_ready(self) -> bool:
        return self.resolve_llm_provider() is not None

    def youtube_ready(self) -> bool:
        return self.client_secret_path.exists() and self.token_path.exists()

    @property
    def niche_terms(self) -> List[str]:
        return [n.strip().lower() for n in self.niche.split(",") if n.strip()]

    # ------------------------------------------------------------ provider
    def resolve_llm_provider(self) -> Optional[str]:
        """Which backend to use, or None when no key is configured.

        `auto` prefers the free tier with the most headroom, so a missing or
        exhausted Gemini quota degrades into another free provider instead of
        a stalled run: Groq first (14,400 requests/day free), then NVIDIA NIM
        (40 rpm), then Gemini, then Anthropic.

        An explicit LLM_PROVIDER always wins, and pinning a provider with no key
        returns None rather than quietly falling back to another one — if you
        asked for a specific provider you get it or you get a clear "key not
        set" error, never a surprise model and a surprise bill.
        """
        want = (self.llm_provider or "auto").strip().lower()
        if want in ("gemini", "anthropic", "groq", "nvidia"):
            return want if self.provider_key(want) else None
        if want != "auto":
            log.warning("unknown LLM_PROVIDER %r; treating as auto", want)
        for name in ("groq", "nvidia", "gemini", "anthropic"):
            if self.provider_key(name):
                return name
        return None

    def provider_key(self, provider: str) -> Optional[str]:
        if provider == "gemini":
            return self.gemini_api_key
        if provider == "anthropic":
            return self.anthropic_api_key
        if provider == "groq":
            return self.groq_api_key
        if provider == "nvidia":
            return self.nvidia_api_key
        return None

    def provider_model(self, provider: str, *, fast: bool = False) -> str:
        if provider == "gemini":
            return self.gemini_model_fast if fast else self.gemini_model
        if provider == "groq":
            return self.groq_model
        if provider == "nvidia":
            return self.nvidia_model
        return self.llm_model_fast if fast else self.llm_model


def get_settings() -> Settings:
    for d in (DATA_DIR, MEDIA_DIR, LOG_DIR, SECRETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return Settings()
