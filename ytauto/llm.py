"""Anthropic-backed content generation.

Two things matter here:

1. **Structured output, not prose parsing.** Every call asks for a JSON schema
   via `output_config.format`, so there is no regex-scraping a code fence and no
   "model returned 2 JSON objects" failure mode. If the API rejects the schema
   (older model / proxy), we transparently fall back to tool-use.

2. **A deterministic offline path.** With no `ANTHROPIC_API_KEY`, `research` /
   `script` / `metadata` fall back to template builders built from the harvested
   topic text. The result is low quality but *structurally valid*, so the whole
   pipeline — render, upload, schedule — is runnable and testable without
   spending a cent. It always logs that quality is degraded.

The offline path is the reason the pipeline can be verified end to end before
anyone has an API key.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import Settings, get_settings
from .logging_setup import get_logger
from .util import clip, norm_text, retry, sentences, slugify, words

log = get_logger("llm")


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """No API key configured."""


class LLMPermanentError(LLMError):
    """A failure that retrying cannot fix: 400, 404, 401, 403.

    Distinguished from a transient 429/5xx because retrying a permanent error
    only multiplies the delay: a Gemini call that 404s in one second would
    otherwise burn four attempts and well over a minute to report the same
    answer.

    `status` carries the HTTP code so callers can still tell a retired model
    (404, worth trying a different one) from a malformed request (400, which
    every model rejects identically).
    """

    def __init__(self, message: str, status: "int | None" = None) -> None:
        super().__init__(message)
        self.status = status


class LLMAuthError(LLMPermanentError):
    """The credential itself is wrong.

    Separate because it is the one permanent failure a different model cannot
    fix: if the key is rejected, every model is rejected, so a fallback chain
    would just spend four more requests to reach the same conclusion.
    """


# ------------------------------------------------------------- dataclasses
@dataclass
class Research:
    """What the model learned about a topic before anything is written."""

    angle: str = ""
    summary: str = ""
    key_facts: List[str] = field(default_factory=list)
    why_now: str = ""
    audience_pull: str = ""
    risks: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.5
    # True only when no LLM produced this (offline fallback, or every call
    # failed). Kept separate from `confidence` because "the model was unsure
    # about a real story" and "no model ran at all" need different handling: the
    # first is a bad topic, the second is a missing API key.
    degraded: bool = False

    @property
    def is_thin(self) -> bool:
        """Too little substance to build a video on."""
        return len(self.key_facts) < 2 or not self.angle


@dataclass
class Script:
    """Narration plus the structural pieces the renderer and uploader need."""

    hook: str = ""
    body: str = ""
    outro: str = ""
    hook_seconds: float = 0.0

    @property
    def text(self) -> str:
        return "\n\n".join(p for p in (self.hook, self.body, self.outro) if p)

    @property
    def word_count(self) -> int:
        return words(self.text)


@dataclass
class Metadata:
    """Everything YouTube needs that is not the video file itself."""

    title: str = ""
    title_b: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    thumbnail_plan: Dict[str, Any] = field(default_factory=dict)
    chapters: List[Dict[str, Any]] = field(default_factory=list)


# ------------------------------------------------------------------ client
def _is_provider_handover(exc: Optional[BaseException]) -> bool:
    """Whether a different provider stands a chance of succeeding.

    Cross-provider failover exists for the daily ceilings, so the cases that
    matter are a spent quota and a saturated model:

    - 429: the day's allowance is gone. Waiting cannot help within a run, but
      another vendor's counter is untouched. This is the case that actually
      happens on Gemini's 20/day.
    - 404: the model id is not served. Another provider will not know that id,
      so this only justifies a handover when no model was pinned.
    - 5xx: capacity. Worth another vendor.

    Not a handover: a 400, because our request is malformed and every vendor
    rejects it the same way, and a 401/403, because a rejected key is a
    configuration error the operator has to see.
    """
    if exc is None:
        return False
    if isinstance(exc, LLMPermanentError):
        return getattr(exc, "status", None) == 404
    return True


class LLM:
    """Provider-agnostic wrapper.

    Dispatches to a backend chosen by `Settings.resolve_llm_provider`. Anthropic
    and Gemini differ in almost every wire detail — auth, system prompt,
    structured-output mechanism, token accounting — so each is hidden behind one
    `complete()` and nothing above this class knows which model is answering.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.s = settings or get_settings()
        self._client: Any = None
        self._backend: Any = None
        self._backends: Optional[Dict[str, Any]] = None
        self._prefer: Optional[str] = None
        self._supports_structured: Optional[bool] = None
        self.usage = {"calls": 0, "in_tokens": 0, "out_tokens": 0}

    @property
    def provider(self) -> Optional[str]:
        """The provider that will be tried first."""
        order = self.provider_order()
        return order[0] if order else None

    @property
    def model(self) -> str:
        p = self.provider or "anthropic"
        return self.s.provider_model(p)

    @property
    def available(self) -> bool:
        return bool(self.provider_order())

    @property
    def backend(self) -> Any:
        """The backend for the preferred provider.

        Kept for the existing call sites; `complete()` goes through
        `_backend_for` so it can walk the provider order.
        """
        name = self.provider
        if name is None:
            raise LLMUnavailable(self._missing_key_message())
        return self._backend_for(name)

    def _missing_key_message(self) -> str:
        pinned = (self.s.llm_provider or "auto").strip().lower()
        if pinned in ("gemini", "anthropic", "groq", "nvidia"):
            var = {
                "gemini": "GEMINI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "groq": "GROQ_API_KEY",
                "nvidia": "NVIDIA_API_KEY",
            }[pinned]
            return f"{var} is not set (LLM_PROVIDER={pinned})"
        return (
            "no LLM key set — set GROQ_API_KEY (free, 14.4K requests/day), "
            "GEMINI_API_KEY (free, 20/day), NVIDIA_API_KEY (free, 40 rpm) "
            "or ANTHROPIC_API_KEY"
        )

    # -- raw call ------------------------------------------------------
    def provider_order(self) -> List[str]:
        """Configured providers, best free headroom first.

        `resolve_llm_provider` alone would freeze one provider for the whole
        process, and on a 20-requests-a-day Gemini quota that means a run dies
        at the first topic. Every provider with a key is a candidate, so an
        exhausted one is stepped over instead.
        """
        pinned = (self.s.llm_provider or "auto").strip().lower()
        if pinned in ("gemini", "anthropic", "groq", "nvidia"):
            return [pinned] if self.s.provider_key(pinned) else []
        out = []
        for name in ("groq", "nvidia", "gemini", "anthropic"):
            if self.s.provider_key(name):
                out.append(name)
        return out

    def complete(
        self,
        prompt: str,
        schema: Dict[str, Any],
        *,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return a dict guaranteed to match `schema`'s top level.

        Tries each configured provider in turn and keeps the first that works,
        so an exhausted daily quota degrades into the next free provider rather
        than into a failed video.
        """
        order = self.provider_order()
        if not order:
            raise LLMUnavailable(self._missing_key_message())

        # An explicit model id belongs to one provider, and a foreign name is a
        # guaranteed 404: sending "gemini-3.8-flash" to Groq, or "llama-3.3-70b"
        # to Gemini, would burn a request to learn what we already know. So the
        # model decides the provider, not the other way round.
        if model:
            owner = [n for n in order if self._model_belongs_to(n, model)]
            order = owner or order[:1]

        last: Optional[BaseException] = None
        for idx, name in enumerate(order):
            if idx:
                if not _is_provider_handover(last):
                    raise last  # type: ignore[misc]
                log.warning(
                    "llm provider %s unavailable (%s); falling over to %s",
                    order[idx - 1], last, name,
                )
            try:
                data = self._backend_for(name).complete(
                    prompt, schema,
                    system=system, model=model, max_tokens=max_tokens,
                )
            except (LLMAuthError, LLMUnavailable):
                # A wrong key is a configuration mistake, not bad luck. Stepping
                # over it would hide a broken key behind a working neighbour,
                # so it is reported instead of swallowed.
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
                continue
            self._prefer = name
            self.usage["calls"] += 1
            return data
        raise last if last else LLMUnavailable(self._missing_key_message())

    def _model_belongs_to(self, provider: str, model: str) -> bool:
        """Whether `model` is one this provider actually serves."""
        if provider == "gemini":
            names = {self.s.gemini_model, self.s.gemini_model_fast}
            try:
                from .llm_gemini import GeminiBackend

                names |= set(GeminiBackend.FALLBACK_MODELS)
            except Exception:  # noqa: BLE001
                pass
            # `models/...` prefixed ids come from the models endpoint.
            return model in names or model.split("/")[-1] in names or model.startswith("gemini")
        if provider in ("groq", "nvidia"):
            try:
                from .llm_openai_compat import OpenAICompatBackend

                names = set(OpenAICompatBackend.FALLBACK_MODELS.get(provider, []))
            except Exception:  # noqa: BLE001
                names = set()
            names.add(self.s.provider_model(provider))
            return model in names
        return model in (self.s.llm_model, self.s.llm_model_fast)

    def _backend_for(self, name: str) -> Any:
        """Cached per-provider backend."""
        if self._backends is None:
            self._backends = {}
        back = self._backends.get(name)
        if back is None:
            if name == "gemini":
                from .llm_gemini import GeminiBackend

                back = GeminiBackend(self.s)
            elif name in ("groq", "nvidia"):
                from .llm_openai_compat import OpenAICompatBackend

                back = OpenAICompatBackend(name, self.s)
            else:
                back = _AnthropicBackend(self, self.s)
            self._backends[name] = back
        return back

    # Kept for the Anthropic backend, which needs the raw SDK handle.
    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.s.anthropic_api_key:
                raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
            from anthropic import Anthropic

            self._client = Anthropic(
                api_key=self.s.anthropic_api_key,
                timeout=float(self.s.llm_timeout_s),
                max_retries=0,  # we do our own backoff, with logging
            )
        return self._client

    def _track(self, resp: Any, model_id: str) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        self.usage["in_tokens"] += getattr(usage, "input_tokens", 0) or 0
        self.usage["out_tokens"] += getattr(usage, "output_tokens", 0) or 0
        log.debug(
            "llm %s: in=%s out=%s",
            model_id,
            getattr(usage, "input_tokens", 0),
            getattr(usage, "output_tokens", 0),
        )


class _AnthropicBackend:
    """The original Anthropic implementation, behind the same `complete()`."""

    name = "anthropic"

    def __init__(self, owner: LLM, settings: Settings) -> None:
        self.owner = owner
        self.s = settings

    def complete(
        self,
        prompt: str,
        schema: Dict[str, Any],
        *,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        model_id = model or self.s.provider_model("anthropic")
        kwargs: Dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens or self.s.llm_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if self.owner._supports_structured is not False:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": schema}
            }

        def _call() -> Dict[str, Any]:
            try:
                resp = self.owner.client.messages.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                if self.owner._supports_structured is not False and _is_schema_unsupported(exc):
                    log.warning("structured output rejected by %s; using tool-use", model_id)
                    self.owner._supports_structured = False
                    kwargs.pop("output_config", None)
                    return _call()
                raise
            self.owner._track(resp, model_id)
            return _extract_json(resp)

        return retry(
            _call,
            attempts=self.s.llm_attempts,
            base_delay=2.0,
            retry_on=(Exception,),
            give_up=lambda e: isinstance(e, LLMPermanentError),
            on_error=lambda n, e: log.warning("llm call failed (%d): %s", n, e),
        )


_llm: Optional[LLM] = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm


# ------------------------------------------------------------- extraction
def _is_schema_unsupported(exc: Exception) -> bool:
    text = str(exc).lower()
    return "output_config" in text or "structured" in text or "json_schema" in text


def _extract_json(resp: Any) -> Dict[str, Any]:
    """Pull the JSON payload out of a response, whichever mode produced it."""
    # Structured output: content[0].text is already the JSON document.
    for block in getattr(resp, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            return _parse_json(getattr(block, "text", "") or "")
        if btype == "tool_use":
            payload = getattr(block, "input", None)
            if isinstance(payload, dict):
                return payload
    raise LLMError(f"no JSON in response (stop_reason={getattr(resp, 'stop_reason', '?')})")


def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise LLMError("empty response body")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: model wrapped it in a fence despite the schema request.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"unparseable JSON: {exc}") from exc
    raise LLMError("no JSON object found in response")


# ---------------------------------------------------------------- schemas
RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "angle": {"type": "string", "description": "One sentence: the specific, non-generic take a viewer would want."},
        "summary": {"type": "string", "description": "2-3 sentence factual summary."},
        "key_facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-6 specific, checkable facts. No filler.",
        },
        "why_now": {"type": "string", "description": "Why this matters this week."},
        "audience_pull": {"type": "string", "description": "The question a viewer arrived with."},
        "risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Claims that are uncertain, unverified, or legally sensitive. Empty if none.",
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "URLs or outlet names supporting the facts.",
        },
    },
    "required": ["angle", "summary", "key_facts", "why_now", "audience_pull", "risks", "sources"],
    "additionalProperties": False,
}

SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "hook": {"type": "string", "description": "The spoken opening. Must land the payoff immediately."},
        "body": {"type": "string", "description": "The main narration, structured with blank lines between beats."},
        "outro": {"type": "string", "description": "Short close + soft call to subscribe."},
    },
    "required": ["hook", "body", "outro"],
    "additionalProperties": False,
}

# Script and metadata in one response.
#
# The free tier allows 20 generateContent requests PER DAY per model per project
# (quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier), so every call
# saved is a call that can be a second video. Metadata is derived from the
# script the model has just written, which is the cheapest possible way to spend
# nothing extra on it.
SCRIPT_WITH_META_SCHEMA = {
    "type": "object",
    "properties": {
        "hook": {"type": "string"},
        "body": {"type": "string"},
        "outro": {"type": "string"},
        "title": {"type": "string"},
        "title_b": {"type": "string"},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "thumbnail_plan": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "emotion": {"type": "string"},
                "subject": {"type": "string"},
                "palette": {"type": "string"},
            },
            "required": ["text", "emotion", "subject", "palette"],
            "additionalProperties": False,
        },
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "time_s": {"type": "integer"},
                    "title": {"type": "string"},
                },
                "required": ["time_s", "title"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["hook", "body", "outro", "title", "title_b", "description", "tags", "thumbnail_plan"],
    "additionalProperties": False,
}

METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "title_b": {"type": "string", "description": "A genuinely different second title for the same video, for A/B testing."},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "thumbnail_plan": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Max 4 words burned into the thumbnail."},
                "emotion": {"type": "string"},
                "subject": {"type": "string"},
                "palette": {"type": "string"},
            },
            "required": ["text", "emotion", "subject", "palette"],
            "additionalProperties": False,
        },
        "chapters": {
            "type": "array",
            "description": "Optional. Must be empty for Shorts.",
            "items": {
                "type": "object",
                "properties": {
                    "time_s": {"type": "integer"},
                    "title": {"type": "string"},
                },
                "required": ["time_s", "title"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "title_b", "description", "tags", "thumbnail_plan", "chapters"],
    "additionalProperties": False,
}
