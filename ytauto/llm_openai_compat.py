"""OpenAI-compatible chat backends: Groq and NVIDIA NIM.

Why these exist
---------------
Gemini's free tier allows 20 generateContent requests per day, per model, per
project (quotaId ``GenerateRequestsPerDayPerProjectPerModel-FreeTier``, verified
from a live quota error). A single long-form video costs two to eight calls, so
the free tier caps a day at a handful of videos.

Groq's free tier allows 14,400 requests a day on ``llama-3.1-8b-instant`` and
1,000 on ``llama-3.3-70b-versatile``, with no credit card, so it is orders of
magnitude less of a bottleneck. NVIDIA's NIM catalog allows 40 requests a minute
on the free developer program.

Cerebras was evaluated and rejected: its "free" tier is a $5, 30-day credit
grant that now requires a verified payment method, and there is no renewing
free allowance.

Both providers speak the OpenAI chat-completions dialect, so one class covers
them; only the base URL, key and model differ.
"""

from __future__ import annotations

import json
import re
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import Settings
from .llm import (
    LLMAuthError,
    LLMError,
    LLMPermanentError,
    LLMUnavailable,
)
from .logging_setup import get_logger
from .util import retry

log = get_logger(__name__)


# Groq free tier: 30 RPM. 30 RPM is one request every 2s, so a small floor keeps
# a retry burst from tripping it. The daily ceilings (14.4K / 1K requests) are
# the real constraint and are far above what this pipeline uses.
_MIN_INTERVAL_S = 2.1

_RETRY_HINT = re.compile(r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s", re.I)


# "for model" and the output/token meters are per model. "requests per minute"
# is deliberately absent: that one is metered per key, so every model on the
# key is already out of budget and handing over cannot help.
_MODEL_SCOPED_METER = re.compile(
    r"for model|\(OTPM\)|\(TPM\)|output tokens per minute|input tokens per minute|"
    r"tokens per minute",
    re.I,
)


def _is_model_scoped(detail: str) -> bool:
    """Whether a 429 is metered per model rather than per key.

    Both providers meter requests per key, so a key-level 429 is worth nothing
    on another model and the retry budget is the only sensible response. But
    Groq also meters output tokens per model -- the error reads "Rate limit
    reached for model `qwen/qwen3.8-27b` ... on output tokens per minute
    (OTPM): Limit 1000" -- and a different model has its own untouched
    allowance. Handing over there is exactly right, and refusing to was why a
    busy minute fell through to a provider with no quota left.
    """
    return bool(_MODEL_SCOPED_METER.search(detail or ""))


def _retry_after_seconds(detail: str) -> Optional[float]:
    """Seconds Groq asks us to wait, when it says."""
    m = _RETRY_HINT.search(detail or "")
    return float(m.group(1)) if m else None

_BASE_DELAY = 3.0


def _strip_fence(text: str) -> str:
    """Unwrap a ```json fenced block.

    JSON mode is a request, not a guarantee: Groq and NIM both sometimes return
    the object inside a fence even when asked not to, and a schema-fidelity
    retry is much more expensive than stripping three lines of punctuation.
    """
    t = (text or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.S)
    return m.group(1).strip() if m else t


def _loads_lenient(text: str) -> Dict[str, Any]:
    """Parse JSON, tolerating a bare object embedded in prose."""
    cleaned = _strip_fence(text)
    try:
        out = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        out = json.loads(cleaned[start : end + 1])
    if not isinstance(out, dict):
        raise ValueError(f"expected a JSON object, got {type(out).__name__}")
    return out


def _schema_hint(schema: Dict[str, Any]) -> str:
    """Spell the schema out in the prompt.

    Neither Groq nor NIM accepts a JSON Schema the way Gemini's
    ``responseSchema`` does; the strongest lever available is
    ``response_format: json_object`` plus the schema written into the prompt.
    """
    return json.dumps(schema, indent=2, sort_keys=True)


class OpenAICompatBackend:
    """One ``chat/completions`` client, pointed at Groq or NVIDIA NIM."""

    name = "openai-compat"

    # Tried in order when the primary model is overloaded or rate limited.
    #
    # Groq's catalogue turns over quickly -- llama-3.3-70b-versatile now answers
    # 404 model_not_found -- so this list is the set that `llm --list-models`
    # reported reachable, minus the models that cannot generate prose: the
    # llama-prompt-guard and gpt-oss-safeguard entries are safety classifiers
    # and the whisper entries are speech-to-text.
    #
    # qwen3.8-27b leads because it is not a reasoning model. The gpt-oss models
    # spend part of max_tokens on visible reasoning, which is what makes Groq
    # answer "max completion tokens reached before generating a valid document"
    # and return unparseable JSON; the small non-reasoning allam is the last
    # resort because it is the least likely to hold a schema together.
    FALLBACK_MODELS: Dict[str, List[str]] = {
        "groq": [
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "allam-2-7b",
        ],
        "nvidia": [
            "openai/gpt-oss-120b",
            "meta/llama-3.3-70b-instruct",
            "meta/llama-3.1-8b-instruct",
        ],
    }

    # These emit reasoning tokens that count against max_tokens before any JSON
    # is produced, so they need the cheapest reasoning setting we can ask for.
    REASONING_MODELS = ("openai/gpt-oss",)

    # A reasoning model that burns the whole budget on thinking returns
    # unparseable output rather than an error we can fall over on.
    REASONING_TOKEN_FLOOR = 16000

    # Groq meters output tokens per minute separately from requests, and the
    # free ceiling observed on qwen3.8-27b is 1000 OTPM -- roughly one
    # 1200-word script. Pacing on requests alone (2.1s) still overruns it,
    # because a single long script consumes the whole minute's budget.
    OUTPUT_TOKEN_LIMIT = 1000
    TOKEN_WINDOW_S = 60.0
    # Assume a worst-case reply when we have no measurement yet, so the first
    # call of a run cannot blow the budget either.
    ASSUMED_OUTPUT_TOKENS = 700
    # Never sit on a rate limit longer than this inside one pipeline step; past
    # it the honest move is to hand over to another provider or fail the run.
    MAX_INLINE_WAIT_S = 30.0

    def _token_window(self) -> Tuple[List[float], int]:
        """(timestamps, tokens) for completions still inside the rolling minute."""
        now = time.monotonic()
        keep = [(t, n) for t, n in self._token_use if now - t < self.TOKEN_WINDOW_S]
        self._token_use = keep
        return keep, sum(n for _, n in keep)

    def _await_token_headroom(self, payload: Optional[Dict[str, Any]]) -> None:
        """Sleep until this call is unlikely to trip the output-token meter.

        Sleeps for the oldest remaining entry when there is no room, rather
        than to the end of the window, so a run keeps its rhythm.
        """
    def _await_token_headroom(self, payload: Optional[Dict[str, Any]]) -> None:
        """Sleep until this call is unlikely to trip the output-token meter.

        Sleeps for the oldest remaining entry when there is no room, rather
        than to the end of the window, so a run keeps its rhythm.
        """
        if self.throttle_s <= 0:
            return
        window, used = self._token_window()
        limit = self.output_token_limit
        asked = int((payload or {}).get("max_tokens") or self.s.llm_max_tokens)
        expected = min(asked, self._last_output_tokens or self.ASSUMED_OUTPUT_TOKENS)
        if used + expected <= limit:
            return
        if not window:
            return
        wait = self.TOKEN_WINDOW_S - (time.monotonic() - window[0][0]) + 0.5
        if wait > 0:
            log.info(
                "%s output-token budget spent (%d/%d), waiting %.0fs",
                self.provider, used, limit, wait,
            )
            time.sleep(wait)

    def _record_tokens(self, data: Dict[str, Any]) -> None:
        usage = (data or {}).get("usage") or {}
        n = int(usage.get("completion_tokens") or 0)
        if n > 0:
            self._last_output_tokens = n
            self._token_use.append((time.monotonic(), n))
            self._token_window()

    def __init__(
        self,
        provider: str,
        settings: Settings,
        *,
        throttle_s: Optional[float] = None,
        fallback: bool = True,
    ) -> None:
        self.provider = provider
        self.name = provider
        self.s = settings
        self.fallback = fallback
        self.throttle_s = _MIN_INTERVAL_S if throttle_s is None else throttle_s
        # Per-minute output-token ceiling. 1000 is the conservative free-tier
        # default (small preview models); bigger models measured far higher
        # (gpt-oss-120b tolerated ~6.5k burst), so CI passes LLM_OTPM to lift it.
        self.output_token_limit = int(os.getenv("LLM_OTPM", str(self.OUTPUT_TOKEN_LIMIT)))
        self._last_call = 0.0
        # Rolling output-token usage, for the per-minute meter.
        self._token_use: List[Tuple[float, int]] = []
        self._last_output_tokens = 0
        self.key = self._key_for(provider, settings)
        self.base_url = self._base_url_for(provider, settings)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _key_for(provider: str, s: Settings) -> str:
        key = s.groq_api_key if provider == "groq" else s.nvidia_api_key
        if not key:
            raise LLMUnavailable(f"{provider.upper()}_API_KEY is not set")
        return key

    @staticmethod
    def _base_url_for(provider: str, s: Settings) -> str:
        if provider == "groq":
            return (s.groq_base_url or "https://api.groq.com/openai/v1").rstrip("/")
        return (
            s.nvidia_base_url
            or "https://integrate.api.nvidia.com/v1"
        ).rstrip("/")

    def default_model(self) -> str:
        if self.provider == "groq":
            return self.s.groq_model or "llama-3.3-70b-versatile"
        return self.s.nvidia_model or "meta/llama-3.3-70b-instruct"

    def list_models(self) -> List[str]:
        r = self._request("get", f"{self.base_url}/models", timeout=30.0)
        data = (r or {}).get("data") or []
        return sorted(str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id"))

    # ------------------------------------------------------------------- wire
    def _request(
        self,
        method: str,
        url: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        import requests  # lazy: keeps `requests` off the import path

        if self.throttle_s > 0 and method == "post":
            gap = time.monotonic() - self._last_call
            if gap < self.throttle_s:
                time.sleep(self.throttle_s - gap)
        if method == "post":
            self._await_token_headroom(payload)
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.request(
                method,
                url,
                headers=headers,
                json=payload,
                timeout=timeout or float(self.s.llm_timeout_s),
            )
        except requests.RequestException as exc:
            raise LLMError(f"{self.provider} transport error: {exc}") from exc
        finally:
            if method == "post":
                self._last_call = time.monotonic()

        if resp.status_code == 200:
            self._record_tokens(resp.json())
            return resp.json()

        detail = (resp.text or "")[:400]
        if resp.status_code in (401, 403):
            raise LLMAuthError(
                f"{self.provider} HTTP {resp.status_code}: {detail} — check the API key.",
                status=resp.status_code,
            )
        if resp.status_code == 429 or resp.status_code >= 500:
            err = LLMError(
                f"{self.provider} HTTP {resp.status_code}: {detail}"
            )
            err.status = resp.status_code  # type: ignore[attr-defined]
            # Groq names the exhausted meter and how long until it refills, e.g.
            # "output tokens per minute (OTPM): Limit 1000, Used 961,
            # Requested 363. Please try again in 19.44s". Without this, a limit
            # that clears in 20 seconds burns the whole model fallback chain and
            # ends the run.
            err.retry_after_s = _retry_after_seconds(detail)  # type: ignore[attr-defined]
            # Not every 429 is alike, and which kind it is decides whether
            # another model is worth trying.
            err.model_scoped = _is_model_scoped(detail)  # type: ignore[attr-defined]
            raise err
        raise LLMPermanentError(
            f"{self.provider} HTTP {resp.status_code}: {detail}",
            status=resp.status_code,
        )

    def _chat(
        self,
        model: str,
        prompt: str,
        schema: Dict[str, Any],
        system: str,
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "Return a single JSON object and nothing else: no prose, no "
                    "markdown, no code fence. It must match this schema exactly:\n\n"
                    f"{_schema_hint(schema)}"
                ),
            }
        )
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": max_tokens or self.s.llm_max_tokens,
            # json_object is the only structured-output knob these APIs expose.
            "response_format": {"type": "json_object"},
        }
        if any(model.startswith(p) for p in self.REASONING_MODELS):
            # Reasoning tokens are billed against max_tokens, so the default
            # budget gets spent before the JSON document is ever closed and
            # Groq rejects the whole response. Cheapest reasoning first.
            payload["reasoning_effort"] = "low"
            payload["max_tokens"] = max(
                payload["max_tokens"], self.REASONING_TOKEN_FLOOR
            )
        data = self._request(
            "post",
            f"{self.base_url}/chat/completions",
            payload=payload,
        )
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"{self.provider} {model}: response had no choices")
        text = (choices[0].get("message") or {}).get("content") or ""
        if not text.strip():
            raise LLMError(f"{self.provider} {model}: empty completion")
        return {"text": text, "usage": data.get("usage") or {}, "model": model}

    # ------------------------------------------------------------------- api
    def complete(
        self,
        prompt: str,
        schema: Dict[str, Any],
        *,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        primary = model or self.default_model()
        candidates = [primary] + [
            m for m in self.FALLBACK_MODELS.get(self.provider, []) if m != primary
        ]
        if not self.fallback:
            candidates = candidates[:1]

        last: Optional[BaseException] = None
        for idx, m in enumerate(candidates):
            if idx:
                # A rate limit the provider told us how long to wait for is
                # not a reason to abandon this provider: the meter refills in
                # seconds. Only wait here if the hint was actionable.
                hint = getattr(last, "retry_after_s", None)
                if hint and 0 < hint <= self.MAX_INLINE_WAIT_S:
                    log.info(
                        "%s rate limited; waiting %.1fs before %s",
                        self.provider, hint, m,
                    )
                    time.sleep(hint)
                elif not _is_handover(last):
                    # Only a capacity or availability problem is worth handing
                    # over. A 400 means our request is malformed and every
                    # model rejects it the same way; an unexplained 429 is a
                    # per-key limit that a different model does not escape.
                    raise last  # type: ignore[misc]
                log.warning(
                    "%s %s unavailable (%s); trying %s",
                    self.provider, primary, last, m,
                )
            try:
                return self._with_schema_retry(
                    m, prompt, schema, system, max_tokens
                )
            except (LLMAuthError, LLMUnavailable):
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
                log.warning("%s %s failed: %s", self.provider, m, exc)
        raise last if last else LLMError("no candidate model available")

    def _with_schema_retry(
        self,
        model: str,
        prompt: str,
        schema: Dict[str, Any],
        system: str,
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """One call, then at most one retry aimed at fidelity.

        A schema miss is not a transport failure, so it gets its own budget:
        re-sending the same prompt plus the previous answer plus the specific
        complaint fixes a wrong shape far more often than a blind retry does,
        and a blind retry would just produce the same wrong shape.
        """
        attempt_extra = ""

        def _once() -> Dict[str, Any]:
            resp = self._chat(model, prompt, schema, system, max_tokens)
            try:
                return _loads_lenient(resp["text"])
            except Exception as exc:  # noqa: BLE001
                raise SchemaMismatch(
                    f"{self.provider} {model} returned unusable JSON: {exc}"
                ) from exc

        try:
            return retry(
                _once,
                attempts=2,
                base_delay=_BASE_DELAY,
                retry_on=(SchemaMismatch,),
                give_up=lambda e: False,
            )
        except SchemaMismatch as exc:
            log.warning("%s; retrying with the schema spelled out again", exc)
            resp = self._chat(
                model,
                prompt + attempt_extra,
                schema,
                system,
                max_tokens,
            )
            return _loads_lenient(resp["text"])

    def note_usage(self, data: Dict[str, Any]) -> None:
        usage = data.get("usage") or {}
        log.debug(
            "%s usage: in=%s out=%s",
            self.provider,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )


class SchemaMismatch(LLMError):
    """The model answered, but not with an object matching the schema."""


def _is_handover(exc: Optional[BaseException]) -> bool:
    """Whether moving to another model could plausibly help.

    - 404: the model name is not on this provider; another name will be.
    - 429: a per-key limit is shared by every model, so another one would fail
      identically. A per-model limit (Groq's output-token meters) is not shared
      and handing over is worth it.
    - 5xx: capacity is per-model on both providers, so hand over.
    """
    if exc is None:
        return False
    if isinstance(exc, LLMUnavailable):
        return False
    status = getattr(exc, "status", None)
    if status == 429:
        return bool(getattr(exc, "model_scoped", False))
    if isinstance(exc, LLMPermanentError):
        return status == 404
    return True
