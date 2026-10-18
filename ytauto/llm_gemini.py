"""Gemini backend for the Generative Language API.

Implemented against the REST endpoint with `requests` rather than the
`google-genai` SDK, for two reasons:

1. `requests` is already a dependency, so enabling this costs no extra install.
2. The SDK pulls in a large transitive tree for what is ultimately one POST.

Free tier: a key from Google AI Studio works with no billing account attached.
The trade-off against Claude is real and worth stating plainly — free-tier
models are less reliable at following a strict JSON schema, so
`_sanitize_schema` trims the schema to the subset Gemini actually honours and
`_parse_json` still guards the result.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .config import Settings
from .llm import LLMAuthError, LLMPermanentError, LLMError, LLMUnavailable
from .logging_setup import get_logger
from .util import retry

log = get_logger("llm.gemini")

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# "This model is currently experiencing high demand" is a queue, not a fault:
# waiting 3s and doubling is too eager, and the observed free-tier failure mode
# was burning all four attempts in ~90s and still not getting through. Start
# much later so an early retry has a real chance of landing.
_OVERLOAD_BASE_DELAY = 12.0
# The free tier allows 20 generateContent requests PER DAY per model per project
# (quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier). This floor is
# not what protects the budget -- the call count does -- but it stops a retry
# storm from burning several of the twenty in a second.
_FREE_TIER_MIN_INTERVAL = 3.2
# The rate-limit window is a minute; one wait covers it.
_RATE_LIMIT_WAIT_S = 60

# Retries on the *primary* model before switching. Kept small on purpose: see
# _with_fallback.
_PRIMARY_ATTEMPTS = 2


def _is_recoverable(exc: BaseException) -> bool:
    """Whether a *different model* stands any chance of succeeding.

    Three cases, and the distinction matters:

    - 404: the model is gone, another one may work. Fall back.
    - 503: capacity is pooled per model, so a busy one can coexist with a free
      one. Fall back.
    - 429: the free tier allows ~20 requests a minute for the whole key, not per
      model. Every alternate is equally rate limited, so walking the fallback
      list here spends the quota that the retry needed to recover it -- the run
      turned a transient 503 into a hard quota exhaustion. Wait instead.

    A 400 is worse still: our own request is malformed and every model rejects
    it identically, so there is nothing to fall back to.
    """
    # 429 is a rate limit on the account, not a per-model rejection: every
    # alternate is throttled identically, so switching models cannot help and
    # only spends the quota the retry needs.
    if _is_rate_limited(exc):
        return False
    if isinstance(exc, LLMPermanentError):
        return getattr(exc, "status", None) == 404
    return True


def _is_rate_limited(exc: BaseException) -> bool:
    return getattr(exc, "status", None) == 429

# Gemini's responseSchema is a subset of JSON Schema. Anything outside it is
# rejected outright, so the full schemas in llm.py are trimmed before sending.
_ALLOWED_KEYS = {
    "type", "description", "properties", "required", "items",
    "enum", "format", "nullable", "propertyOrdering",
}


def _sanitize_schema(node: Any) -> Any:
    """Reduce a JSON Schema to what Gemini's `responseSchema` accepts.

    Notably drops `additionalProperties`, which Gemini rejects outright, and
    `type: "number"` is kept as-is (Gemini accepts both `number` and `integer`).

    `properties` needs special handling: its keys are the *field names* from the
    schema, not keywords, so they must survive even though none of them appear
    in the allowlist.
    """
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                out[key] = {name: _sanitize_schema(sub) for name, sub in value.items()}
            elif key in _ALLOWED_KEYS:
                out[key] = _sanitize_schema(value)
        return out
    if isinstance(node, list):
        return [_sanitize_schema(v) for v in node]
    return node


class GeminiBackend:
    """Retrying wrapper over `models/{model}:generateContent`."""

    name = "gemini"

    # Tried in order when the primary model is overloaded or unavailable.
    # Capacity is pooled per model on the free tier, so a busy `flash-latest`
    # often means only that alias; a different point version is frequently free
    # at the same moment.
    #
    # The 2.5 models are deliberately absent. They still appear in the models
    # listing but return 404 for newer API keys, so listing them as a fallback
    # just spends two requests proving what the error message already says. The
    # numbered models are what the API recommends for a fresh key.
    FALLBACK_MODELS = (
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-flash-lite-latest",
    )

    def __init__(
        self,
        settings: Settings,
        *,
        throttle_s: Optional[float] = None,
        fallback: bool = True,
    ) -> None:
        self.s = settings
        self.key = settings.gemini_api_key
        # Resolved at call time, not as a default argument value, so the
        # constant stays patchable by the test suite.
        self.throttle_s = _FREE_TIER_MIN_INTERVAL if throttle_s is None else throttle_s
        self.fallback = fallback
        self._last_call = 0.0

    # ---------------------------------------------------------------- setup
    def _post(self, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        import requests  # imported lazily: keeps `requests` off the import path

        if not self.key:
            raise LLMUnavailable("GEMINI_API_KEY is not set")

        url = f"{API_ROOT}/{model}:generateContent"
        headers = {"Content-Type": "application/json"}

        # The free tier is rate limited per minute. A small floor between calls
        # keeps a --count 5 run from tripping 429s and burning the retry budget
        # that should be reserved for real failures.
        if self.throttle_s > 0:
            gap = time.monotonic() - self._last_call
            if gap < self.throttle_s:
                time.sleep(self.throttle_s - gap)
        self._last_call = time.monotonic()

        resp = requests.post(
            url, headers=headers, params={"key": self.key}, json=payload, timeout=120
        )
        if resp.status_code >= 400:
            detail = _short(resp.text, 400)
            # 429 and 5xx are worth another go; everything else 4xx is not.
            if resp.status_code in (429,) or resp.status_code >= 500:
                # Retryable, but the status is still carried: 429 means the
                # account is out of quota and a different model will not help,
                # which is not something the type alone can say.
                err = LLMError(f"gemini {model} HTTP {resp.status_code}: {detail}")
                err.status = resp.status_code  # type: ignore[attr-defined]
                raise err

            hint = ""
            if resp.status_code == 404:
                hint = (
                    " — this model is not available to your key. Run "
                    "`ytauto llm --list-models`, or unset GEMINI_MODEL to use the "
                    "default alias, which tracks the current release."
                )
            if resp.status_code in (401, 403):
                raise LLMAuthError(
                    f"gemini HTTP {resp.status_code}: {detail} — check GEMINI_API_KEY.",
                    status=resp.status_code,
                )
            raise LLMPermanentError(
                f"gemini {model} HTTP {resp.status_code}: {detail}{hint}",
                status=resp.status_code,
            )
        return resp.json()

    # ----------------------------------------------------------------- api
    def complete(
        self,
        prompt: str,
        schema: Dict[str, Any],
        *,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        model_id = model or self.s.provider_model("gemini")
        clean = _sanitize_schema(schema)

        config: Dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseSchema": clean,
            # The flash models love to pad. A slightly higher temperature makes
            # the prose less uniform without inviting the JSON drift that a
            # high temperature would.
            "temperature": 0.7,
        }
        if max_tokens or self.s.llm_max_tokens:
            config["maxOutputTokens"] = max_tokens or self.s.llm_max_tokens

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config,
            "safetySettings": _PERMISSIVE_SAFETY,
        }
        # systemInstruction is a top-level field, NOT part of generationConfig.
        # Nesting it there is a 400: "Unknown name \"systemInstruction\"".
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        def _call() -> Dict[str, Any]:
            data = self._post(model_id, payload)
            return _extract(data, model_id)

        def _with_fallback() -> Dict[str, Any]:
            # A 429 is the account's minute budget, not this model's capacity.
            # Wait out the window once on the same model instead of walking the
            # candidate list, where every option is throttled identically and
            # the retries only deepen the exhaustion.
            attempts_used = {"n": 0}

            def _attempt() -> Dict[str, Any]:
                try:
                    return _call()
                except Exception as exc:  # noqa: BLE001
                    if _is_rate_limited(exc) and attempts_used["n"] < 1:
                        attempts_used["n"] += 1
                        log.warning(
                            "gemini %s hit the free-tier rate limit; waiting %ds "
                            "for the window to reset", model_id, _RATE_LIMIT_WAIT_S
                        )
                        time.sleep(_RATE_LIMIT_WAIT_S)
                        return _call()
                    raise

            # A 503 that survives one backoff is not a blip, it is a saturated
            # free-tier pool. Spending four attempts (12+24+48s) on the primary
            # before trying anything else turned a recoverable hiccup into a
            # four-minute stall, so the primary gets a deliberately small budget
            # and then hands over.
            try:
                return retry(
                    _attempt,
                    attempts=min(self.s.llm_attempts, _PRIMARY_ATTEMPTS),
                    base_delay=_OVERLOAD_BASE_DELAY,
                    retry_on=(Exception,),
                    give_up=lambda e: isinstance(e, LLMPermanentError),
                    on_error=lambda n, e: log.warning(
                        "gemini %s call failed (%d): %s", model_id, n, e
                    ),
                )
            except LLMAuthError:
                # Wrong key: every model would fail identically.
                raise
            except Exception as exc:  # noqa: BLE001
                if not self.fallback or not _is_recoverable(exc):
                    raise
                for alt in self._candidates(model_id):
                    log.warning(
                        "gemini %s unavailable (%s); trying %s", model_id, exc, alt
                    )
                    # The alternate gets the same retry budget as the primary.
                    # A single overloaded alternate must not kill the run, which
                    # is what happened when they were tried exactly once.
                    try:
                        return retry(
                            lambda a=alt: _extract(self._post(a, payload), a),
                            attempts=min(self.s.llm_attempts, _PRIMARY_ATTEMPTS),
                            base_delay=_OVERLOAD_BASE_DELAY,
                            retry_on=(Exception,),
                            give_up=lambda e: isinstance(e, LLMPermanentError),
                            on_error=lambda n, e: log.warning(
                                "gemini %s call failed (%d): %s", alt, n, e
                            ),
                        )
                    except Exception as next_exc:  # noqa: BLE001
                        if not _is_recoverable(next_exc):
                            raise
                        log.warning("gemini %s also unavailable: %s", alt, next_exc)
                raise

        return _with_fallback()

    def _candidates(self, primary: str) -> List[str]:
        out = [m for m in self.FALLBACK_MODELS if m != primary]
        for m in (self.s.gemini_model, self.s.gemini_model_fast):
            if m and m != primary and m not in out:
                out.append(m)
        return out

    def note_usage(self, data: Dict[str, Any]) -> None:
        meta = data.get("usageMetadata") or {}
        log.debug(
            "gemini usage: in=%s out=%s",
            meta.get("promptTokenCount"),
            meta.get("candidatesTokenCount"),
        )


# Gemini blocks these categories by default, which is wrong for a news-writing
# tool: discussing a shooting, a hack or a company failure is the job. Only the
# hard blocks stay on.
_PERMISSIVE_SAFETY = [
    {"category": c, "threshold": "BLOCK_ONLY_HIGH"}
    for c in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


def _short(text: str, limit: int = 300) -> str:
    text = (text or "").replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def _extract(data: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """Pull the JSON document out of a generateContent response.

    Checks the finish reason first: a truncated response (MAX_TOKENS) parses as
    invalid JSON, and reporting that as "the model returned garbage" hides the
    real cause.
    """
    candidates = data.get("candidates") or []
    if not candidates:
        block = data.get("promptFeedback") or {}
        raise LLMError(
            f"gemini {model_id} returned no candidates "
            f"(finishReason={block.get('blockReason') or 'n/a'})"
        )
    cand = candidates[0]
    finish = cand.get("finishReason") or "STOP"
    if finish not in ("STOP", "MAX_TOKENS"):
        raise LLMError(f"gemini {model_id} finishReason={finish}")

    parts = ((cand.get("content") or {}).get("parts")) or []
    texts: List[str] = [p.get("text", "") for p in parts if p.get("text")]
    if not texts:
        raise LLMError(f"gemini {model_id} produced no text parts")

    from .llm import _parse_json  # single JSON parser for every provider

    if finish == "MAX_TOKENS":
        # Salvage what we can: a truncated object often has valid leading fields.
        log.warning("gemini %s hit MAX_TOKENS; salvaging partial JSON", model_id)
    return _parse_json("\n".join(texts))
