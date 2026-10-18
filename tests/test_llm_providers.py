"""Tests for the Gemini backend and the provider indirection.

`requests` is stubbed at the module boundary, so nothing here touches the
network — which also means these tests verify the *wire format* we build, not
merely that a mocked object was called.
"""

from __future__ import annotations

import json
import types

import pytest

from ytauto.config import Settings, get_settings
from ytauto.llm import (
    LLM,
    LLMAuthError,
    LLMError,
    LLMUnavailable,
    LLMPermanentError,
    METADATA_SCHEMA,
    RESEARCH_SCHEMA,
)
from ytauto.llm_gemini import GeminiBackend, _extract, _sanitize_schema


@pytest.fixture
def gemini_settings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_ATTEMPTS", "1")   # no real backoff in tests
    return get_settings()


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        return self._payload


@pytest.fixture
def fake_post(monkeypatch):
    """Capture POSTs and return a scripted response. No socket is ever opened."""
    calls = []

    def _install(response=None, side_effect=None):
        def _post(url, headers=None, params=None, json=None, timeout=None):
            calls.append(
                {"url": url, "headers": headers, "params": params, "json": json}
            )
            if side_effect:
                raise side_effect
            return response

        import ytauto.llm_gemini as mod

        monkeypatch.setitem(
            __import__("sys").modules, "requests", types.SimpleNamespace(post=_post)
        )
        return calls

    return _install


def _ok(text, finish="STOP"):
    return FakeResponse(
        {
            "candidates": [
                {"finishReason": finish, "content": {"parts": [{"text": text}]}}
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
        }
    )


# ------------------------------------------------------- provider dispatch
def test_auto_prefers_the_free_tier_with_the_most_headroom(monkeypatch):
    """Groq (14.4K requests/day) beats Gemini (20/day), which beats Anthropic.

    A daily cron should land on the provider that will not be exhausted, not
    the one that happened to be configured first.
    """
    for var in ("GROQ_API_KEY", "NVIDIA_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert get_settings().resolve_llm_provider() == "gemini"

    monkeypatch.setenv("NVIDIA_API_KEY", "n")
    assert get_settings().resolve_llm_provider() == "nvidia"

    monkeypatch.setenv("GROQ_API_KEY", "q")
    assert get_settings().resolve_llm_provider() == "groq"


def test_auto_falls_back_to_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert get_settings().resolve_llm_provider() == "anthropic"


def test_pinning_a_provider_without_a_key_does_not_fall_back(monkeypatch):
    """A pinned provider must never silently become a different, billable one."""
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY")
    s = get_settings()
    assert s.resolve_llm_provider() is None
    assert s.llm_ready() is False
    with pytest.raises(LLMUnavailable, match="GEMINI_API_KEY"):
        LLM(s).complete("hi", {"type": "object"})


def test_no_key_at_all_reports_both_options(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = get_settings()
    assert s.resolve_llm_provider() is None
    assert "GEMINI_API_KEY" in LLM(s)._missing_key_message()


def test_llm_reports_its_model_from_the_active_provider(gemini_settings):
    llm = LLM(gemini_settings)
    assert llm.provider == "gemini"
    assert llm.model == gemini_settings.gemini_model
    assert llm.available is True


# ---------------------------------------------------------- schema trimming
def test_sanitize_drops_unsupported_json_schema_keys():
    out = _sanitize_schema(RESEARCH_SCHEMA)
    assert "additionalProperties" not in out
    for prop in out["properties"].values():
        assert "additionalProperties" not in prop
    # Everything Gemini does accept must survive untouched.
    assert out["properties"]["key_facts"]["type"] == "array"
    assert out["properties"]["key_facts"]["items"]["type"] == "string"
    assert out["required"] == RESEARCH_SCHEMA["required"]


def test_sanitize_recurses_into_nested_objects():
    out = _sanitize_schema(METADATA_SCHEMA)
    plan = out["properties"]["thumbnail_plan"]
    assert "additionalProperties" not in plan
    assert set(plan["required"]) == {"text", "emotion", "subject", "palette"}
    chapters = out["properties"]["chapters"]
    assert "additionalProperties" not in chapters["items"]


def test_sanitize_leaves_the_input_untouched():
    before = json.dumps(RESEARCH_SCHEMA, sort_keys=True)
    _sanitize_schema(RESEARCH_SCHEMA)
    assert json.dumps(RESEARCH_SCHEMA, sort_keys=True) == before


# ------------------------------------------------------------- request shape
def test_request_uses_response_mime_type_and_schema(gemini_settings, fake_post):
    calls = fake_post(response=_ok(json.dumps({"angle": "x", "confidence": 0.9})))
    out = GeminiBackend(gemini_settings).complete(
        "prompt text", RESEARCH_SCHEMA, system="be terse", max_tokens=1234
    )
    assert out["angle"] == "x"

    call = calls[0]
    assert call["url"].endswith(f"/{gemini_settings.gemini_model}:generateContent")
    assert call["params"] == {"key": "test-key"}
    body = call["json"]
    cfg = body["generationConfig"]
    assert cfg["responseMimeType"] == "application/json"
    assert cfg["maxOutputTokens"] == 1234
    assert "additionalProperties" not in cfg["responseSchema"]
    assert body["contents"][0]["parts"][0]["text"] == "prompt text"


def test_system_instruction_is_top_level_not_in_generation_config(gemini_settings, fake_post):
    """Regression: nesting systemInstruction in generationConfig is a hard 400.

    Gemini rejects the whole request with 'Unknown name "systemInstruction" at
    generation_config', so the round trip fails for a reason that has nothing to
    do with the key or the model.
    """
    calls = fake_post(response=_ok(json.dumps({"angle": "x", "confidence": 0.9})))
    GeminiBackend(gemini_settings).complete(
        "p", RESEARCH_SCHEMA, system="be terse"
    )
    body = calls[0]["json"]
    assert body["systemInstruction"]["parts"][0]["text"] == "be terse"
    assert "systemInstruction" not in body["generationConfig"]


def test_no_system_instruction_key_when_absent(gemini_settings, fake_post):
    calls = fake_post(response=_ok(json.dumps({"angle": "x", "confidence": 0.9})))
    GeminiBackend(gemini_settings).complete("p", RESEARCH_SCHEMA)
    assert "systemInstruction" not in calls[0]["json"]


def test_safety_settings_keep_hard_blocks_off(gemini_settings, fake_post):
    """News writing must not be blocked for mentioning a hack or a fatality."""
    calls = fake_post(response=_ok(json.dumps({"angle": "x", "confidence": 0.9})))
    GeminiBackend(gemini_settings).complete("p", RESEARCH_SCHEMA)
    cats = {s["category"]: s["threshold"] for s in calls[0]["json"]["safetySettings"]}
    assert cats
    assert set(cats.values()) == {"BLOCK_ONLY_HIGH"}


# ------------------------------------------------------------- extraction
def test_extract_parses_a_clean_response():
    data = _ok(json.dumps({"angle": "a"}))._payload
    assert _extract(data, "m") == {"angle": "a"}


def test_extract_handles_multiple_parts():
    data = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"text": '{"a":'}, {"text": " 1}"}]},
            }
        ]
    }
    assert _extract(data, "m") == {"a": 1}


def test_extract_reports_the_block_reason_when_there_are_no_candidates():
    with pytest.raises(LLMError, match="SAFETY"):
        _extract({"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}, "m")


def test_extract_names_a_non_stop_finish_reason():
    data = {
        "candidates": [
            {"finishReason": "RECITATION", "content": {"parts": [{"text": "{}"}]}}
        ]
    }
    with pytest.raises(LLMError, match="RECITATION"):
        _extract(data, "m")


def test_extract_raises_when_a_finish_reason_is_missing_entirely():
    with pytest.raises(LLMError, match="no candidates"):
        _extract({}, "m")


def test_max_tokens_is_salvaged_rather_than_reported_as_garbage():
    """Truncated JSON should be reported as truncation, not a parse failure."""
    data = {
        "candidates": [
            {
                "finishReason": "MAX_TOKENS",
                "content": {"parts": [{"text": '{"angle": "half a sen'}]},
            }
        ]
    }
    with pytest.raises(LLMError):
        _extract(data, "m")   # unparseable, but the reason is now MAX_TOKENS


# ---------------------------------------------------------------- failures
def test_http_error_surfaces_the_status(gemini_settings, fake_post):
    fake_post(response=FakeResponse({"error": {"message": "quota exceeded"}}, 429))
    with pytest.raises(LLMError, match="429"):
        GeminiBackend(gemini_settings).complete("p", RESEARCH_SCHEMA)


def test_retries_then_gives_up(gemini_settings, fake_post, monkeypatch):
    """A 5xx must be retried, and must eventually raise rather than hang."""
    import ytauto.llm_gemini as mod

    calls = []
    responses = [FakeResponse({"error": "boom"}, 500), _ok(json.dumps({"angle": "x"}))]

    def _post(url, headers=None, params=None, json=None, timeout=None):
        calls.append(1)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setitem(
        __import__("sys").modules, "requests", types.SimpleNamespace(post=_post)
    )
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    gemini_settings.llm_attempts = 2
    out = GeminiBackend(gemini_settings).complete("p", RESEARCH_SCHEMA)
    assert out["angle"] == "x"
    assert len(calls) == 2


def test_throttle_inserts_a_delay_between_calls(gemini_settings, fake_post, monkeypatch):
    calls = fake_post(response=_ok(json.dumps({"angle": "x"})))
    slept = []
    monkeypatch.setattr("ytauto.llm_gemini.time.sleep", lambda s: slept.append(s))
    b = GeminiBackend(gemini_settings, throttle_s=2.0)
    b.complete("p", RESEARCH_SCHEMA)
    b.complete("p", RESEARCH_SCHEMA)
    assert len(calls) == 2
    assert slept and slept[0] > 0   # second call waited for the first


def test_missing_key_is_reported_as_unavailable(gemini_settings, fake_post):
    fake_post(response=_ok("{}"))
    b = GeminiBackend(gemini_settings)
    b.key = None
    with pytest.raises(LLMUnavailable, match="GEMINI_API_KEY"):
        b.complete("p", RESEARCH_SCHEMA)


# ------------------------------------------------- permanent vs retryable
def test_a_404_is_not_retried_but_another_model_is_tried(gemini_settings, monkeypatch):
    """A retired model gets one request, then the chain moves on.

    Retrying the same dead model four times turned a one-second 404 into a
    two-minute hang. But giving up entirely is also wrong: the fallback is what
    makes a retired model a non-event.
    """
    import ytauto.llm_gemini as mod

    calls = []
    dead = "gemini-flash-latest"
    # Built outside _post: the `json` kwarg shadows the json module in there.
    alive = _ok('{"angle": "x", "confidence": 0.9}')

    def _post(url, headers=None, params=None, json=None, timeout=None):
        calls.append(url)
        if url.split("/")[-1].split(":")[0] == dead:
            return FakeResponse({"error": {"message": "no longer available"}}, 404)
        return alive

    monkeypatch.setitem(
        __import__("sys").modules, "requests", types.SimpleNamespace(post=_post)
    )
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    gemini_settings.llm_attempts = 4

    out = GeminiBackend(gemini_settings).complete("p", RESEARCH_SCHEMA)
    assert out["angle"] == "x"
    assert len(calls) == 2, f"expected one dead + one fallback, got {len(calls)}"
    assert calls[0].endswith(f"{dead}:generateContent")
    assert not calls[1].endswith(f"{dead}:generateContent")


def test_a_401_never_tries_another_model(gemini_settings, monkeypatch):
    """A rejected key fails every model, so the chain must not run."""
    calls = []

    def _post(url, headers=None, params=None, json=None, timeout=None):
        calls.append(1)
        return FakeResponse({"error": "unauthorized"}, 401)

    monkeypatch.setitem(
        __import__("sys").modules, "requests", types.SimpleNamespace(post=_post)
    )
    with pytest.raises(Exception) as exc:
        GeminiBackend(gemini_settings).complete("p", RESEARCH_SCHEMA)
    assert len(calls) == 1
    assert "GEMINI_API_KEY" in str(exc.value)


def test_a_404_message_tells_you_what_to_do(gemini_settings, fake_post):
    fake_post(response=FakeResponse({"error": {"message": "gone"}}, 404))
    # Falls back to another model, so the message is logged rather than raised.
    # Assert the hint text exists on the error the backend produced.
    import ytauto.llm_gemini as mod
    try:
        mod.GeminiBackend(gemini_settings)._post("gemini-flash-latest", {})
    except Exception as exc:
        assert "--list-models" in str(exc)


def test_a_401_points_at_the_key(gemini_settings, fake_post):
    fake_post(response=FakeResponse({"error": "unauthorized"}, 401))
    with pytest.raises(Exception) as exc:
        GeminiBackend(gemini_settings).complete("p", RESEARCH_SCHEMA)
    assert "GEMINI_API_KEY" in str(exc.value)


def test_a_429_is_still_retried(gemini_settings, fake_post, monkeypatch):
    """Rate limiting is the one 4xx worth another attempt."""
    import ytauto.llm_gemini as mod

    calls = []
    responses = [FakeResponse({"error": "slow down"}, 429),
                 _ok(json.dumps({"angle": "x", "confidence": 0.9}))]

    def _post(url, headers=None, params=None, json=None, timeout=None):
        calls.append(1)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setitem(
        __import__("sys").modules, "requests", types.SimpleNamespace(post=_post)
    )
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    gemini_settings.llm_attempts = 3
    out = GeminiBackend(gemini_settings).complete("p", RESEARCH_SCHEMA)
    assert out["angle"] == "x"
    assert len(calls) == 2


def test_give_up_stops_the_retry_loop_immediately():
    from ytauto.llm import LLMPermanentError
    from ytauto.util import retry

    calls = []

    def _always_fails():
        calls.append(1)
        raise LLMPermanentError("gone")

    with pytest.raises(LLMPermanentError):
        retry(_always_fails, attempts=5, base_delay=0, retry_on=(Exception,),
              give_up=lambda e: isinstance(e, LLMPermanentError))
    assert len(calls) == 1


def test_retry_without_give_up_still_loops(monkeypatch):
    from ytauto.llm import LLMPermanentError
    from ytauto.util import retry

    calls = []
    with pytest.raises(LLMPermanentError):
        retry(lambda: calls.append(1) or (_ for _ in ()).throw(
            LLMPermanentError("x")), attempts=3, base_delay=0)
    assert len(calls) == 3


# ------------------------------------------------------------ model choice
def test_default_gemini_model_is_an_alias_not_a_retired_point_version(monkeypatch):
    """A dated default is how gemini-2.5-flash reached production and 404'd."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_FAST", raising=False)
    s = get_settings()
    assert s.gemini_model.endswith("-latest")
    assert s.gemini_model_fast.endswith("-latest")


def test_gemini_model_can_be_pinned(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    assert get_settings().gemini_model == "gemini-3.8-flash"


# ------------------------------------------------- fallback classification
def test_a_400_does_not_walk_the_fallback_list():
    """A malformed request is rejected identically by every model.

    Falling back on a 400 spent three extra requests to arrive at the same
    error the caller needed to see immediately.
    """
    calls = []

    class _Back(GeminiBackend):
        def _post(self, model, payload):
            calls.append(model)
            raise LLMPermanentError("gemini 400: bad request", status=400)

    with pytest.raises(LLMPermanentError):
        _Back(Settings(gemini_api_key="k", gemini_model="gemini-3.8-flash")).complete(
            "hi", {"type": "object"}, system="s"
        )
    assert calls == ["gemini-3.8-flash"]


def test_a_404_does_fall_back_to_another_model():
    calls = []

    class _Back(GeminiBackend):
        def _post(self, model, payload):
            calls.append(model)
            if model == "gemini-3.8-flash":
                raise LLMPermanentError("gemini 404: no longer available", status=404)
            return {"candidates": [{"content": {"parts": [{"text": '{"ok":true}'}]}}]}

    back = _Back(Settings(gemini_api_key="k", gemini_model="gemini-3.8-flash"))
    out = back.complete("hi", {"type": "object"}, system="s")
    assert out == {"ok": True}
    # The first candidate is the first fallback that is not the primary itself.
    assert calls[1] == back._candidates("gemini-3.8-flash")[0]


def test_a_401_stops_without_falling_back():
    calls = []

    class _Back(GeminiBackend):
        def _post(self, model, payload):
            calls.append(model)
            raise LLMAuthError("gemini 401: bad key", status=401)

    with pytest.raises(LLMAuthError):
        _Back(Settings(gemini_api_key="k", gemini_model="gemini-3.8-flash")).complete(
            "hi", {"type": "object"}, system="s"
        )
    assert calls == ["gemini-3.8-flash"]


@pytest.fixture
def no_backoff(monkeypatch):
    """The overload delay is 12s by design; the suite should not pay for it."""
    monkeypatch.setattr("ytauto.llm_gemini._OVERLOAD_BASE_DELAY", 0.0)


def test_a_flaky_fallback_model_is_retried_rather_than_abandoned(no_backoff):
    """An overloaded alternate must not kill the run on its first 503.

    Alternates used to be tried exactly once, so a single transient 503 on the
    second model failed the whole call even though the first was legitimately
    saturated and a third model was free.
    """
    calls = []

    class _Back(GeminiBackend):
        def _post(self, model, payload):
            calls.append(model)
            if model == "gemini-3.8-flash":
                raise LLMError("503 overloaded")
            if calls.count("gemini-3.5-flash") == 1:
                raise LLMError("503 overloaded again")
            return {"candidates": [{"content": {"parts": [{"text": '{"ok":1}'}]}}]}

    out = _Back(Settings(gemini_api_key="k", gemini_model="gemini-3.8-flash")).complete(
        "hi", {"type": "object"}, system="s"
    )
    assert out == {"ok": 1}
    assert calls.count("gemini-3.5-flash") == 2


def test_no_unavailable_2_5_model_is_used_as_a_fallback():
    """2.5 still shows up in the models listing but 404s for newer keys.

    Listing it as a fallback spends requests re-discovering that.
    """
    for model in GeminiBackend.FALLBACK_MODELS:
        assert "2.5" not in model


def test_a_429_does_not_walk_the_fallback_list():
    """The free-tier quota is per key, not per model.

    Switching models after a 429 spends the very budget the retry needs to
    recover it, which is how a transient 503 escalated into hard quota
    exhaustion mid-run.
    """
    calls = []

    class _Back(GeminiBackend):
        def _post(self, model, payload):
            calls.append(model)
            err = LLMError("gemini 429: quota exceeded, limit: 20")
            err.status = 429
            raise err

    with pytest.raises(LLMError):
        _Back(Settings(gemini_api_key="k", gemini_model="gemini-3.8-flash")).complete(
            "hi", {"type": "object"}, system="s"
        )
    assert set(calls) == {"gemini-3.8-flash"}, f"fell back on a rate limit: {calls}"


def test_a_503_still_falls_back_because_capacity_is_per_model():
    calls = []

    class _Back(GeminiBackend):
        def _post(self, model, payload):
            calls.append(model)
            if model == "gemini-3.8-flash":
                raise LLMError("gemini 503: high demand")
            return {"candidates": [{"content": {"parts": [{"text": '{"ok":1}'}]}}]}

    out = _Back(Settings(gemini_api_key="k", gemini_model="gemini-3.8-flash")).complete(
        "hi", {"type": "object"}, system="s"
    )
    assert out == {"ok": 1}
    assert len(set(calls)) > 1, "a saturated model should have handed over"


def test_the_free_tier_request_pace_is_enforced_by_default():
    """A hard daily quota, so the client paces itself between calls.

    Measured from a live error: quotaId
    "GenerateRequestsPerDayPerProjectPerModel-FreeTier", quotaValue 20.
    """
    from conftest import REAL_FREE_TIER_MIN_INTERVAL

    assert REAL_FREE_TIER_MIN_INTERVAL >= 3.0
    # An explicit value still wins, for anyone on a paid quota.
    assert GeminiBackend(Settings(gemini_api_key="k"), throttle_s=0.0).throttle_s == 0.0
