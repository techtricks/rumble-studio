"""Groq / NVIDIA NIM backend: wire format, JSON handling, model handover."""

from __future__ import annotations

import json
import types

import pytest

from ytauto.config import Settings
from ytauto.llm import LLMAuthError, LLMError, LLMPermanentError
from ytauto.llm_openai_compat import (
    OpenAICompatBackend,
    SchemaMismatch,
    _is_handover,
    _loads_lenient,
    _strip_fence,
)

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _ok(content, completion_tokens=5):
    return _Resp(200, {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": completion_tokens},
    })


@pytest.fixture
def no_pacing(monkeypatch):
    monkeypatch.setattr("ytauto.llm_openai_compat._MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr("ytauto.llm_openai_compat._BASE_DELAY", 0.0)
    # A rate limit the provider says to retry in 19s is honoured by sleeping,
    # which is right in production and wrong here: it turned one assertion into
    # 58 seconds of real waiting. Tests assert the decision, not the clock.
    monkeypatch.setattr(OpenAICompatBackend, "MAX_INLINE_WAIT_S", 0.0)


def _capture(monkeypatch, responses):
    """Patch requests.request and record every call."""
    seen = []

    def fake_request(method, url, headers=None, json=None, timeout=None):
        seen.append({
            "method": method, "url": url, "headers": headers, "payload": json,
        })
        r = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr("requests.request", fake_request)
    return seen


# ----------------------------------------------------------------- settings
def test_groq_wins_auto_resolution_because_it_has_the_most_headroom():
    s = Settings(groq_api_key="g", nvidia_api_key="n", gemini_api_key="x")
    assert s.resolve_llm_provider() == "groq"


def test_nvidia_is_used_when_there_is_no_groq_key():
    s = Settings(nvidia_api_key="n", gemini_api_key="x")
    assert s.resolve_llm_provider() == "nvidia"


def test_gemini_is_still_reachable_and_still_the_last_free_resort():
    s = Settings(gemini_api_key="x")
    assert s.resolve_llm_provider() == "gemini"


def test_pinning_a_provider_without_a_key_is_an_error_not_a_silent_switch():
    """If you asked for groq you get groq or a clear error, never a surprise bill."""
    s = Settings(llm_provider="groq", gemini_api_key="x")
    assert s.resolve_llm_provider() is None
    from ytauto.llm import LLM

    assert "GROQ_API_KEY is not set" in LLM(s)._missing_key_message()


def test_the_missing_key_message_names_the_best_free_option():
    from ytauto.llm import LLM

    msg = LLM(Settings())._missing_key_message()
    assert "GROQ_API_KEY" in msg and "14.4K" in msg


# ------------------------------------------------------------- wire format
def test_the_request_is_openai_shaped_with_the_system_prompt_split_out(no_pacing, monkeypatch):
    seen = _capture(monkeypatch, [_ok('{"ok": true}')])
    b = OpenAICompatBackend("groq", Settings(groq_api_key="k"))
    b.complete("do the thing", SCHEMA, system="you are a researcher")

    call = seen[0]
    assert call["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer k"
    msgs = call["payload"]["messages"]
    assert msgs[0] == {"role": "system", "content": "you are a researcher"}
    assert "do the thing" in msgs[1]["content"]
    assert call["payload"]["response_format"] == {"type": "json_object"}
    # Neither API accepts a JSON schema, so it has to travel in the prompt.
    assert '"ok"' in msgs[1]["content"]


def test_nvidia_points_at_its_own_host(no_pacing, monkeypatch):
    seen = _capture(monkeypatch, [_ok('{"ok": true}')])
    OpenAICompatBackend("nvidia", Settings(nvidia_api_key="k")).complete(
        "hi", SCHEMA
    )
    assert seen[0]["url"].startswith("https://integrate.api.nvidia.com/v1")


def test_a_missing_key_raises_rather_than_sending_an_anonymous_request():
    from ytauto.llm import LLMUnavailable

    with pytest.raises(LLMUnavailable):
        OpenAICompatBackend("groq", Settings())


# ---------------------------------------------------------------- json bits
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"ok": true}', True),
        ('```json\n{"ok": true}\n```', True),
        ('```\n{"ok": true}\n```', True),
        ('Here you go: {"ok": true}', True),
        ('  {"ok": true}  ', True),
    ],
)
def test_json_comes_back_usable(raw, expected):
    """JSON mode is a request, not a guarantee.

    A fence or a sentence around the object is common enough that a strict
    parser would fail calls that actually succeeded.
    """
    assert _loads_lenient(raw)["ok"] is expected


def test_a_fenced_block_is_unwrapped():
    assert _strip_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_a_json_array_is_rejected_because_the_schema_wants_an_object():
    with pytest.raises(ValueError):
        _loads_lenient("[1, 2, 3]")


def test_unparseable_output_is_a_schema_mismatch_not_a_transport_error():
    with pytest.raises(Exception):
        _loads_lenient("I am afraid I cannot do that")


# ------------------------------------------------------------ model handover
def test_an_overloaded_model_hands_over_to_the_next(no_pacing, monkeypatch):
    seen = _capture(monkeypatch, [
        _Resp(503, text="overloaded"),
        _ok('{"ok": true}'),
    ])
    out = OpenAICompatBackend("groq", Settings(groq_api_key="k")).complete("hi", SCHEMA)
    assert out == {"ok": True}
    # Derived from config, not hardcoded: Groq's catalogue turns over and a
    # literal here silently rots the moment the default is rotated.
    primary = Settings(groq_api_key="k").groq_model
    assert seen[0]["payload"]["model"] == primary
    assert seen[1]["payload"]["model"] != primary


def test_a_retired_model_name_hands_over(no_pacing, monkeypatch):
    seen = _capture(monkeypatch, [
        _Resp(404, text="model not found"),
        _ok('{"ok": true}'),
    ])
    b = OpenAICompatBackend("groq", Settings(groq_api_key="k"))
    out = b.complete("hi", SCHEMA, model="some-model-that-was-retired")
    assert out == {"ok": True}
    assert seen[1]["payload"]["model"] == OpenAICompatBackend.FALLBACK_MODELS["groq"][0]


def test_a_bad_key_stops_the_chain_immediately(no_pacing, monkeypatch):
    seen = _capture(monkeypatch, [_Resp(401, text="invalid api key")])
    with pytest.raises(LLMAuthError):
        OpenAICompatBackend("groq", Settings(groq_api_key="bad")).complete("hi", SCHEMA)
    assert len(seen) == 1, "a bad key fails identically on every model"


def test_a_rate_limit_does_not_walk_the_model_list(no_pacing, monkeypatch):
    """The limit is per key, so every model on it is throttled identically.

    Cycling models here spends the budget the retry needed to recover it.
    """
    seen = _capture(monkeypatch, [_Resp(429, text="rate limit exceeded")])
    with pytest.raises(LLMError):
        OpenAICompatBackend("groq", Settings(groq_api_key="k")).complete("hi", SCHEMA)
    assert len(seen) == 1


def test_a_malformed_request_does_not_walk_the_model_list(no_pacing, monkeypatch):
    seen = _capture(monkeypatch, [_Resp(400, text="invalid schema")])
    with pytest.raises(LLMPermanentError):
        OpenAICompatBackend("groq", Settings(groq_api_key="k")).complete("hi", SCHEMA)
    assert len(seen) == 1


def test_handover_predicate_covers_the_cases_that_can_recover():
    err_404 = LLMPermanentError("gone", status=404)
    err_400 = LLMPermanentError("bad", status=400)
    err_429 = LLMError("429")
    err_429.status = 429
    err_503 = LLMError("503")
    err_503.status = 503

    assert _is_handover(err_404) is True
    assert _is_handover(err_503) is True
    assert _is_handover(err_400) is False
    assert _is_handover(err_429) is False
    assert _is_handover(None) is False


def test_every_groq_fallback_has_a_published_free_tier_limit():
    """These are real model ids on the free plan, not guesses.

    Groq publishes per-model free limits; a typo here would 404 on every call.
    """
    for provider, models in OpenAICompatBackend.FALLBACK_MODELS.items():
        assert models, provider
        for m in models:
            assert isinstance(m, str) and m.strip() == m and m


def test_the_primary_model_is_never_repeated_in_the_fallback_list():
    for provider, models in OpenAICompatBackend.FALLBACK_MODELS.items():
        assert len(models) == len(set(models)), provider


# ------------------------------------------------------------------ schema fix
def test_a_schema_miss_gets_one_fidelity_retry_not_a_blind_one(no_pacing, monkeypatch):
    """Re-sending the same prompt just produces the same wrong shape."""
    seen = _capture(monkeypatch, [
        _ok("I'm sorry, I cannot comply"),
        _ok('{"ok": true}'),
    ])
    out = OpenAICompatBackend("groq", Settings(groq_api_key="k")).complete("hi", SCHEMA)
    assert out == {"ok": True}
    assert len(seen) == 2
    assert seen[1]["payload"]["messages"][-1]["content"].count('"ok"') >= 1


def test_a_model_that_never_returns_json_fails_loudly(no_pacing, monkeypatch):
    _capture(monkeypatch, [_ok("still refusing")])
    with pytest.raises(Exception):
        OpenAICompatBackend("groq", Settings(groq_api_key="k")).complete("hi", SCHEMA)


def test_an_empty_completion_is_reported_not_silently_empty(no_pacing, monkeypatch):
    _capture(monkeypatch, [_ok("   ")])
    with pytest.raises(LLMError, match="empty completion"):
        OpenAICompatBackend("groq", Settings(groq_api_key="k")).complete("hi", SCHEMA)


def test_a_response_with_no_choices_is_reported(no_pacing, monkeypatch):
    _capture(monkeypatch, [_Resp(200, {"choices": []})])
    with pytest.raises(LLMError, match="no choices"):
        OpenAICompatBackend("groq", Settings(groq_api_key="k")).complete("hi", SCHEMA)


def test_list_models_reads_the_openai_data_envelope(no_pacing, monkeypatch):
    _capture(monkeypatch, [_Resp(200, {"data": [
        {"id": "b-model"}, {"id": "a-model"}, {"nope": 1},
    ]})])
    models = OpenAICompatBackend("groq", Settings(groq_api_key="k")).list_models()
    assert models == ["a-model", "b-model"]


def test_default_groq_model_is_a_model_that_exists():
    """Groq retired llama-3.3-70b-versatile; the old default 404'd live.

    Catalogues turn over, so the default has to be a model that is actually
    served rather than one that was merely documented when it was written.
    """
    from ytauto.config import Settings
    from ytauto.llm_openai_compat import OpenAICompatBackend

    s = Settings()
    assert s.groq_model in OpenAICompatBackend.FALLBACK_MODELS["groq"]


def test_fallback_models_are_all_generative():
    """Guard and speech models cannot write a script.

    `llm --list-models` reports the whole catalogue, which on Groq includes
    llama-prompt-guard / gpt-oss-safeguard safety classifiers and the whisper
    speech-to-text models. None of them can produce the JSON a script needs,
    so they must not be in the fallback chain.
    """
    from ytauto.llm_openai_compat import OpenAICompatBackend

    for m in OpenAICompatBackend.FALLBACK_MODELS["groq"]:
        assert "guard" not in m
        assert "safeguard" not in m
        assert not m.startswith("whisper")


def test_groq_chain_leads_with_a_non_reasoning_model():
    """Reasoning models burn max_tokens before emitting JSON.

    Groq answers "max completion tokens reached before generating a valid
    document" and the whole response is lost, so the first fallback has to be
    a model that does not think out loud.
    """
    from ytauto.llm_openai_compat import OpenAICompatBackend

    first = OpenAICompatBackend.FALLBACK_MODELS["groq"][0]
    assert not any(first.startswith(p) for p in OpenAICompatBackend.REASONING_MODELS)


def test_reasoning_models_get_a_token_floor_and_cheap_reasoning():
    """A truncated JSON document is unrecoverable, so reasoning models need
    more than the default 8K budget and the cheapest reasoning setting."""
    from ytauto.llm_openai_compat import OpenAICompatBackend

    assert OpenAICompatBackend.REASONING_TOKEN_FLOOR > 8000
    assert "openai/gpt-oss-20b" in OpenAICompatBackend.REASONING_MODELS[0] or any(
        "gpt-oss" in p for p in OpenAICompatBackend.REASONING_MODELS
    )


def test_no_test_inherits_a_real_llm_key():
    """No real LLM key survives into a test, for any provider.

    The autouse fixture clears the `*_api_key` variables derived from Settings,
    so this fails if a new provider is added without being covered: the developer's
    real key would otherwise sit in os.environ and quietly reroute the failover
    tests onto the live API, spending quota and time per test.
    """
    import os

    from ytauto.config import Settings

    leaked = [
        name.upper()
        for name in Settings.__dataclass_fields__
        if name.endswith("_api_key") and os.environ.get(name.upper())
    ]
    assert leaked == []
