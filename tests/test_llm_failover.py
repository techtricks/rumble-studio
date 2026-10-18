"""Cross-provider failover: a spent quota should not end the run.

Gemini's free tier is 20 requests a day. If a run picks one provider at startup
and that budget is already spent, every topic fails. These tests pin the
behaviour that keeps a daily schedule alive.
"""

from __future__ import annotations

import pytest

from ytauto.config import Settings
from ytauto.llm import LLM, LLMAuthError, LLMError, LLMUnavailable, LLMPermanentError

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


class _StubBackend:
    def __init__(self, name, behaviour):
        self.name = name
        self.behaviour = behaviour
        self.calls = 0

    def complete(self, prompt, schema, *, system="", model=None, max_tokens=None):
        self.calls += 1
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        return self.behaviour


def _quota_error():
    e = LLMError("HTTP 429: quota exhausted for the day")
    e.status = 429
    return e


def _three_providers():
    s = Settings(gemini_api_key="g", nvidia_api_key="n", groq_api_key="q")
    llm = LLM(s)
    assert llm.provider_order() == ["groq", "nvidia", "gemini"]
    return llm


def test_a_spent_daily_quota_falls_over_to_another_free_provider():
    llm = _three_providers()
    groq = _StubBackend("groq", _quota_error())
    nvidia = _StubBackend("nvidia", {"ok": True})
    gemini = _StubBackend("gemini", {"ok": "never"})
    llm._backends = {"groq": groq, "nvidia": nvidia, "gemini": gemini}

    assert llm.complete("hi", SCHEMA) == {"ok": True}
    assert groq.calls == 1
    assert nvidia.calls == 1
    assert gemini.calls == 0


def test_the_handover_keeps_going_until_one_provider_answers():
    llm = _three_providers()
    llm._backends = {
        "groq": _StubBackend("groq", _quota_error()),
        "nvidia": _StubBackend("nvidia", LLMError("503 high demand")),
        "gemini": _StubBackend("gemini", {"ok": 3}),
    }
    assert llm.complete("hi", SCHEMA) == {"ok": 3}


def test_a_bad_key_is_reported_rather_than_hidden_behind_a_working_provider():
    """Stepping over a 401 would leave a broken key unnoticed indefinitely."""
    llm = LLM(Settings(gemini_api_key="g", groq_api_key="q"))
    nvidia_free = _StubBackend("nvidia", {"ok": True})
    llm._backends = {
        "groq": _StubBackend("groq", LLMAuthError("HTTP 401", status=401)),
        "nvidia": nvidia_free,
    }
    with pytest.raises(LLMAuthError):
        llm.complete("hi", SCHEMA)
    assert nvidia_free.calls == 0


def test_a_malformed_request_is_not_tried_on_every_provider():
    """A 400 is our bug; four vendors will all reject it identically."""
    llm = _three_providers()
    llm._backends = {
        "groq": _StubBackend("groq", LLMPermanentError("400 bad", status=400)),
        "nvidia": _StubBackend("nvidia", {"ok": True}),
        "gemini": _StubBackend("gemini", {"ok": True}),
    }
    with pytest.raises(LLMPermanentError):
        llm.complete("hi", SCHEMA)


def test_with_only_one_provider_configured_a_quota_error_surfaces():
    llm = LLM(Settings(gemini_api_key="g"))
    llm._backends = {"gemini": _StubBackend("gemini", _quota_error())}
    with pytest.raises(LLMError):
        llm.complete("hi", SCHEMA)


def test_pinning_a_provider_keeps_the_handover_within_that_provider():
    """A pinned provider must not silently become a different vendor."""
    llm = LLM(Settings(gemini_api_key="g", groq_api_key="q", llm_provider="gemini"))
    assert llm.provider_order() == ["gemini"]
    llm._backends = {
        "gemini": _StubBackend("gemini", _quota_error()),
        "groq": _StubBackend("groq", {"ok": True}),
    }
    with pytest.raises(LLMError):
        llm.complete("hi", SCHEMA)


def test_an_explicit_model_id_routes_to_the_provider_that_serves_it():
    """A Gemini model name on Groq's API is a guaranteed 404.

    The provider order puts Groq first, so without an ownership check the
    pinned model went to the wrong vendor and the call was wasted.
    """
    llm = LLM(Settings(gemini_api_key="g", groq_api_key="q"))
    groq = _StubBackend("groq", _quota_error())
    gemini = _StubBackend("gemini", {"ok": True})
    llm._backends = {"groq": groq, "gemini": gemini}

    assert llm.complete("hi", SCHEMA, model="gemini-3.8-flash") == {"ok": True}
    assert groq.calls == 0
    assert gemini.calls == 1


def test_an_explicit_groq_model_routes_to_groq():
    llm = LLM(Settings(gemini_api_key="g", groq_api_key="q"))
    groq = _StubBackend("groq", {"ok": True})
    llm._backends = {
        "groq": groq,
        "gemini": _StubBackend("gemini", _quota_error()),
    }
    assert llm.complete("hi", SCHEMA, model="llama-3.3-70b-versatile") == {"ok": True}
    assert groq.calls == 1


def test_no_keys_at_all_reports_every_free_option():
    with pytest.raises(LLMUnavailable):
        LLM(Settings()).complete("hi", SCHEMA)


def test_the_handover_predicate_matches_the_documented_policy():
    from ytauto.llm import _is_provider_handover

    not_found = LLMPermanentError("gone", status=404)
    bad_request = LLMPermanentError("bad", status=400)
    quota = _quota_error()
    capacity = LLMError("503")
    capacity.status = 503

    assert _is_provider_handover(not_found) is True
    assert _is_provider_handover(quota) is True
    assert _is_provider_handover(capacity) is True
    assert _is_provider_handover(bad_request) is False
    assert _is_provider_handover(None) is False
