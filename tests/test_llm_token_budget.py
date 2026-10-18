"""Groq meters output tokens per minute, which request pacing does not cover.

A single 1200-word script is roughly the whole 1000-token free ceiling, so
pacing calls 2.1s apart still trips OTPM mid-pipeline. When it does, Groq says
how long until the meter refills; ignoring that burns the model fallback chain
and ends the run over a limit that clears in twenty seconds.
"""

from __future__ import annotations

import pytest

from ytauto.config import Settings
from tests.test_llm_groq import _Resp, _capture, _ok, no_pacing  # noqa: F401
from ytauto.llm_openai_compat import (
    OpenAICompatBackend,
    _retry_after_seconds,
)

GROQ_OTPM_429 = (
    '{"error":{"message":"Rate limit reached for model `qwen/qwen3.8-27b` in '
    'organization `org_1` service tier `on_demand` on output tokens per minute '
    '(OTPM): Limit 1000, Used 961, Requested 363. Please try again in 19.44s."}}'
)


def test_the_provider_retry_hint_is_parsed():
    assert _retry_after_seconds(GROQ_OTPM_429) == 19.44


def test_no_hint_in_the_message_means_no_hint():
    assert _retry_after_seconds("rate limit reached") is None


def test_a_429_carries_its_retry_hint_on_the_error(no_pacing, monkeypatch):
    seen = _capture(monkeypatch, [_Resp(429, text=GROQ_OTPM_429)])
    b = OpenAICompatBackend("groq", Settings(groq_api_key="k"))
    with pytest.raises(Exception) as ei:
        b.complete("hi", {"type": "object"})
    assert getattr(ei.value, "retry_after_s", None) == 19.44


def test_the_token_meter_records_completion_tokens(no_pacing, monkeypatch):
    _capture(monkeypatch, [_ok('{"ok": true}', completion_tokens=812)])
    b = OpenAICompatBackend("groq", Settings(groq_api_key="k"))
    b.complete("hi", {"type": "object"})
    assert b._last_output_tokens == 812


def test_the_free_output_ceiling_is_smaller_than_one_script():
    """This is why pacing on requests alone is not enough."""
    b = OpenAICompatBackend("groq", Settings(groq_api_key="k"))
    # ~1200 words of narration in the JSON response.
    assert b.ASSUMED_OUTPUT_TOKENS < b.OUTPUT_TOKEN_LIMIT


def test_an_actionable_retry_hint_waits_instead_of_abandoning_the_provider(
    no_pacing, monkeypatch
):
    """A 429 that clears in 20s should not burn the fallback chain.

    Handing over on every rate limit meant one busy minute ended the run: the
    next provider was Gemini, whose free tier is 20 requests a day and was
    already spent. The provider is asked to wait when it says how long.
    """
    from tests.test_llm_groq import _capture

    monkeypatch.setattr(OpenAICompatBackend, "MAX_INLINE_WAIT_S", 30.0)
    slept = []
    monkeypatch.setattr(
        "ytauto.llm_openai_compat.time.sleep", lambda s: slept.append(s)
    )
    _capture(monkeypatch, [_Resp(429, text=GROQ_OTPM_429), _ok('{"ok": true}')])
    b = OpenAICompatBackend("groq", Settings(groq_api_key="k"))
    assert b.complete("hi", {"type": "object"}) == {"ok": True}
    assert slept == [19.44]


def test_a_long_retry_hint_is_not_waited_out(no_pacing, monkeypatch):
    """Past the inline budget, handing over beats blocking a pipeline step."""
    from tests.test_llm_groq import _capture

    slept = []
    monkeypatch.setattr(
        "ytauto.llm_openai_compat.time.sleep", lambda s: slept.append(s)
    )
    # 45s is past MAX_INLINE_WAIT_S (0 under no_pacing), so we must not sit on it.
    detail = GROQ_OTPM_429.replace("19.44s", "45s")
    # One 429 then enough successes to cover the whole fallback chain.
    _capture(monkeypatch, [_Resp(429, text=detail)] + [_ok('{"ok": true}')] * 5)
    b = OpenAICompatBackend("groq", Settings(groq_api_key="k"))
    assert b.complete("hi", {"type": "object"}) == {"ok": True}
    assert 45.0 not in slept


def test_a_narrow_token_budget_throttles_before_the_meter_trips(no_pacing):
    """Self-throttling is the point: one script is the whole free ceiling."""
    b = OpenAICompatBackend("groq", Settings(groq_api_key="k"))
    assert b.OUTPUT_TOKEN_LIMIT == 1000
    assert b.ASSUMED_OUTPUT_TOKENS <= b.OUTPUT_TOKEN_LIMIT


@pytest.mark.parametrize(
    "detail,model_scoped",
    [
        # Groq's output-token meter names the model: another model is untouched.
        ("Rate limit reached for model `qwen/qwen3.8-27b` on output tokens per "
         "minute (OTPM): Limit 1000", True),
        ("Rate limit reached for model `x` on tokens per minute (TPM)", True),
        # A per-key request limit is shared, so handing over cannot help.
        ("Rate limit reached for organization `org_1`: Requests per minute limit "
         "exceeded", False),
        ("Rate limit reached", False),
    ],
)
def test_a_429_is_handover_worthy_only_when_metered_per_model(
    no_pacing, monkeypatch, detail, model_scoped
):
    """A 429 is not one thing.

    The rule used to be "never hand over on a 429, every model shares the
    key's limits". Groq's output-token meter is per model -- the error names it
    -- so refusing to hand over there sent a live run to a provider with no
    quota left. A key-level limit still must not hand over.
    """
    from tests.test_llm_groq import _capture

    _capture(monkeypatch, [_Resp(429, text=detail)] + [_ok('{"ok": true}')] * 5)
    b = OpenAICompatBackend("groq", Settings(groq_api_key="k"))
    if model_scoped:
        assert b.complete("hi", {"type": "object"}) == {"ok": True}
    else:
        with pytest.raises(Exception):
            b.complete("hi", {"type": "object"})
