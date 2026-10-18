"""Padding to length is a real defect even when every word is "true".

A live 1232-word script passed every other gate -- grounded confidence 0.68,
correct duration, on-niche -- while ending in 141 sentences of which twelve
opened with "The test" and eleven with "It is":

    The test is the proof. The rollout is the expansion. The partnership is
    the foundation. The market is the target.

No sentence is duplicated verbatim, so exact-match deduplication sees a clean
script. The frame repeats, and a viewer hears filler.
"""

from __future__ import annotations

from ytauto.validate import check_repetition

PADDED = (
    "Google is testing a direct integration between its AI interface and "
    "Flipkart. The test allows select users to initiate a purchase through "
    "Gemini. The integration connects the conversational layer directly to "
    "Flipkart's inventory. It is a significant shift from Google's previous "
    "role. It is a concrete step in a major market. It is a test of the "
    "future of shopping. The test is the proof. The rollout is the "
    "expansion. The partnership is the foundation. The market is the target. "
    "The technology is the enabler. The experience is the goal. The test is "
    "the first step. The rollout is the next."
)

VARIED = " ".join(
    [
        "Google rebuilt its shopping stack over two years.",
        "Competitors noticed the latency drop within a week.",
        "Analysts remain split on whether margins survive competition.",
        "Early data suggests users finish more sessions per visit.",
        "The company declined to publish a rollout timeline.",
        "Retail partners are rewriting their own fulfilment flows.",
        "Nobody will say what the next integration will be.",
        "Payments still route through the original checkout.",
        "Returns policy differs sharply between regions.",
        "Support volume fell by half last quarter.",
        "Mobile traffic now outnumbers desktop again.",
        "Search quality improved on long queries.",
        "A rival quietly launched a similar pilot.",
        "Watchdog groups asked for disclosure in August.",
        "Advertisers noticed lower click-through rates.",
        "Regulators have not commented publicly yet.",
        "Engineering blog posts went up on Tuesday.",
        "Internal tooling was rewritten from scratch.",
        "Latency fell below two hundred milliseconds.",
        "Cache hit rates improved substantially.",
        "The pilot covers a few hundred listings only.",
        "Checkout still opens a new tab on mobile.",
        "Nothing about pricing has been announced.",
        "One executive called it an early experiment.",
        "Wall Street barely reacted to the news.",
        "Recruiting posts hint at further rollout.",
        "Competitors are watching the same metrics.",
        "Few users noticed anything different.",
        "Accessibility complaints have not surfaced.",
        "Publishers are concerned about traffic.",
        "Checkout flow adds a single confirmation step.",
        "Nothing suggests a hard launch date.",
        "Internal docs reference a staged plan.",
        "Two regions were added last month.",
        "Merchant fees appear unchanged so far.",
        "Analysts want to see a full quarter first.",
        "Several partners declined to comment.",
        "The integration still depends on one API.",
        "Latency spikes remain an open question.",
        "Retention barely moved in early tests.",
    ]
)


def test_a_padded_script_is_flagged():
    assert check_repetition(PADDED)


def test_varied_prose_is_not_flagged():
    """The detector has to survive ordinary writing."""
    assert check_repetition(VARIED) == []


def test_the_flag_names_the_repeated_frame():
    """A warning a human cannot act on is not useful."""
    problems = check_repetition(PADDED)
    assert any("the test" in p for p in problems)


def test_no_flag_without_a_script():
    assert check_repetition("") == []


def test_a_very_short_script_is_left_to_the_length_check():
    """Repetition share is meaningless below a dozen sentences."""
    assert check_repetition("The test is the proof. The test is the proof.") == []


def test_padding_is_reported_as_a_warning_not_an_error():
    """The facts underneath a padded script are still worth keeping.

    Discarding a well-sourced video because its ending is filler would push an
    operator towards ALLOW_DEGRADED=1, which is the worse failure.
    """
    from ytauto.validate import validate_package

    res = validate_package(
        kind="longform",
        title="Google Tests AI Shopping With Flipkart in India",
        title_b="Google Tests AI Shopping With Flipkart in India",
        thumbnail_plan={},
        description="Google is testing a direct integration between Gemini and "
                    "Walmart-owned Flipkart in India, letting select users buy "
                    "products inside the chat interface during a limited pilot.",
        tags=["google", "flipkart", "ai", "shopping", "india", "gemini"],
        script=PADDED,
        research={
            "angle": "Google is becoming a transactional partner, not a "
                     "referral engine.",
            "summary": "A live pilot connects Gemini checkout to Flipkart.",
            "key_facts": [
                "The test runs in India for select users.",
                "Checkout completes inside the chat window.",
            ],
            "why_now": "It is live now, not a concept demo.",
            "audience_pull": "Can I actually pay for things inside an AI chat?",
            "risks": ["Limited user base, no timeline published."],
            "sources": ["techcrunch.com"],
            "confidence": 0.68,
        },
        target_s=480,
        wpm=150,
        niche_terms=["google", "ai", "shopping"],
    )
    errors, warnings = res
    assert any("padded" in w for w in warnings)
    assert not any("padded" in e for e in errors)
