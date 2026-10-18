"""Confidence measured against the source, not asked of the model.

Self-reported confidence is not calibrated across models. Told to "be
pessimistic", qwen3.8-27b returned 0.10 for a well-sourced story and the
pipeline rejected a video it should have kept, while another provider
returns 0.9 for the same input. A publish gate that moves when provider
failover changes the model is not a gate.
"""

from __future__ import annotations

from ytauto.content import grounded_confidence

SRC = (
    "Apple was hit with 5.7 billion dollars in damages on Thursday over haptic "
    "patents. The ruling came after a three-week jury trial in Texas, and the "
    "company said it would appeal."
)


def test_grounded_facts_pass_the_gate():
    facts = [
        "Apple was hit with 5.7 billion dollars in damages",
        "the ruling came after a jury trial in Texas",
        "the company said it would appeal",
    ]
    assert grounded_confidence(facts, SRC) >= 0.4


def test_hallucinated_facts_fail_the_gate():
    """A fact the model never read shares almost no content words."""
    facts = [
        "Tesla will release a robotaxi in Berlin next June",
        "Samsung shipped 40 million foldable handsets this quarter",
    ]
    assert grounded_confidence(facts, SRC) < 0.4


def test_confidence_tracks_grounding_not_vibes():
    """A wholly invented fact scores below one the model actually read."""
    real = ["the ruling came after a jury trial in Texas"]
    made_up = ["Zuckerberg acquired the company for nine billion dollars"]
    assert grounded_confidence([real], SRC) > grounded_confidence([made_up], SRC)


def test_a_single_swapped_detail_does_move_the_score():
    """Texas -> Reykjavik inside an otherwise-quoted sentence."""
    real = grounded_confidence(["the ruling came after a jury trial in Texas"], SRC)
    swapped = grounded_confidence(["the ruling came after a jury trial in Reykjavik"], SRC)
    assert real > swapped


def test_known_weak_spot_paraphrase_only_just_clears_the_gate():
    """Content-word overlap rewards real words over invented ones, but it
    cannot tell a restatement from a distortion.

    "The company intends to contest the ruling" is not what the source says
    ("the company said it would appeal"), yet it still clears 0.4 because
    company and ruling are genuinely present. This method catches wholesale
    fabrication and thin sourcing; it is not a fact-checker. A second source
    or the risks field is what catches a subtle misattribution.
    """
    score = grounded_confidence(["The company intends to contest the ruling"], SRC)
    assert score < 0.5
