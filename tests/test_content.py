"""Content generation, offline fallback, and the publish gates.

The offline path is the one that runs on a machine with no ANTHROPIC_API_KEY,
so it is exercised heavily: it is the difference between "the tool explains
itself" and "the tool produces nothing and you cannot tell why".
"""

from __future__ import annotations

import pytest

from ytauto import validate as V
from ytauto.content import (
    research_topic,
    write_metadata,
    write_script,
)
from ytauto.llm import Research


TOPIC = {
    "id": "t1",
    "title": "Nvidia ships a GPU four times faster than the last one",
    "url": "https://example.com/story",
    "source": "rss",
    "summary": (
        "Nvidia announced a new datacenter GPU on Tuesday. The company says it "
        "runs inference four times faster than the previous part. Analysts "
        "argue the real constraint is power draw, not compute. Pricing was not "
        "disclosed, and shipments start next quarter."
    ),
}


# ------------------------------------------------------------ offline path
def test_offline_research_marks_itself_degraded(settings):
    r = research_topic(TOPIC, settings=settings)
    assert r.degraded is True
    assert r.confidence < 0.4
    assert r.key_facts, "should salvage the source sentences"
    assert r.angle
    assert any("without an LLM" in risk for risk in r.risks)


def test_offline_research_never_invents_sources(settings):
    r = research_topic(TOPIC, settings=settings)
    # Only the topic's own url is allowed; no invented citations.
    assert all(s == TOPIC["url"] for s in r.sources)


def test_offline_research_of_an_empty_topic_is_thin(settings):
    r = research_topic({"title": "", "summary": ""}, settings=settings)
    assert r.is_thin


def test_offline_scripts_have_a_hook_and_a_body(settings):
    r = research_topic(TOPIC, settings=settings)
    s = write_script("longform", TOPIC, r, settings=settings)
    assert s.hook
    assert s.word_count > 40
    assert s.text.startswith(s.hook[:20])


def test_offline_short_is_under_three_minutes(settings):
    r = research_topic(TOPIC, settings=settings)
    s = write_script("shorts", TOPIC, r, settings=settings)
    est = s.word_count / max(1, settings.words_per_minute) * 60
    assert est <= 180, f"Shorts cap is 180s, estimated {est:.0f}s"


def test_offline_metadata_passes_its_own_gates(settings):
    """The offline template must not ship something its own validator rejects."""
    r = research_topic(TOPIC, settings=settings)
    sc = write_script("longform", TOPIC, r, settings=settings)
    m = write_metadata("longform", TOPIC, r, sc, settings=settings)

    assert V.check_title(m.title) == []
    assert V.check_title_b(m.title_b, m.title) == []
    assert V.check_description(m.description) == []
    assert V.check_tags(m.tags) == []
    assert V.check_chapters(m.chapters, "longform") == []


def test_offline_metadata_for_shorts(settings):
    r = research_topic(TOPIC, settings=settings)
    sc = write_script("shorts", TOPIC, r, settings=settings)
    m = write_metadata("shorts", TOPIC, r, sc, settings=settings)
    assert V.check_title(m.title, max_len=55) == []
    assert V.check_chapters(m.chapters, "shorts") == []


# -------------------------------------------------------------- title rules
def test_title_length_is_enforced():
    assert V.check_title("x" * 100)
    assert V.check_title("") == ["title is empty"]


def test_title_b_must_differ_from_title():
    t = "Nvidia ships a faster GPU"
    assert V.check_title_b(t, t)                       # identical
    assert V.check_title_b("", t)                      # missing
    assert V.check_title_b("Power draw is the real GPU limit", t) == []


def test_title_b_catches_a_near_identical_variant():
    a = "Nvidia ships a much faster new GPU"
    assert V.check_title_b(a, a)
    assert V.check_title_b("Nvidia ships a much faster new GPU!", a)


def test_title_b_threshold_documented_limitation():
    """Known weakness, pinned deliberately: Jaccard only catches near-identical
    variants. A light synonym swap ("ships"->"releases") scores 0.56 and passes.
    Tightening this needs a synonym model, so for now the A/B result should be
    eyeballed once, not trusted blindly."""
    a = "Nvidia ships a much faster new GPU"
    b = "Nvidia releases a way faster new GPU"
    assert V.check_title_b(b, a) == []


def test_description_needs_enough_substance():
    assert V.check_description("too short")
    good = "A" * 100 + " https://example.com/source"
    assert V.check_description(good) == []


def test_tag_rules():
    assert V.check_tags([])
    assert V.check_tags(["ai"] * 40)                    # too many
    assert V.check_tags(["a" * 80])                     # too long
    assert V.check_tags(["AI", "gpu", "nvidia"]) == []


def test_script_rules():
    assert V.check_script("word ", target_s=60, wpm=150)      # too short
    long_enough = " ".join(["word"] * 200)                    # ~80s at 150wpm
    assert V.check_script(long_enough, target_s=60, wpm=150)  # too long
    assert V.check_script(long_enough, target_s=90, wpm=150) == []


def test_research_rules():
    assert V.check_research({"angle": "", "key_facts": ["a", "b"], "confidence": 0.9})
    assert V.check_research({"angle": "x", "key_facts": ["a"], "confidence": 0.9})
    assert V.check_research({"angle": "x", "key_facts": ["a", "b"], "confidence": 0.1})
    assert V.check_research({"angle": "x", "key_facts": ["a", "b"], "confidence": 0.9}) == []


def test_research_confidence_handles_garbage():
    problems = V.check_research({"angle": "x", "key_facts": ["a", "b"],
                                 "confidence": "not a number"})
    assert any("confidence" in p for p in problems)


def test_niche_terms_are_required_in_the_metadata():
    assert V.validate_package(
        kind="longform", title="A perfectly reasonable title about chips",
        title_b="Something else entirely about power draw",
        description="d" * 120, tags=["ai", "gpu"], script=" ".join(["w"] * 200),
        thumbnail_plan={"text": "Big claim", "emotion": "shocked",
                        "subject": "a GPU on a dark desk"},
        research={"angle": "a", "key_facts": ["f1", "f2"], "confidence": 0.8},
        chapters=[{"title": "Intro", "time_s": 0}],
        niche_terms=["ai", "gpu"], target_s=60, wpm=150, max_title_len=70,
    )[0] == []


# ------------------------------------------------------ full validation gate
def _pkg(**over):
    base = dict(
        kind="longform",
        title="Nvidia's New GPU Runs AI Four Times Faster",
        title_b="Power draw, not compute, is the real bottleneck",
        description="A" * 100 + " https://example.com/src",
        tags=["ai", "gpu", "nvidia"],
        script=" ".join(["word"] * 220),
        thumbnail_plan={"text": "4x faster", "emotion": "shocked",
                        "subject": "a GPU on a dark desk", "palette": "dark"},
        research={"angle": "power is the limit", "key_facts": ["f1", "f2"],
                  "confidence": 0.8},
        chapters=[{"title": "Intro", "time_s": 0}, {"title": "The catch", "time_s": 30}],
        niche_terms=["ai", "gpu", "nvidia"],
        target_s=60,
        wpm=150,
        max_title_len=70,
    )
    base.update(over)
    return V.validate_package(**base)


def test_a_good_package_passes():
    errors, warnings = _pkg()
    assert errors == [], errors


def test_a_package_with_a_bad_title_fails():
    errors, _ = _pkg(title="x" * 200)
    assert any("title" in e for e in errors)


def test_a_package_with_a_duplicate_title_b_fails():
    errors, _ = _pkg(title_b="Nvidia's New GPU Runs AI Four Times Faster")
    assert any("title_b" in e for e in errors)


def test_a_package_with_no_source_fails():
    errors, _ = _pkg(research={"angle": "a", "key_facts": ["f1", "f2"],
                               "confidence": 0.8}, source_url="")
    # A missing source is a warning, not a hard stop, so it must not error out.
    assert not any("source" in e for e in errors)


def test_off_topic_package_fails_the_niche_check():
    errors, _ = _pkg(
        title="Medieval pottery techniques explained",
        title_b="How to throw a bowl on a wheel",
        description="A" * 100,
        tags=["pottery", "history"],
        niche_terms=["ai", "gpu", "nvidia"],
    )
    assert errors


def test_shorts_must_not_carry_chapters():
    errors, _ = _pkg(kind="shorts",
                     chapters=[{"title": "Intro", "time_s": 0}])
    assert any("chapter" in e for e in errors)


def test_warnings_do_not_block():
    errors, warnings = _pkg(description="#a #b #c " + "A" * 100)
    assert errors == []


# ------------------------------------------------- length vs render target
def test_a_short_script_is_rejected_for_a_longform_target():
    """The bug this pins: 184 words (~74s) passed a 480s longform target.

    check_script only ever complained about being too long, so a model that
    ignored the word-count instruction produced a video that was not the format
    it had been scheduled as, and every gate agreed it was fine.
    """
    from ytauto.validate import check_script

    short = "word " * 184          # ~74s at 150wpm
    problems = check_script(short, target_s=480, wpm=150, kind="longform")
    assert problems, "an undersized longform script must be flagged"
    assert "too short" in problems[0]
    assert "longform" in problems[0]


def test_a_correct_length_script_passes():
    from ytauto.validate import check_script

    good = "word " * 1200          # exactly 480s at 150wpm
    assert check_script(good, target_s=480, wpm=150) == []


def test_an_over_length_script_is_still_flagged():
    from ytauto.validate import check_script

    assert check_script("word " * 2000, target_s=480, wpm=150)


def test_a_slightly_short_script_is_tolerated():
    """Inside the tolerance band a shorter script is fine, not a defect."""
    from ytauto.validate import check_script

    assert check_script("word " * 1000, target_s=480, wpm=150) == []


def test_a_tiny_script_still_fails_immediately():
    from ytauto.validate import check_script

    assert "far too short" in check_script("hi there", target_s=480, wpm=150)[0]


# ------------------------------------------------------- longform beat growth
class _BeatLLM:
    """Stands in for the LLM: a stub script, then N sized beats."""

    def __init__(self, beat_words=140, beat_text=None):
        self.available = True
        self.beat_words = beat_words
        # None means "honour the size the prompt asked for", like a real model.
        self.beat_text = beat_text
        self.beat_calls = 0
        self.script_calls = 0

    def complete(self, prompt, schema, system="", model=None, max_tokens=None):
        if "body beat" in prompt:
            self.beat_calls += 1
            return {"beat": self.beat_text or ("detail " * self._asked(prompt))}
        self.script_calls += 1
        return {"hook": "A cold open claim.", "body": "short body.", "outro": "Close."}

    @staticmethod
    def _asked(prompt):
        """Honour the size the prompt asked for, like a real model does."""
        import re

        m = re.search(r"Write (\d+) words", prompt)
        return int(m.group(1)) if m else 140


def test_a_short_longform_script_grows_to_its_target_length():
    from ytauto.config import Settings
    from ytauto.content import write_script
    from ytauto.llm import Research
    from ytauto.llm import LLM

    s = Settings(longform_target_s=480, words_per_minute=150)
    fake = _BeatLLM()
    out = write_script(
        "longform",
        {"title": "Nvidia ships a faster GPU", "url": "https://x.test/a"},
        Research(
            angle="why it matters",
            summary="s",
            key_facts=["one fact", "two fact", "three fact"],
            why_now="now",
            audience_pull="pull",
            risks=[],
            confidence=0.8,
        ),
        settings=s,
        llm=fake,
    )
    target = int(480 / 60.0 * 150)
    assert out.word_count >= target, f"{out.word_count} < {target}"
    assert fake.beat_calls >= 1


def test_beat_growth_stops_when_research_runs_out():
    """A stub beat means the model had nothing left to add.

    Returning a short script here is deliberate: validation must reject it
    rather than the pipeline padding to hit a runtime it cannot justify.
    """
    from ytauto.config import Settings
    from ytauto.content import write_script
    from ytauto.llm import Research

    s = Settings(longform_target_s=480, words_per_minute=150, longform_max_beats=4)
    fake = _BeatLLM(beat_text="too short")
    out = write_script(
        "longform",
        {"title": "T", "url": "https://x.test/a"},
        Research(
            angle="a", summary="s", key_facts=["f"], why_now="n",
            audience_pull="p", risks=[], confidence=0.5,
        ),
        settings=s,
        llm=fake,
    )
    assert fake.beat_calls == 1, "must stop after the first stub, not keep asking"
    assert out.word_count < 480 / 60.0 * 150


def test_a_script_already_at_target_gets_no_extra_calls():
    from ytauto.config import Settings
    from ytauto.content import write_script
    from ytauto.llm import Research

    s = Settings(longform_target_s=60, words_per_minute=150)

    class _Long(_BeatLLM):
        def complete(self, prompt, schema, system="", model=None, max_tokens=None):
            self.script_calls += 1
            return {
                "hook": "Hook.",
                "body": "word " * 400,
                "outro": "Out.",
            }

    fake = _Long()
    write_script(
        "longform",
        {"title": "T", "url": "https://x.test/a"},
        Research(
            angle="a", summary="s", key_facts=["f"], why_now="n",
            audience_pull="p", risks=[], confidence=0.5,
        ),
        settings=s,
        llm=fake,
    )
    assert fake.beat_calls == 0


def test_shorts_are_never_grown():
    from ytauto.config import Settings
    from ytauto.content import write_script
    from ytauto.llm import Research

    s = Settings(shorts_target_s=45, words_per_minute=150, longform_target_s=480)
    fake = _BeatLLM()
    write_script(
        "shorts",
        {"title": "T", "url": "https://x.test/a"},
        Research(
            angle="a", summary="s", key_facts=["f"], why_now="n",
            audience_pull="p", risks=[], confidence=0.5,
        ),
        settings=s,
        llm=fake,
    )
    assert fake.beat_calls == 0


def test_a_failed_beat_does_not_discard_the_whole_script():
    from ytauto.config import Settings
    from ytauto.content import write_script
    from ytauto.llm import LLMError, Research

    class _Flaky(_BeatLLM):
        def complete(self, prompt, schema, system="", model=None, max_tokens=None):
            if "body beat" in prompt:
                raise LLMError("503 overloaded")
            return super().complete(prompt, schema, system, model, max_tokens)

    s = Settings(longform_target_s=480, words_per_minute=150)
    out = write_script(
        "longform",
        {"title": "T", "url": "https://x.test/a"},
        Research(
            angle="a", summary="s", key_facts=["f"], why_now="n",
            audience_pull="p", risks=[], confidence=0.5,
        ),
        settings=s,
        llm=_Flaky(),
    )
    assert out.word_count > 0


def test_beat_growth_is_bounded_by_the_beat_cap():
    """A model that never reaches the requested size must not loop forever.

    The cap is a safety valve; the result stays short so validation can reject
    the topic rather than the pipeline spinning on it.
    """
    from ytauto.config import Settings
    from ytauto.content import write_script
    from ytauto.llm import Research

    class _Stubborn(_BeatLLM):
        """Always answers a third of what was asked."""

        @staticmethod
        def _asked(prompt):
            import re

            m = re.search(r"Write (\d+) words", prompt)
            return max(40, int(m.group(1)) // 3) if m else 40

    s = Settings(longform_target_s=600, words_per_minute=150, longform_max_beats=5)
    fake = _Stubborn()
    write_script(
        "longform",
        {"title": "T", "url": "https://x.test/a"},
        Research(
            angle="a", summary="s", key_facts=["f"], why_now="n",
            audience_pull="p", risks=[], confidence=0.5,
        ),
        settings=s,
        llm=fake,
    )
    assert fake.beat_calls == 5


# ------------------------------------------------- one-call script + metadata
class _CountingLLM:
    """Records every call so the test can assert the request budget."""

    def __init__(self):
        self.available = True
        self.calls = []

    def complete(self, prompt, schema, system="", model=None, max_tokens=None):
        self.calls.append("beat" if "body beat" in prompt else
                          "script" if "body:" in prompt or "Format exactly" in prompt
                          else "other")
        if "body beat" in prompt:
            return {"beat": "detail " * 200}
        return {
            "hook": "A cold claim.", "body": "short body.", "outro": "Close.",
            "title": "Nvidia ships a faster GPU", "title_b": "A quicker GPU is out",
            "description": "Two paragraphs.", "tags": ["nvidia", "gpu"],
            "thumbnail_plan": {
                "text": "FASTER GPU", "emotion": "surprise",
                "subject": "a chip", "palette": "dark blue",
            },
            "chapters": [],
        }


def test_metadata_costs_no_extra_request():
    """20 generateContent calls a day is the whole budget.

    Metadata used to be its own request, spending a fifth of the day's quota on
    packaging for a script the model had just written and still remembered.
    """
    from ytauto.config import Settings
    from ytauto.content import write_metadata, write_script
    from ytauto.llm import Research

    s = Settings(longform_target_s=480, words_per_minute=150)
    fake = _CountingLLM()
    r = Research(
        angle="a", summary="s", key_facts=["f"], why_now="n",
        audience_pull="p", risks=[], confidence=0.7,
    )
    fields = {}
    write_script("longform", {"title": "T", "url": "https://x.test/a"}, r,
                 settings=s, llm=fake, meta_out=fields)
    assert "title" in fields, "script call should have returned the metadata too"

    meta = write_metadata("longform", {"title": "T", "url": "https://x.test/a"}, r,
                          settings=s, llm=fake, prefetched=fields)
    assert meta.title == "Nvidia ships a faster GPU"
    assert meta.title_b == "A quicker GPU is out"
    assert meta.tags == ["nvidia", "gpu"]
    assert meta.thumbnail_plan["text"] == "FASTER GPU"
    assert fake.calls.count("script") == 1, fake.calls


def test_partial_prefetched_metadata_falls_back_to_a_dedicated_call():
    """A truncated response must not produce a video with no title."""
    from ytauto.config import Settings
    from ytauto.content import write_metadata
    from ytauto.llm import Research

    class _OnlyTags(_CountingLLM):
        def complete(self, prompt, schema, system="", model=None, max_tokens=None):
            self.calls.append("meta")
            return {
                "title": "Real title", "title_b": "Alt", "description": "d",
                "tags": ["a"], "thumbnail_plan": {
                    "text": "T", "emotion": "e", "subject": "s", "palette": "p",
                },
            }

    s = Settings()
    r = Research(angle="a", summary="s", key_facts=["f"], why_now="n",
                 audience_pull="p", risks=[], confidence=0.7)
    fake = _OnlyTags()
    meta = write_metadata("longform", {"title": "T", "url": "u"}, r,
                          settings=s, llm=fake, prefetched={"tags": ["x"]})
    assert meta.title == "Real title"
    assert fake.calls == ["meta"]


def test_an_empty_prefetched_dict_costs_nothing_extra_when_llm_is_absent():
    from ytauto.config import Settings
    from ytauto.content import write_metadata
    from ytauto.llm import Research

    class _NoLLM:
        available = False

    s = Settings()
    r = Research(angle="a", summary="s", key_facts=["f"], why_now="n",
                 audience_pull="p", risks=[], confidence=0.7)
    out = write_metadata("longform", {"title": "Some Topic", "url": "u"}, r,
                         settings=s, llm=_NoLLM())
    assert out.title


# --------------------------------------------------------------------------
# the on-topic gate, and the phrase matching it depends on
# --------------------------------------------------------------------------
def test_tech_story_recognised_without_saying_the_word_tech():
    """The gate used to reject real technology for lacking a brand term.

    A robotaxi fleet is a technology story whether or not the package says
    "tech". The vocabulary is broad enough to recognise the subject; a subject
    that is not technology is still refused.
    """
    from ytauto.config import get_settings
    from ytauto.validate import check_niche

    terms = get_settings().niche_terms
    assert not check_niche("80% of the fleet is in two states", "Waymo expansion", ["self-driving"], terms)
    assert not check_niche("The new chip runs AI four times faster", "Nvidia GPU", ["gpu"], terms)
    assert not check_niche("A solid-state cell breakthrough", "Battery", ["battery"], terms)


def test_non_tech_story_is_still_refused():
    from ytauto.config import get_settings
    from ytauto.validate import check_niche

    terms = get_settings().niche_terms
    assert check_niche("How medieval potters made plates", "A history of pottery", ["history"], terms)
    assert check_niche("A recipe for sourdough", "Baking at home", ["cooking"], terms)


def test_multi_word_niche_terms_match_as_phrases():
    """A phrase term used to be unmatchable, because the words lived in a set.

    Joining a set does not keep word order, so "electric vehicle" could not be
    found in text that said exactly that, and in the rare case the set happened
    to hash in order it would match the two words the other way round.
    """
    from ytauto.validate import check_niche

    assert not check_niche("The car parks itself", "An electric vehicle story", ["ev"], ["electric vehicle"])
    # Order still matters: reversed text is not the phrase.
    assert check_niche("The car parks itself", "A vehicle that is electric", ["ev"], ["electric vehicle"])


def test_single_word_terms_do_not_match_inside_other_words():
    """'ai' inside 'available' is the failure this check was written to avoid."""
    from ytauto.validate import check_niche

    assert check_niche("New details available", "Email settings explained", ["email"], ["ai"])
    assert not check_niche("New AI details", "Artificial intelligence", ["ai"], ["ai"])


def test_niche_vocabulary_keeps_its_four_tag_seeds_first():
    """The first four terms become tags, so the broadest go first."""
    from ytauto.config import get_settings

    terms = get_settings().niche_terms
    assert terms[:4] == ["tech", "technology", "ai", "gadgets"]
    assert len(terms) > 20, "the long tail is what recognises a technology subject"
    assert len(set(terms)) == len(terms), "duplicate terms would waste tag slots"
