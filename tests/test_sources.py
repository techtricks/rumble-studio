"""Source adapters, exercised against saved fixtures — never the live network.

The adapters fetch through `ytauto.sources.base.fetch`, which uses `requests`,
so that is the client every test here stubs. The autouse `_no_network` fixture
turns a missed stub into an immediate failure rather than a live request.
"""

from __future__ import annotations

import pytest
import requests

from ytauto.sources import reddit, rss
from ytauto.sources.base import SourceError

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Example Feed</title>
  <item>
    <title>Nvidia ships a faster AI GPU</title>
    <link>https://example.com/a</link>
    <description>&lt;p&gt;A new datacenter GPU runs inference 4x faster.&lt;/p&gt;</description>
    <pubDate>Sat, 25 Sep 2026 09:00:00 +0000</pubDate>
  </item>
  <item>
    <title>Second story about gadgets</title>
    <link>https://example.com/b</link>
    <description>Some other news entirely.</description>
  </item>
</channel></rss>
"""


class FakeResponse:
    def __init__(self, body, status=200):
        self._body = body.encode() if isinstance(body, str) else body
        self.status_code = status
        self.encoding = "utf-8"
        self.headers = {}
        self.text = self._body.decode("utf-8", "replace")
        self.content = self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self):
        import json

        return json.loads(self.text)


def _stub_get(monkeypatch, body=FEED, status=200):
    """Return a recorder of requested URLs plus the stub install."""
    calls = []

    def _get(url, headers=None, timeout=None, **k):
        calls.append(url)
        return FakeResponse(body, status)

    monkeypatch.setattr(requests, "get", _get)
    return calls


@pytest.fixture(autouse=True)
def _no_throttle(monkeypatch):
    """Reddit sleeps 15s between calls to stay under its rate limit. That is
    correct in production and unacceptable in a test loop."""
    monkeypatch.setattr(reddit.time, "sleep", lambda s: None)


# --------------------------------------------------------------------- rss
def test_rss_parses_items(monkeypatch):
    _stub_get(monkeypatch)
    out = rss.RssSource(["https://example.com/feed"]).harvest(limit=10)
    assert len(out) == 2
    assert out[0].title == "Nvidia ships a faster AI GPU"
    assert out[0].url == "https://example.com/a"
    assert out[0].source == "rss"


def test_rss_sends_a_real_user_agent(monkeypatch):
    seen = {}

    def _get(url, headers=None, timeout=None, **k):
        seen.update(headers or {})
        return FakeResponse(FEED)

    monkeypatch.setattr(requests, "get", _get)
    rss.RssSource(["https://example.com/feed"]).harvest(limit=5)
    # Feeds served to an empty/absent UA are throttled or blocked outright.
    assert seen.get("User-Agent")


def test_rss_strips_html_from_summaries(monkeypatch):
    _stub_get(monkeypatch)
    summary = rss.RssSource(["https://example.com/feed"]).harvest(10)[0].summary
    assert "<p>" not in summary
    assert "4x faster" in summary


def test_rss_normalises_rfc2822_pubdate_to_iso(monkeypatch):
    """The feed hands us "Sat, 25 Sep 2026 09:00:00 +0000". If that is stored
    raw, parse_dt cannot read it later and scoring falls back to created_at,
    making every story look brand new."""
    _stub_get(monkeypatch)
    got = rss.RssSource(["https://example.com/feed"]).harvest(10)[0].published_at
    assert got and got.startswith("2026-09-25"), got
    from ytauto.util import parse_dt

    assert parse_dt(got) is not None


def test_rss_limit_caps_the_total(monkeypatch):
    _stub_get(monkeypatch)
    assert len(rss.RssSource(["https://example.com/feed"]).harvest(1)) == 1
    assert len(rss.RssSource(["https://example.com/feed"]).harvest(5)) == 2


def test_rss_skips_items_without_a_title(monkeypatch):
    _stub_get(monkeypatch, "<rss><channel><item><link>https://e.com/1</link></item></channel></rss>")
    assert rss.RssSource(["https://example.com/feed"]).harvest(5) == []


@pytest.mark.parametrize("status", [404, 429, 500, 503])
def test_fetch_raises_a_source_error(monkeypatch, status):
    """`fetch` is the strict layer: it raises rather than guessing."""
    from ytauto.sources.base import fetch

    _stub_get(monkeypatch, "nope", status=status)
    with pytest.raises(SourceError):
        fetch("https://example.com/feed", attempts=1)


@pytest.mark.parametrize("status", [404, 429, 500])
def test_rss_harvest_survives_a_dead_feed(monkeypatch, status):
    """...but a harvest is not: one dead feed must not lose the others."""
    _stub_get(monkeypatch, "nope", status=status)
    assert rss.RssSource(["https://example.com/feed"]).harvest(5) == []


def test_one_dead_feed_does_not_hide_a_healthy_one(monkeypatch):
    """Keyed on the URL, not a call counter: `fetch` retries a 5xx, so a
    counter would make the dead feed's retries succeed and hide the bug."""
    def _get(url, headers=None, timeout=None, **k):
        if "dead" in url:
            return FakeResponse("gone", 404)
        return FakeResponse(FEED)

    monkeypatch.setattr(requests, "get", _get)
    out = rss.RssSource(
        ["https://dead.example/feed", "https://live.example/feed"]
    ).harvest(10)
    assert len(out) == 2
    assert all(t.url for t in out)


# ------------------------------------------------------------------ reddit
def test_reddit_uses_the_official_rss_endpoint():
    """Reddit's JSON API blocks datacenter IPs, so the adapter must use the
    public .rss feed. Asserting the URL is the entire point of this test."""
    assert reddit.FEED == "https://www.reddit.com/r/{sub}/.rss"
    assert ".json" not in reddit.FEED


def test_reddit_parses_a_feed(monkeypatch):
    _stub_get(monkeypatch)
    out = reddit.RedditSource(["technology"]).harvest(limit=5)
    assert len(out) == 2
    assert all(t.source == "reddit" for t in out)
    assert all(t.raw.get("subreddit") == "technology" for t in out)


def test_reddit_requests_the_feed_url(monkeypatch):
    calls = _stub_get(monkeypatch)
    reddit.RedditSource(["technology"]).harvest(limit=5)
    assert calls == ["https://www.reddit.com/r/technology/.rss"]


def test_reddit_429_is_tolerated(monkeypatch):
    _stub_get(monkeypatch, "slow down", status=429)
    # Rate limiting is expected, not fatal: a blocked sub yields nothing and the
    # other subreddits still get harvested.
    assert reddit.RedditSource(["technology"]).harvest(limit=5) == []


def test_reddit_throttles_between_calls(monkeypatch):
    """Two calls inside the window must wait, or the IP gets banned."""
    slept = []
    monkeypatch.setattr(reddit.time, "sleep", lambda s: slept.append(s))
    src = reddit.RedditSource(["technology"])
    src.harvest(limit=2)
    src.harvest(limit=2)
    assert slept, "a second call inside the window should sleep"
    assert 0 < slept[0] <= reddit.MIN_GAP_SECONDS


# --------------------------------------------------------- reddit self posts
def test_reddit_ama_self_posts_are_rejected():
    """Real r/technology feed shapes, not invented ones.

    A text post's body is `<div class="md">`; a link post's body is a `<table>`
    with a thumbnail. The structural marker is the reliable signal, the title
    patterns are the backstop.
    """
    from ytauto.sources.reddit import _is_self_post

    ama_body = '<!-- SC_OFF --><div class="md"><p>To make its workforce AI native...</div>'
    assert _is_self_post("Hi Reddit, I'm Katie Paul, a Reuters journalist", ama_body)

    # Title backstop when the body is empty.
    assert _is_self_post("AMA: our CTO answers questions", "")
    assert _is_self_post("How do I get into ML?", "")
    assert _is_self_post("Hi Reddit, I'm a data engineer", "")


def test_reddit_link_posts_are_kept():
    from ytauto.sources.reddit import _is_self_post

    link_body = (
        '<table> <tr><td> <a href="https://www.reddit.com/r/technology/comments/1wrc8q4/x/">'
        '<img alt="Nvidia trailers"></a></td></tr></table>'
    )
    assert not _is_self_post("Thieves Stole Nvidia Trailers", link_body)
    assert not _is_self_post("Nvidia ships a GPU four times faster", "")


def test_a_link_post_with_no_article_url_is_still_kept():
    """Reddit's .rss no longer publishes the article link.

    Verified against the live feed: every href in a link post's body points back
    at reddit.com. Rejecting on a missing outbound URL therefore discarded every
    link post and silently emptied the subreddit — a headline with no source is
    weaker material, not unusable material.
    """
    from ytauto.sources.reddit import _is_self_post

    assert not _is_self_post("Thieves Stole Nvidia Trailers", "<table></table>")


def test_a_first_person_project_post_is_kept():
    """Title patterns are a backstop only.

    "I built X" is often the actual news.
    """
    from ytauto.sources.reddit import _is_self_post

    assert not _is_self_post("I built a 3D printer from a $40 stepper motor", "<table></table>")


def test_outbound_link_is_found_in_the_alternate_link():
    """Reddit does not always put the article in <link>.

    Checking `link` alone rejected real stories whose entry only exposed the
    outbound URL as an alternate link, which is how an earlier version lost a
    whole subreddit.
    """
    from ytauto.sources.reddit import _outbound_link

    entry = {
        "link": "https://www.reddit.com/r/technology/comments/abc/title/",
        "links": [
            {"href": "https://www.reddit.com/r/technology/comments/abc/title/"},
            {"href": "https://www.theverge.com/2026/01/nvidia-gpu"},
        ],
    }
    assert _outbound_link(entry, entry["link"]) == "https://www.theverge.com/2026/01/nvidia-gpu"


def test_outbound_link_is_found_in_the_summary_html():
    from ytauto.sources.reddit import _outbound_link

    entry = {
        "link": "https://www.reddit.com/r/gadgets/comments/x/post/",
        "summary": '<table><tr><td><a href="https://arstechnica.com/gadgets/2026/01/apple/">[link]</a></td></tr></table>',
    }
    assert _outbound_link(entry, entry["link"]) == "https://arstechnica.com/gadgets/2026/01/apple/"


def test_a_link_post_whose_article_is_only_in_the_summary_is_kept():
    from ytauto.sources.reddit import RedditSource

    src = RedditSource(["gadgets"], limit_each=5)
    entry = {
        "title": "Apple hit with $5.7 billion in damages over haptic patents",
        "link": "https://www.reddit.com/r/gadgets/comments/x/apple/",
        "summary": '<a href="https://arstechnica.com/gadgets/2026/01/apple/">[link]</a>',
        "updated": "2026-09-27T00:00:00+00:00",
    }
    topic = src._to_topic(entry, "gadgets", 0)
    assert topic is not None
    # The stored URL must be the article, not the reddit permalink.
    assert topic.url == "https://arstechnica.com/gadgets/2026/01/apple/"


def test_reddit_link_posts_are_kept():
    from ytauto.sources.reddit import _is_self_post

    assert not _is_self_post(
        "Nvidia ships a GPU four times faster",
        "https://www.theverge.com/2026/01/nvidia-gpu",
    )
    assert not _is_self_post(
        "Apple hit with $5.7 billion in damages",
        "https://arstechnica.com/gadgets/2026/01/apple/",
    )
    # A real story posted to reddit, linked outward, is still a story.
    assert not _is_self_post(
        "Insurers claim AI is increasing healthcare costs",
        "https://reuters.com/technology/ai-health-costs/",
    )


def test_self_post_filter_is_applied_when_building_a_topic():
    from ytauto.sources.reddit import RedditSource

    src = RedditSource(["technology"], limit_each=5)
    entry = {
        "title": "Hi Reddit, I'm a Reuters journalist who...",
        "link": "https://www.reddit.com/r/technology/comments/abc/hi_reddit/",
        "summary": "self post body",
        "updated": "2026-09-27T00:00:00+00:00",
    }
    assert src._to_topic(entry, "technology", 0) is None

    story = {
        "title": "Nvidia ships a faster GPU",
        "link": "https://www.theverge.com/2026/01/nvidia",
        "summary": "Nvidia announced a new datacenter GPU.",
        "updated": "2026-09-27T00:00:00+00:00",
    }
    topic = src._to_topic(story, "technology", 0)
    assert topic is not None
    assert topic.url == "https://www.theverge.com/2026/01/nvidia"
