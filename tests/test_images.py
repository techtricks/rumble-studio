"""Tests for picture selection and the slideshow background.

Two bugs motivated this file, and both were invisible: every image lookup
returned None because the fetch returned bytes, and an unclosed `<input>` left
the article parser skipping the rest of the page.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ytauto.render import video as vid
from ytauto.sources import article, images


# --------------------------------------------------------------------------
# fetching pictures
# --------------------------------------------------------------------------
def _html(body: str) -> str:
    return f"<html><head><meta charset='utf-8'></head><body>{body}</body></html>"


def test_og_image_found(monkeypatch):
    monkeypatch.setattr(
        images, "fetch",
        lambda *a, **kw: _html(
            "<meta property='og:image' content='https://cdn.example/hero.jpg'>"
        ),
    )
    assert images.og_image("https://example.com/s") == "https://cdn.example/hero.jpg"


def test_og_image_when_attributes_are_reversed(monkeypatch):
    # Real pages put content before property about as often as after it.
    monkeypatch.setattr(
        images, "fetch",
        lambda *a, **kw: _html(
            "<meta content='https://cdn.example/r.jpg' property='og:image'>"
        ),
    )
    assert images.og_image("https://example.com/s") == "https://cdn.example/r.jpg"


def test_og_image_twitter_fallback(monkeypatch):
    monkeypatch.setattr(
        images, "fetch",
        lambda *a, **kw: _html(
            "<meta name='twitter:image' content='https://cdn.example/t.jpg'>"
        ),
    )
    assert images.og_image("https://example.com/s") == "https://cdn.example/t.jpg"


def test_og_image_absent(monkeypatch):
    monkeypatch.setattr(images, "fetch", lambda *a, **kw: _html("<p>no meta</p>"))
    assert images.og_image("https://example.com/s") is None


def test_og_image_resolves_a_relative_url(monkeypatch):
    # Plenty of sites emit a site-relative og:image; dropping it loses the best
    # picture the page has.
    monkeypatch.setattr(
        images, "fetch",
        lambda *a, **kw: _html("<meta property='og:image' content='/relative.jpg'>"),
    )
    assert images.og_image("https://example.com/s") == "https://example.com/relative.jpg"


def test_og_image_ignores_javascript_urls(monkeypatch):
    monkeypatch.setattr(
        images, "fetch",
        lambda *a, **kw: _html(
            "<meta property='og:image' content='javascript:void(0)'>"
        ),
    )
    assert images.og_image("https://example.com/s") is None


def test_og_image_survives_fetch_failure(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(images, "fetch", _boom)
    assert images.og_image("https://example.com/s") is None


def test_og_image_rejects_bad_url():
    assert images.og_image("not-a-url") is None
    assert images.og_image("") is None


def test_article_images_lead_ranks_first(monkeypatch):
    html = _html(
        "<meta property='og:image' content='https://cdn.example/lead.jpg'>"
        "<img src='https://cdn.example/inline1.jpg'>"
        "<img src='https://cdn.example/inline2.jpg'>"
    )
    monkeypatch.setattr(images, "fetch", lambda *a, **kw: html)
    out = images.article_images("https://example.com/s", limit=4)
    assert out[0].url == "https://cdn.example/lead.jpg"
    assert {c.url for c in out} >= {
        "https://cdn.example/lead.jpg",
        "https://cdn.example/inline1.jpg",
    }


def test_article_images_makes_relative_urls_absolute(monkeypatch):
    html = _html("<img src='/uploads/x.jpg'>")
    monkeypatch.setattr(images, "fetch", lambda *a, **kw: html)
    out = images.article_images("https://example.com/s", limit=3)
    assert any(c.url == "https://example.com/uploads/x.jpg" for c in out)


def test_article_images_handles_data_uri(monkeypatch):
    html = _html("<img src='data:image/gif;base64,R0lGOD'>")
    monkeypatch.setattr(images, "fetch", lambda *a, **kw: html)
    out = images.article_images("https://example.com/s", limit=3)
    assert not [c for c in out if c.url.startswith("data:")]


# --------------------------------------------------------------------------
# picking pictures that belong in the video
# --------------------------------------------------------------------------
def test_relevant_accepts_matching_title():
    c = images._Candidate(url="https://x/a.jpg", title="File:Googleplex HQ.jpg")
    assert images._relevant(c, ["Google"])


def test_relevant_rejects_unrelated_file():
    c = images._Candidate(url="https://x/a.jpg", title="File:RF probe with HF valve.jpg")
    assert not images._relevant(c, ["Google", "Flipkart"])


def test_relevant_rejects_short_key_as_substring():
    """'AI' inside 'Prague' is not a match; the word boundary is the whole point."""
    c = images._Candidate(url="https://x/a.jpg", title="File:Tower in Prague.jpg")
    assert not images._relevant(c, ["AI"])
    assert images._relevant(c, ["ai", "Prague"])


def test_relevant_rejects_art_project_filename():
    # A Wikimedia file named "... - Google Art Project.jpg" matched "Google" and
    # put a fifteenth-century painting on a video about a shopping app.
    c = images._Candidate(url="https://x/a.jpg", title="File:Brueghel - Google Art Project.jpg")
    assert not images._usable(c)


def test_usable_rejects_icons_and_logos():
    assert not images._usable(images._Candidate(url="https://x/a.jpg", title="Google App Engine icon.jpg"))


def test_usable_rejects_small_and_portrait():
    assert not images._usable(images._Candidate(url="https://x/a.jpg", width=320, height=240))
    # A tall portrait shot crops badly behind 16:9 captions.
    assert not images._usable(images._Candidate(url="https://x/a.jpg", width=800, height=1400))


def test_usable_accepts_landscape():
    assert images._usable(images._Candidate(url="https://x/a.jpg", width=1920, height=1080))


def test_build_queries_uses_entities_then_facets():
    qs = images.build_queries("Google Tests AI Shopping With Flipkart in India", ["retail"])
    assert qs[0] == "Google"
    # Each later query differs, so slides do not repeat one stock photo.
    assert len(set(qs)) == len(qs)
    assert all(q.strip() for q in qs)


def test_build_queries_drops_stopwords():
    qs = images.build_queries("The Future of the Chip", [])
    assert "the" not in [w.lower() for q in qs for w in q.split()]


def test_build_queries_empty_title():
    assert images.build_queries("", []) == []


def test_collect_dedupes_and_keeps_order(monkeypatch):
    monkeypatch.setattr(images, "article_images", lambda *a, **kw: [
        images._Candidate(url="https://x/a.jpg", score=100.0),
        images._Candidate(url="https://x/a.jpg?v=2", score=99.0),
    ])
    monkeypatch.setattr(images, "build_queries", lambda *a, **kw: [])
    out = images.collect(url="https://example.com/s", title="Google")
    assert len(out) == 1
    assert out[0].url == "https://x/a.jpg"


# --------------------------------------------------------------------------
# downloading
# --------------------------------------------------------------------------
def test_sniff_image_recognises_real_bytes():
    assert images._sniff_image(b"\xff\xd8\xff\xe0rest")
    assert images._sniff_image(b"\x89PNG\r\n\x1a\nrest")
    assert images._sniff_image(b"RIFF\x00\x00\x00\x00WEBPrest")
    # A 404 HTML page served with a .jpg name is not a picture.
    assert not images._sniff_image(b"<!DOCTYPE html><html>404</html>")


def test_download_rejects_html_disguised_as_image(monkeypatch, tmp_path):
    monkeypatch.setattr(images, "fetch", lambda *a, **kw: b"<!DOCTYPE html>" + b"x" * 40_000)
    assert images.download(images._Candidate(url="https://x/a.jpg"), tmp_path) is None


def test_download_rejects_tiny_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(images, "fetch", lambda *a, **kw: b"\xff\xd8\xff" + b"x" * 100)
    assert images.download(images._Candidate(url="https://x/a.jpg"), tmp_path) is None


def test_download_writes_and_caches(monkeypatch, tmp_path):
    # The contract is now "normalize through ffmpeg or drop": a fake payload
    # must be a genuinely decodable JPEG, otherwise download() drops it.
    payload = None
    try:
        import tempfile

        from ytauto.render import ffmpeg as ff

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "j.jpg"
            # payload also has to clear the 25 kB minimum image size — go big
            ff.run(["-f", "lavfi", "-i", "color=c=red:s=4096x4096:d=1",
                    "-frames:v", "1", "-q:v", "2", str(src)])
            payload = src.read_bytes()
    except Exception:
        pytest.skip("ffmpeg unavailable")

    calls = []

    def _fake(*a, **kw):
        calls.append(1)
        return payload

    monkeypatch.setattr(images, "fetch", _fake)
    cand = images._Candidate(url="https://x/a.jpg", creator="Someone", license_name="CC BY 4.0")
    a = images.download(cand, tmp_path)
    assert a and Path(a.path).exists()
    # Second call must come from disk, not the network.
    b = images.download(cand, tmp_path)
    assert b and len(calls) == 1
    # An undecodable asset is dropped outright — it must never reach the
    # renderer raw (that hang swallowed five cycles' renders).
    monkeypatch.setattr(
        images, "fetch", lambda *a, **kw: b"\xff\xd8\xff" + b"x" * 30_000)
    broken = images.download(
        images._Candidate(url="https://x/broken.jpg"), tmp_path)
    assert broken is None


def test_download_survives_fetch_error(monkeypatch, tmp_path):
    def _boom(*a, **kw):
        raise RuntimeError("down")

    monkeypatch.setattr(images, "fetch", _boom)
    assert images.download(images._Candidate(url="https://x/a.jpg"), tmp_path) is None


def test_licence_allows_commercial_use_with_derivatives():
    for ok in ("CC0", "Public domain", "CC BY 4.0", "CC BY-SA 4.0", "BY", "PDM"):
        assert images._licence_ok(ok), ok


def test_licence_blocks_nc_and_nd():
    # This channel monetises and crops every picture, so NC (no commercial use)
    # and ND (no derivatives) are both unusable. An earlier run put two BY-NC-ND
    # Flickr photos on screen.
    for bad in ("CC BY-NC 2.0", "CC BY-ND 3.0", "BY-NC-ND", "CC BY-NC-SA 4.0"):
        assert not images._licence_ok(bad), bad


def test_licence_rejects_unknown_or_empty():
    # Guessing "probably fine" is how a channel monetises a photo it has no
    # right to, so an unrecognised licence is a refusal.
    for bad in ("", "   ", "All rights reserved", "fair use"):
        assert not images._licence_ok(bad), bad


def test_openverse_asks_only_for_usable_licences(monkeypatch):
    seen = {}

    def _fake(url, **kw):
        seen["url"] = url
        return {"results": []}

    monkeypatch.setattr(images, "fetch", _fake)
    images.openverse_images("server racks")
    # all-cc would hand back NC and ND photos.
    assert "license_type=commercial,modification" in seen["url"]
    assert "all-cc" not in seen["url"]


def test_credit_string_carries_licence():
    a = images.ImageAsset(
        path=Path("a.jpg"), url="https://x/a.jpg", creator="A Person",
        license_name="CC BY 4.0", source_page="https://commons.wikimedia.org/wiki/File:A",
    )
    c = a.credit()
    assert "A Person" in c and "CC BY 4.0" in c and "wikimedia" in c


def test_credit_empty_when_no_metadata():
    assert images.ImageAsset(path=Path("a.jpg"), url="https://x/a.jpg").credit() == ""


def test_images_for_video_returns_empty_rather_than_raising(monkeypatch, tmp_path):
    def _boom(*a, **kw):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(images, "collect", _boom)
    # The renderer's gradient fallback depends on this never propagating.
    assert images.images_for_video(url="https://x", title="t", dest_dir=tmp_path) == []


# --------------------------------------------------------------------------
# article parser regression
# --------------------------------------------------------------------------
def test_void_input_does_not_swallow_the_article():
    """<input> has no closing tag, so a skip counter never unwinds."""
    out = article.extract_text(
        "<body><input name='q'><nav>Menu</nav>"
        "<p>The real story text, which must survive the parse.</p></body>"
    )
    assert "real story text" in out
    assert "Menu" not in out


def test_unclosed_chrome_tag_does_not_swallow_the_article():
    out = article.extract_text(
        "<body><div class='wrap'><nav>Menu</nav>"
        "<p>Body copy that has to survive a sloppy close tag.</p></body>"
    )
    assert "Body copy" in out


def test_script_and_style_are_dropped():
    out = article.extract_text(
        "<body><script>var x=1;</script><style>a{color:red}</style><p>Real copy.</p></body>"
    )
    assert "var x" not in out
    assert "color:red" not in out
    assert "Real copy" in out


# --------------------------------------------------------------------------
# slideshow graph
# --------------------------------------------------------------------------
def test_slideshow_single_image():
    g = vid._slideshow_filter([Path("a.jpg")], vid.style_for("longform"), 10.0)
    assert g.count(";") >= 0
    assert g.endswith("[slides]")
    assert "[v0]" in g


def test_slideshow_covers_every_image_input():
    style = vid.style_for("longform")
    g = vid._slideshow_filter([Path(f"{i}.jpg") for i in range(4)], style, 40.0)
    for i in range(4):
        assert f"[{i}:v]" in g
    # One xfade per additional image.
    assert g.count("xfade") == 3


def test_slideshow_offsets_advance():
    g = vid._slideshow_filter([Path(f"{i}.jpg") for i in range(3)], vid.style_for("longform"), 30.0)
    assert "offset=9.6" in g or "offset=9" in g  # 10s segments, 0.4s fade
    assert "offset=19" in g


def test_slideshow_crops_rather_than_stretches():
    g = vid._slideshow_filter([Path("a.jpg")], vid.style_for("longform"), 10.0)
    assert "force_original_aspect_ratio=increase" in g
    assert "crop=" in g
    assert "setdar" not in g  # would distort


def test_slideshow_never_fades_for_longer_than_the_slide():
    # A fade longer than the segment makes ffmpeg fail on a negative duration.
    g = vid._slideshow_filter([Path("a.jpg"), Path("b.jpg")], vid.style_for("short"), 1.0)
    for part in g.split(";"):
        if "xfade" in part and "duration=" in part:
            dur = float(part.split("duration=")[1].split(":")[0])
            assert dur < 0.5


def test_chain_prefixes_a_bare_input_label():
    c = vid._Chain("[0:v]").add("null", "v1")
    assert c.build() == "[0:v]null[v1]"


def test_chain_chains_onto_a_subgraph_label():
    """Chaining onto the subgraph *text* gives '[slides]vignette=..', which
    ffmpeg rejects as trailing garbage."""
    c = vid._Chain("[0:v]null[slides]", label="[slides]").add("vignette=PI/5", "v1")
    built = c.build()
    assert built == "[0:v]null[slides];[slides]vignette=PI/5[v1]"
    # Every statement must be self-contained: one input label, filters, one
    # output label. "[slides]vignette=..." with no separating ';' is what
    # ffmpeg calls trailing garbage and refuses to parse.
    import re as _re

    for part in built.split(";"):
        assert _re.fullmatch(r"\[[\w:-]+\][^;]*\[[\w-]+\]", part), part


def test_render_video_without_images_keeps_the_gradient():
    import inspect

    sig = inspect.signature(vid.render_video)
    assert "images" in sig.parameters
    assert sig.parameters["images"].default is None


def test_config_exposes_the_picture_knobs():
    from ytauto.config import Settings

    s = Settings()
    assert s.images_per_video >= 1
    assert isinstance(s.allow_article_images, bool)


def test_disallowing_article_images_skips_the_article_fetch():
    """With the outlet's copyrighted photo off, its page is never requested."""
    from ytauto.pipeline import render as pr

    class _Row(dict):
        pass

    row = _Row(topic_id="t1", title="Google Something", tags="[]", id="v1")
    calls = []

    class _DB:
        def get_topic(self, tid):
            calls.append(tid)
            return {"url": "https://example.com/story"}

    p = {"images": Path("/tmp/none"), "credits": Path("/tmp/none/credits.txt")}
    pr._image_assets(row, _DB(), "v1", p, limit=0, allow_article=False)
    assert calls == []
