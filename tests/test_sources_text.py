"""Regression tests for turning a fetched response into text.

The article extractor asked `isinstance(body, str)` on the result of a fetch
that returns bytes, so it returned None on every single request. The pipeline
still produced scripts, because it silently fell back to the feed teaser, and
the enrichment looked like it was working. These tests pin the decode.
"""

from __future__ import annotations

import pytest

from ytauto.sources import base


class _Resp:
    def __init__(self, encoding=None, content=b"", payload=None):
        self.encoding = encoding
        self.content = content
        self._payload = payload

    def json(self):
        return self._payload


def test_decode_body_prefers_utf8():
    data = "caf\u00e9 \u2014 na\u00efve".encode("utf-8")
    assert base.decode_body(data) == "caf\u00e9 \u2014 na\u00efve"


def test_decode_body_uses_declared_encoding():
    assert base.decode_body("caf\u00e9".encode("latin-1"), _Resp(encoding="latin-1")) == "caf\u00e9"


def test_decode_body_ignores_requests_placeholder_iso():
    # requests claims ISO-8859-1 for text/* with no charset, which mangles UTF-8.
    # The meta tag is the honest signal and must win.
    html = '<html><head><meta charset="utf-8"></head><body>na\u00efve</body></html>'
    out = base.decode_body(html.encode("utf-8"), _Resp(encoding="ISO-8859-1"))
    assert "na\u00efve" in out


def test_decode_body_reads_charset_from_meta_when_header_absent():
    html = '<html><head><meta charset="utf-8"></head><body>na\u00efve</body></html>'
    out = base.decode_body(html.encode("utf-8"), _Resp(encoding=None))
    assert "na\u00efve" in out


def test_decode_body_never_raises_on_binary():
    assert isinstance(base.decode_body(bytes(range(256))), str)


def _resp(content: bytes, encoding=None, ctype="text/html", payload=None):
    class _R:
        def __init__(self):
            self.status_code = 200
            self.content = content
            self.encoding = encoding
            self.headers = {"content-type": ctype}
            self.text = content.decode("utf-8", "replace")

        def json(self):
            return payload

        def raise_for_status(self):
            return None

    return _R()


def test_fetch_as_text_returns_str(monkeypatch):
    import requests

    monkeypatch.setattr(
        requests, "get",
        lambda *a, **kw: _resp("<p>hi \u00e9</p>".encode("utf-8"), encoding="utf-8"),
    )
    out = base.fetch("https://example.com/a", as_text=True)
    assert isinstance(out, str)
    assert "\u00e9" in out


def test_fetch_default_still_returns_bytes(monkeypatch):
    import requests

    monkeypatch.setattr(
        requests, "get",
        lambda *a, **kw: _resp(b"\x00\x01binary", ctype="application/octet-stream"),
    )
    assert isinstance(base.fetch("https://example.com/a.bin"), bytes)


def test_fetch_as_json_still_works(monkeypatch):
    import requests

    monkeypatch.setattr(
        requests, "get",
        lambda *a, **kw: _resp(b"{}", payload={"a": 1}, ctype="application/json"),
    )
    assert base.fetch("https://example.com/a.json", as_json=True) == {"a": 1}


def test_article_extraction_reads_a_real_byte_response(monkeypatch):
    """The original failure, end to end: bytes in, text out, nav dropped."""
    import requests

    from ytauto.sources import article

    html = (
        b"<html><head><meta charset='utf-8'></head><body>"
        b"<nav>Menu</nav><p>The actual body of the story, with real words in it.</p>"
        b"<footer>legal</footer>"
        b"</body></html>"
    )
    monkeypatch.setattr(requests, "get", lambda *a, **kw: _resp(html, encoding="utf-8"))
    out = article.fetch_article_text("https://example.com/story")
    assert out
    assert "actual body of the story" in out
    assert "Menu" not in out
    assert "legal" not in out
