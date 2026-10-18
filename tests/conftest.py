"""Shared fixtures.

Two layers of isolation, because one is not enough:

1. Module-level env vars set the moment pytest imports this file. `ytauto.config`
   resolves `DB_PATH`, `TOKEN_PATH` and friends into module constants *at import
   time*, and pytest imports test modules during collection — before any fixture
   runs. Env set in a fixture would therefore arrive too late and the suite
   would happily read and write the real `data/ytauto.db`.
2. Fixtures pass explicit paths as well, so a test is correct even if the
   import order changes.

The suite touches no network and no real credentials.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- layer 1: import-time redirection (must stay above any `import ytauto`) ---
_TMP = Path(tempfile.mkdtemp(prefix="ytauto-tests-"))
atexit.register(shutil.rmtree, str(_TMP), True)

os.environ["DATA_DIR"] = str(_TMP)             # moves db + media + logs together
os.environ["DB_PATH"] = str(_TMP / "test.db")
os.environ["TOKEN_PATH"] = str(_TMP / "secrets" / "token.json")
os.environ["CLIENT_SECRET_PATH"] = str(_TMP / "secrets" / "client_secret.json")
os.environ["ANTHROPIC_API_KEY"] = ""          # force the offline path
os.environ["DRY_RUN"] = "1"
os.environ["AUTO_APPROVE"] = "0"
os.environ["RSS_FEEDS"] = "https://example.com/feed"
os.environ["SUBREDDITS"] = ""
os.environ["YOUTUBE_DAILY_QUOTA"] = "10000"
# The fixtures are written against the default technology niche (Waymo,
# self-driving fleet …). A developer's own .env sets NICHE=indian history, and
# load_dotenv() would win because setdefault is not used here — so pin the
# suite to the default like every other setting above. Mirrors
# ytauto.config.DEFAULT_NICHE; keep the two in sync.
os.environ["NICHE"] = (
    "tech,technology,ai,gadgets,"
    "artificial intelligence,machine learning,chatbot,"
    "smartphone,phone,laptop,tablet,headphone,earbuds,wearable,smartwatch,"
    "gaming,console,handheld,gpu,cpu,chip,semiconductor,processor,foundry,"
    "software,app,apps,operating system,update,firmware,driver,open source,"
    "coding,programming,developer,cloud,data centre,data center,server,"
    "cybersecurity,hack,hacker,data breach,encryption,malware,privacy,"
    "robot,robotics,drone,autonomous,self-driving,driverless,robotaxi,"
    "electric vehicle,ev,battery,tesla,space,satellite,nasa,rocket,launch,"
    "quantum,blockchain,crypto,bitcoin,streaming,social media,startup,"
    "internet,browser,search engine,5g,6g,algorithm"
)
# Same reason as NICHE: the production .env runs history sources whose niche
# vocabulary does not overlap the fixtures, so the topic gate is set to 0.
# Fixture expectations (only clearly-niche topics clear discovery) assume the
# default thresholds — pin them too.
os.environ["MIN_NICHE_SCORE"] = "0.22"
os.environ["MIN_TOTAL_SCORE"] = "0.35"
# Same reason: the production .env now runs a Hindi channel with English
# sidecar captions, but the fixtures assert the code's *English* defaults
# (voice pool rotation, caption chunking). Pin the language too.
os.environ["LANGUAGE"] = "English"
os.environ["SUBTITLE_LANGUAGES"] = ""

import ytauto.config as _config  # noqa: E402  (import must follow the env above)

# Hard guarantee, not a hope: if anything resolved a path outside the temp dir,
# fail loudly instead of quietly operating on the developer's real data.
_LEAKED = {
    "DB_PATH": _config.DB_PATH,
    "TOKEN_PATH": _config.TOKEN_PATH,
    "CLIENT_SECRET_PATH": _config.CLIENT_SECRET_PATH,
    "LOG_DIR": _config.LOG_DIR,
}
for _name, _value in _LEAKED.items():
    if _TMP not in Path(_value).parents and Path(_value) != _TMP:
        raise RuntimeError(
            f"test isolation leak: {_name} resolved to {_value}, which is outside {_TMP}"
        )


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any accidental real HTTP call fails the test loudly.

    Blocked at `socket.connect`, the one chokepoint every HTTP stack funnels
    through — urllib, requests, httpx (which the anthropic SDK uses) and the
    Google API client alike.

    The previous version patched each library's entry point instead
    (`urlopen`, `requests.get`, `Session.request`). That missed `socket`
    entirely, so a raw connection still went out: the suite really did reach
    generativelanguage.googleapis.com once a developer's real GEMINI_API_KEY was
    present in .env, and the affected tests hung on retries instead of failing.
    Patching libraries is a losing game; patching the socket is not.
    """
    import socket
    import urllib.request

    import requests

    def _blocked(*a, **k):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError(
            "test attempted a real network call — stub the HTTP client instead"
        )

    def _blocked_connect(self, address, *a, **k):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError(
            f"test attempted a real network call to {address!r} — "
            "stub the HTTP client instead"
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect, raising=True)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_connect, raising=True)
    monkeypatch.setattr(socket, "create_connection", _blocked, raising=True)
    # Keep the friendlier library-level messages for the common case.
    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    monkeypatch.setattr(requests, "get", _blocked)
    monkeypatch.setattr(requests, "post", _blocked)
    monkeypatch.setattr(requests, "put", _blocked)
    # Patch the request METHOD, not the Session class: replacing the class
    # breaks code that subclasses requests.Session *during a test*
    # (google.auth.transport.requests does exactly that, lazily).
    monkeypatch.setattr(requests.Session, "request", _blocked)
    yield


@pytest.fixture(autouse=True)
def _no_ambient_llm_key(monkeypatch):
    """Tests must never inherit a real LLM key from .env.

    `config` calls `load_dotenv()` at import, so a developer's real
    GEMINI_API_KEY is in os.environ for the whole session. Without this, tests
    that expect the offline fallback quietly take the live-API path instead:
    correct-looking tests that spend real quota, take ~30s each, and hang the
    suite on retries. Test behaviour must depend on the test, not the machine.

    The variables are derived from the `*_api_key` fields on Settings rather
    than listed by hand. A hand-written list went stale the moment Groq and
    NVIDIA were added: a real GROQ_API_KEY in .env silently changed which
    provider the failover tests exercised.
    """
    from ytauto.config import Settings

    monkeypatch.setenv("LLM_PROVIDER", "auto")
    for name in Settings.__dataclass_fields__:
        if name.endswith("_api_key"):
            monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture(autouse=True)
def _per_test_paths(tmp_path, monkeypatch):
    """A fresh database and log dir per test.

    Without this the suite shares one database and state leaks between tests —
    a topic created by one test gets scripted by the next, which is exactly how
    the CLI tests started failing for reasons that had nothing to do with what
    they were testing.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("TOKEN_PATH", str(tmp_path / "secrets" / "token.json"))
    monkeypatch.setenv("CLIENT_SECRET_PATH", str(tmp_path / "secrets" / "client_secret.json"))
    yield


@pytest.fixture
def db(tmp_path):
    from ytauto.db import Database

    return Database(tmp_path / "test.db")


@pytest.fixture
def settings():
    from ytauto.config import get_settings

    return get_settings()


@pytest.fixture
def sample_topic():
    return {
        "id": "t1",
        "title": "Nvidia ships a 4x faster GPU",
        "url": "https://example.com/story",
        "source": "rss",
        "summary": (
            "Nvidia announced a new datacenter GPU on Tuesday. The company says "
            "it is four times faster on inference than the previous part. "
            "Analysts argue the real constraint is power draw, not compute."
        ),
        "published_at": "2099-01-01T00:00:00+00:00",
        "total_score": 0.9,
        "niche_score": 0.8,
    }


def test_suite_cannot_reach_the_network():
    """Guard, not a fixture: proves the socket block is armed for every test.

    If this ever fails, the suite can make real API calls — and with a key in
    .env that means real quota and real money.
    """
    import socket

    try:
        socket.create_connection(("generativelanguage.googleapis.com", 443), timeout=2)
    except OSError:
        return
    raise AssertionError("tests can open a socket; the network block is not armed")


# Captured before the fixture below neutralises it, so a test can still assert
# the shipped request pace rather than the test-suite stand-in.
from ytauto import llm_gemini as _llm_gemini

REAL_FREE_TIER_MIN_INTERVAL = _llm_gemini._FREE_TIER_MIN_INTERVAL


@pytest.fixture(autouse=True)
def _no_llm_throttle(monkeypatch):
    """Zero the free-tier request pacing for the suite.

    The real gap is 3.2s to stay under the 20 requests/minute quota, which is
    correct in production and would add a minute of pure sleeping to the tests.
    """
    monkeypatch.setattr("ytauto.llm_gemini._FREE_TIER_MIN_INTERVAL", 0.0)
    monkeypatch.setattr("ytauto.llm_gemini._OVERLOAD_BASE_DELAY", 0.0)
    monkeypatch.setattr("ytauto.llm_gemini._RATE_LIMIT_WAIT_S", 0.0)
