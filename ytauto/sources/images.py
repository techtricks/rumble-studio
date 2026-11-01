"""Real pictures for a script, so the video is not a gradient with captions.

The renderer used to emit an animated gradient with text burned over it. That
is a text slide show: it renders, it passes every gate, and it looks like
nothing happened. A technology channel needs the thing being talked about on
screen, so pictures are gathered from three places, cheapest and most reliable
first:

1. **The source article's own image** -- ``og:image`` plus content images. The
   outlet picked it for this exact story, so it is guaranteed on topic.
2. **Wikimedia Commons** for named entities in the title. Freely licensed,
   high quality, and no credentials. This is what fills the middle of a video
   about a company or a product with no fresh photo.
3. **Openverse** for the broader subject, CC-licensed across Flickr, Wikimedia
   and museum archives. Anonymous access works; no key, no signup.

Every one of these is a public HTTP API, so there is no paywall, no scraping of
a logged-in page, and nothing to pay for. A miss is never an error: the renderer
falls back to the gradient it already had, so a bad day upstream degrades the
video instead of failing it.

Attribution is collected with every asset, because Openverse content requires
credit and getting that wrong on a published video is a real problem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
from urllib.parse import quote, urlparse

from ..logging_setup import get_logger
from .base import fetch

log = get_logger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_ABSOLUTE_RE = re.compile(r"^(?:https?:)?//", re.I)
_IMG_EXT_RE = re.compile(r"\.(?:jpe?g|png|webp)(?:\?|$)", re.I)

# A 1x1 tracking pixel and a 150px site logo both pass "is a url ending in .jpg".
# Anything under this is chrome, not content.
_MIN_IMAGE_BYTES = 25_000
_MIN_WIDTH = 640
_MIN_ASPECT = 1.1
# A channel about technology does not need a picture of a logo, a flag or a
# navigation sprite. These read as filler even when they are perfectly on topic.
_CHROME_TITLE_RE = re.compile(
    r"\b(icon|logo|flag|coat of arms|map|diagram\s*of|chart of|banner|"
    r"navbox|sidebar|wikimedia|commons-logo|user ?box|art project)\b",
    re.I,
)
_MAX_ASPECT = 2.6

# Noise that makes a generic photo-search worse rather than better.
# A monetised YouTube channel is commercial use, and the renderer crops, zooms
# and composites every picture. So the only licences that are actually safe here
# are CC0/PD and CC BY / CC BY-SA. Everything else is a rights problem waiting to
# happen: NC forbids the channel's own monetisation, ND forbids the crop that
# fills the 16:9 frame, and a link-back requirement is not something a burned-in
# video can satisfy.
_ALLOWED_LICENCE_RE = re.compile(
    r"\b(public\s*domain|cc0|cc[- ]?by(?:-sa)?|by(?:-sa)?|pd|pdm)\b", re.I
)
_BLOCKED_LICENCE_RE = re.compile(r"\b(nc|nd|non-?commercial|no-?deriv)\b", re.I)


def _licence_ok(name: str) -> bool:
    """True when a licence permits commercial use and derivatives.

    An empty or unrecognised licence is rejected. Guessing "probably fine" is how
    a channel ends up monetising a photo it had no right to.
    """
    text = (name or "").strip()
    if not text:
        return False
    if _BLOCKED_LICENCE_RE.search(text):
        return False
    return bool(_ALLOWED_LICENCE_RE.search(text))


_STOPWORDS = frozenset(
    """a an the and or of for to in on at by with from as is are was were be
    being new now why how what when who will can could should would you your
    this that these those it its into over under about after before""".split()
)

# Editorial noise: words that say WHEN or HOW a story is framed, not WHAT it
# shows. Image archives index nouns and entities, so searching these returns
# places and artworks that merely contain the word: "5 Bright Planets to Spot
# in November" pulled in a town called Bright (twice), a movie premiere named
# Bright and three places named "Spot" — a whole video of mountains and post
# offices instead of planets.
_MONTHS = frozenset(
    "january february march april may june july august september october "
    "november december".split()
)
_NOISE = frozenset(
    """bright dark spot spots watch watches find finds found look looks
    hear heard makes making made means meant really literally fact facts
    story stories explained explainer meaning future ever never always best
    worst top ultimate complete real true full hidden secret rare weird
    strange crazy beautiful famous deadly dangerous huge massive giant tiny
    big small tests test where""".split()
)

_WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_COMMONS_API = "https://commons.wikimedia.org/w/api.php"
_OPENVERSE_API = "https://api.openverse.org/v1/images/"


@dataclass
class ImageAsset:
    """One downloaded picture plus the credit its licence requires."""

    path: "object"
    url: str
    title: str = ""
    creator: str = ""
    license_name: str = ""
    source_page: str = ""

    def credit(self) -> str:
        if not self.creator and not self.license_name:
            return ""
        who = self.creator or "unknown"
        lic = self.license_name or "see source"
        return f"{who} / {lic} -- {self.source_page or self.url}"


@dataclass
class _Candidate:
    url: str
    title: str = ""
    creator: str = ""
    license_name: str = ""
    source_page: str = ""
    width: int = 0
    height: int = 0
    score: float = 0.0
    meta: Dict[str, str] = field(default_factory=dict)


def _clean(text: str) -> str:
    return _TAG_RE.sub("", text or "").replace("&amp;", "&").replace("&quot;", '"').strip()


def _absolute(url: str, base: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("/"):
        p = urlparse(base)
        return f"{p.scheme or 'https'}://{p.netloc}{url}"
    return url if url.startswith("http") else ""


def _usable(c: _Candidate) -> bool:
    """Reject chrome: logos, sprites, portraits and slivers."""
    if not c.url.startswith("http"):
        return False
    if _CHROME_TITLE_RE.search(c.title or ""):
        return False
    if not _IMG_EXT_RE.search(c.url) and c.width < _MIN_WIDTH:
        return False
    if c.width and c.height:
        if c.width < _MIN_WIDTH:
            return False
        aspect = c.width / float(c.height)
        if aspect < _MIN_ASPECT or aspect > _MAX_ASPECT:
            return False
    return True


def _relevant(c: _Candidate, keys: Sequence[str], domain: Optional[str] = None) -> bool:
    """Does the picture actually look like it is about this story?

    Commons and Openverse are full-text searches, so a query for "Flipkart
    technology" happily returns a scanned cheque and an RF probe: both files
    mention the words somewhere in their description. Requiring a real content
    word from the story in the file's own title is what stops the video filling
    up with technically-matched, visually-meaningless pictures. A stock photo of
    a server rack for a story about a checkout button is the same failure with
    better manners, so this also rejects candidates whose title says nothing at
    all about the subject.

    Word overlap alone is too strict for picture titles, though: a "Planets"
    story legitimately ships "Jupiter and Saturn conjunction in the night sky",
    which shares no word with the query. So a second pass accepts pictures that
    name the story's own domain nouns — planet, telescope, ocean, coral, fort —
    which keeps the topic's *genre* on screen even when the exact entity word
    is missing. The town called Bright still dies here: it names no space noun.
    """
    if not keys and not domain:
        return True
    hay = f"{c.title} {c.meta.get('description', '')}".lower()
    if not hay.strip():
        return False
    for k in keys:
        k = k.lower()
        if len(k) <= 4:
            # Short keys are matched as whole words. As substrings "AI" turns up
            # in "Prague", "iPad" in "spiral" and "AI" in any "Art" qualifier,
            # which is how a seventeenth-century painting got in for a story
            # about a shopping app.
            if re.search(rf"\b{re.escape(k)}\b", hay):
                return True
            continue
        if k in hay:
            return True
        # A light stem so "robots" matches "robot" and "phones" matches "phone".
        if k.rstrip("s") in hay or k + "s" in hay:
            return True
    if domain and _DOMAIN_HINT_RES[domain].search(hay):
        return True
    return False


# --------------------------------------------------------------------------
# 1. the article's own images
# --------------------------------------------------------------------------
def og_image(url: str, *, timeout: int = 20) -> Optional[str]:
    """The article's own social image, if it advertises one."""
    if not url or not url.startswith(("http://", "https://")):
        return None
    try:
        html = fetch(url, timeout=timeout, attempts=2, as_text=True)
    except Exception as exc:  # noqa: BLE001
        log.debug("og:image fetch failed for %s: %s", url[:60], exc)
        return None
    if not isinstance(html, str):
        return None
    for pat, group in (
        (r"""<meta[^>]+property\s*=\s*["']og:image(?::secure_url)?["'][^>]*content\s*=\s*["']([^"']+)["']""", 1),
        (r"""<meta[^>]+content\s*=\s*["']([^"']+)["'][^>]*property\s*=\s*["']og:image(?::secure_url)?["']""", 1),
        (r"""<meta[^>]+(?:name|property)\s*=\s*["']twitter:image["'][^>]*content\s*=\s*["']([^"']+)["']""", 1),
    ):
        m = re.search(pat, html[:200000], re.I)
        if m:
            cand = _absolute(_clean(m.group(group)), url)
            if cand:
                return cand
    return None


def article_images(url: str, *, limit: int = 4, timeout: int = 20) -> List[_Candidate]:
    """The article's lead image plus any content images inside it."""
    if not url or not url.startswith("http"):
        return []
    out: List[_Candidate] = []
    seen = set()

    lead = og_image(url, timeout=timeout)
    if lead and lead not in seen:
        seen.add(lead)
        # The story's own picture: always worth the first slot.
        out.append(_Candidate(url=lead, title="article image", score=100.0))

    try:
        html = fetch(url, timeout=timeout, attempts=2, as_text=True)
    except Exception as exc:  # noqa: BLE001
        log.debug("article image scan failed for %s: %s", url[:60], exc)
        return out
    if not isinstance(html, str):
        return out

    for m in re.finditer(r"""<img[^>]+(?:src|data-src)\s*=\s*["']([^"']+)["']""", html, re.I):
        cand = _absolute(_clean(m.group(1)), url)
        if not cand or cand in seen:
            continue
        # "resize=1200,800" style URLs hide the real filename behind a query.
        if not _IMG_EXT_RE.search(cand.split("?")[0] + "?x"):
            continue
        seen.add(cand)
        out.append(_Candidate(url=cand, title="article image", score=60.0))
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# 2. Wikimedia Commons, for entities
# --------------------------------------------------------------------------
def commons_images(query: str, *, limit: int = 4, timeout: int = 14) -> List[_Candidate]:
    if not query.strip():
        return []
    params = (
        "?action=query&generator=search"
        f"&gsrsearch={quote('filetype:bitmap ' + query)}"
        f"&gsrlimit={max(limit * 2, 6)}&gsrnamespace=6"
        "&prop=imageinfo&iiprop=url|size|extmetadata&iiurlwidth=1280&format=json"
    )
    try:
        data = fetch(_COMMONS_API + params, timeout=timeout, attempts=1, as_json=True)
    except Exception as exc:  # noqa: BLE001
        log.debug("commons lookup failed for %r: %s", query, exc)
        return []
    if not isinstance(data, dict):
        return []

    out: List[_Candidate] = []
    for page in (data.get("query", {}).get("pages") or {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url") or ""
        if not url:
            continue
        meta = info.get("extmetadata") or {}

        def _m(key: str) -> str:
            return _clean((meta.get(key) or {}).get("value", ""))

        c = _Candidate(
            url=url,
            title=_clean(page.get("title", "")),
            creator=_m("Artist"),
            license_name=_m("LicenseShortName"),
            source_page=info.get("descriptionurl", ""),
            width=int(info.get("thumbwidth") or info.get("width") or 0),
            height=int(info.get("thumbheight") or info.get("height") or 0),
            score=40.0,
        )
        if _usable(c) and _licence_ok(c.license_name):
            out.append(c)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# 3. Openverse, for the wider subject
# --------------------------------------------------------------------------
def openverse_images(query: str, *, limit: int = 4, timeout: int = 12) -> List[_Candidate]:
    if not query.strip():
        return []
    params = (
        f"?q={quote(query)}&page_size={max(limit * 2, 8)}"
        # commercial + modification == CC0/PDM/BY/BY-SA. Asking for all-cc
        # pulls in NC and ND pictures that this channel may not use.
        "&license_type=commercial,modification&mature=false"
    )
    try:
        data = fetch(_OPENVERSE_API + params, timeout=timeout, attempts=1, as_json=True)
    except Exception as exc:  # noqa: BLE001
        log.debug("openverse lookup failed for %r: %s", query, exc)
        return []
    if not isinstance(data, dict):
        return []

    out: List[_Candidate] = []
    for r in data.get("results", []):
        c = _Candidate(
            url=r.get("url") or "",
            title=_clean(r.get("title") or ""),
            creator=_clean(r.get("creator") or ""),
            license_name=(_clean(r.get("license") or "")).upper(),
            source_page=r.get("foreign_landing_url") or r.get("url") or "",
            width=int(r.get("width") or 0),
            height=int(r.get("height") or 0),
            score=20.0,
            meta={"description": _clean(r.get("description") or "")[:300]},
        )
        if _usable(c) and _licence_ok(c.license_name):
            out.append(c)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# queries
# --------------------------------------------------------------------------
def _keywords(title: str, tags: Sequence[str] = ()) -> List[str]:
    """Content words worth searching on, most significant first.

    Image archives (Commons, Openverse) index *entities*: "Sanchi", "Stupa",
    "Nalanda". A video title's subtitle — "Sanchi Stupa: The Art of Absence"
    — contributes editorial words like "Art" or "Absence" that search engines
    happily match into unrelated pictures. So the words before the colon come
    first, tags next, and the subtitle is only mined when the entity part
    alone looks too thin to build varied queries from — and even then only
    words that belong to the story's own domain survive, so a sea story
    keeps "Turtles" while "Shoreless" is dropped. Editorial framing words
    (bright, spot, november, story, ...) never survive at all: they are how
    a planets video ended up searching a town called Bright.
    """
    entity, _, subtitle = (title or "").partition(":")
    dom = _domain(f"{title} {' '.join(str(t) for t in tags)}")
    domain_words = _DOMAIN_HINT_RES[dom]
    words: List[str] = []

    def _add(w: str) -> bool:
        low = w.lower()
        if low in _STOPWORDS or low in _MONTHS or low in _NOISE:
            return False
        if any(low == x.lower() for x in words):
            return False
        words.append(w)
        return True

    for w in re.findall(r"[A-Za-z][A-Za-z0-9+.\-]{2,}", entity or title or ""):
        _add(w)
    # Numbers in an entity ("1857", "9-Story") are searchable history anchors.
    for t in tags or []:
        for w in re.findall(r"[A-Za-z][A-Za-z0-9+.\-]{2,}", str(t)):
            if _add(w):
                break
    if len(words) < 4 and subtitle:
        for w in re.findall(r"[A-Za-z][A-Za-z0-9+.\-]{2,}", subtitle):
            if domain_words.search(w):
                _add(w)
    return words


HISTORY_FACETS = [
    "temple", "monument", "fort", "palace", "sculpture", "statue",
    "ruins", "manuscript", "painting", "architecture", "museum",
    "heritage site", "archaeology", "king", "empire", "battle",
]
TECH_FACETS = [
    "technology", "electronics", "gadgets", "data center", "circuit board",
    "server", "smartphone", "chip", "software", "developer",
]
# The sea and space channels need their OWN visual vocabulary. Before these
# existed, a space title fell through to the history default and searched
# "Planets temple" and "Spot palace" — facet words that no planet photo
# carries, so the relevance gate kept whatever matched the entity word alone.
SPACE_FACETS = [
    "night sky", "telescope", "planet", "astronomy", "solar system",
    "observatory", "star cluster", "galaxy", "spacecraft", "moon",
]
SEA_FACETS = [
    "ocean", "marine life", "underwater", "coral reef", "tropical fish",
    "deep sea", "coast", "seabed", "dolphin", "shipwreck",
]
DOMAIN_FACETS = {
    "history": HISTORY_FACETS,
    "tech": TECH_FACETS,
    "space": SPACE_FACETS,
    "sea": SEA_FACETS,
}
_HISTORY_HINT_RE = re.compile(
    r"\b(empire|dynast(?:y|ies)|temple|king|queen|maharaja|maratha|mauryan|mughal|gupta|"
    r"chola|stupa|fort|palace|monument|ruins|ancient|medieval|heritage|"
    r"civilisation|civilization|battle|war|histor(?:y|ic)|archaeolog|"
    r"1857|independence|raj|pillar|manuscript|inscription)\b", re.I)
_TECH_HINT_RE = re.compile(
    r"\b(tech(?:nology)?|ai\b|artificial intelligence|app\b|software|chip|gpu|"
    r"startup|data|robot|drone|battery|smartphone|internet|code|api|cloud)\b", re.I)
_SPACE_HINT_RE = re.compile(
    r"\b(planet|planets|planetary|stars?|starry|galax(?:y|ies)|nebula|"
    r"constellations?|telescopes?|orbit|orbits|asteroids?|comets?|meteors?|"
    r"supernovae?|pulsars?|quasars?|black holes?|neutron|cosmos|cosmic|"
    r"universe|astronom\w*|astronauts?|rockets?|spacex|nasa|esa\b|jwst|"
    r"hubble|voyager|apollo|solar system|eclipses?|auroras?|mars|jupiter|"
    r"saturn|venus|mercury|neptune|uranus|pluto|milky way|moons?|lunar|"
    r"solar|exoplanets?|dark matter|dark energy|gravit\w*|interstellar|"
    r"observatories|observatory|skies|sky|starlight|light years?|pale blue dot)\b",
    re.I)
_SEA_HINT_RE = re.compile(
    r"\b(oceans?|seas?|marine|underwater|corals?|reefs?|fish|fishes|whales?|"
    r"dolphins?|sharks?|seals?|turtles?|octopuses?|squid|jellyfish|jelly|"
    r"eels?|sponges?|plankton|kelp|mangroves?|tides?|tidal|coasts?|coastal|"
    r"deep[- ]sea|divers?|diving|shipwrecks?|submarines?|sargasso|mariana|"
    r"trench|seabeds?|oceanic|brine|lagoons?|atolls?|estuaries|seagrass|"
    r"abyss|hydrothermal)\b", re.I)
_DOMAIN_HINT_RES = {
    "history": _HISTORY_HINT_RE,
    "tech": _TECH_HINT_RE,
    "space": _SPACE_HINT_RE,
    "sea": _SEA_HINT_RE,
}
# Tie-break order: history is the engine's home domain, so it wins a draw.
_DOMAIN_ORDER = ("history", "space", "sea", "tech")


def _domain(text: str) -> str:
    """Which visual vocabulary this story belongs to.

    Counts each domain's hint words in the title+tags and takes the winner —
    a single hard-coded domain cannot serve an engine that now writes history,
    sea AND space videos. A title with no hints at all stays history, which
    is the behaviour every existing channel was tuned against.
    """
    counts = {d: len(_DOMAIN_HINT_RES[d].findall(text or "")) for d in _DOMAIN_ORDER}
    best = max(_DOMAIN_ORDER, key=lambda d: (counts[d], -_DOMAIN_ORDER.index(d)))
    return best if counts[best] else "history"


def build_queries(title: str, tags: Sequence[str] = ()) -> List[str]:
    """Distinct search phrases, so slides do not repeat one generic photo.

    Each phrase pairs a different entity or facet with a different visual
    noun, chosen from the domain the story is actually about. The facets used
    to be a hardcoded technology list, which made a history channel search
    "Art gadgets" and "Absence data center" — then a history/tech-only split
    made a SPACE channel search "Planets temple". _domain() now picks the
    vocabulary from the title itself (history / tech / space / sea), and a
    title whose keywords all got filtered as editorial noise still gets
    domain-vocabulary queries instead of nothing.
    """
    if not (title or "").strip() and not tags:
        return []
    text = f"{title} {' '.join(str(t) for t in tags)}"
    facets = DOMAIN_FACETS[_domain(text)]
    keys = _keywords(title, tags)
    if not keys:
        # Nothing searchable survived the noise filter — search the domain's
        # own visual nouns rather than nothing, so the relevance gate still
        # has something honest to hold pictures against.
        return [f"{facets[0]} {facets[1]}", facets[2]] if len(facets) >= 3 else []
    queries: List[str] = []
    for i, key in enumerate(keys[:4]):
        if i == 0:
            queries.append(key)
        else:
            queries.append(f"{key} {facets[i % len(facets)]}")
    # One entity word ("Planets") would otherwise repeat a single photo all
    # video: pad with the domain's visual vocabulary for variety, skipping a
    # facet that is just the key's own stem ("Planets planet").
    i = 1
    while len(queries) < 3 and i < len(facets):
        if facets[i].split()[0].lower().rstrip("s") != keys[0].lower().rstrip("s"):
            queries.append(f"{keys[0]} {facets[i]}")
        i += 1
    return queries


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def collect(
    *,
    url: str = "",
    title: str = "",
    tags: Sequence[str] = (),
    limit: int = 6,
) -> List[_Candidate]:
    """Best available pictures for a story, most trustworthy source first."""
    cands: List[_Candidate] = []
    if url:
        cands.extend(article_images(url, limit=3))
    keys = _keywords(title, tags)
    dom = _domain(f"{title} {' '.join(str(t) for t in tags)}")
    if not keys:
        # The gate must never be wide open: with no searchable word left,
        # hold every picture to the domain's own vocabulary instead.
        keys = DOMAIN_FACETS[dom][:2]
    for q in build_queries(title, tags):
        if len(cands) >= limit * 2:
            break
        # Commons first for a bare entity name: the licensing is clearer and the
        # pictures are usually more representative of the product.
        cands.extend(c for c in commons_images(q, limit=3)
                     if _relevant(c, keys, domain=dom))
        cands.extend(c for c in openverse_images(q, limit=3)
                     if _relevant(c, keys, domain=dom))

    seen = set()
    out: List[_Candidate] = []
    for c in sorted(cands, key=lambda x: -x.score):
        key = c.url.split("?")[0]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out[:limit]


def _normalize(data: bytes, dest_dir: Path, stem: str) -> Optional[Path]:
    """Decode through ffmpeg into a plain baseline JPEG.

    A corrupt or exotic asset (truncated JPEG, huge PNG, odd colorspace) can
    make the zoompan filter chain stall instead of failing fast — the
    Talikota render hung to a deterministic 30-minute timeout five runs in
    a row before this. Re-encoding at fetch time means only ffmpeg-clean
    pixels ever reach the renderer, and the normalized bytes are what get
    cached, so the cost is paid once per URL.
    """
    try:
        from ..render import ffmpeg as ff
        from pathlib import Path
    except Exception:  # noqa: BLE001 — no usable ffmpeg: keep the raw bytes
        return None
    import tempfile

    src = Path(tempfile.mkstemp(suffix=".img", dir=dest_dir)[1])
    try:
        src.write_bytes(data)
        out = dest_dir / f"{stem}.n.jpg"
        ff.run(
            [
                "-i", str(src), "-frames:v", "1", "-map", "0:v:0",
                "-q:v", "2", str(out),
            ]
        )
        return out
    except Exception:  # noqa: BLE001 — undecodable asset never ships
        log.debug("image normalize failed %s", data[:8])
        return None
    finally:
        try:
            src.unlink()
        except OSError:
            pass


def download(cand: _Candidate, dest_dir: "object", *, timeout: int = 30) -> Optional[ImageAsset]:
    """Fetch a candidate to disk, cached by URL digest, validating the bytes.

    Only ffmpeg-decodable pixels leave this function. An asset that fails
    normalization (corrupt file, or a file that crashes the jpeg/png decoder
    — one Obama PNG cracked ffmpeg's decoder thread outright) is dropped
    outright, never shipped raw: a raw fallback once hung the renderer in an
    infinite "Failed to reallocate parser buffer" loop for five cycles.
    """
    from pathlib import Path

    from ..util import digest

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = digest(cand.url)[:24]

    # A previously normalized twin short-circuits the whole fetch.
    norm = dest_dir / f"{name}.n.jpg"
    if norm.exists() and norm.stat().st_size > 0:
        return ImageAsset(
            path=norm, url=cand.url, title=cand.title, creator=cand.creator,
            license_name=cand.license_name, source_page=cand.source_page,
        )

    suffix = Path(urlparse(cand.url).path).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
        suffix = ".jpg"
    path = dest_dir / f"{name}{suffix}"

    if path.exists() and path.stat().st_size >= _MIN_IMAGE_BYTES:
        data = path.read_bytes()   # older cache: still goes through ffmpeg below
    else:
        try:
            resp = fetch(cand.url, timeout=timeout, attempts=2, as_text=False)
        except Exception as exc:  # noqa: BLE001
            log.debug("image download failed %s: %s", cand.url[:60], exc)
            return None
        if not isinstance(resp, (bytes, bytearray)) or len(resp) < _MIN_IMAGE_BYTES:
            return None
        if not _sniff_image(bytes(resp)):
            return None
        data = bytes(resp)
        path.write_bytes(data)     # cached before normalization so a
        # network blip never re-downloads; removed below if it can't decode.

    normalized = _normalize(data, dest_dir, name)
    if normalized is not None and normalized.exists():
        return ImageAsset(
            path=normalized, url=cand.url, title=cand.title,
            creator=cand.creator, license_name=cand.license_name,
            source_page=cand.source_page,
        )
    # Undecodable: purge the cache entry so it never resurfaces.
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    log.info("dropped undecodable image %s (%s)", name, cand.url[:80])
    return None


def _sniff_image(data: bytes) -> bool:
    """Trust the bytes, not the extension: a 404 page is not a JPEG."""
    return (
        data[:3] == b"\xff\xd8\xff"                       # JPEG
        or data[:8] == b"\x89PNG\r\n\x1a\n"               # PNG
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")  # WEBP
    )


def images_for_video(
    *,
    url: str,
    title: str,
    tags: Sequence[str] = (),
    dest_dir: "object",
    limit: int = 6,
) -> List[ImageAsset]:
    """Everything the renderer needs: local files plus their credits."""
    out: List[ImageAsset] = []
    try:
        cands = collect(url=url, title=title, tags=tags, limit=limit)
    except Exception as exc:  # noqa: BLE001
        # A picture problem must never cost a rendered video. Every caller wants
        # the gradient over an exception, so the guarantee lives here rather
        # than depending on each call site remembering to catch.
        log.warning("image lookup failed for %r: %s", (title or "")[:50], exc)
        return []
    for cand in cands:
        asset = download(cand, dest_dir)
        if asset:
            out.append(asset)
    log.info(
        "collected %d image(s) for %r%s",
        len(out), (title or "")[:50],
        "" if out else " -- renderer will fall back to its gradient",
    )
    return out
