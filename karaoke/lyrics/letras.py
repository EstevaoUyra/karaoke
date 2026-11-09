"""letras.mus.br: the biggest lyrics site for Brazilian songs. Used as a *pointer*, not a source.

Its song pages refuse automated clients (bot protection answers 403 even for robots.txt), so
the lyrics text is never fetched. Only the site's public search-suggestion endpoint is queried,
and the matching song page's URL is saved to ``lyrics/sources/letras.json``. The user (or the web
UI) opens it and pastes the lyrics into ``lyrics/sources/user.txt``, which beats every other
source on the next ``lyrics_fetch`` run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import httpx
from rapidfuzz import fuzz

from karaoke.lyrics.normalize import line_key

SEARCH_URL = "https://solr.sscdn.co/letras/m1/"
SONG_URL = "https://www.letras.mus.br/{artist}/{song}/"
USER_AGENT = "karaoke (https://github.com/EstevaoUyra/karaoke)"
TIMEOUT = 15.0
MIN_SIMILARITY = 80
# Tests set this to an ``httpx.MockTransport``; None = real network.
TRANSPORT: httpx.BaseTransport | None = None

log = logging.getLogger(__name__)


@dataclass
class LetrasHit:
    url: str
    title: str
    artist: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n"


def _similar(a: str, b: str) -> float:
    return fuzz.token_set_ratio(line_key(a), line_key(b))


def parse_suggestions(body: str) -> list[dict]:
    """The endpoint answers JSONP (``LetrasSug({...})``) or plain JSON; returns its docs."""
    body = body.strip()
    if not body.startswith("{"):
        body = body[body.index("(") + 1:body.rindex(")")]
    return json.loads(body).get("response", {}).get("docs", [])


def best_hit(docs: list[dict], title: str, artist: str | None) -> LetrasHit | None:
    best, best_score = None, 0.0
    for d in docs:
        if not str(d.get("id", "")).startswith("mus") or not d.get("dns") or not d.get("url"):
            continue  # artists, albums, …
        t = _similar(title, d.get("txt", ""))
        a = _similar(artist, d.get("art", "")) if artist else 100.0
        if t < MIN_SIMILARITY or a < MIN_SIMILARITY:
            continue
        if t + a > best_score:
            best_score = t + a
            best = LetrasHit(url=SONG_URL.format(artist=d["dns"], song=d["url"]), title=d.get("txt", ""),
                             artist=d.get("art", ""))
    return best


def find(title: str, artist: str | None) -> LetrasHit | None:
    """The letras.mus.br page of this song, or None. Raises httpx.HTTPError on network errors."""
    q = f"{artist} {title}" if artist else title
    with httpx.Client(transport=TRANSPORT, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
        r = c.get(SEARCH_URL, params={"q": q, "wt": "json"})
        r.raise_for_status()
        return best_hit(parse_suggestions(r.text), title, artist)
