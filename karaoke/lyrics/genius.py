"""Genius lyrics (optional: only used when ``settings.genius_token`` / ``GENIUS_ACCESS_TOKEN`` is set).

The API search gives the song page URL; the lyrics text is scraped from the page's
``<div data-lyrics-container="true">`` blocks (``<br>`` = new line, ``data-exclude-from-selection``
blocks such as the contributors header are skipped).
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser

import httpx
from rapidfuzz import fuzz

from karaoke.lyrics.normalize import line_key

API_URL = "https://api.genius.com"
USER_AGENT = "karaoke (https://github.com/EstevaoUyra/karaoke)"
TIMEOUT = 20.0
TRANSPORT: httpx.BaseTransport | None = None  # tests set an httpx.MockTransport
MIN_TITLE_SIMILARITY = 80

log = logging.getLogger(__name__)


@dataclass
class GeniusSong:
    id: int
    title: str
    artist: str
    url: str


@dataclass
class GeniusLyrics:
    song: GeniusSong
    text: str


def make_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, transport=TRANSPORT,
                        follow_redirects=True)


def search(client: httpx.Client, token: str, query: str) -> list[GeniusSong]:
    r = client.get(f"{API_URL}/search", params={"q": query}, headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    out = []
    for hit in r.json().get("response", {}).get("hits", []):
        if hit.get("type", "song") != "song":
            continue
        res = hit.get("result") or {}
        if not res.get("url"):
            continue
        out.append(GeniusSong(id=int(res.get("id", 0)), title=res.get("title") or "",
                              artist=(res.get("primary_artist") or {}).get("name") or "", url=res["url"]))
    return out


class _LyricsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0  # >0 while inside a lyrics container (counts nested divs)
        self.exclude_depth = 0  # >0 while inside an excluded block within a container
        self.blocks: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if self.depth == 0:
            if tag == "div" and a.get("data-lyrics-container") == "true":
                self.depth = 1
                self.blocks.append([])
            return
        if tag == "div":
            self.depth += 1
        if self.exclude_depth:
            if tag == "div":
                self.exclude_depth += 1
            return
        if a.get("data-exclude-from-selection") == "true" and tag == "div":
            self.exclude_depth = 1
            return
        if tag == "br":
            self.blocks[-1].append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.depth and not self.exclude_depth and tag == "br":
            self.blocks[-1].append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.depth == 0 or tag != "div":
            return
        self.depth -= 1
        if self.exclude_depth:
            self.exclude_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.depth and not self.exclude_depth:
            self.blocks[-1].append(data)


def extract_lyrics(page_html: str) -> str:
    """Lyrics text from a Genius song page ("" if there are no lyrics containers)."""
    p = _LyricsParser()
    p.feed(page_html)
    p.close()
    parts = ["".join(b) for b in p.blocks]
    text = html.unescape("\n".join(parts)).replace("\r", "")
    lines = [re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # Section headers often sit right after a line with no blank line between them.
    text = re.sub(r"(?<=[^\n])\n(\[[^\]\n]+\])", r"\n\n\1", text)
    return text + "\n" if text else ""


def find_lyrics(title: str, artist: str | None, token: str,
                client: httpx.Client | None = None) -> GeniusLyrics | None:
    own = client is None
    client = client or make_client()
    try:
        hits = search(client, token, f"{artist} {title}" if artist else title)
        for song in hits:
            if fuzz.token_set_ratio(line_key(song.title), line_key(title)) < MIN_TITLE_SIMILARITY:
                continue
            r = client.get(song.url)
            r.raise_for_status()
            text = extract_lyrics(r.text)
            if text.strip():
                log.info("genius: %s - %s (%s)", song.artist, song.title, song.url)
                return GeniusLyrics(song, text)
        return None
    finally:
        if own:
            client.close()
