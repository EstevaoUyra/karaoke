"""LRCLIB client (https://lrclib.net, free, no key).

Phase 0 finding: one query is not enough. A plain title search for a cover returns only the original
recording, while ``title + qualifier`` ("Captain Fantastic") finds the cover's own synced LRC. So
``find_best`` runs several queries, deduplicates by id and keeps the candidate whose duration is
closest to our audio (synced beats plain at equal distance).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import httpx
from rapidfuzz import fuzz

from karaoke.lyrics.normalize import line_key

BASE_URL = "https://lrclib.net"
USER_AGENT = "karaoke (https://github.com/EstevaoUyra/karaoke)"
TIMEOUT = 20.0
# Tests set this to an ``httpx.MockTransport``; None = real network.
TRANSPORT: httpx.BaseTransport | None = None
# A candidate's track name must be this similar (token_set_ratio on match tokens) to our title.
MIN_TITLE_SIMILARITY = 80
# When we know the artist, a candidate's artist must be this similar too, unless the candidate came
# from a query naming the version (a cover's soundtrack, a riddim): a cover is sung by someone else.
# Real variants score 100 ("Sergio Mendes + Brasil 66" / "Sérgio Mendes & Brasil '66", "Sacal" /
# "Sacal, Pumpkilla"); another artist's song with the same title scores under 50.
MIN_ARTIST_SIMILARITY = 80

log = logging.getLogger(__name__)


@dataclass
class LrclibRecord:
    id: int
    track_name: str
    artist_name: str
    album_name: str | None
    duration: float | None
    instrumental: bool
    plain_lyrics: str | None
    synced_lyrics: str | None

    @property
    def synced(self) -> bool:
        return bool(self.synced_lyrics and self.synced_lyrics.strip())

    @property
    def has_lyrics(self) -> bool:
        return self.synced or bool(self.plain_lyrics and self.plain_lyrics.strip())

    @property
    def url(self) -> str:
        return f"{BASE_URL}/api/get/{self.id}"

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> LrclibRecord:
        dur = d.get("duration")
        return cls(
            id=int(d["id"]),
            track_name=d.get("trackName") or d.get("name") or "",
            artist_name=d.get("artistName") or "",
            album_name=d.get("albumName"),
            duration=float(dur) if dur is not None else None,
            instrumental=bool(d.get("instrumental")),
            plain_lyrics=d.get("plainLyrics"),
            synced_lyrics=d.get("syncedLyrics"),
        )


@dataclass
class LrclibMatch:
    record: LrclibRecord
    delta: float | None  # |record.duration - our duration|; None if either is unknown
    query: str  # which query found it (for the log / SourceInfo note)
    n_candidates: int


def make_client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT,
                        transport=TRANSPORT, follow_redirects=True)


def get(client: httpx.Client, title: str, artist: str, duration: float | None = None,
        album: str | None = None) -> LrclibRecord | None:
    """``/api/get``: exact signature lookup. None on 404."""
    params: dict[str, Any] = {"track_name": title, "artist_name": artist}
    if album:
        params["album_name"] = album
    if duration:
        params["duration"] = round(duration)
    r = client.get("/api/get", params=params)
    if 400 <= r.status_code < 500:  # 404 = not found; 400 = signature rejected
        return None
    r.raise_for_status()
    return LrclibRecord.from_api(r.json())


def search(client: httpx.Client, **params: str) -> list[LrclibRecord]:
    """``/api/search`` with ``q=`` or ``track_name=``/``artist_name=``/``album_name=``."""
    r = client.get("/api/search", params={k: v for k, v in params.items() if v})
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return [LrclibRecord.from_api(d) for d in r.json()]


def queries(title: str, artist: str | None, extra: Iterable[str] = ()) -> list[dict[str, str]]:
    """The ``/api/search`` queries, in order (``/api/get`` comes first, separately)."""
    qs: list[dict[str, str]] = []
    if artist:
        qs.append({"track_name": title, "artist_name": artist})
    for e in extra:
        if e and e.strip():
            qs.append({"q": f"{title} {e.strip()}"})
    qs.append({"track_name": title})
    qs.append({"q": f"{artist} {title}" if artist else title})
    out: list[dict[str, str]] = []
    for q in qs:  # dedupe, keep order
        if q not in out:
            out.append(q)
    return out


def title_similarity(a: str, b: str) -> float:
    return fuzz.token_set_ratio(line_key(a), line_key(b))


def _rank(rec: LrclibRecord, duration: float | None) -> tuple[float, int]:
    if duration is None or rec.duration is None:
        return (float("inf"), 0 if rec.synced else 1)
    return (abs(rec.duration - duration), 0 if rec.synced else 1)


def pick_best(cands: list[LrclibRecord], duration: float | None) -> LrclibRecord | None:
    """Closest duration; synced before plain at equal |Δ|; first seen breaks remaining ties."""
    ok = [c for c in cands if c.has_lyrics and not c.instrumental]
    if not ok:
        return None
    return min(ok, key=lambda c: _rank(c, duration))  # min() is stable: first seen wins ties


def find_best(title: str, artist: str | None = None, duration: float | None = None,
              extra: Iterable[str] = (), album: str | None = None,
              client: httpx.Client | None = None, stop_within: float = 2.0) -> LrclibMatch | None:
    """Run every query, dedupe by id, return the closest-duration candidate (or None).

    Stops early once a synced candidate within ``stop_within`` seconds is found.
    Network errors propagate (the caller decides whether that is fatal).
    """
    own = client is None
    client = client or make_client()
    seen: dict[int, LrclibRecord] = {}
    found_by: dict[int, str] = {}

    def add(recs: Iterable[LrclibRecord], label: str, check_artist: bool = True) -> bool:
        for r in recs:
            if r.id in seen:
                continue
            if title_similarity(r.track_name, title) < MIN_TITLE_SIMILARITY:
                continue
            if check_artist and artist and title_similarity(r.artist_name, artist) < MIN_ARTIST_SIMILARITY:
                log.info("lrclib: skipping #%s %r by %r: not our artist %r", r.id, r.track_name,
                         r.artist_name, artist)
                continue
            seen[r.id] = r
            found_by[r.id] = label
        best = pick_best(list(seen.values()), duration)
        return bool(best and best.synced and duration is not None and best.duration is not None
                    and abs(best.duration - duration) <= stop_within)

    try:
        done = False
        if artist and duration:
            rec = get(client, title, artist, duration, album)
            done = add([rec] if rec else [], f"get artist={artist!r} title={title!r}")
        if not done:
            versions = {f"{title} {e.strip()}" for e in extra if e and e.strip()}
            for q in queries(title, artist, extra):
                label = "search " + " ".join(f"{k}={v!r}" for k, v in q.items())
                if add(search(client, **q), label, check_artist=q.get("q") not in versions):
                    break
    finally:
        if own:
            client.close()

    best = pick_best(list(seen.values()), duration)
    if best is None:
        return None
    delta = abs(best.duration - duration) if (best.duration is not None and duration is not None) else None
    log.info("lrclib: %d candidates, best id=%s Δ=%s via %s", len(seen), best.id, delta, found_by[best.id])
    return LrclibMatch(best, delta, found_by[best.id], len(seen))
