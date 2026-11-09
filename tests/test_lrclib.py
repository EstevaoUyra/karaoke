from __future__ import annotations

import httpx
import pytest

from karaoke.lyrics import lrclib


def rec(id_, track, artist, duration, synced=True, plain=True, album=None):
    return {"id": id_, "trackName": track, "artistName": artist, "albumName": album, "duration": duration,
            "instrumental": False, "plainLyrics": "invented words\n" if plain else None,
            "syncedLyrics": "[00:01.00]invented words\n" if synced else None}


def client_for(handler):
    return httpx.Client(base_url=lrclib.BASE_URL, headers={"User-Agent": lrclib.USER_AGENT},
                        transport=httpx.MockTransport(handler))


def test_queries_order():
    qs = lrclib.queries("Paper Boats", "Invented Band", ["Rainy Soundtrack", ""])
    assert qs == [
        {"track_name": "Paper Boats", "artist_name": "Invented Band"},
        {"q": "Paper Boats Rainy Soundtrack"},
        {"track_name": "Paper Boats"},
        {"q": "Invented Band Paper Boats"},
    ]
    assert lrclib.queries("Paper Boats", None) == [{"track_name": "Paper Boats"}, {"q": "Paper Boats"}]


def test_pick_best_closest_duration_synced_wins_ties():
    cands = [lrclib.LrclibRecord.from_api(r) for r in (
        rec(1, "T", "A", 300.0), rec(2, "T", "A", 221.0, synced=False), rec(3, "T", "A", 219.0),
        rec(4, "T", "A", 220.5, synced=False, plain=False),  # no lyrics at all
    )]
    assert lrclib.pick_best(cands, 220.0).id == 3  # |Δ| 1.0 tie with #2, synced wins
    assert lrclib.pick_best(cands, None).id == 1  # unknown duration: first synced
    assert lrclib.pick_best([], 220.0) is None


def test_find_best_multi_query_dedupe_and_user_agent():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, dict(req.url.params)))
        assert req.headers["user-agent"] == "karaoke (https://github.com/EstevaoUyra/karaoke)"
        if req.url.path == "/api/get":
            return httpx.Response(404, json={"message": "Failed to find specified track"})
        params = dict(req.url.params)
        if params.get("q") == "Paper Boats Rainy Soundtrack":
            return httpx.Response(200, json=[rec(7, "Paper Boats", "Cover Singer", 181.0),
                                             rec(5, "Paper Boats", "Invented Band", 240.0)])
        if params.get("track_name") == "Paper Boats":
            return httpx.Response(200, json=[rec(5, "Paper Boats", "Invented Band", 240.0),
                                             rec(6, "Paper Boats (Live)", "Invented Band", 260.0),
                                             rec(9, "Something Else Entirely", "Other", 180.5)])
        return httpx.Response(200, json=[])

    with client_for(handler) as c:
        m = lrclib.find_best("Paper Boats", "Invented Band", 180.4, extra=["Rainy Soundtrack"], client=c)
    assert m is not None
    assert m.record.id == 7  # #9 is closer but its title doesn't match
    assert m.delta == pytest.approx(0.6)
    assert m.record.synced
    assert "Rainy Soundtrack" in m.query
    assert seen[0][0] == "/api/get"
    assert seen[0][1] == {"track_name": "Paper Boats", "artist_name": "Invented Band", "duration": "180"}
    # stopped early: a synced match within 2 s was found by the second search
    assert len(seen) == 3


def test_find_best_get_hit_stops_immediately():
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, json=rec(1, "Paper Boats", "Invented Band", 180.0))

    with client_for(handler) as c:
        m = lrclib.find_best("Paper Boats", "Invented Band", 181.0, client=c)
    assert m.record.id == 1 and m.delta == 1.0 and calls == ["/api/get"]


def test_find_best_nothing():
    def handler(req):
        return httpx.Response(404) if req.url.path == "/api/get" else httpx.Response(200, json=[])

    with client_for(handler) as c:
        assert lrclib.find_best("Nothing Here", "Nobody", 100.0, client=c) is None


def test_network_error_propagates():
    def handler(req):
        raise httpx.ConnectError("offline", request=req)

    with client_for(handler) as c, pytest.raises(httpx.HTTPError):
        lrclib.find_best("Paper Boats", None, 100.0, client=c)


# --------------------------------------------------------------------------- real LRCLIB (Phase 0 facts)


@pytest.mark.slow
def test_real_captain_fantastic_cover_found_by_extra_term():
    m = lrclib.find_best("Sweet Child O Mine", None, 220.3, extra=["Captain Fantastic"])
    assert m is not None
    assert m.record.synced
    assert m.delta is not None and m.delta <= 2.0
    assert "captain fantastic" in (m.record.album_name or "").lower()


@pytest.mark.slow
def test_real_plain_title_search_returns_only_the_original_lengths():
    with lrclib.make_client() as c:
        recs = lrclib.search(c, track_name="Sweet Child O Mine")
    with_lyrics = [r for r in recs if r.has_lyrics and r.duration]
    assert with_lyrics
    assert all(abs(r.duration - 220.3) > 2 for r in with_lyrics)  # the cover isn't among them
    assert min(r.duration for r in with_lyrics if r.synced) >= 300  # synced = the original's 303-356 s


@pytest.mark.slow
def test_real_sacal_not_on_lrclib():
    assert lrclib.find_best("Vão Querer Me Internar", "Sacal", 153.0) is None
