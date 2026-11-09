from __future__ import annotations

import httpx
import pytest

from karaoke.config import get_settings
from karaoke.lyrics import genius, lrclib
from karaoke.models import JobSettings, LyricsChoice, LyricsSourceName, Meta, StageName
from karaoke.pipeline import run_job
from karaoke.stages import lyrics_fetch
from karaoke.store import JobStore, read_model, write_model

# All lyrics below are invented.

PT_TEXT = """[Verso 1]
Hoje a cidade acordou mais cedo
o café esfriou perto da janela

[Refrão]
eu não sei se volto pra casa
mas a rua ainda é minha
"""

EN_TEXT = """[Verse]
Paper boats are sailing down the gutter
and the rain keeps drumming on the roof

[Chorus]
float away, my little boat, float away
"""


def _settings():
    return get_settings().model_copy(update={"genius_token": None})


def _lrclib_mock(monkeypatch, handler):
    monkeypatch.setattr(lrclib, "TRANSPORT", httpx.MockTransport(handler))


def _demo_record(p, duration=24.0):
    return {"id": 4242, "trackName": "Sing It Loud", "artistName": "Demo Band", "albumName": None,
            "duration": duration, "instrumental": False,
            "plainLyrics": p.lyrics_source("lrclib.txt").read_text(),
            "syncedLyrics": p.lyrics_source("lrclib.lrc").read_text()}


def _run(p, settings=None):
    store = JobStore(p.root.parent)
    run_job(p.id, only=[StageName.lyrics_fetch], store=store, settings=settings or _settings())
    return read_model(p.lyrics_choice, LyricsChoice), p.lyrics_chosen_text.read_text()


def _set_job_settings(p, **kw):
    store = JobStore(p.root.parent)
    job = store.load(p.id)
    job.settings = JobSettings(**kw)
    store.save(job)


@pytest.fixture
def demo_fetch(demo_no_audio, monkeypatch):
    """demo job without chosen.*; LRCLIB serves the demo's own lrc/txt (24 s, matching)."""
    p = demo_no_audio
    expected = p.lyrics_chosen_text.read_text()
    record = _demo_record(p)
    p.lyrics_choice.unlink()
    p.lyrics_chosen_text.unlink()
    requests = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        if req.url.path == "/api/get":
            return httpx.Response(200, json=record)
        return httpx.Response(200, json=[record])

    _lrclib_mock(monkeypatch, handler)
    return p, expected, requests, record


def test_demo_lrclib_synced_matches_expected_chosen_txt(demo_fetch):
    p, expected, requests, _ = demo_fetch
    choice, text = _run(p)
    assert text == expected
    assert choice.chosen == LyricsSourceName.lrclib
    assert choice.use_line_timings is True
    assert choice.language == "en"
    assert requests[0].url.path == "/api/get"
    assert requests[0].headers["user-agent"] == lrclib.USER_AGENT
    files = [(s.name, s.file, s.synced, s.duration_match) for s in choice.sources]
    assert files == [
        (LyricsSourceName.lrclib, "lyrics/sources/lrclib.lrc", True, True),
        (LyricsSourceName.lrclib, "lyrics/sources/lrclib.txt", False, True),
        (LyricsSourceName.youtube, "lyrics/sources/youtube.vtt", True, None),
    ]
    assert choice.sources[0].note.startswith("chosen;")
    assert choice.sources[0].url == "https://lrclib.net/api/get/4242"
    assert all(s.language == "en" for s in choice.sources)


def test_user_txt_wins(demo_fetch):
    p, *_ = demo_fetch
    p.lyrics_source("user.txt").write_text(EN_TEXT)
    choice, text = _run(p)
    assert choice.chosen == LyricsSourceName.user
    assert choice.use_line_timings is False
    assert text.startswith("[verse]\nPaper boats are sailing down the gutter\n")
    assert "\n\n[chorus]\nfloat away, my little boat, float away\n" in text
    assert [s.name for s in choice.sources][0] == LyricsSourceName.user
    assert LyricsSourceName.lrclib in [s.name for s in choice.sources]  # still fetched and listed


def test_duration_mismatch_keeps_text_but_not_timings(demo_no_audio, monkeypatch):
    p = demo_no_audio
    record = _demo_record(p, duration=31.0)  # another recording
    _lrclib_mock(monkeypatch, lambda req: httpx.Response(200, json=record if req.url.path == "/api/get"
                                                         else [record]))
    choice, text = _run(p)
    assert choice.chosen == LyricsSourceName.lrclib
    assert choice.use_line_timings is False
    assert choice.sources[0].duration == 31.0 and choice.sources[0].duration_match is False
    assert text == make_demo_chosen(p)


def make_demo_chosen(p):
    import make_demo

    lines = []
    prev = None
    for section, words in make_demo.TRUTH:
        if section != prev:
            if lines:
                lines.append("")
            lines.append(f"[{section}]")
            prev = section
        lines.append(" ".join(w for w, _, _ in words))
    return "\n".join(lines) + "\n"


def _nothing(req):
    return httpx.Response(404) if req.url.path == "/api/get" else httpx.Response(200, json=[])


def test_nothing_found_falls_back_to_whisper(demo_no_audio, monkeypatch):
    p = demo_no_audio
    p.lyrics_source("youtube.vtt").unlink()
    _lrclib_mock(monkeypatch, _nothing)
    choice, text = _run(p)
    assert choice.chosen == LyricsSourceName.whisper
    assert text == ""
    assert choice.language is None
    assert choice.use_line_timings is False
    assert [s.name for s in choice.sources] == [LyricsSourceName.whisper]
    # a successful lookup that found nothing removes stale files from an earlier run
    assert not p.lyrics_source("lrclib.lrc").exists()
    assert not p.lyrics_source("lrclib.txt").exists()


def test_youtube_manual_subs_used_when_lrclib_has_nothing(demo_no_audio, monkeypatch):
    p = demo_no_audio
    _lrclib_mock(monkeypatch, _nothing)
    choice, text = _run(p)
    assert choice.chosen == LyricsSourceName.youtube
    assert choice.use_line_timings is False  # no duration to check against
    lines = text.strip().split("\n")
    assert lines[0] == "Morning light is on the window,"
    assert "" in lines  # the 5 s instrumental gap starts a new stanza
    assert len([ln for ln in lines if ln]) == 6


def test_auto_captions_are_not_a_lyrics_source(demo_no_audio, monkeypatch):
    p = demo_no_audio
    meta = read_model(p.meta, Meta)
    meta.has_manual_subs = []
    write_model(p.meta, meta)
    _lrclib_mock(monkeypatch, _nothing)
    choice, text = _run(p)
    assert choice.chosen == LyricsSourceName.whisper and text == ""


def test_network_error_reuses_files_from_previous_run(demo_no_audio, monkeypatch):
    p = demo_no_audio

    def offline(req):
        raise httpx.ConnectError("offline", request=req)

    _lrclib_mock(monkeypatch, offline)
    choice, text = _run(p)
    assert choice.chosen == LyricsSourceName.lrclib
    assert choice.sources[0].duration == 24.0  # from the [length:] tag
    assert choice.use_line_timings is True
    assert "cached" in choice.sources[0].note
    assert text == make_demo_chosen(p)


def test_forced_source_and_language_from_settings(demo_fetch):
    p, *_ = demo_fetch
    _set_job_settings(p, lyrics_source="youtube", language="pt")
    choice, _ = _run(p)
    assert choice.chosen == LyricsSourceName.youtube
    assert choice.language == "pt"  # job settings beat detection


def test_forced_whisper(demo_fetch):
    p, *_ = demo_fetch
    _set_job_settings(p, lyrics_source="whisper")
    choice, text = _run(p)
    assert choice.chosen == LyricsSourceName.whisper and text == ""


@pytest.mark.parametrize("text, lang", [(PT_TEXT, "pt"), (EN_TEXT, "en")])
def test_language_detection(demo_no_audio, monkeypatch, text, lang):
    p = demo_no_audio
    _lrclib_mock(monkeypatch, _nothing)
    p.lyrics_source("user.txt").write_text(text)
    meta = read_model(p.meta, Meta)
    meta.youtube_language = "fr"  # never trusted
    write_model(p.meta, meta)
    choice, chosen = _run(p)
    assert choice.chosen == LyricsSourceName.user
    assert choice.language == lang
    assert chosen.startswith("[verse]\n")


def test_detect_language_helper():
    assert lyrics_fetch.detect_language("") is None
    assert lyrics_fetch.detect_language("eu não sei se volto pra casa amanhã") == "pt"
    assert lyrics_fetch.detect_language("I don't know if I'm coming home tomorrow") == "en"


def test_vtt_parser():
    vtt = ("WEBVTT\nKind: captions\nLanguage: en\n\n1\n00:00:01.000 --> 00:00:03.500 align:start\n"
           "<c>♪ first line ♪</c>\n\n00:04.000 --> 00:06.000\n[Music]\n\n"
           "00:00:06.000 --> 00:00:08.000\nsecond line\nthird line\n")
    assert lyrics_fetch.parse_vtt_cues(vtt) == [(1.0, 3.5, ["first line"]),
                                                (6.0, 8.0, ["second line", "third line"])]


# --------------------------------------------------------------------------- genius (mocked)

GENIUS_PAGE = """<html><body><div class="x">not lyrics</div>
<div data-lyrics-container="true" class="Lyrics__Container"><div data-exclude-from-selection="true">
<div>3 Contributors</div>Paper Boats Lyrics</div>[Verse 1]<br/>Paper boats are <a href="/x"><span>sailing</span></a>
down the gutter<br>and the rain &amp; the wind<br/><br/>[Chorus]<br/>float away</div>
<div data-lyrics-container="true">float away, my boat</div></body></html>"""


def test_genius_extract_lyrics():
    text = genius.extract_lyrics(GENIUS_PAGE)
    assert text == ("[Verse 1]\nPaper boats are sailing\ndown the gutter\nand the rain & the wind\n\n"
                    "[Chorus]\nfloat away\nfloat away, my boat\n")


def _genius_handler(req: httpx.Request) -> httpx.Response:
    if req.url.host == "api.genius.com":
        assert req.headers["authorization"] == "Bearer test-token"
        return httpx.Response(200, json={"response": {"hits": [
            {"type": "song", "result": {"id": 1, "title": "Other Song", "url": "https://genius.com/other",
                                        "primary_artist": {"name": "X"}}},
            {"type": "song", "result": {"id": 2, "title": "Sing It Loud", "url": "https://genius.com/sing",
                                        "primary_artist": {"name": "Demo Band"}}},
        ]}})
    assert str(req.url) == "https://genius.com/sing"
    return httpx.Response(200, text=GENIUS_PAGE)


def test_genius_find_lyrics():
    with httpx.Client(transport=httpx.MockTransport(_genius_handler)) as c:
        found = genius.find_lyrics("Sing It Loud", "Demo Band", "test-token", client=c)
    assert found.song.id == 2 and found.text.startswith("[Verse 1]\n")


def test_stage_uses_genius_when_token_set(demo_no_audio, monkeypatch):
    p = demo_no_audio
    _lrclib_mock(monkeypatch, _nothing)
    monkeypatch.setattr(genius, "TRANSPORT", httpx.MockTransport(_genius_handler))
    settings = get_settings().model_copy(update={"genius_token": "test-token"})
    choice, text = _run(p, settings)
    assert choice.chosen == LyricsSourceName.genius
    assert choice.sources[0].url == "https://genius.com/sing"
    assert p.lyrics_source("genius.txt").exists()
    assert text.startswith("[verse]\nPaper boats are sailing\n")
