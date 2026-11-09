from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest
import soundfile as sf

from karaoke.config import get_settings
from karaoke.models import Meta, StageName
from karaoke.pipeline import StageContext
from karaoke.stages import ingest
from karaoke.store import JobStore, read_model

URL = "https://www.youtube.com/watch?v=AAAAAAAAAAA"

# Invented subtitle text.
MANUAL_VTT = "WEBVTT\n\n00:00:00.200 --> 00:00:01.000\n♪ Paper boats ♪\n\n00:00:01.000 --> 00:00:01.900\non the river\n"


def _info(**over: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "id": "AAAAAAAAAAA",
        "title": "Demo Band - Sing It Loud (Official Video) [HD]",
        "uploader": "Demo Band VEVO",
        "duration": 2,
        "thumbnail": "https://i.ytimg.com/vi/AAAAAAAAAAA/maxresdefault.jpg",
        "language": "fr",
        "subtitles": {
            "pl": [{"ext": "vtt", "url": "https://subs.invalid/pl.vtt"}],
            "en": [{"ext": "json3", "url": "https://subs.invalid/en.json3"},
                   {"ext": "vtt", "url": "https://subs.invalid/en.vtt"}],
        },
        "automatic_captions": {"pt": [{"ext": "vtt", "url": "https://subs.invalid/auto-pt.vtt"}]},
    }
    info.update(over)
    return info


def _make_mp4(path: Path, seconds: float = 2.0) -> None:
    ffmpeg = get_settings().ffmpeg_bin
    subprocess.run(
        [ffmpeg, "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"color=c=black:s=64x36:d={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
         "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-ac", "1", str(path)],
        check=True, stdin=subprocess.DEVNULL,
    )


class _Resp:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL: records calls, 'downloads' a tiny mp4, serves subtitles."""

    calls: list[tuple[str, bool]] = []
    opened: list[str] = []
    info: dict[str, Any] = {}
    seconds: float = 2.0

    def __init__(self, opts: dict[str, Any]):
        self.opts = opts
        assert opts["noplaylist"] is True
        assert opts["js_runtimes"] == {get_settings().yt_js_runtime: {}}

    def __enter__(self) -> FakeYDL:
        return self

    def __exit__(self, *a: object) -> None:
        return None

    def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
        FakeYDL.calls.append((url, download))
        info = dict(FakeYDL.info)
        if download:
            out = Path(self.opts["outtmpl"]["default"].replace("%(ext)s", "mp4"))
            hook = self.opts["progress_hooks"][0]
            for i in range(1, 5):
                hook({"status": "downloading", "filename": str(out), "downloaded_bytes": i * 25,
                      "total_bytes": 100})
            _make_mp4(out, FakeYDL.seconds)
            hook({"status": "finished", "filename": str(out), "downloaded_bytes": 100, "total_bytes": 100})
            info["requested_downloads"] = [{"filepath": str(out)}]
        return info

    def urlopen(self, url: str) -> _Resp:
        FakeYDL.opened.append(url)
        return _Resp(MANUAL_VTT.encode())


@pytest.fixture
def fake_ydl(monkeypatch):
    import yt_dlp

    FakeYDL.calls, FakeYDL.opened, FakeYDL.info, FakeYDL.seconds = [], [], _info(), 2.0
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    return FakeYDL


def _ctx(store: JobStore, job_id: str, force: bool = False) -> StageContext:
    return StageContext(store.load(job_id), store.paths(job_id), store, get_settings(), StageName.ingest,
                        logging.getLogger("test.ingest"), force=force)


def test_ingest_downloads_and_writes_everything(store: JobStore, fake_ydl):
    job = store.create(url=URL + "&list=RDxyz")
    ctx = _ctx(store, job.id)
    ingest.run(ctx)
    p = ctx.paths

    assert fake_ydl.calls == [(URL, True)]
    assert p.source_video.exists()
    info = sf.info(str(p.audio))
    assert (info.samplerate, info.channels, info.subtype) == (44100, 2, "PCM_16")

    meta = read_model(p.meta, Meta)
    assert meta.duration == pytest.approx(2.0, abs=0.1)
    assert meta.video_id == "AAAAAAAAAAA" and meta.url == URL
    assert (meta.clean_artist, meta.clean_title, meta.clean_extra) == ("Demo Band", "Sing It Loud", [])
    assert meta.youtube_language == "fr"
    assert sorted(meta.has_manual_subs) == ["en", "pl"]  # automatic captions are not listed

    # en (a song language) beats pl even though pl is listed first; auto captions are never used
    assert fake_ydl.opened == ["https://subs.invalid/en.vtt"]
    assert p.lyrics_source("youtube.vtt").read_text() == MANUAL_VTT
    assert ctx.job.stages[StageName.ingest].progress == 1.0


def test_ingest_is_idempotent(store: JobStore, fake_ydl):
    job = store.create(url=URL)
    ingest.run(_ctx(store, job.id))
    p = store.paths(job.id)
    mtime = p.source_video.stat().st_mtime_ns

    # 1) everything present: no network at all
    fake_ydl.calls.clear()
    ingest.run(_ctx(store, job.id))
    assert fake_ydl.calls == []
    assert p.source_video.stat().st_mtime_ns == mtime

    # 2) meta and audio lost: rebuilt from metadata only, the video is not downloaded again
    p.meta.unlink()
    p.audio.unlink()
    ingest.run(_ctx(store, job.id))
    assert fake_ydl.calls == [(URL, False)]
    assert p.audio.exists() and read_model(p.meta, Meta).clean_title == "Sing It Loud"
    assert p.source_video.stat().st_mtime_ns == mtime


def test_no_manual_subs_means_no_vtt(store: JobStore, fake_ydl):
    fake_ydl.info = _info(subtitles={}, title="Sacal - Vão Querer Me Internar (Tropical Paradise Riddim)",
                          uploader="Sacal Prod")
    job = store.create(url=URL)
    ingest.run(_ctx(store, job.id))
    p = store.paths(job.id)
    assert not p.lyrics_source("youtube.vtt").exists()
    assert fake_ydl.opened == []
    meta = read_model(p.meta, Meta)
    assert meta.has_manual_subs == []
    assert (meta.clean_artist, meta.clean_extra) == ("Sacal", ["Tropical Paradise Riddim"])


def test_music_fields_are_mapped(store: JobStore, fake_ydl):
    fake_ydl.info = _info(subtitles={"pt-BR": [{"ext": "vtt", "url": "https://subs.invalid/pt.vtt"}]},
                          title="Sing It Loud", uploader="Demo Band - Topic", artists=["Demo Band", "Guest"],
                          track="Sing It Loud", album="Invented Album")
    job = store.create(url=URL)
    ingest.run(_ctx(store, job.id))
    meta = read_model(store.paths(job.id).meta, Meta)
    assert (meta.artist, meta.track, meta.album) == ("Demo Band, Guest", "Sing It Loud", "Invented Album")
    assert (meta.clean_artist, meta.clean_title) == ("Demo Band", "Sing It Loud")
    assert fake_ydl.opened == ["https://subs.invalid/pt.vtt"]


@pytest.mark.parametrize(("langs", "hint", "expected"), [
    ([], None, None),
    (["pl", "de"], None, "pl"),
    (["pl", "pt-BR", "en"], None, "pt-BR"),
    (["pt", "en"], "en", "en"),
    (["pt", "en"], "fr", "pt"),
    (["en-GB", "es"], "es", "en-GB"),
])
def test_choose_sub_lang(langs, hint, expected):
    assert ingest.choose_sub_lang(langs, hint) == expected


def test_progress_hook_is_monotonic(store: JobStore):
    job = store.create(url=URL)
    ctx = _ctx(store, job.id)
    seen: list[float] = []
    ctx._emit = lambda e: seen.append(e.progress)
    hook = ingest._Progress(ctx, 0.0, 1.0)
    for f in ("v.mp4", "a.m4a"):
        for b in (0, 30, 60, 90, 100):
            hook({"status": "downloading", "filename": f, "downloaded_bytes": b, "total_bytes": 100})
        hook({"status": "finished", "filename": f, "downloaded_bytes": 100, "total_bytes": 100})
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- real network


_REAL = Path(__file__).resolve().parents[1] / "data" / "jobs"


@pytest.mark.slow
def test_ingest_real_video_without_manual_subs(tmp_path: Path):
    store = JobStore(tmp_path)
    job = store.create(url="https://www.youtube.com/watch?v=f8x_t8wPFcM")
    ingest.run(_ctx(store, job.id))
    p = store.paths(job.id)
    meta = read_model(p.meta, Meta)
    assert meta.duration == pytest.approx(153, abs=1.5)
    assert meta.has_manual_subs == []
    assert not p.lyrics_source("youtube.vtt").exists()
    assert (meta.clean_artist, meta.clean_title) == ("Sacal", "Vão Querer Me Internar")
    assert meta.clean_extra == ["Tropical Paradise Riddim"]
    info = sf.info(str(p.audio))
    assert (info.samplerate, info.channels) == (44100, 2)


@pytest.mark.slow
@pytest.mark.parametrize("video_id", ["Zdh2hot8rjU", "f8x_t8wPFcM"])
def test_titleclean_on_real_info_json(video_id: str):
    """Offline check of the metadata mapping against a real yt-dlp info.json (gitignored data)."""
    import json

    f = _REAL / video_id / "source.info.json"
    if not f.exists():
        pytest.skip("real job data not present")
    info = json.loads(f.read_text())
    from karaoke.lyrics.titleclean import clean

    ct = clean(info["title"], info.get("uploader"), ingest._first_artist(info), info.get("track"))
    expected = {
        "Zdh2hot8rjU": (None, "Sweet Child O Mine", ["Captain Fantastic"], ["en", "pl"]),
        "f8x_t8wPFcM": ("Sacal", "Vão Querer Me Internar", ["Tropical Paradise Riddim"], []),
    }[video_id]
    assert (ct.artist, ct.title, ct.extra) == expected[:3]
    assert sorted(ingest.manual_sub_langs(info)) == expected[3]
    if expected[3]:
        assert ingest.choose_sub_lang(ingest.manual_sub_langs(info), info.get("language")) == "en"
