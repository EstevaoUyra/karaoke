"""The web API against the synthetic demo job (offline, no models)."""

from __future__ import annotations

import asyncio
import io
import shutil
import sys
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from karaoke.api.app import create_app
from karaoke.api.queue import JobQueue, RunRequest
from karaoke.api.sse import EventHub
from karaoke.api.watch import classify
from karaoke.config import get_settings
from karaoke.models import LyricsFinal, StageName, StageStatus, WordOrigin
from karaoke.store import read_model


class Recorder:
    def __init__(self):
        self.events: list[tuple[str, object]] = []

    def __call__(self, type_: str, data: object) -> None:
        self.events.append((type_, data))

    def of(self, type_: str) -> list:
        return [d for t, d in self.events if t == type_]


@pytest.fixture
def client(store, demo):
    app = create_app(store, watch=False, web_dist=None)
    with TestClient(app) as c:
        yield c


def _lyrics(client, job="demo") -> dict:
    r = client.get(f"/api/jobs/{job}/lyrics")
    assert r.status_code == 200
    return r.json()


# --------------------------------------------------------------------------- jobs


def test_list_and_get_job(client):
    jobs = client.get("/api/jobs").json()
    assert [j["id"] for j in jobs] == ["demo"]
    assert jobs[0]["title"] == "Sing It Loud" and jobs[0]["artist"] == "Demo Band"
    assert set(jobs[0]["stages"]) == {s.value for s in StageName}

    d = client.get("/api/jobs/demo").json()
    assert d["meta"]["duration"] == 24.0
    assert d["files"]["instrumental"]["size"] > 0
    assert d["files"]["preview_video"] is None
    assert d["issues"] == []


def test_unknown_or_bad_job_ids_are_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/nothing-here").status_code == 404


def test_create_job_with_pasted_lyrics_does_not_run_when_asked(client, store):
    r = client.post("/api/jobs", json={"url": "https://www.youtube.com/watch?v=Zdh2hot8rjU&list=RD",
                                       "lyrics": "line one\nline two", "run": False})
    assert r.status_code == 201
    assert r.json()["job"]["id"] == "Zdh2hot8rjU" and r.json()["run"] is None
    p = store.paths("Zdh2hot8rjU")
    assert p.lyrics_source("user.txt").read_text() == "line one\nline two\n"
    assert store.load("Zdh2hot8rjU").url == "https://www.youtube.com/watch?v=Zdh2hot8rjU"
    assert client.post("/api/jobs", json={"url": "not a link"}).status_code == 422


def test_user_lyrics_roundtrip(client, demo):
    assert client.put("/api/jobs/demo/lyrics/user", json={"text": "hello\nworld"}).json()["saved"]
    files = {f["name"]: f["text"] for f in client.get("/api/jobs/demo/lyrics/sources").json()["files"]}
    assert files["user.txt"] == "hello\nworld\n" and "lrclib.lrc" in files
    client.put("/api/jobs/demo/lyrics/user", json={"text": "  "})
    assert not demo.lyrics_source("user.txt").exists()


def test_letras_pointer(client, demo):
    assert client.get("/api/jobs/demo/lyrics/sources").json()["letras"] is None
    demo.lyrics_source("letras.json").write_text(
        '{"url": "https://www.letras.mus.br/demo-band/sing-it-loud/", "title": "Sing It Loud", "artist": "Demo Band"}')
    assert client.get("/api/jobs/demo/lyrics/sources").json()["letras"] == {
        "url": "https://www.letras.mus.br/demo-band/sing-it-loud/", "title": "Sing It Loud", "artist": "Demo Band"}
    demo.lyrics_source("letras.json").write_text('{"url": "javascript:alert(1)"}')
    assert client.get("/api/jobs/demo/lyrics/sources").json()["letras"] is None
    r = client.put("/api/jobs/demo/lyrics/user", json={"text": "my pasted lyrics", "run": False})
    assert r.json() == {"saved": True, "run": None}


def test_logs_tail(client, demo):
    demo.log("align").write_text("".join(f"line {i}\n" for i in range(500)))
    r = client.get("/api/jobs/demo/logs/align?lines=3").json()
    assert r["text"] == "line 497\nline 498\nline 499"
    assert client.get("/api/jobs/demo/logs/render").json()["text"] == ""


# --------------------------------------------------------------------------- media


def test_media_supports_range_requests(client, demo):
    size = demo.instrumental.stat().st_size
    full = client.get("/api/jobs/demo/files/stems/instrumental.wav")
    assert full.status_code == 200 and full.headers["content-type"] == "audio/x-wav"
    assert full.headers.get("accept-ranges") == "bytes" and len(full.content) == size
    part = client.get("/api/jobs/demo/files/stems/instrumental.wav", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206
    assert part.content == demo.instrumental.read_bytes()[100:200]
    assert part.headers["content-range"] == f"bytes 100-199/{size}"
    dl = client.get("/api/jobs/demo/files/stems/vocals.wav?download=1")
    assert "attachment" in dl.headers["content-disposition"]


def test_media_refuses_other_files(client):
    for path in ("job.json", "lyrics.final.json", "stems/../job.json", "logs/align.log", "stems/missing.wav"):
        assert client.get(f"/api/jobs/demo/files/{path}").status_code == 404, path


# --------------------------------------------------------------------------- lyrics


def test_save_lyrics_marks_edits_manual_and_validates(client, demo):
    lf = _lyrics(client)
    w = lf["lines"][0]["words"][5]
    w["text"] = "windows,"  # edited by the user...
    lf["lines"][1]["words"][0]["start"] += 0.05  # ...and a nudge, both left with origin "matched"
    r = client.put(f"/api/jobs/demo/lyrics?base={lf['updated_at']}", json=lf)
    assert r.status_code == 200, r.text
    assert r.json()["remarked"] == 2
    saved = read_model(demo.lyrics_final, LyricsFinal)
    assert saved.lines[0].words[5].text == "windows,"
    assert saved.lines[0].words[5].origin == WordOrigin.manual
    assert saved.lines[1].words[0].origin == WordOrigin.manual
    assert saved.lines[1].words[1].origin == WordOrigin.matched  # untouched words keep their origin
    assert sum(w.origin == WordOrigin.manual for ln in saved.lines for w in ln.words) == 2


def test_save_lyrics_rejects_invalid(client, demo):
    lf = _lyrics(client)
    lf["lines"][0]["words"][0]["start"] = 50.0  # outside its line
    r = client.put("/api/jobs/demo/lyrics", json=lf)
    assert r.status_code == 422
    assert any("outside its line bounds" in i["message"] for i in r.json()["detail"]["issues"])
    lf["lines"][0]["words"][0]["bogus"] = 1  # schema: extra fields are forbidden
    assert client.put("/api/jobs/demo/lyrics", json=lf).status_code == 422
    assert read_model(demo.lyrics_final, LyricsFinal).lines[0].words[0].start == 2.0


def test_save_lyrics_conflict_when_changed_on_disk(client, demo):
    lf = _lyrics(client)
    other = read_model(demo.lyrics_final, LyricsFinal)  # e.g. a Claude session saves first
    other.updated_at = other.updated_at.replace(year=2030)
    demo.lyrics_final.write_text(other.model_dump_json())
    r = client.put(f"/api/jobs/demo/lyrics?base={lf['updated_at']}", json=lf)
    assert r.status_code == 409


def test_validate_endpoint(client):
    lf = _lyrics(client)
    lf["lines"][2]["start"] = 1.0  # starts before the previous line
    issues = client.post("/api/jobs/demo/lyrics/validate", json=lf).json()["issues"]
    assert any("out of order" in i["message"] for i in issues)


# --------------------------------------------------------------------------- background


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 36), (200, 40, 90)).save(buf, "PNG")
    return buf.getvalue()


def test_background_upload_and_save(client, demo):
    bg = client.get("/api/jobs/demo/background").json()
    assert bg["scenes"][0]["source"]["type"] == "gradient"
    r = client.post("/api/jobs/demo/background/scenes/s1/image", files={"file": ("photo.png", _png(), "image/png")})
    assert r.status_code == 200, r.text
    scene = r.json()["background"]["scenes"][0]
    assert scene["source"]["type"] == "image" and scene["source"]["origin"] == "upload" and scene["locked"]
    assert (demo.root / scene["source"]["path"]).exists()
    assert client.get(f"/api/jobs/demo/files/{scene['source']['path']}").status_code == 200

    bad = client.post("/api/jobs/demo/background/scenes/s1/image", files={"file": ("x.png", b"nope", "image/png")})
    assert bad.status_code == 422

    bg = r.json()["background"]
    bg["scenes"][0]["dim"] = 0.4
    assert client.put("/api/jobs/demo/background", json=bg).status_code == 200
    bg["scenes"][0]["end"] = 10.0  # leaves the rest of the song uncovered
    assert client.put("/api/jobs/demo/background", json=bg).status_code == 422


# --------------------------------------------------------------------------- settings and queue


def test_render_saves_settings_and_queues_render_only(store, demo):
    rec = Recorder()
    q = JobQueue(store, rec)  # never started: requests stay queued
    app = create_app(store, queue=q, watch=False, web_dist=None)
    q.start = lambda: None  # type: ignore[method-assign]
    with TestClient(app) as c:
        settings = c.get("/api/jobs/demo/settings").json()
        settings["render"]["preset"] = "preview"
        settings["render"]["guide_vocals_db"] = -18
        r = c.post("/api/jobs/demo/render", json={"render": settings["render"]})
        assert r.status_code == 202 and r.json()["only"] == ["render"]
        assert store.load("demo").settings.render.preset == "preview"
        assert c.get("/api/jobs").json()[0]["queue"] == "queued"
        # a second request for the same job replaces the queued one
        c.post("/api/jobs/demo/run", json={"from_stage": "align"})
        snap = c.get("/api/queue").json()
        assert len(snap["queued"]) == 1 and snap["queued"][0]["from_stage"] == "align"
        assert c.post("/api/jobs/demo/cancel").json()["cancelled"]
        assert c.get("/api/queue").json()["queued"] == []
    assert rec.of("queue")


def test_queue_runs_the_real_worker(store, demo):
    """End to end through the child process, on a cheap stage (backgrounds keeps the existing file)."""
    rec = Recorder()
    q = JobQueue(store, rec)
    q.start()
    try:
        q.submit(RunRequest("demo", only=[StageName.backgrounds]))
        assert q.wait_idle(60)
    finally:
        q.stop()
    hist = q.snapshot()["history"]
    assert hist[0]["state"] == "done", hist
    statuses = [(e["stage"], e["status"]) for e in rec.of("stage")]
    assert ("backgrounds", "running") in statuses and ("backgrounds", "done") in statuses
    assert store.load("demo").stages[StageName.backgrounds].status == StageStatus.done


def test_cancel_kills_the_worker_and_marks_the_stage(store, demo):
    job = store.load("demo")
    job.stages[StageName.render].status = StageStatus.running  # what a killed worker leaves behind
    store.save(job)
    rec = Recorder()
    q = JobQueue(store, rec, command=lambda jobs_dir, req: [sys.executable, "-c", "import time; time.sleep(60)"])
    q.start()
    try:
        q.submit(RunRequest("demo", only=[StageName.render]))
        for _ in range(100):
            if q.state_for("demo") == "running" and q._proc is not None:
                break
            time.sleep(0.05)
        t0 = time.monotonic()
        assert q.cancel("demo")
        assert q.wait_idle(10)
        assert time.monotonic() - t0 < 5
    finally:
        q.stop()
    assert q.snapshot()["history"][0]["state"] == "cancelled"
    st = store.load("demo").stages[StageName.render]
    assert st.status == StageStatus.failed and st.error == "cancelled"


@pytest.mark.skipif(not shutil.which(get_settings().ffmpeg_bin), reason="needs ffmpeg")
def test_upload_audio_creates_a_job_past_ingest(store, demo):
    q = JobQueue(store, Recorder())
    q.start = lambda: None  # type: ignore[method-assign]
    app = create_app(store, queue=q, watch=False, web_dist=None)
    with TestClient(app) as c:
        data = demo.vocals.read_bytes()
        r = c.post("/api/jobs/upload", files={"file": ("my song.wav", data, "audio/wav")},
                   data={"title": "My Song", "artist": "Me", "lyrics": "la la la", "run": "true"})
        assert r.status_code == 201, r.text
        job_id = r.json()["job"]["id"]
        assert job_id.startswith("upload-") and r.json()["job"]["title"] == "My Song"
    p = store.paths(job_id)
    assert p.audio.exists() and p.lyrics_source("user.txt").exists()
    from karaoke.pipeline import STAGE_BY_NAME, is_cached

    job = store.load(job_id)
    assert is_cached(STAGE_BY_NAME[StageName.ingest], p, job, get_settings())


# --------------------------------------------------------------------------- events and watcher


def test_hub_fans_out_across_threads():
    hub = EventHub()

    async def main():
        sub = hub.subscribe()
        import threading

        threading.Thread(target=hub.publish, args=("stage", {"x": 1})).start()
        msg = await asyncio.wait_for(sub[1].get(), 2)
        hub.unsubscribe(sub)
        return msg

    assert asyncio.run(main()) == {"event": "stage", "data": {"x": 1}}


def test_watch_classify(tmp_path):
    jobs = tmp_path / "jobs"
    files, created = classify(jobs, [
        jobs / "demo" / "lyrics.final.json",
        jobs / "demo" / ".lyrics.final.json.abc.tmp",
        jobs / "demo" / "output" / ".preview.mp4.part",
        jobs / "demo" / "job.json",
        jobs / "new1",
        tmp_path / "elsewhere.txt",
    ])
    assert files == {"demo": {"lyrics.final.json", "job.json"}}
    assert created == {"demo", "new1"}
