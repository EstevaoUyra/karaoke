from __future__ import annotations

import json

import make_demo
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from karaoke import cli
from karaoke.config import get_settings
from karaoke.models import Background, LyricsFinal, StageName, StageStatus, Transcript
from karaoke.pipeline import ORDER, STAGE_BY_NAME, is_cached, run_job, select_stages
from karaoke.store import JobStore, clean_youtube_url, job_id_for_url, read_model
from karaoke.validate import validate_job


def test_youtube_ids_and_playlist_stripping():
    url = "https://www.youtube.com/watch?v=Zdh2hot8rjU&list=RDZdh2hot8rjU"
    assert job_id_for_url(url) == "Zdh2hot8rjU"
    assert clean_youtube_url(url) == "https://www.youtube.com/watch?v=Zdh2hot8rjU"
    assert job_id_for_url("https://youtu.be/f8x_t8wPFcM?si=x") == "f8x_t8wPFcM"


def test_demo_fixture_is_valid(demo):
    assert validate_job(demo) == []
    lf = read_model(demo.lyrics_final, LyricsFinal)
    assert [ln.section for ln in lf.lines][:3] == ["verse", "verse", "chorus"]
    t = read_model(demo.transcript, Transcript)
    heard = " ".join(w.text.strip() for s in t.segments for w in s.words)
    assert "widow" in heard and "yeah" in heard  # planted errors are there


def test_validation_catches_problems(demo_no_audio):
    p = demo_no_audio
    lf = read_model(p.lyrics_final, LyricsFinal)
    lf.lines[1].words[0].start = 99.0
    p.lyrics_final.write_text(lf.model_dump_json())
    bg = read_model(p.background, Background)
    bg.scenes[0].end = 10.0
    p.background.write_text(bg.model_dump_json())
    msgs = [str(i) for i in validate_job(p)]
    assert any("outside its line bounds" in m for m in msgs)
    assert any("scenes end at 10.00" in m for m in msgs)


def test_schema_rejects_unknown_fields(demo_no_audio):
    raw = json.loads(demo_no_audio.background.read_text())
    raw["scenes"][0]["colour"] = "red"
    with pytest.raises(ValidationError):
        Background.model_validate(raw)


def test_create_job_is_idempotent(store: JobStore):
    a = store.create(url="https://www.youtube.com/watch?v=Zdh2hot8rjU&list=RD")
    b = store.create(url="https://youtu.be/Zdh2hot8rjU")
    assert a.id == b.id == "Zdh2hot8rjU"
    assert [j.id for j in store.list()] == ["Zdh2hot8rjU"]
    assert set(a.stages) == set(StageName)


def test_select_stages():
    assert select_stages() == ORDER
    assert select_stages(StageName.align) == [StageName.align, StageName.melody, StageName.structure,
                                              StageName.backgrounds,
                                              StageName.render]
    assert select_stages(only=[StageName.render, StageName.ingest]) == [StageName.ingest, StageName.render]


def test_old_job_json_gets_new_stages(store: JobStore):
    """A job.json written before the melody stage existed still loads, with melody pending."""
    job = store.create(job_id="old")
    raw = json.loads(store.paths("old").job.read_text())
    del raw["stages"]["melody"]
    store.paths("old").job.write_text(json.dumps(raw))
    job = store.load("old")
    assert list(job.stages) == list(StageName)
    assert job.stages[StageName.melody].status == StageStatus.pending


def test_runner_caches_and_reruns_on_input_change(demo_no_audio, monkeypatch):
    """Use a fake `align` stage to exercise the runner's cache logic."""
    calls = []

    def fake_run(self, ctx):
        calls.append(ctx.stage)
        ctx.progress(0.5, "halfway")
        ctx.paths.lyrics_final.write_text(ctx.paths.lyrics_final.read_text())

    monkeypatch.setattr(type(STAGE_BY_NAME[StageName.align]), "run", fake_run)
    store = JobStore(demo_no_audio.root.parent)
    run_job("demo", only=[StageName.align], store=store)
    job = store.load("demo")
    assert job.stages[StageName.align].status == StageStatus.done
    spec, settings = STAGE_BY_NAME[StageName.align], get_settings()
    assert is_cached(spec, demo_no_audio, job, settings)
    # editing an input invalidates the cache
    demo_no_audio.lyrics_chosen_text.write_text("changed\n")
    assert not is_cached(spec, demo_no_audio, store.load("demo"), settings)
    assert calls == [StageName.align]


def test_runner_records_failures(demo_no_audio, monkeypatch):
    import karaoke.stages.ingest as ingest

    def boom(ctx):
        raise NotImplementedError("planted failure")

    monkeypatch.setattr(ingest, "run", boom)  # in-process stage: no network
    store = JobStore(demo_no_audio.root.parent)
    with pytest.raises(NotImplementedError):
        run_job("demo", only=[StageName.ingest], store=store)
    st = store.load("demo").stages[StageName.ingest]
    assert st.status == StageStatus.failed and "NotImplementedError" in st.error
    assert demo_no_audio.log("ingest").read_text().count("failed") == 1


def test_isolated_stage_runs_in_child_and_reports_errors(demo_no_audio):
    from karaoke.pipeline import IsolatedStageError

    assert STAGE_BY_NAME[StageName.separate].isolated and STAGE_BY_NAME[StageName.transcribe].isolated
    store = JobStore(demo_no_audio.root.parent)
    # demo_no_audio has no audio.wav, so the real separate stage fails inside the child process
    with pytest.raises(IsolatedStageError):
        run_job("demo", only=[StageName.separate], store=store)
    assert store.load("demo").stages[StageName.separate].status == StageStatus.failed
    assert "failed in child process" in demo_no_audio.log("separate").read_text()


def test_cli_status_validate_schema(demo_no_audio, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_store", lambda: JobStore(demo_no_audio.root.parent))
    r = CliRunner()
    assert r.invoke(cli.app, ["status", "demo"]).exit_code == 0
    res = r.invoke(cli.app, ["validate", "demo"])
    assert res.exit_code == 0 and "0 error(s)" in res.output
    assert r.invoke(cli.app, ["schema", "--out", str(tmp_path / "schema")]).exit_code == 0
    assert (tmp_path / "schema" / "lyrics_final.schema.json").exists()


def test_demo_audio_files(demo):
    import soundfile as sf

    for p in (demo.audio, demo.vocals, demo.instrumental):
        info = sf.info(p)
        assert info.samplerate == 44100 and info.channels == 2
        assert abs(info.duration - make_demo.DURATION) < 0.01
