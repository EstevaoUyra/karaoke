"""separate stage. Fast tests mock audio_separator (never imported here); the slow test runs the
real model on the demo job through run_job, which spawns a child process."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from karaoke import audio
from karaoke.config import get_settings
from karaoke.models import Quality, StageName, StageStatus
from karaoke.pipeline import StageContext, run_job
from karaoke.stages import separate

SR = 44100


class FakeSeparator:
    """Writes stems like audio-separator: '<base>_(<Stem>)_<model>.wav' into its output dir."""

    instances: list[FakeSeparator] = []

    def __init__(self, out_dir: Path, stems: dict[str, float], fail: bool = False, delay: float = 0.0):
        self.out_dir, self.stems, self.fail, self.delay = out_dir, stems, fail, delay
        self.model = None
        FakeSeparator.instances.append(self)

    def load_model(self, model_filename):
        self.model = model_filename

    def separate(self, audio_path):
        time.sleep(self.delay)
        names = []
        n = SR  # 1 s
        for stem, value in self.stems.items():
            name = f"{Path(audio_path).stem}_({stem})_{Path(self.model).stem}.wav"
            sf.write(str(self.out_dir / name), np.full((n, 2), value, dtype=np.float32), SR, subtype="FLOAT")
            names.append(name)
        if self.fail:
            raise RuntimeError("boom")
        return names


@pytest.fixture
def ctx(demo, store):
    events = []
    job = store.load(demo.id)
    c = StageContext(job, demo, store, get_settings(), StageName.separate, logging.getLogger("test.separate"),
                     _emit=events.append)
    c.events = events  # type: ignore[attr-defined]
    return c


def _install(monkeypatch, **kw):
    FakeSeparator.instances.clear()
    monkeypatch.setattr(separate, "_make_separator", lambda ctx, out_dir: FakeSeparator(out_dir, **kw))


def _value(path: Path) -> float:
    data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return float(data.mean())


def test_two_stem_model_renames_outputs(ctx, monkeypatch):
    _install(monkeypatch, stems={"Vocals": 0.25, "Instrumental": -0.5})
    ctx.paths.vocals.unlink()
    ctx.paths.instrumental.unlink()
    separate.run(ctx)
    assert FakeSeparator.instances[0].model == get_settings().separator_model
    assert _value(ctx.paths.vocals) == pytest.approx(0.25, abs=1e-4)
    assert _value(ctx.paths.instrumental) == pytest.approx(-0.5, abs=1e-4)
    assert sorted(f.name for f in ctx.paths.stems_dir.iterdir()) == ["instrumental.wav", "vocals.wav"]
    assert ctx.job.stages[StageName.separate].progress == 1.0


def test_demucs_instrumental_is_the_sum_of_other_stems(ctx, monkeypatch):
    _install(monkeypatch, stems={"Drums": 0.1, "Bass": 0.2, "Other": 0.05, "Vocals": 0.3})
    ctx.job.settings.separator_model = "htdemucs_ft.yaml"
    separate.run(ctx)
    assert FakeSeparator.instances[0].model == "htdemucs_ft.yaml"
    assert _value(ctx.paths.vocals) == pytest.approx(0.3, abs=1e-4)
    assert _value(ctx.paths.instrumental) == pytest.approx(0.35, abs=1e-3)
    assert sorted(f.name for f in ctx.paths.stems_dir.iterdir()) == ["instrumental.wav", "vocals.wav"]


def test_sum_never_clips(tmp_path):
    parts = []
    for i, v in enumerate((0.8, 0.7)):
        f = tmp_path / f"{i}.wav"
        sf.write(str(f), np.full((100, 2), v, dtype=np.float32), SR, subtype="FLOAT")
        parts.append(f)
    separate.sum_stems(parts, tmp_path / "sum.wav")
    data, _ = sf.read(str(tmp_path / "sum.wav"))
    assert np.abs(data).max() <= 1.0


def test_model_choice(ctx, monkeypatch):
    s = get_settings()
    assert separate.model_for(ctx.job, s) == s.separator_model
    ctx.job.settings.quality = Quality.hq
    assert separate.model_for(ctx.job, s) == s.separator_model_hq
    _install(monkeypatch, stems={"Vocals": 0.1, "Instrumental": 0.1})
    separate.run(ctx)
    assert FakeSeparator.instances[0].model == s.separator_model_hq
    ctx.job.settings.separator_model = "UVR-MDX-NET-Inst_HQ_3.onnx"
    assert separate.model_for(ctx.job, s) == "UVR-MDX-NET-Inst_HQ_3.onnx"


def test_crash_leaves_previous_outputs_untouched(ctx, monkeypatch):
    before = ctx.paths.vocals.read_bytes()
    _install(monkeypatch, stems={"Vocals": 0.9, "Instrumental": 0.9}, fail=True)
    with pytest.raises(RuntimeError, match="boom"):
        separate.run(ctx)
    assert ctx.paths.vocals.read_bytes() == before
    assert sorted(f.name for f in ctx.paths.stems_dir.iterdir()) == ["instrumental.wav", "vocals.wav"]


def test_missing_vocal_stem_is_an_error(ctx, monkeypatch):
    _install(monkeypatch, stems={"Drums": 0.1, "Bass": 0.1})
    with pytest.raises(RuntimeError, match="no vocal stem"):
        separate.run(ctx)
    assert not any(f.name.startswith(".separate-") for f in ctx.paths.stems_dir.iterdir())


def test_ticker_reports_estimated_progress(ctx, monkeypatch):
    monkeypatch.setattr(separate, "TICK_S", 0.05)
    _install(monkeypatch, stems={"Vocals": 0.1, "Instrumental": 0.1}, delay=0.4)
    separate.run(ctx)
    ticks = [e.progress for e in ctx.events if "separating" in e.message]
    assert len(ticks) >= 3
    assert all(0 < p <= separate.PROGRESS_CAP for p in ticks)
    assert ticks == sorted(ticks)


def test_helpers():
    assert separate.stem_name("audio_(Vocals)_Kim_Vocal_2.wav") == "vocals"
    assert separate.stem_name("audio_(No Vocals)_x.wav") == "no vocals"
    assert separate.expected_rtf("Kim_Vocal_2.onnx") == separate.RTF_MDX
    assert separate.expected_rtf("model_bs_roformer_ep_317_sdr_12.9755.ckpt") == separate.RTF_HEAVY
    assert separate.estimate_progress(50, 100, 1.0) == pytest.approx(0.5)
    assert separate.estimate_progress(500, 100, 1.0) == separate.PROGRESS_CAP
    assert separate.estimate_progress(10, 0, 1.0) == 0.0


@pytest.mark.slow
def test_separate_real_model_on_demo(demo, store):
    demo.vocals.unlink()
    demo.instrumental.unlink()
    job = run_job(demo.id, only=[StageName.separate], store=store)
    assert job.stages[StageName.separate].status == StageStatus.done
    assert audio.duration(demo.vocals) == pytest.approx(audio.duration(demo.audio), abs=0.05)
    assert audio.duration(demo.instrumental) == pytest.approx(audio.duration(demo.audio), abs=0.05)
    assert not any(f.name.startswith(".separate-") for f in demo.stems_dir.iterdir())
