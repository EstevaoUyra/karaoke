"""Stage 2 — separate. Owner: workstream B.

Input:  ``audio.wav``.
Output: ``stems/vocals.wav``, ``stems/instrumental.wav`` (44.1 kHz stereo).

Uses ``audio_separator.separator.Separator`` with
``model_file_dir=ctx.settings.separator_models_dir``. The model is
``job.settings.separator_model`` or, by default, ``settings.separator_model`` (fast) or
``settings.separator_model_hq`` (quality == "hq"). The library's output files are renamed to the
contract names. Demucs-type models give 4+ stems: instrumental = the sum of every non-vocal stem.

The library writes into a temp dir inside ``stems/``; outputs are moved into place only when
both exist, so a crash never leaves half-written stems. Stem levels are kept as in the mix
(no per-stem amplification), so the sum of demucs stems is the true instrumental.
audio-separator has no progress callback: a ticker thread estimates progress from elapsed time
over (duration × expected real-time factor), capped at 0.95.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

from karaoke.pipeline import StageContext

# Expected seconds of compute per second of audio on the 4-core Intel CPU (docs/benchmarks.md).
RTF_MDX = 1.0  # .onnx (MDX-Net, e.g. Kim_Vocal_2)
RTF_HEAVY = 10.0  # Roformer / Demucs / everything else
PROGRESS_CAP = 0.95
TICK_S = 2.0

_STEM = re.compile(r"\(([^)]+)\)")
_VOCAL_NAMES = {"vocals", "vocal"}
_INSTRUMENTAL_NAMES = {"instrumental", "inst", "no vocals", "no_vocals", "novocals", "accompaniment"}


def model_for(job, settings) -> str:
    """Same rule as the stage's cache key in karaoke/pipeline.py."""
    if job.settings.separator_model:
        return job.settings.separator_model
    return settings.separator_model_hq if job.settings.quality == "hq" else settings.separator_model


def expected_rtf(model: str) -> float:
    return RTF_MDX if model.lower().endswith(".onnx") else RTF_HEAVY


def estimate_progress(elapsed: float, duration: float, rtf: float, cap: float = PROGRESS_CAP) -> float:
    if duration <= 0 or rtf <= 0:
        return 0.0
    return min(cap, max(0.0, elapsed / (duration * rtf)))


def stem_name(path: Path | str) -> str:
    """'audio_(Vocals)_Kim_Vocal_2.wav' -> 'vocals'."""
    m = _STEM.search(Path(path).name)
    return (m.group(1) if m else Path(path).stem).strip().lower()


def classify_stems(files: list[Path]) -> tuple[Path, list[Path]]:
    """(vocals file, files that make up the instrumental)."""
    by_name = {stem_name(f): f for f in files}
    vocals = next((f for n, f in by_name.items() if n in _VOCAL_NAMES), None)
    if vocals is None:
        raise RuntimeError(f"separator gave no vocal stem: {sorted(by_name)}")
    inst = [f for n, f in by_name.items() if n in _INSTRUMENTAL_NAMES]
    if inst:
        return vocals, inst[:1]
    rest = [f for f in files if f != vocals]
    if not rest:
        raise RuntimeError(f"separator gave no instrumental stem: {sorted(by_name)}")
    return vocals, rest


def sum_stems(parts: list[Path], out: Path) -> None:
    """Write the sample-wise sum of ``parts`` to ``out`` (scaled down only if it would clip)."""
    import numpy as np
    import soundfile as sf

    total = None
    sr = None
    subtype = None
    for f in parts:
        data, rate = sf.read(str(f), dtype="float32", always_2d=True)
        if total is None:
            total, sr, subtype = data.astype(np.float64), rate, sf.info(str(f)).subtype
            continue
        if rate != sr:
            raise RuntimeError(f"stem sample rates differ: {f.name} {rate} vs {sr}")
        n = min(len(total), len(data))
        total = total[:n] + data[:n]
    peak = float(np.abs(total).max()) if total.size else 0.0
    if peak > 0.999:
        total *= 0.999 / peak
    sf.write(str(out), total.astype(np.float32), sr, subtype=subtype or "PCM_16")


class _Ticker:
    def __init__(self, ctx: StageContext, duration: float, rtf: float, label: str):
        self.ctx, self.duration, self.rtf, self.label = ctx, duration, rtf, label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="separate-progress", daemon=True)
        self._t0 = 0.0

    def _loop(self) -> None:
        while not self._stop.wait(TICK_S):
            el = time.monotonic() - self._t0
            self.ctx.progress(estimate_progress(el, self.duration, self.rtf), f"{self.label} ({el:.0f} s)")

    def __enter__(self) -> _Ticker:
        self._t0 = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def _make_separator(ctx: StageContext, out_dir: Path):
    from audio_separator.separator import Separator

    return Separator(
        log_level=logging.WARNING,
        model_file_dir=str(ctx.settings.separator_models_dir),
        output_dir=str(out_dir),
        output_format="WAV",
        # Keep mix levels (needed for the demucs sum): only scale a stem down if it would clip,
        # and never amplify (the library requires a value > 0; 1e-6 is never reached by music).
        normalization_threshold=1.0,
        amplification_threshold=1e-6,
    )


def run(ctx: StageContext) -> None:
    from karaoke import audio

    p = ctx.paths
    if not p.audio.exists():
        raise FileNotFoundError(f"{p.rel(p.audio)} is missing: run the ingest stage first")
    model = model_for(ctx.job, ctx.settings)
    dur = audio.duration(p.audio)
    rtf = expected_rtf(model)
    p.stems_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".separate-", dir=p.stems_dir))
    try:
        ctx.progress(0.0, f"loading {model}")
        sep = _make_separator(ctx, tmp)
        sep.load_model(model_filename=model)
        t0 = time.monotonic()
        with _Ticker(ctx, dur, rtf, f"separating with {model}"):
            names = sep.separate(str(p.audio))
        took = time.monotonic() - t0
        files = [tmp / Path(n).name for n in names]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise RuntimeError(f"separator outputs not found: {missing}")
        ctx.log.info("separated %.1f s of audio with %s in %.1f s (RTF %.2f): %s", dur, model, took,
                     took / dur if dur else 0.0, [f.name for f in files])
        vocals, inst_parts = classify_stems(files)
        tmp_vocals, tmp_inst = tmp / "vocals.wav", tmp / "instrumental.wav"
        vocals.rename(tmp_vocals)
        if len(inst_parts) == 1:
            inst_parts[0].rename(tmp_inst)
        else:
            ctx.progress(PROGRESS_CAP, f"summing {len(inst_parts)} stems into the instrumental")
            sum_stems(inst_parts, tmp_inst)
        # Both files are complete: move into place (same filesystem, atomic per file).
        tmp_inst.replace(p.instrumental)
        tmp_vocals.replace(p.vocals)
        ctx.progress(1.0, f"{model}: {took:.0f} s (RTF {took / dur if dur else 0:.2f})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
