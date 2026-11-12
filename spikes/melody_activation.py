"""Cache CREPE's full activation (every pitch bin of every frame) for the jobs and for vocadito:
the input of the decoder experiments in spikes/melody_decode.py.

    uv run python spikes/melody_activation.py [--tracker crepe-tiny] [JOB ...]

Writes ``cache/melody.act.<tracker>.npz`` in each job (``act`` as uint8, 255 = 1.0) and
``data/gold/_cache/vocadito_<i>_<tracker>.act.npz``. Imports torch: run it on its own.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

JOBS = Path("data/jobs")
V = Path("data/gold/_sources/vocadito/extracted")
CACHE = Path("data/gold/_cache")


def save(path: Path, wav: Path, model: str) -> None:
    from karaoke.audio import read_mono
    from karaoke.melody.track import crepe_activation

    if path.exists():
        return
    sig, sr = read_mono(wav)
    t0 = time.time()
    act, bin0, hop = crepe_activation(sig, sr, model=model)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, act=np.round(act * 255).astype(np.uint8), bin0=bin0, hop_s=hop)
    dur = len(sig) / sr
    print(f"{path}: {dur:.0f} s of audio in {time.time() - t0:.0f} s (x{(time.time() - t0) / dur:.2f}), "
          f"{path.stat().st_size / 1e6:.1f} MB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="*")
    ap.add_argument("--tracker", default="crepe-tiny")
    ap.add_argument("--no-vocadito", action="store_true")
    args = ap.parse_args()
    model = args.tracker.split("-")[1]
    for d in sorted(JOBS.iterdir()):
        wav = d / "stems" / "vocals.wav"
        if wav.exists() and (not args.jobs or d.name in args.jobs):
            save(d / "cache" / f"melody.act.{args.tracker}.npz", wav, model)
    if not args.no_vocadito and not args.jobs and V.exists():
        for i in range(1, 41):
            save(CACHE / f"vocadito_{i}_{args.tracker}.act.npz", V / f"Audio/vocadito_{i}.wav", model)


if __name__ == "__main__":
    main()
