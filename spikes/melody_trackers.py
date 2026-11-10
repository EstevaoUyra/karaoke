"""Run one F0 tracker on a job's vocal stem and cache its contour, for spikes/eval_melody.py.

    uv run python spikes/melody_trackers.py JOB TRACKER [--hop 0.01]

TRACKER is ``pyin`` (librosa, the original melody stage), ``crepe-full`` or ``crepe-tiny``
(torchcrepe with viterbi decoding; periodicity is the confidence). The contour goes to
``data/jobs/JOB/cache/f0_<tracker>.npz`` (f0 in Hz with NaN where unvoiced, conf 0..1, hop_s,
runtime_s). Torch trackers and ctranslate2 must not share a process: run this on its own.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from karaoke.audio import read_mono
from karaoke.store import JobStore


def track(signal: np.ndarray, sr: int, tracker: str, hop_s: float) -> tuple[np.ndarray, np.ndarray, float]:
    if tracker == "pyin":
        from karaoke.melody.track import pyin_track

        return pyin_track(signal, sr)
    from karaoke.melody.track import crepe_track

    return crepe_track(signal, sr, model=tracker.split("-", 1)[1], hop_s=hop_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("tracker", choices=["pyin", "crepe-full", "crepe-tiny"])
    ap.add_argument("--hop", type=float, default=0.01)
    a = ap.parse_args()
    p = JobStore().paths(a.job)
    signal, sr = read_mono(p.vocals)
    t0 = time.monotonic()
    f0, conf, hop = track(signal, sr, a.tracker, a.hop)
    dt = time.monotonic() - t0
    out = p.root / "cache" / f"f0_{a.tracker}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, f0=f0, conf=conf, hop_s=hop, runtime_s=dt)
    dur = len(signal) / sr
    print(f"{a.job} {a.tracker}: {dt:.1f} s for {dur:.0f} s of audio (RTF {dt / dur:.2f}), "
          f"{np.isfinite(f0).mean() * 100:.0f}% frames voiced -> {out}")


if __name__ == "__main__":
    main()
