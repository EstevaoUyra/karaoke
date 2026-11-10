"""Contour → notes for a job: loudness and word timings from the job folder, then ``notes.segment``.
Shared by the melody stage and spikes/eval_melody.py."""

from __future__ import annotations

import numpy as np

from karaoke.store import JobPaths


def frame_db(signal: np.ndarray, sr: int, hop_s: float, n: int) -> np.ndarray:
    """Frame loudness (dBFS, 64 ms windows) on a contour's grid of ``n`` frames."""
    from karaoke.audio import rms_db

    db = rms_db(signal, sr, frame_s=0.064, hop_s=hop_s)
    return np.r_[db, np.full(max(0, n - len(db)), -120.0)][:n]


def job_words(p: JobPaths) -> list[tuple[float, float]] | None:
    from karaoke.models import LyricsFinal
    from karaoke.store import read_model_opt

    lyrics = read_model_opt(p.lyrics_final, LyricsFinal)
    return [(w.start, w.end) for ln in lyrics.lines for w in ln.words] if lyrics else None


def notes_from_contour(p: JobPaths, f0: np.ndarray, conf: np.ndarray, hop_s: float, tracker,
                       signal: tuple[np.ndarray, int] | None = None):
    """(notes, tuning) for a contour of the job's vocal stem; ``tracker`` is the tracker's name
    (its ``notes.profile``) or a ``notes.Profile``."""
    from karaoke.audio import read_mono
    from karaoke.melody.notes import estimate_tuning, hz_to_midi, profile, segment

    sig, sr = signal if signal is not None else read_mono(p.vocals)
    db = frame_db(sig, sr, hop_s, len(f0))
    words = job_words(p)
    midi = hz_to_midi(f0)
    prof = profile(tracker) if isinstance(tracker, str) else tracker
    tuning = round(estimate_tuning(midi, conf, hop_s, db, prof=prof), 3)
    return segment(midi, conf, hop_s, db=db, words=words, tuning=tuning, prof=prof), tuning
