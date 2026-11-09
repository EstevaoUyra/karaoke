"""The song's beat, so the countdown dots go out in time with the music."""

from __future__ import annotations

from pathlib import Path

MIN_BEAT_S = 0.55  # faster beats are doubled (count every other beat) ...
MAX_BEAT_S = 1.2  # ... slower ones halved


def fold_beat(beat_s: float) -> float:
    while beat_s < MIN_BEAT_S:
        beat_s *= 2
    while beat_s > MAX_BEAT_S:
        beat_s /= 2
    return beat_s


def beat_period(audio: Path | str, max_s: float = 120.0) -> float | None:
    """Seconds per (counted) beat of ``audio``, or None if no tempo is found."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio), sr=22050, mono=True, duration=max_s)
    if len(y) < sr * 5:
        return None
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(bpm) or bpm <= 0:
        return None
    return round(fold_beat(60.0 / bpm), 3)
