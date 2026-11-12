"""Song structure: sections, a loudness curve and beats, from the mix and the synced lyrics.

Backgrounds use it to change scenes where the song changes. The method is deliberately simple:

1. Loudness: RMS of the mix in dB, pooled to ``ENERGY_HOP_S`` and scaled 0..1 between the song's
   quiet and loud percentiles.
2. Boundaries, from two sources:
   - the lyrics: a silence between lines of ``GAP_S`` or more is an instrumental section;
   - the audio: peaks of a novelty curve (how much timbre and harmony differ between the 16
     beats before and after a beat), for songs sung almost without a pause. The strongest
     peaks become boundaries; weaker ones only split sections longer than ``MAX_SECTION_S``.
   An audio boundary that falls inside a sung line moves to the nearest edge of that line, so a
   scene never changes in the middle of a line.
3. Sections shorter than ``MIN_SECTION_S`` are merged into the neighbour of closest loudness.
4. Labels: barely sung sections are intro / instrumental / outro; sung ones are "chorus" when most
   of their lines are repeated elsewhere in the song and they are at least as loud as the typical
   sung section, "verse" otherwise.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

from karaoke.models import Line, Section, Structure

if TYPE_CHECKING:
    import numpy as np

SR = 22050
HOP = 512
ENERGY_HOP_S = 0.5
GAP_S = 8.0  # a pause between lines this long is its own (instrumental) section
MIN_SECTION_S = 8.0
NOVELTY_BEATS = 16  # the audio is compared this many beats before and after each beat
MIN_GAP_S = 10.0  # an audio boundary keeps at least this distance from its neighbours
MAX_SECTION_S = 36.0  # longer sections are split at their strongest audio change
VOCAL_MIN = 0.2  # sections sung for less than this share are instrumental
REPEAT_RATIO = 0.85  # two lines this similar are the same line
REPEAT_APART = 2  # ... if they are at least this many lines apart
CHORUS_REPEATS = 0.5  # share of a section's lines that must be repeated lines

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm_text(line: Line) -> str:
    text = " ".join(w.text for w in line.words).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(_PUNCT.sub(" ", text).split())


def repeated_lines(lines: list[Line]) -> list[bool]:
    """For each line: is (almost) the same text sung elsewhere in the song, a few lines away?"""
    texts = [norm_text(ln) for ln in lines]
    out = [False] * len(lines)
    for i, a in enumerate(texts):
        if len(a.split()) < 2:
            continue
        for j in range(i + REPEAT_APART, len(texts)):
            b = texts[j]
            if a == b or (abs(len(a) - len(b)) <= 0.3 * len(a)
                          and SequenceMatcher(None, a, b, autojunk=False).ratio() >= REPEAT_RATIO):
                out[i] = out[j] = True
    return out


def lyric_bounds(lines: list[Line], duration: float) -> list[float]:
    """Edges of the long pauses between lines (and before the first / after the last line)."""
    out: list[float] = []
    prev_end = 0.0
    for ln in lines:
        if ln.start - prev_end >= GAP_S:
            out += [prev_end, ln.start]
        prev_end = max(prev_end, ln.end)
    if lines and duration - prev_end >= GAP_S:
        out.append(prev_end)
    return sorted({round(t, 3) for t in out if 0.0 < t < duration})


def snap_to_lines(t: float, lines: list[Line]) -> float:
    """Move a boundary that falls inside a sung line to the nearest edge of that line."""
    for ln in lines:
        if ln.start < t < ln.end:
            return ln.start if t - ln.start <= ln.end - t else ln.end
    return t


def _mean(curve: list[float], hop_s: float, start: float, end: float) -> float:
    a, b = int(start / hop_s), max(int(start / hop_s) + 1, int(round(end / hop_s)))
    part = curve[a:b]
    return sum(part) / len(part) if part else 0.0


def drop_short(bounds: list[float], energy: list[float], hop_s: float, duration: float) -> list[float]:
    """Remove boundaries until no section is shorter than ``MIN_SECTION_S`` (the shortest goes
    first, into the neighbour whose loudness is closest)."""
    edges = [0.0, *bounds, duration]
    while len(edges) > 2:
        spans = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        k = min(range(len(spans)), key=spans.__getitem__)
        if spans[k] >= MIN_SECTION_S:
            break
        e = [_mean(energy, hop_s, edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        left = abs(e[k] - e[k - 1]) if k > 0 else float("inf")
        right = abs(e[k] - e[k + 1]) if k + 1 < len(spans) else float("inf")
        del edges[k if left <= right else k + 1]
    return edges[1:-1]


def vocal_share(lines: list[Line], start: float, end: float) -> float:
    sung = sum(max(0.0, min(ln.end, end) - max(ln.start, start)) for ln in lines)
    return min(1.0, sung / (end - start)) if end > start else 0.0


def label_sections(bounds: list[float], lines: list[Line], energy: list[float], hop_s: float,
                   duration: float) -> list[Section]:
    edges = [0.0, *bounds, duration]
    spans = list(zip(edges[:-1], edges[1:], strict=True))
    rep = repeated_lines(lines)
    levels = [_mean(energy, hop_s, a, b) for a, b in spans]
    shares = [vocal_share(lines, a, b) for a, b in spans]
    sung = sorted(e for e, v in zip(levels, shares, strict=True) if v >= VOCAL_MIN)
    typical = sung[len(sung) // 2] if sung else 0.0
    out: list[Section] = []
    for k, (a, b) in enumerate(spans):
        if shares[k] < VOCAL_MIN:
            label = "intro" if k == 0 else "outro" if k == len(spans) - 1 else "instrumental"
        else:
            # a line belongs to the section its middle falls in
            flags = [r for ln, r in zip(lines, rep, strict=True) if a <= (ln.start + ln.end) / 2 < b]
            repeats = sum(flags) / len(flags) if flags else 0.0
            label = "chorus" if repeats >= CHORUS_REPEATS and levels[k] >= typical - 0.02 else "verse"
        out.append(Section(start=round(a, 3), end=round(b, 3), label=label, energy=round(levels[k], 3),
                           vocal=round(shares[k], 3)))
    return out


# --------------------------------------------------------------------------- audio


def energy_curve(y: np.ndarray, sr: int) -> list[float]:
    """Loudness 0..1 per ``ENERGY_HOP_S``: RMS in dB between the song's 5th and 95th percentiles."""
    import librosa
    import numpy as np

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    per = max(1, int(round(ENERGY_HOP_S * sr / HOP)))
    n = len(rms) // per
    if n == 0:
        return []
    pooled = np.sqrt((rms[: n * per].reshape(n, per) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(pooled, 1e-5))
    lo, hi = np.percentile(db, [5, 95])
    if hi - lo < 1.0:
        return [0.5] * n
    return [round(float(v), 3) for v in np.clip((db - lo) / (hi - lo), 0.0, 1.0)]


def novelty(feats: np.ndarray, width: int = NOVELTY_BEATS) -> np.ndarray:
    """How different the ``width`` beats after each beat are from the ``width`` before it."""
    import numpy as np

    n = feats.shape[1]
    out = np.zeros(n)
    csum = np.concatenate([np.zeros((feats.shape[0], 1)), np.cumsum(feats, axis=1)], axis=1)
    for i in range(width, n - width + 1):
        before = (csum[:, i] - csum[:, i - width]) / width
        after = (csum[:, i + width] - csum[:, i]) / width
        out[i] = float(np.linalg.norm(after - before))
    return out


def audio_candidates(y: np.ndarray, sr: int, beat_frames: np.ndarray) -> list[tuple[float, float]]:
    """(time, strength) of every local peak of the timbre + harmony + loudness novelty."""
    import librosa
    import numpy as np

    if len(beat_frames) < 4 * NOVELTY_BEATS:
        return []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP)
    feats = []
    for f in (mfcc, chroma):
        f = librosa.util.sync(f, beat_frames, aggregate=np.median)
        feats.append((f - f.mean(axis=1, keepdims=True)) / (f.std(axis=1, keepdims=True) + 1e-9))
    nov = novelty(np.vstack(feats))
    # sync() columns are the spans between beats, starting at frame 0
    starts = np.concatenate([[0], beat_frames])
    times = librosa.frames_to_time(starts, sr=sr, hop_length=HOP)
    out = []
    for i in range(1, len(nov) - 1):
        if nov[i] > 0 and nov[i] >= nov[i - 1] and nov[i] > nov[i + 1] and i < len(times):
            out.append((float(times[i]), float(nov[i])))
    return out


def pick_bounds(lyric: list[float], cands: list[tuple[float, float]], lines: list[Line],
                duration: float) -> list[float]:
    """Lyrics boundaries first. Then audio candidates, strongest first: the clearly strong ones
    anywhere they fit, the weaker ones only to break up a section longer than ``MAX_SECTION_S``.
    A candidate inside a sung line moves to the nearest edge of that line."""
    out = sorted(lyric)
    if not cands:
        return out
    strengths = sorted(s for _, s in cands)
    strong = strengths[int(0.75 * (len(strengths) - 1))]
    for t, s in sorted(cands, key=lambda c: -c[1]):
        t = round(snap_to_lines(t, lines), 3)
        edges = [0.0, *out, duration]
        k = max(i for i, e in enumerate(edges[:-1]) if e <= t) if 0.0 < t < duration else None
        if k is None:
            continue
        a, b = edges[k], edges[k + 1]
        if t - a < MIN_GAP_S or b - t < MIN_GAP_S:
            continue
        if s >= strong or b - a > MAX_SECTION_S:
            out = sorted([*out, t])
    return out


def analyze(audio: Path | str, lines: list[Line], duration: float) -> Structure:
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio), sr=SR, mono=True)
    energy = energy_curve(y, sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    bpm = float(np.atleast_1d(tempo)[0])
    beats = [round(float(t), 3) for t in librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)]
    try:
        cands = audio_candidates(y, sr, beat_frames)
    except Exception:  # noqa: BLE001 — the lyrics alone still give sections
        cands = []
    bounds = pick_bounds(lyric_bounds(lines, duration), cands, lines, duration)
    bounds = drop_short(bounds, energy, ENERGY_HOP_S, duration)
    return Structure(
        duration=round(duration, 3),
        tempo_bpm=round(bpm, 1) if np.isfinite(bpm) and bpm > 0 else None,
        beats=beats,
        energy_hop_s=ENERGY_HOP_S,
        energy=energy,
        sections=label_sections(bounds, lines, energy, ENERGY_HOP_S, duration),
        params={"method": 1, "audio_candidates": len(cands)},
    )
