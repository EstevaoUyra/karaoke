"""Word-constrained syllable timing and pitches.

Timing uses an ordered dynamic program on a 30 ms grid: vowel/character length supplies a
weak duration prior; a voicing dip or a pitch boundary supplies an acoustic cut cue. A second
pitch pass forbids rests inside the resulting spans, and notes outside them. This two-pass
approximation keeps the whole-song interval context without a song-sized syllable x pitch DP.
"""
from __future__ import annotations

import numpy as np

from karaoke.lyrics.syllables import split_syllables
from karaoke.melody import decode as D
from karaoke.models import LyricsFinal, Note, Syllable

GRID_S = 0.03
TAIL_S = 0.08


def boundaries(parts: list[str], start: float, end: float, support: np.ndarray,
               rough: np.ndarray, hop: float) -> list[float]:
    """Ordered cuts, never empty spans; sub-grid words use exact proportional cuts."""
    weights = np.asarray([max(1, sum(c.isalpha() for c in p)) for p in parts], dtype=float)
    targets = np.r_[0.0, np.cumsum(weights) / weights.sum()]
    count = len(parts)
    if count == 1:
        return [start, end]
    steps = int((end - start) / GRID_S)
    if steps < count:
        return list(start + targets * (end - start))
    grid = np.linspace(start, end, steps + 1)
    frames = np.clip(np.round(grid / hop).astype(int), 0, max(0, len(support) - 1))
    cue = np.zeros(len(grid))
    if len(support):
        cue = 1.0 - support[frames]
        # A stable pitch difference on either side, not a single raw frame's argmax.
        d = max(1, int(0.03 / hop))
        left = rough[np.maximum(0, frames - d)]
        right = rough[np.minimum(len(rough) - 1, frames + d)]
        cue += 0.6 * ((left != right) & (left >= 0) & (right >= 0))
    dp = np.full(len(grid), -np.inf)
    dp[0] = 0
    back = np.zeros((count, len(grid)), dtype=int)
    for syllable in range(1, count):
        new = np.full(len(grid), -np.inf)
        best, arg = -np.inf, 0
        for k in range(syllable, len(grid) - (count - syllable)):
            if dp[k-1] > best:
                best, arg = dp[k-1], k-1
            # Normalised timing keeps acoustic cues local to the expected boundary.
            penalty = 6.0 * ((k / steps - targets[syllable]) * count) ** 2
            new[k] = best + cue[k] - penalty
            back[syllable, k] = arg
        dp = new
    k = int(np.argmax(dp))
    cuts = []
    for syllable in range(count-1, 0, -1):
        cuts.append(float(grid[k]))
        k = back[syllable, k]
    return [start, *reversed(cuts), end]


def syllable_spans(lyrics: LyricsFinal, support: np.ndarray, rough: np.ndarray, hop: float) -> list[Syllable]:
    out = []
    for line in lyrics.lines:
        for wi, word in enumerate(line.words):
            if not word.text.strip():
                continue
            if not np.isfinite([word.start, word.end]).all() or word.start < 0 or word.end <= word.start:
                raise ValueError(f"word {line.id}/{wi} needs finite, positive time bounds")
            parts = split_syllables(word.text, lyrics.language)
            start, end = word.start, word.end
            a, b = max(0, int(np.ceil(start / hop))), min(len(support), int(np.ceil(end / hop)))
            heard = np.flatnonzero(support[a:b] >= 0.15)
            # Word alignments often include a trailing rest. Keep enough duration for every
            # syllable; completely unheard words keep their nominal span with faint notes.
            if len(heard):
                last = (a + int(heard[-1]) + 1) * hop
                earliest = start + min(end - start, len(parts) * GRID_S)
                if end - last >= TAIL_S:
                    end = min(end, max(earliest, last))
            cuts = boundaries(parts, start, end, support, rough, hop)
            out.extend(Syllable(line_id=line.id, word_index=wi, text=part, start=s, end=e)
                       for part, s, e in zip(parts, cuts[:-1], cuts[1:], strict=True))
    return out


def decode_syllables(S: np.ndarray, lo: int, act: np.ndarray, bin0: int, hop: float, tuning: float,
                     lyrics: LyricsFinal, rough: np.ndarray, prior: D.Prior) -> tuple[list[Note], list[Syllable]]:
    support = S.max(axis=1) if len(S) else np.zeros(0)
    syllables = syllable_spans(lyrics, support, rough, hop)
    T = len(S)
    mask = np.zeros(T, dtype=bool)
    change = np.full(T, prior.change)
    times = np.arange(T) * hop
    for sy in syllables:
        mask |= (times >= sy.start) & (times < sy.end)
        change[np.abs(times - sy.start) <= hop / 2] = prior.syllable
    path = D.viterbi(S, np.zeros(T), change, prior, hop, note_ok=mask, rest_ok=~mask)
    # Preserve lead-in context for unheard words, including spans with no frame centres.
    known = np.flatnonzero((path >= 0) & (support >= 0.15))
    if not len(known):
        known = np.flatnonzero(rough >= 0)
    fallback = int(np.clip(60 - lo, 0, S.shape[1] - 1))
    fine, _ = D.peak_pitch(act, bin0, centre=np.where(path >= 0, path + lo + tuning, np.nan))
    notes = []
    for si, sy in enumerate(syllables):
        frames = np.flatnonzero((times >= sy.start) & (times < sy.end))
        if not len(frames):
            near = int(known[np.argmin(np.abs(times[known] - sy.start))]) if len(known) else None
            pitch = int(path[near] if path[near] >= 0 else rough[near]) if near is not None else fallback
            notes.append(Note(start=sy.start, end=sy.end, midi=pitch+lo, pitch=pitch+lo+tuning, conf=0, syllable=si))
            continue
        labels = path[frames].copy()
        # With no evidence anywhere in the recording, a deterministic middle-C placeholder is
        # more useful than the lowest supported CREPE bin. Confidence remains exactly zero.
        if not len(known):
            labels[:] = fallback
        elif not np.any(support[frames] > 1e-8):
            near = int(known[np.argmin(np.abs(times[known] - (sy.start + sy.end) / 2))])
            labels[:] = path[near] if path[near] >= 0 else rough[near]
        edges = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(frames)]
        for a, b in zip(edges, edges[1:], strict=False):
            idx = frames[a:b]
            pitch = int(labels[a])
            start = sy.start if a == 0 else max(sy.start, (int(idx[0]) - 0.5) * hop)
            end = sy.end if b == len(frames) else max(start, (int(frames[b]) - 0.5) * hop)
            if end <= start:
                continue
            values = fine[idx]
            values = values[np.isfinite(values)]
            notes.append(Note(start=start, end=end, midi=pitch+lo,
                              pitch=float(np.median(values)) if len(values) else pitch+lo+tuning,
                              conf=float(S[idx, pitch].mean()), syllable=si))
    notes.sort(key=lambda n: (n.start, n.end))
    return notes, syllables
