"""Snap word timings to the pauses the vocal stem actually has.

Whisper's word timestamps are back to back: when the singer pauses, the silence is folded into
the word before or after it. Rendered as karaoke, the fill then crawls over that word during the
pause. The separated vocal stem shows where the pause is, but not as absolute silence (reverb and
leakage keep a stem at -30..-45 dB between phrases). So a pause is measured *relative to the word
itself*: a stretch of at least ``MIN_PAUSE_S`` that stays ``DROP_DB`` below the word's own loud
level (its ``LOUD_PERCENTILE`` frame).

For each word (``trim_words``):
- a quiet stretch of at least ``LEAD_MIN_PAUSE_S`` at the start moves the start to just before
  the voice onset (shorter ones are left: soft consonants look like pauses to an energy detector);
- the longest quiet stretch after that ends the word (plus a short release);
- whatever is loud after that pause is the next word's onset when that word starts within
  ``ONSET_PULL_S``, so the next word starts there instead.

Pauses are only ever cut out of words, never added. Words the caller marks as fixed (manual and
AI edits) are not touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from karaoke.audio import HOP_S, read_mono, rms_db

MIN_PAUSE_S = 0.3  # shorter dips are consonants and breaths
LEAD_MIN_PAUSE_S = 0.6  # a word's start only moves past a longer pause (sibilants like "s", "sh" are quiet)
DROP_DB = 12.0  # a pause is this far below the word's loud level ...
LOUD_PERCENTILE = 90.0
FLOOR_DB = -60.0  # ... or simply below this (true silence, whatever the word's level)
LEAD_PAD_S = 0.15  # a moved start sits this much before the loud onset (the consonant before the vowel)
TAIL_PAD_S = 0.10  # a trimmed end keeps this much of the release
ONSET_PULL_S = 0.6  # the next word may start at most this much earlier than whisper said
MIN_WORD_S = 0.15
MIN_LOUD_S = 0.08  # a loud part shorter than this before a pause is a click, not the word


@dataclass
class Envelope:
    db: np.ndarray
    hop_s: float = HOP_S

    @classmethod
    def from_file(cls, vocals: Path | str) -> Envelope:
        signal, sr = read_mono(vocals)
        return cls(rms_db(signal, sr))

    def frames(self, s: float, e: float) -> tuple[int, int]:
        n = len(self.db)
        return min(n, max(0, int(round(s / self.hop_s)))), min(n, max(0, int(round(e / self.hop_s))))

    def quiet_runs(self, s: float, e: float) -> list[tuple[float, float]]:
        """Pauses inside [s, e] relative to that span's own loudness, as (start, end) seconds."""
        a, b = self.frames(s, e)
        seg = self.db[a:b]
        if len(seg) < 3:
            return []
        thr = max(float(np.percentile(seg, LOUD_PERCENTILE)) - DROP_DB, min(FLOOR_DB, float(seg.max()) - DROP_DB))
        quiet = seg < thr
        runs = []
        i = 0
        while i < len(quiet):
            if quiet[i]:
                j = i
                while j < len(quiet) and quiet[j]:
                    j += 1
                if (j - i) * self.hop_s >= MIN_PAUSE_S:
                    runs.append(((a + i) * self.hop_s, (a + j) * self.hop_s))
                i = j
            else:
                i += 1
        return runs


def trim_span(env: Envelope, s: float, e: float) -> tuple[float, float, float | None]:
    """(start, end, onset) of a word without its pauses; ``onset`` is where the voice comes back
    after the word's pause (the next word's real start), or None."""
    if e - s <= MIN_WORD_S:
        return s, e, None
    runs = env.quiet_runs(s, e)
    if not runs:
        return s, e, None
    ns, ne, onset = s, e, None
    q0, q1 = runs[0]
    if q0 - s < MIN_LOUD_S:  # starts in a pause: move the start to the onset
        if q1 - q0 >= LEAD_MIN_PAUSE_S:
            ns = max(s, q1 - LEAD_PAD_S)
        runs = runs[1:]
    if runs:
        q0, q1 = max(runs, key=lambda r: r[1] - r[0])
        if q0 - ns >= MIN_LOUD_S:
            ne = min(e, q0 + TAIL_PAD_S)
            onset = q1 if e - q1 >= 0.02 else None
    if ne - ns < MIN_WORD_S:
        return s, e, None
    return round(ns, 3), round(ne, 3), (round(onset, 3) if onset is not None else None)


def trim_words(env: Envelope, times: list[list[float]], fixed: list[bool] | None = None) -> int:
    """Trim pauses out of word timings in place (``times`` = sorted [start, end] pairs).

    Returns how many words changed.
    """
    fixed = fixed or [False] * len(times)
    changed = 0
    for i, (s, e) in enumerate(times):
        if fixed[i]:
            continue
        ns, ne, onset = trim_span(env, s, e)
        if (ns, ne) != (s, e):
            times[i][0], times[i][1] = ns, ne
            changed += 1
        if onset is not None and i + 1 < len(times) and not fixed[i + 1]:
            nxt = times[i + 1]
            if ne <= onset < nxt[0] and nxt[0] - onset <= ONSET_PULL_S:
                nxt[0] = round(max(ne, onset - LEAD_PAD_S), 3)
    return changed
