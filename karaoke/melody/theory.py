"""Keys, key signatures and how a pitch is written on a treble staff.

- ``detect_key``: Krumhansl–Schmuckler. A 12-bin pitch-class weight vector (here: the notes'
  durations) is correlated with the 24 rotated Krumhansl–Kessler profiles; the best correlation
  wins. With a second vector (the instrumental's chroma) the two correlation tables are averaged.
- ``spell``: the letter and accidental of a MIDI pitch in a key. Scale notes use the key's own
  spelling (no accidental is drawn: the key signature covers it). Notes outside the key prefer, in
  order: the raised 6th/7th of a minor key and the lowered 7th of a major key (the usual chromatic
  notes), then a natural sign (a key-signature letter played natural, e.g. F♮ in G major), then the
  key's own direction (sharps in sharp keys and C major / A minor, flats in flat keys).
- ``staff_position``: diatonic steps above the bottom line of a treble staff (E4 = 0, F5 = 8, the
  top line); odd numbers are spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Mode = Literal["major", "minor"]
Accidental = Literal["sharp", "flat", "natural"]

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

LETTERS = "CDEFGAB"
NATURAL_PC = [0, 2, 4, 5, 7, 9, 11]
SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"
MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]
MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]  # natural minor
# key-signature names by number of fifths, -7..7
MAJOR_BY_FIFTHS = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"]
MINOR_BY_FIFTHS = ["Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#"]
# fifths of each major key by tonic pitch class (the spelling with fewer accidentals; F# over Gb)
_MAJOR_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
# minor keys: the relative major's, except Eb minor (6 flats) over D# minor (6 sharps)
_MINOR_FIFTHS = {pc: _MAJOR_FIFTHS[(pc + 3) % 12] for pc in range(12)} | {3: -6}

E4_DIATONIC = 7 * 4 + 2  # diatonic number (7 * octave + letter) of the bottom staff line
# staff positions of the key-signature accidentals on a treble staff, in signature order
SHARP_POSITIONS = [8, 5, 9, 6, 3, 7, 4]  # F5 C5 G5 D5 A4 E5 B4
FLAT_POSITIONS = [4, 7, 3, 6, 2, 5, 1]  # B4 E5 A4 D5 G4 C5 F4


@dataclass(frozen=True)
class Key:
    tonic_pc: int
    mode: Mode

    @property
    def fifths(self) -> int:
        return (_MAJOR_FIFTHS if self.mode == "major" else _MINOR_FIFTHS)[self.tonic_pc % 12]

    @property
    def tonic(self) -> str:
        return (MAJOR_BY_FIFTHS if self.mode == "major" else MINOR_BY_FIFTHS)[self.fifths + 7]

    @property
    def name(self) -> str:
        return f"{self.tonic} {self.mode}"

    @property
    def signature(self) -> list[str]:
        """Accidentals of the key signature in the order they are written, e.g. ['F#', 'C#']."""
        n = self.fifths
        return [f"{c}#" for c in SHARP_ORDER[:n]] if n >= 0 else [f"{c}b" for c in FLAT_ORDER[:-n]]

    def sig_alter(self, letter: int) -> int:
        """How the key signature alters a letter (0 = C .. 6 = B): +1 sharp, -1 flat, 0 natural."""
        c, n = LETTERS[letter], self.fifths
        if n > 0 and c in SHARP_ORDER[:n]:
            return 1
        if n < 0 and c in FLAT_ORDER[:-n]:
            return -1
        return 0

    @property
    def tonic_letter(self) -> int:
        return LETTERS.index(self.tonic[0])

    def scale_pcs(self) -> list[int]:
        steps = MAJOR_STEPS if self.mode == "major" else MINOR_STEPS
        return [(self.tonic_pc + s) % 12 for s in steps]


@dataclass(frozen=True)
class Spelling:
    letter: int  # 0 = C .. 6 = B
    alter: int  # -1 flat, 0 natural, +1 sharp (as played)
    octave: int  # scientific pitch octave of the letter (C4 = middle C)
    accidental: Accidental | None  # what has to be drawn before the note in this key (None: nothing)

    @property
    def name(self) -> str:
        return f"{LETTERS[self.letter]}{'#' if self.alter > 0 else 'b' if self.alter < 0 else ''}{self.octave}"

    @property
    def staff_position(self) -> int:
        return 7 * self.octave + self.letter - E4_DIATONIC


def _wrap(d: int) -> int:
    """A pitch-class difference as -6..5."""
    return (d + 6) % 12 - 6


def spell(midi: int, key: Key) -> Spelling:
    """Letter, accidental and octave of ``midi`` written in ``key`` (see the module docstring)."""
    pc = midi % 12
    direction = 1 if key.fifths >= 0 else -1
    tl = key.tonic_letter
    steps = MAJOR_STEPS if key.mode == "major" else MINOR_STEPS
    best: tuple[int, int, int] | None = None  # (score, letter, alter)
    for letter in range(7):
        alter = _wrap(pc - NATURAL_PC[letter])
        if abs(alter) > 1:
            continue
        sig = key.sig_alter(letter)
        degree = (letter - tl) % 7
        degree_pc = (key.tonic_pc + steps[degree]) % 12
        if alter == sig:
            score = 0  # in the key
        elif key.mode == "minor" and degree in (5, 6) and pc == (degree_pc + 1) % 12:
            score = 1  # raised 6th / 7th (melodic and harmonic minor)
        elif key.mode == "major" and degree == 6 and pc == (degree_pc - 1) % 12:
            score = 1  # lowered 7th (mixolydian, blues)
        elif alter == 0:
            score = 2  # a natural sign cancels the key signature
        elif alter == direction:
            score = 3
        else:
            score = 4
        if best is None or score < best[0]:
            best = (score, letter, alter)
    assert best is not None
    _, letter, alter = best
    octave = (midi - alter - NATURAL_PC[letter]) // 12 - 1
    sig = key.sig_alter(letter)
    acc: Accidental | None = None
    if alter != sig:
        acc = "natural" if alter == 0 else "sharp" if alter > 0 else "flat"
    return Spelling(letter, alter, octave, acc)


def ledger_positions(pos: int) -> list[int]:
    """Staff positions of the ledger lines a note at ``pos`` needs (lines are even positions;
    the staff itself is 0..8)."""
    if pos <= -2:
        return list(range(-2, pos - 1, -2))
    if pos >= 10:
        return list(range(10, pos + 1, 2))
    return []


# --------------------------------------------------------------------------- key detection


@dataclass(frozen=True)
class KeyResult:
    key: Key
    r: float  # correlation of the best key
    runner_up: Key
    r2: float

    @property
    def margin(self) -> float:
        return self.r - self.r2


def key_correlations(weights: np.ndarray) -> list[tuple[float, Key]]:
    """Correlation of a 12-bin pitch-class weight vector with all 24 keys, best first."""
    w = np.asarray(weights, dtype=float)
    out = []
    if w.shape != (12,) or not np.isfinite(w).all() or w.std() == 0:
        return out
    for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(w, np.roll(prof, tonic))[0, 1])
            out.append((r, Key(tonic, mode)))  # type: ignore[arg-type]
    out.sort(key=lambda x: -x[0])
    return out


def detect_key(*vectors: np.ndarray | None, factors: list[float] | None = None) -> KeyResult | None:
    """Krumhansl–Schmuckler key of one or more pitch-class weight vectors (None if all are empty).

    With several vectors (e.g. the melody's notes and the instrumental's chroma) the 24 key
    correlations are averaged, weighted by ``factors``. Averaging correlations rather than adding
    the vectors keeps a peaky note histogram from drowning out a flat but steady chroma."""
    factors = factors or [1.0] * len(vectors)
    total: dict[Key, float] = {}
    wsum = 0.0
    for vec, f in zip(vectors, factors, strict=True):
        corr = key_correlations(vec) if vec is not None else []
        if not corr:
            continue
        wsum += f
        for r, key in corr:
            total[key] = total.get(key, 0.0) + f * r
    if len(total) < 2 or wsum <= 0:
        return None
    ranked = sorted(((r / wsum, k) for k, r in total.items()), key=lambda x: -x[0])
    (r, key), (r2, key2) = ranked[0], ranked[1]
    return KeyResult(key, r, key2, r2)


def pc_histogram(notes: list[tuple[int, float]]) -> np.ndarray:
    """Pitch-class weights from (midi, weight) pairs, e.g. weight = duration x confidence."""
    h = np.zeros(12)
    for midi, w in notes:
        h[midi % 12] += w
    return h


def display_shift(median_midi: float, centre: float = 71.0, max_octaves: int = 2) -> int:
    """Octaves to shift a melody by so its median sits near the staff's middle line (B4 = 71).
    Positive: written higher than it sounds (a low voice; "8vb" under the clef)."""
    x = (centre - median_midi) / 12
    k = int(np.sign(x) * np.floor(abs(x) + 0.5 - 1e-9))  # nearest; a tie keeps the written octave
    return max(-max_octaves, min(max_octaves, k))
