"""Retained pitch-bin activations → a path with a prior on how melodies move.

CREPE gives, for every 10 ms frame, an activation for each of its pitch bins (20 cents apart):
``track.crepe_activation``. The old tracker used its own batch-local Viterbi, then discarded
alternative bins before note segmentation. This decoder keeps alternatives across the song:
weak nearby evidence can beat a brief stronger distant peak, while sustained leaps survive.

1. **Salience per semitone** (``semitone_salience``): the strongest bin of each semitone (on the
   singer's tuning) in every frame. Neighbouring bins can support alternative semitones;
   the context decides.
2. **A Viterbi path over (note | rest) states** (``viterbi``), one state per semitone plus, for
   every semitone, a rest that remembers the last note. The path collects the salience of the notes
   it passes through (a rest collects a fixed threshold instead) and pays for every change of note:
   a fixed cost (``Prior.change``: glides, scoops and vibrato are not notes) plus the cost of the
   interval (``Prior.interval``: minus the log of how often sung melodies move that far, counted on
   the human references in ``data/gold``; see docs/benchmarks.md). The interval also counts
   across a rest at full cost, and is forgotten after a long silence.
3. Callers shape the path with per-frame arrays: where a note may sound (``note_ok``), where a
   rest may (``rest_ok``), what a change costs (``change``: cheaper where a new syllable starts)
   and the rest's threshold (lower inside a sung word).

The path's score is a sum of activations, not of log-probabilities: CREPE's bins are independent
sigmoids (they do not sum to one) and neighbouring frames are far from independent, so the scale
between evidence and prior is a measured constant (``Prior.weight``), not a derived one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from karaoke.melody.track import bin_to_midi

NEG = -1e9

# Minus the log of how often consecutive notes are that many semitones apart (0..12, then
# "more"), relative to a step of one or two: counted over the 4 059 note pairs of the human
# references (Rock Corpus and Hooktheory songs, vocadito clips; spikes/melody_jumps.py and
# docs/benchmarks.md). Unison and steps are free; a fifth costs 2.9, an octave 5.
INTERVAL_COST = (0.0, 0.0, 0.0, 0.7, 1.6, 1.7, 3.5, 2.9, 3.8, 3.8, 4.2, 5.5, 5.0, 6.5)


@dataclass(frozen=True)
class Prior:
    """What the path pays, in units of activation x frames (one frame of a clear note is ~0.8)."""

    weight: float = 2.0  # multiplies the interval cost
    interval: tuple[float, ...] = INTERVAL_COST
    change: float = 4.0  # changing note where no syllable starts
    syllable: float = 0.5  # ... where one does
    on: float = 1.5  # starting a note after a rest; ending one
    off: float = 1.5
    voiced: float = 0.35  # activation a frame needs for a note to beat a rest ...
    word_voiced: float = 0.12  # ... inside a sung word
    forget_s: float = 2.0  # a rest this long forgets the pitch before it

    def table(self, n: int) -> np.ndarray:
        """Interval cost between semitone states i and j (n x n), zero on the diagonal."""
        d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        t = np.asarray(self.interval, dtype=float)
        return self.weight * t[np.minimum(d, len(t) - 1)]


def with_prior(prior: Prior, **changes) -> Prior:
    return replace(prior, **changes)


def semitone_salience(act: np.ndarray, bin0: int, tuning: float, lo: int, hi: int) -> np.ndarray:
    """``S[t, s - lo]``: the strongest activation among the bins that round to semitone ``s``
    (MIDI ``lo..hi``) once ``tuning`` (semitones the singing sits off A440) is taken out."""
    label = np.round(bin_to_midi(bin0 + np.arange(act.shape[1])) - tuning).astype(int)
    out = np.zeros((act.shape[0], hi - lo + 1), dtype=np.float32)
    for s in range(max(lo, int(label.min())), min(hi, int(label.max())) + 1):
        out[:, s - lo] = act[:, label == s].max(axis=1)
    return out


def peak_pitch(act: np.ndarray, bin0: int, centre: np.ndarray | None = None, reach: float = 1.0
               ) -> tuple[np.ndarray, np.ndarray]:
    """(fractional MIDI, activation) of each frame's peak: the weighted mean of the bins around
    the strongest one (``centre`` None), or around the strongest within ``reach`` semitones of
    ``centre[t]`` (NaN there: no pitch for that frame)."""
    T, B = act.shape
    midi = bin_to_midi(bin0 + np.arange(B))
    a = act
    if centre is not None:
        ok = np.abs(midi[None, :] - np.asarray(centre, dtype=float)[:, None]) <= reach
        a = np.where(ok, act, 0.0)
    k = a.argmax(axis=1)
    idx = np.clip(k[:, None] + np.arange(-2, 3)[None, :], 0, B - 1)
    w = np.take_along_axis(a, idx, axis=1).astype(float)
    tot = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pitch = (w * midi[idx]).sum(axis=1) / tot
    peak = a[np.arange(T), k].astype(float)
    pitch[(tot <= 0) | (peak <= 0)] = np.nan
    return pitch, peak


def activation_from_contour(f0: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, int]:
    """An activation with one bump per frame, for trackers that only give a contour (pyin):
    ``(act, bin0)``. The bump is CREPE's training target (a Gaussian, 25 cents wide)."""
    from karaoke.melody.track import CREPE_BINS, midi_to_bin

    f0 = np.asarray(f0, dtype=float)
    ok = np.isfinite(f0) & (f0 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = midi_to_bin(69.0 + 12.0 * np.log2(np.where(ok, f0, 440.0) / 440.0))
    d = (np.arange(CREPE_BINS)[None, :] - b[:, None]) * 20.0
    act = np.exp(-0.5 * (d / 25.0) ** 2) * (np.nan_to_num(np.asarray(conf, dtype=float)) * ok)[:, None]
    return act.astype(np.float32), 0


def viterbi(S: np.ndarray, rest: np.ndarray, change: np.ndarray, prior: Prior, hop_s: float,
            note_ok: np.ndarray | None = None, rest_ok: np.ndarray | None = None) -> np.ndarray:
    """The best path through ``S`` (frames x semitones): the semitone index per frame, -1 = rest.

    ``rest[t]`` is what a rest collects in frame ``t``, ``change[t]`` what changing note costs on
    entering frame ``t``; ``note_ok`` / ``rest_ok`` (bool per frame) forbid notes or rests. A frame
    where both are forbidden is a rest.
    """
    T, n = S.shape
    if T == 0:
        return np.zeros(0, dtype=int)
    C = prior.table(n)
    off_diag = 1.0 - np.eye(n)
    note_ok = np.ones(T, dtype=bool) if note_ok is None else np.asarray(note_ok, dtype=bool)
    rest_ok = np.ones(T, dtype=bool) if rest_ok is None else np.asarray(rest_ok, dtype=bool) | ~note_ok
    # a rest forgets its pitch once the silence has lasted ``forget_s``
    quiet = ~note_ok | (S.max(axis=1) < prior.voiced)
    run = np.zeros(T, dtype=int)
    for t in range(T):
        run[t] = run[t - 1] + 1 if quiet[t] and t else int(quiet[t])
    forget = run == max(1, int(round(prior.forget_s / hop_s)))

    dn = np.full(n, NEG)  # best score ending in note i
    dr = np.zeros(n)  # ... in the rest after note i
    from_n = np.zeros((T, n), dtype=np.int16)  # note j came from note k (k < n) or rest k - n
    from_r = np.zeros((T, n), dtype=bool)  # rest i came from note i
    forgot = {}  # frame → the rest every rest state continues from when the pitch is forgotten
    cols = np.arange(n)
    for t in range(T):
        if forget[t]:
            forgot[t] = int(dr.argmax())
            dr = np.full(n, dr.max())
        cost = C + change[t] * off_diag
        cand = np.concatenate([dn[:, None] - cost, dr[:, None] - cost - prior.on], axis=0)
        k = cand.argmax(axis=0)
        new_n = cand[k, cols] + S[t] if note_ok[t] else np.full(n, NEG)
        leave = dn - prior.off
        took = leave > dr
        new_r = np.where(took, leave, dr) + rest[t] if rest_ok[t] else np.full(n, NEG)
        from_n[t], from_r[t] = k, took
        dn, dr = new_n, new_r
    path = np.full(T, -1, dtype=int)
    in_note = bool(dn.max() > dr.max())
    i = int(dn.argmax() if in_note else dr.argmax())
    for t in range(T - 1, -1, -1):
        if in_note:
            path[t] = i
            k = int(from_n[t, i])
            in_note, i = k < n, k % n
        else:
            in_note = bool(from_r[t, i])
        if not in_note and t in forgot:
            i = forgot[t]
    return path


def path_notes(path: np.ndarray, lo: int, hop_s: float, S: np.ndarray, act: np.ndarray, bin0: int,
               tuning: float, cuts: np.ndarray | None = None):
    """``notes.RawNote`` for every run of one semitone in ``path``; a run is also cut at the frames
    in ``cuts`` (syllable starts). ``pitch`` is the median of the frames' own peaks near the note
    (the note itself where the voice shows none), ``conf`` the note's mean salience."""
    from karaoke.melody.notes import RawNote

    T = len(path)
    cut = np.zeros(T + 1, dtype=bool)
    if cuts is not None and len(cuts):
        cut[np.clip(np.asarray(cuts, dtype=int), 0, T)] = True
    fine, _ = peak_pitch(act, bin0, centre=np.where(path >= 0, path + lo + tuning, np.nan))
    out = []
    a = 0
    while a < T:
        if path[a] < 0:
            a += 1
            continue
        b = a + 1
        while b < T and path[b] == path[a] and not cut[b]:
            b += 1
        midi = int(path[a]) + lo
        seg = fine[a:b]
        seg = seg[np.isfinite(seg)]
        pitch = float(np.median(seg)) if len(seg) else midi + tuning
        out.append(RawNote(max(0.0, (a - 0.5) * hop_s), (b - 0.5) * hop_s, midi, pitch,
                           float(np.mean(S[a:b, path[a]]))))
        a = b
    return out
