"""A pitch contour (pyin f0 + voicing probability) → notes.

A separated vocal stem is not a clean solo voice: it has leakage from the band, reverb and echo
tails, breaths and doubled backing vocals. So the segmentation is conservative; a missing note is
better than a wrong one in a sing-along guide.

1. **Voiced frames**: pyin's own voicing probability at least ``VOICED_PROB``, or at least
   ``STABLE_PROB`` where the pitch holds steady (within ``STABLE_RANGE`` semitones over
   ``STABLE_S``: pyin is unsure of a stem's held notes, but noise and leakage do not hold a pitch).
   The frame must also be within ``LEVEL_DB`` of the stem's loud level (echo tails and leakage are
   pitched but quiet). Runs shorter than ``MIN_NOTE_S`` are dropped.
2. **Tuning**: the singer's (or recording's) offset from A440, the circular mean of the notes'
   distance to the nearest semitone (from a first pass), is taken out before quantising.
3. **Quantise with hysteresis**: the contour (in fractional MIDI) is smoothed with a moving median
   over ``SMOOTH_S`` (about one vibrato cycle), then the current semitone is kept until the
   smoothed pitch is more than ``HYSTERESIS`` semitones away from it. Vibrato and intonation
   jitter stay inside one note.
4. **Merge**: label runs shorter than ``MIN_STATE_S`` (slides, scoops into a note) join the
   neighbour closest in pitch. Each run's pitch is the median of its raw frames; neighbours with the
   same pitch and less than ``MERGE_GAP_S`` between them become one note.
5. **Octave errors**: pyin sometimes jumps an octave for a note or two. A note an octave (±1) away
   from both neighbours, which agree with each other, is moved to their octave.
6. Notes shorter than ``MIN_NOTE_S`` are dropped, and so are notes further than ``RANGE_ST``
   from the melody's median (bass leakage, growls).
7. **Lyrics** (optional, word timings from ``lyrics.final.json``): notes that hardly overlap any
   sung word are dropped unless they are confident and long (word timings are often off by a
   fraction of a second, so a clear note is trusted over them; short or unsure ones are ad-libs or
   leakage), and a note held across a word start is split there, so each syllable gets its own bar.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

VOICED_PROB = 0.5  # pyin voicing probability needed for a frame to count (conservative) ...
STABLE_PROB = 0.1  # ... or this much where the pitch is steady:
STABLE_S = 0.2  # within STABLE_RANGE semitones over a window this long
STABLE_RANGE = 0.6
LEVEL_DB = 25.0  # frames this far below the stem's loud level are leakage or echo
LOUD_PERCENTILE = 95.0
SMOOTH_S = 0.2  # moving-median window (about one vibrato cycle at 5-6 Hz)
HYSTERESIS = 0.7  # semitones the smoothed pitch must leave the current note by to change note
MIN_STATE_S = 0.14  # shorter label runs inside a voiced run are glides/scoops: merged into a neighbour
MERGE_GAP_S = 0.1  # same pitch with a shorter gap: one note
MIN_NOTE_S = 0.12  # shorter notes are dropped
OCTAVE_CONTEXT_S = 0.6  # neighbours closer than this are used to fix octave errors
RANGE_ST = 18  # notes further than this from the melody's median (semitones) are dropped
WORD_PAD_S = 0.25  # word spans are widened by this when matching notes to words
MIN_WORD_OVERLAP = 0.3  # a note must overlap the (padded) words for this fraction of its length ...
WORDLESS_CONF = 0.6  # ... unless its confidence is at least this
WORDLESS_MIN_S = 0.25  # ... and it is at least this long
SPLIT_MIN_S = 0.1  # a word start splits a note only if both parts are at least this long


@dataclass
class RawNote:
    start: float
    end: float
    midi: int
    pitch: float  # median fractional MIDI
    conf: float

    @property
    def dur(self) -> float:
        return self.end - self.start


def hz_to_midi(f0: np.ndarray) -> np.ndarray:
    f0 = np.asarray(f0, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = 69.0 + 12.0 * np.log2(f0 / 440.0)
    m[~np.isfinite(m)] = np.nan
    return m


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where ``mask`` is True."""
    if len(mask) == 0:
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def _moving_median(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) < 3:
        return x.copy()
    h = win // 2
    xp = np.pad(x, h, mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view

    return np.median(sliding_window_view(xp, 2 * h + 1), axis=1)


def stable(midi: np.ndarray, hop_s: float) -> np.ndarray:
    """Frames whose centred ``STABLE_S`` window holds the pitch within ``STABLE_RANGE``."""
    win = max(3, int(round(STABLE_S / hop_s)) | 1)
    h = win // 2
    if len(midi) < win:
        return np.zeros(len(midi), dtype=bool)
    from numpy.lib.stride_tricks import sliding_window_view

    v = sliding_window_view(np.pad(midi, h, constant_values=np.nan), win)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rng = np.nanmax(v, axis=1) - np.nanmin(v, axis=1)
        full = np.isfinite(v).sum(axis=1) >= win - 1
    return full & (rng <= STABLE_RANGE)


def voiced_mask(midi: np.ndarray, prob: np.ndarray, db: np.ndarray | None, hop_s: float) -> np.ndarray:
    pr = np.nan_to_num(prob)
    m = np.isfinite(midi) & ((pr >= VOICED_PROB) | ((pr >= STABLE_PROB) & stable(midi, hop_s)))
    if db is not None and len(db) and m.any():
        db = np.asarray(db[: len(m)], dtype=float)
        if len(db) < len(m):
            db = np.r_[db, np.full(len(m) - len(db), -120.0)]
        ref = float(np.percentile(db[m], LOUD_PERCENTILE))
        m &= db >= ref - LEVEL_DB
    min_frames = max(1, int(round(MIN_NOTE_S / hop_s)))
    for a, b in _runs(m):
        if b - a < min_frames:
            m[a:b] = False
    return m


def quantise(seg: np.ndarray, hop_s: float) -> np.ndarray:
    """Semitone labels for one voiced run (fractional MIDI), with hysteresis and short runs merged."""
    sm = _moving_median(seg, max(1, int(round(SMOOTH_S / hop_s))) | 1)
    labels = np.empty(len(sm), dtype=int)
    cur = int(round(sm[0]))
    for i, v in enumerate(sm):
        if abs(v - cur) > HYSTERESIS:
            cur = int(round(v))
        labels[i] = cur
    min_state = max(1, int(round(MIN_STATE_S / hop_s)))
    while True:  # merge the shortest too-short run into its closest neighbour, until none is left
        runs = _label_runs(labels)
        short = [(b - a, k) for k, (a, b, _) in enumerate(runs) if b - a < min_state]
        if not short or len(runs) == 1:
            break
        _, k = min(short)
        a, b, lab = runs[k]
        nb = [runs[j] for j in (k - 1, k + 1) if 0 <= j < len(runs)]
        mean = float(np.mean(seg[a:b]))
        target = min(nb, key=lambda r: abs(r[2] - mean))[2]
        labels[a:b] = target
    return labels


def _label_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    out = []
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        out.append((i, j, int(labels[i])))
        i = j
    return out


def fix_octaves(notes: list[RawNote]) -> int:
    """Move notes an octave away from both (agreeing) neighbours back to their octave. In place."""
    changed = 0
    for i in range(1, len(notes) - 1):
        a, n, b = notes[i - 1], notes[i], notes[i + 1]
        if n.start - a.end > OCTAVE_CONTEXT_S or b.start - n.end > OCTAVE_CONTEXT_S:
            continue
        if abs(a.midi - b.midi) > 5:
            continue
        ref = (a.midi + b.midi) / 2
        for k in (-12, 12):
            if abs(n.midi + k - ref) <= 5 and abs(n.midi - ref) >= 9:
                n.midi += k
                n.pitch += k
                changed += 1
                break
    return changed


def drop_outliers(notes: list[RawNote]) -> list[RawNote]:
    """Drop notes further than ``RANGE_ST`` from the duration-weighted median pitch (bass leakage
    and growls near pyin's ``fmin``, whistle-like octave errors)."""
    if len(notes) < 5:
        return notes
    order = sorted(notes, key=lambda n: n.midi)
    half = sum(n.dur for n in notes) / 2
    acc = 0.0
    med = order[-1].midi
    for n in order:
        acc += n.dur
        if acc >= half:
            med = n.midi
            break
    return [n for n in notes if abs(n.midi - med) <= RANGE_ST]


def _merge_same(notes: list[RawNote], gap: float) -> list[RawNote]:
    out: list[RawNote] = []
    for n in notes:
        if out and out[-1].midi == n.midi and n.start - out[-1].end <= gap:
            p = out[-1]
            w0, w1 = p.dur, n.dur
            p.pitch = (p.pitch * w0 + n.pitch * w1) / max(w0 + w1, 1e-9)
            p.conf = (p.conf * w0 + n.conf * w1) / max(w0 + w1, 1e-9)
            p.end = n.end
        else:
            out.append(RawNote(n.start, n.end, n.midi, n.pitch, n.conf))
    return out


def filter_by_words(notes: list[RawNote], words: list[tuple[float, float]]) -> list[RawNote]:
    """Keep notes that overlap the (padded) sung words for at least ``MIN_WORD_OVERLAP``, and
    confident, long notes (``WORDLESS_CONF``, ``WORDLESS_MIN_S``) wherever they are."""
    if not words:
        return notes
    spans = sorted((s - WORD_PAD_S, e + WORD_PAD_S) for s, e in words)
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    starts = np.array([m[0] for m in merged])
    out = []
    for n in notes:
        i = int(np.searchsorted(starts, n.end))
        ov = sum(max(0.0, min(n.end, e) - max(n.start, s)) for s, e in merged[max(0, i - 3):i])
        if ov >= MIN_WORD_OVERLAP * n.dur or (n.conf >= WORDLESS_CONF and n.dur >= WORDLESS_MIN_S):
            out.append(n)
    return out


def split_at_words(notes: list[RawNote], word_starts: list[float]) -> list[RawNote]:
    """Split notes at the word starts that fall inside them (a held pitch over two syllables)."""
    ws = np.sort(np.asarray(word_starts, dtype=float))
    out = []
    for n in notes:
        cuts = [t for t in ws[(ws > n.start + SPLIT_MIN_S) & (ws < n.end - SPLIT_MIN_S)]]
        t0 = n.start
        kept = []
        for t in cuts:
            if t - t0 >= SPLIT_MIN_S:
                kept.append(t)
                t0 = t
        edges = [n.start, *kept, n.end]
        for a, b in zip(edges, edges[1:], strict=False):
            out.append(RawNote(a, b, n.midi, n.pitch, n.conf))
    return out


def circular_offset(pitches: list[float] | np.ndarray, weights: list[float] | np.ndarray | None = None) -> float:
    """Weighted circular mean of the pitches' distance to the nearest semitone (-0.5..0.5)."""
    m = np.asarray(pitches, dtype=float)
    w = np.ones(len(m)) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(m)
    if not ok.any() or w[ok].sum() <= 0:
        return 0.0
    z = np.sum(w[ok] * np.exp(2j * np.pi * (m[ok] - np.round(m[ok])))) / w[ok].sum()
    return float(np.angle(z) / (2 * np.pi)) if abs(z) > 1e-6 else 0.0


def estimate_tuning(midi: np.ndarray, prob: np.ndarray, hop_s: float, db: np.ndarray | None = None) -> float:
    """Offset from equal temperament at A440 in semitones (-0.5..0.5): the duration-weighted
    circular mean of the notes' median pitches (a first pass without tuning). Per-note medians,
    not frames: vibrato spreads frames evenly over a semitone and would make the mean arbitrary."""
    notes = segment(midi, prob, hop_s, db=db, tuning=0.0)
    return circular_offset([n.pitch for n in notes], [n.dur for n in notes])


def segment(midi: np.ndarray, prob: np.ndarray, hop_s: float, db: np.ndarray | None = None,
            words: list[tuple[float, float]] | None = None, tuning: float | None = None) -> list[RawNote]:
    """Notes from a contour sampled every ``hop_s`` (frame ``i`` is centred on ``i * hop_s``).

    ``midi`` is fractional MIDI (NaN where pyin found no pitch), ``prob`` pyin's voicing
    probability, ``db`` the frame loudness in dBFS on the same grid (optional), ``words`` sung word
    spans (optional), ``tuning`` the offset to take out (None: ``estimate_tuning``). See the
    module docstring.
    """
    midi = np.asarray(midi, dtype=float)
    prob = np.asarray(prob, dtype=float)[: len(midi)]
    mask = voiced_mask(midi, prob, db, hop_s)
    if tuning is None:
        tuning = estimate_tuning(midi, prob, hop_s, db)
    notes: list[RawNote] = []
    for a, b in _runs(mask):
        seg = midi[a:b]
        labels = quantise(seg - tuning, hop_s)
        for s, e, _ in _label_runs(labels):
            pitch = float(np.median(seg[s:e]))
            notes.append(RawNote(max(0.0, (a + s - 0.5) * hop_s), (a + e - 0.5) * hop_s,
                                 int(round(pitch - tuning)), pitch, float(np.mean(prob[a + s:a + e]))))
    notes = _merge_same(notes, MERGE_GAP_S)
    fix_octaves(notes)
    notes = _merge_same(notes, MERGE_GAP_S)
    notes = [n for n in notes if n.dur >= MIN_NOTE_S - 1e-9]
    notes = drop_outliers(notes)
    if words:
        notes = filter_by_words(notes, words)
        notes = split_at_words(notes, [s for s, _ in words])
    for n in notes:
        n.start, n.end, n.pitch, n.conf = round(n.start, 3), round(n.end, 3), round(n.pitch, 2), round(n.conf, 3)
    return notes
