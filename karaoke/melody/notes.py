"""A pitch contour (f0 + confidence) → notes.

A separated vocal stem is not a clean solo voice: it has leakage from the band, reverb and echo
tails, breaths and doubled backing vocals. Outside the sung words the segmentation is therefore
conservative. Inside them the lyrics are a prior that someone *is* singing, so weaker evidence is
accepted there (a sing-along guide with holes in it was the main complaint about the first
version). The thresholds depend on the tracker (``Profile``, ``profile(name)``): pyin's voicing
probability and CREPE's periodicity are on different scales.

1. **Voiced frames**: the tracker's confidence at least ``Profile.voiced``, or at least
   ``Profile.stable_conf`` where the pitch holds steady (within ``STABLE_RANGE`` semitones over
   ``STABLE_S``: noise and leakage do not hold a pitch). Inside a sung word (padded by
   ``Profile.word_pad``) ``Profile.word_conf`` is enough where the pitch moves smoothly
   (``SMOOTH_RANGE`` over ``STABLE_S``). Frames must be within ``Profile.level_db`` (in words:
   ``Profile.word_level_db``) of the stem's loud level: echo tails and leakage are pitched but quiet.
2. **Bridging**: inside a word, unvoiced gaps up to ``Profile.bridge_s`` between voiced frames
   whose pitches agree (within ``BRIDGE_ST``) are filled (a consonant or a tracker dropout in the
   middle of a held note). Then voiced runs shorter than ``MIN_NOTE_S`` are dropped.
3. **Tuning**: the singer's (or recording's) offset from A440, the circular mean of the notes'
   distance to the nearest semitone (from a first pass), is taken out before quantising.
4. **Quantise with hysteresis**: the contour (in fractional MIDI) is smoothed with a moving median
   over ``SMOOTH_S`` (about one vibrato cycle), then the current semitone is kept until the
   smoothed pitch is more than ``HYSTERESIS`` semitones away from it. Vibrato and intonation
   jitter stay inside one note.
5. **Merge**: label runs shorter than ``MIN_STATE_S`` (slides, scoops into a note) join the
   neighbour closest in pitch. Each run's pitch is the median of its raw frames; neighbours with the
   same pitch and less than ``MERGE_GAP_S`` between them become one note.
6. **Octave errors**: a note an octave (±1) away from both neighbours, which agree with each other,
   is moved to their octave. (A phrase-level version, against the median of the notes within 3 s,
   made the octave disagreement with CREPE-full worse: real melodies leap.)
7. Notes shorter than ``MIN_NOTE_S`` are dropped, and so are notes further than ``RANGE_ST``
   from the melody's median (bass leakage, growls).
8. **Lyrics** (optional, word timings from ``lyrics.final.json``): notes that hardly overlap any
   sung word are dropped unless they are confident and long (word timings are often off by a
   fraction of a second, so a clear note is trusted over them; short or unsure ones are ad-libs or
   leakage); a note held across a word start is split there, so each syllable gets its own bar;
   and a gap of up to ``Profile.close_gap_s`` between two notes of the same word is closed (the
   first note is held until the next one starts).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np

STABLE_S = 0.2  # window for the steadiness tests
STABLE_RANGE = 0.6  # a steady pitch stays within this many semitones over STABLE_S
SMOOTH_RANGE = 2.5  # a smoothly moving (sung) pitch stays within this; noise jumps around
LOUD_PERCENTILE = 95.0
SMOOTH_S = 0.2  # moving-median window (about one vibrato cycle at 5-6 Hz)
HYSTERESIS = 0.7  # semitones the smoothed pitch must leave the current note by to change note
MIN_STATE_S = 0.14  # shorter label runs inside a voiced run are glides/scoops: merged into a neighbour
MERGE_GAP_S = 0.1  # same pitch with a shorter gap: one note
MIN_NOTE_S = 0.12  # shorter notes are dropped
BRIDGE_ST = 1.5  # a dropout is bridged only between pitches this close
OCTAVE_CONTEXT_S = 0.6  # neighbours closer than this are used to fix octave errors
RANGE_ST = 18  # notes further than this from the melody's median (semitones) are dropped
WORD_PAD_S = 0.25  # word spans are widened by this when matching notes to words
MIN_WORD_OVERLAP = 0.3  # a note must overlap the (padded) words for this fraction of its length ...
WORDLESS_MIN_S = 0.25  # ... unless it is at least this long and confident (Profile.wordless_conf)
SPLIT_MIN_S = 0.1  # a word start splits a note only if both parts are at least this long


@dataclass(frozen=True)
class Profile:
    """Confidence thresholds of one tracker (see the module docstring)."""

    voiced: float = 0.5  # confidence enough anywhere ...
    stable_conf: float = 0.1  # ... or this much with a steady pitch
    word_conf: float = 0.1  # ... or this much inside a sung word, with a smoothly moving pitch
    level_db: float = 25.0  # below the stem's loud level: leakage or echo
    word_level_db: float = 30.0
    word_pad: float = 0.1
    bridge_s: float = 0.2
    close_gap_s: float = 0.2
    wordless_conf: float = 0.6
    use_words: bool = True  # False: the first version's behaviour (words only filter and split)


PROFILES = {
    # pyin voicing probability: low on a stem's held notes, so steadiness counts too
    "pyin": Profile(),
    # CREPE periodicity: ~0.6-0.9 on clear singing, ~0.2-0.4 on breathy or doubled singing,
    # under 0.15 on noise
    "crepe": Profile(voiced=0.5, stable_conf=0.2, word_conf=0.15, word_level_db=35.0, wordless_conf=0.6),
    "pyin-v1": Profile(use_words=False, bridge_s=0.0, close_gap_s=0.0, level_db=25.0, word_level_db=25.0),
}


def profile(tracker: str) -> Profile:
    """The profile for a tracker name ("pyin", "crepe-full", "crepe-tiny", ...)."""
    if tracker in PROFILES:
        return PROFILES[tracker]
    return PROFILES["crepe"] if tracker.startswith("crepe") else PROFILES["pyin"]


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


def _range(midi: np.ndarray, hop_s: float) -> np.ndarray:
    """Pitch range (max - min) over a centred ``STABLE_S`` window; inf where it is mostly unpitched."""
    win = max(3, int(round(STABLE_S / hop_s)) | 1)
    h = win // 2
    if len(midi) < win:
        return np.full(len(midi), np.inf)
    from numpy.lib.stride_tricks import sliding_window_view

    v = sliding_window_view(np.pad(midi, h, constant_values=np.nan), win)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rng = np.nanmax(v, axis=1) - np.nanmin(v, axis=1)
        full = np.isfinite(v).sum(axis=1) >= win - 1
    return np.where(full, rng, np.inf)


def stable(midi: np.ndarray, hop_s: float) -> np.ndarray:
    """Frames whose centred ``STABLE_S`` window holds the pitch within ``STABLE_RANGE``."""
    return _range(midi, hop_s) <= STABLE_RANGE


def word_mask(words: list[tuple[float, float]] | None, n: int, hop_s: float, pad: float) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in words or []:
        m[max(0, int(np.floor((s - pad) / hop_s))):min(n, int(np.ceil((e + pad) / hop_s)) + 1)] = True
    return m


def voiced_mask(midi: np.ndarray, prob: np.ndarray, db: np.ndarray | None, hop_s: float,
                in_word: np.ndarray | None = None, prof: Profile = PROFILES["pyin"],
                min_run: bool = True) -> np.ndarray:
    pr = np.nan_to_num(prob)
    rng = _range(midi, hop_s)
    m = np.isfinite(midi) & ((pr >= prof.voiced) | ((pr >= prof.stable_conf) & (rng <= STABLE_RANGE)))
    if in_word is not None and prof.use_words:
        m |= np.isfinite(midi) & in_word & (pr >= prof.word_conf) & (rng <= SMOOTH_RANGE)
    if db is not None and len(db) and m.any():
        db = np.asarray(db[: len(m)], dtype=float)
        if len(db) < len(m):
            db = np.r_[db, np.full(len(m) - len(db), -120.0)]
        ref = float(np.percentile(db[m], LOUD_PERCENTILE))
        floor = np.full(len(m), ref - prof.level_db)
        if in_word is not None and prof.use_words:
            floor[in_word] = ref - prof.word_level_db
        m &= db >= floor
    return drop_short_runs(m, hop_s) if min_run else m


def drop_short_runs(m: np.ndarray, hop_s: float) -> np.ndarray:
    m = m.copy()
    min_frames = max(1, int(round(MIN_NOTE_S / hop_s)))
    for a, b in _runs(m):
        if b - a < min_frames:
            m[a:b] = False
    return m


def bridge(midi: np.ndarray, mask: np.ndarray, in_word: np.ndarray, hop_s: float,
           max_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Fill unvoiced gaps up to ``max_s`` inside words between agreeing pitches (linear in MIDI).
    Returns (midi, mask) copies."""
    midi, mask = midi.copy(), mask.copy()
    if max_s <= 0:
        return midi, mask
    max_n = int(round(max_s / hop_s))
    runs = _runs(mask)
    for (_, b0), (a1, _) in zip(runs, runs[1:], strict=False):
        gap = a1 - b0
        if gap > max_n or not in_word[b0:a1].all():
            continue
        p0, p1 = midi[b0 - 1], midi[a1]
        if not (np.isfinite(p0) and np.isfinite(p1)) or abs(p0 - p1) > BRIDGE_ST:
            continue
        midi[b0:a1] = np.linspace(p0, p1, gap + 2)[1:-1]
        mask[b0:a1] = True
    return midi, mask


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
    and growls near the tracker's ``fmin``, whistle-like octave errors)."""
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


def _merged_spans(words: list[tuple[float, float]], pad: float) -> list[list[float]]:
    merged: list[list[float]] = []
    for s, e in sorted((s - pad, e + pad) for s, e in words):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def filter_by_words(notes: list[RawNote], words: list[tuple[float, float]],
                    wordless_conf: float = PROFILES["pyin"].wordless_conf) -> list[RawNote]:
    """Keep notes that overlap the (padded) sung words for at least ``MIN_WORD_OVERLAP``, and
    confident, long notes (``wordless_conf``, ``WORDLESS_MIN_S``) wherever they are."""
    if not words:
        return notes
    merged = _merged_spans(words, WORD_PAD_S)
    starts = np.array([m[0] for m in merged])
    out = []
    for n in notes:
        i = int(np.searchsorted(starts, n.end))
        ov = sum(max(0.0, min(n.end, e) - max(n.start, s)) for s, e in merged[max(0, i - 3):i])
        if ov >= MIN_WORD_OVERLAP * n.dur or (n.conf >= wordless_conf and n.dur >= WORDLESS_MIN_S):
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


def close_gaps(notes: list[RawNote], words: list[tuple[float, float]], max_gap: float) -> int:
    """Hold a note until the next one starts when the gap between them is at most ``max_gap`` and
    lies inside one sung word (a dropout, not a rest). In place; returns notes changed."""
    if max_gap <= 0 or not words:
        return 0
    spans = _merged_spans(words, 0.0)
    changed = 0
    for a, b in zip(notes, notes[1:], strict=False):
        gap = b.start - a.end
        if 0 < gap <= max_gap and any(s <= a.end and b.start <= e for s, e in spans):
            a.end = b.start
            changed += 1
    return changed


def circular_offset(pitches: list[float] | np.ndarray, weights: list[float] | np.ndarray | None = None) -> float:
    """Weighted circular mean of the pitches' distance to the nearest semitone (-0.5..0.5)."""
    m = np.asarray(pitches, dtype=float)
    w = np.ones(len(m)) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(m)
    if not ok.any() or w[ok].sum() <= 0:
        return 0.0
    z = np.sum(w[ok] * np.exp(2j * np.pi * (m[ok] - np.round(m[ok])))) / w[ok].sum()
    return float(np.angle(z) / (2 * np.pi)) if abs(z) > 1e-6 else 0.0


def estimate_tuning(midi: np.ndarray, prob: np.ndarray, hop_s: float, db: np.ndarray | None = None,
                    prof: Profile = PROFILES["pyin"]) -> float:
    """Offset from equal temperament at A440 in semitones (-0.5..0.5): the duration-weighted
    circular mean of the notes' median pitches (a first pass without tuning or words). Per-note
    medians, not frames: vibrato spreads frames evenly over a semitone and would make the mean
    arbitrary."""
    notes = segment(midi, prob, hop_s, db=db, tuning=0.0, prof=prof)
    return circular_offset([n.pitch for n in notes], [n.dur for n in notes])


def segment(midi: np.ndarray, prob: np.ndarray, hop_s: float, db: np.ndarray | None = None,
            words: list[tuple[float, float]] | None = None, tuning: float | None = None,
            prof: Profile = PROFILES["pyin"]) -> list[RawNote]:
    """Notes from a contour sampled every ``hop_s`` (frame ``i`` is centred on ``i * hop_s``).

    ``midi`` is fractional MIDI (NaN where the tracker found no pitch), ``prob`` the tracker's
    confidence, ``db`` the frame loudness in dBFS on the same grid (optional), ``words`` sung word
    spans (optional), ``tuning`` the offset to take out (None: ``estimate_tuning``), ``prof`` the
    tracker's thresholds. See the module docstring.
    """
    midi = np.asarray(midi, dtype=float)
    prob = np.asarray(prob, dtype=float)[: len(midi)]
    in_word = word_mask(words, len(midi), hop_s, prof.word_pad) if words else None
    mask = voiced_mask(midi, prob, db, hop_s, in_word, prof, min_run=False)
    if in_word is not None and prof.use_words:  # bridge before short runs go: they may join up
        midi, mask = bridge(midi, mask, in_word, hop_s, prof.bridge_s)
    mask = drop_short_runs(mask, hop_s)
    if tuning is None:
        tuning = estimate_tuning(midi, prob, hop_s, db, prof)
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
        notes = filter_by_words(notes, words, prof.wordless_conf)
        notes = split_at_words(notes, [s for s, _ in words])
        if prof.use_words:
            close_gaps(notes, words, prof.close_gap_s)
    for n in notes:
        n.start, n.end, n.pitch, n.conf = round(n.start, 3), round(n.end, 3), round(n.pitch, 2), round(n.conf, 3)
    return notes


def with_profile(prof: Profile, **changes) -> Profile:
    """A copy of ``prof`` with some thresholds changed (for experiments)."""
    return replace(prof, **changes)
