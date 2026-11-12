"""Activation → melodic path and word-constrained syllables; legacy contour helpers for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field

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


def load_contour_midi(p: JobPaths) -> tuple[np.ndarray, float] | None:
    """(fractional MIDI per frame with NaN where there is no pitch, hop seconds) from the melody
    stage's cache (``cache/melody.f0.npz``), or None if it is missing, unreadable or was made from
    another vocal stem. For drawing the voice's real pitch; numpy only (no torch)."""
    from karaoke.melody.notes import hz_to_midi
    from karaoke.store import file_hash

    if not p.melody_f0.exists():
        return None
    try:
        with np.load(p.melody_f0) as z:
            f0, hop_s = z["f0"].astype(float), float(z["hop_s"])
            vhash = str(z["vocals_hash"]) if "vocals_hash" in z.files else None
    except Exception:  # noqa: BLE001 — a broken cache only costs the pitch line
        return None
    if vhash and p.vocals.exists() and file_hash(p.vocals) != vhash:
        return None
    f0[~(f0 > 0)] = np.nan
    return hz_to_midi(f0), hop_s


# --------------------------------------------------------------------------- activation → notes

LEVEL_DB = 25.0  # frames this far below the stem's loud level carry no evidence (echo, leakage) ...
WORD_LEVEL_DB = 35.0  # ... inside a sung word
WORD_PAD_S = 0.1  # words are widened by this where they act as a voicing prior
CUE_S = 0.05  # a change of note is cheap within this of a word start
MIN_NOTE_S = 0.05


@dataclass
class Decoded:
    notes: list  # notes.RawNote
    tuning: float
    midi: np.ndarray  # the voice's own pitch per frame (fractional MIDI, NaN where unpitched)
    conf: np.ndarray
    syllables: list = field(default_factory=list)  # models.Syllable


def level_gate(db: np.ndarray | None, conf: np.ndarray, in_word: np.ndarray | None) -> np.ndarray:
    """1.0 where a frame is loud enough to count as evidence, else 0.0 (see ``LEVEL_DB``)."""
    from karaoke.melody.notes import LOUD_PERCENTILE

    n = len(conf)
    if db is None or not len(db) or not (conf >= 0.5).any():
        return np.ones(n)
    db = np.r_[np.asarray(db, dtype=float)[:n], np.full(max(0, n - len(db)), -120.0)]
    ref = float(np.percentile(db[conf >= 0.5], LOUD_PERCENTILE))
    floor = np.full(n, ref - LEVEL_DB)
    if in_word is not None:
        floor[in_word] = ref - WORD_LEVEL_DB
    return (db >= floor).astype(float)


def decode_job(act: np.ndarray, bin0: int, hop_s: float, db: np.ndarray | None, lyrics=None, prior=None,
               syllables: bool = False, refine: bool = True, tuning: float | None = None) -> Decoded:
    """Notes from a tracker's activation (``decode``), with the lyrics' word timings as a prior.

    Without ``syllables`` the notes go where the voice is heard: a sung word only lowers the
    evidence a note needs and makes a change of note cheap at its start."""
    from karaoke.melody import decode as D
    from karaoke.melody import notes as N

    # Without lyric anchors, a smaller change cost preserves fast solo notes (vocadito).
    prior = prior or (D.Prior() if lyrics is not None else D.Prior(weight=1, change=2, on=0.5, off=0.5))
    T = len(act)
    words = [(w.start, w.end) for ln in lyrics.lines for w in ln.words] if lyrics is not None else []
    midi, conf = D.peak_pitch(act, bin0)
    in_word = N.word_mask(words, T, hop_s, WORD_PAD_S) if words else None
    gate = level_gate(db, conf, in_word)
    if tuning is None:
        tuning = round(N.estimate_tuning(np.where(gate > 0, midi, np.nan), conf, hop_s, db, prof=N.PROFILES["crepe"]), 3)
    from karaoke.melody.track import bin_to_midi

    lo = int(np.ceil(bin_to_midi(bin0) - tuning + 0.5))
    hi = int(np.floor(bin_to_midi(bin0 + act.shape[1] - 1) - tuning - 0.5))
    S = D.semitone_salience(act, bin0, tuning, lo, hi) * gate[:, None].astype(np.float32)
    rest = np.full(T, prior.voiced)
    change = np.full(T, prior.change)
    starts = np.array(sorted({int(round(s / hop_s)) for s, _ in words}), dtype=int)
    if words:
        rest[in_word] = prior.word_voiced
        k = int(round(CUE_S / hop_s))
        for f in starts:
            change[max(0, f - k):min(T, f + k + 1)] = prior.syllable
    path = D.viterbi(S, rest, change, prior, hop_s)
    if syllables and lyrics is not None:
        from karaoke.melody.syllables import decode_syllables

        notes, spans = decode_syllables(S, lo, act, bin0, hop_s, tuning, lyrics, path, prior)
        if refine:
            raw = [N.RawNote(n.start, n.end, n.midi, n.pitch, n.conf) for n in notes]
            N.refine_by_profile(raw, midi - tuning, hop_s)
            for note, revised in zip(notes, raw, strict=True):
                note.midi = revised.midi
                a = max(0, int(np.ceil(note.start / hop_s)))
                b = min(T, int(np.ceil(note.end / hop_s)))
                if b > a:
                    note.conf = float(S[a:b, np.clip(note.midi - lo, 0, S.shape[1] - 1)].mean())
        return Decoded(notes, tuning, midi, conf, spans)
    notes = D.path_notes(path, lo, hop_s, S, act, bin0, tuning)
    notes = [n for n in notes if n.dur >= MIN_NOTE_S - 1e-9]
    notes = N.drop_outliers(notes)
    if words:
        notes = N.split_at_words(notes, [s for s, _ in words])
    if refine:
        N.refine_by_profile(notes, midi - tuning, hop_s)
    for n in notes:
        n.start, n.end, n.pitch, n.conf = round(n.start, 3), round(n.end, 3), round(n.pitch, 2), round(n.conf, 3)
    return Decoded(notes, tuning, midi, conf)
