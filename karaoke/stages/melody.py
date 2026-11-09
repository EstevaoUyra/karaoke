"""Stage — melody: the sung melody and the song's key, for the pitch guide.

Inputs: ``stems/vocals.wav``, ``stems/instrumental.wav`` (chroma for the key, optional),
``lyrics.final.json`` (word timings, optional). Output: ``melody.json``.

1. pyin f0 + voicing probability on the vocal stem, with the parameters of
   ``karaoke.align.voicing`` (16 kHz, 40 ms hop, 70-900 Hz, 1024-sample frames). It runs in 30 s
   chunks (progress, bounded memory) and is cached in ``cache/melody.f0.npz`` by the stem's hash,
   so re-segmenting after a lyrics edit takes a second.
2. ``karaoke.melody.notes.segment``: notes (see there).
3. Key: Krumhansl–Schmuckler (``karaoke.melody.theory``) on the notes' pitch classes weighted by
   duration x confidence, averaged with the same on the instrumental's energy-weighted chroma (the
   harmony is a steadier key cue than a melody, and rap has hardly any melody: Sacal's verses sit
   on three pitches).
4. The display octave shift: the notes' (duration-weighted) median octave against a treble staff.
"""

from __future__ import annotations

import numpy as np

from karaoke.pipeline import StageContext

CHUNK_S = 30.0
CHUNK_PAD_S = 1.0
FMIN, FMAX, FRAME = 70.0, 900.0, 1024  # as karaoke.align.voicing.voicing
CHROMA_WEIGHT = 1.0  # instrumental chroma vs vocal notes in the key estimate
CHROMA_SR = 22050
CHROMA_MAX_S = 600.0
TRACK_VERSION = 1  # bump when pitch_track changes: invalidates cache/melody.f0.npz


def pitch_track(signal: np.ndarray, sr: int, progress=None) -> tuple[np.ndarray, np.ndarray, float]:
    """(f0 Hz with NaN when unvoiced, voicing probability, hop seconds) of a mono signal."""
    import librosa

    from karaoke.align.voicing import PITCH_HOP_S, PITCH_SR

    y = librosa.resample(signal, orig_sr=sr, target_sr=PITCH_SR) if sr != PITCH_SR else signal
    hop = int(PITCH_HOP_S * PITCH_SR)
    n_frames = 1 + len(y) // hop
    f0 = np.full(n_frames, np.nan, dtype=np.float32)
    prob = np.zeros(n_frames, dtype=np.float32)
    chunk = int(CHUNK_S / PITCH_HOP_S)  # frames per chunk
    pad = int(CHUNK_PAD_S / PITCH_HOP_S)
    for c0 in range(0, n_frames, chunk):
        c1 = min(n_frames, c0 + chunk)
        a = max(0, c0 - pad)
        seg = y[a * hop: (c1 + pad) * hop]
        if len(seg) < FRAME:
            continue
        f, _, p = librosa.pyin(seg, fmin=FMIN, fmax=FMAX, sr=PITCH_SR, frame_length=FRAME,
                               hop_length=hop, center=True)
        k0 = c0 - a
        k1 = min(len(f), k0 + (c1 - c0))
        f0[c0:c0 + k1 - k0] = f[k0:k1]
        prob[c0:c0 + k1 - k0] = np.nan_to_num(p[k0:k1])
        if progress:
            progress(c1 / n_frames)
    return f0, prob, PITCH_HOP_S


def frame_db(signal: np.ndarray, sr: int, hop_s: float, n: int) -> np.ndarray:
    from karaoke.audio import rms_db

    db = rms_db(signal, sr, frame_s=FRAME / 16000, hop_s=hop_s)
    return np.r_[db, np.full(max(0, n - len(db)), -120.0)][:n]


def instrumental_chroma(path) -> np.ndarray | None:
    """Energy-weighted mean chroma (12 bins, C first) of the instrumental, or None."""
    import librosa

    y, sr = librosa.load(str(path), sr=CHROMA_SR, mono=True, duration=CHROMA_MAX_S)
    if len(y) < sr * 5:
        return None
    hop = 4096
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=8192, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=8192, hop_length=hop)[0]
    n = min(chroma.shape[1], len(rms))
    w = rms[:n]
    if w.sum() <= 0:
        return None
    return (chroma[:, :n] * w).sum(axis=1) / w.sum()


def weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2)])


def run(ctx: StageContext) -> None:
    from karaoke.audio import read_mono
    from karaoke.melody.notes import estimate_tuning, hz_to_midi, segment
    from karaoke.melody.theory import detect_key, display_shift, pc_histogram
    from karaoke.models import KeyEstimate, LyricsFinal, Melody, Note
    from karaoke.store import file_hash, read_model_opt, write_model

    p = ctx.paths
    if not p.vocals.exists():
        raise FileNotFoundError(f"missing {p.rel(p.vocals)}: run the separate stage first")

    ctx.progress(0.01, "reading the vocal stem")
    signal, sr = read_mono(p.vocals)
    vhash = file_hash(p.vocals)
    cached = None
    if p.melody_f0.exists():  # valid whatever --force says: keyed by the stem's hash
        try:
            z = np.load(p.melody_f0)
            if str(z["vocals_hash"]) == vhash and int(z["version"]) == TRACK_VERSION:
                cached = (z["f0"], z["prob"], float(z["hop_s"]))
        except Exception as exc:  # noqa: BLE001 — a broken cache is recomputed
            ctx.log.warning("ignoring unreadable %s: %s", p.rel(p.melody_f0), exc)
    if cached is None:
        ctx.progress(0.02, "tracking pitch (pyin)")
        f0, prob, hop_s = pitch_track(signal, sr, lambda f: ctx.progress(0.02 + 0.8 * f, "tracking pitch (pyin)"))
        p.melody_f0.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.melody_f0.with_name(".melody.f0.tmp.npz")
        np.savez_compressed(tmp, f0=f0, prob=prob, hop_s=hop_s, vocals_hash=vhash, version=TRACK_VERSION)
        tmp.replace(p.melody_f0)
    else:
        ctx.log.info("pitch contour from cache (%s)", p.rel(p.melody_f0))
        f0, prob, hop_s = cached
    db = frame_db(signal, sr, hop_s, len(f0))

    lyrics = read_model_opt(p.lyrics_final, LyricsFinal)
    words = [(w.start, w.end) for ln in lyrics.lines for w in ln.words] if lyrics else None
    ctx.progress(0.85, "segmenting notes")
    midi = hz_to_midi(f0)
    tuning = round(estimate_tuning(midi, prob, hop_s, db), 3)
    raw = segment(midi, prob, hop_s, db=db, words=words, tuning=tuning)
    notes = [Note(start=n.start, end=n.end, midi=n.midi, pitch=n.pitch, conf=n.conf) for n in raw]

    ctx.progress(0.9, "estimating the key")
    hist = pc_histogram([(n.midi, (n.end - n.start) * n.conf) for n in notes])
    vocal = hist / hist.sum() if hist.sum() > 0 else hist
    chroma = None
    if p.instrumental.exists():
        try:
            chroma = instrumental_chroma(p.instrumental)
        except Exception as exc:  # noqa: BLE001 — the key then comes from the melody alone
            ctx.log.warning("instrumental chroma failed: %s", exc)
    res = detect_key(vocal, chroma, factors=[1.0, CHROMA_WEIGHT])
    params: dict[str, float | int | str | bool] = {"pyin_fmin": FMIN, "pyin_fmax": FMAX, "chroma": chroma is not None,
                                                   "words": bool(words)}
    for label, w in (("vocal", vocal), ("chroma", chroma)):
        r = detect_key(w) if w is not None else None
        if r is not None:
            params[f"key_{label}"] = f"{r.key.name} ({r.r:.2f})"
    key = None
    if res is not None:
        key = KeyEstimate(tonic=res.key.tonic, tonic_pc=res.key.tonic_pc, mode=res.key.mode,
                          fifths=res.key.fifths, accidentals=res.key.signature,
                          confidence=round(res.r, 3), margin=round(res.margin, 3), runner_up=res.runner_up.name)

    median = None
    shift = 0
    if notes:
        median = round(weighted_median([n.pitch for n in notes], [n.end - n.start for n in notes]), 2)
        shift = display_shift(median)
    write_model(p.melody, Melody(hop_s=hop_s, notes=notes, key=key, median_midi=median, tuning=tuning,
                                 display_octave_shift=shift, params=params))
    sung = sum(n.end - n.start for n in notes)
    ctx.log.info("key %s (r=%.2f, runner-up %s), %s", key and f"{key.tonic} {key.mode}", res.r if res else 0,
                 key and key.runner_up, {k: v for k, v in params.items() if k.startswith("key_")})
    ctx.progress(1.0, f"{len(notes)} notes ({sung:.0f} s), key {key.tonic + ' ' + key.mode if key else '?'}, "
                      f"median MIDI {median}, octave shift {shift:+d}")
