"""Stage — melody: the sung melody and the song's key, for the pitch guide.

Inputs: ``stems/vocals.wav``, ``stems/instrumental.wav`` and ``audio.wav`` (chroma for the key,
optional), ``lyrics.final.json`` (word timings, optional). Output: ``melody.json``.

1. F0 of the vocal stem with ``Settings.melody_tracker`` (``karaoke.melody.track``): torchcrepe
   by default (see docs/benchmarks.md, "Melody"), pyin as the fallback when torch is missing or
   fails. The contour is cached in ``cache/melody.f0.npz`` by the stem's hash and the tracker, so
   re-segmenting after a lyrics edit takes a second. CREPE imports torch, so this stage runs in a
   child process (``isolated=True``): torch must never share a process with ctranslate2.
2. ``karaoke.melody.notes.segment`` with the tracker's thresholds and the words as a prior.
3. Key: Krumhansl–Schmuckler (``karaoke.melody.theory``), the correlations of three votes
   averaged: the notes' pitch classes (duration x confidence), the instrumental's chroma (the
   harmony is a steadier cue than a melody, and rap has hardly any melody) and the full mix's.
4. The display octave shift: the notes' (duration-weighted) median octave against a treble staff.
"""

from __future__ import annotations

import numpy as np

from karaoke.pipeline import StageContext

CHROMA_SR = 22050
CHROMA_MAX_S = 600.0
KEY_VOTES = {"vocal": 1.0, "instrumental": 1.0, "mix": 1.0}
TRACK_VERSION = 2  # bump when a tracker changes: invalidates cache/melody.f0.npz


def pitch_track(signal: np.ndarray, sr: int, tracker: str, progress=None) -> tuple[np.ndarray, np.ndarray, float]:
    """(f0 Hz, confidence, hop seconds) with ``tracker`` ("crepe-full", "crepe-tiny" or "pyin")."""
    from karaoke.melody.track import crepe_track, pyin_track

    if tracker.startswith("crepe-"):
        return crepe_track(signal, sr, model=tracker.split("-", 1)[1], progress=progress)
    return pyin_track(signal, sr, progress=progress)


def audio_chroma(path) -> np.ndarray | None:
    """Energy-weighted mean chroma (12 bins, C first) of an audio file, or None."""
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


instrumental_chroma = audio_chroma  # the name tests patch


def weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2)])


def _contour(ctx: StageContext, signal: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float, str]:
    from karaoke.store import file_hash

    p = ctx.paths
    tracker = ctx.settings.melody_tracker
    vhash = file_hash(p.vocals)
    if p.melody_f0.exists():  # valid whatever --force says: keyed by the stem's hash and tracker
        try:
            z = np.load(p.melody_f0)
            if (str(z["vocals_hash"]) == vhash and int(z["version"]) == TRACK_VERSION
                    and str(z["tracker"]) == tracker):
                ctx.log.info("pitch contour from cache (%s, %s)", p.rel(p.melody_f0), tracker)
                return z["f0"], z["conf"], float(z["hop_s"]), tracker
        except Exception as exc:  # noqa: BLE001 — a broken or old cache is recomputed
            ctx.log.info("not using %s: %s", p.rel(p.melody_f0), exc)
    used = tracker
    msg = f"tracking pitch ({tracker})"
    ctx.progress(0.02, msg)
    try:
        f0, conf, hop_s = pitch_track(signal, sr, tracker, lambda f: ctx.progress(0.02 + 0.8 * f, msg))
    except Exception as exc:
        if tracker == "pyin":
            raise
        ctx.log.warning("%s failed (%s); falling back to pyin", tracker, exc)
        used, msg = "pyin", "tracking pitch (pyin fallback)"
        f0, conf, hop_s = pitch_track(signal, sr, "pyin", lambda f: ctx.progress(0.02 + 0.8 * f, msg))
    p.melody_f0.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.melody_f0.with_name(".melody.f0.tmp.npz")
    if used == tracker:  # a fallback contour is not cached under the requested tracker
        np.savez_compressed(tmp, f0=f0, conf=conf, hop_s=hop_s, vocals_hash=vhash, version=TRACK_VERSION,
                            tracker=tracker)
        tmp.replace(p.melody_f0)
    return f0, conf, hop_s, used


def run(ctx: StageContext) -> None:
    from karaoke.audio import read_mono
    from karaoke.melody.pipeline import notes_from_contour
    from karaoke.melody.theory import detect_key, display_shift, pc_histogram
    from karaoke.models import KeyEstimate, Melody, Note
    from karaoke.store import write_model

    p = ctx.paths
    if not p.vocals.exists():
        raise FileNotFoundError(f"missing {p.rel(p.vocals)}: run the separate stage first")

    ctx.progress(0.01, "reading the vocal stem")
    signal, sr = read_mono(p.vocals)
    f0, conf, hop_s, tracker = _contour(ctx, signal, sr)

    ctx.progress(0.85, "segmenting notes")
    raw, tuning = notes_from_contour(p, f0, conf, hop_s, tracker, signal=(signal, sr))
    notes = [Note(start=n.start, end=n.end, midi=n.midi, pitch=n.pitch, conf=n.conf) for n in raw]

    ctx.progress(0.9, "estimating the key")
    hist = pc_histogram([(n.midi, (n.end - n.start) * n.conf) for n in notes])
    votes: dict[str, np.ndarray | None] = {"vocal": hist / hist.sum() if hist.sum() > 0 else None}
    for name, path in (("instrumental", p.instrumental), ("mix", p.audio)):
        votes[name] = None
        if path.exists():
            try:
                votes[name] = instrumental_chroma(path)
            except Exception as exc:  # noqa: BLE001 — the key then comes from the other votes
                ctx.log.warning("%s chroma failed: %s", name, exc)
    names = list(votes)
    res = detect_key(*(votes[k] for k in names), factors=[KEY_VOTES[k] for k in names])
    params: dict[str, float | int | str | bool] = {
        "tracker": tracker, "words": p.lyrics_final.exists(),
        "chroma": votes["instrumental"] is not None, "mix_chroma": votes["mix"] is not None}
    for k in names:
        r = detect_key(votes[k]) if votes[k] is not None else None
        if r is not None:
            params[f"key_{k}"] = f"{r.key.name} ({r.r:.2f})"
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
    write_model(p.melody, Melody(hop_s=round(hop_s, 4), notes=notes, key=key, median_midi=median, tuning=tuning,
                                 display_octave_shift=shift, params=params))
    sung = sum(n.end - n.start for n in notes)
    ctx.log.info("key %s (r=%.2f, runner-up %s), %s", key and f"{key.tonic} {key.mode}", res.r if res else 0,
                 key and key.runner_up, {k: v for k, v in params.items() if k.startswith("key_")})
    ctx.progress(1.0, f"{len(notes)} notes ({sung:.0f} s, {tracker}), key "
                      f"{key.tonic + ' ' + key.mode if key else '?'}, median MIDI {median}, octave shift {shift:+d}")
