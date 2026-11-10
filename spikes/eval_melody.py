"""Objective proxies for melody.json quality (nobody can listen in CI).

    uv run python spikes/eval_melody.py JOB [--notes melody.json ...] [--resegment TRACKER ...]

Each candidate is a list of notes: a melody.json file (``--notes``; default: the job's), or the
notes ``karaoke.melody.notes.segment`` makes *now* from a cached tracker contour
(``--resegment pyin|crepe-full|crepe-tiny``, optionally ``:profile``; contours come from spikes/melody_trackers.py, pyin
also from the stage's own cache/melody.f0.npz).

Metrics (10 ms frames):
- ``cover``: share of sung-word time (``lyrics.final.json`` words) that has a note;
  ``nonote_s``: seconds of sung-word time with no note; ``gap_words``: share of words that no
  note overlaps at all (the UI's "gaps").
- ``cover_v``: the same, over word time where the reference tracker is confidently voiced (singing
  a tracker can hear: the misses that are fixable).
- ``spill``: share of note time more than 0.3 s from any word (ad-libs, leakage, or wrong words).
- ``agree[R]``: share of note frames, where reference tracker R is confident, whose note is within
  50 cents of R's pitch (after the melody's tuning offset); ``oct[R]``: share an octave off.
  R should be a tracker the candidate was *not* made from (pyin vs crepe).
- ``chroma``: the notes are synthesised (3 harmonics) and their CQT chroma compared with the
  vocal stem's on frames where both sound: mean cosine similarity, and ``pc``, the share where the
  vocal chroma's strongest pitch class is the note's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from karaoke.store import JobStore

HOP = 0.01
REF_CONF = {"pyin": 0.5, "crepe-full": 0.5, "crepe-tiny": 0.5}
SYN_SR = 22050


def load_contour(p, tracker: str) -> tuple[np.ndarray, np.ndarray, float] | None:
    f = p.root / "cache" / f"f0_{tracker}.npz"
    if not f.exists() and tracker == "pyin" and p.melody_f0.exists():
        f = p.melody_f0
    if not f.exists():
        return None
    z = np.load(f)
    conf = z["conf"] if "conf" in z else z["prob"]
    return z["f0"].astype(float), conf.astype(float), float(z["hop_s"])


def on_grid(f0: np.ndarray, conf: np.ndarray, hop: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample a contour to the HOP grid (nearest frame)."""
    idx = np.clip(np.round(np.arange(n) * HOP / hop).astype(int), 0, len(f0) - 1)
    return f0[idx], conf[idx]


def mask_of(spans: list[tuple[float, float]], n: int, pad: float = 0.0) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in spans:
        m[max(0, int((s - pad) / HOP)):min(n, int(np.ceil((e + pad) / HOP)))] = True
    return m


def note_grid(notes: list[dict], n: int) -> np.ndarray:
    g = np.full(n, np.nan)
    for x in notes:
        g[max(0, int(x["start"] / HOP)):min(n, int(np.ceil(x["end"] / HOP)))] = x["midi"]
    return g


def synth(notes: list[dict], dur: float, tuning: float) -> np.ndarray:
    y = np.zeros(int(dur * SYN_SR) + 1, dtype=np.float32)
    for x in notes:
        a, b = int(x["start"] * SYN_SR), min(len(y), int(x["end"] * SYN_SR))
        if b <= a:
            continue
        t = np.arange(b - a) / SYN_SR
        f = 440.0 * 2 ** ((x["midi"] + tuning - 69) / 12)
        env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)
        y[a:b] += (env * sum(np.sin(2 * np.pi * f * k * t) / k for k in (1, 2, 3)) * 0.2).astype(np.float32)
    return y


def chroma_scores(notes: list[dict], tuning: float, vocal: np.ndarray, dur: float,
                  only: np.ndarray | None = None) -> tuple[float, float]:
    import librosa

    hop = 512
    cv = librosa.feature.chroma_cqt(y=vocal, sr=SYN_SR, hop_length=hop)
    cs = librosa.feature.chroma_cqt(y=synth(notes, dur, tuning), sr=SYN_SR, hop_length=hop)
    rms = librosa.feature.rms(y=vocal, hop_length=hop)[0]
    n = min(cv.shape[1], cs.shape[1], len(rms))
    cv, cs, rms = cv[:, :n], cs[:, :n], rms[:n]
    db = 20 * np.log10(np.maximum(rms, 1e-9))
    voiced = db > np.percentile(db, 95) - 30
    g = note_grid(notes, int(n * hop / SYN_SR / HOP) + 1)
    t_idx = np.minimum((np.arange(n) * hop / SYN_SR / HOP).astype(int), len(g) - 1)
    has = np.isfinite(g[t_idx]) & voiced
    if only is not None:
        has &= only[np.minimum(t_idx, len(only) - 1)]
    if not has.any():
        return float("nan"), float("nan")
    a, b = cv[:, has], cs[:, has]
    cos = (a * b).sum(0) / np.maximum(np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0), 1e-9)
    pc = np.argmax(a, axis=0) == (np.round(g[t_idx][has] + tuning).astype(int) % 12)
    return float(cos.mean()), float(pc.mean())


def evaluate(job: str, notes: list[dict], tuning: float, refs: list[str], vocal: np.ndarray | None,
             words: list[tuple[float, float]], dur: float, only: np.ndarray | None = None) -> dict[str, float]:
    """Metrics of one candidate; with ``only`` (a HOP-grid mask), the accuracy metrics (agree, oct,
    chroma, pc) are computed on those frames only, e.g. where another candidate also has notes."""
    p = JobStore().paths(job)
    n = int(dur / HOP) + 1
    W = mask_of(words, n)
    Wpad = mask_of(words, n, 0.3)
    g = note_grid(notes, n)
    N = np.isfinite(g)
    out = {"notes": len(notes), "note_s": N.sum() * HOP, "cover": (N & W).sum() / max(1, W.sum()),
           "nonote_s": (W & ~N).sum() * HOP,
           "gap_words": float(np.mean([not N[int(s / HOP):max(int(s / HOP) + 1, int(np.ceil(e / HOP)))].any()
                                       for s, e in words])) if words else float("nan"),
           "spill": (N & ~Wpad).sum() / max(1, N.sum())}
    for r in refs:
        c = load_contour(p, r)
        if c is None:
            continue
        f0, conf = on_grid(*c, n)
        ref = 69 + 12 * np.log2(np.where(np.isfinite(f0) & (f0 > 0), f0, np.nan) / 440)
        ok = np.isfinite(ref) & (conf >= REF_CONF[r])
        out[f"cover_v[{r}]"] = (N & W & ok).sum() / max(1, (W & ok).sum())
        sel = N & ok & (only[:n] if only is not None else True)
        d = ref[sel] - (g[sel] + tuning)
        out[f"agree[{r}]"] = float(np.mean(np.abs(d) <= 0.5)) if sel.any() else float("nan")
        k = np.round(d / 12)
        out[f"oct[{r}]"] = float(np.mean((k != 0) & (np.abs(d - 12 * k) <= 0.5))) if sel.any() else float("nan")
    if vocal is not None:
        out["chroma"], out["pc"] = chroma_scores(notes, tuning, vocal, dur, only)
    return out


def resegment(job: str, spec: str) -> tuple[list[dict], float]:
    """Notes the current karaoke.melody code makes from a cached contour. ``spec`` is a tracker
    name, or ``tracker:profile`` (e.g. ``pyin:pyin-v1``, the first version's thresholds)."""
    from karaoke.melody.notes import profile, with_profile
    from karaoke.melody.pipeline import notes_from_contour

    tracker, _, rest = spec.partition(":")
    name, _, over = rest.partition(":")
    prof = profile(name or tracker)
    if over:  # e.g. crepe-tiny:crepe:word_level_db=35,bridge_s=0.3
        prof = with_profile(prof, **{k: float(v) for k, v in (kv.split("=") for kv in over.split(","))})
    p = JobStore().paths(job)
    c = load_contour(p, tracker)
    if c is None:
        raise SystemExit(f"no cached contour for {tracker}: run spikes/melody_trackers.py {job} {tracker}")
    f0, conf, hop = c
    raw, tuning = notes_from_contour(p, f0, conf, hop, prof)
    return [{"start": x.start, "end": x.end, "midi": x.midi} for x in raw], tuning


def gates(job: str, spec: str) -> None:
    """Where sung-word time is lost, gate by gate, for a contour and a profile (``tracker:profile``)."""
    from karaoke.audio import read_mono
    from karaoke.melody import notes as N
    from karaoke.melody.pipeline import frame_db, job_words, notes_from_contour

    tracker, _, prof_name = spec.partition(":")
    prof = N.profile(prof_name or tracker)
    p = JobStore().paths(job)
    f0, conf, hop = load_contour(p, tracker)
    sig, sr = read_mono(p.vocals)
    db = frame_db(sig, sr, hop, len(f0))
    words = job_words(p) or []
    midi = N.hz_to_midi(f0)
    W = N.word_mask(words, len(f0), hop, 0.0)
    rng = N._range(midi, hop)
    fin = np.isfinite(midi)
    strong = fin & (conf >= prof.voiced)
    steady = fin & (conf >= prof.stable_conf) & (rng <= N.STABLE_RANGE)
    inw = N.word_mask(words, len(f0), hop, prof.word_pad)
    worded = fin & inw & (conf >= prof.word_conf) & (rng <= N.SMOOTH_RANGE) if prof.use_words else np.zeros_like(fin)
    full = N.voiced_mask(midi, conf, db, hop, inw, prof)
    no_level = N.voiced_mask(midi, conf, None, hop, inw, prof)
    tot = W.sum()
    print(f"{job} {spec}: share of sung-word time ({tot * hop:.0f} s) left after each gate")
    for name, m in (("pitch found", fin), (f"conf >= {prof.voiced}", strong), ("+ steady, lower conf", strong | steady),
                    ("+ in-word prior", strong | steady | worded), ("+ min run (no level gate)", no_level),
                    ("+ level gate", full)):
        print(f"  {name:28s} {(m & W).sum() / tot:6.3f}")
    raw, _ = notes_from_contour(p, f0, conf, hop, prof_name or tracker)
    n = int(len(f0) * hop / HOP) + 1
    g = note_grid([{"start": x.start, "end": x.end, "midi": x.midi} for x in raw], n)
    Wh = mask_of(words, n)
    print(f"  {'final notes':28s} {(np.isfinite(g) & Wh).sum() / Wh.sum():6.3f}")


def main() -> None:
    import librosa

    from karaoke.models import LyricsFinal, Meta
    from karaoke.store import read_model

    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--notes", nargs="*", default=None, help="melody.json files")
    ap.add_argument("--resegment", nargs="*", default=[], help="trackers to re-segment with the current code")
    ap.add_argument("--refs", nargs="*", default=["pyin", "crepe-full"])
    ap.add_argument("--common", default=None,
                    help="melody.json: accuracy metrics only where it has notes too (same frames for all)")
    ap.add_argument("--no-chroma", action="store_true")
    ap.add_argument("--gates", nargs="*", default=[], help="tracker[:profile]: where word time is lost")
    a = ap.parse_args()
    for spec in a.gates:
        gates(a.job, spec)
    if a.gates and a.notes is None and not a.resegment:
        return
    p = JobStore().paths(a.job)
    dur = read_model(p.meta, Meta).duration
    lf = read_model(p.lyrics_final, LyricsFinal)
    words = [(w.start, w.end) for ln in lf.lines for w in ln.words]
    vocal = None if a.no_chroma else librosa.load(str(p.vocals), sr=SYN_SR, mono=True)[0]
    cands: list[tuple[str, list[dict], float]] = []
    for f in (a.notes if a.notes is not None else ([str(p.melody)] if not a.resegment else [])):
        m = json.loads(Path(f).read_text())
        cands.append((Path(f).name, m["notes"], float(m.get("tuning", 0.0))))
    for t in a.resegment:
        notes, tuning = resegment(a.job, t)
        cands.append((f"seg:{t}", notes, tuning))
    only = None
    if a.common:
        only = np.isfinite(note_grid(json.loads(Path(a.common).read_text())["notes"], int(dur / HOP) + 1))
    rows = [(name, evaluate(a.job, notes, tuning, a.refs, vocal, words, dur, only)) for name, notes, tuning in cands]
    keys = list(dict.fromkeys(k for _, r in rows for k in r))
    print(f"{a.job}: {sum(e - s for s, e in words):.0f} s of words")
    print(f"{'candidate':24s}" + "".join(f"{k:>18s}" for k in keys))
    for name, r in rows:
        print(f"{name[:24]:24s}" + "".join(
            f"{r.get(k, float('nan')):18.3f}" if isinstance(r.get(k), float) else f"{r.get(k, ''):>18}" for k in keys))


if __name__ == "__main__":
    main()
