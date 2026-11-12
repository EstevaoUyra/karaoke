"""Score the note-making (karaoke.melody.notes.segment) on vocadito: 40 solo-voice clips with
exact, human-made notes by two annotators. The cleanest measure we have of tracker + note-making,
with no separation and no lyrics involved.

    uv run python spikes/score_vocadito.py [--tracker crepe-tiny]

The dataset is local only: ``data/gold/_sources/vocadito/extracted`` (see data/gold/README.md;
https://zenodo.org/records/5578807, CC BY 4.0). Contours are cached in ``data/gold/_cache``.
Imports torch for the CREPE trackers: run it on its own.

Annotator A1 is the reference. The first row scores annotator A2 against A1 with the same
measures: that is how far two careful people agree, so it is the ceiling.

Columns (means over the 40 clips):
- ``found``: share of reference notes during which we have a note.
- ``pitch``: share of reference notes where our overlapping note's median pitch is within 50 cents.
- ``label``: the same for our semitone label (plus the clip's tuning offset): what the guide shows.
- ``on_F``: note onsets matched one-to-one within 50 ms.
- ``note_P`` / ``note_R`` / ``note_F``: onset within 50 ms and pitch within 50 cents.
- ``notes``: our notes per reference note (below 1: fast notes are being merged or dropped).
- ``frames``: share of sung frames where the raw contour is within 50 cents of the annotated f0.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

V = Path("data/gold/_sources/vocadito/extracted")
CACHE = Path("data/gold/_cache")
CLIPS = range(1, 41)
Note = tuple[float, float, float]  # start, end, fractional MIDI


def to_midi(hz: float) -> float:
    return float(69 + 12 * np.log2(hz / 440.0))


def reference(i: int, who: str = "A1") -> list[Note]:
    with open(V / f"Annotations/Notes/vocadito_{i}_notes{who}.csv") as f:
        return [(float(a), float(a) + float(d), to_midi(float(h))) for a, h, d in csv.reader(f)]


def contour(i: int, tracker: str):
    """(fractional MIDI, confidence, hop seconds, frame loudness), tracked once and cached."""
    from karaoke.audio import read_mono
    from karaoke.melody.notes import hz_to_midi
    from karaoke.melody.pipeline import frame_db

    wav = V / f"Audio/vocadito_{i}.wav"
    sig, sr = read_mono(wav)
    path = CACHE / f"vocadito_{i}_{tracker}.npz"
    if not path.exists():
        from karaoke.melody.track import crepe_track, pyin_track

        f0, conf, hop = pyin_track(sig, sr) if tracker == "pyin" else crepe_track(sig, sr, model=tracker.split("-")[1])
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, f0=f0, conf=conf, hop_s=hop)
    z = np.load(path)
    hop = float(z["hop_s"])
    return hz_to_midi(z["f0"]), z["conf"], hop, frame_db(sig, sr, hop, len(z["f0"])), z["f0"]


def our_notes(i: int, tracker: str) -> tuple[list[Note], list[Note]]:
    """Our notes twice: with their median pitch, and with the semitone label (plus tuning)."""
    from karaoke.melody import notes as N

    midi, conf, hop, db, _ = contour(i, tracker)
    prof = N.profile(tracker)
    tuning = N.estimate_tuning(midi, conf, hop, db, prof=prof)
    ns = N.segment(midi, conf, hop, db=db, tuning=tuning, prof=prof)
    return [(n.start, n.end, n.pitch) for n in ns], [(n.start, n.end, n.midi + tuning) for n in ns]


def match(ref: list[Note], est: list[Note], tol_s: float, tol_st: float | None) -> tuple[float, float, float]:
    """Precision, recall, F of one-to-one matches (nearest onset within ``tol_s``; pitch within
    ``tol_st`` semitones unless None)."""
    used: set[int] = set()
    hits = 0
    for a, _, p in ref:
        best = None
        for k, (b, _, q) in enumerate(est):
            if k in used or abs(a - b) > tol_s or (tol_st is not None and abs(p - q) > tol_st):
                continue
            if best is None or abs(a - b) < abs(a - est[best][0]):
                best = k
        if best is not None:
            used.add(best)
            hits += 1
    p = hits / len(est) if est else 0.0
    r = hits / len(ref) if ref else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def at_note(ref: list[Note], est: list[Note]) -> tuple[float, float]:
    """(found, pitch within 50 cents) per reference note, from the estimate that overlaps it most."""
    found = ok = 0
    for a, b, p in ref:
        best, most = None, 0.0
        for s, e, q in est:
            overlap = min(b, e) - max(a, s)
            if overlap > most:
                best, most = q, overlap
        if best is not None:
            found += 1
            ok += abs(best - p) <= 0.5
    return found / len(ref), ok / len(ref)


def frame_accuracy(i: int, tracker: str) -> float:
    with open(V / f"Annotations/F0/vocadito_{i}_f0.csv") as f:
        ref = np.array([[float(x) for x in row] for row in csv.reader(f)])
    *_, hop, _, f0 = contour(i, tracker)
    t = np.arange(len(f0)) * hop
    near = ref[np.clip(np.searchsorted(ref[:, 0], t), 0, len(ref) - 1), 1]
    sung = near > 0
    ok = sung & np.isfinite(f0) & (f0 > 0)
    cents = 12 * np.abs(np.log2(f0[ok] / near[ok]))
    return float(np.sum(cents <= 0.5) / max(1, sung.sum()))


def score(get) -> dict[str, float]:
    names = ("found", "pitch", "label", "on_F", "note_P", "note_R", "note_F", "notes")
    cols: dict[str, list[float]] = {k: [] for k in names}
    for i in CLIPS:
        ref = reference(i)
        frac, lab = get(i)
        found, pitch = at_note(ref, frac)
        p, r, f = match(ref, frac, 0.05, 0.5)
        for k, v in zip(cols, (found, pitch, at_note(ref, lab)[1], match(ref, frac, 0.05, None)[2], p, r, f,
                               len(frac) / len(ref)), strict=True):
            cols[k].append(v)
    return {k: float(np.mean(v)) for k, v in cols.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracker", default="crepe-tiny", choices=["crepe-tiny", "crepe-full", "pyin"])
    args = ap.parse_args()
    if not V.exists():
        raise SystemExit(f"vocadito is not at {V}: see data/gold/README.md")
    rows = {
        "annotator A2 (ceiling)": score(lambda i: (reference(i, "A2"),) * 2),
        f"ours ({args.tracker})": score(lambda i: our_notes(i, args.tracker)),
    }
    head = list(next(iter(rows.values())))
    print(f"{'':24s}" + "".join(f"{h:>8s}" for h in head))
    for name, r in rows.items():
        print(f"{name:24s}" + "".join(f"{r[h]:8.2f}" if h == "notes" else f"{r[h]:8.0%}" for h in head))
    print(f"raw contour within 50 cents on sung frames: {np.mean([frame_accuracy(i, args.tracker) for i in CLIPS]):.1%}")


if __name__ == "__main__":
    main()
