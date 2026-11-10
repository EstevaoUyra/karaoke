"""Score melody.json against the human-made references in data/gold (see data/gold/README.md).

    uv run python spikes/score_gold.py [SONG_ID ...] [--tol 0.15] [--json out.json]

The gold set is local only (gitignored). For each song whose job has a melody.json:

- the reference notes are moved to the audio's time and pitch
  (``t = start * time_scale + offset_s``, ``midi + midi_offset``);
- only the reference's own span is scored (Hooktheory entries cover one section of the song).

Per reference note, "our pitch" is the melody.json note that sounds longest during the first
``PITCH_WINDOW_S`` of the reference note (the references have no trustworthy note ends).

Columns:
- ``ref``: reference notes scored; ``ours``: melody.json notes in the same span.
- ``found``: share of reference notes during which we have any note.
- ``pitch``: share of reference notes where our note is the same semitone (missing counts as wrong).
- ``±1``: within one semitone. ``pc``: same pitch class in any octave. ``oct``: right pitch class,
  wrong octave. For references with an unknown octave only ``pc`` is meaningful.
- ``pitch|found``: the same semitone, over the notes we found: pitch quality apart from coverage.
- ``on_P`` / ``on_R`` / ``on_F``: onsets matched one-to-one within ``--tol`` seconds.
- ``loose``: timing-tolerant ``pitch``: one of our notes has the reference's semitone and sounds
  for at least 50 ms between ``TOL_S`` before the reference onset and ``TOL_S`` after the pitch
  window. The references' onsets are quantised to a grid (17-30 % sit more than 150 ms from ours,
  with no systematic lead or lag), so ``pitch`` is too strict and ``loose`` a little too kind
  (a repeated pitch next door also counts); the truth is in between.
- ``pitch≥`` / ``loose≥``: the same two over reference notes at least ``--min-dur`` seconds long
  (the time to the next onset where the reference has no note ends).

- ``sure``: share of reference notes where our note has confidence of at least ``SURE_CONF`` (the
  guide draws the others faint); ``pitch|sure`` / ``loose|sure``: the two scores over those.

After the songs comes the same split by note length, over all songs together.

- ``lag``: the time shift of our notes (within ±0.3 s) that maximises ``pitch``, and that best
  ``pitch``: a check on the reference's alignment, not a score.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

GOLD = Path("data/gold")
JOBS = Path("data/jobs")
PITCH_WINDOW_S = 0.5
PITCH_SKIP_S = 0.03  # the first moments of a note are the consonant / the scoop
SURE_CONF = 0.4  # as render.roll.UNSURE_CONF: notes below it are drawn faint in the guide
TOL_S = 0.15  # timing slack of the loose score
BUCKETS = ((0.0, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 99.0))  # note lengths, seconds


@dataclass
class N:
    start: float
    end: float
    midi: int


def load_reference(ref: dict) -> tuple[list[N], list[tuple[float, float]], bool]:
    scale, off = ref.get("time_scale") or 1.0, ref.get("offset_s") or 0.0
    shift = ref.get("midi_offset")
    notes = [N(n["start"] * scale + off, n["end"] * scale + off, round(n["midi"]) + (shift or 0))
             for n in ref["notes"]]
    segs = (ref.get("annotation") or {}).get("segments")
    if segs:
        spans = [(s["start"] * scale + off, s["end"] * scale + off) for s in segs]
    else:
        spans = [(min(n.start for n in notes) - 1.0, max(n.end for n in notes) + 1.0)]
    return notes, spans, shift is not None


def in_spans(t: float, spans: list[tuple[float, float]]) -> bool:
    return any(a <= t <= b for a, b in spans)


def our_pitch(ref: N, ours: list[N], lag: float = 0.0) -> int | None:
    a = ref.start + PITCH_SKIP_S
    b = ref.start + max(0.08, min(ref.end - ref.start, PITCH_WINDOW_S))
    best, best_overlap = None, 0.0
    for n in ours:
        overlap = min(b, n.end + lag) - max(a, n.start + lag)
        if overlap > best_overlap:
            best, best_overlap = n.midi, overlap
    return best


def loose_hit(ref: N, ours: list[N]) -> bool:
    a = ref.start - TOL_S
    b = ref.start + max(0.08, min(ref.end - ref.start, PITCH_WINDOW_S)) + TOL_S
    return any(n.midi == ref.midi and min(b, n.end) - max(a, n.start) >= 0.05 for n in ours)


def per_note(refs: list[N], ours: list[N]) -> list[tuple[float, bool, bool, bool]]:
    """(length, found, same semitone, loose hit) for every reference note."""
    out = []
    for r in refs:
        got = our_pitch(r, ours)
        out.append((r.end - r.start, got is not None, got == r.midi, got == r.midi or loose_hit(r, ours)))
    return out


def pitch_scores(refs: list[N], ours: list[N], lag: float = 0.0) -> dict[str, float]:
    got = [our_pitch(r, ours, lag) for r in refs]
    n = len(refs)
    found = [(r, g) for r, g in zip(refs, got, strict=True) if g is not None]
    same = sum(g == r.midi for r, g in found)
    return {
        "found": len(found) / n,
        "pitch": same / n,
        "near": sum(abs(g - r.midi) <= 1 for r, g in found) / n,
        "pc": sum((g - r.midi) % 12 == 0 for r, g in found) / n,
        "oct": sum((g - r.midi) % 12 == 0 and g != r.midi for r, g in found) / n,
        "pitch_found": same / len(found) if found else 0.0,
    }


def onset_scores(refs: list[N], ours: list[N], tol: float) -> dict[str, float]:
    """Greedy one-to-one matching in time order."""
    a, b = sorted(r.start for r in refs), sorted(o.start for o in ours)
    i = j = hits = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tol:
            hits, i, j = hits + 1, i + 1, j + 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    p = hits / len(b) if b else 0.0
    r = hits / len(a) if a else 0.0
    return {"on_P": p, "on_R": r, "on_F": 2 * p * r / (p + r) if p + r else 0.0}


def score(song: Path, tol: float, min_dur: float = 0.2) -> dict | None:
    ref = json.loads((song / "reference.json").read_text())
    job = (ref.get("audio") or {}).get("job_id")
    mel = JOBS / job / "melody.json" if job else None
    if not mel or not mel.exists():
        return None
    refs, spans, octave_known = load_reference(ref)
    refs = [r for r in refs if in_spans(r.start, spans)]
    raw = json.loads(mel.read_text())["notes"]
    ours_all = [N(n["start"], n["end"], n["midi"]) for n in raw]
    sure = [N(n["start"], n["end"], n["midi"]) for n in raw if n.get("conf", 1.0) >= SURE_CONF]
    ours = [o for o in ours_all if in_spans(o.start, spans)]
    out = {"song": song.name, "source": ref["source"].split(",")[0].split(" (")[0][:12], "octave_known": octave_known,
           "ref": len(refs), "ours": len(ours)}
    out.update(pitch_scores(refs, ours_all))
    notes = per_note(refs, ours_all)
    long = [x for x in notes if x[0] >= min_dur]
    out["loose"] = sum(x[3] for x in notes) / len(notes)
    out["pitch_long"] = sum(x[2] for x in long) / len(long) if long else 0.0
    out["loose_long"] = sum(x[3] for x in long) / len(long) if long else 0.0
    out["long_share"] = len(long) / len(notes)
    out["notes"] = notes
    # reference notes where the note we show is one the tracker was sure of
    kept = [(r, our_pitch(r, ours_all)) for r in refs if our_pitch(r, sure) == our_pitch(r, ours_all) is not None]
    out["sure"] = len(kept) / len(refs)
    out["pitch_sure"] = sum(p == r.midi for r, p in kept) / len(kept) if kept else 0.0
    out["loose_sure"] = sum(p == r.midi or loose_hit(r, ours_all) for r, p in kept) / len(kept) if kept else 0.0
    out.update(onset_scores(refs, ours, tol))
    key = "pitch" if octave_known else "pc"
    lags = [k / 100 for k in range(-30, 31, 2)]
    best = max(lags, key=lambda lag: (pitch_scores(refs, ours_all, lag)[key], -abs(lag)))
    out["lag"], out["lag_best"] = best, pitch_scores(refs, ours_all, best)[key]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("songs", nargs="*")
    ap.add_argument("--tol", type=float, default=0.15, help="onset tolerance in seconds")
    ap.add_argument("--min-dur", type=float, default=0.2, help="'long' notes are at least this long (s)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    dirs = sorted(d for d in GOLD.iterdir() if (d / "reference.json").exists() and (not args.songs or d.name in args.songs))
    rows = [r for d in dirs if (r := score(d, args.tol, args.min_dur))]
    cols = ["found", "pitch", "loose", "pitch_long", "loose_long", "sure", "pitch_sure", "loose_sure", "near", "oct", "on_F"]
    head = {"near": "±1", "pitch_long": f"pitch≥{args.min_dur:g}s", "loose_long": f"loose≥{args.min_dur:g}s",
            "pitch_sure": "pitch|sure", "loose_sure": "loose|sure"}
    print(f"{'song':24s} {'ref':>4s} {'ours':>4s} " + " ".join(f"{head.get(c, c):>10s}" for c in cols) + "   lag")
    for r in rows:
        mark = "" if r["octave_known"] else "  (octave unknown: read pc)"
        print(f"{r['song']:24s} {r['ref']:4d} {r['ours']:4d} " + " ".join(f"{r[c]:10.0%}" for c in cols)
              + f"  {r['lag']:+.2f}s → {r['lag_best']:.0%}{mark}")
    known = [r for r in rows if r["octave_known"]]
    if known:
        means = " ".join(f"{sum(r[c] for r in known) / len(known):10.0%}" for c in cols)
        print(f"{'mean (octave known)':24s} {'':9s} {means}")
        pooled = [x for r in known for x in r["notes"]]
        print(f"\nby note length, all {len(pooled)} reference notes of those songs together:")
        print(f"{'length':>14s} {'notes':>6s} {'share':>6s} {'found':>6s} {'pitch':>6s} {'loose':>6s}")
        for lo, hi in BUCKETS:
            part = [x for x in pooled if lo <= x[0] < hi]
            if part:
                name = f"{lo:g}-{hi:g} s" if hi < 99 else f"over {lo:g} s"
                print(f"{name:>14s} {len(part):6d} {len(part) / len(pooled):6.0%} " + " ".join(
                    f"{sum(x[k] for x in part) / len(part):6.0%}" for k in (1, 2, 3)))
    for r in rows:
        r.pop("notes")
    skipped = [d.name for d in dirs if d.name not in {r["song"] for r in rows}]
    if skipped:
        print("no melody.json (not scored):", ", ".join(skipped))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
