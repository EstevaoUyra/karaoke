"""Count the large jumps between consecutive notes of the pitch guide, and how many are wrong.

    uv run python spikes/melody_jumps.py [--melody melody.json] [--min 7] [--gap 0.5] [--list SONG]

Real melodies mostly repeat a pitch or move by one or two semitones, so a guide that often leaps a
fifth or more is suspect. For every job in ``data/jobs`` with the melody file:

- ``pairs``: consecutive notes with at most ``--gap`` seconds between them (a phrase; a leap across
  a long rest is not what a listener hears as a jump);
- ``jumps``: pairs whose pitches differ by at least ``--min`` semitones (7 = a fifth).

Where the job has a human reference (``data/gold``, see spikes/score_gold.py) each jump is judged
against it. The reference note under each of our two notes is the one that overlaps it longest:

- ``right``: both of our notes are the reference's semitone (a real leap, drawn correctly);
- ``leap``: the reference also moves at least a fourth (5) in the same direction: there is a leap,
  our pitches are off;
- ``wrong``: the reference moves a major third or less, or the other way: the jump is ours alone;
- ``noref``: no reference note under one of the two notes.

The reference's own share of jumps (``ref jumps``) is the base rate: how often this melody really
leaps. The same is printed for vocadito (solo clips, exact notes) with ``--vocadito`` when the
notes file from ``score_vocadito.py --dump`` exists.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

GOLD = Path("data/gold")
JOBS = Path("data/jobs")

Note = tuple[float, float, int]  # start, end, midi


def pairs(notes: list[Note], gap: float) -> list[tuple[Note, Note]]:
    return [(a, b) for a, b in zip(notes, notes[1:], strict=False) if b[0] - a[1] <= gap]


def under(n: Note, refs: list[Note]) -> Note | None:
    """The reference note that overlaps ``n`` longest (None: none does)."""
    best, best_ov = None, 0.0
    for r in refs:
        ov = min(n[1], r[1]) - max(n[0], r[0])
        if ov > best_ov:
            best, best_ov = r, ov
    return best


def judge(a: Note, b: Note, refs: list[Note]) -> str:
    ra, rb = under(a, refs), under(b, refs)
    if ra is None or rb is None:
        return "noref"
    if a[2] == ra[2] and b[2] == rb[2]:
        return "right"
    d, rd = b[2] - a[2], rb[2] - ra[2]
    return "leap" if abs(rd) >= 5 and (rd > 0) == (d > 0) else "wrong"


def interval_shares(ps: list[tuple[Note, Note]]) -> dict[str, float]:
    c: Counter[str] = Counter()
    for a, b in ps:
        d = abs(b[2] - a[2])
        bucket = ("0" if d == 0 else "1-2" if d <= 2 else "3-4" if d <= 4 else "5-6" if d <= 6
                  else "7-11" if d <= 11 else "12+")
        c[bucket] += 1
    n = max(1, len(ps))
    return {k: c[k] / n for k in ("0", "1-2", "3-4", "5-6", "7-11", "12+")}


def load_gold() -> dict[str, tuple[str, list[Note], list[tuple[float, float]]]]:
    """job id → (song name, reference notes in audio time, scored spans); octave-known songs only."""
    out = {}
    if not GOLD.exists():
        return out
    for d in sorted(GOLD.iterdir()):
        f = d / "reference.json"
        if not f.exists():
            continue
        ref = json.loads(f.read_text())
        job = (ref.get("audio") or {}).get("job_id")
        if not job or ref.get("midi_offset") is None:
            continue
        scale, off, shift = ref.get("time_scale") or 1.0, ref.get("offset_s") or 0.0, ref["midi_offset"]
        notes = [(n["start"] * scale + off, n["end"] * scale + off, round(n["midi"]) + shift) for n in ref["notes"]]
        segs = (ref.get("annotation") or {}).get("segments")
        spans = ([(s["start"] * scale + off, s["end"] * scale + off) for s in segs] if segs
                 else [(min(n[0] for n in notes) - 1.0, max(n[1] for n in notes) + 1.0)])
        notes = [n for n in notes if any(s <= n[0] <= e for s, e in spans)]
        out[job] = (d.name, notes, spans)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--melody", default="melody.json", help="file name inside each job folder")
    ap.add_argument("--min", type=int, default=7, help="a jump is at least this many semitones")
    ap.add_argument("--gap", type=float, default=0.5, help="notes further apart than this are not a pair (s)")
    ap.add_argument("--list", help="print every jump of this job id or gold song name")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    gold = load_gold()
    rows = []
    tot: Counter[str] = Counter()
    print(f"{'job':12s} {'song':24s} {'notes':>5s} {'pairs':>5s} {'jumps':>5s} {'share':>6s} {'oct+':>4s} | "
          f"{'judged':>6s} {'right':>5s} {'leap':>5s} {'wrong':>5s} {'noref':>5s} | {'ref pairs':>9s} "
          f"{'ref jumps':>9s} {'share':>6s}")
    for d in sorted(JOBS.iterdir()):
        f = d / args.melody
        if not f.exists():
            continue
        notes = [(n["start"], n["end"], n["midi"]) for n in json.loads(f.read_text())["notes"]]
        ps = pairs(notes, args.gap)
        jumps = [(a, b) for a, b in ps if abs(b[2] - a[2]) >= args.min]
        row = {"job": d.name, "song": "", "notes": len(notes), "pairs": len(ps), "jumps": len(jumps),
               "octaves": sum(abs(b[2] - a[2]) >= 12 for a, b in jumps), "intervals": interval_shares(ps)}
        line = (f"{d.name:12s} {{song:24s}} {len(notes):5d} {len(ps):5d} {len(jumps):5d} "
                f"{len(jumps) / max(1, len(ps)):6.1%} {row['octaves']:4d} | ")
        tot.update(notes=len(notes), pairs=len(ps), jumps=len(jumps))
        if d.name in gold:
            name, refs, spans = gold[d.name]
            inside = [(a, b) for a, b in jumps if any(s <= a[0] and b[1] <= e for s, e in spans)]
            verdicts = Counter(judge(a, b, refs) for a, b in inside)
            rps = pairs(refs, args.gap)
            rj = sum(abs(b[2] - a[2]) >= args.min for a, b in rps)
            ours_in = [(a, b) for a, b in ps if any(s <= a[0] and b[1] <= e for s, e in spans)]
            row.update(song=name, judged=len(inside), **{k: verdicts[k] for k in ("right", "leap", "wrong", "noref")},
                       pairs_in_ref=len(ours_in), ref_pairs=len(rps), ref_jumps=rj, ref_intervals=interval_shares(rps))
            line = line.format(song=name) + (
                f"{len(inside):6d} {verdicts['right']:5d} {verdicts['leap']:5d} {verdicts['wrong']:5d} "
                f"{verdicts['noref']:5d} | {len(rps):9d} {rj:9d} {rj / max(1, len(rps)):6.1%}")
            tot.update(judged=len(inside), ref_pairs=len(rps), ref_jumps=rj, pairs_in_ref=len(ours_in),
                       **{k: verdicts[k] for k in ("right", "leap", "wrong", "noref")})
            if args.list in (d.name, name):
                for a, b in inside:
                    ra, rb = under(a, refs), under(b, refs)
                    print(f"    {a[0]:7.2f}-{a[1]:7.2f} {a[2]:3d} → {b[0]:7.2f}-{b[1]:7.2f} {b[2]:3d}  ({b[2] - a[2]:+3d})  "
                          f"ref {ra and ra[2]} → {rb and rb[2]}  {judge(a, b, refs)}")
        else:
            line = line.format(song="")
            if args.list == d.name:
                for a, b in jumps:
                    print(f"    {a[0]:7.2f}-{a[1]:7.2f} {a[2]:3d} → {b[0]:7.2f}-{b[1]:7.2f} {b[2]:3d}  ({b[2] - a[2]:+3d})")
        print(line)
        rows.append(row)
    print(f"\nall jobs: {tot['jumps']} jumps in {tot['pairs']} pairs ({tot['jumps'] / max(1, tot['pairs']):.1%})")
    if tot["judged"]:
        judged = tot["judged"] - tot["noref"]
        print(f"with a reference: {tot['judged']} jumps in {tot['pairs_in_ref']} pairs "
              f"({tot['judged'] / max(1, tot['pairs_in_ref']):.1%}); the references themselves: {tot['ref_jumps']} in "
              f"{tot['ref_pairs']} ({tot['ref_jumps'] / max(1, tot['ref_pairs']):.1%})")
        print(f"  right {tot['right']} ({tot['right'] / max(1, judged):.0%}), leap but pitch off {tot['leap']} "
              f"({tot['leap'] / max(1, judged):.0%}), wrong {tot['wrong']} ({tot['wrong'] / max(1, judged):.0%}) "
              f"of the {judged} with a reference under both notes; {tot['noref']} without")
    keys = ("0", "1-2", "3-4", "5-6", "7-11", "12+")
    print("\nintervals between consecutive notes (semitones), share of pairs:")
    print(f"{'':28s} " + " ".join(f"{k:>6s}" for k in keys))
    for label, sel, field in (("ours, all jobs", rows, "intervals"),
                              ("ours, jobs with a reference", [r for r in rows if r["song"]], "intervals"),
                              ("the references", [r for r in rows if r["song"]], "ref_intervals")):
        w = [r["ref_pairs"] if field == "ref_intervals" else r["pairs"] for r in sel]
        if sum(w):
            print(f"{label:28s} " + " ".join(
                f"{sum(r[field][k] * wi for r, wi in zip(sel, w, strict=True)) / sum(w):6.1%}" for k in keys))
    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "total": dict(tot)}, indent=2))


if __name__ == "__main__":
    main()
