"""Decoder experiments: notes from CREPE's cached activation (spikes/melody_activation.py) with
``karaoke.melody.decode``, scored with the same measures as the notes in melody.json.

    uv run python spikes/melody_decode.py [--set weight=2.5 change=4 ...] [--mode free|syllables]
                                          [--tag exp] [--grid]

Writes ``melody.<tag>.json`` next to each job's melody.json (notes and tuning only) and prints
the jump counts (spikes/melody_jumps.py), the gold scores (spikes/score_gold.py) and the vocadito
scores (spikes/score_vocadito.py) in one line each. numpy only: no torch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

JOBS = Path("data/jobs")
CACHE = Path("data/gold/_cache")
TRACKER = "crepe-tiny"
_mem: dict = {}


def job_inputs(job: str):
    """(act, bin0, hop, db, lyrics) of a job, cached in memory (db also on disk)."""
    if job not in _mem:
        from karaoke.models import LyricsFinal
        from karaoke.store import JobPaths, read_model_opt

        p = JobPaths(JOBS / job)
        z = np.load(p.root / "cache" / f"melody.act.{TRACKER}.npz")
        act, bin0, hop = z["act"].astype(np.float32) / 255.0, int(z["bin0"]), float(z["hop_s"])
        dbf = p.root / "cache" / "melody.db.npy"
        if not dbf.exists():
            from karaoke.audio import read_mono
            from karaoke.melody.pipeline import frame_db

            sig, sr = read_mono(p.vocals)
            np.save(dbf, frame_db(sig, sr, hop, len(act)).astype(np.float32))
        _mem[job] = (act, bin0, hop, np.load(dbf), read_model_opt(p.lyrics_final, LyricsFinal))
    return _mem[job]


def vocadito_inputs(i: int):
    key = f"v{i}"
    if key not in _mem:
        from score_vocadito import V

        z = np.load(CACHE / f"vocadito_{i}_{TRACKER}.act.npz")
        act, bin0, hop = z["act"].astype(np.float32) / 255.0, int(z["bin0"]), float(z["hop_s"])
        dbf = CACHE / f"vocadito_{i}.db.npy"
        if not dbf.exists():
            from karaoke.audio import read_mono
            from karaoke.melody.pipeline import frame_db

            sig, sr = read_mono(V / f"Audio/vocadito_{i}.wav")
            np.save(dbf, frame_db(sig, sr, hop, len(act)).astype(np.float32))
        _mem[key] = (act, bin0, hop, np.load(dbf))
    return _mem[key]


def jobs() -> list[str]:
    return sorted(d.name for d in JOBS.iterdir() if (d / "cache" / f"melody.act.{TRACKER}.npz").exists()
                  and not d.name.startswith("demo"))


def run_jobs(prior, mode: str, tag: str, **kw) -> None:
    from karaoke.melody.pipeline import decode_job

    for job in jobs():
        act, bin0, hop, db, lyrics = job_inputs(job)
        res = decode_job(act, bin0, hop, db, lyrics if mode != "nowords" else None, prior=prior,
                         syllables=mode == "syllables", **kw)
        notes = [n.model_dump() if hasattr(n, "model_dump") else vars(n) for n in res.notes]
        doc = {"tuning": res.tuning, "notes": notes, "params": {"tracker": TRACKER, "syllables": mode == "syllables"},
               "version": 2, "hop_s": hop, "syllables": [s.model_dump() for s in res.syllables]}
        (JOBS / job / f"melody.{tag}.json").write_text(json.dumps(doc))


def coverage_scores(tag: str) -> dict:
    from karaoke.models import Melody
    from karaoke.validate import validate_melody

    words = missing = syllables = missing_syllables = errors = 0
    outside_s = total_s = 0.0
    for job in jobs():
        *_, lyrics = job_inputs(job)
        doc = json.loads((JOBS / job / f"melody.{tag}.json").read_text())
        notes = doc["notes"]
        intervals = sorted((w.start, w.end) for line in lyrics.lines for w in line.words if w.text.strip())
        words += len(intervals)
        missing += sum(not any(min(e, n["end"]) > max(a, n["start"]) for n in notes) for a, e in intervals)
        union = []
        for a, e in intervals:
            if union and a <= union[-1][1]:
                union[-1] = (union[-1][0], max(e, union[-1][1]))
            else:
                union.append((a, e))
        for n in notes:
            duration = n["end"] - n["start"]
            total_s += duration
            inside = sum(max(0, min(e, n["end"]) - max(a, n["start"])) for a, e in union)
            outside_s += max(0, duration - inside)
        if doc.get("syllables"):
            melody = Melody.model_validate(doc)
            errors += len(validate_melody(melody, lyrics))
            syllables += len(melody.syllables)
            linked = {n.syllable for n in melody.notes}
            missing_syllables += sum(i not in linked for i in range(len(melody.syllables)))
    return dict(words=words, missing_words=missing, syllables=syllables, missing_syllables=missing_syllables,
                outside_seconds=outside_s, note_seconds=total_s, validation_errors=errors)


def gold_scores(tag: str) -> dict:
    import score_gold

    rows = [r for d in sorted(score_gold.GOLD.iterdir()) if (d / "reference.json").exists()
            and (r := score_gold.score(d, 0.15, 0.2, f"melody.{tag}.json")) and r["octave_known"]]
    cols = ("found", "pitch", "loose", "near", "oct", "on_F", "pitch_sure", "sure")
    out = {c: float(np.mean([r[c] for r in rows])) for c in cols}
    out["songs"] = {r["song"]: r["pitch"] for r in rows}
    return out


def jump_scores(tag: str) -> dict:
    import melody_jumps as mj

    gold = mj.load_gold()
    tot = dict(notes=0, pairs=0, jumps=0, judged=0, right=0, leap=0, wrong=0, noref=0)
    for job in jobs():
        doc = json.loads((JOBS / job / f"melody.{tag}.json").read_text())
        notes = [(n["start"], n["end"], n["midi"]) for n in doc["notes"]]
        ps = mj.pairs(notes, 0.5)
        ju = [(a, b) for a, b in ps if abs(b[2] - a[2]) >= 7]
        tot["notes"] += len(notes)
        tot["pairs"] += len(ps)
        tot["jumps"] += len(ju)
        if job in gold:
            _, refs, spans = gold[job]
            for a, b in ju:
                if any(s <= a[0] and b[1] <= e for s, e in spans):
                    tot["judged"] += 1
                    tot[mj.judge(a, b, refs)] += 1
    return tot


def vocadito_scores(prior=None, **kw) -> dict:
    import melody_jumps as mj
    import score_vocadito as sv

    from karaoke.melody.pipeline import decode_job

    res = {}
    for i in sv.CLIPS:
        act, bin0, hop, db = vocadito_inputs(i)
        res[i] = decode_job(act, bin0, hop, db, None, prior=prior, **kw)
    out = sv.score(lambda i: ([(n.start, n.end, n.pitch) for n in res[i].notes],
                              [(n.start, n.end, n.midi + res[i].tuning) for n in res[i].notes]))
    ps = [x for i in sv.CLIPS for x in mj.pairs([(n.start, n.end, n.midi) for n in res[i].notes], 0.5)]
    out["jumps"], out["pairs"] = sum(abs(b[2] - a[2]) >= 7 for a, b in ps), len(ps)
    return out


def report(prior, mode: str, tag: str, vocadito: bool = True, **kw) -> dict:
    t0 = time.time()
    run_jobs(prior, mode, tag, **kw)
    t1 = time.time()
    g, j = gold_scores(tag), jump_scores(tag)
    line = (f"jumps {j['jumps']:4d}/{j['pairs']} ({j['jumps'] / max(1, j['pairs']):.1%}) judged {j['judged']:3d}: "
            f"right {j['right']:2d} leap {j['leap']:2d} wrong {j['wrong']:3d} | gold found {g['found']:.1%} "
            f"pitch {g['pitch']:.1%} loose {g['loose']:.1%} ±1 {g['near']:.1%} oct {g['oct']:.1%} on_F {g['on_F']:.1%} "
            f"sure {g['sure']:.0%} pitch|sure {g['pitch_sure']:.1%} | notes {j['notes']}")
    out = {"gold": g, "jumps": j, "coverage": coverage_scores(tag)}
    if vocadito:
        v = vocadito_scores(None if mode == "syllables" else prior, **kw)
        line += (f" | vocadito found {v['found']:.1%} pitch {v['pitch']:.1%} label {v['label']:.1%} "
                 f"note_F {v['note_F']:.1%} notes {v['notes']:.2f} jumps {v['jumps']}/{v['pairs']}")
        out["vocadito"] = v
    print(line + f"  [{t1 - t0:.0f} s decode]", flush=True)
    print("coverage", out["coverage"], flush=True)
    return out


def main() -> None:
    from karaoke.melody.decode import Prior, with_prior

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", nargs="*", default=[], help="Prior fields, e.g. weight=2.5 change=4")
    ap.add_argument("--mode", default="free", choices=["free", "syllables", "nowords"])
    ap.add_argument("--tag", default="exp")
    ap.add_argument("--no-vocadito", action="store_true")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--songs", action="store_true", help="print the per-song gold pitch")
    ap.add_argument("--grid", nargs="*", default=[], help="field=v1,v2,... : one run per value, others fixed")
    ap.add_argument("--json", type=Path, help="write aggregate metrics (no lyrics/audio)")
    ap.add_argument("--score", help="only score the existing melody.<TAG>.json files")
    args = ap.parse_args()
    if args.score:
        g, j = gold_scores(args.score), jump_scores(args.score)
        print(f"jumps {j['jumps']:4d}/{j['pairs']} ({j['jumps'] / max(1, j['pairs']):.1%}) judged {j['judged']:3d}: "
              f"right {j['right']:2d} leap {j['leap']:2d} wrong {j['wrong']:3d} | gold found {g['found']:.1%} "
              f"pitch {g['pitch']:.1%} loose {g['loose']:.1%} ±1 {g['near']:.1%} oct {g['oct']:.1%} on_F {g['on_F']:.1%} "
              f"sure {g['sure']:.0%} pitch|sure {g['pitch_sure']:.1%} | notes {j['notes']}")
        print(" ".join(f"{k} {v:.0%}" for k, v in g["songs"].items()))
        return
    prior = with_prior(Prior(), **{k: float(v) for k, v in (s.split("=") for s in args.set)})
    kw = {"refine": not args.no_refine}
    if not args.grid:
        print(prior)
        out = report(prior, args.mode, args.tag, not args.no_vocadito, **kw)
        if args.json:
            args.json.write_text(json.dumps(out, indent=2) + "\n")
        if args.songs:
            print(" ".join(f"{k} {v:.0%}" for k, v in out["gold"]["songs"].items()))
        return
    for spec in args.grid:
        field, values = spec.split("=")
        for v in values.split(","):
            print(f"{field}={v:6s}", end=" ")
            report(with_prior(prior, **{field: float(v)}), args.mode, args.tag, not args.no_vocadito, **kw)


if __name__ == "__main__":
    main()
