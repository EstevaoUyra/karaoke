"""Phase 1 acceptance check for workstream B (separate + transcribe) on a real job.

    uv run python spikes/check_transcribe.py Zdh2hot8rjU                  # EN cover, scored vs its LRC
    uv run python spikes/check_transcribe.py Zdh2hot8rjU --prompt hotwords
    uv run python spikes/check_transcribe.py f8x_t8wPFcM --separate       # PT-BR, no stems yet

The job is COPIED into a temporary JobStore (``--workdir`` keeps it for re-runs), and the real
stages run through ``run_job(only=[...])``, so torch (separate) and ctranslate2 (transcribe) each
get their own child process. Nothing in data/jobs/<id> is modified.

When ``lyrics/sources/lrclib.lrc`` exists it is the reference: the fraction of LRC lyric words
matched by whisper words (difflib over normalized tokens, as in spikes/bench_alignment.py) and the
per-20 s coverage (reference words by LRC line time vs words whisper heard). With ``--prompt``
the LRC text becomes the temp job's chosen.txt, to measure the lyrics prompt.
Results are appended to data/bench/transcribe_<id>.json.

Phase 1 results (2026-11-09, shared 4-core CPU, load average 4-10 from other agents and a
Homebrew build; times swing ±25 % with the load). Cover = Zdh2hot8rjU, 237 LRC words, 220 s:

| Config (cover, language given) | Encoder passes | Decode time | Words matched |
|---|---|---|---|
| Phase 0: Silero VAD on, sequential (re-measured) | ~4 | 31 s | 46 % |
| Phase 0: VAD off, sequential (Phase 0 number) | — | 376 s | 76 % (69 % by this script) |
| energy clips ≤ 30 s, sequential, pad 0.2 | ~15 | 260 s | 70 % (loops: 495 words heard) |
| energy clips ≤ 15 s, sequential | 17-19 | 155-170 s | 72-76 % |
| **energy clips ≤ 30 s, batched, pad 0.4, merge 1.0 (default)** | 9 | **78-102 s** | **76 %** |
| same, pad 0.2 / 0.3 / 0.6 | 8 / 8 / 10 | | 70 / 69 / 76 % |
| same, clips ≤ 20 s / ≤ 25 s | 12 / 11 | | 73 / 74 % |
| same + lyrics ``initial_prompt`` / ``hotwords`` | 9 | ~82 s | 44 / 50 % |
| batched with timestamp tokens | 9 + 4 retries | 160 s | 74 % (4 loops) |

Language detection (when no language is given) adds one encoder pass (~8-15 s incl. model load).
PT-BR (f8x_t8wPFcM, 153 s, no reference lyrics): Kim_Vocal_2 separation 129 s (RTF 0.84);
transcription 7 clips + 1 dropout retry, 340 words, language detected pt (p = 1.00), 78-109 s.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from unidecode import unidecode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from karaoke import audio  # noqa: E402
from karaoke.models import Job, LyricsChoice, LyricsSourceName, StageName  # noqa: E402
from karaoke.pipeline import run_job  # noqa: E402
from karaoke.stages import transcribe as tr_stage  # noqa: E402  (light: faster_whisper is imported lazily)
from karaoke.store import JobStore, read_model, write_model  # noqa: E402

WINDOW = 20.0


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", unidecode(w.lower()))


def read_lrc(p: Path) -> list[tuple[float, str]]:
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", line)
        if m and m[3].strip():
            rows.append((int(m[1]) * 60 + float(m[2]), m[3].strip()))
    return rows


def lrc_words(lrc: list[tuple[float, str]], total: float) -> list[tuple[str, float]]:
    """Reference words with a time estimate: spread evenly between this line start and the next."""
    out = []
    for i, (t, text) in enumerate(lrc):
        nxt = lrc[i + 1][0] if i + 1 < len(lrc) else min(total, t + 5)
        ws = [norm(w) for w in text.split() if norm(w)]
        for k, w in enumerate(ws):
            out.append((w, t + (nxt - t) * k / max(1, len(ws))))
    return out


def score(transcript, lrc: list[tuple[float, str]] | None, total: float) -> dict:
    heard = [(norm(w.text), w.start) for s in transcript.segments for w in s.words if norm(w.text)]
    res: dict = {"words_heard": len(heard)}
    nwin = int(total // WINDOW) + 1
    per = [{"t": int(i * WINDOW), "heard": 0} for i in range(nwin)]
    for _, t in heard:
        per[min(nwin - 1, int(t // WINDOW))]["heard"] += 1
    if lrc:
        ref = lrc_words(lrc, total)
        sm = difflib.SequenceMatcher(a=[w for w, _ in ref], b=[w for w, _ in heard], autojunk=False)
        matched = {a + k for a, _, n in sm.get_matching_blocks() for k in range(n)}
        res["ref_words"] = len(ref)
        res["matched"] = len(matched)
        res["matched_ratio"] = round(len(matched) / len(ref), 3)
        for d in per:
            d["ref"] = 0
            d["matched"] = 0
        for i, (_, t) in enumerate(ref):
            d = per[min(nwin - 1, int(t // WINDOW))]
            d["ref"] += 1
            d["matched"] += i in matched
    res["per_20s"] = per
    return res


def make_temp_job(src_id: str, store: JobStore, *, with_audio: bool, prompt_from_lrc: bool,
                  lang: str | None) -> None:
    src = JobStore().paths(src_id)
    dst = store.paths(src_id)
    dst.ensure_dirs()
    settings = read_model(src.job, Job).settings
    write_model(dst.job, Job(id=src_id, settings=settings))
    if with_audio:
        if not dst.audio.exists():
            os.symlink(src.audio, dst.audio)
    elif not dst.vocals.exists():
        os.symlink(src.vocals, dst.vocals)
    lrc = src.lyrics_source("lrclib.lrc")
    if prompt_from_lrc and lrc.exists():
        dst.lyrics_chosen_text.write_text("\n".join(t for _, t in read_lrc(lrc)) + "\n", encoding="utf-8")
    else:
        dst.lyrics_chosen_text.unlink(missing_ok=True)
    if lang:
        write_model(dst.lyrics_choice, LyricsChoice(chosen=LyricsSourceName.lrclib, language=lang))
    else:
        dst.lyrics_choice.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--separate", action="store_true", help="run the separate stage first (from audio.wav)")
    ap.add_argument("--prompt", choices=["none", "initial", "hotwords"], default="none")
    ap.add_argument("--lang", default=None, help="language as lyrics_fetch would give it (default: detect)")
    ap.add_argument("--workdir", type=Path, default=None, help="keep the temp job store here")
    args = ap.parse_args()

    work = args.workdir or Path(tempfile.mkdtemp(prefix="karaoke-check-"))
    store = JobStore(work / "jobs")
    store.jobs_dir.mkdir(parents=True, exist_ok=True)
    src = JobStore().paths(args.job_id)
    make_temp_job(args.job_id, store, with_audio=args.separate, prompt_from_lrc=args.prompt != "none",
                  lang=args.lang)
    os.environ["KARAOKE_WHISPER_PROMPT"] = args.prompt  # inherited by the spawned child
    p = store.paths(args.job_id)
    out: dict = {"job": args.job_id, "prompt": args.prompt, "lang_given": args.lang}

    if args.separate and not (p.vocals.exists() and p.instrumental.exists()):
        t0 = time.monotonic()
        run_job(args.job_id, only=[StageName.separate], store=store)
        out["separate_wall_s"] = round(time.monotonic() - t0, 1)
        print(f"separate: {out['separate_wall_s']} s")

    regions = audio.voice_regions(p.vocals, max_region_s=tr_stage.CLIP_MAX_LEN)  # same as the stage
    clips = audio.pack_clips(regions, tr_stage.CLIP_MAX_LEN, tr_stage.CLIP_MAX_GAP)
    total = audio.duration(p.vocals)
    out.update(duration=round(total, 1), regions=len(regions), voiced_s=round(audio.coverage(regions), 1),
               clips=len(clips), clip_s=round(audio.coverage(clips), 1))

    t0 = time.monotonic()
    run_job(args.job_id, only=[StageName.transcribe], store=store)
    out["transcribe_wall_s"] = round(time.monotonic() - t0, 1)
    log = p.log("transcribe").read_text(encoding="utf-8")
    m = re.findall(r"transcribed \d+ segments, \d+ words in ([\d.]+) s", log)
    out["decode_s"] = float(m[-1]) if m else None

    tr = store.transcript(args.job_id)
    out["language"] = tr.language
    out["language_probability"] = tr.language_probability
    lrc_path = src.lyrics_source("lrclib.lrc")
    out.update(score(tr, read_lrc(lrc_path) if lrc_path.exists() else None, total))

    print(json.dumps({k: v for k, v in out.items() if k != "per_20s"}, indent=2))
    for d in out["per_20s"]:
        ref = f"ref {d['ref']:3d}  matched {d['matched']:3d}  " if "ref" in d else ""
        print(f"  {d['t']:4d}-{d['t'] + int(WINDOW):<4d} {ref}heard {d['heard']:3d}")
    bench = ROOT / "data" / "bench" / f"transcribe_{args.job_id}.json"
    bench.parent.mkdir(parents=True, exist_ok=True)
    runs = json.loads(bench.read_text()) if bench.exists() else []
    runs.append(out)
    bench.write_text(json.dumps(runs, indent=2))
    if not args.workdir:
        shutil.rmtree(work, ignore_errors=True)
    print(f"temp job: {p.root}" if args.workdir else "")


if __name__ == "__main__":
    main()
