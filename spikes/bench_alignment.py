"""Phase 0 spike: how good are line timings from (a) whisper word timestamps vs (b) stable-ts forced alignment?

Each phase runs in its own process: torch and ctranslate2 crash or deadlock together on Intel macOS.

    uv run python spikes/bench_alignment.py Zdh2hot8rjU --phase sep --model Kim_Vocal_2.onnx   # torch
    uv run python spikes/bench_alignment.py Zdh2hot8rjU --phase whisper                        # ctranslate2 only
    uv run python spikes/bench_alignment.py Zdh2hot8rjU --phase stable                         # torch only

Reference = the duration-matched synced LRC in data/jobs/<id>/lyrics/sources/lrclib.lrc (hand-made, about ±0.3 s).
Metric = |predicted line start - LRC line start|.
(a) whisper: faster-whisper turbo on the vocal stem, then a greedy difflib match from lyric words to
    whisper words; a line starts at its first matched word (a crude stand-in for workstream D's DP).
(b) stable: stable-ts ``align()`` of the LRC text with openai-whisper "turbo" (PyTorch) on the vocal stem.
Results are merged into data/bench/alignment_<id>.json.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import statistics
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from unidecode import unidecode  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", unidecode(w.lower()))


def read_lrc(p: Path) -> list[tuple[float, str]]:
    rows = []
    for line in p.read_text().splitlines():
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", line)
        if m and m[3].strip():
            rows.append((int(m[1]) * 60 + float(m[2]), m[3].strip()))
    return rows


def ensure_stems(job: Path, model: str) -> None:
    if (job / "stems" / "vocals.wav").exists():
        return
    from audio_separator.separator import Separator

    tmp = job / "stems" / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    sep = Separator(output_dir=str(tmp), model_file_dir=str(Path.home() / ".cache/audio-separator-models"))
    sep.load_model(model_filename=model)
    t0 = time.monotonic()
    files = [tmp / Path(f).name for f in sep.separate(str(job / "audio.wav"))]
    print(f"separated full song in {time.monotonic() - t0:.0f}s")
    for f in files:
        stem = re.search(r"\(([^)]+)\)", f.name).group(1).lower()
        if stem in ("vocals", "instrumental"):
            shutil.move(f, job / "stems" / f"{stem}.wav")
    shutil.rmtree(tmp)


def errors(pred: list[float | None], ref: list[float]) -> dict:
    e = [abs(p - r) for p, r in zip(pred, ref, strict=True) if p is not None]
    return {
        "lines": len(ref), "found": len(e),
        "median_err": round(statistics.median(e), 3) if e else None,
        "p90_err": round(sorted(e)[int(0.9 * (len(e) - 1))], 3) if e else None,
        "within_0.3s": round(sum(x <= 0.3 for x in e) / len(ref), 3),
        "within_0.5s": round(sum(x <= 0.5 for x in e) / len(ref), 3),
        "worst": sorted((round(x, 2) for x in e), reverse=True)[:5],
    }


def phase_whisper(voc: Path, lrc: list[tuple[float, str]], lang: str) -> dict:
    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=4)
    t0 = time.monotonic()
    segs, _ = model.transcribe(str(voc), language=lang, word_timestamps=True, vad_filter=True,
                               condition_on_previous_text=False)
    ww = [w for s in segs for w in (s.words or [])]
    took = round(time.monotonic() - t0, 1)
    hyp = [norm(w.word) for w in ww]
    lyric_words, line_of = [], []
    for i, (_, text) in enumerate(lrc):
        for w in text.split():
            lyric_words.append(norm(w))
            line_of.append(i)
    sm = difflib.SequenceMatcher(a=lyric_words, b=hyp, autojunk=False)
    word_map = {a + k: b + k for a, b, n in sm.get_matching_blocks() for k in range(n)}
    starts: list[float | None] = []
    for li in range(len(lrc)):
        idxs = [i for i, ln in enumerate(line_of) if ln == li]
        starts.append(ww[word_map[idxs[0]]].start if idxs and idxs[0] in word_map else None)
    return {"seconds": took, "word_match_ratio": round(len(word_map) / len(lyric_words), 3),
            "line_starts": errors(starts, [t for t, _ in lrc])}


def phase_stable(voc: Path, lrc: list[tuple[float, str]], lang: str) -> dict:
    import stable_whisper

    model = stable_whisper.load_model("turbo", device="cpu")
    text = "\n".join(t for _, t in lrc)
    t0 = time.monotonic()
    al = model.align(str(voc), text, language=lang)
    took = round(time.monotonic() - t0, 1)
    aw = [w for s in al.segments for w in s.words]
    starts: list[float | None] = []
    k = 0
    for _, line in lrc:
        starts.append(aw[k].start if k < len(aw) else None)
        k += len(line.split())
    return {"seconds": took, "aligned_words": [len(aw), k], "line_starts": errors(starts, [t for t, _ in lrc])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--phase", choices=["sep", "whisper", "stable"], required=True)
    ap.add_argument("--model", default="Kim_Vocal_2.onnx", help="separator model (sep phase)")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()
    job = ROOT / "data" / "jobs" / args.job_id
    if args.phase == "sep":
        ensure_stems(job, args.model)
        return
    lrc = read_lrc(job / "lyrics" / "sources" / "lrclib.lrc")
    voc = job / "stems" / "vocals.wav"
    res = phase_whisper(voc, lrc, args.lang) if args.phase == "whisper" else phase_stable(voc, lrc, args.lang)
    out_path = ROOT / "data" / "bench" / f"alignment_{args.job_id}.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out[args.phase] = res
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({args.phase: res}, indent=2))


if __name__ == "__main__":
    main()
