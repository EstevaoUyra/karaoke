"""Phase 0 spike: compare separation models on CPU (speed + how well whisper hears the stems).

    uv run python spikes/bench_separation.py --phase sep   # separate: all models x both songs
    uv run python spikes/bench_separation.py --phase asr   # then whisper on every stem

For each song a 45 s excerpt is cut from data/jobs/<id>/audio.wav, separated by every model
into data/bench/<id>/<model>/{vocals,instrumental}.wav (keep these and listen!), then:
- sep_s / rtf: wall time and real-time factor (seconds of compute per second of audio)
- vocals WER: whisper on the vocal stem vs the reference lyrics (when known)
- leak_words: words whisper finds (prob > 0.5) in the INSTRUMENTAL (lower = cleaner karaoke)
Baselines: whisper on the original mix. Results go to data/bench/results.json.
"""

from __future__ import annotations

import os

# torch and ctranslate2 each bundle libiomp5; allow both in one process (see karaoke/__init__.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from rapidfuzz.distance import Levenshtein
from unidecode import unidecode

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "data" / "jobs"
BENCH = ROOT / "data" / "bench"

SONGS = {
    "Zdh2hot8rjU": {"lang": "en", "start": 20.0},  # Sweet Child O' Mine (Captain Fantastic)
    "f8x_t8wPFcM": {"lang": "pt", "start": 30.0},  # Sacal - Vão Querer Me Internar
}
EXCERPT = 45.0
MODELS = [
    "UVR-MDX-NET-Inst_HQ_3.onnx",
    "Kim_Vocal_2.onnx",
    "htdemucs_ft.yaml",
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
]


def norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", unidecode(text.lower()))


def wer(ref: list[str], hyp: list[str]) -> float:
    return Levenshtein.distance(ref, hyp) / max(1, len(ref))


def lrc_reference(job: Path, start: float, end: float) -> list[str] | None:
    lrc = job / "lyrics" / "sources" / "lrclib.lrc"
    if not lrc.exists():
        return None
    words: list[str] = []
    rows = []
    for line in lrc.read_text().splitlines():
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", line)
        if m:
            rows.append((int(m[1]) * 60 + float(m[2]), m[3]))
    for (t, text), nxt in zip(rows, rows[1:] + [(1e9, "")], strict=True):
        if t >= start - 0.5 and nxt[0] <= end + 0.5:
            words += norm_words(text)
    return words


def cut_excerpt(job_id: str, start: float) -> Path:
    out = BENCH / job_id / "excerpt.wav"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-t", str(EXCERPT),
                        "-i", str(JOBS / job_id / "audio.wav"), "-c:a", "pcm_s16le", str(out)], check=True)
    return out


def separate(excerpt: Path, model: str) -> tuple[Path, Path, float]:
    from audio_separator.separator import Separator

    out_dir = excerpt.parent / model.rsplit(".", 1)[0]
    voc, ins = out_dir / "vocals.wav", out_dir / "instrumental.wav"
    timing = out_dir / "time.json"
    if voc.exists() and ins.exists() and timing.exists():
        return voc, ins, json.loads(timing.read_text())["sep_s"]
    out_dir.mkdir(parents=True, exist_ok=True)
    sep = Separator(output_dir=str(out_dir), model_file_dir=str(Path.home() / ".cache/audio-separator-models"),
                    output_format="WAV")
    sep.load_model(model_filename=model)
    t0 = time.monotonic()
    files = [out_dir / Path(f).name for f in sep.separate(str(excerpt))]
    dt = time.monotonic() - t0
    stems = {re.search(r"\(([^)]+)\)", f.name).group(1).lower(): f for f in files}
    shutil.move(stems.pop("vocals"), voc)
    if "instrumental" in stems:
        shutil.move(stems.pop("instrumental"), ins)
    else:  # demucs: instrumental = drums + bass + other (+ guitar/piano)
        import numpy as np
        import soundfile as sf
        parts = [sf.read(p) for p in stems.values()]
        sf.write(ins, np.sum([d for d, _ in parts], axis=0), parts[0][1])
    for f in stems.values():
        f.unlink(missing_ok=True)
    timing.write_text(json.dumps({"sep_s": dt}))
    return voc, ins, dt


_MODEL = None


def transcribe(path: Path, lang: str) -> tuple[list[str], float, list[dict]]:
    global _MODEL
    from faster_whisper import WhisperModel

    if _MODEL is None:
        _MODEL = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=4)
    t0 = time.monotonic()
    segs, _ = _MODEL.transcribe(str(path), language=lang, word_timestamps=True, vad_filter=True,
                                condition_on_previous_text=False)
    words = [w for s in segs for w in (s.words or [])]
    dt = time.monotonic() - t0
    return [w.word for w in words], dt, [{"w": w.word, "s": w.start, "e": w.end, "p": w.probability} for w in words]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--songs", nargs="*", default=list(SONGS))
    ap.add_argument("--phase", choices=["sep", "asr"], required=True,
                    help="torch (separation) and ctranslate2 (whisper) deadlock in one process: run sep, then asr")
    args = ap.parse_args()
    results_path = BENCH / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for song in args.songs:
        cfg = SONGS[song]
        excerpt = cut_excerpt(song, cfg["start"])
        ref = lrc_reference(JOBS / song, cfg["start"], cfg["start"] + EXCERPT)
        r = results.setdefault(song, {"lang": cfg["lang"], "reference_words": len(ref) if ref else None})
        if args.phase == "sep":
            for model in args.models:
                try:
                    _, _, dt = separate(excerpt, model)
                    print(song, model, f"separated in {dt:.0f}s", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(song, model, "ERROR", exc, flush=True)
            continue
        if "mix" not in r:
            hyp, wt, words = transcribe(excerpt, cfg["lang"])
            r["mix"] = {"whisper_s": wt, "wer": wer(ref, norm_words(" ".join(hyp))) if ref else None,
                        "text": " ".join(hyp).strip()}
            print(song, "mix", r["mix"]["wer"], flush=True)
            results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        for model in args.models:
            if model in r:
                continue
            out_dir = excerpt.parent / model.rsplit(".", 1)[0]
            voc, ins, timing = out_dir / "vocals.wav", out_dir / "instrumental.wav", out_dir / "time.json"
            if not timing.exists():
                r[model] = {"error": "separation failed (see sep phase)"}
                continue
            sep_s = json.loads(timing.read_text())["sep_s"]
            hyp, wt, _ = transcribe(voc, cfg["lang"])
            leak, _, leak_words = transcribe(ins, cfg["lang"])
            r[model] = {
                "sep_s": round(sep_s, 1), "rtf": round(sep_s / EXCERPT, 2), "whisper_s": round(wt, 1),
                "wer": round(wer(ref, norm_words(" ".join(hyp))), 3) if ref else None,
                "leak_words": sum(1 for w in leak_words if w["p"] > 0.5),
                "leak_text": " ".join(leak).strip(),
                "text": " ".join(hyp).strip(),
            }
            print(song, model, {k: v for k, v in r[model].items() if "text" not in k}, flush=True)
            results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
