"""Make a whisper transcript of a job's vocal stem for alignment benchmarks (workstream D).

Runs faster-whisper turbo (ctranslate2 only, no torch, so it is safe in its own process) on
``data/jobs/<id>/stems/vocals.wav`` and writes a ``models.Transcript`` to
``data/bench/transcript_<id>.json``. About 6 minutes of CPU for a 3:40 song (no VAD).

    uv run python spikes/make_transcript.py Zdh2hot8rjU [--lang en]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from karaoke.models import Transcript, TSegment, TWord  # noqa: E402
from karaoke.store import write_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()
    voc = ROOT / "data" / "jobs" / args.job_id / "stems" / "vocals.wav"
    out = ROOT / "data" / "bench" / f"transcript_{args.job_id}.json"

    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    t0 = time.monotonic()
    segs, info = model.transcribe(str(voc), language=args.lang, word_timestamps=True, vad_filter=False,
                                  condition_on_previous_text=False)
    segments = []
    for s in segs:
        words = [TWord(text=w.word, start=round(w.start, 3), end=round(w.end, 3),
                       prob=round(w.probability, 3)) for w in (s.words or [])]
        segments.append(TSegment(start=round(s.start, 3), end=round(s.end, 3), text=s.text, words=words,
                                 avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob))
        print(f"{s.end:7.1f}s  {len(words)} words", flush=True)
    tr = Transcript(model="large-v3-turbo", language=info.language,
                    language_probability=info.language_probability, audio="vocals", segments=segments)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_model(out, tr)
    n = sum(len(s.words) for s in segments)
    print(f"wrote {out} ({len(segments)} segments, {n} words) in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
