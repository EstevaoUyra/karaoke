"""Real-song check (slow): the EN cover job with its duration-matched synced LRC.

Needs local, gitignored data: ``data/jobs/Zdh2hot8rjU/lyrics/sources/lrclib.lrc`` and a whisper
transcript at ``data/bench/transcript_Zdh2hot8rjU.json`` (``spikes/make_transcript.py``). The
lyric text is built from the LRC lines (stanzas at its empty break lines) and line starts are
compared with the LRC starts. No lyric text is printed.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest

from karaoke.align.core import align_lyrics
from karaoke.align.tokens import parse_chosen_text, parse_lrc
from karaoke.models import Transcript
from karaoke.store import read_model
from karaoke.validate import validate_lyrics

ROOT = Path(__file__).resolve().parents[1]
JOB = "Zdh2hot8rjU"
LRC = ROOT / "data" / "jobs" / JOB / "lyrics" / "sources" / "lrclib.lrc"
TRANSCRIPT = ROOT / "data" / "bench" / f"transcript_{JOB}.json"


def _errors(use_line_timings: bool):
    lrc = parse_lrc(LRC.read_text(encoding="utf-8"))
    rows = [r for r in lrc if r.text]
    lines = parse_chosen_text("\n".join(r.text for r in lrc))  # empty LRC lines -> stanza breaks
    assert len(lines) == len(rows)
    tr = read_model(TRANSCRIPT, Transcript)
    t0 = time.perf_counter()
    lf, stats = align_lyrics(lines, tr, lrc=lrc, use_line_timings=use_line_timings, duration=220.0)
    took = time.perf_counter() - t0
    assert [i for i in validate_lyrics(lf, 220.0) if i.severity == "error"] == []
    errs = [abs(ln.start - r.start) for ln, r in zip(lf.lines, rows, strict=True)]
    return errs, took, stats


def _summary(errs: list[float]) -> str:
    n = len(errs)
    return (f"{sum(e <= 0.3 for e in errs)}/{n} <=0.3s, {sum(e <= 0.5 for e in errs)}/{n} <=0.5s, "
            f"median {statistics.median(errs):.2f}s, worst {max(errs):.1f}s")


@pytest.mark.slow
@pytest.mark.skipif(not (LRC.exists() and TRANSCRIPT.exists()), reason="real-song data not present")
@pytest.mark.parametrize("use_line_timings", [False, True])
def test_real_song_line_starts(use_line_timings):
    errs, took, stats = _errors(use_line_timings)
    print(f"\nuse_line_timings={use_line_timings}: all {_summary(errs)}; "
          f"lines 1-36 {_summary(errs[:36])}; {took:.2f}s; "
          f"{stats.matched + stats.fuzzy}/{stats.words} words heard, {stats.loop_words_dropped} loop words dropped")
    assert took < 1.0
    within = sum(e <= 0.5 for e in errs) / len(errs)
    if use_line_timings:
        assert within >= 0.8
    else:
        # The transcript has a 26 s whisper repetition loop over the outro (lines 37-53), so only
        # the part before it is held to a bound; see the workstream D report.
        assert sum(e <= 0.5 for e in errs[:36]) / 36 >= 0.75
        assert max(errs[:36]) < 5.0
