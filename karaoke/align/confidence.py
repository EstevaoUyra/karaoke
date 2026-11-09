"""Per-word confidence and per-line flags."""

from __future__ import annotations

from karaoke.models import Line, WordOrigin

LOW_CONFIDENCE = "low_confidence"
TIMING_SUSPECT = "timing_suspect"
COMPUTED_FLAGS = {LOW_CONFIDENCE, TIMING_SUSPECT}

MAX_LINE_S = 15.0
MAX_WORD_S = 4.0
LRC_DISAGREE_S = 1.0


def word_conf(kind: str, prob: float, sim: float = 1.0, anchored: bool = False) -> float:
    """kind: matched | fuzzy | sub (a misheard word at the right place) | interpolated."""
    if kind == "matched":
        c = 0.55 + 0.45 * prob
    elif kind == "fuzzy":
        c = (0.3 + 0.4 * prob) * (0.6 + 0.4 * sim)
    elif kind == "sub":
        c = 0.25 + 0.1 * prob
    else:
        c = 0.35 if anchored else 0.2
    return round(max(0.0, min(1.0, c)), 3)


def line_flags(line: Line, raw_durations: list[float], lrc_start: float | None) -> list[str]:
    """``low_confidence``: < 60 % of words heard by whisper, or mean conf < 0.5.
    ``timing_suspect``: implausible word durations, a line longer than 15 s, or a start far from
    the trusted LRC start (``lrc_start`` is only passed when line timings are trusted)."""
    flags = [f for f in line.flags if f not in COMPUTED_FLAGS]
    words = line.words
    heard = sum(w.origin in (WordOrigin.matched, WordOrigin.fuzzy, WordOrigin.manual, WordOrigin.ai)
                for w in words)
    mean_conf = sum(w.conf for w in words) / len(words)
    if heard / len(words) < 0.6 or mean_conf < 0.5:
        flags.append(LOW_CONFIDENCE)
    short = sum(d < 0.05 for d in raw_durations)
    suspect = (
        any(d > MAX_WORD_S for d in raw_durations)
        or (len(words) >= 2 and short * 2 >= len(words))
        or line.end - line.start > MAX_LINE_S
        or (lrc_start is not None and abs(line.start - lrc_start) > LRC_DISAGREE_S)
    )
    if suspect:
        flags.append(TIMING_SUSPECT)
    return flags
