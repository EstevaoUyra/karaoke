"""No lyrics text (``chosen == whisper``): lines come straight from the transcript.

Segments are split at pauses longer than 0.7 s, then chunks longer than 10 words are split at
their largest internal pauses. Every line is flagged ``low_confidence`` for review.
"""

from __future__ import annotations

from karaoke.align.confidence import LOW_CONFIDENCE, word_conf
from karaoke.align.interpolate import enforce_monotonic
from karaoke.align.preserve import line_id
from karaoke.align.tokens import WToken
from karaoke.models import Line, Word, WordOrigin

PAUSE_S = 0.7
MAX_WORDS = 10


def _split_long(chunk: list[WToken]) -> list[list[WToken]]:
    if len(chunk) <= MAX_WORDS:
        return [chunk]
    # split at the largest pause that leaves both sides non-empty, preferring the middle
    best, best_score = None, -1.0
    for k in range(2, len(chunk) - 1):
        gap = chunk[k].start - chunk[k - 1].end
        balance = 1 - abs(k - len(chunk) / 2) / len(chunk)
        score = gap + 0.05 * balance
        if score > best_score:
            best, best_score = k, score
    assert best is not None
    return _split_long(chunk[:best]) + _split_long(chunk[best:])


def whisper_lines(tokens: list[WToken], duration: float | None) -> list[Line]:
    chunks: list[list[WToken]] = []
    for t in tokens:
        if chunks and t.segment == chunks[-1][-1].segment and t.start - chunks[-1][-1].end <= PAUSE_S:
            chunks[-1].append(t)
        else:
            chunks.append([t])
    groups = [g for c in chunks for g in _split_long(c)]
    times = [[t.start, t.end] for g in groups for t in g]
    enforce_monotonic(times, [False] * len(times), duration)
    lines: list[Line] = []
    k = 0
    for g in groups:
        words = []
        for t in g:
            s, e = times[k]
            k += 1
            words.append(Word(text=t.text, start=s, end=e, conf=round(min(0.5, word_conf("matched", t.prob)), 3),
                              origin=WordOrigin.matched))
        lines.append(Line(id=line_id(len(lines)), start=words[0].start, end=max(w.end for w in words),
                          words=words, flags=[LOW_CONFIDENCE]))
    return lines
