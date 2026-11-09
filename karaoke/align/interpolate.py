"""Timing for lyric words whisper did not hear, plus the final monotonic clean-up.

A run of untimed lyric words sits between two timed words P (ending at ``lo``) and N (starting at
``hi``). The run is cut into per-line chunks. Each word gets a "natural" duration from its
length; when the window is longer than the words need, the spare time goes to the line breaks
(the singer breathes or the band plays between lines, not inside a line). So:
- the tail of P's line is sung right after P,
- the head of N's line is sung right before N,
- whole lines in between share the remaining time,
- a run inside one line is spread proportionally between P and N.
LRC anchors, when trusted, override this per line (see ``core``).
"""

from __future__ import annotations

from dataclasses import dataclass

from karaoke.align.tokens import letters, match_key

MIN_WORD = 0.08
BREAK_GAP = 0.4  # default pause between lines when a run has no bound on one side
LINE_PAUSE = 0.6  # pause between two lines of one stanza when a longer gap exists elsewhere
WORD_STRETCH = 1.25  # words may be this much longer than natural before the rest goes to breaks


# Sung vocables are held much longer than their spelling suggests ("oh" ~ 0.5 s, not 0.25 s).
VOCABLES = {"oh", "ohh", "ooh", "oooh", "o", "ah", "aah", "ahh", "whoa", "woah", "wo", "woo", "yeah", "yeh",
            "na", "la", "hey", "uh", "uhh", "mm", "mmm", "hmm", "ay", "ai", "eh", "ei", "ae", "u", "uu"}


def natural_duration(text: str) -> float:
    """Rough sung duration of a word: longer words take longer; vocables are held."""
    if match_key(text) in VOCABLES:
        return 0.55
    return min(0.9, max(0.18, 0.12 + 0.06 * letters(text)))


@dataclass
class Slot:
    start: float
    end: float


def spread(chunks: list[list[float]], lo: float, hi: float, breaks: list[str | None]) -> list[list[Slot]]:
    """Place chunks (lists of natural durations, one chunk per line) inside [lo, hi].

    ``breaks`` has ``len(chunks) + 1`` entries: before the first chunk, between chunks, after the
    last. Each is None (the chunk continues the neighbouring timed word's line), "line" (an
    ordinary line break) or "stanza" (a blank line or section change in the lyrics). Spare time
    goes to the breaks; when there are stanza breaks, ordinary breaks get a normal pause
    (``LINE_PAUSE``) and the stanza breaks share the rest, because instrumentals sit between
    stanzas, not between the lines of one.
    """
    assert len(breaks) == len(chunks) + 1
    nat = [d for c in chunks for d in c]
    total = sum(nat)
    width = max(0.0, hi - lo)
    n_line = sum(b == "line" for b in breaks)
    n_stanza = sum(b == "stanza" for b in breaks)
    gaps = [0.0] * len(breaks)
    if n_line + n_stanza == 0 or width <= total * WORD_STRETCH:
        scale = width / total if total > 0 else 0.0
    else:
        scale = WORD_STRETCH
        excess = width - total * scale
        if n_stanza:
            line_gap = min(LINE_PAUSE, excess / (n_line + n_stanza))
            stanza_gap = (excess - line_gap * n_line) / n_stanza
        else:
            line_gap = stanza_gap = excess / n_line
        gaps = [line_gap if b == "line" else stanza_gap if b == "stanza" else 0.0 for b in breaks]
    out: list[list[Slot]] = []
    t = lo + gaps[0]
    for ci, c in enumerate(chunks):
        if ci:
            t += gaps[ci]
        row = []
        for d in c:
            row.append(Slot(t, t + d * scale))
            t += d * scale
        out.append(row)
    return out


def place_from(durs: list[float], start: float, hi: float | None = None, stretch: float = 1.0) -> list[Slot]:
    """Words back to back from ``start``; compressed to end by ``hi`` if they would not fit."""
    total = sum(durs) * stretch
    scale = stretch
    if hi is not None and start + total > hi and sum(durs) > 0:
        scale = max(0.0, hi - start) / sum(durs)
    out = []
    t = start
    for d in durs:
        out.append(Slot(t, t + d * scale))
        t += d * scale
    return out


def enforce_monotonic(times: list[list[float]], fixed: list[bool], duration: float | None,
                      min_word: float = MIN_WORD) -> None:
    """In place: words in song order become non-overlapping, at least ``min_word`` long, within
    [0, duration]. ``fixed`` words (manual/ai) are never moved; others make room for them."""
    n = len(times)
    # Upper bound for each movable word: the start of the next fixed word.
    cap = [float("inf")] * n
    nxt = float("inf")
    for i in range(n - 1, -1, -1):
        cap[i] = nxt
        if fixed[i]:
            nxt = times[i][0]
    if duration is not None:
        cap = [min(c, duration) for c in cap]
    prev_end = 0.0
    prev_start = 0.0
    for i in range(n):
        s, e = times[i]
        if fixed[i]:
            prev_end = max(prev_end, e)
            prev_start = s
            continue
        s = max(s, prev_end, 0.0)
        e = max(e, s + min_word)
        if e > cap[i]:
            e = cap[i]
            s = min(s, e - min_word)
            s = max(s, prev_start)
            e = max(e, s)
        times[i] = [round(s, 3), round(e, 3)]
        prev_end = max(prev_end, times[i][1])
        prev_start = times[i][0]
