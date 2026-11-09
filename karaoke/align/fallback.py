"""No lyrics text (``chosen == whisper``): lines come straight from the transcript.

Whisper's segments are windows of audio, not lyric lines: a segment often ends mid-phrase, and
the next one starts with the rest of it ("boy Eles vão…"). So segments are ignored and the word
stream is cut where the *meaning* breaks:

1. a new line starts after ``.!?``, before a capitalised word (whisper capitalises sentence
   starts; the English "I" does not count), and at pauses longer than ``PAUSE_S``;
2. a lone short word is merged into its neighbour when the two are sung together;
3. lines longer than ``MAX_CHARS`` are split at their best break (``karaoke.linebreak``).

Whisper's trailing "…" on a word it cut off is dropped, and the word after it is not treated as
a sentence start. Every line is flagged ``low_confidence`` for review.
"""

from __future__ import annotations

import dataclasses

from karaoke.align.confidence import LOW_CONFIDENCE, word_conf
from karaoke.align.interpolate import enforce_monotonic
from karaoke.align.preserve import line_id
from karaoke.align.tokens import WToken
from karaoke.linebreak import SENTENCE_END, W, split_to_fit, starts_sentence
from karaoke.models import Line, Word, WordOrigin

PAUSE_S = 2.0  # shorter pauses stay inside a line (the fill waits); longer ones also get the dots
MAX_CHARS = 42
MERGE_GAP_S = 0.5  # a lone word joins a neighbour sung closer than this
SHORT_CHARS = 7  # ... when it is at most this long

ELLIPSES = ("...", "…")


def _clean(tokens: list[WToken]) -> list[WToken]:
    out: list[WToken] = []
    cut = False  # the previous word was cut off with "…"
    for t in tokens:
        text = t.text
        if cut and starts_sentence(text):
            text = text[0].lower() + text[1:]
        cut = text.endswith(ELLIPSES)
        for e in ELLIPSES:
            if text.endswith(e) and len(text) > len(e):
                text = text[: -len(e)].rstrip()
        out.append(dataclasses.replace(t, text=text))
    return out


def phrases(tokens: list[WToken]) -> list[list[WToken]]:
    """Cut the word stream at sentence ends, sentence starts and pauses."""
    out: list[list[WToken]] = []
    for t in tokens:
        if out:
            prev = out[-1][-1]
            new = (prev.text.endswith(SENTENCE_END) or starts_sentence(t.text)
                   or t.start - prev.end > PAUSE_S)
            if not new:
                out[-1].append(t)
                continue
        out.append([t])
    return out


def _text(chunk: list[WToken]) -> str:
    return " ".join(t.text for t in chunk)


def _merge_short(chunks: list[list[WToken]]) -> list[list[WToken]]:
    out = [list(c) for c in chunks]
    i = 0
    while i < len(out):
        c = out[i]
        if len(c) == 1 and len(c[0].text) <= SHORT_CHARS and len(out) > 1:
            nxt = out[i + 1] if i + 1 < len(out) else None
            prv = out[i - 1] if i > 0 else None
            if nxt and nxt[0].start - c[-1].end <= MERGE_GAP_S and len(_text(c + nxt)) <= MAX_CHARS:
                out[i + 1] = c + nxt
                del out[i]
                continue
            if prv and c[0].start - prv[-1].end <= MERGE_GAP_S and len(_text(prv + c)) <= MAX_CHARS:
                out[i - 1] = prv + c
                del out[i]
                continue
        i += 1
    return out


def _split_long(chunk: list[WToken]) -> list[list[WToken]]:
    ws = [W(t.text, t.start, t.end) for t in chunk]
    return [[chunk[i] for i in part] for part in split_to_fit(ws, len, MAX_CHARS)]


def group_lines(tokens: list[WToken]) -> list[list[WToken]]:
    groups = _merge_short(phrases(_clean(tokens)))
    return [g for c in groups for g in _split_long(c)]


def whisper_lines(tokens: list[WToken], duration: float | None, envelope=None) -> list[Line]:
    if envelope is not None:
        from karaoke.align.voicing import trim_words

        times = [[t.start, t.end] for t in tokens]
        trim_words(envelope, times)
        tokens = [dataclasses.replace(t, start=s, end=e) for t, (s, e) in zip(tokens, times, strict=True)]
    groups = group_lines(tokens)
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
