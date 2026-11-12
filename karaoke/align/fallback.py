"""No lyrics text (``chosen == whisper``): lines come straight from the transcript.

Whisper's segments are windows of audio, not lyric lines: a segment often ends mid-phrase, and
the next one starts with the rest of it ("boy Eles vão…"). So segments are ignored and the word
stream is cut where the *meaning* breaks:

1. a new line starts after ``.!?``, before a capitalised word (whisper capitalises sentence
   starts; the English "I" does not count), and at pauses longer than ``PAUSE_S``;
2. a short piece (a word or two) is merged into its neighbour when the two are sung together;
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
SHORT_LINE_CHARS = 9  # ... when it is at most this long ("King com", "Hugo de": whisper capitalises names)

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


# A line does not end on these (PT/EN articles, prepositions, conjunctions): "Rio Preto com a / Paraíba"
# would cut a phrase, so a capitalised word after one of them is a name, not a sentence start.
NO_LINE_END = {"a", "o", "as", "os", "um", "uma", "de", "do", "da", "dos", "das", "e", "em", "no", "na", "nos",
               "nas", "com", "pra", "pro", "para", "por", "que", "the", "an", "of", "and", "to", "in", "on", "with",
               "for", "at", "from"}


def _name_follows(prev: str, word: str) -> bool:
    """Is a capitalised ``word`` part of a name or phrase begun before it ("São Paulo", "de Janeiro")?"""
    p = prev.strip()
    if p.endswith((",", ";", ":")) or p.endswith(SENTENCE_END):
        return False
    bare_prev = p.strip("\"'“”‘’()").lower()
    return bare_prev in NO_LINE_END or starts_sentence(p)


def phrases(tokens: list[WToken]) -> list[list[WToken]]:
    """Cut the word stream at sentence ends, sentence starts and pauses."""
    out: list[list[WToken]] = []
    for t in tokens:
        if out:
            prev = out[-1][-1]
            new = (prev.text.endswith(SENTENCE_END)
                   or (starts_sentence(t.text) and not _name_follows(prev.text, t.text))
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
        if len(_text(c)) <= SHORT_LINE_CHARS and len(out) > 1:
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
        trim_words(envelope, times, texts=[t.text for t in tokens])
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


# --------------------------------------------------------------------------- filling gaps in lyrics

FROM_WHISPER = "from_whisper"  # line flag: the words are what whisper heard, not written lyrics
FILL_MIN_WORDS = 4  # fewer heard words outside every line are an ad-lib or noise, not a missing passage
FILL_MIN_DISTINCT = 3  # "yeah yeah yeah yeah" is not a passage either
FILL_MIN_PROB = 0.45  # mean whisper probability; lower is mostly guessing in instrumentals
FILL_MARGIN_S = 0.25  # a heard word this close to a line belongs to that line's neighbourhood


def fill_id(index: int) -> str:
    return f"w{index + 1:03d}"


def fill_gaps(free: list[WToken], covered: list[tuple[float, float]], duration: float | None,
              envelope=None) -> tuple[list[Line], set[int]]:
    """Lines for what whisper heard where no line is: lyrics sites often have only part of a song.

    ``free`` are the whisper words no lyric word took; ``covered`` the (start, end) of every line
    there is. Words inside or next to a line are left alone (an ad-lib, a backing answer, a word
    the lyrics spell differently). Words between two lines (or before the first, after the last)
    become lines when there are enough of them and whisper was fairly sure, built like a song
    with no lyrics at all and flagged ``from_whisper`` and ``low_confidence`` for review.
    Returns the lines (ids ``w001``…, numbered later) and the indices of the words used."""
    spans = sorted(covered)
    groups: list[list[WToken]] = []
    last_gap = -1
    for t in sorted(free, key=lambda t: t.start):
        if any(t.start < e + FILL_MARGIN_S and t.end > s - FILL_MARGIN_S for s, e in spans):
            continue
        gap = sum(1 for s, _ in spans if s < t.start)  # which gap between lines
        # a new run in another gap, or after a pause: each run is judged on its own
        if groups and gap == last_gap and t.start - groups[-1][-1].end <= PAUSE_S:
            groups[-1].append(t)
        else:
            groups.append([t])
        last_gap = gap
    lines: list[Line] = []
    used: set[int] = set()
    for g in groups:
        if (len(g) < FILL_MIN_WORDS or len({t.key for t in g}) < FILL_MIN_DISTINCT
                or sum(t.prob for t in g) / len(g) < FILL_MIN_PROB):
            continue
        for ln in whisper_lines(g, duration, envelope):
            ln.flags = [LOW_CONFIDENCE, FROM_WHISPER]
            lines.append(ln)
        used |= {t.index for t in g}
    lines.sort(key=lambda ln: ln.start)
    for i, ln in enumerate(lines):
        ln.id = fill_id(i)
    return lines, used
