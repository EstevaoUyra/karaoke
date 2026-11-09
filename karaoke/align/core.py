"""Lyrics text + whisper transcript (+ optional LRC anchors) -> LyricsFinal.

Pure function, no file IO: ``karaoke/stages/align.py`` reads the job files and calls
``align_lyrics``. Steps:

1. Tokenize lyrics (display text + match key) and whisper words; drop whisper repetition loops.
2. LRC anchors (only when ``use_line_timings``): map lyric lines to LRC lines by text, run an
   unconstrained DP once to estimate a global LRC offset, then penalize pairing a lyric word
   with a whisper word more than ~1 s outside its line's LRC window (``penalty``). Choruses
   share words; their LRC windows do not, so this removes repetition confusion.
3. Global DP (``dp.align``).
4. Matched words take whisper timing. In each gap between two matched pairs, a lyric run and a
   whisper run that fill the same slot inside one line are paired as substitutions (whisper
   misheard the word but timed it); everything else in the gap is unmatched.
5. Manual/AI words from the previous result are put back and act as fixed anchors.
6. Untimed words: LRC line placement where trusted, otherwise ``interpolate.spread``.
7. Monotonic, non-overlapping times with a minimum duration; line bounds; confidence; flags.
"""

from __future__ import annotations

import bisect
import statistics
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz

from karaoke.align import dp
from karaoke.align.confidence import line_flags, word_conf
from karaoke.align.fallback import whisper_lines
from karaoke.align.interpolate import (
    BREAK_GAP,
    enforce_monotonic,
    natural_duration,
    place_from,
    spread,
)
from karaoke.align.preserve import KEEP, Preserved, collect, line_id
from karaoke.align.tokens import (
    LrcLine,
    LyricLine,
    WToken,
    drop_hallucination_loops,
    match_key,
    whisper_tokens,
)
from karaoke.align.voicing import Envelope, trim_words
from karaoke.models import (
    Line,
    LyricsFinal,
    LyricsSourceName,
    Transcript,
    UnmatchedAudio,
    Word,
    WordOrigin,
)

# LRC window: no penalty within this many seconds outside [line start, next LRC timestamp];
# beyond it the pair score drops by PENALTY_SLOPE per second, down to -PENALTY_CAP.
LRC_TOLERANCE = 1.0
PENALTY_SLOPE = 2.0
PENALTY_CAP = 8.0
SUB_CONTIGUITY = 1.0  # a substituted whisper run must touch its same-line neighbours within this
UNMATCHED_JOIN = 1.5  # unmatched whisper words closer than this form one unmatched_audio entry
LONG_WORD_S = 3.0
LRC_MIN_OFFSET = 0.5  # smaller median differences are whisper's own bias, not an LRC offset
# Extra cost per unheard lyric word placed where whisper leaves no time for it (see dp.align).
ROOM_COST = 0.3
ROOM_FULL = 3.0  # seconds of silence between whisper words that make an unheard word free


@dataclass
class Anchor:
    start: float  # LRC line start (after offset correction)
    end: float  # next LRC timestamp (a break or the next line)


@dataclass
class _W:
    """One lyric word being aligned."""

    line: int
    stanza: int
    pos: int  # index within its line
    text: str
    key: str
    start: float | None = None
    end: float | None = None
    kind: str = "interpolated"  # matched | fuzzy | sub | interpolated | kept
    prob: float = 0.0
    sim: float = 0.0
    anchored: bool = False  # placed from an LRC line window
    kept: Word | None = None


@dataclass
class AlignStats:
    words: int = 0
    matched: int = 0
    fuzzy: int = 0
    substituted: int = 0
    interpolated: int = 0
    kept: int = 0
    whisper_words: int = 0
    whisper_unmatched: int = 0
    loop_words_dropped: int = 0
    lrc_lines_mapped: int = 0
    lrc_offset: float | None = None
    lrc_disabled_reason: str | None = None
    pauses_trimmed: int = 0
    lost_edits: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- LRC anchors


def map_lines_to_lrc(lines: list[LyricLine], lrc: list[LrcLine], duration: float | None) -> dict[int, Anchor]:
    """Order-preserving best match of lyric lines to LRC text lines (free gaps)."""
    rows = [(k, r) for k, r in enumerate(lrc) if r.text.strip()]
    if not rows or not lines:
        return {}
    a = [" ".join(ln.keys) for ln in lines]
    b = [" ".join(match_key(w) for w in r.text.split()) for _, r in rows]
    n, m = len(a), len(b)
    sim = [[fuzz.ratio(x, y) / 100.0 for y in b] for x in a]
    score = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = sim[i - 1][j - 1]
            diag = score[i - 1, j - 1] + s if s >= 0.5 else -1.0
            score[i, j] = max(score[i - 1, j], score[i, j - 1], diag)
    out: dict[int, Anchor] = {}
    i, j = n, m
    while i > 0 and j > 0:
        s = sim[i - 1][j - 1]
        if s >= 0.5 and score[i, j] == score[i - 1, j - 1] + s:
            k, r = rows[j - 1]
            nxt = lrc[k + 1].start if k + 1 < len(lrc) else (duration if duration else r.start + 8.0)
            out[i - 1] = Anchor(r.start, max(nxt, r.start + 0.5))
            i, j = i - 1, j - 1
        elif score[i, j] == score[i - 1, j]:
            i -= 1
        else:
            j -= 1
    return out


def lrc_penalty(words: list[_W], toks: list[WToken], anchors: dict[int, Anchor]) -> np.ndarray:
    lo = np.array([anchors[w.line].start if w.line in anchors else np.nan for w in words])
    hi = np.array([anchors[w.line].end if w.line in anchors else np.nan for w in words])
    t = np.array([(x.start + x.end) / 2 for x in toks])
    d = np.maximum(np.maximum(lo[:, None] - t[None, :], t[None, :] - hi[:, None]), 0.0)
    d = np.nan_to_num(d, nan=0.0)
    return -np.minimum(PENALTY_CAP, PENALTY_SLOPE * np.maximum(0.0, d - LRC_TOLERANCE))


def _first_word_times(ops: list[dp.Op], words: list[_W], toks: list[WToken]) -> dict[int, float]:
    out: dict[int, float] = {}
    for op in ops:
        if op.kind in ("match", "split", "merge") and op.li[0] < len(words):
            w = words[op.li[0]]
            if w.pos == 0:
                out[w.line] = toks[op.wi[0]].start
    return out


def estimate_offset(ops: list[dp.Op], words: list[_W], toks: list[WToken],
                    anchors: dict[int, Anchor]) -> tuple[float, float]:
    """Median (whisper line start - LRC line start) and the share of lines that agree with it."""
    starts = _first_word_times(ops, words, toks)
    diffs = [starts[k] - anchors[k].start for k in starts if k in anchors]
    if len(diffs) < 3:
        return 0.0, 1.0
    med = statistics.median(diffs)
    agree = sum(abs(d - med) <= 1.5 for d in diffs) / len(diffs)
    return med, agree


# --------------------------------------------------------------------------- main


def align_lyrics(
    lines: list[LyricLine],
    transcript: Transcript,
    *,
    lrc: list[LrcLine] | None = None,
    use_line_timings: bool = False,
    duration: float | None = None,
    language: str | None = None,
    text_source: LyricsSourceName = LyricsSourceName.lrclib,
    previous: LyricsFinal | None = None,
    params: dp.DPParams | None = None,
    envelope: Envelope | None = None,
) -> tuple[LyricsFinal, AlignStats]:
    """``envelope`` (the vocal stem's loudness) lets pauses be cut out of words, see ``voicing``."""
    stats = AlignStats()
    toks = whisper_tokens(transcript)
    if text_source == LyricsSourceName.whisper or not lines:
        return _whisper_fallback(toks, duration, language, previous, stats, envelope)

    words = [_W(line=ln.index, stanza=ln.stanza, pos=k, text=t, key=match_key(t))
             for ln in lines for k, t in enumerate(ln.words)]
    keys = [w.key for w in words]
    toks, looped = drop_hallucination_loops(toks, keys)
    stats.loop_words_dropped = len(looped)
    stats.words, stats.whisper_words = len(words), len(toks)
    if duration is None:
        ends = [t.end for t in toks] + [r.start for r in (lrc or [])]
        duration = (max(ends) + 5.0) if ends else None

    anchors: dict[int, Anchor] = {}
    if use_line_timings and lrc:
        anchors = map_lines_to_lrc(lines, lrc, duration)
        stats.lrc_lines_mapped = len(anchors)

    wkeys = [t.key for t in toks]
    probs = [t.prob for t in toks]
    room = room_cost(toks, duration)
    quality, wscale = whisper_quality(toks)
    extra = {"lgap_extra": room, "quality": quality, "wgap_scale": wscale}
    penalty = None
    if anchors:
        ops0 = dp.align(keys, wkeys, probs, None, params, **extra)
        off, agree = estimate_offset(ops0, words, toks, anchors)
        if agree < 0.5:
            stats.lrc_disabled_reason = f"LRC disagrees with whisper (only {agree:.0%} of lines agree)"
            anchors = {}
            ops = ops0
        else:
            if abs(off) >= LRC_MIN_OFFSET:
                anchors = {k: Anchor(a.start + off, a.end + off) for k, a in anchors.items()}
            stats.lrc_offset = round(off, 3)
            penalty = lrc_penalty(words, toks, anchors)
            ops = dp.align(keys, wkeys, probs, penalty, params, **extra)
    else:
        ops = dp.align(keys, wkeys, probs, None, params, **extra)

    used_w = _apply_ops(ops, words, toks)
    _trim_line_starts(words)
    used_w |= _substitutions(ops, words, toks, anchors)

    preserved = collect(previous, lines)
    stats.lost_edits = preserved.lost
    for w in words:
        old = preserved.words.get((w.line, w.pos))
        if old is not None:
            w.kept, w.kind, w.start, w.end = old, "kept", old.start, old.end

    _clip_long_words(words, anchors)
    _interpolate(words, anchors, duration)

    times = [[float(w.start), float(w.end)] for w in words]  # type: ignore[arg-type]
    raw = [e - s for s, e in times]
    fixed = [w.kept is not None for w in words]
    enforce_monotonic(times, fixed, duration)
    if envelope is not None:
        stats.pauses_trimmed = trim_words(envelope, times, fixed)

    out_lines = _build_lines(lines, words, times, raw, anchors if use_line_timings else {}, preserved)
    unmatched = _unmatched(toks, used_w, looped)
    for w in words:
        if w.kind == "matched":
            stats.matched += 1
        elif w.kind == "fuzzy":
            stats.fuzzy += 1
        elif w.kind == "sub":
            stats.substituted += 1
        elif w.kind == "kept":
            stats.kept += 1
        else:
            stats.interpolated += 1
    stats.whisper_unmatched = len(toks) - len(used_w)
    lf = LyricsFinal(language=language, text_source=text_source, lines=out_lines, unmatched_audio=unmatched)
    return lf, stats


# --------------------------------------------------------------------------- steps


def room_cost(toks: list[WToken], duration: float | None) -> np.ndarray:
    """Per DP column j: cost of an unheard lyric word between whisper words j-1 and j."""
    if not toks:
        return np.zeros(1)
    starts = np.array([t.start for t in toks])
    ends = np.array([t.end for t in toks])
    room = np.empty(len(toks) + 1)
    room[0] = starts[0]
    room[1:-1] = starts[1:] - ends[:-1]
    room[-1] = (duration - ends[-1]) if duration is not None else ROOM_FULL
    return ROOM_COST * np.clip(1.0 - room / ROOM_FULL, 0.0, 1.0)


def whisper_quality(toks: list[WToken]) -> tuple[np.ndarray, np.ndarray]:
    """(match multiplier, skip-cost multiplier) per whisper word. Near-zero-length words are what
    whisper emits when it loops or squeezes words it did not really hear; they count less."""
    d = np.array([t.end - t.start for t in toks])
    prob = np.array([t.prob for t in toks])
    q = np.clip(d / 0.15, 0.3, 1.0)
    return q, q * (0.4 + 0.6 * prob)


def _clip_long_words(words: list[_W], anchors: dict[int, Anchor]) -> None:
    """Whisper sometimes stretches a word over the following silence or over lines it did not
    hear (a 6 s "mine"). Clip a heard word that runs past its line's LRC window, or that is very
    long and followed by unheard lyric words that need the time."""
    for k, w in enumerate(words):
        if w.kind not in ("matched", "fuzzy", "sub") or w.start is None or w.end is None:
            continue
        a = anchors.get(w.line)
        if a is not None and w.end > a.end + 0.5:
            w.end = max(w.start + 0.08, a.end)
        nxt = words[k + 1] if k + 1 < len(words) else None
        if nxt is not None and nxt.start is None and w.end - w.start > LONG_WORD_S:
            w.end = w.start + max(1.5, 2.5 * natural_duration(w.text))


def _trim_line_starts(words: list[_W]) -> None:
    """Whisper often stretches the first word after a pause back into the silence (a 1.5 s
    "Sweet" that is really sung in its last 0.6 s). For a heard line-initial word much longer
    than its natural duration, keep half of the excess."""
    for w in words:
        if w.pos != 0 or w.kind not in ("matched", "fuzzy") or w.start is None or w.end is None:
            continue
        d = w.end - w.start
        nat = natural_duration(w.text)
        cap = max(0.5, 1.5 * nat)
        if d > 2 * nat and d > cap:
            w.start = w.end - (cap + 0.5 * (d - cap))


def _apply_ops(ops: list[dp.Op], words: list[_W], toks: list[WToken]) -> set[int]:
    used: set[int] = set()
    for op in ops:
        if op.kind not in ("match", "split", "merge"):
            continue
        kind = "matched" if op.sim >= 0.999 else "fuzzy"
        ws = [toks[j] for j in op.wi]
        used.update(op.wi)
        prob = sum(t.prob for t in ws) / len(ws)
        if op.kind == "merge":
            # two lyric words in one whisper word: split its span by natural duration
            t = ws[0]
            a, b = (words[i] for i in op.li)
            da, db = natural_duration(a.text), natural_duration(b.text)
            mid = t.start + (t.end - t.start) * da / (da + db)
            for w, s, e in ((a, t.start, mid), (b, mid, t.end)):
                w.start, w.end, w.kind, w.prob, w.sim = s, e, kind, prob, op.sim
        else:
            w = words[op.li[0]]
            w.start, w.end, w.kind, w.prob, w.sim = ws[0].start, ws[-1].end, kind, prob, op.sim
    return used


def _regions(ops: list[dp.Op]) -> list[tuple[int | None, int | None, list[int], list[int]]]:
    """Gaps between pair ops: (lyric index before, lyric index after, lyric gaps, whisper gaps)."""
    out = []
    prev: int | None = None
    lg: list[int] = []
    wg: list[int] = []
    for op in ops:
        if op.kind == "lgap":
            lg.append(op.li[0])
        elif op.kind == "wgap":
            wg.append(op.wi[0])
        else:
            if lg or wg:
                out.append((prev, op.li[0], lg, wg))
            lg, wg = [], []
            prev = op.li[-1]
    if lg or wg:
        out.append((prev, None, lg, wg))
    return out


def _substitutions(ops: list[dp.Op], words: list[_W], toks: list[WToken], anchors: dict[int, Anchor]) -> set[int]:
    """Pair a lyric gap run with the whisper gap run in the same slot of the same line.

    Whisper misheard these words ("widow" for "window" is fuzzy; "out" for "loud" may not be), but
    it heard *something* exactly there, so its timing beats interpolation. Conditions: the lyric
    run is inside one line, the word counts are close, the whisper run touches the neighbouring
    matched words of that same line (within ``SUB_CONTIGUITY``), and it lies in the LRC window.
    """
    used: set[int] = set()
    for prev, nxt, lg, wg in _regions(ops):
        if not lg or not wg:
            continue
        line = words[lg[0]].line
        if any(words[i].line != line for i in lg):
            continue
        k, m = len(lg), len(wg)
        if abs(k - m) > max(1, k // 2):
            continue
        w0, w1 = toks[wg[0]], toks[wg[-1]]
        checks = 0
        ok = True
        if prev is not None and words[prev].line == line and words[prev].end is not None:
            checks += 1
            ok &= w0.start - words[prev].end <= SUB_CONTIGUITY
        if nxt is not None and words[nxt].line == line and words[nxt].start is not None:
            checks += 1
            ok &= words[nxt].start - w1.end <= SUB_CONTIGUITY
        if line in anchors:
            a = anchors[line]
            ok &= w0.start >= a.start - LRC_TOLERANCE and w1.end <= a.end + LRC_TOLERANCE
            checks += 1
        if not ok or checks == 0:
            continue
        lw = [words[i] for i in lg]
        if k == m:
            for w, t in zip(lw, (toks[j] for j in wg), strict=True):
                w.start, w.end, w.kind, w.prob = t.start, t.end, "sub", t.prob
        else:
            slots = place_from([natural_duration(w.text) for w in lw], w0.start, w1.end)
            span = w1.end - w0.start
            total = sum(s.end - s.start for s in slots) or 1.0
            t = w0.start
            prob = sum(toks[j].prob for j in wg) / m
            for w, s in zip(lw, slots, strict=True):
                d = (s.end - s.start) / total * span
                w.start, w.end, w.kind, w.prob = t, t + d, "sub", prob
                t += d
        used.update(wg)
    return used


def _interpolate(words: list[_W], anchors: dict[int, Anchor], duration: float | None) -> None:
    n = len(words)
    i = 0
    while i < n:
        if words[i].start is not None:
            i += 1
            continue
        j = i
        while j < n and words[j].start is None:
            j += 1
        _fill_run(words, i, j, anchors, duration)
        i = j


def _fill_run(words: list[_W], i: int, j: int, anchors: dict[int, Anchor], duration: float | None) -> None:
    """Time words[i:j] (all untimed) between words[i-1] and words[j] (timed, when they exist)."""
    p = words[i - 1] if i > 0 else None
    q = words[j] if j < len(words) else None
    lo = p.end if p is not None else None
    hi = q.start if q is not None else None
    run = words[i:j]
    chunks: list[list[_W]] = []
    for w in run:
        if chunks and chunks[-1][-1].line == w.line:
            chunks[-1].append(w)
        else:
            chunks.append([w])
    # 1) chunks of lines with trusted LRC anchors are placed from their own window
    placed: list[bool] = [False] * len(chunks)
    for ci, c in enumerate(chunks):
        a = anchors.get(c[0].line)
        if a is None:
            continue
        is_tail = p is not None and p.line == c[0].line
        is_head = q is not None and q.line == c[-1].line
        durs = [natural_duration(w.text) for w in c]
        win_lo = max(a.start, lo) if lo is not None else a.start
        win_hi = min(a.end, hi) if hi is not None else a.end
        if is_tail and is_head:
            continue  # inside a line: proportional fill below is better
        if is_tail:
            assert lo is not None
            slots = place_from(durs, lo, max(lo + 0.08 * len(c), min(a.end, hi) if hi is not None else a.end))
        elif is_head:
            assert hi is not None
            nat = sum(durs)
            start = a.start if hi - a.start <= max(3 * nat, nat + 1.5) else hi - 1.5 * nat
            if lo is not None:
                start = max(start, lo)
            slots = place_from(durs, start, hi, stretch=max(1.0, (hi - start) / nat) if nat else 1.0)
        else:
            if win_hi - win_lo < 0.08 * len(c):
                continue
            width = win_hi - win_lo
            nat = sum(durs)
            stretch = min(max(1.0, 0.85 * width / nat), 1.5 / (nat / len(c))) if nat else 1.0
            slots = place_from(durs, win_lo, win_hi, stretch=max(1.0, stretch))
        for w, s in zip(c, slots, strict=True):
            w.start, w.end, w.anchored = s.start, s.end, True
        placed[ci] = True
    # 2) the rest: spread groups of consecutive unplaced chunks between their timed neighbours
    ci = 0
    while ci < len(chunks):
        if placed[ci]:
            ci += 1
            continue
        cj = ci
        while cj < len(chunks) and not placed[cj]:
            cj += 1
        group = chunks[ci:cj]
        before = chunks[ci - 1][-1] if ci > 0 else p
        after = chunks[cj][0] if cj < len(chunks) else q
        g_lo = before.end if before is not None else None
        g_hi = after.start if after is not None else None
        durs = [[natural_duration(w.text) for w in c] for c in group]
        need = sum(map(sum, durs)) + BREAK_GAP * len(group)
        if g_lo is None and g_hi is None:
            g_lo, g_hi = 0.0, need
        elif g_lo is None:
            assert g_hi is not None
            g_lo = max(0.0, g_hi - need)
        elif g_hi is None:
            g_hi = g_lo + need
            if duration is not None:
                g_hi = min(g_hi, duration)
        seq: list[_W | None] = [before] + [c[0] for c in group] + [after]
        ends: list[_W | None] = [before] + [c[-1] for c in group] + [after]
        breaks = [_break_kind(ends[k], seq[k + 1]) for k in range(len(group) + 1)]
        slots = spread(durs, g_lo, max(g_lo, g_hi), breaks)
        for c, row in zip(group, slots, strict=True):
            for w, s in zip(c, row, strict=True):
                w.start, w.end = s.start, s.end
        ci = cj


def _break_kind(a: _W | None, b: _W | None) -> str | None:
    if a is None or b is None:
        return "stanza"
    if a.line == b.line:
        return None
    return "line" if a.stanza == b.stanza else "stanza"


def _build_lines(lines: list[LyricLine], words: list[_W], times: list[list[float]], raw: list[float],
                 anchors: dict[int, Anchor], preserved: Preserved) -> list[Line]:
    out: list[Line] = []
    k = 0
    for ln in lines:
        ws: list[Word] = []
        durs: list[float] = []
        for _ in ln.words:
            w = words[k]
            s, e = times[k]
            durs.append(raw[k])
            k += 1
            if w.kept is not None:
                ws.append(w.kept.model_copy())
                continue
            if w.kind in ("matched", "fuzzy"):
                origin = WordOrigin(w.kind)
            else:
                origin = WordOrigin.interpolated
            ws.append(Word(text=w.text, start=s, end=e, conf=word_conf(w.kind, w.prob, w.sim, w.anchored),
                           origin=origin))
        line = Line(id=line_id(ln.index), section=ln.section, start=min(x.start for x in ws),
                    end=max(x.end for x in ws), words=ws, flags=preserved.flags.get(ln.index, []))
        a = anchors.get(ln.index)
        line.flags = line_flags(line, durs, a.start if a else None)
        out.append(line)
    return out


def _compact(texts: list[str]) -> str:
    out: list[str] = []
    i = 0
    while i < len(texts):
        j = i
        while j < len(texts) and match_key(texts[j]) == match_key(texts[i]):
            j += 1
        out += [texts[i]] * (j - i) if j - i <= 3 else [f"{texts[i]} (x{j - i})"]
        i = j
    return " ".join(out)


def _unmatched(toks: list[WToken], used: set[int], looped: list[WToken]) -> list[UnmatchedAudio]:
    free = sorted([t for t in toks if t.index not in used] + looped, key=lambda t: t.start)
    used_starts = sorted(toks[i].start for i in used)
    groups: list[list[WToken]] = []
    for t in free:
        if groups:
            last = groups[-1][-1]
            between = bisect.bisect_left(used_starts, t.start) - bisect.bisect_right(used_starts, last.start)
            if t.start - last.end <= UNMATCHED_JOIN and between <= 0:
                groups[-1].append(t)
                continue
        groups.append([t])
    return [UnmatchedAudio(start=round(g[0].start, 3), end=round(max(t.end for t in g), 3),
                           text=_compact([t.text for t in g])) for g in groups]


def _whisper_fallback(toks: list[WToken], duration: float | None, language: str | None,
                      previous: LyricsFinal | None, stats: AlignStats,
                      envelope: Envelope | None = None) -> tuple[LyricsFinal, AlignStats]:
    out = whisper_lines(toks, duration, envelope)
    if previous is not None:
        pseudo = [LyricLine(index=i, text=" ".join(w.text for w in ln.words), section=None,
                            words=[w.text for w in ln.words]) for i, ln in enumerate(out)]
        pres = collect(previous, pseudo)
        stats.lost_edits = pres.lost
        for (li, wi), old in pres.words.items():
            out[li].words[wi] = old.model_copy()
        for ln in out:
            if any(w.origin in KEEP for w in ln.words):
                ln.start = min(w.start for w in ln.words)
                ln.end = max(w.end for w in ln.words)
    stats.words = stats.matched = sum(len(ln.words) for ln in out)
    stats.whisper_words = len(toks)
    return LyricsFinal(language=language, text_source=LyricsSourceName.whisper, lines=out), stats
