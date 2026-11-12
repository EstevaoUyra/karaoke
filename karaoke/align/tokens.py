"""Tokens for alignment: lyric words (display text plus match key) and whisper words.

Also the small parsers align needs: ``chosen.txt`` (``[section]`` headers, one sung line per
line) and LRC (``[mm:ss.xx]text``). Workstream C has its own normalizer and LRC parser in
``karaoke/lyrics``; these are kept separate on purpose until the two are deduplicated.

Match keys: lowercase, accents folded, punctuation dropped. A few informal forms map to one
canonical key (pra/para, tá/está, cê/você, gonna/going to, 'cause/because, o'/of). Multi-word
expansions are concatenated ("gonna" -> "goingto"), so a 2:1 merge in the DP compares
"going"+"to" with "gonna" exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from unidecode import unidecode

from karaoke.models import Transcript

# Raw forms (lowercased, apostrophes unified) checked before punctuation is stripped.
_RAW_EQUIV = {
    "o'": "of",
    "'cause": "because",
    "'cos": "because",
    "'coz": "because",
    "'em": "them",
    "'til": "until",
    "'till": "until",
    "'bout": "about",
    "'round": "around",
}
# Keys (after folding/stripping) mapped to a canonical key.
_KEY_EQUIV = {
    # EN
    "gonna": "goingto",
    "wanna": "wantto",
    "gotta": "gotto",
    "cause": "because",
    "cuz": "because",
    "coz": "because",
    "cos": "because",
    "till": "until",
    "til": "until",
    "ya": "you",
    "yeh": "yeah",
    "yea": "yeah",
    "ok": "okay",
    # PT
    "pra": "para",
    "pro": "parao",
    "pras": "paraas",
    "pros": "paraos",
    "ta": "esta",
    "tava": "estava",
    "tavam": "estavam",
    "tao": "estao",
    "tamo": "estamos",
    "tamos": "estamos",
    "ce": "voce",
    "ces": "voces",
    "vc": "voce",
    "vcs": "voces",
}
_APOS = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def match_key(text: str) -> str:
    """Normalized key used for matching (never displayed)."""
    raw = text.strip().lower().translate(_APOS).strip(".,;:!?\"()[]{}«»“”")
    if raw in _RAW_EQUIV:
        return _RAW_EQUIV[raw]
    if len(raw) > 3 and raw.endswith("in'"):  # nothin' -> nothing
        raw = raw[:-1] + "g"
    key = _NON_ALNUM.sub("", unidecode(raw))
    return _KEY_EQUIV.get(key, key)


def letters(text: str) -> int:
    return len(_NON_ALNUM.sub("", unidecode(text.lower())))


# --------------------------------------------------------------------------- lyrics text


@dataclass
class LyricLine:
    index: int  # 0-based position in the song
    text: str
    section: str | None
    words: list[str]  # display texts (punctuation-only tokens glued to a neighbour)
    stanza: int = 0  # increments at blank lines and section headers

    @property
    def keys(self) -> list[str]:
        return [match_key(w) for w in self.words]


_HEADER = re.compile(r"^\s*\[([^\]]*)\]\s*$")


def section_label(header: str) -> str | None:
    """'[Verse 1: Someone]' -> 'verse'; '[Chorus x2]' -> 'chorus'."""
    s = header.split(":", 1)[0].strip().lower()
    s = re.sub(r"\s*(x\s*\d+|\d+x|\(\s*x?\d+\s*\)|\d+)\s*$", "", s).strip()
    return s or None


def split_words(text: str) -> list[str]:
    """Whitespace tokens; tokens without letters or digits are glued to the previous word."""
    out: list[str] = []
    pending = ""
    for tok in text.split():
        if match_key(tok) or tok.strip("'’") and letters(tok):
            if pending:
                tok = pending + " " + tok
                pending = ""
            out.append(tok)
        elif out:
            out[-1] = out[-1] + " " + tok
        else:
            pending = (pending + " " + tok).strip()
    if pending and out:
        out[-1] += " " + pending
    return out


def parse_chosen_text(text: str) -> list[LyricLine]:
    lines: list[LyricLine] = []
    section: str | None = None
    stanza = 0
    for raw in text.splitlines():
        if not raw.strip():
            stanza += 1
            continue
        m = _HEADER.match(raw)
        if m:
            section = section_label(m.group(1))
            stanza += 1
            continue
        words = split_words(raw)
        if words:
            lines.append(LyricLine(index=len(lines), text=raw.strip(), section=section, words=words, stanza=stanza))
    stanzas = sorted({ln.stanza for ln in lines})
    renum = {k: i for i, k in enumerate(stanzas)}
    for ln in lines:
        ln.stanza = renum[ln.stanza]
    return lines


# --------------------------------------------------------------------------- LRC


@dataclass(frozen=True)
class LrcLine:
    start: float
    text: str  # "" = break (instrumental, end of the previous line)


_LRC_TS = re.compile(r"\[(\d+):(\d+(?:[.:]\d+)?)\]")
_LRC_OFFSET = re.compile(r"^\s*\[offset:\s*([+-]?\d+)\s*\]", re.I)


def parse_lrc(text: str) -> list[LrcLine]:
    """Timestamped lines sorted by time. ``[offset:+ms]`` is applied (positive = earlier)."""
    rows: list[LrcLine] = []
    offset = 0.0
    for raw in text.splitlines():
        mo = _LRC_OFFSET.match(raw)
        if mo:
            offset = int(mo.group(1)) / 1000.0
            continue
        stamps = []
        pos = 0
        while m := _LRC_TS.match(raw, pos):
            stamps.append(int(m.group(1)) * 60 + float(m.group(2).replace(":", ".")))
            pos = m.end()
        if not stamps:
            continue
        body = re.sub(r"<\d+:\d+(?:\.\d+)?>", "", raw[pos:]).strip()  # drop enhanced-LRC word tags
        rows += [LrcLine(start=max(0.0, t - offset), text=body) for t in stamps]
    rows.sort(key=lambda r: r.start)
    return rows


# --------------------------------------------------------------------------- whisper


@dataclass
class WToken:
    index: int  # position in the flat whisper word list
    text: str  # stripped display text as whisper wrote it
    key: str
    start: float
    end: float
    prob: float
    segment: int = 0
    tags: set[str] = field(default_factory=set)


def whisper_tokens(tr: Transcript) -> list[WToken]:
    """Flat list of whisper words with a non-empty key, in time order."""
    out: list[WToken] = []
    for si, seg in enumerate(tr.segments):
        for w in seg.words:
            key = match_key(w.text)
            if not key:
                continue
            out.append(WToken(index=len(out), text=w.text.strip(), key=key, start=float(w.start),
                              end=max(float(w.end), float(w.start)), prob=float(w.prob), segment=si))
    out.sort(key=lambda t: (t.start, t.index))
    for i, t in enumerate(out):
        t.index = i
    return out


LOOP_SUNG_STEP = 0.25  # seconds; see drop_hallucination_loops


def _max_run(keys: list[str], phrase: list[str]) -> int:
    """Most back-to-back copies of ``phrase`` anywhere in ``keys``."""
    p = len(phrase)
    best = 0
    i = 0
    while i + p <= len(keys):
        j = i
        while j + p <= len(keys) and keys[j:j + p] == phrase:
            j += p
        if j > i:
            best = max(best, (j - i) // p)
            i = j
        else:
            i += 1
    return best


def drop_hallucination_loops(tokens: list[WToken], lyric_keys: list[str] | None = None,
                             min_repeats: int = 8) -> tuple[list[WToken], list[WToken]]:
    """Remove whisper repetition loops: the same 1–3-word phrase repeated many times in a row.

    Songs repeat words ("sweet, sweet"), but far more back-to-back copies than the lyrics ever
    contain is whisper looping on a held note or on silence, and its timestamps are garbage.
    A loop is dropped when it has at least ``min_repeats`` copies and more than twice the
    longest run of that phrase in the lyrics (plus 2). The first copies, up to the lyrics' own
    run length, are kept; the rest are returned separately (reported as ``unmatched_audio``).

    A run whose copies are all at least ``LOOP_SUNG_STEP`` apart is kept whole: that is a sung
    hook on the beat, misheard ("toça" x9 for "dança"), and its times are the beats. Whisper's
    loops stack copies on the same instant (every loop in the jobs so far has a 0.0 s step).
    """
    lyric_keys = lyric_keys or []
    keys = [t.key for t in tokens]
    keep = [True] * len(tokens)
    i = 0
    n = len(tokens)
    while i < n:
        hit = None
        for p in (1, 2, 3):
            phrase = keys[i:i + p]
            if len(phrase) < p:
                break
            j = i + p
            while j + p <= n and keys[j:j + p] == phrase:
                j += p
            reps = (j - i) // p
            if reps < min_repeats:
                continue
            lyric_reps = _max_run(lyric_keys, phrase)
            starts = [tokens[i + r * p].start for r in range(reps)]
            if min(b - a for a, b in zip(starts, starts[1:], strict=False)) >= LOOP_SUNG_STEP:
                continue
            if reps >= 2 * lyric_reps + 2:
                hit = (j, p, max(1, lyric_reps))
                break
        if hit:
            j, p, keep_reps = hit
            for k in range(i + keep_reps * p, j):
                keep[k] = False
            i = j
        else:
            i += 1
    kept = [t for t, k in zip(tokens, keep, strict=True) if k]
    dropped = [t for t, k in zip(tokens, keep, strict=True) if not k]
    for i, t in enumerate(kept):
        t.index = i
    return kept, dropped
