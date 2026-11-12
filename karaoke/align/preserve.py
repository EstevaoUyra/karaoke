"""Carry human (``manual``) and AI (``ai``) word edits from a previous lyrics.final.json.

Those words are never overwritten by ``align``. They are found in the new lines:
1. by line id plus word index, when the line is otherwise unchanged (same word count and every
   non-edited word has the same match key);
2. otherwise by text: the most similar new line (prefer the same position), then the word with
   the edited word's key nearest to its old index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from karaoke.align.tokens import LyricLine, match_key
from karaoke.models import Line, LyricsFinal, Word, WordOrigin

KEEP = (WordOrigin.manual, WordOrigin.ai)


def line_id(index: int) -> str:
    return f"l{index + 1:03d}"


@dataclass
class Preserved:
    words: dict[tuple[int, int], Word]  # (line index, word index) -> old word, kept exactly
    flags: dict[int, list[str]]  # line index -> old non-computed flags (e.g. ai_edited)
    lost: list[str]  # descriptions of edits that could not be placed
    # Old lines kept whole, beside the lyrics: lines a person or the AI added (every word edited,
    # matching no lyric line, e.g. a spoken intro) and edited lines filled in from whisper ("w001").
    standalone: list[Line] = field(default_factory=list)


def _joined(words: list[str]) -> str:
    return " ".join(match_key(w) for w in words)


def collect(previous: LyricsFinal | None, lines: list[LyricLine]) -> Preserved:
    out = Preserved({}, {}, [])
    if previous is None:
        return out
    by_id = {line_id(ln.index): ln for ln in lines}
    used: set[tuple[int, int]] = set()
    new_texts = [_joined(ln.words) for ln in lines]
    for old in previous.lines:
        keep = [(i, w) for i, w in enumerate(old.words) if w.origin in KEEP]
        if not keep:
            continue
        if old.id.startswith("w"):  # filled in from whisper, then edited: there is no lyric line for it
            out.standalone.append(old.model_copy(deep=True))
            continue
        target = _same_line(old, by_id.get(old.id))
        if target is not None:
            for i, w in keep:
                out.words[(target.index, i)] = w
                used.add((target.index, i))
            out.flags[target.index] = list(old.flags)
            continue
        old_text = _joined([w.text for w in old.words])
        old_idx = int(old.id[1:]) - 1 if old.id[1:].isdigit() else 0
        ranked = sorted(range(len(lines)), key=lambda k: (-fuzz.ratio(old_text, new_texts[k]), abs(k - old_idx)))
        cand = next((k for k in ranked[:1] if fuzz.ratio(old_text, new_texts[k]) >= 70), None)
        added = len(keep) == len(old.words)  # every word edited: possibly a line added by hand
        if added and cand is not None and any((cand, j) in used for j in range(len(lines[cand].words))):
            cand = None  # the lyric line already holds another old line: this one is an added repeat
        if cand is None:
            if added:  # a line added by hand: keep it whole
                out.standalone.append(old.model_copy(deep=True))
            else:
                out.lost += [f"{old.id} word {i} {w.text!r}" for i, w in keep]
            continue
        keys = lines[cand].keys
        for i, w in keep:
            k = match_key(w.text)
            options = [j for j, kk in enumerate(keys) if kk == k and (cand, j) not in used]
            if not options and len(keys) == len(old.words) and (cand, i) not in used:
                options = [i]
            if not options:
                out.lost.append(f"{old.id} word {i} {w.text!r}")
                continue
            j = min(options, key=lambda j: abs(j - i))
            out.words[(cand, j)] = w
            used.add((cand, j))
        out.flags.setdefault(cand, list(old.flags))
    return out


def _same_line(old: Line, new: LyricLine | None) -> LyricLine | None:
    if new is None or len(new.words) != len(old.words):
        return None
    for w, nw in zip(old.words, new.words, strict=True):
        if w.origin not in KEEP and match_key(w.text) != match_key(nw):
            return None
    return new
