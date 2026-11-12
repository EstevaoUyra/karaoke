"""Japanese songs: whisper hears kanji and kana, the lyrics are romaji. Make both romaji words.

Whisper (``language="ja"``) returns words that are a character or a few characters long, with
times. The lyrics are romaji in the usual singer's spelling ("hana ni natte"). To align them:

1. Each transcript segment is parsed as one text with fugashi (MeCab + UniDic), so kanji get their
   reading from context. Every character keeps the time of the whisper word it came from
   (spread evenly across that word), so a morpheme's time is that of its characters.
2. Morphemes are grouped into words the way romaji lyrics are written: auxiliary verbs, suffixes
   and the connective te/de stay with the word before ("nat" + "te" -> "natte"); case and topic
   particles stand alone ("hana", "ni").
3. Each word is romanized from its *pronunciation* in context (cutlet, Hepburn), so the particle
   wa/e/o is spelled as sung and a small tsu doubles the next consonant.
4. ``romaji_key`` folds the spellings that differ between romaji styles (long vowels as "ou",
   "oo", a macron, or a single vowel; "wo" for the particle) on both sides before matching.

Heavy imports (fugashi, cutlet, the UniDic dictionary) happen on first use.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from karaoke.align.tokens import WToken, match_key
from karaoke.models import Transcript

_JA = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")  # kana, CJK ideographs, half-width kana


def is_japanese(text: str) -> bool:
    return bool(_JA.search(text))


# --------------------------------------------------------------------------- keys


_LONG = [("ou", "o"), ("oo", "o"), ("uu", "u"), ("aa", "a"), ("ii", "i"), ("ee", "e")]


def romaji_key(key: str) -> str:
    """Fold romaji spelling variants in a ``match_key`` (already lowercase ASCII)."""
    if key == "wo":
        return "o"
    for a, b in _LONG:
        key = key.replace(a, b)
    return key


# --------------------------------------------------------------------------- tokens


@lru_cache(maxsize=1)
def _tools():
    import cutlet
    import fugashi

    cut = cutlet.Cutlet()
    cut.use_foreign_spelling = False  # pronunciations are katakana: "イル" must stay "iru", not "ile"
    return fugashi.Tagger(), cut


def _joins_previous(pos1: str, pos2: str) -> bool:
    """Does this morpheme belong to the word before it in written romaji?"""
    if pos1 in ("助動詞", "接尾辞"):  # auxiliary verb (ta, nai, masu…), suffix
        return True
    return pos1 == "助詞" and pos2 == "接続助詞"  # connective particle te/de, ba, kara…


def _pron(node) -> str:
    f = node.feature
    for attr in ("pron", "kana"):
        v = getattr(f, attr, None)
        if v and v != "*":
            return v
    return node.surface


def romaji_tokens(tr: Transcript) -> list[WToken]:
    """Whisper's Japanese words as romaji words with times (see the module docstring)."""
    tagger, cut = _tools()
    out: list[WToken] = []
    for si, seg in enumerate(tr.segments):
        chars: list[tuple[str, float, float, float]] = []  # (char, start, end, prob)
        for w in seg.words:
            text = "".join(c for c in w.text if not c.isspace())
            if not text:
                continue
            s, e = float(w.start), max(float(w.end), float(w.start))
            step = (e - s) / len(text)
            for k, c in enumerate(text):
                chars.append((c, s + k * step, s + (k + 1) * step, float(w.prob)))
        if not chars:
            continue
        text = "".join(c for c, *_ in chars)
        groups: list[list[tuple[str, int, int]]] = []  # per word: (pronunciation, first char, end char)
        pos = 0
        for node in tagger(text):
            surf = node.surface
            i = text.find(surf, pos) if surf else -1
            if i < 0:
                continue
            j = i + len(surf)
            pos = j
            f = node.feature
            item = (_pron(node), i, j)
            if groups and _joins_previous(getattr(f, "pos1", ""), getattr(f, "pos2", "")):
                groups[-1].append(item)
            else:
                groups.append([item])
        for g in groups:
            kana = "".join(p for p, _, _ in g)
            i0, i1 = g[0][1], g[-1][2]
            roma = cut.romaji(unicodedata.normalize("NFKC", kana)).strip()
            key = romaji_key(match_key(roma.replace(" ", "")))
            if not key:
                continue
            span = chars[i0:i1]
            out.append(WToken(index=len(out), text=roma.lower(), key=key, start=span[0][1],
                              end=span[-1][2], prob=min(p for *_, p in span), segment=si))
    out.sort(key=lambda t: (t.start, t.index))
    for i, t in enumerate(out):
        t.index = i
    return out
