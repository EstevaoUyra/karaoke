"""Conservative written syllables for PT-BR / English; pieces preserve the original text.

These are pronunciation heuristics, not a sung-language recognizer. Unsupported scripts,
letter spellings, numbers and punctuation-only tokens remain one piece. Singing can elide or
extend written syllables; timing and melismas are handled separately by melody.syllables.
"""
from __future__ import annotations

import re
import unicodedata

VOWELS = set('aeiouáàâãéêíóôõúü')
DIPHTHONGS = {'ai', 'au', 'ei', 'eu', 'oi', 'ou', 'ui', 'ão', 'ãe', 'õe'}
ONSETS = {'br', 'cr', 'dr', 'fr', 'gr', 'pr', 'tr', 'vr', 'bl', 'cl', 'fl', 'gl', 'pl', 'ch', 'lh', 'nh'}


def split_syllables(text: str, language: str | None) -> list[str]:
    """At least one piece, concatenating exactly to ``text``; never strips punctuation."""
    lang = (language or '').lower().split('-')[0]
    if lang not in {'pt', 'en'}:
        return [text]
    letters = [c for c in text if c.isalpha()]
    if not letters or any('LATIN' not in unicodedata.name(c, '') for c in letters) or any(c.isdigit() for c in text):
        return [text]
    # Preserve explicit repeats (hee-hee, mi-mi-mine), but do not guess spoken letter names.
    if '-' in text.rstrip('-'):
        parts = re.findall(r'[^-]+-*', text)
        if ''.join(parts) != text or any(sum(c.isalpha() for c in p) <= 1 for p in parts):
            return [text]
        return [s for p in parts for s in split_syllables(p, lang) if s]
    # Consonant-only contractions must not make the stem's silent e audible (there's,
    # we've). Keep the suffix attached to the last piece, including trailing punctuation.
    if lang == 'en' and (m := re.fullmatch(r"(.+?)(['’](?:s|d|ll|re|ve))([^\w]*)", text, re.IGNORECASE)):
        stem, suffix, punctuation = m.groups()
        if not (suffix.lower().endswith('s') and re.search(r'(s|z|x|sh|ch)$', stem, re.IGNORECASE)):
            pieces = split_syllables(stem, lang)
            pieces[-1] += suffix + punctuation
            return pieces
    lower = text.lower()
    vowel = [i for i, c in enumerate(lower) if c in VOWELS or (lang == 'en' and c == 'y' and i > 0)]
    if lang == 'pt':
        vowel = [i for i in vowel if not (lower[i] == 'u' and i > 0 and lower[i-1] in 'qg'
                                         and i+1 < len(lower) and lower[i+1] in 'eiéê')]
    if not vowel:
        return [text]
    nuclei = [[vowel[0]]]
    for i in vowel[1:]:
        prev = nuclei[-1][-1]
        pair = lower[prev:i+1]
        if i == prev+1 and (lang == 'en' or pair in DIPHTHONGS):
            nuclei[-1].append(i)
        else:
            nuclei.append([i])
    if lang == 'en' and len(nuclei) > 1:
        last = max(i for i, c in enumerate(lower) if c.isalpha())
        # Silent final e, except consonant + le (table, little).
        if nuclei[-1] == [last] and lower[last] == 'e' and not (
                last >= 2 and lower[last-1] == 'l' and lower[last-2] not in VOWELS):
            nuclei.pop()
        # Common unsung -ed ending; wanted/needed retain their final syllable.
        elif lower[:last+1].endswith('ed') and nuclei[-1] == [last-1] and last >= 2 and lower[last-2] not in 'td':
            nuclei.pop()
    cuts = []
    for a, b in zip(nuclei, nuclei[1:], strict=False):
        left, right = a[-1], b[0]
        consonants = [i for i in range(left+1, right) if lower[i].isalpha()]
        cut = right
        if consonants:
            cut = consonants[-1]
            if len(consonants) >= 2 and (lower[consonants[-2]:right] in ONSETS or (
                    lang == 'en' and lower[right:] == 'e' and lower[right-1] == 'l')):
                cut = consonants[-2]
        cuts.append(cut)
    edges = [0, *cuts, len(text)]
    return [text[a:b] for a, b in zip(edges, edges[1:], strict=False)]
