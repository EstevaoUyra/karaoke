"""Lyrics text normalization: section parsing, repeat expansion, the chosen.txt format, match tokens.

``parse_lyrics_text`` turns lyrics as lyrics sites write them into stanzas with section labels and
every repeat marker expanded. ``to_chosen_text`` writes the canonical ``lyrics/chosen.txt``:

    [verse]
    first sung line
    second sung line

    [chorus]
    ...

Section headers are written only when the section changes (consecutive stanzas of the same section
are merged), stanzas without a section are separated by a blank line and get no header. Parsing a
chosen.txt back with ``parse_lyrics_text`` gives the same sections and lines.

``match_tokens`` / ``normalize_word`` produce the tokens the aligner compares: lowercase, no
punctuation, accents folded, hyphens split, informal PT/EN forms expanded (``EQUIVALENTS``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- sections

# folded keyword → canonical section name
_SECTION_WORDS: dict[str, str] = {
    "chorus": "chorus", "refrain": "chorus", "refrao": "chorus", "coro": "chorus",
    "hook": "hook",
    "verse": "verse", "verso": "verse", "estrofe": "verse", "parte": "verse",
    "pre-chorus": "pre-chorus", "prechorus": "pre-chorus", "pre chorus": "pre-chorus",
    "pre-refrao": "pre-chorus", "pre refrao": "pre-chorus", "prerefrao": "pre-chorus",
    "post-chorus": "post-chorus", "postchorus": "post-chorus", "post chorus": "post-chorus",
    "pos-refrao": "post-chorus",
    "bridge": "bridge", "ponte": "bridge",
    "intro": "intro", "introducao": "intro",
    "outro": "outro", "final": "outro", "encerramento": "outro",
    "interlude": "interlude", "interludio": "interlude",
    "instrumental": "instrumental", "solo": "instrumental", "guitar solo": "instrumental",
    "break": "break", "breakdown": "break",
    "spoken": "spoken", "falado": "spoken",
    "rap": "rap",
}
_REPEAT_WORDS = ("repeat", "repetir", "repete", "bis")

_HEADER_BRACKET = re.compile(r"^\[\s*([^\]]+?)\s*\]$")
_HEADER_COLON = re.compile(r"^([^\W\d][\w\s\-]*?)\s*:\s*(.*)$")
# repeat count inside a header / at the end of a line
_COUNT = r"(?:[x×]\s*(?P<a>\d+)|(?P<b>\d+)\s*[x×]|(?P<rep>repeat|repetir|repete|bis)(?:\s*[x×]?\s*(?P<c>\d+)\s*[x×]?)?)"
_HEADER_COUNT = re.compile(rf"[\s(]*{_COUNT}[\s)]*$", re.IGNORECASE)
_LINE_COUNT = re.compile(rf"\s*(?:[\(\[]\s*{_COUNT}\s*[\)\]]|\s[x×](?P<d>\d+))\s*$", re.IGNORECASE)
_ONLY_COUNT = re.compile(rf"^[\(\[]?\s*{_COUNT}\s*[\)\]]?$", re.IGNORECASE)
_NUM_SUFFIX = re.compile(r"^(?P<word>.*?)\s*(?P<num>\d+)$")


@dataclass
class Stanza:
    section: str | None  # canonical: "verse", "chorus", "bridge", ... (None = unlabeled)
    lines: list[str] = field(default_factory=list)
    label: str | None = None  # section plus number, used to find what a bare header repeats: "verse 2"
    singer: str | None = None  # "[Chorus: Name]" → "Name"


@dataclass
class _Header:
    section: str
    label: str
    count: int | None  # explicit repeat count, None = not given
    repeat: bool  # header says "repeat" ("[Repeat Chorus]")
    singer: str | None


def _count_of(m: re.Match) -> int:
    for g in ("a", "b", "c", "d"):
        v = m.groupdict().get(g)
        if v:
            return max(1, int(v))
    return 2  # "(repeat)" / "(bis)" without a number: sung twice


def _parse_header_body(body: str, require_known: bool) -> _Header | None:
    body = body.strip()
    singer = None
    if ":" in body:
        body, singer = (s.strip() for s in body.split(":", 1))
        singer = singer or None
    count = None
    repeat = False
    m = _HEADER_COUNT.search(body)
    if m and m.start() > 0:
        count = _count_of(m) if (m.group("a") or m.group("b") or m.group("c")) else None
        repeat = bool(m.group("rep"))
        if count is None and (m.group("rep") or "").lower() == "bis":
            count = 2
        body = body[: m.start()].strip()
    words = fold(body)
    for rw in _REPEAT_WORDS:  # "[Repeat Chorus]"
        if words.startswith(rw + " "):
            words = words[len(rw) + 1:].strip()
            repeat = True
    words = re.sub(r"\s+", " ", words.strip(" -–—.()"))
    num = ""
    nm = _NUM_SUFFIX.match(words)
    if nm and nm.group("word"):
        words, num = nm.group("word").strip(), nm.group("num")
    section = _SECTION_WORDS.get(words)
    if section is None:
        if require_known or not words:
            return None
        section = words
    label = f"{section} {num}" if num else section
    return _Header(section, label, count, repeat, singer)


def _parse_header(line: str) -> _Header | None:
    m = _HEADER_BRACKET.match(line)
    if m:
        return _parse_header_body(m.group(1), require_known=False)
    m = _HEADER_COLON.match(line)
    if m and not m.group(2):  # "Chorus:" alone on its line
        return _parse_header_body(m.group(1), require_known=True)
    if m and _ONLY_COUNT.match(m.group(2)):  # "Chorus: (x2)"
        return _parse_header_body(f"{m.group(1)} {m.group(2)}", require_known=True)
    return None


def _clean_line(line: str) -> str:
    line = line.replace("♪", " ").replace("♫", " ")
    return re.sub(r"\s+", " ", line).strip()


def parse_lyrics_text(text: str) -> list[Stanza]:
    """Lyrics text → stanzas with sections, every repeat marker expanded.

    - Headers: ``[Chorus]``, ``[Verse 2]``, ``[Refrão]``, ``[Ponte]``, ``[Chorus: Singer]``, ``Chorus:``.
    - ``[Chorus x2]`` / ``[Refrão 2x]`` over lines: the stanza is sung that many times.
    - A bare header (no lines below it) whose section appeared before repeats that section's lines,
      once or ``xN`` times (``[Chorus x1]`` after one chorus → two choruses in total).
    - ``(x2)`` / ``(2x)`` / ``(repeat)`` / ``(bis)`` at the end of a line: that line repeated.
    - The same marker alone on a line: the stanza it closes (or the previous one) repeated.
    """
    stanzas: list[Stanza] = []
    cur: Stanza | None = None
    cur_header: _Header | None = None

    def find_previous(h: _Header) -> Stanza | None:
        for st in reversed(stanzas):
            if st.lines and st.label == h.label:
                return st
        if h.label == h.section:  # "[Chorus]" may repeat "[Chorus 1]"
            for st in reversed(stanzas):
                if st.lines and st.section == h.section:
                    return st
        return None

    def close() -> None:
        nonlocal cur, cur_header
        if cur is None:
            return
        h = cur_header
        if cur.lines:
            n = h.count if h and h.count else 1
            stanzas.extend(Stanza(cur.section, list(cur.lines), cur.label, cur.singer) for _ in range(n))
        elif h is not None:
            prev = find_previous(h)
            if prev is not None:
                n = h.count or 1
                stanzas.extend(Stanza(prev.section, list(prev.lines), prev.label, h.singer or prev.singer)
                               for _ in range(n))
        cur, cur_header = None, None

    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            close()
            continue
        m = _ONLY_COUNT.match(line)
        if m:  # "(x2)" alone: repeat the stanza it closes, else the previous one
            n = _count_of(m)
            if cur is not None and cur.lines:
                cur.lines = cur.lines * n
            elif stanzas:
                last = stanzas[-1]
                stanzas.extend(Stanza(last.section, list(last.lines), last.label, last.singer) for _ in range(n - 1))
            continue
        h = _parse_header(line)
        if h is not None:
            close()
            cur = Stanza(h.section, [], h.label, h.singer)
            cur_header = h
            continue
        n = 1
        m = _LINE_COUNT.search(line)
        if m and m.start() > 0:
            n = _count_of(m)
            line = line[: m.start()].rstrip(" ,;")
        if cur is None:
            cur = Stanza(None, [])
        cur.lines.extend([line] * n)
    close()
    stanzas = [st for st in stanzas if st.lines]
    if is_title_cased([ln for st in stanzas for ln in st.lines]):
        for st in stanzas:
            st.lines = [sentence_case(ln) for ln in st.lines]
    return stanzas


# --------------------------------------------------------------------------- capitals

# Some lyrics sites write Every Word With A Capital. When most words past the first of their line do,
# the song is put in sentence case. Normal lyrics stay well under this (names, "I", the line start).
TITLE_CASE_SHARE = 0.5
_KEEP_CAPS = {"I", "I'm", "I'll", "I've", "I'd", "I’m", "I’ll", "I’ve", "I’d"}


def _letters(word: str) -> str:
    return "".join(c for c in word if c.isalpha())


def is_title_cased(lines: list[str]) -> bool:
    n = caps = 0
    for ln in lines:
        for w in ln.split()[1:]:
            a = _letters(w)
            if a:
                n += 1
                caps += a[0].isupper()
    return n >= 10 and caps / n > TITLE_CASE_SHARE


def _titled(word: str) -> bool:
    """'Quero', 'Ve-la', 'Natty-Party': a capital, then small letters. Not 'SOM', 'KayaMan', 'I'."""
    parts = [_letters(x) for x in re.split(r"[-‐]", word)]
    parts = [x for x in parts if x]
    if word.strip(".,!?;:\"'()") in _KEEP_CAPS:
        return False
    return bool(parts) and parts[0][0].isupper() and all(x[1:] == x[1:].lower() for x in parts)


def sentence_case_words(words: list[str]) -> list[str]:
    """Lowercase the title-cased words of one line, except the first and any after . ! or ?"""
    out = []
    start = True
    for w in words:
        out.append(w if start or not _titled(w) else w.lower())
        start = w.rstrip("\"')»”").endswith((".", "!", "?"))
    return out


def sentence_case(line: str) -> str:
    return " ".join(sentence_case_words(line.split()))


def flatten(stanzas: list[Stanza]) -> list[tuple[str | None, str]]:
    """(section, line) for every sung line."""
    return [(st.section, ln) for st in stanzas for ln in st.lines]


def to_chosen_text(stanzas: list[Stanza]) -> str:
    """Stanzas → the exact chosen.txt format (see module docstring). Empty input → ""."""
    out: list[str] = []
    prev: str | None = None
    for st in stanzas:
        if not st.lines:
            continue
        if st.section is not None and st.section == prev:
            out.extend(st.lines)
            continue
        if out:
            out.append("")
        if st.section:
            out.append(f"[{st.section}]")
        out.extend(st.lines)
        prev = st.section
    return "\n".join(out) + "\n" if out else ""


# --------------------------------------------------------------------------- match tokens

_APOS = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'"})
_SPLIT = re.compile(r"[\s\-‐‑‒–—/_]+")
_EDGE = re.compile(r"^[^\w']+|[^\w']+$")

# Informal form (lowercase, with accents/apostrophes) → canonical tokens (already folded).
EQUIVALENTS: dict[str, tuple[str, ...]] = {
    # PT-BR
    "pra": ("para",), "pras": ("para", "as"), "pro": ("para", "o"), "pros": ("para", "os"),
    "prum": ("para", "um"), "pruma": ("para", "uma"),
    "tá": ("esta",), "tás": ("estas",), "tô": ("estou",), "tava": ("estava",), "tavam": ("estavam",),
    "tamo": ("estamos",), "tamos": ("estamos",),
    "cê": ("voce",), "cês": ("voces",), "ocê": ("voce",), "vc": ("voce",),
    "né": ("nao", "e"),
    # EN
    "gonna": ("going", "to"), "wanna": ("want", "to"), "gotta": ("got", "to"), "gimme": ("give", "me"),
    "'cause": ("because",), "cause": ("because",), "'cos": ("because",), "cos": ("because",),
    "cuz": ("because",), "'coz": ("because",), "coz": ("because",),
    "'til": ("until",), "til": ("until",), "till": ("until",),
    "o'": ("of",), "ya": ("you",), "'em": ("them",),
}


def fold(text: str) -> str:
    """Lowercase and strip accents (for comparisons only; display keeps the original)."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _token(s: str) -> str:
    return re.sub(r"[\W_]+", "", fold(s))


def normalize_word(word: str, expand: bool = True) -> list[str]:
    """Tokens for one display word: 0 (pure punctuation), 1, or more (hyphens, ``gonna`` → going to)."""
    out: list[str] = []
    for part in _SPLIT.split(word.lower().translate(_APOS)):
        p = _EDGE.sub("", part)
        if not p:
            continue
        if expand:
            hit = EQUIVALENTS.get(p) or EQUIVALENTS.get(p.strip("'"))
            if hit:
                out.extend(hit)
                continue
            if len(p) > 4 and p.endswith("in'"):  # lovin' → loving
                p = p[:-1] + "g"
        tok = _token(p)
        if tok:
            out.append(tok)
    return out


def match_tokens(text: str, expand: bool = True) -> list[str]:
    """Text → matching tokens: lowercase, no punctuation, accents folded, hyphens split,
    informal forms expanded (``expand=False`` keeps them as written)."""
    return [t for w in text.split() for t in normalize_word(w, expand)]


def line_key(text: str) -> str:
    """A comparable key for a whole line."""
    return " ".join(match_tokens(text))
