"""Turn a YouTube title (+ uploader / yt-dlp music fields) into artist, title and search qualifiers.

``clean(title, uploader, artist, track)`` is a pure function (no IO). It:

- prefers yt-dlp's ``track``/``artist`` when present;
- strips noise in EN and PT-BR: "(Official Video)", "[HD]", "Lyrics", "Letra", "Clipe Oficial",
  "Áudio Oficial", "ft./feat./part." sections, hashtags, "VEVO" / "- Topic" in the uploader, ...;
- keeps qualifiers that identify the *recording* in ``extra`` instead of discarding them
  ("Tropical Paradise Riddim", "Ao Vivo", "Live at ...", "Remix", the film of a soundtrack ...),
  because the lyrics search uses them to find the right version (see PLAN.md §4, LRCLIB lookup);
- never uses an untrusted uploader (e.g. a lyrics channel) as the artist. The uploader becomes
  the artist only for "X - Topic", "XVEVO" and "X Official" channels; otherwise it is only used
  to decide which side of "A - B" is the artist.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class CleanTitle:
    artist: str | None
    title: str
    extra: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- patterns

_I = re.IGNORECASE

# Phrases that never carry information about the recording. Matched as whole words; longest first.
_NOISE_PHRASES = [
    r"official\s+(?:hd\s+)?(?:music\s+|lyrics?\s+|4k\s+)?(?:video|vídeo|audio|áudio|visuali[sz]er|clip|mv)",
    r"(?:music|lyrics?|letra|performance|animated)\s+(?:video|vídeo)(?:\s+oficial|\s+official)?",
    r"(?:vídeo|video|clipe|clip|videoclipe|áudio|audio|lyric|letra)\s+oficial",
    r"videoclipe",
    r"(?:with|w/)\s+lyrics",
    r"com\s+letra",
    r"lyrics?",
    r"letra",
    r"legendado(?:\s+pt(?:-br)?)?",
    r"tradução",
    r"traduzido",
    r"traducida",
    r"visuali[sz]er",
    r"official",
    r"oficial",
    r"(?:full\s+)?hd",
    r"hq",
    r"uhd",
    r"[48]k(?:\s+remaster(?:ed)?)?",
    r"(?:1080|720|2160)p",
    r"m/?v",
    r"(?:\d{4}\s+)?remaster(?:ed|izado|izada)?(?:\s+(?:version|versão|\d{4}))*",
    r"explicit",
    r"lançamento(?:\s+\d{4})?",
    r"out\s+now",
    r"(?:new|nova)\s+(?:song|música|musica)",
    r"audio",
    r"áudio",
    r"video",
    r"vídeo",
    r"clipe",
]
_NOISE_RE = re.compile(r"(?<!\w)(?:" + "|".join(_NOISE_PHRASES) + r")(?!\w)", _I)

# Trailing noise outside brackets is matched more strictly (no bare "video"/"audio": "Home Video").
_TRAILING_NOISE_RE = re.compile(
    r"(?:^|[\s\-–|:/,])+(?:"
    + "|".join(p for p in _NOISE_PHRASES if p not in ("audio", "áudio", "video", "vídeo", "clipe", "official",
                                                       "oficial", "explicit", "m/?v"))
    + r")\s*$",
    _I,
)

# "ft. X", "feat. X", "featuring X", "part. X" (PT), "participação (especial) de X" — to the end.
_FEAT_RE = re.compile(
    r"(?:^|\s)(?:ft\.?|feat\.?|featuring|part\.|participação(?:\s+especial)?(?:\s+de)?)\s+.*$", _I
)
_FEAT_START_RE = re.compile(r"^\s*(?:ft\.?|feat\.?|featuring|part\.|participação)(?:\s|$)", _I)
_PROD_RE = re.compile(r"^\s*(?:prod(?:\.|uced)?|produção|produzido)(?:\s+(?:by|por))?(?:\s|$)", _I)

# Words that describe a qualifier's *kind* rather than name it ("Captain Fantastic soundtrack" →
# "Captain Fantastic"). Removed from qualifiers; if nothing is left, the qualifier is dropped.
_CONTEXT_RE = re.compile(
    r"(?<!\w)(?:original\s+(?:motion\s+picture\s+)?soundtrack|soundtrack|o\.?s\.?t\.?|"
    r"trilha\s+sonora(?:\s+(?:original|oficial))?(?:\s+(?:de|do|da|dos|das))?|"
    r"from\s+the\s+(?:movie|film|series|motion\s+picture)|do\s+filme|da\s+novela|da\s+série|"
    r"cover(?:\s+(?:by|de|por))?|versão\s+de)(?!\w)",
    _I,
)

_LEADING_FROM_RE = re.compile(r"^\s*(?:from|by|de|do|da)\s+", _I)

# A " - " part that is a qualifier rather than an artist or a title.
_QUALIFIER_PART_RE = re.compile(
    r"(?<!\w)(?:soundtrack|o\.?s\.?t\.?|trilha\s+sonora|cover|remix|version|versão|unplugged|acoustic|"
    r"acústico|riddim|radio\s+edit|extended\s+mix)(?!\w)"
    r"|^\s*(?:live|ao\s+vivo|en\s+vivo)(?:\s+(?:at|in|from|em|no|na|@)\b.*)?\s*$",
    _I,
)

_BRACKETS = {"(": ")", "[": "]", "{": "}", "【": "】", "「": "」", "『": "』", "《": "》"}
_BRACKET_RE = re.compile(r"\(([^()]*)\)|\[([^\[\]]*)\]|\{([^{}]*)\}|【([^】]*)】|「([^」]*)」|『([^』]*)』|《([^》]*)》")
_QUOTED_RE = re.compile(r"[\"“”]([^\"“”]+)[\"“”]")
_HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
_DASH_RE = re.compile(r"\s+[-–—―~]\s+|\s*[–—―]\s*")  # " - " (hyphen needs spaces) or any long dash
_PIPE_RE = re.compile(r"\s*(?:\||//|｜)\s*")

_STRIP_CHARS = " \t-–—―~|:;,./\\·•*+_\"“”'‘’"

_SMALL_WORDS = {
    "a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "vs",
    "e", "o", "os", "da", "das", "de", "do", "dos", "em", "na", "nas", "no", "nos", "com", "para",
}

_UPLOADER_TRUSTED_RE = re.compile(
    r"^(?P<name>.+?)(?:\s+-\s+topic|\s*vevo|\s+(?:official|oficial)(?:\s+channel|\s+canal)?)\s*$", _I
)
_UPLOADER_SUFFIX_RE = re.compile(r"(?:\s+-\s+topic|\s*vevo|\s+(?:official|oficial)(?:\s+channel)?|\s+tv)\s*$", _I)


# --------------------------------------------------------------------------- helpers


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strip(s: str) -> str:
    return _ws(s).strip(_STRIP_CHARS).strip()


def _key(s: str | None) -> str:
    """Comparison key: casefolded, accents folded, alphanumerics only."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s.casefold())
    return "".join(c for c in s if c.isalnum())


def _same(a: str | None, b: str | None) -> bool:
    ka, kb = _key(a), _key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    shorter, longer = sorted((ka, kb), key=len)
    return len(shorter) >= 4 and longer.startswith(shorter)


def _titlecase(s: str) -> str:
    """Capitalise all-lowercase words (except small words after the first): 'captain fantastic' → 'Captain Fantastic'."""
    out = []
    for i, w in enumerate(s.split(" ")):
        if w.islower() and (i == 0 or w not in _SMALL_WORDS):
            w = w[0].upper() + w[1:]
        out.append(w)
    return " ".join(out)


def _remove_noise(s: str) -> str:
    return _strip(_NOISE_RE.sub(" ", s))


def _strip_trailing_noise(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = _strip(_TRAILING_NOISE_RE.sub("", s))
    return s


def _remove_feat(s: str) -> str:
    return _strip(_FEAT_RE.sub("", s))


def _qualifier(content: str) -> str | None:
    """Clean a bracket/part qualifier; None if it carries nothing useful for the lyrics search."""
    content = _HASHTAG_RE.sub(" ", content)
    if _FEAT_START_RE.match(content) or _PROD_RE.match(content):
        return None
    content = _remove_feat(_remove_noise(content))
    content = _strip(_CONTEXT_RE.sub(" ", content))
    content = _strip(_LEADING_FROM_RE.sub("", content))
    content = _strip(_remove_noise(content))
    if not content or not any(c.isalnum() for c in content):
        return None
    return _titlecase(content)


def _clean_name(s: str) -> str:
    s = _HASHTAG_RE.sub(" ", s)
    return _remove_feat(_strip_trailing_noise(_strip(s)))


def _uploader_artist(uploader: str | None) -> str | None:
    """The artist name if the uploader is an artist channel ('X - Topic', 'XVEVO', 'X Official')."""
    if not uploader:
        return None
    m = _UPLOADER_TRUSTED_RE.match(uploader.strip())
    return _strip(m.group("name")) if m else None


def _uploader_name(uploader: str | None) -> str | None:
    if not uploader:
        return None
    return _strip(_UPLOADER_SUFFIX_RE.sub("", uploader.strip())) or None


def _add(extra: list[str], q: str | None) -> None:
    if q and _key(q) not in {_key(e) for e in extra}:
        extra.append(q)


def _extract_brackets(s: str, extra: list[str]) -> str:
    """Remove every bracket group from ``s``; recording qualifiers among them go to ``extra``."""
    while True:
        m = _BRACKET_RE.search(s)
        if not m:
            break
        content = next(g for g in m.groups() if g is not None)
        _add(extra, _qualifier(content))
        s = s[: m.start()] + " " + s[m.end():]
    # unbalanced leftovers
    return _ws(re.sub(r"[()\[\]{}【】「」『』《》]", " ", s))


# --------------------------------------------------------------------------- main


def _parse(title: str, uploader: str | None, artist_hint: str | None, track_hint: str | None) -> CleanTitle:
    extra: list[str] = []
    s = unicodedata.normalize("NFC", title)
    s = _HASHTAG_RE.sub(" ", s)

    # 'Artist "Song" (Official Video)' — a quoted title.
    quoted = _QUOTED_RE.search(s)
    if quoted and not _DASH_RE.search(s):
        before, after = s[: quoted.start()], s[quoted.end():]
        rest = _strip(_extract_brackets(after, extra))
        if rest:
            _add(extra, _qualifier(rest))
        t = _clean_name(_extract_brackets(quoted.group(1), extra))
        a = _clean_name(_extract_brackets(before, extra)) or None
        return CleanTitle(a or artist_hint or _uploader_artist(uploader), t, extra)

    s = _extract_brackets(s, extra)

    # "Artist - Song | Channel | Official Video": keep the first segment with content.
    segments = [seg for seg in _PIPE_RE.split(s) if _strip(_strip_trailing_noise(seg))]
    if segments:
        main = next((seg for seg in segments if _DASH_RE.search(seg)), segments[0])
        s = main

    parts = [_strip(_strip_trailing_noise(p)) for p in _DASH_RE.split(s)]
    parts = [p for p in parts if p and _remove_noise(p)]
    if not parts:
        return CleanTitle(artist_hint or _uploader_artist(uploader), _strip(title), extra)

    # Qualifier parts ("Captain fantastic soundtrack", "Live at X") — never the first part.
    kept = [parts[0]]
    for p in parts[1:]:
        if _QUALIFIER_PART_RE.search(p) and not _same(p, track_hint):
            _add(extra, _qualifier(p))
        else:
            kept.append(p)
    parts = [_clean_name(p) for p in kept]
    parts = [p for p in parts if p]
    if not parts:
        return CleanTitle(artist_hint or _uploader_artist(uploader), _strip(title), extra)

    if len(parts) == 1:
        return CleanTitle(artist_hint or _uploader_artist(uploader), parts[0], extra)

    a, b = parts[0], parts[1]
    for p in parts[2:]:
        _add(extra, _qualifier(p))

    hints = [h for h in (artist_hint, _uploader_name(uploader)) if h]
    if track_hint and _same(a, track_hint) and not _same(b, track_hint):
        a, b = b, a
    elif not (track_hint and _same(b, track_hint)):
        if any(_same(b, h) for h in hints) and not any(_same(a, h) for h in hints):
            a, b = b, a  # "Song - Artist"
    return CleanTitle(a, b, extra)


def clean(
    title: str, uploader: str | None = None, artist: str | None = None, track: str | None = None
) -> CleanTitle:
    """Clean a YouTube title for the lyrics lookup. See the module docstring."""
    artist = _clean_name(_UPLOADER_SUFFIX_RE.sub("", artist)) if artist and artist.strip() else None
    track = track.strip() if track and track.strip() else None
    parsed = _parse(title or "", uploader, artist, track)

    if track:
        extra: list[str] = []
        t = _clean_name(_extract_brackets(track, extra))
        tparts = [_strip(p) for p in _DASH_RE.split(t)]
        t = tparts[0] or _strip(track)
        for p in tparts[1:]:
            _add(extra, _qualifier(p))
        for q in parsed.extra:
            _add(extra, q)
        a = artist
        if not a:
            a = parsed.artist if _same(parsed.title, t) else (parsed.artist or _uploader_artist(uploader))
        return CleanTitle(a or None, t, extra)

    if artist:
        return CleanTitle(artist, parsed.title, parsed.extra)
    return parsed
