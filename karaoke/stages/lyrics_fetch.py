"""Stage 3 — lyrics_fetch. Owner: workstream C.

Inputs: ``meta.json``, optional ``lyrics/sources/user.txt``, optional ``lyrics/sources/youtube.vtt``.
Outputs:
- ``lyrics/sources/lrclib.lrc`` / ``lrclib.txt`` (if found), ``genius.txt`` (if a token is set and it is found).
- ``lyrics/chosen.json`` (models.LyricsChoice), with every source found and the chosen one.
- ``lyrics/chosen.txt``: the canonical text for alignment. One sung line per line, with blank lines
  between stanzas. Section headers become ``[chorus]``-style lines. ``[Chorus x2]``/"(repeat)"
  markers are EXPANDED.

Priority: ``user`` > ``lrclib`` (synced, then plain) > ``genius`` > ``youtube`` (manual subs) > ``whisper``.
``job.settings.lyrics_source`` forces a source.
- LRCLIB lookup: ``GET https://lrclib.net/api/get`` with artist/title/duration, then fall back to
  ``/api/search``. Send a User-Agent of ``karaoke (https://github.com/EstevaoUyra/karaoke)``.
- ``use_line_timings`` is True only for synced LRC with ``|duration - meta.duration| <= 2 s``.
- Language comes from ``job.settings.language``, then from lingua detection on the text (pt/en only).
  NEVER use ``meta.youtube_language``.
- If nothing is found, choose ``whisper``: write an empty chosen.txt. Alignment then uses the
  transcript text and flags every line for review.

Implementation notes:
- Every source found is listed in chosen.json (``lrclib`` can appear twice: the ``.lrc`` and the ``.txt``).
- When the synced LRC is chosen, chosen.txt is built from its lines. LRC has no section headers, so
  stanzas come from its empty (instrumental break) lines; if the plain text of the same record has
  headers and, once expanded, the same line sequence, its sections and stanzas are used instead.
- A network error keeps whatever LRCLIB/Genius files a previous run wrote (and uses them); a
  successful lookup that finds nothing deletes them.
- ``youtube.vtt`` counts only if ``meta.has_manual_subs`` is non-empty (auto captions are just ASR).
  It is parsed by the small private parser below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from karaoke.pipeline import StageContext

if TYPE_CHECKING:
    from karaoke.lyrics.normalize import Stanza

DURATION_TOLERANCE = 2.0  # seconds; same rule as SourceInfo.duration_match
YOUTUBE_STANZA_GAP = 4.0  # seconds of silence between subtitle cues that starts a new stanza


@dataclass
class _Candidate:
    name: str  # LyricsSourceName value
    file: str  # relative to the job dir
    stanzas: list[Stanza]
    synced: bool = False
    duration: float | None = None
    url: str | None = None
    note: str = ""
    language: str | None = None

    def duration_match(self, ours: float) -> bool | None:
        if self.duration is None:
            return None
        return abs(self.duration - ours) <= DURATION_TOLERANCE


# --------------------------------------------------------------------------- helpers


@lru_cache(maxsize=1)
def _detector():
    from lingua import Language, LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.PORTUGUESE).build()


def detect_language(text: str) -> str | None:
    """'pt' / 'en' / None (no text or undecidable). Only PT and EN are considered."""
    from lingua import Language

    if not text or not re.search(r"\w", text):
        return None
    lang = _detector().detect_language_of(text)
    return {Language.ENGLISH: "en", Language.PORTUGUESE: "pt"}.get(lang)


_VTT_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_VTT_TAG = re.compile(r"<[^>]+>")
_NON_LYRIC = re.compile(r"^[\[(][^\])]*[\])]$")  # "[Music]", "(applause)"


def _vtt_seconds(m: re.Match) -> float:
    h, mi, s, frac = m.group(1), m.group(2), m.group(3), m.group(4)
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + int(frac) / 10 ** len(frac)


def parse_vtt_cues(text: str) -> list[tuple[float, float, list[str]]]:
    """WebVTT → [(start, end, [text lines])], markup and ♪ removed, non-lyric cues dropped."""
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        rows = block.strip("\n").split("\n")
        for i, row in enumerate(rows):
            if "-->" not in row:
                continue
            a, b = row.split("-->", 1)
            ma, mb = _VTT_TIME.search(a), _VTT_TIME.search(b)
            if not (ma and mb):
                break
            lines = []
            for t in rows[i + 1:]:
                t = _VTT_TAG.sub("", t).replace("♪", " ").replace("♫", " ")
                t = re.sub(r"&nbsp;", " ", t)
                t = re.sub(r"\s+", " ", t).strip()
                if t and not _NON_LYRIC.match(t):
                    lines.append(t)
            if lines:
                cues.append((_vtt_seconds(ma), _vtt_seconds(mb), lines))
            break
    return cues


def _vtt_stanzas(text: str) -> list[Stanza]:
    from karaoke.lyrics.normalize import Stanza

    stanzas: list[Stanza] = []
    prev_end = None
    for start, end, lines in parse_vtt_cues(text):
        if not stanzas or (prev_end is not None and start - prev_end >= YOUTUBE_STANZA_GAP):
            stanzas.append(Stanza(None, []))
        stanzas[-1].lines.extend(lines)
        prev_end = end
    return stanzas


def lrc_stanzas(lrc_text: str, plain_text: str | None) -> tuple[list[Stanza], bool]:
    """Stanzas for chosen.txt from a synced LRC. Returns (stanzas, labels_transferred)."""
    from rapidfuzz import fuzz

    from karaoke.lyrics.lrc import parse_lrc, split_stanzas
    from karaoke.lyrics.normalize import Stanza, line_key, parse_lyrics_text

    lrc_lines = parse_lrc(lrc_text)
    base = [Stanza(None, [ln.text for ln in st]) for st in split_stanzas(lrc_lines)]
    if not plain_text:
        return base, False
    plain = parse_lyrics_text(plain_text)
    if not any(st.section for st in plain):
        return base, False
    sung = [ln.text for ln in lrc_lines if ln.text]
    plain_lines = [ln for st in plain for ln in st.lines]
    if len(sung) != len(plain_lines):
        return base, False
    if any(fuzz.ratio(line_key(a), line_key(b)) < 85 for a, b in zip(sung, plain_lines, strict=True)):
        return base, False
    out, i = [], 0
    for st in plain:  # plain structure, LRC wording
        n = len(st.lines)
        out.append(Stanza(st.section, sung[i:i + n], st.label, st.singer))
        i += n
    return out, True


def _write_or_remove(path, text: str | None) -> None:
    from karaoke.store import atomic_write_text

    if text and text.strip():
        atomic_write_text(path, text if text.endswith("\n") else text + "\n")
    else:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- sources


def _lrclib_candidates(ctx: StageContext, title: str, artist: str | None, meta) -> list[_Candidate]:
    import httpx

    from karaoke.lyrics import lrclib
    from karaoke.lyrics.lrc import format_time, lrc_duration, parse_lrc_tags
    from karaoke.lyrics.normalize import parse_lyrics_text

    p = ctx.paths
    lrc_path, txt_path = p.lyrics_source("lrclib.lrc"), p.lyrics_source("lrclib.txt")
    try:
        match = lrclib.find_best(title, artist, meta.duration, extra=meta.clean_extra, album=meta.album)
    except httpx.HTTPError as exc:
        ctx.log.warning("LRCLIB unreachable (%s); using files from a previous run if any", exc)
        out = []
        lrc_text = lrc_path.read_text(encoding="utf-8") if lrc_path.exists() else None
        txt_text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else None
        if lrc_text:
            st, _ = lrc_stanzas(lrc_text, txt_text)
            out.append(_Candidate("lrclib", p.rel(lrc_path), st, synced=True, duration=lrc_duration(lrc_text),
                                  note="cached (LRCLIB unreachable)"))
        if txt_text:
            out.append(_Candidate("lrclib", p.rel(txt_path), parse_lyrics_text(txt_text),
                                  note="cached (LRCLIB unreachable)"))
        return out

    if match is None:
        ctx.log.info("LRCLIB: nothing found for %r / %r", artist, title)
        lrc_path.unlink(missing_ok=True)
        txt_path.unlink(missing_ok=True)
        return []

    rec = match.record
    note = f"LRCLIB #{rec.id}: {rec.artist_name} - {rec.track_name}"
    if rec.album_name:
        note += f" [{rec.album_name}]"
    note += f"; {match.n_candidates} candidates; found by {match.query}"
    lrc_text = None
    if rec.synced:
        existing = parse_lrc_tags(rec.synced_lyrics)
        tags = {"ar": rec.artist_name, "ti": rec.track_name, "al": rec.album_name or "",
                "length": format_time(rec.duration) if rec.duration else ""}
        head = "".join(f"[{k}:{v}]\n" for k, v in tags.items() if v and k not in existing)
        lrc_text = head + rec.synced_lyrics.strip("\n") + "\n"
    plain = rec.plain_lyrics if rec.plain_lyrics and rec.plain_lyrics.strip() else None
    _write_or_remove(lrc_path, lrc_text)
    _write_or_remove(txt_path, plain)

    out = []
    if lrc_text:
        st, transferred = lrc_stanzas(lrc_text, plain)
        out.append(_Candidate("lrclib", p.rel(lrc_path), st, synced=True, duration=rec.duration, url=rec.url,
                              note=note + ("; sections from the plain text" if transferred else "")))
    if plain:
        out.append(_Candidate("lrclib", p.rel(txt_path), parse_lyrics_text(plain), duration=rec.duration,
                              url=rec.url, note=note))
    return out


def _genius_candidate(ctx: StageContext, title: str, artist: str | None) -> _Candidate | None:
    from karaoke.lyrics.normalize import parse_lyrics_text

    p = ctx.paths
    path = p.lyrics_source("genius.txt")
    token = ctx.settings.genius_token
    if not token:
        if path.exists() and path.read_text(encoding="utf-8").strip():
            return _Candidate("genius", p.rel(path), parse_lyrics_text(path.read_text(encoding="utf-8")),
                              note="existing file (no Genius token set)")
        return None
    import httpx

    from karaoke.lyrics import genius

    try:
        found = genius.find_lyrics(title, artist, token)
    except httpx.HTTPError as exc:
        ctx.log.warning("Genius unreachable (%s)", exc)
        if path.exists() and path.read_text(encoding="utf-8").strip():
            return _Candidate("genius", p.rel(path), parse_lyrics_text(path.read_text(encoding="utf-8")),
                              note="cached (Genius unreachable)")
        return None
    if found is None:
        path.unlink(missing_ok=True)
        return None
    _write_or_remove(path, found.text)
    return _Candidate("genius", p.rel(path), parse_lyrics_text(found.text), url=found.song.url,
                      note=f"Genius #{found.song.id}: {found.song.artist} - {found.song.title}")


def _youtube_candidate(ctx: StageContext, meta) -> _Candidate | None:
    p = ctx.paths
    path = p.lyrics_source("youtube.vtt")
    if not path.exists():
        return None
    if not meta.has_manual_subs:
        ctx.log.info("youtube.vtt ignored: meta.has_manual_subs is empty (automatic captions are not lyrics)")
        return None
    st = _vtt_stanzas(path.read_text(encoding="utf-8", errors="replace"))
    if not any(s.lines for s in st):
        return None
    url = meta.url or (f"https://www.youtube.com/watch?v={meta.video_id}" if meta.video_id else None)
    return _Candidate("youtube", p.rel(path), st, synced=True, url=url,
                      note="manual subtitles: " + ", ".join(meta.has_manual_subs))


# --------------------------------------------------------------------------- choice


def _choose(cands: list[_Candidate], forced: str | None, log) -> _Candidate | None:
    """cands are already in priority order."""
    usable = [c for c in cands if any(s.lines for s in c.stanzas)]
    if forced:
        if forced == "whisper":
            return None
        pick = [c for c in usable if c.name == forced]
        if pick:
            return pick[0]
        log.warning("forced lyrics source %r not available; falling back to the priority order", forced)
    return usable[0] if usable else None


def run(ctx: StageContext) -> None:
    from karaoke.lyrics.normalize import parse_lyrics_text, to_chosen_text
    from karaoke.models import LyricsChoice, LyricsSourceName, Meta, SourceInfo
    from karaoke.store import atomic_write_text, read_model, write_model

    p = ctx.paths
    if not p.meta.exists():
        raise FileNotFoundError(f"{p.rel(p.meta)} is missing: run ingest first")
    meta = read_model(p.meta, Meta)
    p.lyrics_sources_dir.mkdir(parents=True, exist_ok=True)
    forced = ctx.job.settings.lyrics_source
    if forced is not None:
        forced = LyricsSourceName(forced).value  # raises on an unknown name

    title = meta.clean_title or meta.track or meta.title
    artist = meta.clean_artist or meta.artist
    cands: list[_Candidate] = []

    user = p.lyrics_source("user.txt")
    if user.exists() and user.read_text(encoding="utf-8").strip():
        cands.append(_Candidate("user", p.rel(user), parse_lyrics_text(user.read_text(encoding="utf-8")),
                                note="pasted by the user"))

    ctx.progress(0.1, f"LRCLIB: {artist or '?'} - {title}")
    cands.extend(_lrclib_candidates(ctx, title, artist, meta))
    ctx.progress(0.5, "Genius" if ctx.settings.genius_token else "Genius skipped (no token)")
    g = _genius_candidate(ctx, title, artist)
    if g:
        cands.append(g)
    ctx.progress(0.7, "YouTube subtitles")
    y = _youtube_candidate(ctx, meta)
    if y:
        cands.append(y)

    chosen = _choose(cands, forced, ctx.log)
    text = to_chosen_text(chosen.stanzas) if chosen else ""

    ctx.progress(0.85, "detecting language")
    for c in cands:
        c.language = detect_language("\n".join(ln for s in c.stanzas for ln in s.lines))
    language = ctx.job.settings.language or (chosen.language if chosen else None)

    sources = [
        SourceInfo(name=LyricsSourceName(c.name), file=c.file, synced=c.synced, language=c.language,
                   duration=c.duration, duration_match=c.duration_match(meta.duration), url=c.url,
                   note=("chosen; " if c is chosen else "") + c.note)
        for c in cands
    ]
    if chosen is None:
        why = "forced by job settings" if forced == "whisper" else "no lyrics found"
        sources.append(SourceInfo(name=LyricsSourceName.whisper, file=p.rel(p.transcript),
                                  note=f"chosen; {why}: align uses the transcript text"))
    use_line_timings = bool(chosen and chosen.synced and chosen.duration_match(meta.duration))
    choice = LyricsChoice(
        chosen=LyricsSourceName(chosen.name) if chosen else LyricsSourceName.whisper,
        language=language,
        use_line_timings=use_line_timings,
        sources=sources,
    )
    atomic_write_text(p.lyrics_chosen_text, text)
    write_model(p.lyrics_choice, choice)

    if chosen:
        delta = abs(chosen.duration - meta.duration) if chosen.duration is not None else None
        msg = (f"{chosen.name} ({chosen.file}), "
               f"{f'Δ{delta:.1f} s, ' if delta is not None else ''}"
               f"line timings {'on' if use_line_timings else 'off'}, language {language}")
    else:
        msg = "no lyrics found: whisper fallback"
    ctx.progress(1.0, msg)
