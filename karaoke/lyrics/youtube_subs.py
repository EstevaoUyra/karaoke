"""Parse YouTube WebVTT subtitles into clean cues.

``parse_vtt(path)`` returns ``list[Cue]`` with:

- styling removed: ``<c.colorE5E5E5>``, ``<i>``, ``<b>``, ``<v Speaker>``, inline word timestamps
  (``<00:00:01.500>``), HTML entities, cue settings (``align:start position:0%``);
- music symbols removed (♪ ♫ ♬ ♩ 🎵 🎶) and pure sound annotations dropped ("[Music]", "[Música]",
  "[Applause]"); other bracketed text is kept, it may be a lyric;
- the rolling duplicates of auto-style VTTs merged: in those files each line is shown first while
  it is being "typed", then again as the top line of the next cue (and in 10 ms filler cues), so
  every line that already appeared in the previous cue is dropped. A cue keeps the timing of the
  first appearance of its lines. This only applies to files with auto-caption markup (inline
  timestamps / ``<c>`` tags), because manual subs may repeat a line on purpose (a chorus);
- exact duplicate cues (same text and times) and empty cues dropped.

A cue's ``text`` keeps its line breaks (``\\n``): in manual lyric subs each line is usually one sung line.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str

    @property
    def lines(self) -> list[str]:
        return self.text.split("\n")


_TS = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_TIMING_RE = re.compile(rf"^\s*{_TS}\s+-->\s+{_TS}")
_TAG_RE = re.compile(r"<[^>]*>")
_INLINE_TS_RE = re.compile(r"<\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3}>")
_MUSIC_RE = re.compile(r"[♪♫♬♩🎵🎶🎼]")
_SOUND_TAG_RE = re.compile(
    r"^\s*[\[(]\s*(?:music|música|musica|instrumental|applause|aplausos|laughter|risos|silence|silêncio|"
    r"no audio|sem áudio|cheering|vocalizing|inaudible|inaudível)\s*[\])]\s*$",
    re.IGNORECASE,
)
_BLOCK_KEYWORDS = ("NOTE", "STYLE", "REGION")


def _seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def _clean_line(line: str) -> str:
    line = _TAG_RE.sub("", line)
    line = html.unescape(line).replace("​", "").replace("\xa0", " ")
    line = _MUSIC_RE.sub(" ", line)
    line = re.sub(r"\s+", " ", line).strip()
    if _SOUND_TAG_RE.match(line):
        return ""
    return line


def _raw_cues(text: str) -> list[tuple[float, float, list[str]]]:
    # Blocks are separated by EMPTY lines; whitespace-only lines belong to the cue (auto-captions
    # use them). A timing line always starts a new cue, so sloppy separators still work.
    blocks = re.split(r"\n{2,}", text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿"))
    cues: list[tuple[float, float, list[str]]] = []
    for block in blocks:
        lines = block.split("\n")
        if lines[0].strip().split(" ")[0] in _BLOCK_KEYWORDS:
            continue
        current: list[str] | None = None  # body of the cue being read, None before a timing line
        for i, line in enumerate(lines):
            m = _TIMING_RE.match(line)
            if m:
                current = []
                cues.append((_seconds(*m.groups()[:4]), _seconds(*m.groups()[4:]), current))
            elif current is not None:
                if i + 1 < len(lines) and _TIMING_RE.match(lines[i + 1]):
                    continue  # the identifier of the next cue
                current.append(line)
    return cues


def parse_vtt_text(text: str) -> list[Cue]:
    raw = _raw_cues(text)
    rolling = bool(_INLINE_TS_RE.search(text) or "<c>" in text or "<c." in text)

    out: list[Cue] = []
    prev_lines: set[str] = set()
    for start, end, body in raw:
        lines = [cl for cl in (_clean_line(x) for x in body) if cl]
        if rolling:
            cur = set(lines)
            lines = [x for x in lines if x not in prev_lines]
            prev_lines = cur
        if not lines:
            continue
        cue = Cue(round(start, 3), round(end, 3), "\n".join(lines))
        if out and out[-1] == cue:
            continue
        out.append(cue)
    return out


def parse_vtt(path: Path | str) -> list[Cue]:
    """Parse a .vtt file. See the module docstring."""
    return parse_vtt_text(Path(path).read_text(encoding="utf-8", errors="replace"))


def cues_to_text(cues: list[Cue]) -> str:
    """Plain lyrics text: one cue line per line."""
    return "\n".join(line for c in cues for line in c.lines)
