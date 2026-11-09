"""LRC (line-synced lyrics) parsing and formatting.

Handles ``[mm:ss]``, ``[mm:ss.xx]``, ``[mm:ss.xxx]`` and ``[mm:ss:xx]`` timestamps, several timestamps
on one line (``[00:12.00][00:48.00]repeated line``), ID tags (``[ar:]``, ``[ti:]``, ``[al:]``,
``[length:]``, ``[offset:]``) and enhanced-LRC word stamps (``<00:12.34>``, stripped).
A timestamped line with no text is an instrumental break: it is kept with ``text == ""``.
"""

from __future__ import annotations

import re
from typing import NamedTuple

_TS = re.compile(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
_TAG = re.compile(r"^\[([a-zA-Z#]+):(.*)\]\s*$")
_WORD_TS = re.compile(r"<\d{1,3}:\d{1,2}(?:[.:]\d{1,3})?>")


class LrcLine(NamedTuple):
    time: float  # seconds
    text: str  # "" = instrumental break


def _ts_seconds(m: re.Match) -> float:
    minutes, seconds, frac = m.group(1), m.group(2), m.group(3)
    t = int(minutes) * 60 + int(seconds)
    if frac:
        t += int(frac) / (10 ** len(frac))
    return t


def parse_time(s: str) -> float | None:
    """'03:40.50' / '3:40' / '220' / '220.5' → seconds."""
    s = s.strip()
    m = _TS.fullmatch(f"[{s}]")
    if m:
        return _ts_seconds(m)
    try:
        return float(s)
    except ValueError:
        return None


def parse_lrc_tags(text: str) -> dict[str, str]:
    """ID tags, keys lowercased: {'ar': 'Artist', 'ti': 'Title', 'length': '03:40.00', ...}."""
    tags: dict[str, str] = {}
    for raw in text.splitlines():
        m = _TAG.match(raw.strip())
        if m and not _TS.match(raw.strip()):
            tags[m.group(1).lower()] = m.group(2).strip()
    return tags


def parse_lrc(text: str) -> list[LrcLine]:
    """Timestamped lines sorted by time (stable for equal times). ``[offset:]`` is applied."""
    tags = parse_lrc_tags(text)
    offset = 0.0
    if "offset" in tags:
        try:
            offset = int(tags["offset"]) / 1000.0  # positive offset = lyrics come earlier
        except ValueError:
            offset = 0.0
    out: list[LrcLine] = []
    for raw in text.splitlines():
        line = raw.strip()
        times: list[float] = []
        while True:
            m = _TS.match(line)
            if not m:
                break
            times.append(_ts_seconds(m))
            line = line[m.end():].lstrip()
        if not times:
            continue
        body = re.sub(r"\s+", " ", _WORD_TS.sub("", line)).strip()
        for t in times:
            out.append(LrcLine(max(0.0, round(t - offset, 3)), body))
    out.sort(key=lambda ln: ln.time)
    return out


def lrc_duration(text: str) -> float | None:
    """Duration from the ``[length:]`` tag, if any."""
    length = parse_lrc_tags(text).get("length")
    return parse_time(length) if length else None


def split_stanzas(lines: list[LrcLine]) -> list[list[LrcLine]]:
    """Group sung lines into stanzas, using empty (instrumental break) lines as boundaries."""
    stanzas: list[list[LrcLine]] = []
    cur: list[LrcLine] = []
    for ln in lines:
        if ln.text:
            cur.append(ln)
        elif cur:
            stanzas.append(cur)
            cur = []
    if cur:
        stanzas.append(cur)
    return stanzas


def format_time(t: float) -> str:
    """Seconds → 'mm:ss.xx'."""
    cs = int(round(max(0.0, t) * 100))
    return f"{cs // 6000:02d}:{(cs % 6000) // 100:02d}.{cs % 100:02d}"


def format_lrc(lines: list[LrcLine], tags: dict[str, str] | None = None) -> str:
    """Inverse of :func:`parse_lrc` (+ :func:`parse_lrc_tags`). Tags come first, one line each."""
    head = "".join(f"[{k}:{v}]\n" for k, v in (tags or {}).items() if v is not None and v != "")
    body = "".join(f"[{format_time(ln.time)}]{ln.text}\n" for ln in lines)
    return head + body
