"""lyrics.final.json → ASS karaoke subtitles (rendered by libass inside ffmpeg).

Layout (classic two-row karaoke):

- Lines alternate between a top and a bottom row, so a line never moves when it becomes current.
  The row parity restarts on the top row after every instrumental gap.
- A line first appears *dimmed* (unsung colour, semi-transparent) while the line before it is
  sung, then becomes active ``lead_in_s`` before its first word. The active event starts with a
  ``\\k`` spacer, so the ``\\kf`` fill starts exactly on the first word.
- Each word gets a ``\\kf`` covering the word; gaps up to ``FOLD_GAP_S`` before the next word are
  folded into it (smooth fill), longer gaps get their own ``\\k`` spacer (the fill waits).
- Before a line that follows a gap longer than ``countdown_gap_s``: three dots that disappear one
  per beat (1 s) over the last 3 s.
- A title card (title / artist) during the intro when the first line starts after ``TITLE_MIN_INTRO_S``.

All times are rounded to centiseconds on an absolute grid, so the ``\\k`` sums are exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from karaoke.models import Line, LyricsFinal, Meta, RenderSettings

BASE_HEIGHT = 1080
FOLD_GAP_S = 0.35  # gaps up to this are folded into the previous word's fill
POST_HOLD_S = 0.4  # keep a finished line on screen this long
PREVIEW_AFTER_GAP_S = 3.5  # after a gap, the next line appears (dimmed) this long before it starts
COUNTDOWN_BEATS = 3
COUNTDOWN_BEAT_S = 1.0
TITLE_MIN_INTRO_S = 3.0
ROW_TOP_Y = 0.70  # centre of the top row, as a fraction of the height
ROW_SPACING = 1.6  # distance between the rows, in font sizes
MAX_TEXT_WIDTH = 0.92  # lines wider than this fraction of the width are shrunk
CHAR_WIDTH_EM = 0.58  # rough average glyph advance of a bold sans font, in font sizes

_HEX = re.compile(r"^#?([0-9a-fA-F]{6})([0-9a-fA-F]{2})?$")


# --------------------------------------------------------------------------- primitives


def hex_to_ass(color: str, alpha: int | None = None) -> str:
    """'#RRGGBB' (or '#RRGGBBAA', AA = CSS opacity) → ASS '&HAABBGGRR'.

    ASS alpha is inverted (00 = opaque, FF = transparent). ``alpha`` (ASS convention) overrides.
    """
    m = _HEX.match(color.strip())
    if not m:
        raise ValueError(f"not a hex colour: {color!r}")
    rgb = m.group(1)
    r, g, b = rgb[0:2], rgb[2:4], rgb[4:6]
    if alpha is None:
        alpha = 255 - int(m.group(2), 16) if m.group(2) else 0
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def ass_bgr(color: str) -> str:
    """'#RRGGBB' → '&HBBGGRR&' for \\1c-style override tags."""
    return "&H" + hex_to_ass(color)[4:] + "&"


def ass_time(cs: int) -> str:
    """Centiseconds → 'H:MM:SS.cc'."""
    cs = max(0, int(cs))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def cs(t: float) -> int:
    return int(round(t * 100))


def escape_text(text: str) -> str:
    """Make arbitrary text safe inside an ASS Dialogue: no override blocks, no \\N/\\h escapes."""
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("\\", "\\⁠")  # a word joiner after the backslash breaks \N, \n, \h
    return text.replace("{", "｛").replace("}", "｝")  # full-width braces


def karaoke_text(line: Line, start_cs: int) -> tuple[str, list[int]]:
    """The ``\\k``-tagged text of an active line whose Dialogue event starts at ``start_cs``.

    Returns (text, durations): durations are the per-syllable centiseconds after the lead-in
    spacer, so ``sum(durations)`` is the line duration (first word start → last word end).
    """
    words = line.words
    w0 = cs(words[0].start)
    lead = max(0, w0 - start_cs)
    parts = [f"{{\\k{lead}}}"]
    durs: list[int] = []
    cur = start_cs + lead
    for j, w in enumerate(words):
        txt = escape_text(w.text.strip())
        if j + 1 < len(words):
            nxt = words[j + 1].start
            if nxt - w.end <= FOLD_GAP_S:
                end = max(cur, cs(nxt))
                parts.append(f"{{\\kf{end - cur}}}{txt} ")
                durs.append(end - cur)
                cur = end
            else:
                end = max(cur, cs(w.end))
                parts.append(f"{{\\kf{end - cur}}}{txt}")
                durs.append(end - cur)
                cur = end
                gap_end = max(cur, cs(nxt))
                parts.append(f"{{\\k{gap_end - cur}}} ")
                durs.append(gap_end - cur)
                cur = gap_end
        else:
            end = max(cur, cs(w.end))
            parts.append(f"{{\\kf{end - cur}}}{txt}")
            durs.append(end - cur)
            cur = end
    return "".join(parts), durs


def plain_text(line: Line) -> str:
    return " ".join(escape_text(w.text.strip()) for w in line.words)


def _est_width(text: str, font_size: float) -> float:
    return len(text) * CHAR_WIDTH_EM * font_size


def _circle(r: int) -> str:
    """An ASS drawing (\\p1) of a filled circle of radius r, bounding box (0,0)-(2r,2r)."""
    k = round(r * 0.5523)
    c = r
    return (f"m {c} 0 b {c + k} 0 {2 * r} {c - k} {2 * r} {c} "
            f"b {2 * r} {c + k} {c + k} {2 * r} {c} {2 * r} "
            f"b {c - k} {2 * r} 0 {c + k} 0 {c} "
            f"b 0 {c - k} {c - k} 0 {c} 0")


# --------------------------------------------------------------------------- events


@dataclass
class Event:
    start: int  # cs
    end: int  # cs
    style: str
    text: str
    layer: int = 0
    name: str = ""  # kind of event: "line", "next", "countdown", "title"

    def __str__(self) -> str:
        return (f"Dialogue: {self.layer},{ass_time(self.start)},{ass_time(self.end)},{self.style},"
                f"{self.name},0,0,0,,{self.text}")


@dataclass
class _Sched:
    line: Line
    s: float  # first word start
    e: float  # last word end
    end: float  # line end (>= e)
    block_start: bool
    row: int
    act_start: float = 0.0
    act_end: float = 0.0


def _schedule(lines: list[Line], rs: RenderSettings, duration: float) -> list[_Sched]:
    out: list[_Sched] = []
    for i, ln in enumerate(lines):
        s, e = ln.words[0].start, max(ln.words[-1].end, ln.words[0].start)
        end = max(ln.end, e)
        block = i == 0 or s - out[-1].end > rs.countdown_gap_s
        row = 0 if block else 1 - out[-1].row
        out.append(_Sched(ln, s, e, end, block, row))
    for sc in out:
        sc.act_start = max(0.0, sc.s - rs.lead_in_s)
        sc.act_end = min(duration, sc.end + POST_HOLD_S) if duration > sc.end else sc.end
    row_free = [0.0, 0.0]
    last_end = 0.0
    for i, sc in enumerate(out):
        # the next line in this row: hand the row over at its lead-in (but never before we finish)
        j = i + 2
        if j < len(out) and not out[i + 1].block_start and not out[j].block_start:
            sc.act_end = min(sc.act_end, max(sc.end, out[j].act_start))
        free = max(row_free[sc.row], last_end if sc.block_start else 0.0)
        sc.act_start = min(max(sc.act_start, free), sc.s)
        row_free[sc.row] = sc.act_end
        last_end = max(last_end, sc.act_end)
    return out


def build_events(lyrics: LyricsFinal, meta: Meta, rs: RenderSettings, resolution: tuple[int, int]) -> list[Event]:
    W, H = resolution
    scale = H / BASE_HEIGHT
    fs = rs.font_size * scale
    row_y = [ROW_TOP_Y * H, ROW_TOP_Y * H + ROW_SPACING * fs]
    lines = sorted((ln for ln in lyrics.lines if ln.words), key=lambda ln: ln.words[0].start)
    duration = meta.duration
    sched = _schedule(lines, rs, duration)
    events: list[Event] = []
    dim = f"\\1c{ass_bgr(rs.color_unsung)}\\1a&H70&\\3a&HA0&\\4a&HD0&"

    for i, sc in enumerate(sched):
        text_plain = plain_text(sc.line)
        size_tag = ""
        w = _est_width(text_plain, fs)
        if w > MAX_TEXT_WIDTH * W:
            size_tag = f"\\fs{fs * MAX_TEXT_WIDTH * W / w:.1f}"
        pos = f"\\an5\\pos({W / 2:.0f},{row_y[sc.row]:.0f})"

        # dimmed "next line" phase
        if sc.block_start:
            prev_end = sched[i - 1].act_end if i else 0.0
            pv_start = max(prev_end, sc.s - max(rs.lead_in_s, PREVIEW_AFTER_GAP_S))
            fade_in = 300
        else:
            pv_start = sched[i - 1].act_start
            if i >= 2 and not sc.block_start and not sched[i - 1].block_start:
                pv_start = max(pv_start, sched[i - 2].act_end)
            fade_in = 150
        a0, a1 = cs(pv_start), cs(sc.act_start)
        if a1 - a0 >= 5:
            events.append(Event(a0, a1, "Karaoke", f"{{{pos}{size_tag}{dim}\\fad({min(fade_in, (a1 - a0) * 5)},0)}}"
                                f"{text_plain}", name="next"))
        else:
            a1 = a0 = cs(sc.act_start)

        # active phase
        start_cs, end_cs = a1, cs(sc.act_end)
        ktext, _ = karaoke_text(sc.line, start_cs)
        fade = "\\fad(0,150)" if a1 > a0 else "\\fad(150,150)"
        events.append(Event(start_cs, max(end_cs, start_cs + 1), "Karaoke",
                            f"{{{pos}{size_tag}{fade}}}{ktext}", layer=1, name="line"))

        # countdown before lines after a long gap (and before the first line after a long intro)
        gap = sc.s - (sched[i - 1].end if i else 0.0)
        if sc.block_start and gap > rs.countdown_gap_s:
            events += _countdown(sc, sched[i - 1].act_end if i else 0.0, W, row_y[0], fs, rs)

    if rs.show_title_card:
        events += _title_card(meta, sched, W, H, fs, rs)
    events.sort(key=lambda e: (e.start, e.layer))
    return events


def _countdown(sc: _Sched, free_from: float, W: int, top_y: float, fs: float, rs: RenderSettings) -> list[Event]:
    r = max(2, round(fs * 0.22))
    spacing = r * 3.2
    y = top_y - 1.15 * fs
    appear = max(free_from, sc.s - COUNTDOWN_BEATS * COUNTDOWN_BEAT_S - 0.5)
    out = []
    for k in range(COUNTDOWN_BEATS):
        gone = sc.s - (COUNTDOWN_BEATS - 1 - k) * COUNTDOWN_BEAT_S
        a, b = cs(appear), cs(gone)
        if b - a < 10:
            continue
        x = W / 2 + (k - (COUNTDOWN_BEATS - 1) / 2) * spacing
        out.append(Event(a, b, "Countdown",
                         f"{{\\an5\\pos({x:.0f},{y:.0f})\\fad(250,120)\\p1}}{_circle(r)}{{\\p0}}",
                         layer=2, name="countdown"))
    return out


def _title_card(meta: Meta, sched: list[_Sched], W: int, H: int, fs: float, rs: RenderSettings) -> list[Event]:
    if sched:
        if sched[0].s <= TITLE_MIN_INTRO_S:
            return []
        end = sched[0].act_start - 0.25
    else:
        end = min(meta.duration, 8.0)
    if end < 1.0:
        return []
    title = escape_text(meta.clean_title or meta.track or meta.title)
    artist = escape_text(meta.clean_artist or meta.artist or meta.uploader or "")
    tfs = fs * 1.5
    if _est_width(title, tfs) > MAX_TEXT_WIDTH * W:
        tfs = MAX_TEXT_WIDTH * W / (len(title) * CHAR_WIDTH_EM)
    afs = fs * 0.95
    if artist and _est_width(artist, afs) > MAX_TEXT_WIDTH * W:
        afs = MAX_TEXT_WIDTH * W / (len(artist) * CHAR_WIDTH_EM)
    fade_in, fade_out = min(600, int(end * 250)), min(500, int(end * 250))
    text = f"{{\\an5\\pos({W / 2:.0f},{H * 0.42:.0f})\\fad({fade_in},{fade_out})\\fs{tfs:.1f}}}{title}"
    if artist:
        text += f"\\N{{\\fs{afs * 0.5:.1f}}} \\N{{\\fs{afs:.1f}\\1c{ass_bgr(rs.color_sung)}}}{artist}"
    return [Event(0, cs(end), "Title", text, layer=0, name="title")]


# --------------------------------------------------------------------------- document


def _style(name: str, font: str, size: float, primary: str, secondary: str, outline: str, back: str,
           bold: bool, outline_w: float, shadow_w: float) -> str:
    return (f"Style: {name},{font},{size:.1f},{primary},{secondary},{outline},{back},"
            f"{-1 if bold else 0},0,0,0,100,100,0,0,1,{outline_w:.1f},{shadow_w:.1f},5,10,10,10,1")


def build_ass(lyrics: LyricsFinal, meta: Meta, rs: RenderSettings, resolution: tuple[int, int]) -> str:
    """The complete .ass document for this song at ``resolution`` (PlayRes = video size)."""
    W, H = resolution
    scale = H / BASE_HEIGHT
    fs = rs.font_size * scale
    bold = "bold" not in rs.font.lower()  # fonts like "Arial Rounded MT Bold" are already bold
    sung, unsung, outline = (hex_to_ass(rs.color_sung), hex_to_ass(rs.color_unsung),
                             hex_to_ass(rs.color_outline))
    shadow = hex_to_ass(rs.color_outline, alpha=0x80)
    styles = [
        # karaoke: Primary = after the fill (sung), Secondary = before the fill (unsung)
        _style("Karaoke", rs.font, fs, sung, unsung, outline, shadow, bold, 3.5 * scale, 1.5 * scale),
        _style("Countdown", rs.font, fs, sung, sung, outline, shadow, bold, 2.5 * scale, 1.0 * scale),
        _style("Title", rs.font, fs * 1.5, hex_to_ass(rs.color_unsung), unsung, outline, shadow, bold,
               4.0 * scale, 2.0 * scale),
    ]
    header = [
        "[Script Info]",
        "; generated by karaoke.render.ass",
        f"Title: {escape_text(meta.clean_title or meta.title)}",
        "ScriptType: v4.00+",
        f"PlayResX: {W}",
        f"PlayResY: {H}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        *styles,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events = build_events(lyrics, meta, rs, resolution)
    return "\n".join(header + [str(e) for e in events]) + "\n"
