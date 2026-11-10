"""lyrics.final.json → ASS karaoke subtitles (rendered by libass inside ffmpeg).

Layout (classic multi-row karaoke):

- The font size never changes. A lyric line too wide for a row is split into display lines at its
  best break (``render.layout``, ``karaoke.linebreak``).
- Lines are grouped in *blocks* separated by gaps longer than ``countdown_gap_s``. Two rows are
  used, or three where the lines come faster than ``FAST_LINE_S`` apart (so there is always time
  to read ahead). Lines cycle through the rows top to bottom, so a line never moves when it
  becomes current; a row is handed to the next line as soon as its line is sung, and at most
  that many lines are on screen at once.
- A line first appears *dimmed* (unsung colour, semi-transparent), then becomes active
  ``lead_in_s`` before its first word. The active event starts with a ``\\k`` spacer, so the
  ``\\kf`` fill starts exactly on the first word.
- Each word gets a ``\\kf`` covering the word; gaps up to ``FOLD_GAP_S`` before the next word are
  folded into it (smooth fill), longer gaps get their own ``\\k`` spacer (the fill waits).
- Every pause of at least ``PAUSE_DOTS_MIN_S`` (the intro, between lines, or inside a line) gets
  three dots above the rows that go out one per beat; the last one goes out as the singing resumes.
  The beat comes from the song's tempo when the caller knows it.
- A title card (title / artist) during the intro when the first line starts after ``TITLE_MIN_INTRO_S``.

All times are rounded to centiseconds on an absolute grid, so the ``\\k`` sums are exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from karaoke.models import Line, LyricsFinal, Meta, RenderSettings
from karaoke.render.layout import display_lines, text_measurer

BASE_HEIGHT = 1080
FOLD_GAP_S = 0.35  # gaps up to this are folded into the previous word's fill
POST_HOLD_S = 0.4  # keep a finished line on screen this long
PREVIEW_AFTER_GAP_S = 3.5  # after a gap, the next line appears (dimmed) this long before it starts
COUNTDOWN_BEATS = 3
COUNTDOWN_BEAT_S = 1.0  # dot beat when the song's tempo is unknown
PAUSE_DOTS_MIN_S = 2.0  # pauses at least this long get the three dots
DOTS_AFTER_S = 0.3  # the dots never appear sooner than this after the voice stops
MIN_DOT_BEAT_S = 0.35  # pauses too short for dots this far apart get none
FAST_LINE_S = 2.0  # where the median line-to-line time is below this, three rows are used
FAST_WINDOW = 5  # lines in that median
MIN_RUN = 4  # fewest lines in a stretch of 2 or 3 rows
OUTLINE = 3.5  # text outline at 1080p
TITLE_MIN_INTRO_S = 3.0
ROW_TOP_Y = 0.70  # centre of the top row, as a fraction of the height
ROW_SPACING = 1.6  # distance between the rows, in font sizes
MAX_TEXT_WIDTH = 0.90  # lines wider than this fraction of the width are split (titles shrink)
SOFT_TEXT_WIDTH = 0.62  # ... and lines wider than this, if they have a strong break
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
    block: int  # index of the block (lines between gaps longer than countdown_gap_s)
    k: int  # position in the block
    rows: int = 2  # rows of the layout this line is shown in (2, or 3 in fast parts)
    row: int = 0
    pv_start: float = 0.0  # the line appears (dimmed)
    act_start: float = 0.0  # the line becomes active (its \\k lead-in starts)
    act_end: float = 0.0


def line_rows(starts: list[float]) -> list[int]:
    """Rows per line of a block: 2, or 3 where the lines come faster than ``FAST_LINE_S`` apart
    (the median over a window of ``FAST_WINDOW`` lines). Runs shorter than ``MIN_RUN`` lines take
    the previous run's value, so the layout does not flicker."""
    n = len(starts)
    if n < MIN_RUN:
        return [2] * n
    pitch = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    pitch.append(pitch[-1])
    half = FAST_WINDOW // 2
    rows = []
    for i in range(n):
        win = sorted(pitch[max(0, i - half):i + half + 1])
        rows.append(3 if win[len(win) // 2] < FAST_LINE_S else 2)
    i = 0
    while i < n:
        j = i
        while j < n and rows[j] == rows[i]:
            j += 1
        if j - i < MIN_RUN and (i > 0 or j < n):
            fill = rows[i - 1] if i > 0 else rows[j]
            rows[i:j] = [fill] * (j - i)
        i = j
    return rows


def next_row(prev_row: int | None, prev_rows: int, rows: int) -> int:
    """The row after ``prev_row`` (of a ``prev_rows`` layout) in a ``rows`` layout: the next one
    down on screen, or the top one. Layouts share their bottom rows (see ``row_ys``)."""
    if prev_row is None:
        return 0
    slot = prev_row + (3 - prev_rows)  # physical slot, 0 = highest of three
    for r in range(rows):
        if r + (3 - rows) > slot:
            return r
    return 0


def _schedule(lines: list[Line], rs: RenderSettings, duration: float) -> list[_Sched]:
    out: list[_Sched] = []
    for ln in lines:
        s, e = ln.words[0].start, max(ln.words[-1].end, ln.words[0].start)
        new_block = not out or s - out[-1].end > rs.countdown_gap_s
        block = out[-1].block + 1 if out and new_block else (out[-1].block if out else 0)
        k = 0 if new_block else out[-1].k + 1
        out.append(_Sched(ln, s, e, max(ln.end, e), block, k))
    blocks: dict[int, list[_Sched]] = {}
    for sc in out:
        blocks.setdefault(sc.block, []).append(sc)
    prev_block_end = 0.0
    for b in sorted(blocks):
        bl = blocks[b]
        prev: _Sched | None = None
        for sc, n in zip(bl, line_rows([sc.s for sc in bl]), strict=True):
            sc.rows = n
            sc.row = next_row(prev.row if prev else None, prev.rows if prev else n, n)
            prev = sc
        slot = [sc.row + 3 - sc.rows for sc in bl]
        for k, sc in enumerate(bl):
            sc.act_end = min(duration, sc.end + POST_HOLD_S) if duration > sc.end else sc.end
            nxt = next((bl[j] for j in range(k + 1, len(bl)) if slot[j] == slot[k]), None)
            if nxt is not None:  # hand the row over in time for the line that reuses it
                sc.act_end = min(sc.act_end, max(sc.end, nxt.s - rs.lead_in_s), nxt.s - 0.05)
        appear = max(prev_block_end, bl[0].s - max(rs.lead_in_s, PREVIEW_AFTER_GAP_S), 0.0)
        free = {sl: appear for sl in range(3)}
        for k, sc in enumerate(bl):
            # at most ``rows`` lines on screen: a line shows up when the one rows-1 before it is sung
            ahead = k - (sc.rows - 1)
            shown = bl[ahead].act_start if ahead > 0 else appear
            start = max(free[slot[k]], shown)
            sc.pv_start = min(start, sc.s)
            sc.act_start = min(max(sc.s - rs.lead_in_s, start), sc.s)
            free[slot[k]] = sc.act_end
        prev_block_end = max(sc.act_end for sc in bl)
    return out


def row_ys(n: int, H: int, fs: float) -> list[float]:
    """Row centres for an n-row block; the bottom row stays put, extra rows stack upwards."""
    bottom = ROW_TOP_Y * H + ROW_SPACING * fs
    return [bottom - (n - 1 - r) * ROW_SPACING * fs for r in range(n)]


def pauses(sched: list[_Sched], min_gap: float) -> list[tuple[float, float, _Sched]]:
    """(silence start, next word start, line of that word) for every pause of at least
    ``min_gap``: the intro, gaps between lines and gaps inside a line."""
    out = []
    prev_end = 0.0
    for sc in sched:
        for j, w in enumerate(sc.line.words):
            if j == 0 or w.start - sc.line.words[j - 1].end >= min_gap:
                last = prev_end if j == 0 else sc.line.words[j - 1].end
                if w.start - last >= min_gap:
                    out.append((last, w.start, sc))
        prev_end = max(prev_end, sc.e)
    return out


def build_events(lyrics: LyricsFinal, meta: Meta, rs: RenderSettings, resolution: tuple[int, int],
                 beat_s: float | None = None) -> list[Event]:
    W, H = resolution
    scale = H / BASE_HEIGHT
    fs = rs.font_size * scale
    width = text_measurer(rs.font, fs, OUTLINE * scale)
    raw = sorted((ln for ln in lyrics.lines if ln.words), key=lambda ln: ln.words[0].start)
    lines = display_lines(raw, width, MAX_TEXT_WIDTH * W, SOFT_TEXT_WIDTH * W)
    duration = meta.duration
    sched = _schedule(lines, rs, duration)
    events: list[Event] = []
    dim = f"\\1c{ass_bgr(rs.color_unsung)}\\1a&H70&\\3a&HA0&\\4a&HD0&"

    for sc in sched:
        text_plain = plain_text(sc.line)
        y = row_ys(sc.rows, H, fs)[sc.row]
        pos = f"\\an5\\pos({W / 2:.0f},{y:.0f})"

        # dimmed "next line" phase
        a0, a1 = cs(sc.pv_start), cs(sc.act_start)
        fade_in = 300 if sc.k == 0 else 150
        if a1 - a0 >= 5:
            events.append(Event(a0, a1, "Karaoke", f"{{{pos}{dim}\\fad({min(fade_in, (a1 - a0) * 5)},0)}}"
                                f"{text_plain}", name="next"))
        else:
            a0 = a1

        # active phase
        start_cs, end_cs = a1, cs(sc.act_end)
        ktext, _ = karaoke_text(sc.line, start_cs)
        fade = "\\fad(0,150)" if a1 > a0 else "\\fad(150,150)"
        events.append(Event(start_cs, max(end_cs, start_cs + 1), "Karaoke",
                            f"{{{pos}{fade}}}{ktext}", layer=1, name="line"))

    for silence, nxt, sc in pauses(sched, PAUSE_DOTS_MIN_S):
        top = row_ys(sc.rows, H, fs)[0]
        events += _countdown(silence, nxt, W, top, fs, beat_s or COUNTDOWN_BEAT_S)

    if rs.show_title_card:
        events += _title_card(meta, sched, W, H, fs, rs)
    events.sort(key=lambda e: (e.start, e.layer))
    return events


def _countdown(silence: float, start: float, W: int, top_y: float, fs: float, beat: float) -> list[Event]:
    """Three dots over the row block: they appear ``COUNTDOWN_BEATS`` beats before ``start`` and
    go out one per beat, the last one exactly when the singing comes back."""
    room = start - silence - DOTS_AFTER_S
    beat = min(beat, room / COUNTDOWN_BEATS)
    if beat < MIN_DOT_BEAT_S:
        return []
    r = max(2, round(fs * 0.22))
    spacing = r * 3.2
    y = top_y - 1.15 * fs
    appear = start - COUNTDOWN_BEATS * beat
    fade_in = int(min(250, (appear - silence) * 1000))
    appear -= fade_in / 1000
    out = []
    for k in range(COUNTDOWN_BEATS):
        gone = start - (COUNTDOWN_BEATS - 1 - k) * beat
        a, b = cs(appear), cs(gone)
        x = W / 2 + (k - (COUNTDOWN_BEATS - 1) / 2) * spacing
        out.append(Event(a, b, "Countdown",
                         f"{{\\an5\\pos({x:.0f},{y:.0f})\\fad({fade_in},120)\\p1}}{_circle(r)}{{\\p0}}",
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


def build_ass(lyrics: LyricsFinal, meta: Meta, rs: RenderSettings, resolution: tuple[int, int],
              beat_s: float | None = None) -> str:
    """The complete .ass document for this song at ``resolution`` (PlayRes = video size).

    ``beat_s`` (seconds per beat, see ``render.tempo``) paces the countdown dots."""
    W, H = resolution
    scale = H / BASE_HEIGHT
    fs = rs.font_size * scale
    bold = "bold" not in rs.font.lower()  # fonts like "Arial Rounded MT Bold" are already bold
    sung, unsung, outline = (hex_to_ass(rs.color_sung), hex_to_ass(rs.color_unsung),
                             hex_to_ass(rs.color_outline))
    shadow = hex_to_ass(rs.color_outline, alpha=0x80)
    styles = [
        # karaoke: Primary = after the fill (sung), Secondary = before the fill (unsung)
        _style("Karaoke", rs.font, fs, sung, unsung, outline, shadow, bold, OUTLINE * scale, 1.5 * scale),
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
    events = build_events(lyrics, meta, rs, resolution, beat_s)
    return "\n".join(header + [str(e) for e in events]) + "\n"
