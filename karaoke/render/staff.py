"""Pitch guide: the sung melody (``melody.json``) as bars on a treble staff, drawn in the ASS.

Layout (top of the screen, above the title card and far above the lyric rows):

    ┌──────────────────────────────────────────────────────────────────────────┐
    │ 𝄞 ♯♯ │ ▬▬ ▬  ┃   ▬▬▬▬    ▬▬  ▬                                          │
    └──────────────────────────────────────────────────────────────────────────┘
      clef, key  past  playhead  upcoming notes (scroll right → left)

- **Scrolling, not paged.** Notes scroll from right to left past a fixed playhead
  (``pitch_trail_s`` of past, ``pitch_lookahead_s`` of future), at a constant speed, so a bar's
  length is always its duration on one time scale, the eye stays on one spot and the next phrase
  is always visible. A paged display would make the playhead jump back at every page turn, and
  the lyrics below already page (row by row).
- A real five-line treble staff with the key signature of the detected key. Each bar sits on the
  line or space of its diatonic pitch (``melody.theory.spell``); notes outside the key get a small
  ♯/♭/♮ just before the bar; ledger lines for notes beyond the staff.
- **Octave shift**: if the melody's median is far from the staff (``Melody.display_octave_shift``),
  all notes are written one octave higher (low voices: "8vb" under the clef, as on a tenor clef)
  or lower ("8va" above it).
- **Highlight**: each note is drawn twice with the same ``\\move``: in the unsung colour, clipped to
  the note area, and in the sung colour, clipped to the left of the playhead. As a bar crosses the
  playhead it fills left to right, exactly like the lyric ``\\kf`` fill.
- Everything is libass vector drawing (``\\p``) in the same ASS as the lyrics, so it is still one
  ffmpeg pass. Only the clef is a font glyph (U+1D11E, e.g. "Apple Symbols", found with
  fontconfig), placed from the font's own metrics; without such a font a vector clef is drawn.
- The staff only shows while notes are near (gaps longer than ``STAFF_GAP_S`` hide it).
"""

from __future__ import annotations

import functools
import math
import shutil
import struct
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from karaoke.melody.theory import FLAT_POSITIONS, SHARP_POSITIONS, Key, ledger_positions, spell
from karaoke.models import Melody, RenderSettings

CLEF_CHAR = "\U0001d11e"
CLEF_FONTS = ("Apple Symbols", "Bravura", "Noto Music", "FreeSerif", "Segoe UI Symbol", "DejaVu Sans")

SPACE = 0.021  # staff space, as a fraction of the video height
PANEL_TOP = 0.035  # top of the panel, as a fraction of the height
PANEL_MARGIN_X = 0.04  # panel inset from the left/right edges, as a fraction of the width
PAD_SPACES = 3.5  # panel extends this many spaces above the top line and below the bottom line
POS_MIN, POS_MAX = -6, 14  # staff positions that fit in the panel (3 ledger lines either side);
#                           notes beyond them are left out rather than drawn at a wrong height
BAR_HEIGHT = 0.72  # note bar thickness, in spaces (thinner than a space: lines stay visible)
BAR_GAP = 0.10  # each bar is shortened by this many spaces at both ends (repeated notes separate)
LINE_W = 0.11  # staff line thickness, in spaces
LEDGER_EXT = 0.45  # ledger lines stick out this far on each side of the bar, in spaces
CLEF_X, CLEF_HEIGHT = 0.6, 7.0  # clef left edge and height, in spaces
CLEF_BELOW_G = 2.6  # the clef reaches this many spaces below its G line
KEY_GAP, KEY_STEP = 0.5, 1.05  # key signature: gap after the clef, advance per accidental
NOTES_GAP = 0.9  # gap between the key signature and the note area
ACC_SCALE = 0.85  # accidentals before notes are drawn at this size (key signature: 1)
ACC_OFFSET = 0.8  # ... centred this many spaces left of the bar
STAFF_GAP_S = 6.0  # hide the staff during longer gaps between notes
FADE_MS = 400

PANEL_ALPHA = 0x80  # ASS alpha (00 opaque, FF transparent)
STAFF_ALPHA = 0x50
NOTE_ALPHA = 0x10


# --------------------------------------------------------------------------- vector shapes


class Shape:
    """An ASS drawing (``\\p3``: coordinates in quarter pixels) built from float coordinates."""

    P = 3
    K = 2 ** (P - 1)

    def __init__(self) -> None:
        self.parts: list[str] = []

    def _xy(self, x: float, y: float) -> str:
        return f"{round(x * self.K)} {round(y * self.K)}"

    def polygon(self, pts: list[tuple[float, float]]) -> Shape:
        if len(pts) >= 3:
            self.parts.append("m " + self._xy(*pts[0]) + " l " + " ".join(self._xy(*p) for p in pts[1:]))
        return self

    def rect(self, x0: float, y0: float, x1: float, y1: float) -> Shape:
        return self.polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    def rounded_rect(self, x0: float, y0: float, x1: float, y1: float, r: float) -> Shape:
        r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
        if r <= 0.01:
            return self.rect(x0, y0, x1, y1)
        k = r * (1 - 0.5523)
        xy = self._xy
        self.parts.append(
            f"m {xy(x0 + r, y0)} l {xy(x1 - r, y0)} b {xy(x1 - k, y0)} {xy(x1, y0 + k)} {xy(x1, y0 + r)} "
            f"l {xy(x1, y1 - r)} b {xy(x1, y1 - k)} {xy(x1 - k, y1)} {xy(x1 - r, y1)} "
            f"l {xy(x0 + r, y1)} b {xy(x0 + k, y1)} {xy(x0, y1 - k)} {xy(x0, y1 - r)} "
            f"l {xy(x0, y0 + r)} b {xy(x0, y0 + k)} {xy(x0 + k, y0)} {xy(x0 + r, y0)}")
        return self

    def stroke(self, pts: list[tuple[float, float]], widths: list[float] | float) -> Shape:
        """A thick polyline as one polygon (left edge forward, right edge back)."""
        n = len(pts)
        if n < 2:
            return self
        ws = [widths] * n if isinstance(widths, (int, float)) else list(widths)
        left, right = [], []
        for i in range(n):
            a, b = pts[max(0, i - 1)], pts[min(n - 1, i + 1)]
            dx, dy = b[0] - a[0], b[1] - a[1]
            d = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / d * ws[i] / 2, dx / d * ws[i] / 2
            left.append((pts[i][0] + nx, pts[i][1] + ny))
            right.append((pts[i][0] - nx, pts[i][1] - ny))
        return self.polygon(left + right[::-1])

    def __str__(self) -> str:
        return " ".join(self.parts)


def _bezier(p0, p1, p2, p3, n: int = 16) -> list[tuple[float, float]]:
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                    u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))
    return out


def accidental(shape: Shape, kind: str, cx: float, cy: float, sp: float, scale: float = 1.0) -> Shape:
    """Draw a ♯/♭/♮ centred on (cx, cy) (the note's line or space) into ``shape``.
    Designed in staff spaces with y up; ``sp`` is the staff space in pixels."""
    u = sp * scale

    def P(x: float, y: float) -> tuple[float, float]:
        return (cx + x * u, cy - y * u)

    thin, thick = 0.13 / scale, 0.26  # thin strokes keep their pixel width when scaled down
    if kind == "sharp":
        shape.stroke([P(-0.22, -1.25), P(-0.22, 1.15)], thin * u)
        shape.stroke([P(0.22, -1.15), P(0.22, 1.25)], thin * u)
        for y0 in (-0.42, 0.42):
            shape.stroke([P(-0.48, y0 - 0.13), P(0.48, y0 + 0.13)], thick * u)
    elif kind == "natural":
        shape.stroke([P(-0.22, -0.5), P(-0.22, 1.35)], thin * u)
        shape.stroke([P(0.22, -1.35), P(0.22, 0.5)], thin * u)
        for y0 in (-0.3, 0.3):
            shape.stroke([P(-0.22, y0 - 0.1), P(0.22, y0 + 0.1)], thick * u)
    elif kind == "flat":
        shape.stroke([P(-0.32, -0.42), P(-0.32, 1.85)], thin * u)
        bowl = _bezier((-0.32, -0.42), (0.75, 0.15), (0.6, 0.95), (-0.32, 0.38))
        n = len(bowl) - 1
        widths = [(0.06 + 0.2 * math.sin(math.pi * i / n) ** 1.5) * u for i in range(n + 1)]
        shape.stroke([P(x, y) for x, y in bowl], widths)
    else:
        raise ValueError(kind)
    return shape


def vector_clef(shape: Shape, x: float, g_y: float, sp: float) -> Shape:
    """A treble clef drawn as strokes (used when no font has U+1D11E). ``g_y`` is the G line."""
    def P(px: float, py: float) -> tuple[float, float]:
        return (x + (px + 1.2) * sp, g_y - py * sp)

    spine = (_bezier((0.05, -0.05), (-0.75, 0.15), (-0.55, -1.25), (0.25, -1.15), 8)
             + _bezier((0.25, -1.15), (1.25, -1.0), (1.2, 0.85), (0.15, 0.95), 8)[1:]
             + _bezier((0.15, 0.95), (-1.0, 1.1), (-1.05, 2.0), (0.35, 3.15), 10)[1:]
             + _bezier((0.35, 3.15), (0.95, 3.7), (0.85, 4.6), (0.45, 4.4), 8)[1:]
             + _bezier((0.45, 4.4), (-0.25, 4.1), (0.0, 1.5), (0.35, -0.6), 10)[1:]
             + _bezier((0.35, -0.6), (0.55, -1.8), (0.45, -2.5), (-0.2, -2.45), 8)[1:])
    n = len(spine)
    widths = [sp * (0.16 + 0.2 * math.sin(math.pi * min(1.0, i / (n * 0.45))) ** 2) for i in range(n)]
    shape.stroke([P(a, b) for a, b in spine], widths)
    cx, cy = P(-0.3, -2.25)
    r = 0.33 * sp
    return shape.rounded_rect(cx - r, cy - r, cx + r, cy + r, r)


# --------------------------------------------------------------------------- clef glyph


@dataclass(frozen=True)
class ClefGlyph:
    family: str
    file: Path
    height_em: float  # glyph height in ems
    top_em: float  # glyph top above the baseline, in ems
    left_em: float  # left side bearing, in ems
    width_em: float
    line_em: float  # libass font size per em: (winAscent + winDescent) / unitsPerEm
    ascent_em: float  # winAscent / unitsPerEm


def _win_metrics(path: Path) -> tuple[float, float] | None:
    """(winAscent, winDescent) / unitsPerEm of a TrueType/OpenType file (libass sizes by these)."""
    try:
        d = path.read_bytes()
        if d[:4] not in (b"\x00\x01\x00\x00", b"OTTO", b"true"):
            return None
        n = struct.unpack(">H", d[4:6])[0]
        tables = {}
        for i in range(n):
            tag, _, off, ln = struct.unpack(">4sIII", d[12 + 16 * i: 28 + 16 * i])
            tables[tag] = d[off: off + ln]
        upem = struct.unpack(">H", tables[b"head"][18:20])[0]
        asc, desc = struct.unpack(">HH", tables[b"OS/2"][74:78])
        return asc / upem, desc / upem
    except (OSError, KeyError, struct.error, IndexError):
        return None


@functools.lru_cache(maxsize=1)
def clef_glyph() -> ClefGlyph | None:
    """A font with U+1D11E that libass can find (via fontconfig), with its metrics, or None."""
    exe = shutil.which("fc-list")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, ":charset=1d11e", "family", "file"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    found: dict[str, Path] = {}
    for line in out.stdout.splitlines():
        if ":" not in line:
            continue
        file, _, fam = line.partition(":")
        fam = fam.strip().split(",")[0]
        if fam and not fam.startswith(".") and Path(file).suffix.lower() in (".ttf", ".otf"):
            found.setdefault(fam, Path(file))
    family = next((f for f in CLEF_FONTS if f in found), None)
    if family is None:
        return None
    win = _win_metrics(found[family])
    if win is None:
        return None
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(str(found[family]), 1000)
        x0, top, x1, bottom = font.getbbox(CLEF_CHAR, anchor="ls")
    except Exception:  # noqa: BLE001 — unreadable font
        return None
    if bottom - top <= 0:
        return None
    return ClefGlyph(family, found[family], (bottom - top) / 1000, -top / 1000, x0 / 1000, (x1 - x0) / 1000,
                     win[0] + win[1], win[0])


# --------------------------------------------------------------------------- layout


@dataclass(frozen=True)
class StaffLayout:
    W: int
    H: int
    sp: float  # staff space, px
    panel: tuple[float, float, float, float]  # x0, y0, x1, y1
    bottom_line_y: float
    clef_x: float
    key_x: float
    notes_left: float
    playhead_x: float
    notes_right: float
    px_per_s: float

    def y(self, pos: float) -> float:
        """Screen y of a staff position (0 = bottom line, 8 = top line)."""
        return self.bottom_line_y - pos * self.sp / 2

    @property
    def top_line_y(self) -> float:
        return self.y(8)


def staff_layout(W: int, H: int, rs: RenderSettings, n_key: int, clef_width_sp: float = 3.2) -> StaffLayout:
    sp = SPACE * H
    x0, x1 = PANEL_MARGIN_X * W, (1 - PANEL_MARGIN_X) * W
    y0 = PANEL_TOP * H
    top_line = y0 + PAD_SPACES * sp
    bottom_line = top_line + 4 * sp
    y1 = bottom_line + PAD_SPACES * sp
    clef_x = x0 + CLEF_X * sp
    key_x = clef_x + (clef_width_sp + KEY_GAP) * sp
    notes_left = key_x + (n_key * KEY_STEP + NOTES_GAP) * sp
    notes_right = x1 - 0.6 * sp
    span = max(0.5, rs.pitch_trail_s + rs.pitch_lookahead_s)
    v = (notes_right - notes_left) / span
    return StaffLayout(W, H, sp, (x0, y0, x1, y1), bottom_line, clef_x, key_x, notes_left,
                       notes_left + rs.pitch_trail_s * v, notes_right, v)


# --------------------------------------------------------------------------- events


@dataclass(frozen=True)
class GuideNote:
    start: float
    end: float
    pos: int  # staff position (POS_MIN..POS_MAX)
    accidental: str | None
    ledgers: list[int]
    written: int  # written MIDI (after the octave shift)
    end_trim: float = 0.0  # seconds from this note's start to the next note's, when that one has an accidental


def guide_notes(melody: Melody, min_conf: float = 0.0) -> tuple[Key, list[GuideNote]]:
    k = melody.key
    key = Key(k.tonic_pc, k.mode) if k is not None else Key(0, "major")
    out = []
    for n in melody.notes:
        if n.conf < min_conf or n.end <= n.start:
            continue
        written = n.midi + 12 * melody.display_octave_shift
        sp = spell(written, key)
        pos = sp.staff_position
        if not POS_MIN <= pos <= POS_MAX:
            continue  # off the panel even after the octave shift: likely a tracking error, not drawn
        out.append(GuideNote(n.start, n.end, pos, sp.accidental, ledger_positions(pos), written))
    for i, (a, b) in enumerate(zip(out, out[1:], strict=False)):
        if b.accidental and b.start > a.start:
            out[i] = replace(a, end_trim=b.start - a.start)
    return key, out


def staff_windows(notes: list[GuideNote], rs: RenderSettings, duration: float) -> list[tuple[float, float]]:
    """When the staff is on screen: around the notes, with gaps under ``STAFF_GAP_S`` bridged."""
    spans = sorted((max(0.0, n.start - rs.pitch_lookahead_s - 0.6), min(duration, n.end + rs.pitch_trail_s + 0.6))
                   for n in notes)
    out: list[list[float]] = []
    for s, e in spans:
        if out and s - out[-1][1] < STAFF_GAP_S:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if e > s]


def pitch_style(rs: RenderSettings, resolution: tuple[int, int]) -> str:
    """The ASS style line used by every pitch-guide event."""
    from karaoke.render.ass import hex_to_ass

    sp = SPACE * resolution[1]
    c = hex_to_ass(rs.color_unsung)
    o = hex_to_ass(rs.color_outline)
    return (f"Style: Pitch,{rs.font},{1.6 * sp:.1f},{c},{c},{o},{o},0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1")


def pitch_events(melody: Melody, rs: RenderSettings, resolution: tuple[int, int], duration: float) -> list:
    """All pitch-guide events (``karaoke.render.ass.Event``) for ``melody``."""
    from karaoke.render.ass import Event, ass_bgr, cs

    W, H = resolution
    key, notes = guide_notes(melody, rs.pitch_min_conf)
    if not notes:
        return []
    glyph = clef_glyph()
    clef_w = (glyph.width_em / glyph.height_em * CLEF_HEIGHT + 0.2) if glyph else 3.2
    L = staff_layout(W, H, rs, abs(key.fifths), clef_w)
    sp = L.sp
    x0, y0, x1, y1 = L.panel
    unsung, sung, outline = ass_bgr(rs.color_unsung), ass_bgr(rs.color_sung), ass_bgr(rs.color_outline)
    clip_all = f"\\clip({L.notes_left:.0f},{y0:.0f},{L.notes_right:.0f},{y1:.0f})"
    clip_sung = f"\\clip({L.notes_left:.0f},{y0:.0f},{L.playhead_x:.0f},{y1:.0f})"
    bord = f"\\bord{max(1.0, 0.09 * sp):.1f}\\3c{outline}\\3a&H40&"
    ev: list = []

    def static(a: int, b: int, layer: int, name: str, tags: str, body: str, pos: str = "\\pos(0,0)") -> None:
        ev.append(Event(a, b, "Pitch", f"{{\\an7{pos}\\fad({FADE_MS},{FADE_MS}){tags}}}{body}",
                        layer=layer, name=name))

    # the staff itself, per window
    for ws, we in staff_windows(notes, rs, duration):
        a, b = cs(ws), cs(we)
        panel = Shape().rounded_rect(x0, y0, x1, y1, 0.8 * sp)
        static(a, b, 10, "pitch-panel", f"\\1c{ass_bgr(rs.color_outline)}\\1a&H{PANEL_ALPHA:02X}&\\p{Shape.P}",
               f"{panel}{{\\p0}}")
        lines = Shape()
        lw = LINE_W * sp
        for i in range(5):
            y = L.y(2 * i)
            lines.rect(L.clef_x - 0.2 * sp, y - lw / 2, L.notes_right, y + lw / 2)
        static(a, b, 11, "pitch-staff", f"\\1c{unsung}\\1a&H{STAFF_ALPHA:02X}&\\p{Shape.P}", f"{lines}{{\\p0}}")
        g_y = L.y(2)
        if glyph:
            em = CLEF_HEIGHT * sp / glyph.height_em
            bottom = g_y + CLEF_BELOW_G * sp
            baseline = bottom - (glyph.height_em - glyph.top_em) * em
            static(a, b, 12, "pitch-clef",
                   f"\\fn{glyph.family}\\fs{em * glyph.line_em:.1f}\\b0\\1c{unsung}\\1a&H{NOTE_ALPHA:02X}&",
                   CLEF_CHAR, pos=f"\\pos({L.clef_x - glyph.left_em * em:.1f},{baseline - glyph.ascent_em * em:.1f})")
        else:
            static(a, b, 12, "pitch-clef", f"\\1c{unsung}\\1a&H{NOTE_ALPHA:02X}&\\p{Shape.P}",
                   f"{vector_clef(Shape(), L.clef_x, g_y, sp)}{{\\p0}}")
        if key.fifths:
            sig = Shape()
            kind, positions = ("sharp", SHARP_POSITIONS) if key.fifths > 0 else ("flat", FLAT_POSITIONS)
            for i, pos in enumerate(positions[: abs(key.fifths)]):
                accidental(sig, kind, L.key_x + (i + 0.5) * KEY_STEP * sp, L.y(pos), sp)
            static(a, b, 12, "pitch-key", f"\\1c{unsung}\\1a&H{NOTE_ALPHA:02X}&\\p{Shape.P}", f"{sig}{{\\p0}}")
        shift = melody.display_octave_shift
        if shift:
            label = ("8vb" if shift > 0 else "8va") if abs(shift) == 1 else ("15mb" if shift > 0 else "15ma")
            cx = L.clef_x + clef_w * sp / 2
            if shift > 0:
                yy, an = g_y + (CLEF_BELOW_G + 0.15) * sp, 8
            else:
                yy, an = g_y - (CLEF_HEIGHT - CLEF_BELOW_G + 0.1) * sp, 2
            ev.append(Event(a, b, "Pitch", f"{{\\an{an}\\pos({cx:.1f},{yy:.1f})\\fad({FADE_MS},{FADE_MS})\\i1"
                                           f"\\fs{1.5 * sp:.1f}\\1c{unsung}\\1a&H{NOTE_ALPHA:02X}&}}{label}",
                            layer=12, name="pitch-octave"))
        head = Shape()
        hw = 0.16 * sp
        head.rect(L.playhead_x - hw / 2, y0 + 0.5 * sp, L.playhead_x + hw / 2, y1 - 0.5 * sp)
        head.polygon([(L.playhead_x - 0.55 * sp, y0 + 0.35 * sp), (L.playhead_x + 0.55 * sp, y0 + 0.35 * sp),
                      (L.playhead_x, y0 + 1.0 * sp)])
        static(a, b, 16, "pitch-playhead", f"\\1c{sung}{bord}\\p{Shape.P}", f"{head}{{\\p0}}")

    # the notes: a bar (unsung), the same bar in the sung colour left of the playhead, and their
    # ledger lines and accidental; all moving together
    v = L.px_per_s
    bh = BAR_HEIGHT * sp
    p = f"\\p{Shape.P}"
    gx = BAR_GAP * sp
    lw = LINE_W * sp
    for n in notes:
        # on screen from when its accidental enters on the right until its ledger lines leave on the left
        t_in = max(0.0, n.start - (L.notes_right - L.playhead_x + (ACC_OFFSET + 1.0) * sp) / v)
        t_out = min(max(duration, n.end), n.end + (L.playhead_x - L.notes_left + LEDGER_EXT * sp) / v)
        a, b = cs(t_in), cs(t_out)
        if b <= a:
            continue
        xa = L.playhead_x + (n.start - a / 100) * v  # the bar's origin is at the playhead at n.start
        xb = L.playhead_x + (n.start - b / 100) * v
        move = f"\\an7\\move({xa:.1f},0,{xb:.1f},0,0,{(b - a) * 10})"
        length = max(bh, (n.end - n.start) * v - 2 * gx)
        if n.end_trim:  # leave room for the next note's accidental
            length = max(bh, min(length, n.end_trim * v - gx - (ACC_OFFSET + 0.55 * ACC_SCALE) * sp))
        y = L.y(n.pos)
        bar = Shape().rounded_rect(gx, y - bh / 2, gx + length, y + bh / 2, bh / 2)
        ev.append(Event(a, b, "Pitch", f"{{{move}{clip_all}\\1c{unsung}\\1a&H{NOTE_ALPHA:02X}&{bord}{p}}}{bar}{{\\p0}}",
                        layer=13, name="pitch-note"))
        ev.append(Event(a, b, "Pitch", f"{{{move}{clip_sung}\\1c{sung}{bord}{p}}}{bar}{{\\p0}}",
                        layer=14, name="pitch-sung"))
        if n.ledgers:
            deco = Shape()
            for lp in n.ledgers:
                ly = L.y(lp)
                deco.rect(gx - LEDGER_EXT * sp, ly - lw / 2, gx + length + LEDGER_EXT * sp, ly + lw / 2)
            ev.append(Event(a, b, "Pitch", f"{{{move}{clip_all}\\1c{unsung}\\1a&H{STAFF_ALPHA:02X}&{p}}}"
                                           f"{deco}{{\\p0}}", layer=12, name="pitch-ledger"))
        if n.accidental:
            acc = accidental(Shape(), n.accidental, gx - ACC_OFFSET * sp, y, sp, ACC_SCALE)
            ev.append(Event(a, b, "Pitch", f"{{{move}{clip_all}\\1c{unsung}{bord}{p}}}{acc}{{\\p0}}",
                            layer=15, name="pitch-accidental"))
    return ev
