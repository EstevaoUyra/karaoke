"""Pitch guide, piano-roll style (``render.pitch_style = "roll"``, the default): one row per
semitone, drawn in the ASS like ``render.staff``.

    ┌──────────────────────────────────────────────────────────────────────────┐
    │ D5 ─────────────┃──────────▬▬▬▬──────────────────────────────────────────│
    │        ▬▬▬ ▬▬▬▬▬┃▬▬▬                ▬▬▬▬▬▬                               │
    │ D4 ─────────────┃────────────────────────────────────────────────────────│
    │        Sweet love┃ of       mine    oh                                    │
    └──────────────────────────────────────────────────────────────────────────┘
      note names   playhead   upcoming notes and their words (scroll right → left)

Why not the staff by default: on a staff a scale step is half a space and a semitone (F / F♯) no
step at all, so a melody looks flat and its chromatic moves are only a small accidental. Here:

- **One row per semitone, and only the rows the song uses** (its range by sung time, padded; at
  least ``MIN_ROWS``), so the lane's whole height shows the melody's shape. Rows of the key's scale
  have a faint line; the tonic's rows a stronger one and the note name ("D4", sounding octave).
- **The voice's real pitch** (the raw contour, ``cache/melody.f0.npz``) is a thin line over each
  bar: scoops, slides and vibrato show, and a bar the voice does not hold (speech-like singing,
  rap) is drawn faint, so only the line and the rhythm are left.
- **Doubtful notes are faint too**: a note the CREPE tracker was unsure of (``UNSURE_CONF``) is
  right far less often than a confident one (docs/benchmarks.md), so it should not look the same.
- **Words under the notes**: each word scrolls along the bottom of the lane, starting under its
  first note, and fills at the playhead like the bars. (Squeezed to its time slot when the singing
  is fast.)
- A pitch held over several syllables is one bar with thin slits at the syllable starts.
- Bars fade in and out at the lane's edges, and the panel is light enough for a picture behind it.
"""

from __future__ import annotations

from dataclasses import dataclass

from karaoke.melody.theory import Key
from karaoke.models import LyricsFinal, Melody, RenderSettings
from karaoke.render.staff import Shape, staff_windows

PANEL_TOP = 0.03  # fractions of the video height / width
PANEL_HEIGHT = (0.2, 0.3)  # grows with the number of rows, so that a row is about ROW_H
ROW_H = 0.0098
PANEL_MARGIN_X = 0.04
PAD = 0.012  # inside the panel, above the top row and left/right
GUTTER = 0.036  # width of the note-name column (fraction of the height)
RAIL = 0.042  # height of the word rail at the bottom of the panel
WORD_FS = 0.027  # word font size
LABEL_FS = 0.02
MIN_ROWS = 12  # the lane never shows less than an octave: a two-note melody is not a mountain range
MAX_ROWS = 26
RANGE_TAIL = 0.01  # notes in the lowest / highest 1 % of sung time may fall off a full lane
BAR_FILL = 0.9  # bar thickness, in rows
BAR_MAX = 0.018  # ... but at most this fraction of the height
JOIN_S = 0.06  # same pitch and closer than this: one bar with a slit
SLIT = 0.0013  # half the slit's width
CURVE_W = 0.0022  # thickness of the pitch line
CURVE_STEP_S = 0.03
CURVE_RANGE = 3.0  # contour points further than this from the note (semitones) are tracker glitches
CURVE_COLOR = "#FFC94D"
STEADY_ST = 0.6  # a frame "holds" the note within this many semitones
UNSURE_CONF = 0.4  # CREPE notes below this confidence are drawn faint (measured: 37 % right vs 74 %)
STEADY_MIN = 0.5  # bars the voice holds for less than this share of their length are drawn faint
WORD_MIN_SCALE = 0.6  # words are squeezed to their time slot down to this
FADE_MS = 400
NOTE_FADE_MS = 220

PANEL_ALPHA = 0xA0  # ASS alpha (00 opaque, FF transparent)
ROW_ALPHA = 0xE2
TONIC_ALPHA = 0x98
NOTE_ALPHA = 0x10
FAINT_ALPHA = 0xA8
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


@dataclass(frozen=True)
class RollNote:
    start: float
    end: float
    midi: int
    join_left: bool = False  # continues the previous bar (same pitch, no rest)
    join_right: bool = False
    steady: float = 1.0  # share of the note the voice holds its pitch (1 without a contour)
    sure: bool = True  # False: the tracker was not confident of this note


@dataclass(frozen=True)
class RollLayout:
    W: int
    H: int
    panel: tuple[float, float, float, float]  # x0, y0, x1, y1
    lane: tuple[float, float]  # top, bottom of the rows
    rail_y: float  # centre of the word rail
    lo: int  # MIDI of the bottom row
    rows: int
    notes_left: float
    playhead_x: float
    notes_right: float
    px_per_s: float

    @property
    def row_h(self) -> float:
        return (self.lane[1] - self.lane[0]) / self.rows

    def y(self, midi: float) -> float:
        """Screen y of the centre of a (fractional) MIDI pitch's row."""
        return self.lane[1] - (midi - self.lo + 0.5) * self.row_h


def pitch_range(notes: list[tuple[int, float]]) -> tuple[int, int]:
    """(lowest, highest) MIDI row for (midi, seconds) notes: the whole range if it fits in
    ``MAX_ROWS``, else without the ``RANGE_TAIL`` extremes; padded by a row, at least ``MIN_ROWS``."""
    if not notes:
        return 60, 60 + MIN_ROWS - 1
    lo, hi = min(m for m, _ in notes) - 1, max(m for m, _ in notes) + 1
    if hi - lo + 1 > MAX_ROWS:
        order = sorted(notes)
        total = sum(d for _, d in order)
        acc, plo, phi = 0.0, order[0][0], order[-1][0]
        for m, d in order:
            if acc <= RANGE_TAIL * total:
                plo = m
            acc += d
            if acc >= (1 - RANGE_TAIL) * total:
                phi = m
                break
        lo, hi = plo - 1, phi + 1
        if hi - lo + 1 > MAX_ROWS:  # still too wide: keep the middle
            cut = hi - lo + 1 - MAX_ROWS
            lo, hi = lo + cut // 2, hi - (cut - cut // 2)
    while hi - lo + 1 < MIN_ROWS:
        if (hi - lo) % 2:
            lo -= 1
        else:
            hi += 1
    return lo, hi


def roll_layout(W: int, H: int, rs: RenderSettings, lo: int, hi: int) -> RollLayout:
    x0, x1 = PANEL_MARGIN_X * W, (1 - PANEL_MARGIN_X) * W
    y0 = PANEL_TOP * H
    pad = PAD * H
    rail = RAIL * H
    rows = hi - lo + 1
    y1 = y0 + min(PANEL_HEIGHT[1] * H, max(PANEL_HEIGHT[0] * H, rows * ROW_H * H + rail + 1.3 * pad))
    notes_left = x0 + pad + GUTTER * H
    notes_right = x1 - pad
    span = max(0.5, rs.pitch_trail_s + rs.pitch_lookahead_s)
    v = (notes_right - notes_left) / span
    return RollLayout(W, H, (x0, y0, x1, y1), (y0 + pad, y1 - rail - 0.3 * pad), y1 - rail / 2 - 0.15 * pad, lo,
                      rows, notes_left, notes_left + rs.pitch_trail_s * v, notes_right, v)


def steadiness(contour: tuple, start: float, end: float, midi: float) -> float:
    """Share of the contour's pitched frames in [start, end) within ``STEADY_ST`` of ``midi``."""
    import numpy as np

    pitch, hop_s = contour
    seg = pitch[max(0, int(round(start / hop_s))):max(0, int(round(end / hop_s)))]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 3:
        return 1.0
    return float(np.mean(np.abs(seg - midi) <= STEADY_ST))


def roll_notes(melody: Melody, lo: int, hi: int, min_conf: float = 0.0, contour: tuple | None = None) -> list[RollNote]:
    """The notes inside the lane, with their joins. ``contour`` is (fractional MIDI per frame with
    NaN where unpitched, hop seconds), on the recording's tuning (``melody.tuning`` is added)."""
    kept = [n for n in melody.notes if n.conf >= min_conf and n.end > n.start and lo <= n.midi <= hi]
    # only CREPE's confidence is calibrated for this; pyin's is low on held notes of a stem
    floor = UNSURE_CONF if str(melody.params.get("tracker", "")).startswith("crepe") else 0.0
    out = []
    for i, n in enumerate(kept):
        prev = kept[i - 1] if i else None
        nxt = kept[i + 1] if i + 1 < len(kept) else None
        steady = steadiness(contour, n.start, n.end, n.midi + melody.tuning) if contour is not None else 1.0
        out.append(RollNote(n.start, n.end, n.midi,
                            prev is not None and prev.midi == n.midi and n.start - prev.end < JOIN_S,
                            nxt is not None and nxt.midi == n.midi and nxt.start - n.end < JOIN_S, steady,
                            n.conf >= floor))
    return out


def _bar(x0: float, y0: float, x1: float, y1: float, round_left: bool, round_right: bool) -> Shape:
    """A bar with a round end where it is free and a square one where it joins its neighbour."""
    s = Shape()
    r = min((y1 - y0) / 2, (x1 - x0) / 2)
    if r <= 0.3 or not (round_left or round_right):
        return s.rect(x0, y0, x1, y1)
    k = r * (1 - 0.5523)
    xy = s._xy
    path = [f"m {xy(x0 + (r if round_left else 0), y0)} l {xy(x1 - (r if round_right else 0), y0)}"]
    if round_right:
        path.append(f"b {xy(x1 - k, y0)} {xy(x1, y0 + k)} {xy(x1, y0 + r)} l {xy(x1, y1 - r)} "
                    f"b {xy(x1, y1 - k)} {xy(x1 - k, y1)} {xy(x1 - r, y1)}")
    else:
        path.append(f"l {xy(x1, y1)}")
    path.append(f"l {xy(x0 + (r if round_left else 0), y1)}")
    if round_left:
        path.append(f"b {xy(x0 + k, y1)} {xy(x0, y1 - k)} {xy(x0, y1 - r)} l {xy(x0, y0 + r)} "
                    f"b {xy(x0, y0 + k)} {xy(x0 + k, y0)} {xy(x0 + r, y0)}")
    else:
        path.append(f"l {xy(x0, y0)}")
    s.parts.append(" ".join(path))
    return s


def curve_points(contour: tuple, n: RollNote, tuning: float) -> list[tuple[float, float]]:
    """(seconds after the note's start, MIDI on the equal-tempered grid) along the note."""
    import numpy as np

    pitch, hop_s = contour
    step = max(1, int(round(CURVE_STEP_S / hop_s)))
    a, b = max(0, int(round(n.start / hop_s))), min(len(pitch), int(round(n.end / hop_s)) + 1)
    pts = []
    for i in range(a, b, step):
        block = pitch[i:min(b, i + step)]
        block = block[np.isfinite(block)]
        if not len(block):
            continue
        m = float(np.median(block)) - tuning
        if abs(m - n.midi) <= CURVE_RANGE:
            pts.append((min(n.end - n.start, max(0.0, (i + (step - 1) / 2) * hop_s - n.start)), m))
    return pts


def tonic_label(midi: int, key: Key) -> str:
    names = FLAT_NAMES if key.fifths < 0 else NAMES
    return f"{names[midi % 12]}{midi // 12 - 1}"


def pitch_events(melody: Melody, rs: RenderSettings, resolution: tuple[int, int], duration: float,
                 lyrics: LyricsFinal | None = None, contour: tuple | None = None) -> list:
    """All pitch-guide events (``karaoke.render.ass.Event``) of the piano-roll guide."""
    from karaoke.render.ass import Event, ass_bgr, cs, escape_text
    from karaoke.render.layout import text_measurer

    W, H = resolution
    k = melody.key
    key = Key(k.tonic_pc, k.mode) if k is not None else None
    usable = [(n.midi, n.end - n.start) for n in melody.notes if n.conf >= rs.pitch_min_conf and n.end > n.start]
    if not usable:
        return []
    lo, hi = pitch_range(usable)
    notes = roll_notes(melody, lo, hi, rs.pitch_min_conf, contour)
    if not notes:
        return []
    L = roll_layout(W, H, rs, lo, hi)
    x0, y0, x1, y1 = L.panel
    v = L.px_per_s
    unsung, sung, outline = ass_bgr(rs.color_unsung), ass_bgr(rs.color_sung), ass_bgr(rs.color_outline)
    lane_clip = f"\\clip({L.notes_left:.0f},{y0:.0f},{L.notes_right:.0f},{y1:.0f})"
    sung_clip = f"\\clip({L.notes_left:.0f},{y0:.0f},{L.playhead_x:.0f},{y1:.0f})"
    bord = f"\\bord{max(1.0, 0.0016 * H):.1f}\\3c{outline}\\3a&H40&"
    p = f"\\p{Shape.P}"
    ev: list = []

    words = [w for ln in lyrics.lines for w in ln.words if w.text.strip()] if lyrics is not None else []
    words.sort(key=lambda w: w.start)

    def static(a: int, b: int, layer: int, name: str, tags: str, body: str, pos: str = "\\an7\\pos(0,0)") -> None:
        ev.append(Event(a, b, "Pitch", f"{{{pos}\\fad({FADE_MS},{FADE_MS}){tags}}}{body}", layer=layer, name=name))

    in_key = set(key.scale_pcs()) if key is not None else set()
    line_w = max(1.0, 0.0014 * H)
    for ws, we in staff_windows([*notes, *words], rs, duration):
        a, b = cs(ws), cs(we)
        static(a, b, 10, "pitch-panel", f"\\1c{outline}\\1a&H{PANEL_ALPHA:02X}&{p}",
               f"{Shape().rounded_rect(x0, y0, x1, y1, 0.017 * H)}{{\\p0}}")
        rows, tonics = Shape(), Shape()
        for m in range(lo, hi + 1):
            if key is None or m % 12 not in in_key:
                continue
            tonic = m % 12 == key.tonic_pc
            y = L.y(m)
            (tonics if tonic else rows).rect(L.notes_left, y - line_w / 2, L.notes_right, y + line_w / 2)
            if tonic:
                ev.append(Event(a, b, "Pitch", f"{{\\an6\\pos({L.notes_left - 0.006 * H:.1f},{y:.1f})"
                                               f"\\fad({FADE_MS},{FADE_MS})\\fs{LABEL_FS * H:.1f}\\1c{unsung}"
                                               f"\\1a&H40&\\bord0}}{tonic_label(m, key)}",
                                layer=12, name="pitch-label"))
        if rows.parts:
            static(a, b, 11, "pitch-rows", f"\\1c{unsung}\\1a&H{ROW_ALPHA:02X}&{p}", f"{rows}{{\\p0}}")
        if tonics.parts:
            static(a, b, 11, "pitch-rows", f"\\1c{unsung}\\1a&H{TONIC_ALPHA:02X}&{p}", f"{tonics}{{\\p0}}")
        head = Shape()
        hw = 0.003 * H
        head.rect(L.playhead_x - hw / 2, y0 + 0.008 * H, L.playhead_x + hw / 2, y1 - 0.008 * H)
        head.polygon([(L.playhead_x - 0.011 * H, y0 + 0.006 * H), (L.playhead_x + 0.011 * H, y0 + 0.006 * H),
                      (L.playhead_x, y0 + 0.02 * H)])
        static(a, b, 16, "pitch-playhead", f"\\1c{sung}{bord}{p}", f"{head}{{\\p0}}")

    def moving(start: float, lead_px: float, tail_px: float, end: float) -> tuple[int, int, float, float] | None:
        """(a, b, xa, xb): on screen from when the thing enters on the right until it has left."""
        t_in = max(0.0, start - (L.notes_right - L.playhead_x + lead_px) / v)
        t_out = min(max(duration, end), end + (L.playhead_x - L.notes_left + tail_px) / v)
        a, b = cs(t_in), cs(t_out)
        if b <= a:
            return None
        return a, b, L.playhead_x + (start - a / 100) * v, L.playhead_x + (start - b / 100) * v

    bh = min(BAR_FILL * L.row_h, BAR_MAX * H)
    slit = SLIT * H
    fade = f"\\fad({NOTE_FADE_MS},{NOTE_FADE_MS})"
    for n in notes:
        m = moving(n.start, 0.0, 0.0, n.end)
        if m is None:
            continue
        a, b, xa, xb = m
        move = f"\\an7\\move({xa:.1f},0,{xb:.1f},0,0,{(b - a) * 10}){fade}"
        left = slit if n.join_left else 0.0
        right = (n.end - n.start) * v - (slit if n.join_right else 0.0)
        if not n.join_right:
            right = max(right, left + bh)
        y = L.y(n.midi)
        bar = _bar(left, y - bh / 2, max(right, left + 1.0), y + bh / 2, not n.join_left, not n.join_right)
        faint = n.steady < STEADY_MIN or not n.sure
        alpha = FAINT_ALPHA if faint else NOTE_ALPHA
        ev.append(Event(a, b, "Pitch", f"{{{move}{lane_clip}\\1c{unsung}\\1a&H{alpha:02X}&{bord}{p}}}{bar}{{\\p0}}",
                        layer=13, name="pitch-faint" if faint else "pitch-note"))
        ev.append(Event(a, b, "Pitch", f"{{{move}{sung_clip}\\1c{sung}\\1a&H{alpha:02X}&{bord}{p}}}{bar}{{\\p0}}",
                        layer=14, name="pitch-sung"))
        if contour is not None:
            pts = curve_points(contour, n, melody.tuning)
            if len(pts) >= 2:
                line = Shape().stroke([(t * v, L.y(mm)) for t, mm in pts], CURVE_W * H)
                ev.append(Event(a, b, "Pitch", f"{{{move}{lane_clip}\\1c{ass_bgr(CURVE_COLOR)}\\1a&H18&\\bord"
                                               f"{max(0.8, 0.001 * H):.1f}\\3c{outline}\\3a&H60&{p}}}{line}{{\\p0}}",
                                layer=15, name="pitch-curve"))

    if words:
        fs = WORD_FS * H
        width = text_measurer(rs.font, fs)
        gap = 0.3 * fs
        wbord = f"\\bord{max(1.0, 0.0016 * H):.1f}\\3c{outline}\\3a&H30&"
        for i, w in enumerate(words):
            text = w.text.strip()
            wpx = max(1.0, width(text))
            scale = 1.0
            if i + 1 < len(words):
                slot = (words[i + 1].start - w.start) * v - gap
                scale = max(WORD_MIN_SCALE, min(1.0, slot / wpx))
            m = moving(w.start, 0.0, wpx * scale, w.start)
            if m is None:
                continue
            a, b, xa, xb = m
            tags = (f"\\an4\\move({xa:.1f},{L.rail_y:.1f},{xb:.1f},{L.rail_y:.1f},0,{(b - a) * 10}){fade}"
                    f"\\fs{fs:.1f}\\fscx{scale * 100:.0f}{wbord}")
            body = escape_text(text)
            ev.append(Event(a, b, "Pitch", f"{{{tags}{lane_clip}\\1c{unsung}\\1a&H{NOTE_ALPHA:02X}&}}{body}",
                            layer=13, name="pitch-word"))
            ev.append(Event(a, b, "Pitch", f"{{{tags}{sung_clip}\\1c{sung}}}{body}", layer=14, name="pitch-word-sung"))
    return ev
