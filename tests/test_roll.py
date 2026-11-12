"""Piano-roll pitch guide: rows, joins, the pitch line, faint bars and the word rail."""

from __future__ import annotations

import re

import make_demo
import numpy as np

from karaoke.models import KeyEstimate, Melody, Meta, Note, RenderSettings
from karaoke.render import roll
from karaoke.render.ass import build_ass

RES = (1280, 720)
MOVE = re.compile(r"\\move\(([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),0,(\d+)\)")
D_MAJOR = KeyEstimate(tonic="D", tonic_pc=2, mode="major", fifths=2, accidentals=["F#", "C#"], confidence=0.8)
RS = RenderSettings(show_pitch=True)


def melody(notes: list[tuple[float, float, int]], key: KeyEstimate | None = D_MAJOR, tuning: float = 0.0) -> Melody:
    return Melody(hop_s=0.01, notes=[Note(start=s, end=e, midi=m, pitch=float(m)) for s, e, m in notes],
                  key=key, tuning=tuning)


def events(mel: Melody, rs: RenderSettings = RS, duration: float = 24.0, **kw):
    return roll.pitch_events(mel, rs, RES, duration, **kw)


def named(ev, name: str):
    return [e for e in ev if e.name == name]


def x_at(e, t: float) -> float:
    x1, _, x2, _, dur = (float(v) for v in MOVE.search(e.text).groups())
    return x1 + (x2 - x1) * (t - e.start / 100) / (dur / 1000)


def ys(e) -> list[float]:
    """Screen y of every point of an event's drawing."""
    body = e.text.split("}", 1)[1].split("{")[0]
    nums = [int(v) for v in re.findall(r"-?\d+", body)]
    return [v / roll.Shape.K for v in nums[1::2]]


def test_roll_is_the_default_style_and_staff_still_selectable():
    lf, m = make_demo.truth_lyrics(), melody([(1.0, 2.0, 67)])
    meta = Meta(title="Demo", duration=24.0)
    doc = build_ass(lf, meta, RS, RES, melody=m)
    assert "pitch-word" in doc and "pitch-clef" not in doc
    staff_doc = build_ass(lf, meta, RenderSettings(show_pitch=True, pitch_style="staff"), RES, melody=m)
    assert "pitch-clef" in staff_doc and "pitch-word" not in staff_doc


def test_range_fits_the_song():
    assert roll.pitch_range([(62, 1.0), (64, 1.0)]) == (58, 69)  # at least an octave, the notes in the middle
    assert roll.pitch_range([(55, 1.0), (76, 1.0)]) == (54, 77)  # the whole range, padded by a row
    lo, hi = roll.pitch_range([(40, 0.05), (60, 10.0), (72, 10.0), (95, 0.05)])
    assert (lo, hi) == (59, 73)  # rare extremes do not flatten the lane


def test_each_semitone_has_its_own_row_and_higher_is_up():
    ev = named(events(melody([(1, 2, 66), (3, 4, 67), (5, 6, 69)])), "pitch-note")
    y = [float(np.mean(ys(e))) for e in ev]
    assert y[0] > y[1] > y[2]
    assert abs((y[0] - y[1]) * 2 - (y[1] - y[2])) < 0.6  # a semitone is half a whole tone


def test_bars_reach_the_playhead_when_sung_and_fill_there():
    ev = events(melody([(5.0, 6.0, 67)]))
    L = roll.roll_layout(*RES, RS, *roll.pitch_range([(67, 1.0)]))
    bar, sung = named(ev, "pitch-note")[0], named(ev, "pitch-sung")[0]
    assert abs(x_at(bar, 5.0) - L.playhead_x) < 1.0
    assert abs(x_at(bar, 4.0) - (L.playhead_x + L.px_per_s)) < 1.0
    assert f"\\clip({L.notes_left:.0f}," in sung.text and f",{L.playhead_x:.0f}," in sung.text


def test_a_pitch_held_over_syllables_is_one_bar_with_slits():
    notes = roll.roll_notes(melody([(1.0, 1.5, 67), (1.5, 2.0, 67), (2.0, 2.5, 69), (3.0, 3.5, 69)]), 60, 72)
    assert [(n.join_left, n.join_right) for n in notes] == [(False, True), (True, False), (False, False),
                                                           (False, False)]
    free = named(events(melody([(1.0, 1.5, 67)])), "pitch-note")[0]
    joined = named(events(melody([(1.0, 1.5, 67), (1.5, 2.0, 67)])), "pitch-note")[0]
    assert " b " in free.text and joined.text.count(" b ") < free.text.count(" b ")  # one square end


def test_tonic_rows_are_labelled():
    labels = [e.text.rsplit("}", 1)[1] for e in named(events(melody([(1, 2, 62), (3, 4, 74)])), "pitch-label")]
    assert labels == ["D4", "D5"]
    assert not named(events(melody([(1, 2, 62)], key=None)), "pitch-label")


def test_pitch_line_follows_the_contour_and_unsteady_bars_are_faint():
    hop = 0.01
    pitch = np.full(1000, np.nan)
    pitch[100:200] = 67.0 + 0.3  # held, on a recording tuned 0.3 semitones sharp
    pitch[300:400] = np.linspace(64.0, 70.0, 100) + 0.3  # a glide through the note: not held
    m = melody([(1.0, 2.0, 67), (3.0, 4.0, 67)], tuning=0.3)
    ev = events(m, contour=(pitch, hop))
    assert [e.name for e in ev if e.name in ("pitch-note", "pitch-faint")] == ["pitch-note", "pitch-faint"]
    curves = named(ev, "pitch-curve")
    assert len(curves) == 2
    bar_y = float(np.mean(ys(named(ev, "pitch-note")[0])))
    assert abs(float(np.mean(ys(curves[0]))) - bar_y) < 1.0  # the tuning offset is taken out
    assert max(ys(curves[1])) - min(ys(curves[1])) > 3 * (max(ys(curves[0])) - min(ys(curves[0])))
    assert not named(events(m), "pitch-curve") and not named(events(m), "pitch-faint")  # no contour: bars only


def test_words_scroll_under_their_notes():
    lf = make_demo.truth_lyrics()
    first = lf.lines[0].words[0]
    m = melody([(first.start, first.end, 67)])
    ev = events(m, lyrics=lf)
    words = named(ev, "pitch-word")
    assert len(words) == sum(len(ln.words) for ln in lf.lines) == len(named(ev, "pitch-word-sung"))
    L = roll.roll_layout(*RES, RS, *roll.pitch_range([(67, 1.0)]))
    bar = named(ev, "pitch-note")[0]
    assert abs(x_at(words[0], first.start) - x_at(bar, first.start)) < 1.0  # the word starts under its note
    assert words[0].text.endswith(first.text)
    _, y, _, _, _ = (float(v) for v in MOVE.search(words[0].text).groups())
    assert L.lane[1] < y < L.panel[3]  # in the rail, below the rows


def test_fast_words_are_squeezed_to_their_slot():
    lf = make_demo.truth_lyrics()
    w = lf.lines[0].words
    w[1].start = w[0].start + 0.05  # the next word starts almost at once
    ev = named(events(melody([(w[0].start, w[0].end, 67)]), lyrics=lf), "pitch-word")
    assert f"\\fscx{roll.WORD_MIN_SCALE * 100:.0f}" in ev[0].text


def test_panel_grows_with_the_range_and_hides_in_long_gaps():
    narrow = roll.roll_layout(*RES, RS, 60, 71)
    wide = roll.roll_layout(*RES, RS, 54, 77)
    assert wide.panel[3] > narrow.panel[3] and wide.panel[3] <= (roll.PANEL_TOP + roll.PANEL_HEIGHT[1]) * RES[1] + 1
    assert wide.row_h >= 0.9 * roll.ROW_H * RES[1]
    panels = named(events(melody([(2.0, 3.0, 67), (40.0, 41.0, 67)]), duration=60.0), "pitch-panel")
    assert len(panels) == 2
