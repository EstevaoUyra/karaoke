"""Pitch guide: ASS event structure, scrolling geometry and the melody stage (fake pitch track)."""

from __future__ import annotations

import re

import make_demo
import numpy as np
import pytest

from karaoke.models import KeyEstimate, Melody, Meta, Note, RenderSettings, StageName, StageStatus
from karaoke.pipeline import ORDER, run_job
from karaoke.render import staff
from karaoke.render.ass import build_ass
from karaoke.store import JobStore, read_model

RES = (1280, 720)
MOVE = re.compile(r"\\move\(([-\d.]+),0,([-\d.]+),0,0,(\d+)\)")
CLIP = re.compile(r"\\clip\((\d+),(\d+),(\d+),(\d+)\)")
G_MAJOR = KeyEstimate(tonic="G", tonic_pc=7, mode="major", fifths=1, accidentals=["F#"], confidence=0.8)


def melody(notes: list[tuple[float, float, int]], key: KeyEstimate | None = G_MAJOR, shift: int = 0) -> Melody:
    return Melody(hop_s=0.04, notes=[Note(start=s, end=e, midi=m, pitch=float(m)) for s, e, m in notes],
                  key=key, display_octave_shift=shift)


def meta(duration: float = 24.0) -> Meta:
    return Meta(title="Demo Band - Sing It Loud", duration=duration, clean_title="Sing It Loud")


def pitch_events(mel: Melody, rs: RenderSettings | None = None, duration: float = 24.0):
    rs = rs or RenderSettings(show_pitch=True)
    return staff.pitch_events(mel, rs, RES, duration)


def times(ev) -> tuple[float, float]:
    return ev.start / 100, ev.end / 100


def x_at(ev, t: float) -> float:
    """Where a moving event's origin is at song time ``t``."""
    x1, x2, dur = (float(v) for v in MOVE.search(ev.text).groups())
    s = ev.start / 100
    return x1 + (x2 - x1) * (t - s) / (int(dur) / 1000)


@pytest.fixture(autouse=True)
def no_font(monkeypatch):
    """Tests do not depend on the machine's fonts: the vector clef unless a test says otherwise."""
    monkeypatch.setattr(staff, "clef_glyph", lambda: None)


def test_off_by_default_and_without_melody():
    lf, m = make_demo.truth_lyrics(), melody([(1.0, 1.5, 67)])
    assert "Pitch" not in build_ass(lf, meta(), RenderSettings(), RES, melody=m)
    assert "Pitch" not in build_ass(lf, meta(), RenderSettings(show_pitch=True), RES)
    doc = build_ass(lf, meta(), RenderSettings(show_pitch=True), RES, melody=m)
    styles = doc.split("[Events]")[0]
    assert "Style: Pitch," in styles
    assert doc.count("Dialogue:") > build_ass(lf, meta(), RenderSettings(), RES).count("Dialogue:")


def test_event_structure():
    m = melody([(5.0, 5.5, 67), (5.6, 6.4, 69), (6.5, 7.0, 66)])
    ev = pitch_events(m)
    names = [e.name for e in ev]
    for n in ("pitch-panel", "pitch-staff", "pitch-clef", "pitch-key", "pitch-playhead"):
        assert names.count(n) == 1
    assert names.count("pitch-note") == names.count("pitch-sung") == 3
    assert "pitch-accidental" not in names and "pitch-ledger" not in names  # all in G major, on the staff
    assert "pitch-octave" not in names
    assert all(e.style == "Pitch" for e in ev)
    panel = next(e for e in ev if e.name == "pitch-panel")
    for e in ev:  # everything on screen only while the staff is
        assert panel.start <= e.start and e.end <= panel.end


def test_bars_scroll_and_fill_at_the_playhead():
    rs = RenderSettings(show_pitch=True)
    L = staff.staff_layout(*RES, rs, n_key=1, clef_width_sp=3.2)
    m = melody([(10.0, 11.0, 67)])
    ev = pitch_events(m, rs)
    note = next(e for e in ev if e.name == "pitch-note")
    sung = next(e for e in ev if e.name == "pitch-sung")
    assert MOVE.search(note.text).groups() == MOVE.search(sung.text).groups()
    # the bar's start reaches the playhead exactly when the note starts, its end when it ends
    head = next(e for e in ev if e.name == "pitch-playhead")
    ph_clip = int(CLIP.search(sung.text).group(3))
    gap = staff.BAR_GAP * L.sp  # the bar is drawn from +gap to +duration*speed-gap of its origin
    assert x_at(note, 10.0) + gap == pytest.approx(ph_clip + gap, abs=1.0)
    bar_end = x_at(note, 11.0) + 1.0 * L.px_per_s - gap
    assert bar_end == pytest.approx(ph_clip - gap, abs=1.0)
    # it enters about pitch_lookahead_s early and leaves pitch_trail_s after it ends
    s, e = times(note)
    assert 10.0 - rs.pitch_lookahead_s - 0.3 < s < 10.0 - rs.pitch_lookahead_s + 0.05
    assert 11.0 + rs.pitch_trail_s - 0.05 < e < 11.0 + rs.pitch_trail_s + 0.3
    # the sung copy is clipped at the playhead, the unsung one to the whole note area
    nx0, _, nx1, _ = (int(v) for v in CLIP.search(note.text).groups())
    sx0, _, sx1, _ = (int(v) for v in CLIP.search(sung.text).groups())
    assert sx0 == nx0 and sx1 < nx1
    assert head.layer > sung.layer > note.layer
    assert "\\p3" in note.text and "\\1c&HF7C34F&" in sung.text  # sung colour (#4FC3F7)


def test_bars_sit_on_their_line_or_space():
    rs = RenderSettings(show_pitch=True)
    ev = pitch_events(melody([(1, 2, 64), (3, 4, 66), (5, 6, 71), (7, 8, 77)]), rs)  # E4 F#4 B4 F5
    L = staff.staff_layout(*RES, rs, n_key=1)
    centres = []
    for e in (e for e in ev if e.name == "pitch-note"):
        ys = [int(v) / 4 for v in re.findall(r"-?\d+ (-?\d+)", e.text.split("\\p3}")[1])]
        centres.append((min(ys) + max(ys)) / 2)
    assert centres == pytest.approx([L.y(0), L.y(1), L.y(4), L.y(8)], abs=0.5)


def test_accidentals_against_the_key_signature():
    # in G major: F#4 nothing, F4 natural, G#4 sharp, Bb4 flat
    ev = pitch_events(melody([(11, 12, 66), (13, 14, 65), (15, 16, 68), (17, 18, 70)]))
    notes = {times(e): x_at(e, 20) for e in ev if e.name == "pitch-note"}
    acc = [e for e in ev if e.name == "pitch-accidental"]
    assert [times(e) for e in acc] == list(notes)[1:]  # one per chromatic note, none for F#
    assert all(x_at(e, 20) == notes[times(e)] for e in acc)  # moving with its bar


def test_key_signature_drawing_follows_the_key():
    def sig(fifths: int, tonic: str, pc: int) -> str:
        k = KeyEstimate(tonic=tonic, tonic_pc=pc, mode="major", fifths=fifths)
        return next((e.text for e in pitch_events(melody([(1, 2, 67)], k)) if e.name == "pitch-key"), "")

    assert sig(0, "C", 0) == ""
    one, three = sig(1, "G", 7), sig(-3, "Eb", 3)
    # a sharp is four strokes, a flat two (stem and bowl)
    assert one.count(" m ") + one.count("}m ") == 4
    assert three.count(" m ") + three.count("}m ") == 6


def test_ledger_lines_and_out_of_range_notes():
    ev = pitch_events(melody([(1, 2, 60), (3, 4, 57), (5, 6, 84), (7, 8, 67), (9, 10, 40)]))
    ledgers = [e for e in ev if e.name == "pitch-ledger"]
    counts = [e.text.count(" m ") + e.text.count("}m ") for e in ledgers]
    assert counts == [1, 2, 2]  # C4: one line; A3: two; C6: two (A5, C6)
    assert len([e for e in ev if e.name == "pitch-note"]) == 4  # E2 would be off the panel: left out


def test_octave_shift_label():
    low = melody([(1, 2, 55), (3, 4, 57)], shift=1)  # a low voice written an octave up
    ev = pitch_events(low)
    label = [e for e in ev if e.name == "pitch-octave"]
    assert len(label) == 1 and label[0].text.endswith("8vb")
    assert not [e for e in ev if e.name == "pitch-ledger"]  # G3/A3 written as G4/A4: on the staff
    high = pitch_events(melody([(1, 2, 88)], shift=-1))
    assert [e.text[-3:] for e in high if e.name == "pitch-octave"] == ["8va"]


def test_staff_hides_during_long_gaps():
    rs = RenderSettings(show_pitch=True)
    m = melody([(10, 11, 67), (12, 13, 69), (40, 41, 67)])
    panels = [times(e) for e in pitch_events(m, rs, duration=60.0) if e.name == "pitch-panel"]
    assert len(panels) == 2
    assert panels[0][1] < 20 and panels[1][0] > 30
    one = melody([(10, 11, 67), (10 + staff.STAFF_GAP_S, 17, 69)])
    assert len([e for e in pitch_events(one, rs, duration=60.0) if e.name == "pitch-panel"]) == 1


def test_min_conf_hides_unsure_notes():
    m = melody([(1, 2, 67), (3, 4, 69)])
    m.notes[1].conf = 0.3
    ev = pitch_events(m, RenderSettings(show_pitch=True, pitch_min_conf=0.5))
    assert len([e for e in ev if e.name == "pitch-note"]) == 1


def test_clef_from_a_font_or_drawn(monkeypatch):
    ev = pitch_events(melody([(1, 2, 67)]))
    clef = next(e for e in ev if e.name == "pitch-clef")
    assert "\\p3" in clef.text  # no font: vector drawing
    glyph = staff.ClefGlyph("Some Music Font", None, height_em=0.733, top_em=0.656, left_em=0.0,  # type: ignore[arg-type]
                            width_em=0.335, line_em=1.05, ascent_em=0.88)
    monkeypatch.setattr(staff, "clef_glyph", lambda: glyph)
    ev = pitch_events(melody([(1, 2, 67)]))
    clef = next(e for e in ev if e.name == "pitch-clef")
    assert "\\fnSome Music Font" in clef.text and clef.text.endswith(staff.CLEF_CHAR)
    assert clef.text.count("\\pos(") == 1  # libass uses the first \pos only


def test_accidental_shapes_are_closed_polygons():
    for kind in ("sharp", "flat", "natural"):
        s = str(staff.accidental(staff.Shape(), kind, 100, 100, 15))
        assert s.startswith("m ") and " l " in s
    with pytest.raises(ValueError):
        staff.accidental(staff.Shape(), "double-sharp", 0, 0, 10)


# --------------------------------------------------------------------------- the stage


def _fake_track(calls: list, fail: set[str] = frozenset()):
    def fake_track(signal, sr, tracker, progress=None):
        calls.append(tracker)
        if tracker in fail:
            raise RuntimeError(f"{tracker} is broken")
        hop = 0.04
        n = int(len(signal) / sr / hop) + 1
        f0 = np.full(n, np.nan, dtype=np.float32)
        lf = make_demo.truth_lyrics()
        scale = [67, 71, 74, 72, 67, 69, 66, 67]  # G major (G B D, C, F# leading tone), mid-staff
        for i, w in enumerate(x for ln in lf.lines for x in ln.words):
            f0[int(w.start / hop) + 1:int(w.end / hop)] = 440 * 2 ** ((scale[i % len(scale)] - 69) / 12)
        prob = np.where(np.isfinite(f0), 0.9, 0.0).astype(np.float32)
        from karaoke.melody.decode import activation_from_contour

        act, bin0 = activation_from_contour(f0, prob)
        act = np.round(act * 255).astype(np.uint8).astype(np.float32) / 255
        return f0, prob, hop, act, bin0

    return fake_track


@pytest.fixture
def in_process(monkeypatch):
    """Run the (isolated) melody stage in this process, so the fake tracker is used."""
    from dataclasses import replace

    from karaoke import pipeline

    spec = pipeline.STAGE_BY_NAME[StageName.melody]
    assert spec.isolated  # CREPE imports torch: never in the same process as ctranslate2
    monkeypatch.setitem(pipeline.STAGE_BY_NAME, StageName.melody, replace(spec, isolated=False))


def test_melody_stage_with_a_fake_pitch_track(demo, monkeypatch, in_process):
    import karaoke.stages.melody as stage
    from karaoke.config import get_settings

    calls: list[str] = []
    monkeypatch.setattr(stage, "pitch_track", _fake_track(calls))
    monkeypatch.setattr(stage, "instrumental_chroma", lambda path: None)
    store = JobStore(demo.root.parent)
    settings = get_settings().model_copy(update={"melody_tracker": "crepe-tiny"})
    assert ORDER.index(StageName.melody) == ORDER.index(StageName.align) + 1
    run_job("demo", only=[StageName.melody], store=store, settings=settings)
    assert store.load("demo").stages[StageName.melody].status == StageStatus.done
    m = read_model(demo.melody, Melody)
    assert len(m.notes) >= 10 and {n.midi for n in m.notes} <= {66, 67, 69, 71, 72, 74}
    assert m.key is not None and (m.key.tonic, m.key.mode, m.key.accidentals) == ("G", "major", ["F#"])
    assert m.display_octave_shift == 0 and abs(m.tuning) < 0.05 and m.params["tracker"] == "crepe-tiny"
    assert demo.melody_f0.exists() and calls == ["crepe-tiny"]
    assert m.syllables and all(n.syllable is not None for n in m.notes)
    with np.load(demo.melody_f0) as cache:
        assert cache['act'].dtype == np.uint8 and int(cache['version']) == stage.TRACK_VERSION
        assert len(cache['act']) == len(cache['f0']) == len(cache['conf'])
    # Force a second decode, so this checks the activation cache rather than a skipped stage.
    run_job("demo", only=[StageName.melody], store=store, settings=settings, force=True)
    again = read_model(demo.melody, Melody)
    assert again.notes == m.notes and again.syllables == m.syllables
    assert calls == ["crepe-tiny"]
    # ... but another tracker does not reuse it
    run_job("demo", only=[StageName.melody], store=store,
            settings=settings.model_copy(update={"melody_tracker": "pyin"}))
    assert calls == ["crepe-tiny", "pyin"]


def test_melody_stage_falls_back_to_pyin(demo, monkeypatch, in_process):
    import karaoke.stages.melody as stage
    from karaoke.config import get_settings

    calls: list[str] = []
    monkeypatch.setattr(stage, "pitch_track", _fake_track(calls, fail={"crepe-full"}))
    monkeypatch.setattr(stage, "instrumental_chroma", lambda path: None)
    store = JobStore(demo.root.parent)
    settings = get_settings().model_copy(update={"melody_tracker": "crepe-full"})
    run_job("demo", only=[StageName.melody], store=store, settings=settings)
    assert calls == ["crepe-full", "pyin"]
    assert read_model(demo.melody, Melody).params["tracker"] == "pyin"
