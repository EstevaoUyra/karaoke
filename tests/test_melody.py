"""Melody extraction on synthetic pitch contours, key detection and staff spelling (no audio)."""

from __future__ import annotations

import numpy as np
import pytest

from karaoke.melody import notes as N
from karaoke.melody.theory import (
    Key,
    detect_key,
    display_shift,
    ledger_positions,
    pc_histogram,
    spell,
)

HOP = 0.04


def contour(parts: list[tuple[float, float | None]], vibrato: float = 0.0, jitter: float = 0.05,
            glide: int = 2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(midi, prob) frames for [(seconds, midi or None for silence)], with vibrato (semitones,
    5.5 Hz), random jitter and ``glide`` frames of slide between notes."""
    rng = np.random.default_rng(seed)
    midi: list[float] = []
    for dur, m in parts:
        n = int(round(dur / HOP))
        if m is None:
            midi += [np.nan] * n
            continue
        t = np.arange(n) * HOP
        seg = m + vibrato * np.sin(2 * np.pi * 5.5 * t) + rng.normal(0, jitter, n)
        if glide and midi and np.isfinite(midi[-1]):
            prev = midi[-1]
            k = min(glide, n)
            seg[:k] = np.linspace(prev, m, k + 2)[1:-1]
        midi += list(seg)
    arr = np.asarray(midi)
    prob = np.where(np.isfinite(arr), 0.9, 0.0)
    return arr, prob


def summary(notes):
    return [(n.midi, round(n.start, 2), round(n.end, 2)) for n in notes]


# --------------------------------------------------------------------------- segmentation


def test_segments_steady_notes():
    midi, prob = contour([(0.4, None), (0.5, 60), (0.3, 62), (0.6, 64), (0.4, None), (0.4, 65)])
    notes = N.segment(midi, prob, HOP)
    assert [n.midi for n in notes] == [60, 62, 64, 65]
    for n, (s, e) in zip(notes, [(0.4, 0.9), (0.9, 1.2), (1.2, 1.8), (2.2, 2.6)], strict=True):
        assert n.start == pytest.approx(s, abs=0.07) and n.end == pytest.approx(e, abs=0.07)
    assert all(0.85 <= n.conf <= 0.95 for n in notes)


def test_vibrato_and_jitter_stay_one_note():
    midi, prob = contour([(2.0, 67)], vibrato=0.5, jitter=0.1)
    notes = N.segment(midi, prob, HOP)
    assert summary(notes) == [(67, 0.0, 1.98)]
    assert notes[0].pitch == pytest.approx(67, abs=0.15)


def test_short_blips_and_scoops_are_merged_or_dropped():
    # a 0.08 s blip between phrases is dropped; a 0.08 s scoop into a note joins the note
    midi, prob = contour([(0.5, 60), (0.3, None), (0.08, 72), (0.3, None), (0.08, 63), (0.6, 65)])
    notes = N.segment(midi, prob, HOP)
    assert [n.midi for n in notes] == [60, 65]


def test_tuning_offset_is_taken_out():
    # a recording 40 cents sharp, sung with some intonation error: rounding alone gets 62 and 65 wrong
    sung = [60.28, 62.52, 64.3, 65.52, 67.35]
    midi, prob = contour([(0.5, m) for m in sung], jitter=0.02, glide=0)
    assert N.estimate_tuning(midi, prob, HOP) == pytest.approx(0.39, abs=0.03)
    assert [n.midi for n in N.segment(midi, prob, HOP)] == [60, 62, 64, 65, 67]
    assert [n.midi for n in N.segment(midi, prob, HOP, tuning=0.0)] == [60, 63, 64, 66, 67]
    midi, prob = contour([(2.0, 67)], vibrato=0.5, jitter=0.1)  # vibrato frames do not bias it
    assert abs(N.estimate_tuning(midi, prob, HOP)) < 0.1


def test_unsure_frames_count_only_where_the_pitch_holds():
    midi, prob = contour([(0.6, 62), (0.4, None), (0.6, 64)], jitter=0.02)
    prob[:] = np.where(np.isfinite(midi), 0.2, 0.0)  # pyin unsure everywhere
    notes = N.segment(midi, prob, HOP)
    assert [n.midi for n in notes] == [62, 64]  # steady, so trusted
    rng = np.random.default_rng(3)
    noisy = np.where(np.isfinite(midi), 63 + rng.uniform(-4, 4, len(midi)), np.nan)
    assert N.segment(noisy, prob, HOP) == []  # unsure and unsteady: leakage, not singing


def test_quiet_echo_tail_is_dropped():
    midi, prob = contour([(0.6, 62), (0.6, 62.0)], glide=0)
    midi[15:] = 69.0  # the second half is an echo of another note ...
    db = np.full(len(midi), -12.0)
    db[15:] = -50.0  # ... far quieter than the singing
    notes = N.segment(midi, prob, HOP, db=db)
    assert [n.midi for n in notes] == [62]


def test_octave_errors_are_folded_back():
    midi, prob = contour([(0.4, 64), (0.2, 76), (0.4, 65)], jitter=0.02)
    assert [n.midi for n in N.segment(midi, prob, HOP)] == [64, 65]  # 76 → 64, merged with it


def test_outliers_far_from_the_melody_are_dropped():
    parts = [(0.4, 64), (0.2, None), (0.4, 67), (0.2, None), (0.4, 65), (0.2, None), (0.5, 38),
             (0.2, None), (0.4, 62), (0.2, None), (0.4, 64)]
    midi, prob = contour(parts)
    assert 38 not in [n.midi for n in N.segment(midi, prob, HOP)]


def test_words_split_held_notes_and_filter_unsure_ones():
    # one held 60 over two words, then an unsure short note with no word, then a long sure one
    midi, prob = contour([(1.0, 60), (0.5, None), (0.2, 67), (0.5, None), (0.6, 64)])
    prob[int(1.5 / HOP):int(1.7 / HOP)] = 0.55
    words = [(0.0, 0.5), (0.5, 1.0)]
    notes = N.segment(midi, prob, HOP, words=words)
    assert summary(notes)[:2] == [(60, 0.0, 0.5), (60, 0.5, 0.98)]
    assert [n.midi for n in notes] == [60, 60, 64]  # 67 dropped (unsure, wordless); 64 kept (sure, long)


# --------------------------------------------------------------------------- key detection


def scale_notes(key: Key, octave: int = 4) -> list[tuple[int, float]]:
    """A melody over the key's scale, with the tonic, third and fifth weighted like a real tune."""
    base = 12 * (octave + 1) + key.tonic_pc
    steps = [0, 2, 4, 5, 7, 9, 11] if key.mode == "major" else [0, 2, 3, 5, 7, 8, 11]  # harmonic minor
    weight = {0: 3.0, 2: 1.5, 4: 2.0}
    return [(base + s, weight.get(i, 1.0)) for i, s in enumerate(steps)]


@pytest.mark.parametrize("key", [Key(0, "major"), Key(7, "major"), Key(5, "major"), Key(2, "major"),
                                 Key(9, "minor"), Key(4, "minor"), Key(2, "minor"), Key(3, "major")])
def test_krumhansl_schmuckler_on_scales(key: Key):
    res = detect_key(pc_histogram(scale_notes(key)))
    assert res is not None and res.key == key and res.r > 0.7


def test_detect_key_averages_two_sources():
    melody = pc_histogram([(62, 2.0), (63, 2.0), (60, 1.0)])  # rap on three pitches: ambiguous
    chroma = np.roll([6.3, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 4.8, 4.0, 2.7, 3.3, 3.2], 7)  # G minor harmony
    both = detect_key(melody, chroma)
    assert both is not None and both.key == Key(7, "minor")
    assert detect_key(None, None) is None and detect_key(np.zeros(12)) is None


def test_key_signatures():
    assert Key(7, "major").signature == ["F#"] and Key(7, "major").tonic == "G"
    assert Key(5, "major").signature == ["Bb"] and Key(5, "major").tonic == "F"
    assert Key(4, "minor").fifths == 1 and Key(4, "minor").tonic == "E"
    assert Key(1, "major").name == "Db major" and Key(1, "major").fifths == -5
    assert Key(6, "major").name == "F# major" and Key(6, "major").fifths == 6
    assert Key(3, "minor").name == "Eb minor" and len(Key(3, "minor").signature) == 6
    assert Key(10, "minor").signature == ["Bb", "Eb", "Ab", "Db", "Gb"]


# --------------------------------------------------------------------------- staff spelling


def test_staff_positions_and_ledger_lines():
    c = Key(0, "major")
    assert spell(64, c).staff_position == 0  # E4: bottom line
    assert spell(77, c).staff_position == 8  # F5: top line
    assert spell(60, c).staff_position == -2  # C4: first ledger line below
    assert spell(81, c).staff_position == 10  # A5: first ledger line above
    assert ledger_positions(-2) == [-2] and ledger_positions(-5) == [-2, -4]
    assert ledger_positions(11) == [10] and ledger_positions(12) == [10, 12]
    assert ledger_positions(4) == [] and ledger_positions(-1) == [] and ledger_positions(9) == []


def test_spelling_in_g_major():
    g = Key(7, "major")
    fs = spell(66, g)  # F#4 is in the key: on the F space, nothing drawn
    assert (fs.name, fs.staff_position, fs.accidental) == ("F#4", 1, None)
    f = spell(65, g)  # F4 natural: same space, with a natural sign
    assert (f.name, f.staff_position, f.accidental) == ("F4", 1, "natural")
    gs = spell(68, g)
    assert (gs.name, gs.accidental) == ("G#4", "sharp")


def test_spelling_in_flat_and_minor_keys():
    f = Key(5, "major")
    assert (spell(70, f).name, spell(70, f).accidental) == ("Bb4", None)
    assert (spell(71, f).name, spell(71, f).accidental) == ("B4", "natural")
    d_minor = Key(2, "minor")
    assert (spell(61, d_minor).name, spell(61, d_minor).accidental) == ("C#4", "sharp")  # leading tone, not Db
    a_minor = Key(9, "minor")
    assert (spell(68, a_minor).name, spell(68, a_minor).accidental) == ("G#4", "sharp")
    c = Key(0, "major")
    assert (spell(70, c).name, spell(70, c).accidental) == ("Bb4", "flat")  # lowered 7th
    eb = Key(3, "major")
    assert spell(63, eb).name == "Eb4" and spell(63, eb).accidental is None
    assert spell(71, eb).accidental == "natural"  # B natural in Eb major


def test_b_sharp_and_c_flat_octaves():
    cs_major = Key(1, "major")  # Db major in practice (5 flats): C natural is in the scale
    assert spell(60, cs_major).name == "C4"
    fs = Key(6, "major")  # F# major: E# is the 7th degree, written on the E line of its octave
    es = spell(65, fs)
    assert (es.name, es.staff_position, es.accidental) == ("E#4", 0, None)


def test_display_shift():
    assert display_shift(57.0) == 1  # a low voice: written an octave up (8vb clef)
    assert display_shift(68.0) == 0
    assert display_shift(84.0) == -1
    assert display_shift(65.0) == 0  # a tie keeps the written octave


# --------------------------------------------------------------------------- words as a prior


def test_words_let_unsure_singing_through():
    # a sung word where the tracker is unsure (conf 0.2) and the pitch wobbles (not "steady")
    midi, prob = contour([(0.3, None), (0.8, 64), (0.3, None)], jitter=0.45, seed=5)
    prob[np.isfinite(midi)] = 0.2
    assert N.segment(midi, prob, HOP, prof=N.PROFILES["pyin-v1"]) == []
    notes = N.segment(midi, prob, HOP, words=[(0.3, 1.1)], prof=N.profile("crepe-tiny"))
    assert [n.midi for n in notes] == [64] and notes[0].dur > 0.6
    # the same frames with no word there stay out
    assert N.segment(midi, prob, HOP, words=[(5.0, 6.0)], prof=N.profile("crepe-tiny")) == []


def test_dropouts_inside_a_word_are_bridged():
    midi, prob = contour([(0.5, 62), (0.16, None), (0.5, 62.3)], jitter=0.02, glide=0)
    words = [(0.0, 1.16)]
    bridged = N.segment(midi, prob, HOP, words=words)
    assert [(n.midi, round(n.start, 1), round(n.end, 1)) for n in bridged] == [(62, 0.0, 1.1)]
    # a dropout between words, or a long one, is a rest
    assert len(N.segment(midi, prob, HOP, words=[(0.0, 0.5), (0.66, 1.16)])) == 2
    long_gap, p2 = contour([(0.5, 62), (0.5, None), (0.5, 62)], jitter=0.02, glide=0)
    assert len(N.segment(long_gap, p2, HOP, words=[(0.0, 1.5)])) == 2


def test_gaps_between_notes_of_a_word_are_closed():
    a = [N.RawNote(0.0, 0.4, 60, 60.0, 0.9), N.RawNote(0.55, 0.9, 62, 62.0, 0.9), N.RawNote(1.5, 2.0, 64, 64.0, 0.9)]
    assert N.close_gaps(a, [(0.0, 0.9), (1.4, 2.0)], 0.2) == 1
    assert a[0].end == 0.55 and a[1].end == 0.9  # 0.9-1.5 is between words: a rest


def test_profiles_by_tracker():
    assert N.profile("crepe-full") is N.profile("crepe-tiny") is N.PROFILES["crepe"]
    assert N.profile("pyin") is N.PROFILES["pyin"]
    assert N.profile("something-else") is N.PROFILES["pyin"]
