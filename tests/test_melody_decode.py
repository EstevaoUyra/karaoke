"""Decoder regressions: alternative pitches, rests, silence and continuous backtracking."""
import numpy as np
import pytest

from karaoke.melody.decode import Prior, activation_from_contour, peak_pitch, viterbi


def path(s, prior=None, rest=0.1, **kwargs):
    p = prior or Prior()
    return viterbi(s, np.full(len(s), rest), np.full(len(s), p.change), p, 0.01, **kwargs)


def test_supported_nearby_pitch_beats_brief_distant_peak():
    s = np.zeros((100, 13))
    s[:, 0] = 0.8
    s[40:45, 0], s[40:45, 9] = 0.5, 0.9
    assert np.all(path(s) == 0)
    s[30:80, 0], s[30:80, 9] = 0.0, 0.95
    assert np.all(path(s)[40:70] == 9)  # strong, sustained real leaps survive


def test_one_frame_rest_cannot_bypass_change_cost():
    s = np.zeros((101, 3))
    s[:50, 0], s[51:, 1] = 0.9, 0.9
    s[51:, 0] = 0.8
    p = Prior(weight=0, change=20, on=0, off=0)
    assert 1 not in path(s, p)
    assert 1 in path(s, Prior(weight=0, change=1, on=0, off=0))


def test_forgetting_rest_backtracks_through_the_actual_previous_pitch():
    s = np.zeros((400, 13))
    s[:50, 0], s[350:, 12] = 0.9, 0.9
    a = path(s)
    assert np.all(a[:50] == 0)
    assert np.all(a[50:350] == -1)
    assert np.all(a[350:] == 12)
    assert np.array_equal(a, path(s))


def test_constraints_and_empty_audio():
    assert path(np.zeros((0, 3))).size == 0
    assert np.all(path(np.zeros((10, 3))) == -1)
    s = np.full((100, 3), 0.9)
    ok = np.zeros(100, dtype=bool)
    ok[20:80] = True
    a = path(s, note_ok=ok, rest_ok=~ok)
    assert np.all(a[~ok] == -1)
    assert np.all(a[ok] >= 0)


def test_pyin_activation_preserves_pitch_and_unvoiced_frames():
    a, b = activation_from_contour(np.array([440., np.nan, 220.]), np.array([0.9, 0.8, 0.7]))
    m, c = peak_pitch(a, b)
    assert m[[0, 2]] == pytest.approx([69, 57], abs=0.05)
    assert np.isnan(m[1]) and c[1] == 0
