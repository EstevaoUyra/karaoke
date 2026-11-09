from __future__ import annotations

import make_demo
import numpy as np
import pytest
import soundfile as sf

from karaoke import audio

SR = 16000


def _write(path, signal, sr=SR):
    sf.write(str(path), np.stack([signal, signal], axis=1).astype(np.float32), sr, subtype="PCM_16")
    return path


def _bursts(total, bursts, sr=SR, amp=0.3, noise_db=-80.0, seed=0):
    """Sine bursts at (start, end, gain_db) on a faint noise floor."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(total * sr)) / sr
    sig = rng.normal(0, 10 ** (noise_db / 20), len(t))
    for s, e, gain in bursts:
        idx = (t >= s) & (t < e)
        sig[idx] += amp * 10 ** (gain / 20) * np.sin(2 * np.pi * 330 * t[idx])
    return sig


def _close(got, want, tol=0.06):
    assert len(got) == len(want), (got, want)
    for (gs, ge), (ws, we) in zip(got, want, strict=True):
        assert abs(gs - ws) <= tol and abs(ge - we) <= tol, (got, want)


def test_duration(tmp_path):
    p = _write(tmp_path / "a.wav", np.zeros(int(2.5 * SR)))
    assert audio.duration(p) == pytest.approx(2.5)


def test_voice_regions_on_tone_bursts(tmp_path):
    sig = _bursts(10.0, [(1.0, 2.0, 0), (2.3, 3.0, 0), (5.0, 6.0, 0), (8.0, 8.05, 0)])
    p = _write(tmp_path / "v.wav", sig)
    got = audio.voice_regions(p, merge_gap_s=0.6, pad_s=0.2)
    # 2.0→2.3 gap (< 0.6 s) merged; the 50 ms blip dropped; 0.2 s padding.
    _close(got, [(0.8, 3.2), (4.8, 6.2)])
    # 1.5 s gap stays split; defaults pad by PAD_S
    sig = _bursts(10.0, [(1.0, 2.0, 0), (3.5, 4.0, 0)])
    pad = audio.PAD_S
    _close(audio.voice_regions(_write(tmp_path / "w.wav", sig)), [(1.0 - pad, 2.0 + pad), (3.5 - pad, 4.0 + pad)])


def test_quiet_leakage_is_not_voice(tmp_path):
    # Bleed 45 dB below the singing must not open a region.
    sig = _bursts(10.0, [(1.0, 4.0, 0), (5.0, 9.0, -45)])
    got = audio.voice_regions(_write(tmp_path / "v.wav", sig), pad_s=0.2)
    _close(got, [(0.8, 4.2)])


def test_hysteresis_keeps_a_dipping_note(tmp_path):
    # Loud, then a dip to -34 dB (between ON -30 and OFF -38), then loud again: one region.
    sig = _bursts(8.0, [(1.0, 2.0, 0), (2.0, 3.0, -34), (3.0, 4.0, 0)])
    got = audio.voice_regions(_write(tmp_path / "v.wav", sig), merge_gap_s=0.0, pad_s=0.2)
    _close(got, [(0.8, 4.2)])
    # A deeper dip (-42 dB) closes the region.
    sig = _bursts(8.0, [(1.0, 2.0, 0), (2.0, 3.0, -42), (3.0, 4.0, 0)])
    got = audio.voice_regions(_write(tmp_path / "w.wav", sig), merge_gap_s=0.0, pad_s=0.0)
    _close(got, [(1.0, 2.0), (3.0, 4.0)])


def test_silence_gives_no_regions(tmp_path):
    assert audio.voice_regions(_write(tmp_path / "s.wav", np.zeros(3 * SR))) == []


def test_voice_regions_on_demo_vocals(demo):
    """The demo vocal stem has tones exactly at the TRUTH word times."""
    got = audio.voice_regions(demo.vocals, merge_gap_s=0.6)  # (the 1.0 s verse→chorus gap is borderline at 1.0)
    lines = [(w[0][1], w[-1][2]) for _, w in make_demo.TRUTH]
    # Words inside a line are < 0.6 s apart → merged; lines 1-2 (gap 0.2 s) and 3-4 merge too.
    want = audio.merge_regions(lines, 0.6)
    want = audio.merge_regions([(s - audio.PAD_S, e + audio.PAD_S) for s, e in want], 0.0)  # padded may touch
    _close(got, want, tol=0.08)
    truth_words = [(s, e) for _, ws in make_demo.TRUTH for _, s, e in ws]
    for s, e in truth_words:  # every sung word is inside a region
        assert any(rs <= s and e <= re_ for rs, re_ in got)


def test_merge_and_pack():
    assert audio.merge_regions([(3, 4), (0, 1), (1.3, 2)], 0.5) == [(0, 2), (3, 4)]
    regions = [(0.0, 5.0), (6.0, 10.0), (20.0, 22.0), (23.0, 48.0), (49.0, 55.0)]
    # 6-10 joins (gap 1 s); 20 is 10 s away; 23-48 joins 20-22; 49-55 would make it > 30 s.
    assert audio.pack_clips(regions, max_len=30, max_gap=2) == [(0.0, 10.0), (20.0, 48.0), (49.0, 55.0)]
    assert audio.pack_clips([]) == []
    assert audio.coverage([(0, 1.5), (2, 3)]) == pytest.approx(2.5)


def test_rms_db_frames():
    sig = np.ones(SR, dtype=np.float32) * 0.5
    db = audio.rms_db(sig, SR, frame_s=0.05, hop_s=0.01)
    assert len(db) == 100
    assert db[50] == pytest.approx(20 * np.log10(0.5), abs=0.01)


def test_long_regions_split_at_quiet_points(tmp_path):
    # 3 s phrases separated by 0.3 s breaths (merged into one long region), 24 s in all.
    bursts = [(1.0 + k * 3.3, 4.0 + k * 3.3, 0) for k in range(7)]
    p = _write(tmp_path / "v.wav", _bursts(26.0, bursts))
    (whole,) = audio.voice_regions(p)
    pieces = audio.voice_regions(p, max_region_s=10.0)
    assert len(pieces) >= 3
    assert pieces[0][0] == whole[0] and pieces[-1][1] == whole[1]
    for (s, e), (s2, _) in zip(pieces, pieces[1:], strict=False):
        assert e == s2 and e - s <= 10.0 + 1e-6
        # every cut lands in a breath, not inside a phrase
        assert any(4.0 + k * 3.3 - 0.03 <= e <= 4.3 + k * 3.3 + 0.03 for k in range(7)), e
