"""Reference scoring must not reward or penalize splitting a held pitch."""
from pathlib import Path


def test_aggregate_pitch_is_invariant_to_syllable_splits_and_duplicate_overlaps(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "spikes"))
    from audit_melody_gold import aggregate_pitch
    from score_gold import N

    ref = N(0, 1, 60)
    held = [N(0, .18, 60), N(.18, .25, 62)]
    split = [N(0, .06, 60), N(.06, .12, 60), N(.12, .18, 60), N(.18, .25, 62)]
    assert aggregate_pitch(ref, held) == aggregate_pitch(ref, split) == 60
    assert aggregate_pitch(ref, split + [N(.18, .25, 62)] * 5) == 60
    assert aggregate_pitch(ref, []) is None
