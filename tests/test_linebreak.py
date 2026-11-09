from __future__ import annotations

import numpy as np

from karaoke.align.fallback import group_lines
from karaoke.align.tokens import WToken, match_key
from karaoke.align.voicing import Envelope, trim_span, trim_words
from karaoke.linebreak import W, split_to_fit, starts_sentence

# --------------------------------------------------------------------------- linebreak


def _words(text: str, pitch: float = 0.4) -> list[W]:
    return [W(t, i * pitch, i * pitch + 0.3) for i, t in enumerate(text.split())]


def _lines(words: list[W], parts: list[list[int]]) -> list[str]:
    return [" ".join(words[i].text for i in p) for p in parts]


def test_short_lines_are_left_alone():
    ws = _words("we keep the window open")
    assert split_to_fit(ws, len, 40) == [list(range(5))]


def test_split_prefers_punctuation_over_balance():
    ws = _words("we kept on walking, the night was cold and the road was long")
    assert _lines(ws, split_to_fit(ws, len, 40)) == ["we kept on walking,", "the night was cold and the road was long"]


def test_split_prefers_a_pause():
    ws = _words("we kept on walking the night was cold and long")
    ws = ws[:4] + [W(w.text, w.start + 1.5, w.end + 1.5) for w in ws[4:]]  # 1.5 s pause after "walking"
    assert _lines(ws, split_to_fit(ws, len, 30)) == ["we kept on walking", "the night was cold and long"]


def test_split_never_leaves_an_article_dangling():
    ws = _words("I dream about the open sea every night")
    parts = _lines(ws, split_to_fit(ws, len, 25))
    assert not any(p.split()[-1] in ("the", "about") for p in parts[:-1])


def test_soft_width_splits_only_at_strong_breaks():
    ws = _words("sweet love of mine (sweet, sweet)")
    assert _lines(ws, split_to_fit(ws, len, 60, soft_width=20)) == ["sweet love of mine", "(sweet, sweet)"]
    plain = _words("sweet love of mine every single day")
    assert split_to_fit(plain, len, 60, soft_width=20) == [list(range(7))]


def test_very_long_lines_split_more_than_once():
    ws = _words(" ".join(f"word{i}" for i in range(30)))
    parts = split_to_fit(ws, len, 40)
    assert len(parts) >= 4 and all(len(" ".join(ws[i].text for i in p)) <= 40 for p in parts)
    assert [i for p in parts for i in p] == list(range(30))


def test_starts_sentence():
    assert starts_sentence("Eles") and starts_sentence("(Sweet")
    assert not starts_sentence("I") and not starts_sentence("I'm") and not starts_sentence("eles")


# --------------------------------------------------------------------------- whisper fallback


def _toks(spec: list[tuple[str, float, float]]) -> list[WToken]:
    return [WToken(index=i, text=t, key=match_key(t), start=s, end=e, prob=0.9) for i, (t, s, e) in enumerate(spec)]


def _seq(text: str, t0: float, pitch: float = 0.3) -> list[tuple[str, float, float]]:
    return [(w, t0 + i * pitch, t0 + i * pitch + 0.25) for i, w in enumerate(text.split())]


def test_fallback_breaks_at_capitals_and_reattaches_tails():
    # whisper segments would have cut "boy" onto the next line; segments are ignored now
    toks = _toks(_seq("Nós vamos dançar boy Nós vamos dançar Mas eu não vou", 1.0))
    assert [" ".join(t.text for t in g) for g in group_lines(toks)] == [
        "Nós vamos dançar boy", "Nós vamos dançar", "Mas eu não vou"]


def test_fallback_drops_ellipses_and_continues_the_sentence():
    toks = _toks(_seq("a minha cabeça não para de... Pensar", 1.0))
    lines = [" ".join(t.text for t in g) for g in group_lines(toks)]
    assert lines == ["a minha cabeça não para de pensar"]


def test_fallback_splits_long_runs_and_merges_lone_words():
    toks = _toks(_seq("Vem", 1.0) + _seq("dançar comigo na beira do mar e depois voltar para casa cantando", 1.3))
    lines = [" ".join(t.text for t in g) for g in group_lines(toks)]
    assert lines[0].startswith("Vem dançar")  # the lone "Vem" joins what follows it
    assert len(lines) == 2 and all(len(x) <= 42 for x in lines)


def test_fallback_breaks_at_pauses():
    toks = _toks(_seq("we go on", 1.0) + _seq("and on", 4.0))
    assert [" ".join(t.text for t in g) for g in group_lines(toks)] == ["we go on", "and on"]


# --------------------------------------------------------------------------- voicing


def _env(segments: list[tuple[float, float, float]], total: float = 10.0) -> Envelope:
    """dB envelope (10 ms frames): -70 dB silence, -40 dB 'reverb floor', with loud segments."""
    db = np.full(int(total / 0.01), -42.0)
    for s, e, level in segments:
        db[int(s / 0.01):int(e / 0.01)] = level
    return Envelope(db)


def test_trailing_pause_is_cut_out_of_a_word():
    env = _env([(1.0, 1.6, -20.0), (3.0, 3.5, -20.0)])  # sung, then a pause at reverb level
    s, e, onset = trim_span(env, 1.0, 3.0)
    assert abs(e - 1.7) < 0.02 and s == 1.0 and onset is None


def test_leading_pause_moves_the_start_to_the_onset():
    env = _env([(2.2, 3.0, -20.0)])
    s, e, _ = trim_span(env, 1.0, 3.0)
    assert abs(s - 2.05) < 0.02 and e == 3.0
    assert trim_span(_env([(1.4, 3.0, -20.0)]), 1.0, 3.0)[0] == 1.0  # 0.4 s: a soft consonant, kept


def test_pause_then_early_onset_pulls_the_next_word():
    # the word "ends" at 3.2 per whisper, but the voice came back at 2.9 for the next word
    env = _env([(1.0, 1.5, -20.0), (2.9, 4.0, -20.0)])
    times = [[1.0, 3.2], [3.2, 4.0]]
    assert trim_words(env, times) == 1
    assert abs(times[0][1] - 1.6) < 0.02
    assert abs(times[1][0] - 2.75) < 0.02


def test_continuous_singing_and_fixed_words_are_untouched():
    env = _env([(1.0, 4.0, -20.0)])
    times = [[1.0, 2.5], [2.5, 4.0]]
    assert trim_words(env, times) == 0 and times == [[1.0, 2.5], [2.5, 4.0]]
    env2 = _env([(1.0, 1.5, -20.0)])
    fixed = [[1.0, 3.0]]
    assert trim_words(env2, fixed, [True]) == 0 and fixed == [[1.0, 3.0]]
