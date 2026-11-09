from __future__ import annotations

import pytest

from karaoke.align.tokens import (
    WToken,
    drop_hallucination_loops,
    match_key,
    parse_chosen_text,
    parse_lrc,
    section_label,
    split_words,
)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("pra", "para"),
        ("Tá", "está"),
        ("cê", "Você"),
        ("'cause", "because"),
        ("cuz", "because"),
        ("o'", "of"),
        ("Nothin'", "nothing"),
        ("Coração,", "coracao"),
        ("“Hello!”", "hello"),
        ("don’t", "don't"),
    ],
)
def test_match_key_equivalences(a, b):
    assert match_key(a) == match_key(b)


def test_multiword_forms_concatenate():
    assert match_key("gonna") == match_key("going") + match_key("to")
    assert match_key("wanna") == "want" + "to"


def test_split_words_glues_punctuation():
    assert split_words("go on — run away !") == ["go", "on —", "run", "away !"]
    assert split_words("... and then") == ["... and", "then"]


def test_parse_chosen_text_sections_and_lines():
    text = "[Verse 1: Someone]\nfirst line here\nsecond line\n\n[Chorus x2]\nla la\n\n[Bridge]\n\n  \nend now\n"
    lines = parse_chosen_text(text)
    assert [ln.text for ln in lines] == ["first line here", "second line", "la la", "end now"]
    assert [ln.section for ln in lines] == ["verse", "verse", "chorus", "bridge"]
    assert [ln.index for ln in lines] == [0, 1, 2, 3]
    assert section_label("Pre-Chorus") == "pre-chorus"


def test_parse_lrc():
    lrc = "[ar:Someone]\n[offset:+500]\n[00:10.50]first line\n[00:05.00][01:05.00]repeated line\n[00:12.00]\n"
    rows = parse_lrc(lrc)
    assert [(r.start, r.text) for r in rows] == [
        (4.5, "repeated line"), (10.0, "first line"), (11.5, ""), (64.5, "repeated line")]


def _tok(i, key):
    return WToken(index=i, text=key, key=key, start=i * 0.2, end=i * 0.2 + 0.1, prob=0.9)


def test_hallucination_loops_respect_lyrics():
    toks = [_tok(0, "hello")] + [_tok(i, "la") for i in range(1, 31)] + [_tok(31, "bye")]
    kept, dropped = drop_hallucination_loops(toks, ["hello", "la", "la", "bye"])
    assert [t.key for t in kept] == ["hello", "la", "la", "bye"]
    assert len(dropped) == 28
    # the lyrics themselves contain a long run: nothing is dropped
    kept, dropped = drop_hallucination_loops(toks, ["la"] * 30)
    assert not dropped and len(kept) == 32
