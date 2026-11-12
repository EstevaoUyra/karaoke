from __future__ import annotations

import make_demo
import pytest

from karaoke.lyrics.normalize import (
    flatten,
    line_key,
    match_tokens,
    normalize_word,
    parse_lyrics_text,
    to_chosen_text,
)

# All lyrics below are invented.


def test_demo_plain_text_expands_to_demo_chosen_txt(demo_no_audio):
    p = demo_no_audio
    stanzas = parse_lyrics_text(p.lyrics_source("lrclib.txt").read_text())
    assert [s.section for s in stanzas] == ["verse", "chorus", "chorus"]
    assert to_chosen_text(stanzas) == p.lyrics_chosen_text.read_text()


def test_chosen_txt_roundtrip(demo_no_audio):
    text = demo_no_audio.lyrics_chosen_text.read_text()
    assert to_chosen_text(parse_lyrics_text(text)) == text
    assert len(flatten(parse_lyrics_text(text))) == len(make_demo.TRUTH)


def test_headers_pt_en_and_colon_style():
    text = "[Verse 1]\na\n\n[Refrão]\nb\n\n[Ponte]\nc\n\nChorus:\nd\n\n[Pre-Chorus: Somebody]\ne\n\nIntro:\nf\n"
    st = parse_lyrics_text(text)
    assert [(s.section, s.lines) for s in st] == [
        ("verse", ["a"]), ("chorus", ["b"]), ("bridge", ["c"]), ("chorus", ["d"]), ("pre-chorus", ["e"]),
        ("intro", ["f"]),
    ]
    assert st[0].label == "verse 1"
    assert st[4].singer == "Somebody"


def test_unlabeled_and_lyric_lines_with_colons_are_not_headers():
    st = parse_lyrics_text("she said: come home\nthe night is long\n\nsecond block\n")
    assert [(s.section, s.lines) for s in st] == [(None, ["she said: come home", "the night is long"]),
                                                  (None, ["second block"])]


@pytest.mark.parametrize("header", ["[Chorus x2]", "[Refrão 2x]", "[Chorus (x2)]", "[Refrão bis]"])
def test_header_count_over_lines(header):
    st = parse_lyrics_text(f"{header}\nla la\nlo lo\n")
    assert [s.section for s in st] == ["chorus", "chorus"]
    assert flatten(st) == [("chorus", "la la"), ("chorus", "lo lo")] * 2


@pytest.mark.parametrize("bare, total", [("[Chorus]", 2), ("[Chorus x1]", 2), ("[Chorus x2]", 3),
                                         ("[Repeat Chorus]", 2), ("Refrão:", 2), ("[Refrão 2x]", 3)])
def test_bare_header_repeats_earlier_section(bare, total):
    text = f"[Chorus]\nsing out\nsing loud\n\n[Verse 2]\nnew words\n\n{bare}\n"
    st = parse_lyrics_text(text)
    choruses = [s for s in st if s.section == "chorus"]
    assert len(choruses) == total
    assert all(s.lines == ["sing out", "sing loud"] for s in choruses)
    assert st[-1].section == "chorus"


def test_bare_numbered_header_does_not_repeat_other_number():
    st = parse_lyrics_text("[Verse 1]\nfirst\n\n[Verse 2]\n\n[Chorus]\nhey\n")
    assert flatten(st) == [("verse", "first"), ("chorus", "hey")]


@pytest.mark.parametrize("marker, n", [("(x2)", 2), ("(2x)", 2), ("(repeat)", 2), ("(x3)", 3), ("x2", 2),
                                       ("[x2]", 2), ("(bis)", 2)])
def test_line_end_markers(marker, n):
    st = parse_lyrics_text(f"oh the river {marker}\nand the sea\n")
    assert st[0].lines == ["oh the river"] * n + ["and the sea"]


def test_standalone_marker_repeats_stanza():
    st = parse_lyrics_text("one\ntwo\n(x2)\n\nthree\n")
    assert flatten(st) == [(None, "one"), (None, "two"), (None, "one"), (None, "two"), (None, "three")]
    st = parse_lyrics_text("[Chorus]\nhey\n\n(repeat)\n")
    assert flatten(st) == [("chorus", "hey"), ("chorus", "hey")]


def test_unknown_bracket_lines_are_dropped_not_sung():
    st = parse_lyrics_text("[Instrumental]\n\n[Verse]\nhello there\n")
    assert flatten(st) == [("verse", "hello there")]


def test_to_chosen_text_layout():
    st = parse_lyrics_text("a\nb\n\nc\n\n[Chorus]\nd\n\n[Chorus]\ne\n\nf\n")
    assert to_chosen_text(st) == "a\nb\n\nc\n\n[chorus]\nd\ne\n\nf\n"
    assert to_chosen_text([]) == ""


def test_match_tokens_basic():
    assert match_tokens("Coração, VOCÊ não-sabe!") == ["coracao", "voce", "nao", "sabe"]
    assert match_tokens("nobody's listening — anymore...") == ["nobodys", "listening", "anymore"]
    assert normalize_word("...") == []


@pytest.mark.parametrize("informal, formal", [
    ("pra", "para"), ("pro", "para o"), ("tá", "está"), ("tô", "estou"), ("cê", "você"), ("né", "não é"),
    ("gonna", "going to"), ("wanna", "want to"), ("'cause", "because"), ("’til", "until"), ("o'", "of"),
    ("ya", "you"), ("lovin'", "loving"),
])
def test_equivalences(informal, formal):
    assert match_tokens(informal) == match_tokens(formal)
    assert match_tokens(informal, expand=False) != match_tokens(formal) or informal == formal


def test_portuguese_to_is_not_english_to():
    assert match_tokens("tô") == ["estou"]
    assert match_tokens("to") == ["to"]
    assert line_key("Sweet Child O' Mine") == "sweet child of mine"


def test_title_cased_lyrics_become_sentence_case():
    from karaoke.lyrics.normalize import is_title_cased, parse_lyrics_text, sentence_case

    text = ("[Refrão]\nQuero Um Som Que Me Agrada\nToca De Noite E De Dia\n"
            "E Ninguém Se Incomoda, Ninguém Se Incomoda.\nPra Ve-la No SOM, KayaMan Vem. Agita\n")
    lines = [ln for st in parse_lyrics_text(text) for ln in st.lines]
    assert lines == ["Quero um som que me agrada", "Toca de noite e de dia",
                     "E ninguém se incomoda, ninguém se incomoda.", "Pra ve-la no SOM, KayaMan vem. Agita"]
    assert sentence_case("And I Said I'm Here") == "And I said I'm here"
    # ordinary lyrics with a few names keep them
    normal = ["Eu vi a Babilônia cair", "I saw the light in Rio de Janeiro", "e Jah me guia no caminho",
              "a gente canta junto"]
    assert not is_title_cased(normal)
    assert [ln for st in parse_lyrics_text("\n".join(normal)) for ln in st.lines] == normal
