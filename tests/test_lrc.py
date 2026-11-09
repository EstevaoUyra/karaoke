from __future__ import annotations

from karaoke.lyrics.lrc import (
    LrcLine,
    format_lrc,
    format_time,
    lrc_duration,
    parse_lrc,
    parse_lrc_tags,
    parse_time,
    split_stanzas,
)

SAMPLE = """[ar:Invented Band]
[ti:Paper Boats]
[length:03:05.50]
[00:01.00]paper boats along the gutter
[00:04.5]rain is drumming on the roof
[00:08.00]
[00:20.00][01:10.250]float away, float away
[00:23.00]<00:23.00>word <00:23.40>stamps <00:23.90>here
[01:00]no fraction here
"""


def test_parse_tags_and_duration():
    tags = parse_lrc_tags(SAMPLE)
    assert tags == {"ar": "Invented Band", "ti": "Paper Boats", "length": "03:05.50"}
    assert lrc_duration(SAMPLE) == 185.5
    assert lrc_duration("[00:01.00]x") is None


def test_parse_lines_multiple_stamps_and_breaks():
    lines = parse_lrc(SAMPLE)
    assert lines == [
        LrcLine(1.0, "paper boats along the gutter"),
        LrcLine(4.5, "rain is drumming on the roof"),
        LrcLine(8.0, ""),  # instrumental break
        LrcLine(20.0, "float away, float away"),
        LrcLine(23.0, "word stamps here"),
        LrcLine(60.0, "no fraction here"),
        LrcLine(70.25, "float away, float away"),
    ]


def test_offset_tag():
    lines = parse_lrc("[offset:+500]\n[00:10.00]a\n[00:00.20]b\n")
    assert lines == [LrcLine(0.0, "b"), LrcLine(9.5, "a")]


def test_split_stanzas():
    st = split_stanzas(parse_lrc(SAMPLE))
    assert [[ln.text for ln in s] for s in st] == [
        ["paper boats along the gutter", "rain is drumming on the roof"],
        ["float away, float away", "word stamps here", "no fraction here", "float away, float away"],
    ]


def test_format_roundtrip():
    lines = parse_lrc(SAMPLE)
    out = format_lrc(lines, {"ar": "Invented Band", "length": format_time(185.5)})
    assert out.startswith("[ar:Invented Band]\n[length:03:05.50]\n[00:01.00]paper boats")
    assert parse_lrc(out) == lines
    assert format_time(61.239) == "01:01.24"
    assert parse_time("03:40.5") == 220.5
    assert parse_time("220") == 220.0
