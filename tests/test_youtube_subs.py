from __future__ import annotations

from pathlib import Path

import pytest

from karaoke.lyrics.youtube_subs import Cue, cues_to_text, parse_vtt, parse_vtt_text

# Invented lyrics only.
MANUAL = """﻿WEBVTT
Kind: captions
Language: en

NOTE this block is a comment
and spans two lines

STYLE
::cue { color: yellow }

1
00:00:01.000 --> 00:00:03.500 align:middle line:90%
♪ Paper <i>boats</i> on the river ♪

2
00:00:03.500 --> 00:00:06.250
<v Singer>Sailing &amp; sinking</v>
all the way home

00:00:06.250 --> 00:00:07.000
♪♪

00:00:07.000 --> 00:00:09.000
[Music]

00:01:09.000 --> 00:01:10.500
Hey hey

00:01:10.500 --> 00:01:12.000
Hey hey

00:01:12.000 --> 00:01:13.000
(hey hey)
"""

# Auto-caption style: rolling two-line window with inline word timestamps and 10 ms filler cues.
ROLLING = """WEBVTT
Kind: captions
Language: pt

00:00:01.000 --> 00:00:03.000 align:start position:0%
{SP}
olha<00:00:01.400><c> o</c><00:00:01.600><c> barco</c>

00:00:03.000 --> 00:00:03.010 align:start position:0%
olha o barco
{SP}

00:00:03.010 --> 00:00:05.000 align:start position:0%
olha o barco
no<00:00:03.500><c> rio</c><00:00:04.000><c> azul</c>

00:00:05.000 --> 00:00:05.010 align:start position:0%
no rio azul
{SP}

00:00:05.010 --> 00:00:07.000 align:start position:0%
no rio azul
<c.colorE5E5E5>[Música]</c>

00:00:07.000 --> 00:00:09.000 align:start position:0%
[Música]
vai<00:00:07.500><c> e</c><00:00:08.000><c> volta</c>
""".replace("{SP}", " ")  # whitespace-only lines, as YouTube writes them


def test_manual_vtt():
    cues = parse_vtt_text(MANUAL)
    assert cues == [
        Cue(1.0, 3.5, "Paper boats on the river"),
        Cue(3.5, 6.25, "Sailing & sinking\nall the way home"),
        Cue(69.0, 70.5, "Hey hey"),
        Cue(70.5, 72.0, "Hey hey"),  # manual subs keep intentional repeats
        Cue(72.0, 73.0, "(hey hey)"),  # bracketed lyrics are kept, only sound tags are dropped
    ]
    assert cues[1].lines == ["Sailing & sinking", "all the way home"]


def test_rolling_auto_vtt_is_deduplicated():
    cues = parse_vtt_text(ROLLING)
    assert cues == [
        Cue(1.0, 3.0, "olha o barco"),
        Cue(3.01, 5.0, "no rio azul"),
        Cue(7.0, 9.0, "vai e volta"),
    ]
    assert cues_to_text(cues) == "olha o barco\nno rio azul\nvai e volta"


def test_timestamps_without_hours_and_crlf(tmp_path: Path):
    p = tmp_path / "x.vtt"
    p.write_bytes(b"WEBVTT\r\n\r\n01:02.5 --> 01:04.250\r\nLinha um\r\n\r\n01:04.250 --> 01:04.250\r\n \r\n")
    assert parse_vtt(p) == [Cue(62.5, 64.25, "Linha um")]


def test_empty_file():
    assert parse_vtt_text("WEBVTT\n\n") == []


_REAL = Path(__file__).resolve().parents[1] / "data" / "jobs" / "Zdh2hot8rjU" / "source.en.vtt"


@pytest.mark.slow
@pytest.mark.skipif(not _REAL.exists(), reason="real job data not present")
def test_real_manual_subs_parse():
    cues = parse_vtt(_REAL)
    assert len(cues) >= 30
    assert all(c.text.strip() and c.end >= c.start for c in cues)
    assert all(a.start <= b.start for a, b in zip(cues, cues[1:], strict=False))
    assert not any("<" in c.text or "♪" in c.text for c in cues)
