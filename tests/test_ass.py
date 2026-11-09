from __future__ import annotations

import re

import make_demo
import pytest

from karaoke.models import Line, LyricsFinal, LyricsSourceName, Meta, RenderSettings, Word
from karaoke.render.ass import (
    ass_bgr,
    ass_time,
    build_ass,
    build_events,
    escape_text,
    hex_to_ass,
    karaoke_text,
)

RES = (1280, 720)
K_TAG = re.compile(r"\\(kf|k)(\d+)")
POS = re.compile(r"\\pos\((\d+),(\d+)\)")


def meta(duration: float = 24.0) -> Meta:
    return Meta(title="Demo Band - Sing It Loud (Official Video)", duration=duration,
                clean_title="Sing It Loud", clean_artist="Demo Band")


def demo_events(rs: RenderSettings | None = None):
    return build_events(make_demo.truth_lyrics(), meta(), rs or RenderSettings(), RES)


def shifted(lf: LyricsFinal, dt: float) -> LyricsFinal:
    lf = lf.model_copy(deep=True)
    for ln in lf.lines:
        ln.start += dt
        ln.end += dt
        for w in ln.words:
            w.start += dt
            w.end += dt
    return lf


# --------------------------------------------------------------------------- primitives


def test_hex_to_ass_colour_conversion():
    assert hex_to_ass("#4FC3F7") == "&H00F7C34F"  # RR GG BB → BB GG RR
    assert hex_to_ass("#FF0000") == "&H000000FF"
    assert hex_to_ass("0000ff") == "&H00FF0000"
    assert hex_to_ass("#4fc3f7", alpha=0x80) == "&H80F7C34F"
    assert hex_to_ass("#FFFFFF00") == "&HFFFFFFFF"  # CSS alpha 00 (transparent) → ASS FF
    assert hex_to_ass("#FFFFFFFF") == "&H00FFFFFF"
    assert ass_bgr("#4FC3F7") == "&HF7C34F&"
    with pytest.raises(ValueError):
        hex_to_ass("blue")


def test_ass_time():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(6150) == "0:01:01.50"
    assert ass_time(360000 + 5) == "1:00:00.05"


def test_escape_text():
    out = escape_text("a{b}c\\Nd\ne")
    assert "{" not in out and "}" not in out
    assert "\\N" not in out  # the backslash can no longer start an escape
    assert "\n" not in out


def test_kf_covers_words_and_folds_short_gaps():
    line = Line(id="x", start=10.0, end=13.0, words=[
        Word(text="one", start=10.00, end=10.40),
        Word(text="two", start=10.50, end=10.90),  # 0.1 s gap → folded into "one"
        Word(text="three", start=12.00, end=13.00),  # 1.1 s gap → its own \k spacer
    ])
    text, durs = karaoke_text(line, start_cs=850)
    tags = K_TAG.findall(text)
    assert tags[0] == ("k", "150")  # lead-in spacer: fill starts exactly on the first word
    assert tags[1:] == [("kf", "50"), ("kf", "40"), ("k", "110"), ("kf", "100")]
    assert sum(durs) == 300


# --------------------------------------------------------------------------- demo song


def test_demo_kf_sums_match_line_durations():
    lf = make_demo.truth_lyrics()
    lines = [e for e in demo_events() if e.name == "line"]
    assert len(lines) == len(lf.lines)
    for ev, ln in zip(lines, lf.lines, strict=True):
        tags = [(k, int(v)) for k, v in K_TAG.findall(ev.text)]
        assert tags[0][0] == "k"
        lead, body = tags[0][1], tags[1:]
        assert ev.start + lead == pytest.approx(round(ln.words[0].start * 100), abs=1)
        assert sum(v for _, v in body) == pytest.approx((ln.words[-1].end - ln.words[0].start) * 100, abs=1)
        assert sum(1 for k, _ in body if k == "kf") == len(ln.words)
        assert ev.end >= round(ln.end * 100)


def test_lead_in_and_two_rows():
    rs = RenderSettings()
    events = demo_events(rs)
    lines = [e for e in events if e.name == "line"]
    # every line becomes active lead_in_s before its first word (line 1 has 2 s of intro)
    lf = make_demo.truth_lyrics()
    for ev, ln in zip(lines, lf.lines, strict=True):
        assert ev.start == round((ln.words[0].start - rs.lead_in_s) * 100)
    rows = [int(POS.search(e.text).group(2)) for e in lines]
    # alternating rows; the second chorus (after the gap) restarts on the top row
    assert rows[0] < rows[1] and rows[2] == rows[0] and rows[3] == rows[1] and rows[4] == rows[0]
    # within a row, events never overlap in time
    lyric = [e for e in events if e.name in ("line", "next")]
    for y in set(rows):
        evs = sorted((e for e in lyric if int(POS.search(e.text).group(2)) == y), key=lambda e: e.start)
        for a, b in zip(evs, evs[1:], strict=False):
            assert b.start >= a.end, (a, b)


def test_next_line_is_dimmed_then_becomes_active_in_place():
    events = demo_events()
    nxt = [e for e in events if e.name == "next"]
    lines = [e for e in events if e.name == "line"]
    # line 2 is shown dimmed while line 1 is sung, then becomes active at the same position
    pre = next(e for e in nxt if "coffee" in e.text)
    act = next(e for e in lines if "coffee" in e.text)
    assert pre.end == act.start
    assert POS.search(pre.text).group(0) == POS.search(act.text).group(0)
    assert "\\1a&H70&" in pre.text and "\\kf" not in pre.text
    assert ass_bgr(RenderSettings().color_unsung) in pre.text


def test_countdown_dots_before_every_pause():
    events = demo_events()
    dots = sorted((e for e in events if e.name == "countdown"), key=lambda e: e.end)
    # the 2 s intro (beat squeezed to fit) and chorus 2 after a 5.1 s gap (1 s beats)
    assert len(dots) == 6
    assert [e.end for e in dots[:3]][-1] == 200  # the last intro dot goes out on the first word
    assert [e.end for e in dots[3:]] == [1550, 1650, 1750]
    assert all(e.start == 1425 for e in dots[3:])  # 3 beats before, plus the fade-in
    assert all("\\p1" in e.text for e in dots)
    # the song's beat paces them
    lf = make_demo.truth_lyrics()
    paced = [e for e in build_events(lf, meta(), RenderSettings(), RES, beat_s=0.5) if e.name == "countdown"]
    assert sorted(e.end for e in paced)[3:] == [1650, 1700, 1750]


def test_pause_inside_a_line_holds_the_fill_and_shows_dots():
    words = [Word(text="hold", start=5.0, end=5.5), Word(text="on", start=5.6, end=6.0),
             Word(text="then", start=9.5, end=9.9), Word(text="go", start=10.0, end=10.5)]
    lf = LyricsFinal(text_source=LyricsSourceName.user,
                     lines=[Line(id="l1", start=5.0, end=10.5, words=words)])
    events = build_events(lf, meta(), RenderSettings(), RES)
    act = next(e for e in events if e.name == "line")
    tags = [(k, int(v)) for k, v in K_TAG.findall(act.text)]
    assert ("k", 350) in tags  # the fill waits 3.5 s instead of crawling over "on"
    dots = sorted(e.end for e in events if e.name == "countdown")
    assert dots[-3:] == [750, 850, 950]


def test_title_card_only_with_a_long_intro():
    rs = RenderSettings()
    # demo: first word at 2.0 s (< 3 s) → no title card
    assert not [e for e in demo_events(rs) if e.name == "title"]
    lf = shifted(make_demo.truth_lyrics(), 6.0)  # first word at 8.0 s
    events = build_events(lf, meta(30.0), rs, RES)
    titles = [e for e in events if e.name == "title"]
    assert len(titles) == 1
    t = titles[0]
    assert t.start == 0 and "Sing It Loud" in t.text and "Demo Band" in t.text
    first_active = min(e.start for e in events if e.name == "line")
    assert t.end < first_active and "\\fad(" in t.text
    # the first line after an 8 s intro gets a countdown too
    assert sorted(e.end for e in events if e.name == "countdown")[:3] == [600, 700, 800]
    assert not [e for e in build_events(lf, meta(30.0), RenderSettings(show_title_card=False), RES)
                if e.name == "title"]


def test_document_header_scales_with_resolution():
    rs = RenderSettings()
    doc720 = build_ass(make_demo.truth_lyrics(), meta(), rs, RES)
    doc1080 = build_ass(make_demo.truth_lyrics(), meta(), rs, (1920, 1080))
    assert "PlayResX: 1280\nPlayResY: 720" in doc720
    assert "Style: Karaoke,Arial Rounded MT Bold,42.7,&H00F7C34F,&H00FFFFFF,&H00000000" in doc720
    assert "Style: Karaoke,Arial Rounded MT Bold,64.0," in doc1080
    assert doc720.count("\nDialogue:") == len(demo_events())


def test_braces_are_escaped_and_long_lines_split_without_shrinking():
    words = [Word(text="{sneaky}", start=2.0, end=2.5)] + [
        Word(text=f"wordnumber{i}", start=2.5 + i * 0.3, end=2.8 + i * 0.3) for i in range(12)
    ]
    lf = LyricsFinal(text_source=LyricsSourceName.user,
                     lines=[Line(id="l1", start=2.0, end=words[-1].end, words=words)])
    events = build_events(lf, meta(), RenderSettings(), RES)
    acts = [e for e in events if e.name == "line"]
    # only our own override blocks remain: every "{" opens a tag block
    assert all(re.sub(r"\{\\[^}]*\}", "", a.text).count("{") == 0 for a in acts)
    assert "\uff5bsneaky\uff5d" in acts[0].text
    assert len(acts) >= 2  # too wide for one 1280 px row: split, never shrunk
    assert not any("\\fs" in e.text for e in events if e.name in ("line", "next"))
    assert sum(len(K_TAG.findall(a.text)) - 1 for a in acts) == len(words)  # every word once
    assert len({POS.search(a.text).group(2) for a in acts}) >= 2


def test_long_line_splits_at_the_parenthesis():
    texts = "Sweet child of mine (sweet, sweet, sweet love of mine)".split()
    words = [Word(text=t, start=2.0 + 0.4 * i, end=2.35 + 0.4 * i) for i, t in enumerate(texts)]
    lf = LyricsFinal(text_source=LyricsSourceName.user,
                     lines=[Line(id="l1", start=2.0, end=words[-1].end, words=words)])
    acts = [e for e in build_events(lf, meta(), RenderSettings(), RES) if e.name == "line"]
    plain = [re.sub(r"\{[^}]*\}", "", a.text).strip() for a in acts]
    assert plain == ["Sweet child of mine", "(sweet, sweet, sweet love of mine)"]


def _fast_song(n: int = 12, pitch: float = 1.4) -> LyricsFinal:
    lines = []
    for i in range(n):
        t = 3.0 + i * pitch
        ws = [Word(text=f"w{i}a", start=t, end=t + 0.5), Word(text=f"w{i}b", start=t + 0.6, end=t + 1.2)]
        lines.append(Line(id=f"l{i}", start=t, end=t + 1.2, words=ws))
    return LyricsFinal(text_source=LyricsSourceName.user, lines=lines)


@pytest.mark.parametrize(("pitch", "rows"), [(1.4, 3), (2.6, 2)])
def test_fast_blocks_use_three_rows(pitch, rows):
    events = build_events(_fast_song(pitch=pitch), meta(60.0), RenderSettings(), RES)
    lines = [e for e in events if e.name == "line"]
    ys = [int(POS.search(e.text).group(2)) for e in lines]
    assert len(set(ys)) == rows
    assert ys[:rows] == sorted(ys[:rows])  # top to bottom, then around again
    assert ys[rows] == ys[0]
    # within a row, events never overlap in time, and each line is up before its first word
    lyric = [e for e in events if e.name in ("line", "next")]
    for y in set(ys):
        evs = sorted((e for e in lyric if int(POS.search(e.text).group(2)) == y), key=lambda e: e.start)
        for a, b in zip(evs, evs[1:], strict=False):
            assert b.start >= a.end, (a, b)
    for ev, ln in zip(lines, _fast_song(pitch=pitch).lines, strict=True):
        assert ev.start < round(ln.words[0].start * 100)


def test_empty_lyrics_gives_title_only():
    lf = LyricsFinal(text_source=LyricsSourceName.whisper, lines=[])
    events = build_events(lf, meta(), RenderSettings(), RES)
    assert [e.name for e in events] == ["title"]


def test_rows_follow_the_pace_inside_one_block():
    # slow lines (3 s apart), then fast ones (1.2 s apart), with no gap between them
    lines, t = [], 3.0
    for i in range(14):
        ws = [Word(text=f"w{i}a", start=t, end=t + 0.4), Word(text=f"w{i}b", start=t + 0.5, end=t + 1.0)]
        lines.append(Line(id=f"l{i}", start=t, end=t + 1.0, words=ws))
        t += 3.0 if i < 6 else 1.2
    lf = LyricsFinal(text_source=LyricsSourceName.user, lines=lines)
    events = build_events(lf, meta(60.0), RenderSettings(), RES)
    ys = [int(POS.search(e.text).group(2)) for e in events if e.name == "line"]
    assert len(set(ys[:5])) == 2 and len(set(ys[-6:])) == 3
    lyric = [e for e in events if e.name in ("line", "next")]
    for y in set(ys):
        evs = sorted((e for e in lyric if int(POS.search(e.text).group(2)) == y), key=lambda e: e.start)
        for a, b in zip(evs, evs[1:], strict=False):
            assert b.start >= a.end, (a, b)
    # never more than three lyric lines on screen
    for e in lyric:
        assert sum(1 for o in lyric if o.start <= e.start < o.end) <= 3
