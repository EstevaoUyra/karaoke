"""Lyrics that cover only part of a song: what whisper heard where no line is becomes lines; lines
added by hand survive a new alignment; the video opens with the title (invented words, not a song)."""

from __future__ import annotations

import pytest
from test_align import assert_valid, sing, transcript

from karaoke.align.core import align_lyrics, fill_missing
from karaoke.align.fallback import FROM_WHISPER, phrases
from karaoke.align.tokens import WToken, parse_chosen_text
from karaoke.models import (
    JobSettings,
    Line,
    LyricsFinal,
    LyricsSourceName,
    Section,
    Structure,
    Word,
    WordOrigin,
)

WRITTEN = ["the river runs below the town", "we carry lanterns through the rain"]
MISSING = ["silver birds are flying over the mountain", "nobody knows the way back home tonight"]


def _song():
    """Two written lines, then 6 s later two lines the lyrics site does not have."""
    truth = sing(WRITTEN + MISSING, section_breaks={2}, section_gap=6.0)
    return parse_chosen_text("\n".join(WRITTEN)), truth


def test_heard_lines_missing_from_the_lyrics_are_kept():
    lines, truth = _song()
    lf, stats = align_lyrics(lines, transcript(truth), text_source=LyricsSourceName.user)
    assert_valid(lf)
    written = [ln for ln in lf.lines if ln.id.startswith("l")]
    filled = [ln for ln in lf.lines if ln.id.startswith("w")]
    assert [" ".join(w.text for w in ln.words) for ln in written] == WRITTEN
    heard = " ".join(w.text for ln in filled for w in ln.words)
    assert heard == " ".join(MISSING)  # whisper's words (lowercase, no punctuation in this fake)
    assert filled[0].words[0].start == truth[2][0][1]
    assert all(FROM_WHISPER in ln.flags and "low_confidence" in ln.flags for ln in filled)
    assert lf.unmatched_audio == [] and stats.filled_words == sum(len(s.split()) for s in MISSING)


def test_ad_libs_noise_and_doubtful_words_are_not_lines():
    lines, truth = _song()
    flat = [w for r in truth[:2] for w in r]
    end = flat[-1][2]
    adlibs = [("yeah", end + 3.0, end + 3.3), ("yeah", end + 3.4, end + 3.7), ("yeah", end + 3.8, end + 4.1),
              ("yeah", end + 4.2, end + 4.5)]  # repeated: not a passage
    doubtful = [(w, s + end + 8, e + end + 8) for w, s, e in truth[2]]  # a real line, but whisper unsure
    tr = transcript(flat + adlibs + doubtful, probs={len(flat) + len(adlibs) + i: 0.2 for i in range(len(doubtful))})
    lf, stats = align_lyrics(lines, tr, text_source=LyricsSourceName.user)
    assert stats.filled_lines == 0 and [ln.id for ln in lf.lines] == ["l001", "l002"]
    assert len(lf.unmatched_audio) >= 2  # still listed for review


def test_words_next_to_a_line_are_not_filled_over():
    """A backing answer inside a written line's time stays unmatched audio, not a line."""
    lines, truth = _song()
    flat = [w for r in truth[:2] for w in r]
    a, b = flat[0][1], flat[5][2]
    backing = [("oh", a + 0.1, a + 0.3), ("come", a + 0.4, a + 0.6), ("on", a + 0.7, a + 0.9), ("now", b - 0.3, b - 0.1)]
    lf, stats = align_lyrics(lines, transcript(flat + backing), text_source=LyricsSourceName.user)
    assert stats.filled_lines == 0


def _added(lid: str, words: list[tuple[str, float, float]]) -> Line:
    ws = [Word(text=t, start=s, end=e, conf=0.4, origin=WordOrigin.ai) for t, s, e in words]
    return Line(id=lid, start=ws[0].start, end=ws[-1].end, words=ws, flags=["ai_edited"])


def test_lines_added_by_hand_survive_a_new_alignment():
    lines, truth = _song()
    tr = transcript(truth)
    lf, _ = align_lyrics(lines, tr, text_source=LyricsSourceName.user)
    # a reviewer wrote the missing passage in, with the lyric spelling, under ids the lyrics do not have
    added = [_added("l003", [(w.capitalize() if i == 0 else w, s, e) for i, (w, s, e) in enumerate(truth[2])]),
             _added("l004", list(truth[3]))]
    before = LyricsFinal(text_source=LyricsSourceName.user, lines=[ln for ln in lf.lines if ln.id.startswith("l")] + added)
    again, stats = align_lyrics(lines, tr, text_source=LyricsSourceName.user, previous=before)
    assert_valid(again)
    assert stats.lost_edits == [] and stats.kept_lines == 2 and stats.filled_lines == 0
    kept = {ln.id: ln for ln in again.lines}
    assert kept["l003"].words[0].text == "Silver" and kept["l004"] == added[1]


def test_an_edited_whisper_line_is_kept():
    lines, truth = _song()
    tr = transcript(truth)
    lf, _ = align_lyrics(lines, tr, text_source=LyricsSourceName.user)
    w1 = next(ln for ln in lf.lines if ln.id == "w001")
    w1.words[0] = w1.words[0].model_copy(update={"text": "Silver", "origin": WordOrigin.manual})
    again, stats = align_lyrics(lines, tr, text_source=LyricsSourceName.user, previous=lf)
    assert_valid(again)
    assert next(ln for ln in again.lines if ln.id == "w001").words[0].text == "Silver"
    assert stats.kept_lines == 1 and stats.lost_edits == []


def test_fill_missing_leaves_a_reviewed_file_alone():
    lines, truth = _song()
    tr = transcript(truth)
    lf, _ = align_lyrics(lines, transcript([w for r in truth[:2] for w in r]), text_source=LyricsSourceName.user)
    reviewed = lf.model_copy(deep=True)
    w0 = reviewed.lines[0].words[0]
    reviewed.lines[0].words[0] = w0.model_copy(update={"start": w0.start + 0.05})
    out, n = fill_missing(reviewed, tr)
    assert n >= 1 and out.lines[:2] == reviewed.lines[:2]
    assert all(ln.id.startswith("w") and FROM_WHISPER in ln.flags for ln in out.lines[2:])
    assert fill_missing(out, tr)[1] == 0  # nothing left to fill


def _tok(i: int, text: str, s: float) -> WToken:
    return WToken(index=i, text=text, key=text.lower(), start=s, end=s + 0.3, prob=0.9, segment=0)


def test_names_stay_on_one_line():
    words = "a conexão de São Paulo com a Paraíba".split()
    toks = [_tok(i, w, i * 0.4) for i, w in enumerate(words)]
    assert [[t.text for t in p] for p in phrases(toks)] == [words]
    # a capital after a comma or a full stop still starts a line
    toks = [_tok(0, "fogo,", 0), _tok(1, "Eu", 0.4), _tok(2, "quero", 0.8)]
    assert len(phrases(toks)) == 2


def test_a_word_is_not_moved_far_across_a_long_unpitched_stretch():
    import numpy as np

    from karaoke.align.voicing import Envelope, retime_across_pauses

    hop = 0.01
    db = np.full(2000, -20.0)  # loud all along (speech), 20 s
    voiced = np.zeros(2000)
    voiced[:300] = 1.0  # pitched singing up to 3 s, then speech with no pitch until the end
    env = Envelope(db=db, hop_s=hop, voiced_prob=voiced, pitch_hop_s=hop)
    times = [[2.5, 2.9], [3.3, 3.7]]  # the second word is spoken, inside the "pause"
    retime_across_pauses(env, times, texts=["mais", "cara"])
    assert times[1][0] == pytest.approx(3.3)


def test_the_video_opens_with_the_title_when_singing_starts_early():
    from karaoke.render.ass import INTRO_MIN_S, lead_pad_s, shift_times
    from karaoke.youtube import chapters

    rs = JobSettings().render
    lf = LyricsFinal(text_source=LyricsSourceName.user, lines=[_added("l001", [("go", 0.8, 1.2)])])
    assert lead_pad_s(lf, rs) == pytest.approx(INTRO_MIN_S - 0.8)
    late = shift_times(lf, 10.0)
    assert lead_pad_s(late, rs) == 0.0 and late.lines[0].words[0].start == pytest.approx(10.8)
    assert lead_pad_s(lf, rs.model_copy(update={"show_title_card": False})) == 0.0
    st = Structure(duration=60, sections=[Section(start=0, end=20, label="verse", energy=0.5, vocal=1),
                                          Section(start=20, end=40, label="chorus", energy=0.9, vocal=1),
                                          Section(start=40, end=60, label="verse", energy=0.5, vocal=1)])
    assert [s for s, _ in chapters(st, "en", 5.2)] == [0.0, 25.2, 45.2]
