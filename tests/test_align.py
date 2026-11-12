"""Alignment tests: the demo job end to end, plus synthetic perturbations of invented lyrics."""

from __future__ import annotations

import random
import time

import make_demo
import pytest

from karaoke.align import dp
from karaoke.align.core import align_lyrics
from karaoke.align.fallback import FROM_WHISPER
from karaoke.align.tokens import LrcLine, match_key, parse_chosen_text
from karaoke.models import (
    LyricsChoice,
    LyricsFinal,
    LyricsSourceName,
    StageName,
    Transcript,
    TSegment,
    TWord,
    WordOrigin,
)
from karaoke.pipeline import run_job
from karaoke.store import JobStore, read_model, write_model
from karaoke.validate import validate_lyrics

HEARD = (WordOrigin.matched, WordOrigin.fuzzy)

# --------------------------------------------------------------------------- invented song

VERSE1 = ["Paper boats along the river", "carry letters I never sent"]
CHORUS = ["Hold the lantern higher now", "let the harbor see us glow"]
VERSE2 = ["Gravel roads and borrowed engines", "humming tunes we used to know"]
BRIDGE = ["Somewhere past the silver hills", "a quiet morning waits for me"]


def song_text(*sections: tuple[str, list[str]]) -> str:
    out = []
    for name, lines in sections:
        out.append(f"[{name}]")
        out += lines
        out.append("")
    return "\n".join(out)


def sing(lines: list[str], start: float = 1.0, line_gap: float = 0.6, word_gap: float = 0.05,
         section_breaks: set[int] | None = None, section_gap: float = 4.0) -> list[list[tuple[str, float, float]]]:
    """Ground-truth timing: each word lasts 0.15 + 0.05/letter."""
    t = start
    out = []
    for li, line in enumerate(lines):
        if li and section_breaks and li in section_breaks:
            t += section_gap
        row = []
        for w in line.split():
            d = 0.15 + 0.05 * len(match_key(w))
            row.append((w, round(t, 3), round(t + d, 3)))
            t += d + word_gap
        out.append(row)
        t += line_gap
    return out


def transcript(rows: list[list[tuple[str, float, float]]] | list[tuple[str, float, float]], prob: float = 0.9,
               probs: dict[int, float] | None = None) -> Transcript:
    """Whisper-like transcript: lowercase words without punctuation, segments split at 1 s pauses."""
    flat = [w for r in rows for w in r] if rows and isinstance(rows[0], list) else list(rows)
    flat = sorted(flat, key=lambda w: w[1])
    segs: list[list[TWord]] = []
    for i, (w, s, e) in enumerate(flat):
        tw = TWord(text=" " + w.strip(",.!?").lower(), start=s, end=e, prob=(probs or {}).get(i, prob))
        if segs and s - segs[-1][-1].end <= 1.0:
            segs[-1].append(tw)
        else:
            segs.append([tw])
    return Transcript(model="test", language="en", segments=[
        TSegment(start=g[0].start, end=g[-1].end, text="".join(w.text for w in g), words=g) for g in segs])


def lrc_for(rows: list[list[tuple[str, float, float]]]) -> list[LrcLine]:
    return [LrcLine(r[0][1], " ".join(w for w, _, _ in r)) for r in rows]


def flat_words(lf: LyricsFinal):
    return [w for ln in lf.lines for w in ln.words]


def assert_valid(lf: LyricsFinal, duration: float | None = None) -> None:
    assert [str(i) for i in validate_lyrics(lf, duration)] == []
    for ln in lf.lines:
        for a, b in zip(ln.words, ln.words[1:], strict=False):
            assert b.start >= a.end - 1e-6, f"overlap in {ln.id}"
        for w in ln.words:
            assert w.end - w.start >= 0.08 - 1e-6


def start_errors(lf: LyricsFinal, truth: list[list[tuple[str, float, float]]]):
    out = []
    for ln, row in zip(lf.lines, truth, strict=True):
        assert [w.text for w in ln.words] == [w for w, _, _ in row]
        for w, (_, s, _) in zip(ln.words, row, strict=True):
            out.append((w, w.start - s))
    return out


# --------------------------------------------------------------------------- demo job


@pytest.mark.parametrize("use_line_timings", [True, False])
def test_demo_job(demo_no_audio, jobs_dir, use_line_timings):
    p = demo_no_audio
    p.lyrics_final.unlink()
    choice = read_model(p.lyrics_choice, LyricsChoice)
    choice.use_line_timings = use_line_timings
    write_model(p.lyrics_choice, choice)

    job = run_job("demo", only=[StageName.align], store=JobStore(jobs_dir))
    assert job.stages[StageName.align].status.value == "done"
    lf = read_model(p.lyrics_final, LyricsFinal)
    assert_valid(lf, make_demo.DURATION)
    assert [ln.id for ln in lf.lines] == [f"l{i:03d}" for i in range(1, 7)]
    assert [ln.section for ln in lf.lines] == [s for s, _ in make_demo.TRUTH]

    errs = []
    for li, (ln, (_, truth)) in enumerate(zip(lf.lines, make_demo.TRUTH, strict=True)):
        assert [w.text for w in ln.words] == [w for w, _, _ in truth]
        for wi, (w, (_, s, _)) in enumerate(zip(ln.words, truth, strict=True)):
            if (li, wi) == (4, 0):  # the dropped first "Sing" of chorus 2
                assert w.origin == WordOrigin.interpolated
                assert abs(w.start - s) < 0.4
            elif w.origin in HEARD:
                errs.append(abs(w.start - s))
                assert abs(w.start - s) < 0.15, (ln.id, w.text, w.start, s)
            else:  # "loud" heard as "out": not a match, but whisper's timing is used
                assert (li, wi) == (2, 5) and abs(w.start - s) < 0.15
    assert len(errs) == 27
    assert [u.text for u in lf.unmatched_audio] == ["yeah"]
    assert lf.unmatched_audio[0].start == pytest.approx(make_demo.ADLIB[1])
    assert lf.text_source == LyricsSourceName.lrclib and lf.language == "en"


# --------------------------------------------------------------------------- perturbations


def _song():
    lines = VERSE1 + CHORUS + VERSE2 + CHORUS + BRIDGE + CHORUS
    text = song_text(("verse", VERSE1), ("chorus", CHORUS), ("verse", VERSE2), ("chorus", CHORUS),
                     ("bridge", BRIDGE), ("chorus", CHORUS))
    truth = sing(lines, section_breaks={2, 4, 6, 8, 10})
    return parse_chosen_text(text), truth


@pytest.mark.parametrize("use_line_timings", [False, True])
def test_perfect_transcript(use_line_timings):
    lines, truth = _song()
    lf, stats = align_lyrics(lines, transcript(truth), lrc=lrc_for(truth), use_line_timings=use_line_timings)
    assert_valid(lf)
    for w, err in start_errors(lf, truth):
        assert w.origin == WordOrigin.matched and abs(err) < 1e-6
    assert lf.unmatched_audio == [] and all(not ln.flags for ln in lf.lines)
    assert stats.matched == stats.words


@pytest.mark.parametrize("shift", [1.5, 6.0])
def test_global_shift_against_lrc(shift):
    """The transcript is offset from the LRC (another edit of the same recording)."""
    lines, truth = _song()
    heard = [[(w, s + shift, e + shift) for w, s, e in r] for r in truth]
    lf, stats = align_lyrics(lines, transcript(heard), lrc=lrc_for(truth), use_line_timings=True)
    assert_valid(lf)
    assert stats.lrc_offset == pytest.approx(shift, abs=0.05)
    for w, err in start_errors(lf, heard):
        assert w.origin == WordOrigin.matched and abs(err) < 1e-6
    assert not any("timing_suspect" in ln.flags for ln in lf.lines)


@pytest.mark.parametrize("use_line_timings", [False, True])
def test_twenty_percent_deleted(use_line_timings):
    lines, truth = _song()
    rng = random.Random(3)
    flat = [w for r in truth for w in r]
    drop = set(rng.sample(range(len(flat)), len(flat) // 5))
    heard = [w for i, w in enumerate(flat) if i not in drop]
    lf, stats = align_lyrics(lines, transcript(heard), lrc=lrc_for(truth), use_line_timings=use_line_timings)
    assert_valid(lf)
    errs = start_errors(lf, truth)
    assert stats.matched == len(heard)
    for w, err in errs:
        if w.origin in HEARD:
            assert abs(err) < 1e-6
    interp = [abs(e) for w, e in errs if w.origin == WordOrigin.interpolated]
    assert len(interp) == len(drop)
    assert max(interp) < 0.6 and sum(interp) / len(interp) < 0.25
    assert lf.unmatched_audio == []


def test_adlib_insertions():
    lines, truth = _song()
    flat = [w for r in truth for w in r]
    # in the instrumental gaps, at a line end, and one sung over a line (backing vocal)
    adlibs = [("yeah", 7.0, 7.4), ("oh", 16.0, 16.4), ("baby", 16.5, 16.9), ("come", 26.0, 26.3), ("on", 26.35, 26.6),
              ("hey", 41.0, 41.2)]
    lf, stats = align_lyrics(lines, transcript(flat + adlibs))
    assert_valid(lf)
    for w, err in start_errors(lf, truth):
        assert w.origin == WordOrigin.matched and abs(err) < 1e-6
    assert [(u.text, u.start) for u in lf.unmatched_audio] == [
        ("yeah", 7.0), ("oh baby", 16.0), ("come on", 26.0), ("hey", 41.0)]


def _sections_song(lyric_sections: list[str], sung_sections: list[str]):
    parts = {"V1": ("verse", VERSE1), "V2": ("verse", VERSE2), "C": ("chorus", CHORUS), "B": ("bridge", BRIDGE)}
    lines = parse_chosen_text(song_text(*(parts[s] for s in lyric_sections)))
    sung = [ln for s in sung_sections for ln in parts[s][1]]
    breaks = {2 * k for k in range(1, len(sung_sections))}
    truth = sing(sung, section_breaks=breaks)
    blocks = [truth[2 * k:2 * k + 2] for k in range(len(sung_sections))]
    return lines, truth, blocks


@pytest.mark.parametrize("sung", [["V1", "C", "V2", "C", "C"], ["V1", "C", "C", "V2", "C"]])
def test_chorus_sung_more_often_than_written(sung):
    lines, truth, blocks = _sections_song(["V1", "C", "V2", "C"], sung)
    lf, stats = align_lyrics(lines, transcript(truth))
    assert_valid(lf)
    words = flat_words(lf)
    assert all(w.origin == WordOrigin.matched for w in words)
    written = [ln for ln in lf.lines if ln.id.startswith("l")]
    # every lyric line maps onto exactly one sung copy, word for word
    copies = [b for b in blocks]
    for k, ln in enumerate(written):
        starts = [w.start for w in ln.words]
        cands = [[s for _, s, _ in r] for b in copies for r in b if " ".join(w for w, _, _ in r) == lines[k].text]
        assert starts in cands, ln.id
    verse2 = written[4]
    assert verse2.start == blocks[sung.index("V2")][0][0][1]
    assert all(not ln.flags for ln in written)
    # the chorus sung once more than written is not dropped: it becomes lines from what whisper heard
    filled = [ln for ln in lf.lines if ln.id.startswith("w")]
    chorus_words = sum(len(ln.split()) for ln in CHORUS)
    assert sum(len(ln.words) for ln in filled) == chorus_words == stats.filled_words
    assert all(FROM_WHISPER in ln.flags for ln in filled) and lf.unmatched_audio == []
    assert filled[0].start in {b[0][0][1] for b, s in zip(blocks, sung, strict=True) if s == "C"}


def test_chorus_written_more_often_than_sung():
    lines, truth, blocks = _sections_song(["V1", "C", "C", "V2", "C"], ["V1", "C", "V2", "C"])
    lf, _ = align_lyrics(lines, transcript(truth))
    assert_valid(lf)
    lyric_choruses = [lf.lines[2:4], lf.lines[4:6]]
    interpolated = [all(w.origin == WordOrigin.interpolated for ln in c for w in ln.words) for c in lyric_choruses]
    matched = [all(w.origin == WordOrigin.matched for ln in c for w in ln.words) for c in lyric_choruses]
    assert sorted(interpolated) == [False, True] and sorted(matched) == [False, True]
    missing = lyric_choruses[interpolated.index(True)]
    assert all("low_confidence" in ln.flags for ln in missing)
    # the unsung chorus is placed between its neighbours, not on top of them
    assert lf.lines[1].end <= missing[0].start and missing[-1].end <= lf.lines[6].start
    for ln in lf.lines[6:]:
        assert all(w.origin == WordOrigin.matched for w in ln.words) and not ln.flags
    assert lf.unmatched_audio == []


def test_whisper_hallucinates_repeated_phrase():
    lines, truth = _song()
    flat = [w for r in truth for w in r]
    # after the first chorus whisper repeats its last line three times over the instrumental gap
    last = truth[3]
    t0 = last[-1][2] + 0.5
    fake, probs = [], {}
    for k in range(3):
        for w, s, e in last:
            off = t0 + k * 1.2 + (s - last[0][1]) * 0.3
            fake.append((w, round(off, 3), round(off + (e - s) * 0.3, 3)))
    heard = sorted(flat + fake, key=lambda w: w[1])
    for i, w in enumerate(heard):
        if w in fake:
            probs[i] = 0.35
    lf, _ = align_lyrics(lines, transcript(heard, probs=probs))
    assert_valid(lf)
    for w, err in start_errors(lf, truth):
        assert w.origin == WordOrigin.matched and abs(err) < 1e-6
    assert sum(len(u.text.split()) for u in lf.unmatched_audio) == len(fake)


def test_whisper_loop_is_dropped_and_reported():
    lines, truth = _song()
    flat = [w for r in truth for w in r]
    gap_start = truth[3][-1][2] + 0.3
    loop = [("glow", round(gap_start + 0.1 * k, 3), round(gap_start + 0.1 * k + 0.08, 3)) for k in range(40)]
    lf, stats = align_lyrics(lines, transcript(flat + loop))
    assert_valid(lf)
    for w, err in start_errors(lf, truth):
        assert w.origin == WordOrigin.matched and abs(err) < 1e-6
    assert stats.loop_words_dropped >= 38
    assert len(lf.unmatched_audio) == 1 and "(x" in lf.unmatched_audio[0].text


def test_sung_repetition_on_the_beat_is_not_a_loop():
    """A hook whisper mishears nine times on a steady beat ("toça" for "dança") is singing: its
    copies are distinct beats. A real loop stacks copies on the same instant."""
    from karaoke.align.tokens import drop_hallucination_loops, whisper_tokens

    sung = [("tossa", round(10 + 0.45 * k, 2), round(10.3 + 0.45 * k, 2)) for k in range(9)]
    loop = [("glow", 20.0 + 0.1 * (k // 3), 20.05 + 0.1 * (k // 3)) for k in range(12)]
    kept, dropped = drop_hallucination_loops(whisper_tokens(transcript(sung + loop)), ["dance"] * 9)
    assert [t.key for t in kept].count("tossa") == 9
    assert [t.key for t in dropped] == ["glow"] * 11


def test_portuguese_accents_and_informal_forms():
    lyrics = ["Tá na hora de ir pra casa", "Cê sabe que o coração não para", "Eu tô aqui, meu bem"]
    lines = parse_chosen_text(song_text(("verso", lyrics)))
    truth = sing(lyrics)
    heard_text = ["está na hora de ir para casa", "você sabe que o coracao nao para", "eu to aqui meu bem"]
    heard = [[(h, s, e) for h, (_, s, e) in zip(ht.split(), row, strict=True)]
             for ht, row in zip(heard_text, truth, strict=True)]
    lf, stats = align_lyrics(lines, transcript(heard), language="pt")
    assert_valid(lf)
    for w, err in start_errors(lf, truth):
        assert w.origin in HEARD and abs(err) < 1e-6
    assert [w.text for w in lf.lines[0].words][:1] == ["Tá"]
    assert lf.lines[1].words[4].text == "coração"
    assert lf.lines[0].section == "verso" and lf.language == "pt"


def test_english_contractions_split_and_merge():
    lyrics = ["I'm gonna find you 'cause I know", "nobody's waiting anymore"]
    lines = parse_chosen_text("\n".join(lyrics))
    heard = [("I'm", 1.0, 1.2), ("going", 1.25, 1.5), ("to", 1.5, 1.6), ("find", 1.65, 1.9), ("you", 1.95, 2.1),
             ("because", 2.15, 2.5), ("I", 2.55, 2.6), ("know", 2.65, 3.0),
             ("nobody", 4.0, 4.4), ("waiting", 4.45, 4.9), ("any", 4.95, 5.2), ("more", 5.2, 5.5)]
    lf, _ = align_lyrics(lines, transcript(heard))
    assert_valid(lf)
    words = flat_words(lf)
    assert all(w.origin in HEARD for w in words)
    gonna = lf.lines[0].words[1]
    assert (gonna.start, gonna.end) == (1.25, 1.6)
    anymore = lf.lines[1].words[2]
    assert (anymore.start, anymore.end) == (4.95, 5.5) and anymore.origin == WordOrigin.matched
    assert lf.lines[1].words[0].origin == WordOrigin.fuzzy  # nobody's ~ nobody


def test_two_lyric_words_in_one_whisper_word():
    lines = parse_chosen_text("we are going to dance tonight")
    heard = [("we", 1.0, 1.2), ("are", 1.25, 1.4), ("gonna", 1.45, 1.9), ("dance", 1.95, 2.3), ("tonight", 2.35, 2.9)]
    lf, _ = align_lyrics(lines, transcript(heard))
    assert_valid(lf)
    going, to = lf.lines[0].words[2:4]
    assert going.start == 1.45 and to.end == 1.9 and going.end == to.start
    assert going.origin == to.origin == WordOrigin.matched


def test_misheard_word_takes_whisper_timing():
    lines = parse_chosen_text("the cat sat on the mat today")
    heard = [("the", 1.0, 1.1), ("cat", 1.2, 1.5), ("sat", 1.6, 1.9), ("on", 2.0, 2.1), ("the", 2.2, 2.3),
             ("rug", 2.4, 2.9), ("today", 3.0, 3.5)]
    lf, _ = align_lyrics(lines, transcript(heard))
    mat = lf.lines[0].words[5]
    assert (mat.start, mat.end) == (2.4, 2.9) and mat.origin == WordOrigin.interpolated
    assert lf.unmatched_audio == []


def test_unheard_line_with_lrc_anchor():
    lines, truth = _song()
    heard = [w for k, r in enumerate(truth) if k != 5 for w in r]  # whisper misses a whole verse line
    lf, _ = align_lyrics(lines, transcript(heard), lrc=lrc_for(truth), use_line_timings=True)
    assert_valid(lf)
    line = lf.lines[5]
    assert all(w.origin == WordOrigin.interpolated for w in line.words)
    assert line.start == pytest.approx(truth[5][0][1], abs=0.01)
    assert "low_confidence" in line.flags
    # without the anchor it is still placed sensibly, right after the previous line's words
    lf2, _ = align_lyrics(lines, transcript(heard))
    assert abs(lf2.lines[5].start - truth[5][0][1]) < 1.0


def test_small_lrc_offset_moves_unheard_lines_too():
    """The LRC runs 0.2 s ahead of the voice. Heard lines show whisper's clock, so a line whisper
    missed must be placed on the same clock, not on the raw LRC time (it would light up early)."""
    lines, truth = _song()
    heard_rows = [[(w, s + 0.2, e + 0.2) for w, s, e in r] for r in truth]
    heard = [w for k, r in enumerate(heard_rows) if k != 5 for w in r]
    lf, stats = align_lyrics(lines, transcript(heard), lrc=lrc_for(truth), use_line_timings=True)
    assert_valid(lf)
    assert stats.lrc_offset == pytest.approx(0.2, abs=0.02)
    assert lf.lines[5].start == pytest.approx(heard_rows[5][0][1], abs=0.02)


def test_misheard_hook_over_several_lines_lands_on_the_heard_beats():
    """A chopped hook: whisper hears one wrong word per beat ("dumbo" for each "down low") on
    lines it cannot match at all. The words must sit on those beats, line by line."""
    hook = ["Down low, down low, down low, down low", "Down low, down low, down low, down low"]
    lyric = ["Paper boats along the river", *hook, "carry letters I never sent"]
    lines = parse_chosen_text(song_text(("verse", lyric)))
    truth = sing(lyric, line_gap=0.5)
    beats = [(r[k][1], r[k + 1][2]) for r in truth[1:3] for k in range(0, 8, 2)]
    heard = truth[0] + [("dumbo", s, e) for s, e in beats] + truth[3]
    lf, _ = align_lyrics(lines, transcript(heard), lrc=lrc_for(truth), use_line_timings=True)
    assert_valid(lf)
    for ln, row in zip(lf.lines[1:3], truth[1:3], strict=True):
        for k in range(0, 8, 2):
            assert ln.words[k].start == pytest.approx(row[k][1], abs=0.02)  # "Down" on its beat
            assert row[k][1] < ln.words[k + 1].start < row[k + 1][2]  # "low" inside the same beat
    assert lf.unmatched_audio == []


def test_lrc_prevents_chorus_jump():
    """Whisper misses the whole second chorus; without anchors the DP has no way to know which
    chorus it heard, with anchors each chorus stays in its own LRC window."""
    lines, truth = _song()
    heard = [w for k, r in enumerate(truth) if k not in (6, 7) for w in r]
    lf, _ = align_lyrics(lines, transcript(heard), lrc=lrc_for(truth), use_line_timings=True)
    assert_valid(lf)
    for k in (2, 3, 10, 11):
        assert all(w.origin == WordOrigin.matched for w in lf.lines[k].words)
        assert lf.lines[k].start == truth[k][0][1]
    for k in (6, 7):
        assert lf.lines[k].start == pytest.approx(truth[k][0][1], abs=0.01)


# --------------------------------------------------------------------------- edits, fallback, perf


def _demo_run(p, jobs_dir) -> LyricsFinal:
    run_job("demo", only=[StageName.align], store=JobStore(jobs_dir))
    return read_model(p.lyrics_final, LyricsFinal)


def test_manual_and_ai_edits_are_preserved(demo_no_audio, jobs_dir):
    p = demo_no_audio
    p.lyrics_final.unlink()
    lf = _demo_run(p, jobs_dir)
    w = lf.lines[1].words[2]
    w.start, w.end, w.origin, w.conf = 5.90, 6.10, WordOrigin.manual, 1.0
    ai = lf.lines[3].words[1]
    ai.text, ai.origin = "listenin'", WordOrigin.ai
    lf.lines[3].flags.append("ai_edited")
    write_model(p.lyrics_final, lf)

    lf2 = _demo_run(p, jobs_dir)
    assert_valid(lf2, make_demo.DURATION)
    w2 = lf2.lines[1].words[2]
    assert (w2.text, w2.start, w2.end, w2.origin) == ("by", 5.90, 6.10, WordOrigin.manual)
    assert lf2.lines[3].words[1].text == "listenin'" and lf2.lines[3].words[1].origin == WordOrigin.ai
    assert "ai_edited" in lf2.lines[3].flags

    # a new line inserted at the top shifts every id: edits follow their text
    text = p.lyrics_chosen_text.read_text()
    p.lyrics_chosen_text.write_text("[intro]\nhello hello\n\n" + text)
    lf3 = _demo_run(p, jobs_dir)
    assert len(lf3.lines) == 7
    w3 = lf3.lines[2].words[2]
    assert (w3.text, w3.start, w3.end, w3.origin) == ("by", 5.90, 6.10, WordOrigin.manual)
    assert lf3.lines[4].words[1].origin == WordOrigin.ai


def test_whisper_fallback(demo_no_audio, jobs_dir):
    p = demo_no_audio
    p.lyrics_final.unlink()
    write_model(p.lyrics_choice, LyricsChoice(chosen=LyricsSourceName.whisper, language="en"))
    lf = _demo_run(p, jobs_dir)
    assert_valid(lf, make_demo.DURATION)
    assert lf.text_source == LyricsSourceName.whisper
    assert all(ln.flags == ["low_confidence"] for ln in lf.lines)
    assert all(len(ln.words) <= 10 for ln in lf.lines)
    tr = read_model(p.transcript, Transcript)
    assert sum(len(ln.words) for ln in lf.lines) == sum(len(s.words) for s in tr.segments)
    assert lf.lines[0].words[0].text == "Morning"
    assert [ln.id for ln in lf.lines] == [f"l{i:03d}" for i in range(1, len(lf.lines) + 1)]


def test_fallback_splits_long_segments():
    words = [(f"w{k}", round(k * 0.4, 2), round(k * 0.4 + 0.3, 2)) for k in range(25)]
    words += [(f"x{k}", round(20 + k * 0.4, 2), round(20 + k * 0.4 + 0.3, 2)) for k in range(4)]
    tr = Transcript(model="t", language="en", segments=[TSegment(start=0, end=22, text="", words=[
        TWord(text=" " + w, start=s, end=e, prob=0.9) for w, s, e in words])])
    lf, _ = align_lyrics([], tr, text_source=LyricsSourceName.whisper)
    assert [len(ln.words) for ln in lf.lines][-1] == 4  # the pause before the x-words splits
    assert all(5 <= len(ln.words) <= 10 for ln in lf.lines[:-1]) and sum(len(ln.words) for ln in lf.lines) == 29
    assert_valid(lf)


def test_performance_600_words():
    rng = random.Random(1)
    vocab = [f"{a}{b}" for a in ("ka", "lo", "mi", "su", "re", "ta", "vo", "ne") for b in ("ra", "to", "li", "an", "e")]
    vocab += ["the", "and", "love", "night", "oh", "you", "me", "light", "heart", "go"]
    words = [rng.choice(vocab) for _ in range(600)]
    lines = parse_chosen_text("\n".join(" ".join(words[k:k + 6]) for k in range(0, 600, 6)))
    heard, t = [], 1.0
    for w in words:
        if rng.random() < 0.1:
            continue
        if rng.random() < 0.1:
            heard.append((rng.choice(vocab), t, t + 0.2))
            t += 0.25
        heard.append((w, t, t + 0.3))
        t += 0.35
    while len(heard) < 600:
        heard.append((rng.choice(vocab), t, t + 0.2))
        t += 0.3
    tr = transcript(heard[:600])
    t0 = time.perf_counter()
    lf, stats = align_lyrics(lines, tr)
    took = time.perf_counter() - t0
    assert took < 1.0, f"{took:.2f}s"
    assert stats.words == 600 and stats.whisper_words == 600
    assert stats.matched + stats.fuzzy > 450
    assert_valid(lf)


def test_dp_ops_cover_everything():
    a = ["sing", "it", "loud", "sing", "it", "loud"]
    b = ["it", "loud", "yeah", "sing", "it", "out"]
    ops = dp.align(a, b)
    assert sorted(i for op in ops for i in op.li) == list(range(len(a)))
    assert sorted(j for op in ops for j in op.wi) == list(range(len(b)))
    assert [op.kind for op in ops if op.kind == "match"] == ["match"] * 4
