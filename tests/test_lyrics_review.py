"""Lyrics review: the brief's suspects, safe edits, and the seeded-error job as a measure."""

from __future__ import annotations

import make_demo
import pytest

from karaoke import lyrics_review as R
from karaoke.models import LyricsFinal, WordOrigin
from karaoke.store import read_model, write_model

# What a careful reviewer does with the seeded job (the karaoke-lyrics-review skill, by hand).
FIX = [
    {"op": "shift_line", "line": "l001", "by": 0.9},
    {"op": "replace_word", "line": "l002", "word": 1, "text": "cooling"},
    {"op": "delete_line", "line": "l007"},
    {"op": "set_line", "line": "l005", "words": [
        {"text": "Sing", "start": 17.5, "end": 17.88}, {"text": "it", "start": 17.91, "end": 18.08},
        {"text": "loud,", "start": 18.18, "end": 18.63}, {"text": "sing", "start": 18.63, "end": 19.02},
        {"text": "it", "start": 19.04, "end": 19.2}, {"text": "loud", "start": 19.27, "end": 19.49}]},
    {"op": "insert_line", "words": [{"text": "nobody's", "start": 19.69, "end": 20.44},
                                    {"text": "listening", "start": 20.45, "end": 21.11},
                                    {"text": "anymore", "start": 21.14, "end": 21.87}]},
    {"op": "clear_unmatched", "start": 19.7, "end": 21.9},
]


@pytest.fixture
def seeded(jobs_dir):
    return make_demo.build_seeded(jobs_dir)


def final(p) -> LyricsFinal:
    return read_model(p.lyrics_final, LyricsFinal)


def test_the_clean_demo_has_only_the_decoys_as_suspects(demo_no_audio):
    kinds = {(s["line"], s["kind"]) for s in R.brief(demo_no_audio)["suspects"]}
    print(kinds)
    assert (None, "unmatched_audio") in kinds and ("l004", "text_differs") in kinds
    assert not {k for _, k in kinds} & {"not_heard", "unsure_words", "overlap"}
    assert R.compare(final(demo_no_audio), make_demo.truth_lyrics())["total_errors"] == 0


def test_the_brief_points_at_every_seeded_error(seeded):
    b = R.brief(seeded)
    found = {(s["line"], s["kind"]) for s in b["suspects"]}
    assert ("l001", "start_off") in found  # the early line
    assert ("l002", "unsure_words") in found  # the wrong word
    assert ("l007", "not_heard") in found  # the line nobody sings
    assert ("l005", "unsure_words") in found and ("l005", "start_off") in found  # the interpolated line
    missing = [s for s in b["suspects"] if s["kind"] == "unmatched_audio" and s["at"] > 19]
    assert missing and [h["text"] for h in missing[0]["whisper_heard"]][-1] == "anymore"  # the missing line
    assert "cooling" in b["other_sources"]["youtube.vtt"]
    line = next(ln for ln in b["lines"] if ln["id"] == "l002")
    assert [h["text"] for h in line["whisper_heard"]][1] == "cooling" and line["words"][1]["conf"] < 0.5


def test_the_fixes_bring_the_seeded_job_back_to_the_truth(seeded):
    truth = make_demo.truth_lyrics()
    before = R.compare(final(seeded), truth)
    assert before["total_errors"] >= 15 and before["wrong_words"] == 1 and before["extra_words"] == 4
    assert len(R.apply_edits(seeded, FIX, dry_run=True)) == len(FIX)
    assert R.compare(final(seeded), truth) == before  # a dry run saves nothing
    R.apply_edits(seeded, FIX)
    after = R.compare(final(seeded), truth)
    assert after["total_errors"] == 0 and after["worst_timing_s"] <= 0.05
    lf = final(seeded)
    assert [ln.words[0].start for ln in lf.lines] == sorted(ln.words[0].start for ln in lf.lines)
    assert len({ln.id for ln in lf.lines}) == len(lf.lines) == 6
    changed = [ln for ln in lf.lines if R.AI_FLAG in ln.flags]
    assert len(changed) == 4 and all("timing_suspect" not in ln.flags for ln in changed)
    assert lf.lines[1].words[1].origin == WordOrigin.ai and lf.lines[1].words[0].origin == WordOrigin.matched
    assert [u.text for u in lf.unmatched_audio] == ["yeah"]  # the ad-lib stays unmatched
    left = {(s["line"], s["kind"]) for s in R.brief(seeded)["suspects"]}
    assert left == {("l004", "text_differs"), (None, "unmatched_audio")}  # only the decoys


def test_hand_edited_words_and_invalid_results_are_refused(seeded):
    lf = final(seeded)
    lf.lines[1].words[1].origin = WordOrigin.manual
    write_model(seeded.lyrics_final, lf)
    saved = seeded.lyrics_final.read_bytes()
    with pytest.raises(R.EditError, match="by hand"):
        R.apply_edits(seeded, [{"op": "replace_word", "line": "l002", "word": 1, "text": "cooling"}])
    with pytest.raises(R.EditError, match="not valid"):  # a line moved on top of the next one
        R.apply_edits(seeded, [{"op": "shift_line", "line": "l003", "by": 1.5}])
    with pytest.raises(R.EditError, match="bad reference|no line"):
        R.apply_edits(seeded, [{"op": "delete_line", "line": "l999"}])
    with pytest.raises(R.EditError, match="unknown op"):
        R.apply_edits(seeded, [{"op": "rewrite_everything"}])
    assert seeded.lyrics_final.read_bytes() == saved


def test_estimates_stay_on_the_suspects_list(seeded):
    edits = [dict(e) for e in FIX]
    edits[3]["estimate"] = True  # l005: say its times were placed, not heard
    R.apply_edits(seeded, edits)
    lf = final(seeded)
    l5 = next(ln for ln in lf.lines if ln.id == "l005")
    assert R.ESTIMATE_FLAG in l5.flags and all(w.conf == R.ESTIMATE_CONF and w.origin == WordOrigin.ai for w in l5.words)
    assert ("l005", "estimate") in {(s["line"], s["kind"]) for s in R.brief(seeded)["suspects"]}


def test_brief_reports_contract_problems_and_reads_compactly(seeded):
    lf = final(seeded)
    lf.lines[-1].words[-1].end = 99.0  # past the end of the 24 s song
    lf.lines[-1].end = 99.0
    write_model(seeded.lyrics_final, lf)
    b = R.brief(seeded)
    assert any("outside audio duration" in v for v in b["validation"])
    by_line = {}
    for s in b["suspects"]:
        by_line.setdefault(s["line"], []).append(s["kind"])
    assert "low_confidence" not in by_line["l005"]  # said already by unsure_words
    text = R.compact(b)
    assert text.splitlines()[1].startswith("VALIDATION:")
    assert "! l002" in text and "heard: coffee@" in text and "\n  l003" in text
    assert len(text) < len(__import__("json").dumps(b)) / 2


def test_more_words_heard_than_written_is_a_suspect(demo_no_audio):
    lf = final(demo_no_audio)
    l3, l4 = lf.lines[2], lf.lines[3]  # the two chorus lines, sung back to back
    l3.words = l3.words[:3]  # the lyrics have only half of what is sung in l003's time
    l3.end = l4.start - 0.05
    write_model(demo_no_audio.lyrics_final, lf)
    assert ("l003", "extra_heard") in {(s["line"], s["kind"]) for s in R.brief(demo_no_audio)["suspects"]}
