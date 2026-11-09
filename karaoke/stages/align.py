"""Stage 5 — align. Owner: workstream D.

Inputs: ``transcript.whisper.json``, ``lyrics/chosen.json``, ``lyrics/chosen.txt``, optional ``lyrics/sources/lrclib.lrc``.
Output: ``lyrics.final.json`` (models.LyricsFinal).

The lyrics text decides the WORDS and whisper decides the TIMING (PLAN.md §4 "Alignment"):
1. Global Needleman-Wunsch over normalized tokens, scored with rapidfuzz similarity.
2. Matched words take whisper timing (``origin`` matched or fuzzy). Unmatched runs are interpolated
   between their neighbours (``origin`` interpolated).
3. When ``chosen.json.use_line_timings`` is set, LRC line starts act as anchors/constraints.
4. Whisper words with no lyric match go into ``unmatched_audio``.
5. Each line gets a confidence score and ``flags`` (``low_confidence``, ``timing_suspect``).
6. If a previous lyrics.final.json exists, words with ``origin`` manual or ai are PRESERVED
   (matched by line id plus word index, or by text).
If chosen is ``whisper``, lines come from transcript segments and are all flagged ``low_confidence``.
"""

from karaoke.pipeline import StageContext


def run(ctx: StageContext) -> None:
    raise NotImplementedError("align: Phase 1, workstream D")
