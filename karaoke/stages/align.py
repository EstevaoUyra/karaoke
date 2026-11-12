"""Stage 5 — align. Owner: workstream D.

Inputs: ``transcript.whisper.json``, ``lyrics/chosen.json``, ``lyrics/chosen.txt``, optional ``lyrics/sources/lrclib.lrc``,
``stems/vocals.wav`` (its loudness shows the pauses that whisper folds into words, see ``align.voicing``).
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

The algorithm lives in ``karaoke/align/core.py``; this module only reads and writes job files.
"""

from __future__ import annotations

import time

from karaoke.pipeline import StageContext


def run(ctx: StageContext) -> None:
    from karaoke.align.core import align_lyrics
    from karaoke.align.tokens import parse_chosen_text, parse_lrc
    from karaoke.models import LyricsChoice, LyricsFinal, LyricsSourceName, Meta, Transcript
    from karaoke.store import read_model, read_model_opt, write_model
    from karaoke.validate import validate_lyrics

    p = ctx.paths
    t0 = time.monotonic()
    transcript = read_model(p.transcript, Transcript)
    choice = read_model_opt(p.lyrics_choice, LyricsChoice) or LyricsChoice(chosen=LyricsSourceName.whisper)
    meta = read_model_opt(p.meta, Meta)
    previous = None
    if p.lyrics_final.exists():
        try:
            previous = read_model(p.lyrics_final, LyricsFinal)
        except Exception as exc:  # noqa: BLE001 — a broken old file must not block a re-run
            ctx.log.warning("ignoring unreadable previous lyrics.final.json: %s", exc)

    lines = []
    source = choice.chosen
    if source != LyricsSourceName.whisper:
        if p.lyrics_chosen_text.exists():
            lines = parse_chosen_text(p.lyrics_chosen_text.read_text(encoding="utf-8"))
        if not lines:
            ctx.log.warning("chosen.txt is missing or empty; falling back to the whisper text")
            source = LyricsSourceName.whisper
    lrc = None
    lrc_path = p.lyrics_source("lrclib.lrc")
    if choice.use_line_timings and lrc_path.exists():
        lrc = parse_lrc(lrc_path.read_text(encoding="utf-8"))
    envelope = None
    if p.vocals.exists():
        from karaoke.align.voicing import Envelope

        envelope = Envelope.from_file(p.vocals)
    else:
        ctx.log.warning("no vocal stem: pauses cannot be trimmed out of word timings")
    ctx.progress(0.2, f"aligning {sum(len(ln.words) for ln in lines)} lyric words")

    lf, stats = align_lyrics(
        lines, transcript, lrc=lrc, use_line_timings=bool(choice.use_line_timings and lrc),
        duration=meta.duration if meta else None,
        language=choice.language or (transcript.language if transcript.language in ("pt", "en", "ja") else None),
        text_source=source, previous=previous, envelope=envelope,
    )
    for issue in validate_lyrics(lf, meta.duration if meta else None):
        ctx.log.warning("validation: %s", issue)
    for lost in stats.lost_edits:
        ctx.log.warning("could not place a preserved edit: %s", lost)
    if stats.lrc_disabled_reason:
        ctx.log.warning("LRC anchors ignored: %s", stats.lrc_disabled_reason)
    write_model(p.lyrics_final, lf)
    flagged = sum(bool(ln.flags) for ln in lf.lines)
    msg = (f"{len(lf.lines)} lines, {stats.matched + stats.fuzzy}/{stats.words} words heard, "
           f"{stats.interpolated} interpolated, {flagged} lines flagged, "
           f"{stats.pauses_trimmed} pauses trimmed, {len(lf.unmatched_audio)} unmatched audio regions "
           f"({time.monotonic() - t0:.2f}s)")
    ctx.log.info("stats: %s", stats)
    ctx.progress(1.0, msg)
