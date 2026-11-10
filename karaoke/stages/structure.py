"""Stage — structure. Sections, loudness and beats of the song (see ``karaoke.structure``).

Inputs: ``audio.wav``, ``lyrics.final.json``, ``meta.json`` (duration).
Output: ``structure.json`` (models.Structure).
"""

from karaoke.pipeline import StageContext


def run(ctx: StageContext) -> None:
    from karaoke.models import LyricsFinal, Meta
    from karaoke.store import read_model, write_model
    from karaoke.structure import analyze

    p = ctx.paths
    meta = read_model(p.meta, Meta)
    lyrics = read_model(p.lyrics_final, LyricsFinal)
    ctx.progress(0.05, "listening for sections, loudness and beats")
    st = analyze(p.audio, lyrics.lines, meta.duration)
    write_model(p.structure, st)
    summary = ", ".join(f"{s.label} {s.start:.0f}s" for s in st.sections)
    ctx.log.info("sections: %s", summary)
    ctx.progress(1.0, f"{len(st.sections)} sections, {st.tempo_bpm or '?'} bpm")
