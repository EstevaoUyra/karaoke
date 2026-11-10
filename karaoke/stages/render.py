"""Stage 7 — render. Owner: workstream E.

Inputs: ``stems/instrumental.wav`` (and ``stems/vocals.wav`` if ``render.guide_vocals_db`` is set),
``lyrics.final.json``, ``background.json``, ``assets/bg/``, ``meta.json``.
Outputs: ``output/karaoke.ass``, plus ``output/preview.mp4`` (preset "preview": 720p, ultrafast)
or ``output/karaoke.mp4`` (preset "final": 1080p, x264 CRF 20, AAC 192k).

1. ``karaoke.render.ass`` builds the ASS from lyrics.final.json and RenderSettings. Each word gets
   a ``\\kf`` fill. Show the current line plus the next (dimmed), each appearing ``lead_in_s``
   early. Show a countdown before lines that follow a gap longer than ``countdown_gap_s``, and a
   title card (clean_title / clean_artist) during the intro.
2. ``karaoke.render.compositor`` turns background.json into a video filtergraph: color, gradient,
   image (with zoompan Ken Burns or pan), video, dim/blur, and xfade transitions.
3. One ffmpeg call: the background graph, then the ``ass`` filter, then the audio mix → output.
   ``ctx.settings.ffmpeg_bin`` MUST have libass (Homebrew ``ffmpeg-full``).
Report progress by parsing ffmpeg's ``-progress pipe:1`` output.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from karaoke.pipeline import StageContext

PREVIEW_SIZE = (1280, 720)


@dataclass(frozen=True)
class Encode:
    preset: str
    crf: int
    audio_bitrate: str


ENCODES = {
    "preview": Encode("ultrafast", 28, "128k"),
    "final": Encode("medium", 20, "192k"),
}


def build_command(
    *, bg_inputs: list[str], bg_filter: str, frames: int, fps: int, ass_name: str,
    instrumental: Path, vocals: Path | None, guide_db: float | None, duration: float,
    encode: Encode, out: Path, fontsdir: Path | None = None,
) -> list[str]:
    """ffmpeg args (without the binary) for background + ASS + audio → mp4."""
    n_bg = sum(1 for a in bg_inputs if a == "-i")
    args = ["-y", *bg_inputs, "-i", str(instrumental)]
    a_inst = n_bg
    ass = f"ass=filename={ass_name}"
    if fontsdir:
        ass += f":fontsdir={fontsdir}"
    graph = [bg_filter, f"[bg]{ass}[v]"]
    fmt = "aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo"
    pad = f"atrim=end={duration:.3f},apad=whole_dur={duration:.3f}"
    if guide_db is not None and vocals is not None:
        args += ["-i", str(vocals)]
        graph.append(f"[{a_inst}:a]{fmt}[inst]")
        graph.append(f"[{a_inst + 1}:a]{fmt},volume={guide_db:.1f}dB[voc]")
        graph.append(f"[inst][voc]amix=inputs=2:duration=first:normalize=0,{pad}[a]")
    else:
        graph.append(f"[{a_inst}:a]{fmt},{pad}[a]")
    args += [
        "-filter_complex", ";".join(graph),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", encode.preset, "-crf", str(encode.crf), "-pix_fmt", "yuv420p",
        "-r", str(fps), "-frames:v", str(frames),
        "-c:a", "aac", "-b:a", encode.audio_bitrate,
        "-t", f"{duration:.3f}",
        "-movflags", "+faststart",
        "-f", "mp4", str(out),
    ]
    return args


def run(ctx: StageContext) -> None:
    from karaoke.models import Background, LyricsFinal, Melody, Meta
    from karaoke.render.ass import build_ass
    from karaoke.render.compositor import build_graph
    from karaoke.render.ffmpeg import has_libass, run_ffmpeg
    from karaoke.stages.backgrounds import default_background
    from karaoke.store import atomic_write_text, read_model, read_model_opt

    p = ctx.paths
    rs = ctx.job.settings.render
    ffmpeg_bin = ctx.settings.ffmpeg_bin
    if not has_libass(ffmpeg_bin):
        raise RuntimeError(
            f"ffmpeg at {ffmpeg_bin!r} has no libass (no 'ass' filter), so it cannot burn karaoke "
            "subtitles. Install Homebrew's ffmpeg-full (`brew install ffmpeg-full`) or point "
            "KARAOKE_FFMPEG / KARAOKE_FFPROBE at an ffmpeg built with --enable-libass."
        )

    meta = read_model(p.meta, Meta)
    lyrics = read_model(p.lyrics_final, LyricsFinal)
    bg = read_model_opt(p.background, Background)
    if bg is None:
        ctx.log.warning("no background.json; rendering the default background")
        bg = default_background(meta.duration)
    if not p.instrumental.exists():
        raise FileNotFoundError(f"missing {p.rel(p.instrumental)}: run the separate stage first")
    guide_db = rs.guide_vocals_db
    if guide_db is not None and not p.vocals.exists():
        ctx.log.warning("guide_vocals_db is set but %s is missing; rendering without guide vocals",
                        p.rel(p.vocals))
        guide_db = None

    size = PREVIEW_SIZE if rs.preset == "preview" else tuple(bg.resolution)
    size = (size[0] - size[0] % 2, size[1] - size[1] % 2)
    encode = ENCODES[rs.preset]
    out = p.preview_video if rs.preset == "preview" else p.final_video
    duration = meta.duration

    ctx.progress(0.01, "finding the beat for the countdown dots")
    try:
        from karaoke.render.tempo import beat_period

        beat_s = beat_period(p.instrumental)
    except Exception as exc:  # noqa: BLE001 — the dots fall back to 1 s beats
        ctx.log.warning("tempo detection failed: %s", exc)
        beat_s = None
    ctx.log.info("countdown beat: %s s", beat_s)
    ctx.progress(0.02, f"building subtitles ({size[0]}x{size[1]})")
    melody = contour = None
    if rs.show_pitch:
        melody = read_model_opt(p.melody, Melody)
        if melody is None:
            ctx.log.warning("show_pitch is on but %s is missing (run the melody stage); no pitch guide",
                            p.rel(p.melody))
        elif melody.key is not None:
            ctx.log.info("pitch guide: %d notes, key %s %s, octave shift %+d", len(melody.notes),
                         melody.key.tonic, melody.key.mode, melody.display_octave_shift)
        if melody is not None and rs.pitch_style == "roll":
            from karaoke.melody.pipeline import load_contour_midi

            contour = load_contour_midi(p)
            if contour is None:
                ctx.log.info("no pitch contour (%s): the guide has bars only", p.rel(p.melody_f0))
    ass_text = build_ass(lyrics, meta, rs, size, beat_s, melody, contour)
    atomic_write_text(p.ass, ass_text)

    p.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="karaoke-render-") as tmp:
        work = Path(tmp)
        # ffmpeg runs inside the workdir and reads the ASS by a plain relative name: no filtergraph
        # escaping of arbitrary job paths.
        shutil.copyfile(p.ass, work / "karaoke.ass")
        ctx.progress(0.02, "preparing background")
        g = build_graph(bg, p.root, work, size=size, duration=duration)
        part = out.with_name(f".{out.name}.part")
        args = build_command(
            bg_inputs=g.input_args, bg_filter=g.filter, frames=g.frames, fps=g.fps, ass_name="karaoke.ass",
            instrumental=p.instrumental.resolve(), vocals=p.vocals.resolve() if guide_db is not None else None,
            guide_db=guide_db, duration=duration, encode=encode, out=part.resolve(),
        )

        def on_progress(t: float) -> None:
            frac = min(1.0, t / duration) if duration > 0 else 0.0
            ctx.progress(0.05 + 0.94 * frac, f"encoding {t:5.1f}/{duration:.1f}s")

        ctx.progress(0.05, f"encoding {rs.preset} with {ffmpeg_bin}")
        try:
            run_ffmpeg(ffmpeg_bin, args, on_progress=on_progress, log=ctx.log, cwd=work)
            os.replace(part, out)
        finally:
            part.unlink(missing_ok=True)
    ctx.progress(1.0, f"wrote {p.rel(out)}")
