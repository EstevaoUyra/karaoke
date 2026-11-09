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

from karaoke.pipeline import StageContext


def run(ctx: StageContext) -> None:
    raise NotImplementedError("render: Phase 1, workstream E")
