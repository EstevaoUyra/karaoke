"""Stage 1 — ingest. Owner: workstream A.

Input:  ``ctx.job.url`` (already stripped of playlist params by ``store.clean_youtube_url``).
Output:
- ``source.mp4``: best video ≤1080p with audio (kept as a possible background, ``video`` scene type).
- ``audio.wav``: 44.1 kHz stereo PCM s16, extracted with ``ctx.settings.ffmpeg_bin``.
- ``meta.json`` (models.Meta): ``duration`` measured from audio.wav. ``clean_artist``/``clean_title``
  come from ``karaoke.lyrics.titleclean.clean(title, uploader, artist, track)``.
- ``lyrics/sources/youtube.vtt``: only MANUAL subtitles. Prefer the song language (pt/en);
  otherwise the first manual track. Auto-captions are ignored.

Use the yt-dlp Python API (``yt_dlp.YoutubeDL``) with ``noplaylist=True`` and
``js_runtimes={ctx.settings.yt_js_runtime: {}}``. Report download progress through ``ctx.progress``.
"""

from karaoke.pipeline import StageContext


def run(ctx: StageContext) -> None:
    raise NotImplementedError("ingest: Phase 1, workstream A")
