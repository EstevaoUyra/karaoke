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

Idempotent: an existing ``source.mp4`` is never downloaded again. The metadata is fetched again
(``extract_info(download=False)``, no media) only when ``meta.json`` is missing, the stage is
forced, or manual subs are known but ``youtube.vtt`` is missing. ``audio.wav`` is re-extracted when
it is missing, older than ``source.mp4``, or the stage is forced.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from karaoke.pipeline import StageContext

# Video ≤1080p, preferring H.264 + AAC (plays everywhere, no re-encode when merged into mp4).
FORMAT = (
    "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/"
    "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/"
    "bv*[height<=1080]+ba/"
    "b[height<=1080]/b"
)
SUB_LANGS = ("pt", "en")  # the song languages we support (PLAN.md: PT-BR and EN)


# --------------------------------------------------------------------------- helpers


def _base_lang(code: str) -> str:
    return code.split("-")[0].split("_")[0].lower()


def manual_sub_langs(info: dict[str, Any]) -> list[str]:
    """Language codes of the MANUAL subtitle tracks (``subtitles``; ``automatic_captions`` ignored)."""
    subs = info.get("subtitles") or {}
    return [lang for lang, fmts in subs.items() if fmts and lang != "live_chat"]


def choose_sub_lang(langs: list[str], hint: str | None = None) -> str | None:
    """Prefer pt/en (the one matching ``hint`` if both exist), else the first manual track."""
    if not langs:
        return None
    ours = [lang for lang in langs if _base_lang(lang) in SUB_LANGS]
    if ours:
        if hint and _base_lang(hint) in SUB_LANGS:
            for lang in ours:
                if _base_lang(lang) == _base_lang(hint):
                    return lang
        return ours[0]
    return langs[0]


def _vtt_url(info: dict[str, Any], lang: str) -> str | None:
    for fmt in (info.get("subtitles") or {}).get(lang) or []:
        if fmt.get("ext") == "vtt" and fmt.get("url"):
            return fmt["url"]
    return None


def _first_artist(info: dict[str, Any]) -> str | None:
    artists = info.get("artists") or info.get("creators")
    if isinstance(artists, list) and artists:
        return str(artists[0])
    a = info.get("artist") or info.get("creator")
    return str(a).split(", ")[0] if a else None


def _all_artists(info: dict[str, Any]) -> str | None:
    artists = info.get("artists") or info.get("creators")
    if isinstance(artists, list) and artists:
        return ", ".join(str(a) for a in artists)
    return info.get("artist") or info.get("creator") or None


def _ffmpeg_location(ffmpeg_bin: str) -> str | None:
    return ffmpeg_bin if os.path.isabs(ffmpeg_bin) and os.path.exists(ffmpeg_bin) else None


def extract_audio(ffmpeg_bin: str, src: Path, dst: Path) -> None:
    """``src`` → 44.1 kHz stereo PCM s16 WAV at ``dst`` (atomic)."""
    tmp = dst.with_name(f".{dst.name}.tmp")
    cmd = [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
           "-vn", "-map", "0:a:0", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", "-f", "wav", str(tmp)]
    try:
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed: {r.stderr.strip()[-2000:]}")
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


class _Progress:
    """yt-dlp progress hook → ctx.progress, mapped into [lo, hi], monotonic and throttled to 1 % steps.

    yt-dlp downloads the video and the audio stream in turn, so each file counts as half.
    """

    def __init__(self, ctx: StageContext, lo: float, hi: float):
        self.ctx, self.lo, self.hi = ctx, lo, hi
        self.files: dict[str, float] = {}
        self.last = -1.0

    def __call__(self, d: dict[str, Any]) -> None:
        status = d.get("status")
        if status not in ("downloading", "finished"):
            return
        done = float(d.get("downloaded_bytes") or 0)
        total = float(d.get("total_bytes") or d.get("total_bytes_estimate") or 0)
        frac = 1.0 if status == "finished" else (min(1.0, done / total) if total else 0.0)
        self.files[str(d.get("filename") or "")] = frac
        overall = sum(self.files.values()) / max(len(self.files), 2)
        value = self.lo + (self.hi - self.lo) * min(1.0, overall)
        if value - self.last >= 0.01 or (status == "finished" and value > self.last):
            self.last = value
            what = f"{done / 1e6:.1f}/{total / 1e6:.1f} MB" if total else f"{done / 1e6:.1f} MB"
            self.ctx.progress(value, f"downloading file {len(self.files)}: {what}")


# --------------------------------------------------------------------------- stage


def run(ctx: StageContext) -> None:
    import yt_dlp

    from karaoke.lyrics.titleclean import clean
    from karaoke.lyrics.youtube_subs import parse_vtt_text
    from karaoke.models import Meta
    from karaoke.store import atomic_write_text, read_model_opt, write_model

    p = ctx.paths
    p.ensure_dirs()
    url = ctx.job.url
    vtt_path = p.lyrics_source("youtube.vtt")
    old_meta = read_model_opt(p.meta, Meta)

    have_video = p.source_video.exists()
    need_info = (
        not have_video
        or ctx.force
        or old_meta is None
        or (bool(old_meta.has_manual_subs) and not vtt_path.exists())
    )
    if need_info and not url:
        raise ValueError(f"job {ctx.job.id} has no url")

    opts: dict[str, Any] = {
        "format": FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": {"default": str(p.source_video.with_suffix("")) + ".%(ext)s"},
        "noplaylist": True,
        "js_runtimes": {ctx.settings.yt_js_runtime: {}},
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [_Progress(ctx, 0.05, 0.85)],
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
        "writesubtitles": False,
        "writeautomaticsub": False,
        "logger": _YtLogger(ctx),
    }
    loc = _ffmpeg_location(ctx.settings.ffmpeg_bin)
    if loc:
        opts["ffmpeg_location"] = loc

    info: dict[str, Any] | None = None
    vtt_text: str | None = None
    if need_info:
        with yt_dlp.YoutubeDL(opts) as ydl:
            if have_video:
                ctx.progress(0.02, "fetching metadata (source.mp4 already downloaded)")
                info = ydl.extract_info(url, download=False)
            else:
                ctx.progress(0.02, "downloading video")
                info = ydl.extract_info(url, download=True)
                _settle_video(info, p.source_video)
            assert info is not None
            lang = choose_sub_lang(manual_sub_langs(info), info.get("language"))
            sub_url = _vtt_url(info, lang) if lang else None
            if sub_url:
                ctx.progress(0.86, f"fetching manual subtitles ({lang})")
                try:
                    vtt_text = ydl.urlopen(sub_url).read().decode("utf-8", errors="replace")
                except Exception as exc:  # noqa: BLE001 — subtitles are optional
                    ctx.log.warning("could not fetch %s subtitles: %s", lang, exc)
            elif lang:
                ctx.log.warning("manual subtitles %s have no vtt format; skipped", lang)

    if not p.source_video.exists():
        raise RuntimeError("download finished but source.mp4 is missing")

    stale_audio = not p.audio.exists() or p.audio.stat().st_mtime < p.source_video.stat().st_mtime
    if stale_audio or ctx.force:
        ctx.progress(0.88, "extracting audio.wav")
        extract_audio(ctx.settings.ffmpeg_bin, p.source_video, p.audio)
    duration = round(audio_duration(p.audio), 3)

    if vtt_text is not None:
        cues = parse_vtt_text(vtt_text)
        if cues:
            atomic_write_text(vtt_path, vtt_text)
            ctx.log.info("wrote youtube.vtt (%d cues)", len(cues))
        else:
            ctx.log.warning("manual subtitles were empty; youtube.vtt not written")

    if info is not None:
        title = info.get("title") or info.get("fulltitle") or ctx.job.id
        uploader = info.get("uploader") or info.get("channel")
        ct = clean(title, uploader=uploader, artist=_first_artist(info), track=info.get("track"))
        meta = Meta(
            url=url,
            video_id=info.get("id"),
            title=title,
            uploader=uploader,
            artist=_all_artists(info),
            track=info.get("track"),
            album=info.get("album"),
            duration=duration,
            thumbnail=info.get("thumbnail"),
            youtube_language=info.get("language"),
            has_manual_subs=manual_sub_langs(info),
            clean_artist=ct.artist,
            clean_title=ct.title,
            clean_extra=ct.extra,
        )
    else:
        assert old_meta is not None
        meta = old_meta.model_copy(update={"duration": duration})
    write_model(p.meta, meta)
    msg = f"{meta.clean_artist or '?'} — {meta.clean_title} ({duration:.1f} s)"
    ctx.progress(1.0, msg)


def _settle_video(info: dict[str, Any], target: Path) -> None:
    """Make sure the downloaded file ends up at ``target`` (yt-dlp may pick another extension)."""
    if target.exists():
        return
    for d in info.get("requested_downloads") or []:
        fp = d.get("filepath")
        if fp and Path(fp).exists():
            os.replace(fp, target)
            return


class _YtLogger:
    """Route yt-dlp messages to the stage log instead of stdout."""

    def __init__(self, ctx: StageContext):
        self.log = ctx.log

    def debug(self, msg: str) -> None:
        if not msg.startswith("[debug] "):
            self.log.debug(msg)

    def info(self, msg: str) -> None:
        self.log.info(msg)

    def warning(self, msg: str) -> None:
        self.log.warning(msg)

    def error(self, msg: str) -> None:
        self.log.error(msg)
