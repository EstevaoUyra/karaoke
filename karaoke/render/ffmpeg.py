"""Small ffmpeg/ffprobe helpers: capability checks, a progress-reporting runner, probing."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

ProgressCallback = Callable[[float], None]  # called with ffmpeg's out_time in seconds


class FFmpegError(RuntimeError):
    """ffmpeg exited with an error. ``stderr_tail`` holds its last lines of output."""

    def __init__(self, message: str, stderr_tail: str = ""):
        super().__init__(message + (f"\n{stderr_tail}" if stderr_tail else ""))
        self.stderr_tail = stderr_tail


@lru_cache(maxsize=16)
def filters(ffmpeg_bin: str) -> frozenset[str]:
    """Names of the filters this ffmpeg build has (empty if the binary can't run)."""
    try:
        out = subprocess.run([ffmpeg_bin, "-hide_banner", "-filters"], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        # " TSC ass               V->V       Render ASS subtitles ..."
        if len(parts) >= 3 and "->" in parts[2]:
            names.add(parts[1])
    return frozenset(names)


def has_filter(ffmpeg_bin: str, name: str) -> bool:
    return name in filters(ffmpeg_bin)


def has_libass(ffmpeg_bin: str) -> bool:
    """True if this ffmpeg can burn ASS subtitles (the ``ass`` filter needs libass)."""
    return has_filter(ffmpeg_bin, "ass")


def _parse_out_time(key: str, value: str) -> float | None:
    if value in ("", "N/A") or value.startswith("-"):
        return None
    try:
        if key in ("out_time_us", "out_time_ms"):  # both are microseconds (a long-standing ffmpeg quirk)
            return int(value) / 1_000_000
        h, m, s = value.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        return None


def run_ffmpeg(
    ffmpeg_bin: str,
    args: Sequence[str],
    *,
    on_progress: ProgressCallback | None = None,
    log: logging.Logger | None = None,
    cwd: Path | None = None,
) -> None:
    """Run ``ffmpeg <args>`` with ``-progress pipe:1`` and stream out_time (s) to ``on_progress``.

    Raises FFmpegError with the tail of stderr on a non-zero exit.
    """
    cmd = [ffmpeg_bin, "-hide_banner", "-nostdin", "-loglevel", "warning", "-nostats",
           "-progress", "pipe:1", *args]
    if log:
        log.info("ffmpeg: %s", subprocess.list2cmdline(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd,
                            bufsize=1)
    tail: deque[str] = deque(maxlen=40)

    def _drain_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                tail.append(line)
                if log:
                    log.info("ffmpeg| %s", line)

    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key in ("out_time_us", "out_time") and on_progress:
                secs = _parse_out_time(key, value)
                if secs is not None:
                    on_progress(secs)
        proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        t.join(timeout=5)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg failed with exit code {proc.returncode}", "\n".join(tail))


def ffprobe(path: Path | str, ffprobe_bin: str) -> dict:
    """``ffprobe -show_format -show_streams`` as a dict."""
    out = subprocess.run(
        [ffprobe_bin, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}", out.stderr.strip())
    return json.loads(out.stdout)


def probe_duration(path: Path | str, ffprobe_bin: str) -> float:
    return float(ffprobe(path, ffprobe_bin)["format"]["duration"])


def extract_frame(ffmpeg_bin: str, video: Path, t: float, out_png: Path) -> None:
    """Write the frame at time ``t`` (s) of ``video`` to ``out_png``."""
    run_ffmpeg(ffmpeg_bin, ["-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", str(out_png)])
