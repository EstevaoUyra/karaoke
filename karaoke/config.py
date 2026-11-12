"""Runtime configuration. Values come from env vars (prefix ``KARAOKE_``) or ``.env``."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# Homebrew's plain `ffmpeg` has no libass. Candidates with libass, in order: the minimal build from
# scripts/build_ffmpeg.sh, then Homebrew `ffmpeg-full` (keg-only; impractical to build on Intel).
_FFMPEG_FULL_CANDIDATES = [
    Path.home() / ".local" / "opt" / "ffmpeg-karaoke" / "bin",
    Path("/usr/local/opt/ffmpeg-full/bin"),
    Path("/opt/homebrew/opt/ffmpeg-full/bin"),
]


def _default_ffmpeg_dir() -> Path | None:
    for d in _FFMPEG_FULL_CANDIDATES:
        if (d / "ffmpeg").exists():
            return d
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KARAOKE_", env_file=REPO_ROOT / ".env", extra="ignore"
    )

    data_dir: Path = REPO_ROOT / "data"

    # --- binaries ---
    ffmpeg: str = Field(default="")  # resolved in `ffmpeg_bin`
    ffprobe: str = Field(default="")
    yt_js_runtime: str = "node"  # yt-dlp needs a JS runtime for YouTube

    # --- models ---
    whisper_model: str = "large-v3-turbo"  # faster-whisper alias → mobiuslabsgmbh/faster-whisper-large-v3-turbo
    whisper_compute_type: str = "int8"
    whisper_threads: int = Field(default_factory=lambda: os.cpu_count() or 4)
    separator_model: str = "Kim_Vocal_2.onnx"  # fast default (~1x real time); see docs/benchmarks.md
    separator_model_hq: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"  # ~10x real time
    melody_tracker: str = "crepe-tiny"  # "crepe-tiny" | "crepe-full" | "pyin"; see docs/benchmarks.md
    separator_models_dir: Path = Path.home() / ".cache" / "audio-separator-models"

    # --- optional API keys (read from the unprefixed env names shown in .env.example) ---
    genius_token: str | None = Field(default=None, validation_alias="GENIUS_ACCESS_TOKEN")
    pexels_api_key: str | None = Field(default=None, validation_alias="PEXELS_API_KEY")
    unsplash_access_key: str | None = Field(default=None, validation_alias="UNSPLASH_ACCESS_KEY")
    # An email or a URL of yours. Wikimedia asks automated clients to say who runs them (in the
    # User-Agent) and refuses many picture downloads from the ones that do not.
    contact: str | None = None  # KARAOKE_CONTACT

    # --- web ---
    host: str = "127.0.0.1"
    port: int = 8765

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def ffmpeg_bin(self) -> str:
        if self.ffmpeg:
            return self.ffmpeg
        d = _default_ffmpeg_dir()
        return str(d / "ffmpeg") if d else (shutil.which("ffmpeg") or "ffmpeg")

    @property
    def ffprobe_bin(self) -> str:
        if self.ffprobe:
            return self.ffprobe
        d = _default_ffmpeg_dir()
        return str(d / "ffprobe") if d else (shutil.which("ffprobe") or "ffprobe")


@lru_cache
def get_settings() -> Settings:
    return Settings()
