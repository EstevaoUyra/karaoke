"""Job folder layout and IO. The job folder is the API: UI, CLI and AI skills all use these paths."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from karaoke.config import get_settings
from karaoke.models import (
    Background,
    Job,
    LyricsChoice,
    LyricsFinal,
    Melody,
    Meta,
    Transcript,
    now,
)

M = TypeVar("M", bound=BaseModel)

_YT_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})")


@dataclass(frozen=True)
class JobPaths:
    """Every file a job can contain. Relative layout is part of the frozen contract."""

    root: Path

    @property
    def id(self) -> str:
        return self.root.name

    # metadata
    @property
    def job(self) -> Path:
        return self.root / "job.json"

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    # media
    @property
    def source_video(self) -> Path:
        return self.root / "source.mp4"

    @property
    def audio(self) -> Path:
        return self.root / "audio.wav"

    @property
    def stems_dir(self) -> Path:
        return self.root / "stems"

    @property
    def vocals(self) -> Path:
        return self.stems_dir / "vocals.wav"

    @property
    def instrumental(self) -> Path:
        return self.stems_dir / "instrumental.wav"

    # lyrics
    @property
    def lyrics_dir(self) -> Path:
        return self.root / "lyrics"

    @property
    def lyrics_sources_dir(self) -> Path:
        return self.lyrics_dir / "sources"

    def lyrics_source(self, filename: str) -> Path:
        """e.g. lyrics_source('lrclib.lrc'), ('lrclib.txt'), ('genius.txt'), ('youtube.vtt'), ('user.txt')."""
        return self.lyrics_sources_dir / filename

    @property
    def lyrics_choice(self) -> Path:
        return self.lyrics_dir / "chosen.json"

    @property
    def lyrics_chosen_text(self) -> Path:
        return self.lyrics_dir / "chosen.txt"

    @property
    def transcript(self) -> Path:
        return self.root / "transcript.whisper.json"

    @property
    def lyrics_final(self) -> Path:
        return self.root / "lyrics.final.json"

    @property
    def review_notes(self) -> Path:
        return self.root / "review.md"  # written by the lyrics-review skill

    # melody (pitch guide)
    @property
    def melody(self) -> Path:
        return self.root / "melody.json"

    @property
    def melody_f0(self) -> Path:
        return self.root / "cache" / "melody.f0.npz"  # raw pyin contour, reused while vocals.wav is unchanged

    # backgrounds
    @property
    def background(self) -> Path:
        return self.root / "background.json"

    @property
    def art_direction(self) -> Path:
        return self.root / "art_direction.md"  # written by the backgrounds skill

    @property
    def bg_assets_dir(self) -> Path:
        return self.root / "assets" / "bg"

    @property
    def bg_credits(self) -> Path:
        return self.bg_assets_dir / "credits.json"

    # output
    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def ass(self) -> Path:
        return self.output_dir / "karaoke.ass"

    @property
    def preview_video(self) -> Path:
        return self.output_dir / "preview.mp4"

    @property
    def final_video(self) -> Path:
        return self.output_dir / "karaoke.mp4"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def log(self, stage: str) -> Path:
        return self.logs_dir / f"{stage}.log"

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root))

    def ensure_dirs(self) -> None:
        for d in (self.stems_dir, self.lyrics_sources_dir, self.bg_assets_dir, self.output_dir,
                  self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- generic IO


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_model(path: Path, model: BaseModel) -> None:
    atomic_write_text(path, model.model_dump_json(indent=2) + "\n")


def read_model(path: Path, cls: type[M]) -> M:
    return cls.model_validate_json(path.read_text(encoding="utf-8"))


def read_model_opt(path: Path, cls: type[M]) -> M | None:
    return read_model(path, cls) if path.exists() else None


# --------------------------------------------------------------------------- hashing


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inputs_hash(paths: Iterable[Path], extra: object = None) -> str:
    """Hash of input files (missing files hash as 'missing') plus any JSON-able settings."""
    h = hashlib.sha1()
    for p in sorted(paths, key=str):
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    h.update(f"{f}:{file_hash(f)}".encode())
        else:
            h.update(f"{p}:{file_hash(p) if p.exists() else 'missing'}".encode())
    h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- job store


def job_id_for_url(url: str) -> str:
    m = _YT_ID.search(url)
    if m:
        return m.group(1)
    return "url-" + hashlib.sha1(url.encode()).hexdigest()[:10]


def clean_youtube_url(url: str) -> str:
    """Drop playlist/radio params: keep only the video."""
    m = _YT_ID.search(url)
    return f"https://www.youtube.com/watch?v={m.group(1)}" if m else url


class JobStore:
    def __init__(self, jobs_dir: Path | None = None):
        self.jobs_dir = jobs_dir or get_settings().jobs_dir

    def paths(self, job_id: str) -> JobPaths:
        return JobPaths(self.jobs_dir / job_id)

    def exists(self, job_id: str) -> bool:
        return self.paths(job_id).job.exists()

    def create(self, url: str | None = None, job_id: str | None = None) -> Job:
        if url:
            url = clean_youtube_url(url)
        job_id = job_id or (job_id_for_url(url) if url else None)
        if not job_id:
            raise ValueError("need a url or a job_id")
        p = self.paths(job_id)
        if p.job.exists():
            return self.load(job_id)
        p.ensure_dirs()
        job = Job(id=job_id, url=url)
        self.save(job)
        return job

    def load(self, job_id: str) -> Job:
        return read_model(self.paths(job_id).job, Job)

    def save(self, job: Job) -> None:
        job.updated_at = now()
        write_model(self.paths(job.id).job, job)

    def list(self) -> list[Job]:
        if not self.jobs_dir.exists():
            return []
        jobs = [read_model(d / "job.json", Job) for d in self.jobs_dir.iterdir() if (d / "job.json").exists()]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    # typed accessors for the main files
    def meta(self, job_id: str) -> Meta | None:
        return read_model_opt(self.paths(job_id).meta, Meta)

    def transcript(self, job_id: str) -> Transcript | None:
        return read_model_opt(self.paths(job_id).transcript, Transcript)

    def lyrics_choice(self, job_id: str) -> LyricsChoice | None:
        return read_model_opt(self.paths(job_id).lyrics_choice, LyricsChoice)

    def lyrics_final(self, job_id: str) -> LyricsFinal | None:
        return read_model_opt(self.paths(job_id).lyrics_final, LyricsFinal)

    def background(self, job_id: str) -> Background | None:
        return read_model_opt(self.paths(job_id).background, Background)

    def melody(self, job_id: str) -> Melody | None:
        return read_model_opt(self.paths(job_id).melody, Melody)
