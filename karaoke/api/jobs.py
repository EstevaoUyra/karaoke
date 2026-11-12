"""Jobs: list, create (YouTube URL or uploaded audio), run / re-run from a stage, cancel, logs, media."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from karaoke.api.deps import issue_dict, job_paths, require_idle, state
from karaoke.api.queue import RunRequest
from karaoke.models import Job, JobSettings, Meta, StageName, StageState, StageStatus, now
from karaoke.pipeline import ORDER, STAGE_BY_NAME, stage_hash
from karaoke.store import JobPaths, JobStore, atomic_write_text, read_model, read_model_opt
from karaoke.validate import validate_job

router = APIRouter(prefix="/api")

# Files the browser may fetch from a job folder (audio/video/images). JSON goes through typed routes.
MEDIA_PREFIXES = ("stems/", "output/", "assets/")
MEDIA_FILES = ("audio.wav", "source.mp4")
AUDIO_UPLOAD_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".mkv", ".mov"}


# --------------------------------------------------------------------------- summaries


def _file_info(p: Path) -> dict | None:
    if not p.exists():
        return None
    s = p.stat()
    return {"size": s.st_size, "mtime": s.st_mtime}


def job_summary(request: Request, job: Job) -> dict:
    st = state(request)
    meta = _safe_meta(st.store.paths(job.id))
    return {
        "id": job.id,
        "url": job.url,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "title": (meta.clean_title or meta.title) if meta else None,
        "artist": (meta.clean_artist or meta.artist or meta.uploader) if meta else None,
        "duration": meta.duration if meta else None,
        "thumbnail": meta.thumbnail if meta else None,
        "stages": {name.value: {"status": s.status.value, "progress": s.progress} for name, s in job.stages.items()},
        "queue": st.queue.state_for(job.id),
    }


def _safe_meta(p: JobPaths) -> Meta | None:
    try:
        return read_model_opt(p.meta, Meta)
    except Exception:  # noqa: BLE001 — a broken meta.json shows up in `issues`, not as a 500
        return None


def _list_jobs(store: JobStore) -> list[Job]:
    if not store.jobs_dir.exists():
        return []
    jobs = []
    for d in store.jobs_dir.iterdir():
        if (d / "job.json").exists():
            try:
                jobs.append(read_model(d / "job.json", Job))
            except Exception:  # noqa: BLE001 — skip unreadable jobs instead of failing the list
                continue
    return sorted(jobs, key=lambda j: j.created_at, reverse=True)


@router.get("/config")
def config(request: Request) -> dict:
    s = state(request).settings
    return {"stages": [s.value for s in ORDER], "jobs_dir": str(state(request).store.jobs_dir),
            "host": s.host, "port": s.port}


@router.get("/jobs")
def list_jobs(request: Request) -> list[dict]:
    return [job_summary(request, j) for j in _list_jobs(state(request).store)]


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    p = job_paths(request, job_id)
    st = state(request)
    job = st.store.load(job_id)
    return {
        "job": job.model_dump(mode="json"),
        "meta": (m.model_dump(mode="json") if (m := _safe_meta(p)) else None),
        "summary": job_summary(request, job),
        "files": {
            "audio": _file_info(p.audio),
            "instrumental": _file_info(p.instrumental),
            "vocals": _file_info(p.vocals),
            "lyrics_final": _file_info(p.lyrics_final),
            "background": _file_info(p.background),
            "preview_video": _file_info(p.preview_video),
            "final_video": _file_info(p.final_video),
            "user_lyrics": _file_info(p.lyrics_source("user.txt")),
            "review_notes": _file_info(p.review_notes),
            "melody": _file_info(p.melody),
            "structure": _file_info(p.structure),
        },
        "issues": [issue_dict(i) for i in validate_job(p)],
    }


# --------------------------------------------------------------------------- create


class CreateJob(BaseModel):
    url: str = Field(min_length=1)
    lyrics: str | None = None  # pasted lyrics → lyrics/sources/user.txt (always wins)
    run: bool = True


def _write_user_lyrics(p: JobPaths, text: str | None) -> None:
    f = p.lyrics_source("user.txt")
    if text and text.strip():
        atomic_write_text(f, text.strip() + "\n")
    else:
        f.unlink(missing_ok=True)


@router.post("/jobs", status_code=201)
def create_job(request: Request, body: CreateJob) -> dict:
    url = body.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(422, "paste a full link, starting with https://")
    st = state(request)
    job = st.store.create(url=url)
    if body.lyrics is not None and body.lyrics.strip():
        _write_user_lyrics(st.store.paths(job.id), body.lyrics)
    run = st.queue.submit(RunRequest(job.id)).to_dict() if body.run else None
    return {"job": job_summary(request, job), "run": run}


@router.post("/jobs/upload", status_code=201)
def upload_audio(
    request: Request,
    file: Annotated[UploadFile, File()],
    title: Annotated[str, Form()] = "",
    artist: Annotated[str, Form()] = "",
    lyrics: Annotated[str, Form()] = "",
    run: Annotated[bool, Form()] = True,
) -> dict:
    """Create a job from an audio (or video) file instead of a YouTube link.

    The API does the ingest step itself (audio.wav + meta.json) and records the ingest stage as done,
    so the pipeline continues with separation.
    """
    from karaoke.stages.ingest import audio_duration, extract_audio

    st = state(request)
    name = Path(file.filename or "upload").name
    if Path(name).suffix.lower() not in AUDIO_UPLOAD_SUFFIXES:
        raise HTTPException(422, f"unsupported file type {Path(name).suffix!r}: use wav, mp3, m4a, flac or ogg")
    with tempfile.TemporaryDirectory(prefix="karaoke-upload-") as tmp:
        src = Path(tmp) / f"upload{Path(name).suffix.lower()}"
        h = hashlib.sha1()
        with src.open("wb") as out:
            while chunk := file.file.read(1 << 20):
                h.update(chunk)
                out.write(chunk)
        job_id = "upload-" + h.hexdigest()[:10]
        store = st.store
        if store.exists(job_id):
            job = store.load(job_id)
            return {"job": job_summary(request, job), "run": None}
        p = store.paths(job_id)
        p.ensure_dirs()
        try:
            extract_audio(st.settings.ffmpeg_bin, src, p.audio)
            duration = round(audio_duration(p.audio), 3)
        except Exception as exc:
            shutil.rmtree(p.root, ignore_errors=True)
            raise HTTPException(422, f"could not read the audio in {name}: {exc}") from exc

    stem = Path(name).stem
    meta = Meta(title=title.strip() or stem, artist=artist.strip() or None, duration=duration,
                clean_title=title.strip() or stem, clean_artist=artist.strip() or None)
    from karaoke.store import write_model

    write_model(p.meta, meta)
    job = Job(id=job_id)
    spec = STAGE_BY_NAME[StageName.ingest]
    job.stages[StageName.ingest] = StageState(
        status=StageStatus.done, input_hash=stage_hash(spec, p, job, st.settings), started_at=now(),
        finished_at=now(), duration_s=0.0, progress=1.0, message=f"uploaded {name} ({duration:.1f} s)",
    )
    store.save(job)
    if lyrics.strip():
        _write_user_lyrics(p, lyrics)
    req = st.queue.submit(RunRequest(job_id)).to_dict() if run else None
    return {"job": job_summary(request, job), "run": req}


# --------------------------------------------------------------------------- run / cancel / queue


class RunBody(BaseModel):
    from_stage: StageName | None = None
    only: list[StageName] | None = None
    force: bool = False


@router.post("/jobs/{job_id}/run", status_code=202)
def run_job_route(request: Request, job_id: str, body: RunBody | None = None) -> dict:
    job_paths(request, job_id)
    body = body or RunBody()
    req = RunRequest(job_id, from_stage=body.from_stage, only=body.only, force=body.force)
    return state(request).queue.submit(req).to_dict()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> dict:
    job_paths(request, job_id)
    return {"cancelled": state(request).queue.cancel(job_id)}


@router.get("/queue")
def queue_snapshot(request: Request) -> dict:
    return state(request).queue.snapshot()


# --------------------------------------------------------------------------- settings, user lyrics


@router.get("/jobs/{job_id}/settings")
def get_settings_route(request: Request, job_id: str) -> dict:
    job_paths(request, job_id)
    return state(request).store.load(job_id).settings.model_dump(mode="json")


@router.put("/jobs/{job_id}/settings")
def put_settings(request: Request, job_id: str, body: JobSettings) -> dict:
    job_paths(request, job_id)
    require_idle(request, job_id)
    store = state(request).store
    job = store.load(job_id)
    job.settings = body
    store.save(job)
    return job.settings.model_dump(mode="json")


class UserLyrics(BaseModel):
    text: str = ""
    run: bool = False  # re-run from lyrics_fetch so the pasted text is aligned


@router.get("/jobs/{job_id}/lyrics/sources")
def lyrics_sources(request: Request, job_id: str) -> dict:
    p = job_paths(request, job_id)
    files = []
    if p.lyrics_sources_dir.exists():
        for f in sorted(p.lyrics_sources_dir.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                files.append({"name": f.name, "text": f.read_text(encoding="utf-8", errors="replace")})
    chosen = state(request).store.lyrics_choice(job_id)
    chosen_text = p.lyrics_chosen_text.read_text(encoding="utf-8") if p.lyrics_chosen_text.exists() else None
    return {"chosen": chosen.model_dump(mode="json") if chosen else None, "chosen_text": chosen_text,
            "files": files, "letras": _letras_pointer(p)}


def _letras_pointer(p: JobPaths) -> dict | None:
    """lyrics/sources/letras.json (written by lyrics_fetch): the song's letras.mus.br page, which the
    user opens to copy the lyrics from (the site blocks automated clients, so we never fetch it)."""
    f = p.lyrics_source("letras.json")
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    url = d.get("url") if isinstance(d, dict) else None
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        return None
    return {"url": url, "title": str(d.get("title") or ""), "artist": str(d.get("artist") or "")}


@router.put("/jobs/{job_id}/lyrics/user")
def put_user_lyrics(request: Request, job_id: str, body: UserLyrics) -> dict:
    p = job_paths(request, job_id)
    _write_user_lyrics(p, body.text)
    run = None
    if body.run:
        run = state(request).queue.submit(RunRequest(job_id, from_stage=StageName.lyrics_fetch)).to_dict()
    return {"saved": bool(body.text.strip()), "run": run}


# --------------------------------------------------------------------------- logs and media


@router.get("/jobs/{job_id}/logs/{stage}")
def stage_log(request: Request, job_id: str, stage: StageName, lines: int = 200) -> dict:
    p = job_paths(request, job_id)
    f = p.log(stage.value)
    if not f.exists():
        return {"stage": stage.value, "text": "", "size": 0}
    size = f.stat().st_size
    with f.open("rb") as fh:  # read only the end of big logs
        fh.seek(max(0, size - 256 * 1024))
        text = fh.read().decode("utf-8", errors="replace")
    tail = "\n".join(text.splitlines()[-max(1, min(lines, 5000)):])
    return {"stage": stage.value, "text": tail, "size": size}


@router.get("/jobs/{job_id}/files/{path:path}")
def job_file(request: Request, job_id: str, path: str, download: bool = False) -> FileResponse:
    """Stems, images and videos. FileResponse answers HTTP Range requests, so media can seek."""
    p = job_paths(request, job_id)
    if not (path in MEDIA_FILES or path.startswith(MEDIA_PREFIXES)):
        raise HTTPException(404, "not a media file")
    f = (p.root / path).resolve()
    root = p.root.resolve()
    if root not in f.parents or not f.is_file() or any(part.startswith(".") for part in f.relative_to(root).parts):
        raise HTTPException(404, f"no file {path}")
    media_type = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
    filename = f"{job_id}-{f.name}" if download else None
    return FileResponse(f, media_type=media_type, filename=filename,
                        content_disposition_type="attachment" if download else "inline",
                        headers={"Cache-Control": "no-cache"})
