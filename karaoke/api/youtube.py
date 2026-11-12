"""YouTube: the draft of a job's upload (output/youtube.json), the upload itself, and later edits.

Uploads run in a thread of their own, one at a time (the pipeline's queue is for stages). Each
reports progress as a ``youtube`` event ({job_id, status, message}); when it ends a ``jobs`` event
refreshes the song list, where uploaded songs carry a YouTube badge. The first upload with no
saved sign-in opens Google's sign-in page in the browser of the machine running the server.
"""

from __future__ import annotations

import threading
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from karaoke import youtube as yt
from karaoke.api.deps import job_paths, state
from karaoke.models import YouTubeUpload
from karaoke.store import JobPaths, read_model_opt, write_model

router = APIRouter(prefix="/api")

_lock = threading.Lock()  # one upload at a time
_tasks: dict[str, dict[str, Any]] = {}  # job id -> {"status": queued|running|done|failed, "message": str}


def summary(p: JobPaths) -> dict | None:
    """What the song list shows about a job's YouTube upload."""
    try:
        y = read_model_opt(p.youtube, YouTubeUpload)
    except Exception:  # noqa: BLE001 — a broken youtube.json shows as not uploaded
        return None
    task = _tasks.get(p.id)
    if not y and not task:
        return None
    return {
        "video_id": y.video_id if y else None,
        "privacy": y.privacy if y else None,
        "uploaded_at": y.uploaded_at if y else None,
        "task": task["status"] if task and task["status"] in ("queued", "running") else None,
    }


def _status(request: Request, p: JobPaths) -> dict:
    s = state(request).settings
    try:
        draft = read_model_opt(p.youtube, YouTubeUpload)
        bad = None
    except Exception as exc:  # noqa: BLE001
        draft, bad = None, f"output/youtube.json cannot be read: {exc}"
    return {
        "draft": draft.model_dump(mode="json") if draft else None,
        "errors": ([bad] if bad else []) + (yt.check(draft) if draft else []),
        "video_ready": p.final_video.exists(),
        "video_changed": bool(draft and draft.video_sha1 and p.final_video.exists()
                              and p.final_video.stat().st_mtime > draft.uploaded_at.timestamp()
                              if draft and draft.uploaded_at else False),
        "client_secret": s.youtube_client_secret.exists(),
        "signed_in": s.youtube_token.exists(),
        "task": _tasks.get(p.id),
    }


@router.get("/jobs/{job_id}/youtube")
def get_youtube(request: Request, job_id: str) -> dict:
    return _status(request, job_paths(request, job_id))


class DraftBody(BaseModel):
    redo: bool = False


@router.post("/jobs/{job_id}/youtube/draft")
def make_draft(request: Request, job_id: str, body: DraftBody | None = None) -> dict:
    p = job_paths(request, job_id)
    try:
        yt.draft(p, redo=bool(body and body.redo))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _status(request, p)


class DraftEdit(BaseModel):
    title: str
    description: str
    tags: list[str]
    privacy: Literal["private", "unlisted", "public"]
    playlist: str | None = None


@router.put("/jobs/{job_id}/youtube")
def save_draft(request: Request, job_id: str, body: DraftEdit) -> dict:
    """Save the fields a person edits; everything else in youtube.json is kept."""
    p = job_paths(request, job_id)
    y = read_model_opt(p.youtube, YouTubeUpload)
    if y is None:
        raise HTTPException(409, "no draft yet")
    tags = [t.strip() for t in body.tags if t.strip()]
    write_model(p.youtube, y.model_copy(update={**body.model_dump(), "tags": tags,
                                                "playlist": (body.playlist or "").strip() or None}))
    return _status(request, p)


def _publish(request: Request, job_id: str, status: str, message: str) -> None:
    _tasks[job_id] = {"status": status, "message": message}
    hub = state(request).hub
    hub.publish("youtube", {"job_id": job_id, "status": status, "message": message})
    if status in ("queued", "done", "failed"):
        hub.publish("jobs", {"job_id": job_id})


class UploadBody(BaseModel):
    again: bool = False


@router.post("/jobs/{job_id}/youtube/upload", status_code=202)
def start_upload(request: Request, job_id: str, body: UploadBody | None = None) -> dict:
    p = job_paths(request, job_id)
    if (_tasks.get(job_id) or {}).get("status") in ("queued", "running"):
        raise HTTPException(409, "this song is already uploading")
    y = read_model_opt(p.youtube, YouTubeUpload)
    if y is None:
        raise HTTPException(409, "no draft yet")
    again = bool(body and body.again)
    if y.video_id and not again:
        raise HTTPException(409, f"already on YouTube: https://youtu.be/{y.video_id}")
    errors = yt.check(y)
    if errors:
        raise HTTPException(422, "; ".join(errors))
    if not p.final_video.exists():
        raise HTTPException(409, "render the final video first")
    if not state(request).settings.youtube_client_secret.exists():
        raise HTTPException(409, yt.SETUP_HELP.format(path=state(request).settings.youtube_client_secret))

    def work() -> None:
        with _lock:
            signed_in = state(request).settings.youtube_token.exists()
            _publish(request, job_id, "running", "uploading…" if signed_in else
                     "sign in to Google in the browser window that opened")
            try:
                out = yt.upload(p, again=again, log=lambda m: _publish(request, job_id, "running", m))
                _publish(request, job_id, "done", f"on YouTube ({out.privacy}): https://youtu.be/{out.video_id}")
            except Exception as exc:  # noqa: BLE001 — shown in the UI
                _publish(request, job_id, "failed", str(exc))

    _publish(request, job_id, "queued", "waiting for another upload to finish" if _lock.locked() else "starting")
    threading.Thread(target=work, name=f"youtube-{job_id}", daemon=True).start()
    return _status(request, p)


class UpdateBody(BaseModel):
    privacy: Literal["private", "unlisted", "public"] | None = None


@router.post("/jobs/{job_id}/youtube/update")
def update_video(request: Request, job_id: str, body: UpdateBody | None = None) -> dict:
    """Send the saved title, description, tags and privacy to the video already uploaded."""
    p = job_paths(request, job_id)
    msgs: list[str] = []
    try:
        yt.update(p, privacy=body.privacy if body else None, log=msgs.append)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — YouTube refused or the network failed
        raise HTTPException(502, f"YouTube: {exc}") from exc
    state(request).hub.publish("jobs", {"job_id": job_id})
    return {**_status(request, p), "messages": msgs}
