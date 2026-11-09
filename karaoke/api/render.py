"""Render: save the render settings and queue the render stage on its own."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from karaoke.api.deps import job_paths, require_idle, state
from karaoke.api.queue import RunRequest
from karaoke.models import RenderSettings, StageName

router = APIRouter(prefix="/api")


class RenderBody(BaseModel):
    render: RenderSettings | None = None  # None: keep the job's current settings


@router.post("/jobs/{job_id}/render", status_code=202)
def render(request: Request, job_id: str, body: RenderBody | None = None) -> dict:
    job_paths(request, job_id)
    st = state(request)
    if body and body.render:
        require_idle(request, job_id)
        job = st.store.load(job_id)
        job.settings.render = body.render
        st.store.save(job)
    req = st.queue.submit(RunRequest(job_id, only=[StageName.render]))
    return req.to_dict()
