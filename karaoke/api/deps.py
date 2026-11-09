"""Shared state and helpers for the API routers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from fastapi import HTTPException, Request

from karaoke.api.queue import JobQueue
from karaoke.api.sse import EventHub
from karaoke.config import Settings
from karaoke.store import JobPaths, JobStore
from karaoke.validate import Issue

JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class AppState:
    store: JobStore
    settings: Settings
    hub: EventHub
    queue: JobQueue


def state(request: Request) -> AppState:
    return request.app.state.karaoke


def job_paths(request: Request, job_id: str) -> JobPaths:
    """Paths of an existing job, or 404. Also guards against path tricks in the id."""
    if not JOB_ID.match(job_id):
        raise HTTPException(404, f"no job {job_id!r}")
    st = state(request)
    if not st.store.exists(job_id):
        raise HTTPException(404, f"no job {job_id!r}")
    return st.store.paths(job_id)


def issue_dict(i: Issue) -> dict:
    return {"severity": i.severity, "file": i.file, "where": i.where, "message": i.message}


def require_idle(request: Request, job_id: str) -> None:
    """job.json is written by the worker while a job runs: refuse settings changes meanwhile."""
    q = state(request).queue.state_for(job_id)
    if q == "running":
        raise HTTPException(409, "the job is running: wait for it to finish or cancel it")
