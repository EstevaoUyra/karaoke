"""Watch the jobs folder and publish ``files`` / ``jobs`` events, so edits made outside the UI
(a Claude session, the CLI, a text editor) show up live in every open tab."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path

from karaoke.api.sse import EventHub

IGNORED_SUFFIXES = (".tmp", ".part")


def classify(jobs_dir: Path, changed: Iterable[str | Path]) -> tuple[dict[str, set[str]], set[str]]:
    """Map changed paths to ``({job_id: {relative paths}}, {job ids whose folder itself changed})``.

    Hidden files (atomic-write temp files ``.x.tmp``, ffmpeg's ``.x.part``, ``.DS_Store``) are ignored.
    """
    files: dict[str, set[str]] = {}
    jobs: set[str] = set()
    root = jobs_dir.resolve()
    for raw in changed:
        try:
            rel = Path(raw).resolve().relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if not parts or any(p.startswith(".") for p in parts) or rel.name.endswith(IGNORED_SUFFIXES):
            continue
        if len(parts) == 1:
            jobs.add(parts[0])
            continue
        files.setdefault(parts[0], set()).add("/".join(parts[1:]))
        if parts[1:] == ("job.json",):
            jobs.add(parts[0])
    return files, jobs


async def watch_jobs(jobs_dir: Path, hub: EventHub, stop: asyncio.Event, debounce_ms: int = 250) -> None:
    from watchfiles import awatch

    jobs_dir.mkdir(parents=True, exist_ok=True)
    async for changes in awatch(jobs_dir, debounce=debounce_ms, step=50, stop_event=stop, recursive=True):
        files, jobs = classify(jobs_dir, (path for _, path in changes))
        for job_id, paths in files.items():
            hub.publish("files", {"job_id": job_id, "paths": sorted(paths)})
        for job_id in jobs:
            hub.publish("jobs", {"job_id": job_id})
