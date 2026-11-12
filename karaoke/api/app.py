"""FastAPI app: REST over the job folder, a single-worker job queue, SSE, and the built web UI.

    uv run karaoke serve            # http://127.0.0.1:8765

The UI is built into ``web/dist`` (``npm run build`` in ``web/``) and served from ``/``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from sse_starlette import EventSourceResponse

from karaoke.api import backgrounds, jobs, lyrics, melody, render, youtube
from karaoke.api.deps import AppState, state
from karaoke.api.queue import JobQueue
from karaoke.api.sse import EventHub
from karaoke.api.watch import watch_jobs
from karaoke.config import REPO_ROOT, Settings, get_settings
from karaoke.store import JobStore

WEB_DIST = REPO_ROOT / "web" / "dist"

_NO_BUILD = """<!doctype html><meta charset="utf-8"><title>karaoke</title>
<body style="font-family: system-ui; max-width: 40rem; margin: 4rem auto; line-height: 1.5">
<h1>The web UI is not built yet</h1>
<p>Run <code>npm install &amp;&amp; npm run build</code> in <code>web/</code>, then reload.
For development, run <code>npm run dev</code> there and open the address it prints.</p>
<p>The API is up: <a href="/api/jobs">/api/jobs</a>.</p></body>"""


def create_app(
    store: JobStore | None = None,
    settings: Settings | None = None,
    *,
    queue: JobQueue | None = None,
    watch: bool = True,
    web_dist: Path | None = WEB_DIST,
) -> FastAPI:
    settings = settings or get_settings()
    store = store or JobStore(settings.jobs_dir)
    store.jobs_dir.mkdir(parents=True, exist_ok=True)
    hub = EventHub()
    queue = queue or JobQueue(store, hub.publish)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        queue.start()
        stop = asyncio.Event()
        task = asyncio.create_task(watch_jobs(store.jobs_dir, hub, stop)) if watch else None
        try:
            yield
        finally:
            stop.set()
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await asyncio.to_thread(queue.stop)

    app = FastAPI(title="karaoke", lifespan=lifespan)
    app.state.karaoke = AppState(store=store, settings=settings, hub=hub, queue=queue)
    for r in (jobs.router, lyrics.router, melody.router, backgrounds.router, render.router, youtube.router):
        app.include_router(r)

    @app.get("/api/events")
    async def events(request: Request) -> EventSourceResponse:
        return EventSourceResponse(state(request).hub.stream(), ping=15)

    @app.get("/api/{rest:path}", include_in_schema=False)
    def api_404(rest: str) -> None:
        raise HTTPException(404, f"no API route /api/{rest}")

    dist = web_dist if web_dist and (web_dist / "index.html").exists() else None

    @app.get("/{path:path}", include_in_schema=False, response_model=None)
    def spa(path: str) -> FileResponse | HTMLResponse:
        if dist is None:
            return HTMLResponse(_NO_BUILD)
        f = (dist / path).resolve()
        if path and f.is_file() and dist.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def serve(host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    s = get_settings()
    uvicorn.run(create_app(settings=s), host=host or s.host, port=port or s.port, log_level="info")
