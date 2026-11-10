"""Server-sent events: one hub that every part of the backend publishes to, one stream per browser tab.

Event types (``event:`` field of the SSE message, JSON ``data``):

- ``stage``: a ``pipeline.StageEvent`` from the running job (status, progress, message, error).
- ``queue``: the job queue changed (``JobQueue.snapshot()``).
- ``files``: files changed inside a job folder (``{"job_id": ..., "paths": [...]}``), from the folder
  watcher. Edits made by a Claude session or the CLI show up in the UI this way.
- ``jobs``: a job folder appeared or disappeared.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from typing import Any


class EventHub:
    """Thread-safe fan-out: ``publish`` may be called from any thread (the queue worker does)."""

    def __init__(self, maxsize: int = 1000):
        self._subs: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    @property
    def subscribers(self) -> int:
        return len(self._subs)

    def publish(self, type_: str, data: Any) -> None:
        msg = {"event": type_, "data": data}
        with self._lock:
            subs = list(self._subs)
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(_put, q, msg)
            except RuntimeError:  # loop closed: the subscriber is gone
                with self._lock:
                    self._subs.discard((loop, q))

    def subscribe(self) -> tuple[asyncio.AbstractEventLoop, asyncio.Queue]:
        sub = (asyncio.get_running_loop(), asyncio.Queue(self._maxsize))
        with self._lock:
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: tuple[asyncio.AbstractEventLoop, asyncio.Queue]) -> None:
        with self._lock:
            self._subs.discard(sub)

    async def stream(self) -> AsyncIterator[dict[str, str]]:
        """Messages formatted for ``sse_starlette.EventSourceResponse``."""
        sub = self.subscribe()
        try:
            yield format_event("hello", {})
            while True:
                msg = await sub[1].get()
                yield format_event(msg["event"], msg["data"])
        finally:
            self.unsubscribe(sub)


def _put(q: asyncio.Queue, msg: dict) -> None:
    try:
        q.put_nowait(msg)
    except asyncio.QueueFull:  # a stalled tab: drop rather than grow without bound
        pass


def format_event(type_: str, data: Any) -> dict[str, str]:
    return {"event": type_, "data": json.dumps(data, default=str)}
