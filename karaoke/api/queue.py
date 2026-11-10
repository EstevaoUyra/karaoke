"""Single-worker job queue: one pipeline run at a time (the machine is CPU-bound).

Each run is a ``karaoke.api.worker`` child process in its own session, so ``cancel`` can kill it
together with everything it started (isolated stages, ffmpeg).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

from karaoke.api.worker import EVENT_PREFIX
from karaoke.models import StageName, StageStatus, now
from karaoke.store import JobStore

RunState = Literal["queued", "running", "done", "failed", "cancelled"]
Publish = Callable[[str, object], None]


@dataclass
class RunRequest:
    job_id: str
    from_stage: StageName | None = None
    only: list[StageName] | None = None
    force: bool = False
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    state: RunState = "queued"
    error: str | None = None
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancel_requested: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id, "job_id": self.job_id,
            "from_stage": self.from_stage.value if self.from_stage else None,
            "only": [s.value for s in self.only] if self.only else None,
            "force": self.force, "state": self.state, "error": self.error,
            "queued_at": self.queued_at, "started_at": self.started_at, "finished_at": self.finished_at,
        }


def worker_command(jobs_dir: Path, req: RunRequest) -> list[str]:
    cmd = [sys.executable, "-m", "karaoke.api.worker", str(jobs_dir), req.job_id]
    if req.from_stage:
        cmd += ["--from", req.from_stage.value]
    if req.only:
        cmd += ["--only", *[s.value for s in req.only]]
    if req.force:
        cmd.append("--force")
    return cmd


class JobQueue:
    def __init__(self, store: JobStore, publish: Publish,
                 command: Callable[[Path, RunRequest], list[str]] = worker_command,
                 kill_grace_s: float = 5.0):
        self.store = store
        self.publish = publish
        self.command = command
        self.kill_grace_s = kill_grace_s
        self._pending: deque[RunRequest] = deque()
        self._history: deque[RunRequest] = deque(maxlen=30)
        self._running: RunRequest | None = None
        self._proc: subprocess.Popen | None = None
        self._cv = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread is None:
            self._stop = False
            self._thread = threading.Thread(target=self._loop, name="karaoke-queue", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._pending.clear()
            self._cv.notify_all()
        if self._running:
            self.cancel(self._running.job_id)
        if self._thread:
            self._thread.join(timeout=self.kill_grace_s + 5)
            self._thread = None

    # ------------------------------------------------------------------ API

    def submit(self, req: RunRequest) -> RunRequest:
        """Queue a run. A job already waiting in the queue gets its request replaced (one entry per job)."""
        with self._cv:
            for i, old in enumerate(self._pending):
                if old.job_id == req.job_id:
                    self._pending[i] = req
                    break
            else:
                self._pending.append(req)
            self._cv.notify_all()
        self._publish_queue()
        return req

    def cancel(self, job_id: str) -> bool:
        """Remove a queued run of this job, or kill its running one. True if anything was cancelled."""
        cancelled = False
        with self._cv:
            for old in list(self._pending):
                if old.job_id == job_id:
                    self._pending.remove(old)
                    old.state = "cancelled"
                    old.finished_at = time.time()
                    self._history.appendleft(old)
                    cancelled = True
            run, proc = self._running, self._proc
            if run and run.job_id == job_id and proc is not None and proc.poll() is None:
                run.cancel_requested = True
                _signal_group(proc, signal.SIGTERM)
                threading.Timer(self.kill_grace_s, _kill_if_alive, args=(proc,)).start()
                cancelled = True
        if cancelled:
            self._publish_queue()
        return cancelled

    def state_for(self, job_id: str) -> RunState | None:
        with self._cv:
            if self._running and self._running.job_id == job_id:
                return "running"
            if any(r.job_id == job_id for r in self._pending):
                return "queued"
        return None

    def snapshot(self) -> dict:
        with self._cv:
            return {
                "running": self._running.to_dict() if self._running else None,
                "queued": [r.to_dict() for r in self._pending],
                "history": [r.to_dict() for r in self._history],
            }

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """For tests and shutdown: block until nothing is running or queued."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._cv:
                if self._running is None and not self._pending:
                    return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------------ worker

    def _publish_queue(self) -> None:
        self.publish("queue", self.snapshot())

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                req = self._pending.popleft()
                req.state = "running"
                req.started_at = time.time()
                self._running = req
            self._publish_queue()
            try:
                self._execute(req)
            except Exception as exc:  # noqa: BLE001 — never let the worker thread die
                req.state = "failed"
                req.error = f"{type(exc).__name__}: {exc}"
            finally:
                req.finished_at = time.time()
                with self._cv:
                    self._running = None
                    self._proc = None
                    self._history.appendleft(req)
                self._publish_queue()
                self.publish("jobs", {"job_id": req.job_id})

    def _execute(self, req: RunRequest) -> None:
        cmd = self.command(self.store.jobs_dir, req)
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True,
                                bufsize=1, start_new_session=True)
        with self._cv:
            self._proc = proc
            if req.cancel_requested:  # cancelled while starting
                _signal_group(proc, signal.SIGTERM)
        error: str | None = None
        assert proc.stdout is not None
        for raw in proc.stdout:
            if not raw.startswith(EVENT_PREFIX):
                continue
            try:
                msg = json.loads(raw[len(EVENT_PREFIX):])
            except json.JSONDecodeError:
                continue
            kind = msg.pop("kind", None)
            if kind == "event":
                self.publish("stage", msg)
            elif kind == "error":
                error = msg.get("error")
        rc = proc.wait()
        if req.cancel_requested:
            req.state = "cancelled"
            self._mark_interrupted(req.job_id, "cancelled")
        elif rc == 0:
            req.state = "done"
        else:
            req.state = "failed"
            req.error = error or f"worker exited with code {rc}"
            self._mark_interrupted(req.job_id, req.error)

    def _mark_interrupted(self, job_id: str, reason: str) -> None:
        """A killed worker leaves its stage 'running' in job.json: record it as failed."""
        try:
            job = self.store.load(job_id)
        except Exception:  # noqa: BLE001
            return
        changed = False
        for name, st in job.stages.items():
            if st.status == StageStatus.running:
                st.status = StageStatus.failed
                st.error = st.error or reason
                st.finished_at = now()
                changed = True
                self.publish("stage", {"job_id": job_id, "stage": StageName(name).value, "status": "failed",
                                       "progress": st.progress, "message": "", "error": st.error})
        if changed:
            self.store.save(job)


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


def _kill_if_alive(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        _signal_group(proc, signal.SIGKILL)
