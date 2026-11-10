"""Child process that runs ``pipeline.run_job`` for the API's job queue.

    python -m karaoke.api.worker <jobs_dir> <job_id> [--from STAGE] [--only STAGE ...] [--force]

The server never runs stages in its own process: a cancel kills this process's whole session
(the isolated separate/transcribe grandchildren and ffmpeg included), and the server stays light.
Stage events are written to stdout as JSON lines prefixed with ``EVENT_PREFIX``; any other output
(libraries that print) is ignored by the reader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVENT_PREFIX = "@@karaoke "


def _emit(payload: dict) -> None:
    sys.stdout.write(EVENT_PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    from karaoke.models import StageName
    from karaoke.pipeline import StageEvent, run_job
    from karaoke.store import JobStore

    ap = argparse.ArgumentParser(prog="karaoke.api.worker")
    ap.add_argument("jobs_dir")
    ap.add_argument("job_id")
    ap.add_argument("--from", dest="from_stage", type=StageName, default=None)
    ap.add_argument("--only", nargs="*", type=StageName, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    def on_event(e: StageEvent) -> None:
        _emit({"kind": "event", "job_id": e.job_id, "stage": e.stage.value, "status": e.status.value,
               "progress": e.progress, "message": e.message, "error": e.error})

    try:
        run_job(a.job_id, from_stage=a.from_stage, only=a.only or None, force=a.force,
                store=JobStore(Path(a.jobs_dir)), on_event=on_event)
    except Exception as exc:  # noqa: BLE001 — reported to the server, details are in logs/<stage>.log
        _emit({"kind": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 1
    _emit({"kind": "ok"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
