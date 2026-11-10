"""Stage registry and runner.

Each stage lives in ``karaoke/stages/<name>.py`` and exposes ``run(ctx: StageContext) -> None``.
The registry below (inputs/outputs/settings per stage) is part of the frozen contract: it decides
when a stage is cached. Stage modules are imported lazily so the CLI stays fast (no torch import).
"""

from __future__ import annotations

import importlib
import logging
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from karaoke.config import Settings, get_settings
from karaoke.models import Job, StageName, StageState, StageStatus, now
from karaoke.store import JobPaths, JobStore, inputs_hash


@dataclass
class StageEvent:
    job_id: str
    stage: StageName
    status: StageStatus
    progress: float = 0.0
    message: str = ""
    error: str | None = None


EventCallback = Callable[[StageEvent], None]


@dataclass
class StageContext:
    job: Job
    paths: JobPaths
    store: JobStore
    settings: Settings
    stage: StageName
    log: logging.Logger
    force: bool = False
    _emit: EventCallback | None = None
    _last_save: float = 0.0

    def progress(self, fraction: float, message: str = "") -> None:
        """Report progress (0..1). Cheap to call often; persisted at most every 0.5 s."""
        st = self.job.stages[self.stage]
        st.progress = max(0.0, min(1.0, fraction))
        st.message = message
        if message:
            self.log.info("%3.0f%% %s", st.progress * 100, message)
        t = time.monotonic()
        if t - self._last_save > 0.5:
            self.store.save(self.job)
            self._last_save = t
        if self._emit:
            self._emit(StageEvent(self.job.id, self.stage, StageStatus.running, st.progress, message))


@dataclass(frozen=True)
class StageSpec:
    name: StageName
    # files whose content decides whether the stage must re-run
    inputs: Callable[[JobPaths], list[Path]]
    # files that must exist after a successful run
    outputs: Callable[[JobPaths, Job], list[Path]]
    # settings that decide whether the stage must re-run
    settings: Callable[[Job, Settings], object] = field(default=lambda job, s: None)
    # Run in a spawned child process. Required for torch (separate) and ctranslate2 (transcribe):
    # their two libiomp5 copies deadlock if both run in one process. Also frees model RAM.
    isolated: bool = False

    @property
    def module(self) -> str:
        return f"karaoke.stages.{self.name.value}"

    def run(self, ctx: StageContext) -> None:
        importlib.import_module(self.module).run(ctx)


def _render_output(p: JobPaths, job: Job) -> list[Path]:
    return [p.ass, p.preview_video if job.settings.render.preset == "preview" else p.final_video]


STAGES: list[StageSpec] = [
    StageSpec(
        StageName.ingest,
        inputs=lambda p: [],
        outputs=lambda p, j: [p.meta, p.audio],
        settings=lambda j, s: {"url": j.url},
    ),
    StageSpec(
        StageName.separate,
        inputs=lambda p: [p.audio],
        outputs=lambda p, j: [p.vocals, p.instrumental],
        settings=lambda j, s: {
            "model": j.settings.separator_model
            or (s.separator_model_hq if j.settings.quality == "hq" else s.separator_model)
        },
        isolated=True,
    ),
    StageSpec(
        StageName.lyrics_fetch,
        # lrclib/genius files are OUTPUTS of this stage; only user/youtube texts are inputs.
        inputs=lambda p: [p.meta, p.lyrics_source("user.txt"), p.lyrics_source("youtube.vtt")],
        outputs=lambda p, j: [p.lyrics_choice, p.lyrics_chosen_text],
        settings=lambda j, s: {"force_source": j.settings.lyrics_source, "language": j.settings.language},
    ),
    StageSpec(
        StageName.transcribe,
        # The lyrics prompt is a soft input (not hashed): editing lyrics must not re-run whisper.
        inputs=lambda p: [p.vocals],
        outputs=lambda p, j: [p.transcript],
        # "method" is bumped whenever the clip/retry logic changes, so old transcripts are redone.
        settings=lambda j, s: {"model": s.whisper_model, "language": j.settings.language,
                               "prompt": os.environ.get("KARAOKE_WHISPER_PROMPT", "none"), "method": 3},
        isolated=True,
    ),
    StageSpec(
        StageName.align,
        inputs=lambda p: [p.transcript, p.lyrics_choice, p.lyrics_chosen_text, p.lyrics_source("lrclib.lrc"),
                          p.meta, p.vocals],
        outputs=lambda p, j: [p.lyrics_final],
        settings=lambda j, s: {"method": 3},  # bumped when the algorithm changes
    ),
    StageSpec(
        StageName.melody,
        # After align (not straight after transcribe): the sung words are a prior for voicing and
        # split held notes at syllables. The F0 contour is cached by vocals.wav hash and tracker
        # (``melody_f0``), so a lyrics edit only re-segments. Instrumental and mix chroma help the key.
        inputs=lambda p: [p.vocals, p.instrumental, p.audio, p.lyrics_final],
        outputs=lambda p, j: [p.melody],
        settings=lambda j, s: {"method": 2, "tracker": s.melody_tracker},  # bumped when the algorithm changes
        isolated=True,  # torchcrepe imports torch: never in a process with ctranslate2
    ),
    StageSpec(
        StageName.backgrounds,
        # Only creates the default background.json if none exists (never overwrites user/AI work).
        inputs=lambda p: [p.meta],
        outputs=lambda p, j: [p.background],
    ),
    StageSpec(
        StageName.render,
        inputs=lambda p: [p.instrumental, p.vocals, p.lyrics_final, p.background, p.bg_assets_dir, p.meta,
                          p.melody],
        outputs=_render_output,
        settings=lambda j, s: j.settings.render.model_dump(),
    ),
]

STAGE_BY_NAME: dict[StageName, StageSpec] = {s.name: s for s in STAGES}
ORDER: list[StageName] = [s.name for s in STAGES]


def _stage_logger(paths: JobPaths, stage: StageName) -> tuple[logging.Logger, logging.Handler]:
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"karaoke.job.{paths.id}.{stage.value}")
    log.setLevel(logging.INFO)
    h = logging.FileHandler(paths.log(stage.value), encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)
    return log, h


def stage_hash(spec: StageSpec, paths: JobPaths, job: Job, settings: Settings) -> str:
    return inputs_hash(spec.inputs(paths), spec.settings(job, settings))


def is_cached(spec: StageSpec, paths: JobPaths, job: Job, settings: Settings) -> bool:
    st = job.stages[spec.name]
    return (
        st.status in (StageStatus.done, StageStatus.skipped)
        and st.input_hash == stage_hash(spec, paths, job, settings)
        and all(o.exists() for o in spec.outputs(paths, job))
    )


def select_stages(
    from_stage: StageName | None = None, only: list[StageName] | None = None
) -> list[StageName]:
    if only:
        return [s for s in ORDER if s in only]
    if from_stage:
        return ORDER[ORDER.index(from_stage):]
    return list(ORDER)


class IsolatedStageError(RuntimeError):
    """A stage that ran in a child process failed (details are in logs/<stage>.log)."""


def _child_main(stage: str, job_id: str, jobs_dir: str, settings: Settings, forced: bool, q) -> None:
    store = JobStore(Path(jobs_dir))
    paths = store.paths(job_id)
    name = StageName(stage)
    job = store.load(job_id)
    log, handler = _stage_logger(paths, name)
    ctx = StageContext(job, paths, store, settings, name, log, force=forced,
                       _emit=lambda e: q.put(("event", e)))
    try:
        STAGE_BY_NAME[name].run(ctx)
        store.save(ctx.job)
        q.put(("ok", None))
    except BaseException as exc:  # noqa: BLE001 — reported to the parent
        log.error("failed in child process\n%s", traceback.format_exc())
        q.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        log.removeHandler(handler)
        handler.close()


def _run_isolated(spec: StageSpec, job_id: str, store: JobStore, settings: Settings, forced: bool,
                  emit: EventCallback) -> None:
    import multiprocessing as mp
    import queue as queue_mod

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_child_main, name=f"karaoke-{spec.name.value}",
                       args=(spec.name.value, job_id, str(store.jobs_dir), settings, forced, q))
    proc.start()
    result: tuple[str, str | None] | None = None
    try:
        while result is None:
            try:
                kind, payload = q.get(timeout=0.5)
            except queue_mod.Empty:
                if not proc.is_alive():
                    break
                continue
            if kind == "event":
                emit(payload)
            else:
                result = (kind, payload)
        proc.join()
    except BaseException:
        proc.terminate()  # Ctrl-C / cancel: don't leave the model running
        proc.join()
        raise
    if result is None:
        raise IsolatedStageError(f"{spec.name} child process died (exit code {proc.exitcode})")
    if result[0] == "error":
        raise IsolatedStageError(result[1])


def run_job(
    job_id: str,
    *,
    from_stage: StageName | None = None,
    only: list[StageName] | None = None,
    force: bool = False,
    store: JobStore | None = None,
    settings: Settings | None = None,
    on_event: EventCallback | None = None,
) -> Job:
    """Run stages in order. Cached stages are skipped unless forced.

    ``from_stage`` forces that stage and continues with the following ones (each still cached
    if its inputs did not change). ``only`` runs exactly the listed stages. Stops at the first
    failure (the error is stored in job.json and re-raised).
    """
    store = store or JobStore()
    settings = settings or get_settings()
    paths = store.paths(job_id)
    job = store.load(job_id)
    emit = on_event or (lambda e: None)

    for name in select_stages(from_stage, only):
        spec = STAGE_BY_NAME[name]
        forced = force or name == from_stage or bool(only)
        if not forced and is_cached(spec, paths, job, settings):
            job.stages[name].status = StageStatus.skipped
            job.stages[name].message = "cached"
            store.save(job)
            emit(StageEvent(job_id, name, StageStatus.skipped, 1.0, "cached"))
            continue

        h = stage_hash(spec, paths, job, settings)
        job.stages[name] = StageState(status=StageStatus.running, started_at=now())
        store.save(job)
        emit(StageEvent(job_id, name, StageStatus.running, 0.0, "started"))
        log, handler = _stage_logger(paths, name)
        ctx = StageContext(job, paths, store, settings, name, log, force=forced, _emit=on_event)
        t0 = time.monotonic()
        try:
            if spec.isolated:
                _run_isolated(spec, job_id, store, settings, forced, emit)
                job = store.load(job_id)  # pick up what the child wrote (progress, message)
                ctx.job = job
            else:
                spec.run(ctx)
            missing = [paths.rel(o) for o in spec.outputs(paths, job) if not o.exists()]
            if missing:
                raise RuntimeError(f"stage {name} finished but outputs are missing: {missing}")
        except BaseException as exc:
            st = job.stages[name]
            st.status = StageStatus.failed
            st.error = f"{type(exc).__name__}: {exc}"
            st.finished_at = now()
            st.duration_s = time.monotonic() - t0
            log.error("failed\n%s", traceback.format_exc())
            store.save(job)
            emit(StageEvent(job_id, name, StageStatus.failed, st.progress, "", st.error))
            raise
        finally:
            log.removeHandler(handler)
            handler.close()

        st = job.stages[name]
        st.status = StageStatus.done
        st.input_hash = h
        st.progress = 1.0
        st.finished_at = now()
        st.duration_s = round(time.monotonic() - t0, 2)
        store.save(job)
        emit(StageEvent(job_id, name, StageStatus.done, 1.0, st.message))
    return job
