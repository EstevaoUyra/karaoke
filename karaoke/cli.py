"""`karaoke` command-line interface. Everything the UI and the AI skills do goes through here or the store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from karaoke.models import SCHEMA_MODELS, StageName, StageStatus
from karaoke.pipeline import StageEvent, run_job
from karaoke.store import JobStore
from karaoke.validate import validate_job

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Karaoke video generator.")
console = Console()

_STATUS_STYLE = {
    StageStatus.pending: "dim",
    StageStatus.running: "yellow",
    StageStatus.done: "green",
    StageStatus.skipped: "cyan",
    StageStatus.failed: "red",
}


def _store() -> JobStore:
    return JobStore()


def _print_event(e: StageEvent) -> None:
    if e.status == StageStatus.running and e.message not in ("", "started"):
        console.print(f"  [dim]{e.stage.value:<13}{e.progress:4.0%} {e.message}[/]")
    elif e.status != StageStatus.running or e.message == "started":
        style = _STATUS_STYLE[e.status]
        console.print(f"[{style}]{e.stage.value:<13}{e.status.value}[/] {e.error or e.message}")


@app.command()
def new(
    url: str,
    run: Annotated[bool, typer.Option("--run", help="Run the whole pipeline right away.")] = False,
) -> None:
    """Create a job from a YouTube URL (playlist params are stripped)."""
    job = _store().create(url=url)
    console.print(f"job [bold]{job.id}[/] → {_store().paths(job.id).root}")
    if run:
        run_job(job.id, on_event=_print_event)


@app.command("run")
def run_cmd(
    job_id: str,
    from_stage: Annotated[StageName | None, typer.Option("--from", help="Force this stage and continue.")] = None,
    only: Annotated[list[StageName] | None, typer.Option("--only", help="Run exactly these stages.")] = None,
    force: Annotated[bool, typer.Option(help="Ignore the cache for every stage.")] = False,
) -> None:
    """Run the pipeline (cached stages are skipped)."""
    try:
        run_job(job_id, from_stage=from_stage, only=only, force=force, on_event=_print_event)
    except Exception as exc:  # noqa: BLE001 — the event printer already showed the stage error
        raise typer.Exit(1) from exc


@app.command()
def stage(job_id: str, name: StageName) -> None:
    """Run a single stage (always re-runs it)."""
    run_cmd(job_id, only=[name])


@app.command()
def status(job_id: str) -> None:
    """Show stage status for a job."""
    store = _store()
    job = store.load(job_id)
    meta = store.meta(job_id)
    title = f"{meta.clean_artist or meta.uploader} — {meta.clean_title or meta.title}" if meta else job.url
    t = Table(title=f"{job.id}: {title}")
    for col in ("stage", "status", "time", "message"):
        t.add_column(col)
    for name, st in job.stages.items():
        t.add_row(
            name.value,
            f"[{_STATUS_STYLE[st.status]}]{st.status.value}[/]",
            f"{st.duration_s:.1f}s" if st.duration_s is not None else "",
            st.error or st.message,
        )
    console.print(t)


@app.command("list")
def list_cmd() -> None:
    """List jobs."""
    store = _store()
    for job in store.list():
        meta = store.meta(job.id)
        done = sum(st.status in (StageStatus.done, StageStatus.skipped) for st in job.stages.values())
        console.print(f"{job.id:<14} {done}/{len(job.stages)}  {meta.title if meta else job.url}")


@app.command()
def validate(job_id: str) -> None:
    """Check every job file against the contracts. Exit code 1 on errors."""
    issues = validate_job(_store().paths(job_id))
    for i in issues:
        console.print(f"[{'red' if i.severity == 'error' else 'yellow'}]{i}[/]")
    errors = sum(i.severity == "error" for i in issues)
    console.print(f"{errors} error(s), {len(issues) - errors} warning(s)")
    raise typer.Exit(1 if errors else 0)


@app.command()
def background(
    job_id: str,
    preset: Annotated[str, typer.Argument(help="gradient, palette, cover, stills, video or aurora.")],
    replace_locked: Annotated[bool, typer.Option(help="Also replace scenes the user locked.")] = False,
) -> None:
    """Replace background.json with an instant preset (uses structure.json for the scene changes)."""
    from karaoke.bg_presets import PresetError, apply_preset
    from karaoke.config import get_settings

    try:
        bg = apply_preset(preset, _store().paths(job_id), get_settings().ffmpeg_bin, replace_locked=replace_locked)
    except PresetError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    console.print(f"background.json: {len(bg.scenes)} scene(s), style {bg.style}")


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Default: KARAOKE_HOST or 127.0.0.1.")] = None,
    port: Annotated[int | None, typer.Option(help="Default: KARAOKE_PORT or 8765.")] = None,
) -> None:
    """Start the web UI and API (build the UI first: `npm run build` in web/)."""
    from karaoke.api.app import serve as run_server

    run_server(host, port)


@app.command()
def schema(out: Path = Path("docs/schema")) -> None:
    """Export JSON Schemas for every contract file (consumed by the web UI's types)."""
    out.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMA_MODELS.items():
        (out / f"{name}.schema.json").write_text(json.dumps(model.model_json_schema(), indent=2) + "\n")
    console.print(f"wrote {len(SCHEMA_MODELS)} schemas to {out}")


if __name__ == "__main__":
    app()
