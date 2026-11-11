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


@app.command("lyrics-brief")
def lyrics_brief(
    job_id: str,
    compact: Annotated[bool, typer.Option(help="Readable text: one row per line, its suspects under it.")] = False,
) -> None:
    """Print the synced lyrics with the evidence around them (JSON): what Whisper heard under each
    line, flags, unmatched audio, the other lyric sources, and a list of suspects."""
    from karaoke import lyrics_review

    b = lyrics_review.brief(_store().paths(job_id))
    print(lyrics_review.compact(b) if compact else json.dumps(b, indent=2, ensure_ascii=False))


@app.command("lyrics-apply")
def lyrics_apply(
    job_id: str,
    edits: Annotated[Path, typer.Argument(help="A JSON file: a list of edits (see the karaoke-lyrics-review skill).")],
    dry_run: Annotated[bool, typer.Option(help="Check the edits and print what they would do; save nothing.")] = False,
) -> None:
    """Apply a list of edits to lyrics.final.json. All or nothing; hand-edited words are never touched."""
    from karaoke.lyrics_review import EditError, apply_edits

    try:
        log = apply_edits(_store().paths(job_id), json.loads(edits.read_text()), dry_run=dry_run)
    except EditError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    for line in log:
        console.print(line)
    console.print(f"{len(log)} edit(s) {'checked, nothing saved' if dry_run else 'applied'}")


@app.command("bg-brief")
def bg_brief(job_id: str) -> None:
    """Print what an art director needs to know about the job (JSON): sections with their lines,
    materials, the cover's colours, the current scenes."""
    from karaoke.bg_tools import brief
    from karaoke.config import get_settings

    print(json.dumps(brief(_store().paths(job_id), get_settings().ffmpeg_bin), indent=2, ensure_ascii=False))


@app.command("bg-sheet")
def bg_sheet(
    job_id: str,
    out: Annotated[Path | None, typer.Option(help="Default: output/bg_sheet.jpg in the job.")] = None,
) -> None:
    """Draw every scene with the lyrics over it into one picture, and print each scene's legibility."""
    from karaoke.bg_tools import contact_sheet
    from karaoke.config import get_settings

    p = _store().paths(job_id)
    out = out or p.output_dir / "bg_sheet.jpg"
    for row in contact_sheet(p, out, get_settings().ffmpeg_bin):
        console.print(f"{row['scene']:<6} at {row['shown_at']:>6.1f}s  {row['section']:<12} behind the lyrics: "
                      f"mean {row['mean']:.2f}  brightest {row['p95']:.2f}  detail {row['detail']:.2f}")
    console.print(f"wrote {out}")


@app.command("bg-check")
def bg_check(
    job_id: str,
    before: Annotated[Path | None, typer.Option(help="A copy of background.json from before the changes: "
                                                     "its locked scenes must be unchanged.")] = None,
    fast: Annotated[bool, typer.Option(help="Skip drawing the scenes (no legibility check).")] = False,
) -> None:
    """Check background.json: the contract, locked scenes, scene changes, legibility. Exit 1 on errors."""
    from karaoke.bg_tools import check
    from karaoke.config import get_settings
    from karaoke.models import Background
    from karaoke.store import read_model

    old = read_model(before, Background) if before else None
    found = check(_store().paths(job_id), None if fast else get_settings().ffmpeg_bin, old)
    for f in found:
        console.print(f"[{'red' if f.severity == 'error' else 'yellow'}]{f}[/]")
    errors = sum(f.severity == "error" for f in found)
    console.print(f"{errors} error(s), {len(found) - errors} warning(s)")
    raise typer.Exit(1 if errors else 0)


def _picture_keys() -> dict[str, str | None]:
    from karaoke.config import get_settings

    s = get_settings()
    return {"unsplash": s.unsplash_access_key, "pexels": s.pexels_api_key}


@app.command("bg-search")
def bg_search(
    job_id: str,
    query: str,
    source: Annotated[list[str] | None, typer.Option(help="wikimedia, unsplash or pexels; repeatable. "
                                                          "Default: every source that is switched on.")] = None,
    n: Annotated[int, typer.Option(help="Candidates per source.")] = 8,
    name: Annotated[str | None, typer.Option(help="Name of this search (default: from the query).")] = None,
    any_shape: Annotated[bool, typer.Option(help="Also portrait and square pictures.")] = False,
) -> None:
    """Search free pictures for a background. Prints the candidates (JSON), each with a `pick`
    for bg-image, and draws their thumbnails into one numbered picture to look at."""
    from karaoke import bg_search as bs

    p = _store().paths(job_id)
    try:
        cands, notes = bs.search(query, _picture_keys(), source, n, landscape=not any_shape)
    except bs.SearchError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    name = bs.slug(name or query)
    sheet = p.bg_search_dir / f"{name}.jpg"
    looks = []
    if cands:  # a search that found nothing leaves the last one of that name alone
        bs.save_search(p, name, query, cands)
        looks = bs.candidate_sheet(cands, sheet)
    print(json.dumps({
        "query": query, "notes": notes, "sheet": str(sheet) if cands else None,
        "candidates": [{"pick": f"{name}:{k + 1}", "source": c.provider, "title": c.title,
                        "size": [c.width, c.height], "license": c.license, "credit": c.credit, "page": c.page,
                        **look} for k, (c, look) in enumerate(zip(cands, looks, strict=True))],
    }, indent=2, ensure_ascii=False))


@app.command("bg-image")
def bg_image(
    job_id: str,
    url: Annotated[str | None, typer.Argument(help="An https:// picture; or use --pick.")] = None,
    pick: Annotated[str | None, typer.Option(help="A candidate from bg-search, e.g. 'night-city:3': "
                                                  "its credit and licence are recorded for you.")] = None,
    credit: Annotated[str | None, typer.Option(help="With a URL: author and title, as the source states them.")] = None,
    license_: Annotated[str | None, typer.Option("--license", help="With a URL: e.g. 'CC BY-SA 4.0'.")] = None,
    page: Annotated[str | None, typer.Option(help="With a URL: the page describing the picture.")] = None,
) -> None:
    """Download a picture into the job's assets/bg and record its credit. Prints the path to use
    in background.json (with --pick: the scene's whole `source` object)."""
    from karaoke import bg_search as bs
    from karaoke.bg_tools import add_image

    p = _store().paths(job_id)
    try:
        if pick:
            _, source = bs.add_candidate(p, bs.load_pick(p, pick), _picture_keys())
            print(json.dumps(source, indent=2, ensure_ascii=False))
            return
        if not (url and credit and license_):
            raise ValueError("give --pick <search>:<number>, or a URL with --credit and --license")
        dest = add_image(p, url, credit, license_, page)
    except Exception as exc:  # noqa: BLE001 — shown to the caller
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    print(p.rel(dest))


@app.command("bg-image-rm")
def bg_image_rm(
    job_id: str,
    name: Annotated[str, typer.Argument(help="File name in assets/bg, e.g. img-1a2b3c.jpg.")],
) -> None:
    """Delete a downloaded picture you decided not to use, and its credit entry."""
    from karaoke.bg_tools import remove_image

    try:
        remove_image(_store().paths(job_id), name)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    console.print(f"removed {name}")


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
