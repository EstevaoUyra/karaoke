# karaoke — notes for Claude

The plan and architecture are in `PLAN.md`; read it first. Benchmark findings are in `docs/benchmarks.md`.

## Commands
- `uv run pytest`: fast tests (offline, fixtures only). `uv run pytest -m slow` runs real models and the network.
- `uv run ruff check karaoke tests spikes`
- `uv run karaoke --help`: commands `new`, `run [--from STAGE | --only STAGE]`, `stage`, `status`, `list`, `validate`, `schema`, `serve`.
- Web UI: `cd web && npm install && npm run build`, then `uv run karaoke serve` (http://127.0.0.1:8765). `npm test` / `npm run typecheck` in `web/`.
- `uv run python tests/fixtures/make_demo.py data/jobs` creates the synthetic `demo` job (24 s, known ground truth).

## Machine constraints (Intel Mac, CPU only)
- Python **3.12** only. Version caps, all set in `pyproject.toml`:
  - `torch==2.2.2` and `numpy<2`.
  - onnxruntime ≤1.23 and llvmlite/numba ≤0.45/0.62, the last versions with Intel-Mac wheels.
  - `av<17`, because faster-whisper 1.2.1 breaks on PyAV 17+.
- **Never run torch (separation, stable-ts) and faster-whisper/ctranslate2 in the same process.**
  - Their two libiomp5 copies deadlock.
  - `separate` and `transcribe` are `isolated=True` stages in `karaoke/pipeline.py` and run in spawned child processes.
- ffmpeg with libass is a minimal build from `scripts/build_ffmpeg.sh` in `~/.local/opt/ffmpeg-karaoke`. Homebrew `ffmpeg-full` is impractical on Intel: it has no bottles and compiles llvm and rust.
  - Use `get_settings().ffmpeg_bin`, never a bare `ffmpeg`, for anything that renders subtitles.
- Whisper: energy-based voice regions plus the batched pipeline. Never use the Silero VAD (it drops sections). A lyrics prompt hurts accuracy and is opt-in via `KARAOKE_WHISPER_PROMPT`.
- yt-dlp needs a JS runtime for YouTube: pass `js_runtimes={"node": {}}` (CLI: `--js-runtimes node`).

## Contracts (frozen; changes go through the main session)
- `karaoke/models.py` is the pydantic model for every job file.
- `karaoke/store.py` (`JobPaths`) holds every path in a job folder. Never hardcode a path inside a job.
- `karaoke/pipeline.py` lists each stage's inputs, outputs and settings, which decide caching.
- Stage modules expose `run(ctx: StageContext)`. Use `ctx.paths`, `ctx.progress()` and `ctx.log`, and do heavy imports inside `run`.
- Words whose `origin` is `manual` or `ai` in `lyrics.final.json` are never overwritten by `align`. Scenes with `locked: true` are never changed by the AI.
- Job data lives in `data/` (gitignored). Fetched lyrics, audio and images must never be committed: the repo is public.

## Languages
Songs are PT-BR and EN. Never trust YouTube's `language` field (it reports `fr` for a Brazilian song).
