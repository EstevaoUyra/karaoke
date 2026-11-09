# karaoke

Paste a YouTube link and get a karaoke video: separated instrumental, word-synced and verified lyrics, and an optional AI-directed background.

> Status: Phase 1 (command-line pipeline) is done. The web UI (Phase 2) and the AI skills (Phase 3) are next. See [PLAN.md](PLAN.md).

## How it works

```
YouTube → download → vocal separation ─┬→ whisper (on the vocal stem) ─┐
                    lyrics (LRCLIB, …) ─┴──────────────────────────────┴→ align → background → render (ASS + ffmpeg)
```

Fetched lyrics decide the words and Whisper decides the timing. Two Claude Code skills can then
review the lyrics and art-direct the background.

## Setup (macOS)

```bash
brew install libass x264 nasm pkgconf
scripts/build_ffmpeg.sh         # ~15 min: FFmpeg with libass in ~/.local/opt/ffmpeg-karaoke
uv sync                         # Python 3.12 venv with all dependencies
cp .env.example .env            # optional API keys
```

Homebrew's `ffmpeg` can't burn in subtitles (no libass). `ffmpeg-full` can, but on Intel Macs it
compiles llvm and rust from source, which takes hours. The script builds only what's needed.
On Apple Silicon, `brew install ffmpeg-full` and skipping the script is simpler.

## Use

```bash
uv run karaoke new "https://www.youtube.com/watch?v=..." --run
uv run karaoke status <job_id>
```

Each job lives in `data/jobs/<youtube id>/`. Its layout is documented in [PLAN.md §3](PLAN.md).

## Development

```bash
uv run pytest                   # offline tests on a synthetic fixture song
uv run ruff check karaoke tests
```
