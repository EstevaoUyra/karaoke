# karaoke

Paste a YouTube link and get a karaoke video: separated instrumental, word-synced and verified lyrics, and an optional AI-directed background.

> Status: Phase 0 done (contracts, fixtures, benchmarks). The pipeline stages are being built. See [PLAN.md](PLAN.md).

## How it works

```
YouTube → download → vocal separation ─┬→ whisper (on the vocal stem) ─┐
                    lyrics (LRCLIB, …) ─┴──────────────────────────────┴→ align → background → render (ASS + ffmpeg)
```

Fetched lyrics decide the words and Whisper decides the timing. Two Claude Code skills can then
review the lyrics and art-direct the background.

## Setup (macOS)

```bash
brew install ffmpeg-full        # needs libass for subtitles (keg-only, doesn't replace ffmpeg)
uv sync                         # Python 3.12 venv with all dependencies
cp .env.example .env            # optional API keys
```

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
