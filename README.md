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

## Web UI

```bash
cd web && npm install && npm run build && cd ..   # once, and after UI changes
uv run karaoke serve                              # http://127.0.0.1:8765
```

Paste a YouTube link to make a song; follow the steps live; fix the lyrics while they play over the
instrumental (click a word to jump there, nudge it by 50 ms, retype a line, or tap Space along with the
song); pick a background image; render a preview or the final video. Edits are saved to
`lyrics.final.json` as `manual` words, which a re-sync never overwrites. Changes made to the job folder
by the CLI or a Claude session show up in the open page without a reload.

For UI work, run `npm run dev` in `web/` next to `karaoke serve` (Vite proxies `/api`).
`npm test` runs the UI tests, `npm run typecheck` the type check, and `npm run gen:types`
regenerates `web/src/types/generated/` from `docs/schema/` after `uv run karaoke schema`.

## Development

```bash
uv run pytest                   # offline tests on a synthetic fixture song
uv run ruff check karaoke tests
```
