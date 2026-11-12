# Karaoke Generator — Plan

Paste a YouTube link in a browser UI and get back a karaoke video: the instrumental track, word-synced lyrics that are 100% correct, and a background that is either a simple default or art-directed by AI.

---

## 1. Environment findings (checked 2026-11-09)

| Item | Finding | Consequence |
|---|---|---|
| Machine | Intel i5-1038NG7 (4 cores), 16 GB, macOS 15.7, **no CUDA/MPS** | Everything runs on the CPU. One job at a time, and stages are cached so we never redo work. |
| Whisper | `faster-whisper` + `mobiuslabsgmbh/faster-whisper-large-v3-turbo` already in the HF cache | We reuse it (CTranslate2, int8 on CPU). |
| PyTorch | **The last Intel-Mac wheel is torch 2.2.2, which supports Python ≤ 3.12** | The project is pinned to **Python 3.12** (via `uv`), and we need `numpy<2` because torch 2.2 was built against numpy 1.x. |
| Separation libs | `audio-separator` 0.24.1 and `onnxruntime` 1.23.2 work on py3.12/x86_64 | Several packages dropped Intel-Mac wheels, so `pyproject.toml` pins onnxruntime ≤1.23 (`required-environments`), llvmlite ≤0.45.1 / numba ≤0.62.1, and `av<17` (faster-whisper 1.2.1 breaks on PyAV 17+). |
| ffmpeg | Homebrew ffmpeg 8.1 **has no libass** (no `ass`/`subtitles` filter) | We use `brew install ffmpeg-full` (keg-only, so it won't replace the current ffmpeg); `config.py` finds it automatically. On Intel some of its dependencies have no bottles and compile from source (slow, one time only). |
| yt-dlp | The global one is 2026.03 (old). YouTube now needs a JS runtime. | We pin a recent `yt-dlp[default]` inside the venv and use `node` (installed) as the JS runtime. |
| `claude` CLI | v2.1.319, logged in via claude.ai subscription. **`claude -p` verified working without API credits.** It uses the subscription's usage limits. | The UI can run the AI skills headless. Interactive sessions remain a first-class way to run them (see §6). |
| Test songs | `Zdh2hot8rjU`: *Sweet Child O' Mine*, Captain Fantastic soundtrack cover, 3:40, EN, **manual EN subs**. `f8x_t8wPFcM`: *Sacal – Vão Querer Me Internar* (Tropical Paradise Riddim), 2:33, PT-BR dancehall, **no manual subs, and YouTube reports its language as `fr`**. | These are good stress tests: a cover whose LRC timings come from the original recording, and fast PT-BR vocals with metadata we can't trust. |

---

## 2. Pipeline

```
YouTube URL
   │
   ▼
[1 ingest] yt-dlp → source.mp4, audio.wav (44.1k stereo), meta.json (title, artist, duration),
           youtube subs (manual > auto) → lyrics/sources/youtube.vtt
   │
   ├──────────────────────────┐
   ▼                          ▼
[2 separate]                [3 lyrics_fetch]  (runs in parallel with 2, network-bound)
 audio-separator             LRCLIB (synced LRC + plain, free, no key)
 → stems/vocals.wav          Genius (optional, token)
 → stems/instrumental.wav    YouTube manual subs (from ingest)
   │                         user-pasted text (UI), which always wins
   ▼                          │
[4 transcribe]                │
 faster-whisper turbo on      │
 vocals.wav (not the mix),    │
 word_timestamps, VAD         │
 → transcript.whisper.json    │
   │                          │
   └────────────┬─────────────┘
                ▼
[5 align]  lyrics text = source of truth for the WORDS
           whisper     = source of truth for the TIMING
           LRC line timestamps = anchors/constraints
           → lyrics.final.json (+ per-word confidence, flags)
                │
                ▼
     (optional) AI skill: karaoke-lyrics-review  ── edits lyrics.final.json
                │
                ▼
[5b melody] torchcrepe-tiny f0 on vocals.wav (pyin fallback) → notes, with the sung words as a
            voicing prior; key (Krumhansl–Schmuckler: notes + instrumental + mix chroma) → melody.json
                │
                ▼
[5c structure] librosa on audio.wav + the synced lines → structure.json: sections (intro / verse /
            chorus / instrumental / outro, each with loudness and sung share), a loudness curve
            and beats. Boundaries: long pauses between lines, then peaks of audio novelty;
            never inside a sung line. Labels are a guess, for pacing scenes only.
                │
                ▼
[6 backgrounds] default: background.json with 1 scene (solid colour / gradient / single image)
     (optional) instant presets (`karaoke background <job> <preset>`, Background tab), one scene
                per section: palette (gradients in the cover's colours), cover (blurred
                thumbnail), stills (frames of the source video), video (the source video, blurred),
                aurora (a `generated` scene: moving colour that follows loudness, beats and
                sections, drawn with numpy in `render/generated.py`; also `pulse` and `drift`)
     (optional) the karaoke-backgrounds skill (`.claude/skills/karaoke-backgrounds/`): reads
                `karaoke bg-brief`, writes art_direction.md and background.json, checks with
                `bg-check` (contract, locked scenes, legibility) and `bg-sheet` (contact sheet)
     (optional) AI skill: karaoke-backgrounds ── art-directed scene timeline + images
     (optional) UI: replace or lock any scene (upload / URL / web search)
                │
                ▼
[7 render]  ASS karaoke subtitles (\kf fill) + background compositor + instrumental
            (+ optional guide-vocals at −18 dB) → ffmpeg(libass, x264) → output/karaoke.mp4
```

### Design principles
1. **The job folder is the API.** Every stage reads and writes plain files in `data/jobs/<id>/`. The web UI, the CLI and the Claude skills all work on the same files. That is what lets an AI skill "just edit the JSON and re-render."
2. **Stages are idempotent and cached.** A stage is skipped if its outputs exist and its inputs haven't changed (a hash goes in `job.json`). Editing the lyrics only re-runs `render`.
3. **One CLI mirrors everything:** `karaoke new <url>`, `karaoke run <job> [--from align]`, `karaoke stage <job> render`, `karaoke validate <job>`. The skills' helper scripts call this CLI.
4. **Separation and transcription are swappable** behind a small interface (`Separator`, `Transcriber`) so models can change after benchmarking.
5. **Heavy ML stages run in child processes** (`isolated=True` in `karaoke/pipeline.py`). This was found in Phase 0: torch and ctranslate2 each bundle libiomp5 on Intel macOS, and once both have run in one process, torch deadlocks. Child processes also return each model's RAM, which matters on a 16 GB machine.

---

## 3. Job folder contract (frozen in Phase 0, so parallel work is safe)

```
data/jobs/<job_id>/                 # job_id = the YouTube video id (re-adding the same URL reuses the job)
  job.json                          # status per stage, input hashes, settings, errors
  meta.json                         # url, title, artist, track, duration, thumbnail
  source.mp4  audio.wav
  stems/vocals.wav  stems/instrumental.wav
  lyrics/sources/{lrclib.lrc, lrclib.txt, genius.txt, youtube.vtt, user.txt}
  lyrics/chosen.txt                 # canonical text used for alignment (+ which source)
  transcript.whisper.json           # segments + words {text,start,end,prob}
  lyrics.final.json                 # ← the important one
  melody.json                       # sung notes + key, for the optional pitch guide (render.show_pitch)
  cache/melody.f0.npz               # raw pyin contour, reused while vocals.wav is unchanged
  structure.json                    # sections, loudness curve, beats (for backgrounds)
  background.json
  assets/bg/*.jpg                   # downloaded/uploaded images + credits.json
  output/karaoke.ass  output/preview.mp4  output/karaoke.mp4
  logs/*.log
```

**`lyrics.final.json`**
```json
{
  "version": 1, "language": "pt", "text_source": "lrclib",
  "lines": [
    { "id": "l012", "section": "chorus", "start": 61.20, "end": 64.85,
      "words": [
        { "text": "Olha", "start": 61.20, "end": 61.55, "conf": 0.94, "origin": "matched" },
        { "text": "que",  "start": 61.55, "end": 61.70, "conf": 0.40, "origin": "interpolated" }
      ],
      "flags": ["low_confidence"] }
  ]
}
```
`origin` is one of `matched | fuzzy | interpolated | manual | ai`. Manual edits are never overwritten by a re-run of `align`.

**`background.json`**
```json
{
  "version": 1, "resolution": [1920, 1080], "style": "default",
  "scenes": [
    { "id": "s1", "start": 0, "end": 31.5,
      "source": { "type": "image", "path": "assets/bg/s1.jpg",
                  "origin": "wikimedia", "url": "…", "credit": "…" },
      "motion": "kenburns-in", "transition": { "type": "fade", "duration": 1.2 },
      "dim": 0.45, "locked": false, "reason": "intro — band live photo" }
  ]
}
```
`source.type` is `color | gradient | image | video`. A `locked: true` scene (set by the user in the UI) is never changed by the AI skill.

---

## 4. Key technical decisions

### Vocal separation
- The wrapper is **`audio-separator`**, which runs UVR's MDX-Net, Demucs and Roformer models behind one API, so switching models is a config change.
- **Phase 0 benchmark** on 2 songs (one PT, one EN), measuring time on this CPU and listening quality:
  - `UVR-MDX-NET-Inst_HQ_3` / `Kim_Vocal_2` (ONNX, fast on CPU)
  - `htdemucs_ft` (good quality, slower)
  - `MelBand Roformer` (best quality on paper, likely very slow on CPU)
- Expected default: MDX-Net for speed, with a "high quality" setting that switches to Roformer/Demucs.

### Transcription
- faster-whisper large-v3-turbo, `compute_type=int8`, `word_timestamps=True`, run **on the vocal stem**.
- **We don't use the Silero VAD filter** (Phase 0: it dropped half the cover). Voice regions come from the vocal stem's RMS energy and are passed as `clip_timestamps` (see `docs/benchmarks.md`).
- Songs are **PT-BR and EN**. The language is detected from the fetched lyrics text (`lingua-language-detector`, restricted to pt/en), then from Whisper's own detection on the vocal stem. **YouTube's `language` field is never trusted**: the Sacal video reports `fr`.
- Pass the fetched lyrics as `initial_prompt` (first ~200 tokens) to bias the vocabulary.
- Reuse the `find_alignment` empty-window guard from `exercito/debate-live/whisper_runtime.py`.

### Lyrics sources (priority)
`user-pasted` > `LRCLIB synced` > `LRCLIB plain` > `Genius` > `YouTube manual subs` > `whisper` (fallback only)
- **LRCLIB** is free, needs no key, and often has line-synced LRC. That is a big win, because its line timings become anchors.
  - **We use its timings only if the LRCLIB duration matches our audio within ±2 s.** Otherwise it is probably another recording (a cover, a live version, a radio edit), and we keep the text only. Example: the Captain Fantastic cover would otherwise inherit Guns N' Roses' timings.
- Title/artist cleanup first: strip "(Official Video)", "ft.", "[HD]" etc. from the YouTube title, and prefer yt-dlp's `track`/`artist` fields when present.
- **LRCLIB lookup tries several queries and picks the closest duration** (Phase 0 finding).
  - A plain "Sweet Child O' Mine" search returns only Guns N' Roses versions (303–356 s).
  - Searching with the title remainder ("Captain Fantastic") finds the cover's own synced LRC (220 s vs our 3:40).
  - So title qualifiers like "soundtrack" or "cover" are kept as extra search terms, not just stripped.
- Sacal is on no lyrics site (LRCLIB: 0 results, and the YouTube description has no lyrics). For independent PT-BR artists, the fallback (whisper text, then AI review with web search, then the paste box) is the *main* path, not an edge case.
- YouTube **auto** subs are just ASR (no better than whisper). Only **manual** subs count as a lyrics source.
- Expand `[Chorus x2]` / "(repeat)" markers, and strip section headers into `section` labels.
- Fetched lyrics live only in `data/` (gitignored). Lyrics are copyrighted, so they are for personal use and never committed.

### Alignment (the core of "100% correct")
1. Normalize tokens: lowercase, strip punctuation, fold accents for matching only, keep originals for display.
2. **Global sequence alignment** (Needleman-Wunsch with `rapidfuzz` similarity as the score) between the lyric words and the whisper words. Choruses repeat, so local/greedy matching would jump to the wrong repetition.
3. Matched lyric words take whisper's timing. Runs of unmatched words get **interpolated** between their matched neighbours, weighted by syllable/character length.
4. If LRC line timestamps exist, they **constrain** the alignment (band DP per line) and fix whole-line drift.
5. Whisper words with no lyric match are recorded as `unmatched_audio` (possible ad-libs or a missing repetition) and flagged for review.
6. Per-line confidence comes from the match ratio, whisper probability and timing sanity (no overlaps, plausible word durations).
7. **Optional refinement (later):** forced alignment. Phase 0 result: whole-song stable-ts `align()` placed every line, but with a 0.51 s median error, up to 8.7 s drift and 3× real-time cost. It is not used by default. A per-line-window version between anchors may come back later.

### Rendering
- **ASS subtitles with `\kf` tags** (smooth left-to-right fill), rendered by libass inside ffmpeg. This is the classic karaoke look, cheap to render, and font/colour/outline are all styleable.
- Layout: current line plus the next line (dimmed). Lines appear ~1.5 s before they start. A **countdown indicator** shows before lines that follow a gap of more than 4 s. A title card (title / artist) covers the intro.
- Background compositor: build one background video from `background.json` with `ffmpeg` (`zoompan` for Ken Burns, `xfade` for transitions), then overlay the ASS. Plan B if the filtergraph gets unwieldy: render background frames with Pillow/numpy and pipe them to ffmpeg.
- Two outputs: a `preview` (720p, ultrafast) and a `final` (1080p, CRF 22).
- **Optional pitch guide** (`render.show_pitch`, off by default): the sung melody from `melody.json`
  at the top of the screen. Notes scroll right to left past a fixed playhead and fill with the sung
  colour as they cross it. It is drawn in the same ASS (libass vector drawings, `\move` + `\clip`),
  so rendering is still one ffmpeg pass. Two styles (`render.pitch_style`):
  - `roll` (default, `karaoke/render/roll.py`): one row per semitone over the song's own range, the
    voice's real pitch (`cache/melody.f0.npz`) as a thin line over each bar, faint bars where the
    voice does not hold a pitch, and each word scrolling under its notes.
  - `staff` (`karaoke/render/staff.py`): a real treble staff (clef, key signature, ledger lines,
    ♯/♭/♮ for notes outside the key, "8vb"/"8va" octave shift), for people who read music.
- **The in-browser preview needs no rendering.** The UI plays `instrumental.wav` with an HTML/CSS lyric overlay driven by `lyrics.final.json`, so lyric edits can be checked instantly.

### Backend / UI
- **FastAPI** with a single-worker job queue (CPU-bound, one job at a time) and progress over **SSE**.
- **Vite + React + TypeScript** in `web/`, built and served by FastAPI, with no separate server in production.
- The **backend watches each job folder** (`watchfiles`) and pushes changes to the UI over SSE. Edits made by any Claude session therefore show up live in the open UI, with no manual refresh.
- AI integration is described in §6, "How skills are run."

---

## 5. UI screens

1. **Home**: URL box (plus "paste lyrics manually" and "upload audio" alternatives) and a job list with status.
2. **Job / Pipeline**: stage cards (ingest → render) with status, timings, logs and "re-run from here."
3. **Lyrics**:
   - A source picker with diff view (LRCLIB vs Genius vs YouTube vs Whisper).
   - The line/word editor, with low-confidence words highlighted.
   - A **play-along preview** and **tap-to-sync** (press space on each word/line while it plays).
   - Nudge buttons (±50 ms) and an **"Ask AI to review"** button.
4. **Background**:
   - A scene timeline over the waveform and section markers.
   - Per scene: replace (upload / URL / web image search), lock, set duration/motion/dim.
   - Instant presets (`karaoke/bg_presets.py`): gradient, palette, cover, stills, video, aurora. Each replaces every scene; locked scenes need a confirmation.
   - An **"AI art direction"** button.
5. **Render**: resolution, font/colours, guide-vocal level, then render, play and download.

---

## 6. Claude skills (`.claude/skills/`)

### How skills are run
1. **Interactive (primary):** open Claude Code in this repo and type `/karaoke-lyrics-review <job>` or `/karaoke-backgrounds <job>`. This is best for back-and-forth such as "make the chorus images warmer." Each AI panel in the UI shows the exact command with a copy button.
2. **One-click from the UI (secondary):** the "Ask AI" button runs `claude -p "/<skill> <job>" --allowedTools …` headless and streams its log into the UI. This was verified to work on the subscription, and it uses the same usage limits.
   - One run at a time, cancellable.
   - The allowed tools are restricted to the repo, the job folder, the helper scripts and web search/fetch.
3. **Development and testing:** skills are built and tested by subagents in the main dev session, against seeded test jobs, with no `claude -p` involved.
   - Lyrics review: a job with known injected errors, scored on how many are found and fixed.
   - Backgrounds: real jobs, judged on the contact sheet plus `bg_timeline.py validate`.

Either way, a skill only edits files in the job folder and calls the CLI, so all three paths behave identically.

### `karaoke-lyrics-review`
- **Inputs:** a job id.
- **Process:**
  1. Read `lyrics.final.json`, every file in `lyrics/sources/` and `transcript.whisper.json`.
  2. Focus on flagged lines and `unmatched_audio` regions, and decide case by case: is it a misheard word, a missing chorus repetition, an ad-lib, a wrong language variant, or a bad line break?
  3. It may web-search the lyrics to settle disagreements between sources.
- **Helper scripts:**
  - `lyrics_diff.py`: a side-by-side of sources vs whisper per line.
  - `lyrics_edit.py`: apply edits as a JSON patch, preserving timings where possible and marking `origin: ai`.
  - `karaoke validate`.
  - `karaoke stage <job> align --respect-manual`.
- **Output:** an edited `lyrics.final.json` plus `review.md`, a change log with a reason for each edit.

### `karaoke-backgrounds` (creative and opinionated)
- **Process:**
  1. Look at the song: title, artist, lyrics themes, section structure (from lyrics plus audio energy via `librosa` RMS/onsets) and mood.
  2. Write a short **art direction** (`art_direction.md`): visual concept, palette, pacing rules. For example: "change scenes only on section boundaries, hold ≥ 8 s, use band photos for the chorus, nature for the verses, and darken everything to 45 % so the lyrics stay readable."
  3. Search, download and curate the images.
  4. Write `background.json` and render a contact sheet and preview for self-check.
- **Helper scripts:**
  - `audio_sections.py`: energy curve and section boundaries.
  - `img_search.py` (built as `karaoke/bg_search.py`, CLI `bg-search`): Wikimedia Commons, free with licence info. The Pexels and Unsplash adapters switch on only when `PEXELS_API_KEY` / `UNSPLASH_ACCESS_KEY` appear in `.env`.
  - `img_fetch.py`: download, validate resolution, crop to 16:9 and record credits.
  - `bg_timeline.py`: set/validate scenes and snap to section boundaries.
  - `contact_sheet.py`: a grid of the chosen images with timestamps, so Claude can *look* at its choices.
- **Rules:** never touch `locked` scenes, always keep lyric legibility (dim/blur), and record each image's credit and source.

---

## 7. Repository layout

```
karaoke/
  PLAN.md  README.md  pyproject.toml (uv, py3.12)  .gitignore  .env.example
  karaoke/
    config.py                 # paths, ffmpeg binary, model choices (env + .env)
    models.py                 # pydantic: Job, Meta, LyricsFinal, Background, Transcript
    store.py                  # job folder IO, hashing, stage cache
    pipeline.py               # stage registry, runner, progress events
    cli.py                    # typer CLI
    stages/  ingest.py separate.py transcribe.py lyrics_fetch.py align.py backgrounds.py render.py
    lyrics/  lrclib.py genius.py youtube_subs.py normalize.py titleclean.py
    align/   dp.py interpolate.py confidence.py
    render/  ass.py compositor.py ffmpeg.py
    api/     app.py jobs.py lyrics.py backgrounds.py render.py ai.py sse.py
  web/                        # Vite + React + TS
  .claude/skills/karaoke-lyrics-review/  SKILL.md  scripts/
  .claude/skills/karaoke-backgrounds/    SKILL.md  scripts/
  tests/  fixtures/jobs/demo/            # tiny synthetic job folder: 15 s audio, fake transcript, lyrics
  data/                                  # gitignored
```

---

## 8. Phases and parallel workstreams

### Phase 0: Foundation and spikes (main session, sequential, ~1 sitting)
- `git init`, `pyproject.toml` (py3.12, torch 2.2.2, numpy<2, all deps declared up front so parallel agents never edit it), `uv sync`.
- `brew install ffmpeg-full`, then a libass smoke test.
- **Contracts:** `models.py` (pydantic + exported JSON Schema), `store.py`, the stage interface, the CLI skeleton.
- **Fixture job** in `tests/fixtures/jobs/demo/` so every workstream can be developed and tested without downloading anything.
- **Spikes** on the 2 test songs (§10):
  - The separation model benchmark.
  - Whisper on the stem vs the mix.
  - DP alignment vs stable-ts forced alignment.
  - Results go to `docs/benchmarks.md`.
- **Exit criteria:** `karaoke stage demo <name>` dispatches to stub stages, and the contracts are frozen.
- Then the first commit, and the public GitHub repo `karaoke` is created and pushed.

### Phase 1: Pipeline MVP (parallel subagents, each owns disjoint files)

| Agent | Owns | Delivers |
|---|---|---|
| **A: ingest** | `stages/ingest.py`, `lyrics/titleclean.py`, `lyrics/youtube_subs.py` | yt-dlp download, audio extraction, metadata, manual-subs → text + timings |
| **B: separate + transcribe** | `stages/separate.py`, `stages/transcribe.py` | audio-separator wrapper (model from config), faster-whisper wrapper |
| **C: lyrics fetch** | `stages/lyrics_fetch.py`, `lyrics/lrclib.py`, `lyrics/genius.py`, `lyrics/normalize.py` | source fetchers, chorus expansion, section parsing, `chosen.txt` |
| **D: align** | `stages/align.py`, `align/*` | DP alignment, interpolation, LRC anchors, confidence (tested on fixtures plus synthetic perturbations) |
| **E: render** | `stages/render.py`, `stages/backgrounds.py` (default), `render/*` | ASS generator, compositor (colour/image/slideshow), preview/final presets |

Then the **integration step** (main session): an end-to-end run on 2 real songs and fixes.
- **MVP is done when** `karaoke new <url> && karaoke run <job>` produces a watchable mp4 with correct lyrics on a solid background.

### Phase 2: Web app (parallel)

| Agent | Owns | Delivers |
|---|---|---|
| **F: API** | `api/*` | REST plus SSE, job queue, file uploads, "re-run from stage", serving stems and images to the UI |
| **G: UI** | `web/*` | Screens 1–5 against a mocked API first (typed from the JSON Schema), then wired to the real one |

### Phase 3: AI skills (parallel)

| Agent | Owns | Delivers |
|---|---|---|
| **H: lyrics-review skill** | `.claude/skills/karaoke-lyrics-review/` | SKILL.md and helper scripts, plus `tests/skills/seed_errors.py`, which makes a copy of a real job with injected mistakes |
| **I: backgrounds skill** | `.claude/skills/karaoke-backgrounds/` | SKILL.md, helper scripts, compositor features it needs (Ken Burns, xfade) |
| **F (continued)** | `api/ai.py`, UI AI panels | Copy-command panels, the "Ask AI" button (headless `claude -p` with log streaming) and folder-watch live refresh |

**Skill evaluation (main session):**
1. Fresh subagents act as the "user's Claude," following each SKILL.md on test jobs: the seeded-error job for lyrics review, both real songs for backgrounds.
2. I review their results and tighten the SKILL.md and scripts.
3. I repeat until the lyrics skill fixes the seeded errors without breaking correct lines.

### Phase 4: Polish
Quality presets, duet/multi-singer colours, a renders gallery, a README with setup steps, and GitHub CI (lint plus unit tests on fixtures).

### How parallel work is kept safe
- Contracts (`models.py`, the job folder layout, the CLI stage signature) and **all dependencies** are frozen in Phase 0. Agents never edit `pyproject.toml`, `models.py` or other agents' files. Contract changes come back to the main session.
- Each agent delivers unit tests that run against `tests/fixtures/` with no network and no heavy models, plus a short note on what it verified.
- The main session reviews diffs, runs the full test suite and does the real end-to-end run between waves.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| CPU-only separation is slow (Roformer may take 10–20 min per song) | Use MDX-Net by default, cache stems, keep "HQ" opt-in |
| No lyrics found (obscure songs) | Fall back to whisper text plus the AI review skill (web search) plus the paste box |
| Lyrics source differs from what's actually sung (repetitions, live versions) | DP alignment flags `unmatched_audio`, then AI review and tap-to-sync in the UI |
| YouTube / yt-dlp breakage | Pin and update yt-dlp in the venv, plus an "upload audio file" fallback |
| torch/numpy pin drift on Intel Mac | Lock with `uv.lock`, pin `torch==2.2.2`, `numpy<2` |
| Image licensing for backgrounds | Prefer Wikimedia/Pexels/Unsplash and record credits. Personal use. |

---

## 10. Decisions (answered 2026-11-09)
1. **Languages:** PT-BR and EN. Normalization handles accent folding and contractions for both (EN: "gonna"/"going to," "'cause"; PT: "pra"/"para," "tá"/"está," "cê"/"você").
2. **GitHub:** a **public** repo named **`karaoke`**, created with `gh` at the end of Phase 0. Fetched lyrics, audio and images never enter the repo (`data/` is gitignored).
3. **API keys:** none for now. Genius, Pexels and Unsplash stay optional, enabled via `.env` later (`.env.example` lists them).
4. **Installs:** approved, `brew install ffmpeg-full` plus a uv-managed Python 3.12.
5. **Test songs:**
   - EN: `https://www.youtube.com/watch?v=Zdh2hot8rjU`, *Sweet Child O' Mine* (Captain Fantastic cover). It has manual subs and an LRCLIB duration mismatch is likely.
   - PT-BR: `https://www.youtube.com/watch?v=f8x_t8wPFcM`, *Sacal – Vão Querer Me Internar*. It has fast dancehall vocals, no manual subs and may not be on LRCLIB, so it is the hardest case for alignment and AI review.
   - Playlist parameters (`&list=…`) are always stripped (`--no-playlist`).
6. **AI skills:** `claude -p` works on the subscription, so the UI keeps a one-click option. Interactive sessions are the primary path, and skill testing is done by subagents (§6).
