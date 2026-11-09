# Phase 0 benchmarks (2026-11-09)

Machine: Intel i5-1038NG7 (4 cores), 16 GB, CPU only. Scripts: `spikes/bench_separation.py` and `spikes/bench_alignment.py`.
Raw results go to `data/bench/` (gitignored). The separated 45 s excerpts stay there for listening:
`data/bench/<video id>/<model>/{vocals,instrumental}.wav`.

> **Timing caveat:** a Homebrew source build (`ffmpeg-full` dependencies) shared the CPU during the
> first song's runs. The same model was about 3× faster on the second song once the build quietened down
> (MDX: RTF 2.7 → 0.9), so the second song's numbers are the more representative ones.

## Separation: 45 s excerpts

RTF = seconds of compute per second of audio (lower is faster). WER = whisper large-v3-turbo on the
**vocal stem**, scored against the duration-matched LRCLIB lyrics of the exact recording (the EN cover only).
Sacal has no reference lyrics, so it is compared by agreement between models.

| Model | RTF EN (contended) | RTF PT | WER EN vocals | Full 3:40 song, est. |
|---|---|---|---|---|
| Original mix (no separation) | — | — | **0.674** | — |
| `UVR-MDX-NET-Inst_HQ_3.onnx` | 2.74 | **0.88** | 0.326 | ~3–4 min |
| `Kim_Vocal_2.onnx` | 2.57 | **0.88** | 0.279 | ~3–4 min |
| `htdemucs_ft.yaml` | 10.1 | 7.22 | **0.209** | ~25 min |
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | 12.5 | 9.49 | **0.209** | ~35 min |

- **Separating before whisper is essential.** On the mix, whisper dropped most of the singing on both songs: EN WER 0.67, and the PT excerpt gave 50 words vs about 142 from any stem.
- **PT (Sacal):** the stems of all four models produce nearly the same transcript (4–7 % word distance from each other). Model choice barely matters for transcription there.
- **EN (the cover):** the heavier models give about 25 % fewer errors (0.21 vs 0.28). Part of the remaining 0.21 is the reference itself: LRCLIB lines differ slightly from what is sung, and whisper skips vocalisations like "oh, oh, oh."
- **Vocal leakage proxy** (words whisper finds in the instrumental): 0 for every model. The proxy can't tell them apart, so **instrumental quality has to be judged by ear.**
- Whisper itself is cheap by comparison: about 0.2–0.5 × real time on the vocal stem (9–21 s per 45 s excerpt).

### Decision
- **Fast default:** `Kim_Vocal_2.onnx`. It ties for fastest (under 1× real time) and has the best fast-tier vocals for whisper.
- **HQ option:** `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (BS-Roformer). It ties demucs on WER, and its published SDR is much higher. It costs about 10× real time.
- **Open: listen to the instrumentals** in `data/bench/*/` before freezing this. `UVR-MDX-NET-Inst_HQ_3` is trained for the instrumental and may sound cleaner as the karaoke track, even if its vocals are worse for whisper.
  - If so, the fast tier could become a two-model split: Kim → vocals for whisper, Inst HQ 3 → instrumental for the video. Each costs about 1× real time.
- Worth trying later: `mel_band_roformer_karaoke_*`, which keeps the backing vocals in the instrumental (the classic karaoke sound), and the CoreML provider for ONNX models.

## Full-song separation
`Kim_Vocal_2.onnx` on the full 3:40 cover: **218 s (RTF 0.99)**, CPU mode. audio-separator found no
acceleration, so CoreML for ONNX is untested.

## Transcription: the VAD filter drops whole sections

faster-whisper turbo on the full-song vocal stem of the cover. Columns count lyric words (reference) vs words whisper
heard, per 20 s window:

| Setting | Time | Words heard / 237 | Lyric words matched | Per-20 s coverage |
|---|---|---|---|---|
| `vad_filter=True` (Silero defaults) | 40 s | 113 | 46 % | normal until 100 s, then **almost nothing from 100–200 s** |
| `vad_filter=False` | 376 s | 194 | **76 %** | close to the reference in every window |

- **The Silero VAD with default thresholds treats long stretches of the separated vocals as silence.** That loses half the song.
- Without VAD the coverage is right, but whisper is about 9× slower (it decodes every 30 s window, including instrumental ones).
- **Fix for workstream B:** compute our own voice-activity regions from the **RMS energy of the vocal stem**, then pass them to faster-whisper as `clip_timestamps` (`vad_filter=False`). The instrumental is already removed, so energy is a reliable VAD here.
  - If energy alone is too noisy, try Silero with a low threshold (`vad_parameters={"threshold": 0.2, "min_silence_duration_ms": 2000}`).
  - **Acceptance:** at least the no-VAD coverage (≥ 75 % matched on the cover) in at most 2× the VAD-on time.

## Alignment: whisper word timestamps vs stable-ts forced alignment

The reference is the 53 line starts of the cover's own synced LRC (hand-made, about ±0.3 s). We measure line start error.

| Method | Time | Lines placed | Median err | p90 err | ≤ 0.3 s | ≤ 0.5 s |
|---|---|---|---|---|---|---|
| (a) whisper (VAD on) + greedy difflib match | 29 s | 18 / 53 | **0.20 s** | 1.1 s | 19 % | 25 % |
| (b) stable-ts `align()` of the LRC text, openai-whisper turbo (PyTorch) | **678 s** | **53 / 53** | 0.51 s | 5.1 s | 26 % | 45 % |

- (a) is precise where whisper heard the line, but it only placed 18 lines (the VAD problem above). The greedy matcher also produced two catastrophic errors (18 s and 26 s) by jumping to the wrong chorus repetition. **This confirms the plan: global DP alignment, not greedy matching.**
- (b) places everything, but drifts by several seconds in places (worst 8.7 s) and costs 3× real time on top of separation.
- **stable-ts cannot share a process with faster-whisper** (segfault; that is the torch + ctranslate2 conflict again). It has to use the PyTorch whisper weights (a further 1.5 GB) in its own process.

### Decision
- **Primary aligner: workstream D's DP** over whisper words (run without the lossy VAD), plus interpolation, plus **LRC line anchors when the duration matches** (as it does for the cover).
- **stable-ts is not a dependency.** It stays in the `spike` group as an optional experiment for later: forced alignment *per line window* between DP/LRC anchors could fix the interpolated gaps much more cheaply than whole-song alignment.
- Workstream D's acceptance on the cover: ≥ 90 % of lines within 0.5 s of the LRC starts. Workstream B's VAD fix has to land first.

# Phase 1 end-to-end (2026-11-09)

`uv run karaoke run <id>` on both test songs, run one after the other. The final 1080p video is written to `output/karaoke.mp4`.

| Stage | Cover (220 s, EN) | Sacal (153 s, PT-BR) |
|---|---|---|
| ingest | 5.7 s (mp4 already present) | 5.8 s |
| separate (Kim_Vocal_2) | 208 s (RTF 0.87) | 137 s (RTF 0.80) |
| lyrics_fetch | 0.8 s: LRCLIB synced, Δ0.3 s, line timings on | 2.0 s: nothing found, whisper fallback |
| transcribe | 77 s, 195 words | 81 s, 340 words, detected `pt` |
| align | 0.1 s, 189/237 words heard, 16 lines flagged | 0.1 s, 51 lines, all flagged (no lyrics source) |
| render (1080p, x264 medium) | 92 s | ~60 s |
| **total** | **about 6.5 min** | **about 4.75 min** |

**Alignment on the cover** (line starts vs the cover's own synced LRC, with LRC anchors on):
- All 53/53 lines are placed.
- Median error is 0.18 s. 58 % of lines are within 0.3 s and 74 % within 0.5 s.
- The worst line is 2.4 s off. There are no wrong-repetition jumps, which Phase 0 had at 18–26 s.
- 48 words were interpolated; most are vocalisations ("oh") and the dense outro.
- The ≥ 90 %-within-0.5 s target is **not met yet**. The remaining misses are what the lyrics-review skill and the UI's tap-to-sync are for.

**Visual check of the frames:**
- The fill tracks mid-word, the next line waits dimmed in the alternate row, and instrumental gaps are empty until the countdown.
- PT accents render correctly.
- **Known issue:** in whisper-fallback songs (no lyrics source), lines follow whisper's phrasing. Sometimes a phrase is split across lines, and whisper adds a trailing "…". The AI review skill should repair the line breaks.

## Rendering fixes (2026-11-09)

- **Pauses.** Whisper's word timestamps are back to back, so a pause was folded into a word and the
  fill crawled across it. `align` now reads the vocal stem's loudness (`karaoke/align/voicing.py`)
  and cuts pauses out of words, judged against each word's own loud level. Absolute silence doesn't
  work, because reverb and leakage keep the stem at -30 to -45 dB between phrases.
  - A word's start only moves past a pause of 0.6 s or more. With 0.3 s, quiet sibilants ("s",
    "sh") counted as silence, and line starts got later (cover median error 0.18 → 0.33 s).
  - With the 0.6 s rule, cover line starts are unchanged: 31/53 within 0.3 s, 39/53 within 0.5 s,
    median 0.18 s.
  - Sacal is sung almost continuously, so it barely changes.
- **Pauses that stay loud (Sacal).** Loudness alone missed Sacal's pauses. At 16–20 s the stem
  stays at -25 dB (breath, noise, leakage) with no singing. At 22–27 s it is the echo of a held
  "boy". In both places whisper stretched "Eles vão" across the gap, so the fill crawled over it.
  `voicing.py` now uses two more signals:
  - **pyin pitch.** A stretch of 1 s or more with no pitch counts as a pause if it's at least 12 dB
    quieter than the singing around it, or if whisper stretched a word across it.
  - **Syllable onsets.** These are sharp 10 dB rises. Words in a stretch of 1.5 s or more with no
    onset are moved, but only if whisper made them at least 3× their natural length and the next
    word starts at the next onset. The word before keeps its held note until the note fades.
  - In Sacal, only those two places changed (5 words). In the cover, 7 line starts moved later.
    At each one the stem is silent (−77 dB) at the LRC time and the new start matches the audio
    onset, so the LRC is early there. Against the LRC the cover now scores 28/53 within 0.3 s,
    37/53 within 0.5 s, median 0.28 s.
  - Pitch tracking adds about 20–35 s to `align` per song.
- **Dots.** Every pause of 2 s or more gets the three dots: the intro, between lines and inside a
  line. They go out one per beat, on a beat from librosa's tempo of the instrumental, folded into
  0.55–1.2 s. The cover's beat is 0.98 s, Sacal's 0.79 s.
- **Line breaks.**
  - The font size is now fixed. Widths are measured with the real font file, scaled the way libass
    sizes fonts: predicted 1446 px, rendered 1443 px.
  - Too-wide lines, and wide lines with a strong break (punctuation, parentheses), are split by
    `karaoke/linebreak.py`.
  - Whisper-only lyrics are grouped by sentence starts and punctuation instead of whisper segments.
- **Rows.** There are 2 rows, or 3 wherever lines come less than 2 s apart (the median over 5
  lines). Sacal switches to 3 rows when the rap starts, at 0:42; the cover only during its fast
  "Sweet" run.

## Missed phrase at a clip cut (2026-11-09)

- **The miss.** Whisper skipped Sacal's "Ou vão querer me internar" (29.9–31.3 s).
  - Whisper's 30 s window forced a cut in a long rap stretch. The quietest frame was a 0.1 s
    breath at 29.85 s, right before the phrase, so the next clip started with no lead-in, and
    whisper often drops words at a clip's very start.
- **Overlapping clips didn't work.** Overlapping split clips by 1 s made whisper decode other clips
  worse (340 → 295 words), so it was reverted.
- **What changed:**
  - Cuts now go in the quietest 0.3 s stretch, not the quietest frame. With it, whisper caught the
    phrase in its main pass (344 words).
  - **Gap fill:** gaps between words of 0.8 s or more, with loud voice and at least 2 syllable
    onsets, are re-decoded with 1 s of context, and only new confident words inside the gap are
    kept.
  - Without the onset requirement, the gap fill invented a word in a loud noise pause, so the
    requirement stays.
- **letras.mus.br.** It is searched through its public suggestion endpoint, and the song URL is
  saved to `lyrics/sources/letras.json`. Its pages refuse automated clients (403, even
  `robots.txt`), so the text has to be pasted by the user into `lyrics/sources/user.txt`.
