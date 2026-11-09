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
