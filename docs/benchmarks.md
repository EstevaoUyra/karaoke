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

# Melody (pitch guide) accuracy (2026-11-09)

Complaint: the pitch guide "has a lot of missing places and does not seem so correct". Nobody can
listen in CI, so `spikes/eval_melody.py` measures proxies against the two test songs' lyrics
(`lyrics.final.json` word timings) and independent F0 trackers (`spikes/melody_trackers.py`):

- `cover`: share of sung-word time with a note (`nonote`: seconds without), `gap words`: words no
  note overlaps (the UI's "gaps"), `spill`: note time > 0.3 s from any word.
- `agree[R]`: note frames within 50 cents of tracker R where R is confident; `oct[R]`: an octave off.
- `pc`: the vocal stem's strongest CQT chroma bin equals the note's pitch class (notes synthesised to
  audio for the `chroma` cosine, which tells the same story).

**Where the old notes went missing** (`--gates`, share of sung-word time left after each gate):
pyin finds a pitch in 88 % / 94 % (EN / PT) of word time, but only 40 % / 26 % passes its voicing
probability ≥ 0.5; the steady-pitch rule adds ~11 points; short-run removal, the level gate and the
word filter cost a few points each. pyin's voicing probability is low on a separated stem's singing,
so the confidence gate, not the tracker's pitch, made the holes.

**Trackers** (CPU, 4 cores; 10 ms hop for CREPE):

| tracker | RTF | EN cover | PT cover | EN agree[pyin] | EN oct[pyin] |
|---|---|---|---|---|---|
| pyin (old gates) | 0.10 | 0.46 | 0.28 | (same tracker) | — |
| pyin (new gates) | 0.10 | 0.63 | 0.72 | (same tracker) | — |
| torchcrepe tiny | 0.17-0.23 | 0.92 | 0.94 | 0.82 | 2.3 % |
| torchcrepe full | 2.6-2.7 | 0.94 | 0.96 | 0.82 | 1.0 % |

**Before / after** (old: pyin + old gates; new: crepe-tiny + word prior; `@old` = accuracy on the
frames the old version also had notes):

| song | version | notes | cover | nonote | gap words | spill | agree[pyin] | agree[crepe-full] | oct[crepe-full] | pc | pc @old |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EN | old | 242 | 0.457 | 72 s | 26.6 % | 11.2 % | (0.89) | 0.831 | 2.6 % | 0.79 | 0.79 |
| EN | new | 381 | 0.919 | 11 s | 3.0 % | 7.4 % | 0.813 | 0.756 | 1.4 % | 0.69 | 0.77 |
| PT | old | 144 | 0.284 | 75 s | 45.9 % | 2.5 % | (0.94) | 0.878 | 0 % | 0.84 | 0.84 |
| PT | new | 377 | 0.940 | 6 s | 1.7 % | 2.7 % | 0.858 | 0.746 | 0.1 % | 0.67 | 0.83 |

- Two trackers agree on only ~83-87 % of confident frames even where both are confident (the
  ceiling of `agree`). On the frames the old version covered, the new notes are as accurate
  (pc @old 0.77-0.83 vs 0.79-0.84). The new frames are harder (breathy, low, rap: speech-like
  pitch, where chroma is weak), which lowers the totals.
- Where pyin and CREPE disagree, CREPE-tiny and -full usually agree with each other and pyin jumps
  octaves (EN 155-159 s: pyin 55/64, both CREPEs 74-76).
- **Choice: torchcrepe tiny** (MIT, weights bundled with the package). Full CREPE adds 2-4 points of
  coverage and halves the octave disagreement, but costs ~10 min per song on this CPU; it is
  available as `KARAOKE_MELODY_TRACKER=crepe-full`. pyin is the fallback when torch fails.
  The stage is now `isolated=True` (torch).
- Not tried: RMVPE (code Apache-2.0, but the official repo publishes no weights; the copy RVC
  redistributes has only a repo-level MIT tag, so the weights' licence is unclear), basic-pitch
  (polyphonic note transcription, needs extra deps).
- Gates now: words are a voicing prior (CREPE periodicity ≥ 0.15 with a smoothly moving pitch,
  level gate 35 dB instead of 25 inside words); dropouts ≤ 0.2 s inside a word are bridged
  before short runs are dropped; gaps ≤ 0.2 s between notes of one word are closed. A phrase-level
  octave fix (median of ±3 s) was tried and made `oct[crepe-full]` worse (1.4 → 2.6 %): dropped.
- Key: a third vote, the full mix's chroma, joins the notes and the instrumental (correlations
  averaged). EN: D major (votes: notes D major, instrumental A major, mix D major; margin 0.21).
  PT: G minor (notes D minor, instrumental G minor 0.87, mix D minor 0.42; margin 0.12).
- Stage runtime (crepe-tiny, child process included): EN 43 s for 3:40, PT 32 s for 2:33.
- **Snapping to the key: tried, not adopted.** Idea: a note outside the key whose pitch sits between
  two semitones (≥ 0.25 from its own) moves to the neighbour inside the key, to draw fewer
  accidentals. Checked against the vocal stem's CQT chroma over each such note: the chroma preferred
  the original note more often than the in-key neighbour (EN 9 vs 5, PT 12 vs 5; no margin from
  0.2 to 0.35 made the in-key side win clearly). The out-of-key notes are mostly real (blue notes,
  speech-like lines), so the notes are left as tracked.

## Melody against human references (2026-11-10)

First score against annotations made by people: `spikes/score_gold.py` on the local gold set
(`data/gold`, not in the repo: Rock Corpus transcriptions of whole songs, Hooktheory lead sheets of
one section). crepe-tiny, melody method 2. "Pitch" = our note is the reference's semitone, over all
reference notes; "found" = we have any note there; onsets matched within 0.15 s.

| Song | Ref notes | Found | Pitch | ±1 semitone | Onset F |
|---|---|---|---|---|---|
| Deu Onda (section) | 42 | 100 % | 86 % | 90 % | 77 % |
| Mas Que Nada (section) | 28 | 100 % | 75 % | 75 % | 55 % |
| Nothing Compares 2 U | 331 | 97 % | 70 % | 89 % | 66 % |
| Imagine | 220 | 100 % | 69 % | 85 % | 74 % |
| Tudo Que Você Podia Ser (sections) | 119 | 73 % | 56 % | 61 % | 65 % |
| Respect | 246 | 88 % | 47 % | 63 % | 62 % |
| Bohemian Rhapsody | 539 | 88 % | 47 % | 57 % | 66 % |
| Billie Jean | 509 | 92 % | 45 % | 66 % | 66 % |

- The most common error is one semitone flat (Imagine 15 %, Nothing Compares 2 U 14 %, Respect 12 %
  of found notes), far more often than one semitone sharp. Not yet explained: scoops at the note
  start, the tuning offset, or the transcribers writing the intended note are all candidates.
- Octave errors are rare (0–4 %) except in Bohemian Rhapsody's stacked voices.
- Mas Que Nada: 21 % of notes are a fourth above the reference: we follow a harmony voice.
- Ai Se Eu Te Pego is left out: the reference's octave is unknown and only 28 % of pitch classes
  agree, with a cluster 13 semitones off. The reference's alignment was never confirmed.
- The references are not exact: Rock Corpus onsets are quantised to a grid and have no note ends,
  Hooktheory rhythm is simplified. Treat differences of a few points as noise.

## Melody: fast notes, measured on solo clips (2026-11-10, melody method 3)

Why the scores above are low, in order of weight:

1. **The references and the stems, not the tracker.** Reading the raw contour directly under each
   reference note scores the same as our notes; crepe-full scores the same as crepe-tiny (Imagine
   69 vs 70 %, Respect 54 vs 51 %, Billie Jean 50 vs 49 %; pyin is far worse on stems). Where the
   voice is steady the contour agrees with the reference 85–90 % of the time (Imagine, Nothing
   Compares 2 U); the misses are where a reference note spans a moving pitch or a backing-vocal
   answer (Respect: 22 % on reference notes longer than 0.5 s).
2. **Two people do not agree either.** On vocadito (40 solo clips, exact notes by two annotators;
   `spikes/score_vocadito.py`) annotator A2 has a note within 50 cents of A1's on 80 % of A1's notes.
3. **Note-making lost fast notes.** On vocadito the contour is within 50 cents on 97.9 % of sung
   frames and under 96 % of the exact reference notes, but our notes reached only 73 %: notes around
   0.1 s were merged into a neighbour (17 % of reference notes) or dropped (9 %).

Change (method 3): a short run is kept when the voice sits on the semitone for at least 0.08 s
(`MIN_SHORT_S`, range ≤ 0.6 semitones, centred), hysteresis 0.7 → 0.5, voiced runs and notes kept
down to 0.05 s. Merely lowering the minimums scored the same but put a passing note on every leap.

| vocadito, per reference note | A2 vs A1 | method 2 | method 3 |
|---|---|---|---|
| we have a note | 92 % | 91 % | 97 % |
| median pitch within 50 cents | 80 % | 73 % | 81 % |
| semitone label within 50 cents | 80 % | 69 % | 77 % |
| onset 50 ms + pitch 50 cents, F | 74 % | 62 % | 67 % |
| precision of that | 81 % | 78 % | 75 % |

| Full songs, exact semitone | method 2 | method 3 |
|---|---|---|
| Deu Onda | 86 % | 86 % |
| Mas Que Nada | 75 % | 75 % |
| Nothing Compares 2 U | 70 % | 75 % |
| Imagine | 69 % | 71 % |
| Tudo Que Você Podia Ser | 56 % | 61 % |
| Respect | 47 % | 52 % |
| Bohemian Rhapsody | 47 % | 49 % |
| Billie Jean | 45 % | 46 % |

The cost: about 15 % more notes on full songs and onset precision down 3–5 points.
Not tried: a lead-vocal model to take backing vocals out of the stem, the likely next gain on
songs like Respect and Bohemian Rhapsody.

## Melody: the song's own pitch profile (2026-11-10, melody method 4)

Music knowledge as a prior, tested against the human references this time.

- **Adopted: the song's own profile** (`notes.refine_by_profile`). A note moves one semitone when
  the voice wavers on it (≤ 70 % of its frames on its own semitone), part of it already sits on the
  neighbour (≥ 15 %), and the neighbour's pitch class has ≥ 2× more of the song's note time. No key
  estimate: a blues or a modulating song has a wider profile and is corrected less. Steady notes
  never move. Needs 30 s of notes, so vocadito's short clips are untouched (scores identical).
- **Not adopted: snapping to the estimated key.** Out-of-key is a strong sign of an error (30–50 %
  of our wrong notes, under 5 % of the right ones on diatonic songs), and a plain snap gains a lot
  on Billie Jean (+59 notes / −6) and Imagine (+18 / 0), but it loses on Respect (+11 / −31),
  Bohemian Rhapsody (+14 / −36) and Nothing Compares 2 U (+27 / −31, where our key estimate is
  probably wrong). The profile rule gets most of the gain without the losses.
- **Not adopted: voting across repeated lines.** For words in lines sung more than once, the
  majority pitch replaced a differing one. With audio support required it moved 21 notes over the
  whole set (+0.1 point); without it, it lost. Singers vary the repeats.

| Full songs, exact semitone | method 2 | method 3 | method 4 |
|---|---|---|---|
| Deu Onda | 86 % | 86 % | 86 % |
| Mas Que Nada | 75 % | 75 % | 79 % |
| Nothing Compares 2 U | 70 % | 75 % | 78 % |
| Imagine | 69 % | 71 % | 78 % |
| Tudo Que Você Podia Ser | 56 % | 61 % | 65 % |
| Respect | 47 % | 52 % | 55 % |
| Billie Jean | 45 % | 46 % | 52 % |
| Bohemian Rhapsody | 47 % | 49 % | 49 % |
| Mean | 61.9 % | 64.4 % | 67.7 % |

The three thresholds were chosen on these same eight songs (a coarse grid; the neighbouring
settings score within 0.5 point and none makes any song worse), so the gain is not a held-out
number.

## Melody: error patterns, separation and harmony tests (2026-11-10, method 4)

`spikes/score_gold.py` now also prints a timing-tolerant score (`loose`: our note of the right
semitone sounds within 0.15 s of the scored window), both scores over notes of at least 0.2 s,
and a split by note length.

| Mean of the 8 songs | all notes | notes ≥ 0.2 s |
|---|---|---|
| exact semitone at the reference's instant | 68 % | 72 % |
| exact semitone, timing-tolerant | 76 % | 80 % |
| within a semitone, timing-tolerant | 82 % | 85 % |

| Note length | share of notes | exact | timing-tolerant |
|---|---|---|---|
| under 0.15 s | 9 % | 47 % | 62 % |
| 0.15–0.3 s | 40 % | 56 % | 67 % |
| 0.3–0.6 s | 35 % | 64 % | 74 % |
| over 0.6 s | 17 % | 69 % | 79 % |

What happens to a reference note (all songs pooled): correct 60.5 %, right pitch within 0.25 s
12.1 %, no note 6.0 % (27 % on Tudo Que Você Podia Ser), 1–2 semitones off 9.2 %, a third to a
sixth off 7.0 %, octave 2.2 %, other 3.0 %. Onsets have no systematic lead or lag (medians within
±30 ms), but 17–30 % differ by more than 150 ms: the references are quantised.

Tried, not adopted:
- **Lead/backing separation** (`UVR_MDXNET_KARA_2.onnx` on the vocal stem, about 1× real time,
  then crepe-tiny on the lead, loudness gate from the full stem). Exact semitone: Bohemian Rhapsody
  49 → 56 %, Imagine 78 → 79 %, Respect 55 → 52 %, Billie Jean 52 → 47 %, Mas Que Nada 79 → 64 %
  (it removes the group vocal that is the melody). On Respect the stem and the lead agree on 85 %
  of notes and are right on only 56 % of those: backing vocals are not what is wrong there.
- **Trackers as an ensemble** (crepe-tiny, crepe-full, pyin on the same stem): any of the three
  right gains 5–14 points as an upper bound; agreement between the two CREPE models is a good
  doubt signal (ours is right 56–73 % when they agree, 21–44 % when they differ). Not built:
  crepe-full costs 2.4–2.9× real time.
- **Chords from the instrumental** (24 triads, Viterbi-smoothed, as the tie-break for wavering
  notes): 64.4 → 64.8 % alone; the instrumental's chroma itself 66.0 %; neither adds anything
  on top of the song-profile rule (67.6 %), and a local profile (±20–40 s) is slightly worse.

## Melody: where the remaining errors are (2026-11-10, method 4)

- **Confidence predicts correctness.** By our note's CREPE confidence (all eight songs):

  | confidence | share of notes | exact | timing-tolerant |
  |---|---|---|---|
  | under 0.3 | 7 % | 16 % | 34 % |
  | 0.3–0.45 | 10 % | 43 % | 56 % |
  | 0.45–0.6 | 17 % | 56 % | 69 % |
  | 0.6–0.75 | 31 % | 70 % | 79 % |
  | over 0.75 | 34 % | 80 % | 89 % |

  The roll guide now draws notes under 0.4 faint (`render.roll.UNSURE_CONF`), as it already did for
  unsteady ones. Solid notes (74 % of all): 74 % exact, 83 % timing-tolerant. Faint: 37 % / 53 %.
  `score_gold.py` prints the scores over the sure notes (`pitch|sure`: mean 75 %, loose 82 %).
- **Bohemian Rhapsody is two songs.** The solo ballad (50–135 s) scores 75–100 % per passage;
  the a cappella opening (29–50 s) and the opera section (190–245 s) score 10–40 %, with our
  notes 8–24 semitones below the transcribed line: a choir, where "the melody" is a choice.
- **Tudo Que Você Podia Ser**: every missing note is in 80–100 s, where the vocal stem is silent
  (−55 dB) and there are no lyrics: the lead sheet covers an unsung passage. On the notes we do
  have it scores 89 %.
- **Respect and Billie Jean**: errors are spread evenly over the song (35–80 % per 20 notes) with
  no constant offset, and the stem and the isolated lead agree. Note by note, many are fast
  alternations in the transcription (66–64–66 at 0.13 s) where the voice stays near one pitch and
  only dips: the transcribers wrote the intended figure. Not fixable from the audio alone.

## Melody: full activation and syllable constraints (2026-11-11, method 5)

[Full report, reproduction commands and reference audit](melody-decoder.md).

On 11 jobs, large jumps fall from 230/4,126 to 73/4,861; jumps classified as wrong against the
reference fall from 92 to 29. All 2,810 words and 3,796 inferred syllables have pitches, with
zero note time outside words. This is a coverage/continuity gain, **not a demonstrated overall
pitch-accuracy gain**: eight-song exact reference agreement falls 67.6 → 60.4%, timing-tolerant
76.1 → 72.1%, onset F 67.3 → 61.0%. Vocadito A1 pitch improves 80.9 → 84.2%, while note F falls
67.0 → 64.4%; A2 note F falls 66.8 → 57.5%. Settings and priors used these same examples.

Treat the old “gold” terminology as human reference annotations, not infallible truth. Official
revised Rock Corpus annotations disagree at 19 shared pitch labels across four of these songs;
our local “verified” status only checked numerical alignment. Vocadito annotators agree at
73.96% note F under our scorer. Timing/weak-octave/subset and segmentation-invariant sensitivity
checks are in the report; they do not reverse the overall full-song comparison. No references
were edited to favour the candidate, and no listening adjudication is claimed.
