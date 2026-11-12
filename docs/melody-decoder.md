# Melody decoder and syllable guide — 2026-11-11

This implementation guarantees a pitch for every **inferred** syllable, within the existing
word timings. It reduces large jumps substantially, but agrees less closely with the existing
full-song reference labels. It is a reviewable candidate, not a demonstrated overall accuracy
improvement. The reference audit below does not make that loss disappear.

## Where the pitch was collapsed

At base commit `2463e9f`, `karaoke/melody/track.py:80` calls `torchcrepe.predict` with
`decoder=torchcrepe.decode.viterbi`. Thus the tracker already used all allowed pitch-bin
activations inside a Viterbi decoder, rather than selecting an independent argmax first.
The collapse occurs at `track.py:83–87`: only its chosen F0 and the activation of that bin
survive. `karaoke/stages/melody.py:95` writes those arrays to `cache/melody.f0.npz`.
`karaoke/melody/notes.py:441` then segments that single contour; alternative peaks cannot be
recovered by the later octave correction or song-profile correction.

Verified against the locally installed torchcrepe **0.0.24**:

- `model.py:108` returns independent sigmoid activations. They are not a categorical probability
  distribution. `decode.py:66` then applies softmax; among allowed bins, the maximum ratio becomes
  at most e, compressing the pitch evidence.
- `decode.py:59–61` uses a triangular continuity rule: nonzero only for distances below 12 bins
  (2.4 semitones). It models neither note changes nor voiced/rest states.
- `core.py:117–132` decodes inference batches separately: with our batch size 256 and 10 ms hop,
  the Viterbi path restarts every 2.56 s.
- `convert.py:12–22,51–57` adds random triangular ±20-cent dither to decoded pitch.

The new stage calls `track.crepe_activation`, retains every bin within the existing 70–900 Hz
search range, and caches rounded uint8 activation (`act`, 255 = 1), `bin0`, and `hop_s` along
with `f0`/`conf`. The latter still serve the contour endpoint and renderers; they are now a
deterministic local weighted peak, masked below activation 0.12. The note decoder consumes the
activation array, not that peak. Storage quantisation and semitone pooling still discard fine
detail; this is not a claim to retain an unmodified 360-bin floating-point model output.

`decode.semitone_salience` pools the strongest bin per tuned semitone. `decode.viterbi` then
chooses a whole-sequence path with interval and note-change costs, explicit rests and remembered
pitch across silence. A one-frame rest pays the same interval/change cost plus on/off costs,
so it cannot bypass the melodic prior. Long low-evidence silence forgets the previous pitch.
Scores sum sigmoid salience, not log probabilities; weights are empirical. The interval table
was estimated from these same references, so the evaluation is **not held out**.

The handover's activation analysis found the reference pitch among the strongest two pooled
peaks at only 13 of 120 wrong-jump endpoints. That is evidence against an easy alternative-peak
fix, but does not establish that the reference is correct or that backing vocals caused the
mismatch. Those possibilities need listening and independent annotation.

## Syllable design and contract

Previously the word file had no syllables. The renderer's apparent syllable slits were note
boundaries, often created by cutting held notes at **word** starts.

1. `lyrics.syllables.split_syllables` supplies dependency-free PT-BR rules and English vowel
   heuristics. Pieces preserve the complete word text. Unsupported scripts, digits and spelled
   letter tokens fall back to one piece. These estimates are imperfect pronunciations.
2. `melody.syllables.boundaries` places ordered cuts on an approximately 30 ms grid. Character
   lengths give a weak timing prior; confidence dips and preliminary pitch changes favour cuts.
   Very short words use proportional sub-grid spans rather than dropping syllables.
3. A second whole-song Viterbi pass forbids notes outside those spans and rests inside them.
   This is a two-pass approximation, not joint syllable/pitch optimisation. Strong sustained
   changes can yield multiple notes in one syllable (melisma).
4. Words with detectable trailing silence can end their last syllable early. Completely
   unheard syllables retain a span, infer a neighbouring pitch and have zero support; all-silent
   input uses an explicitly zero-confidence middle-C placeholder. Invalid/zero-duration word
   timings fail clearly instead of pretending the coverage guarantee was met.
5. `validate_melody` checks word coverage, exact text reconstruction, note links, containment,
   and notes tiling every syllable without internal gaps. Overlapping input words keep their
   own spans; no guarantee of a single simultaneous voice is made for overlapping words.
6. No lyric file uses free note/rest decoding. Its lighter prior (`weight=1, change=2,
   on=off=0.5`) preserves more fast solo notes; the constrained guide uses
   `weight=2, change=4, on=off=1.5, syllable=0.5`.

**Contract change:** `Melody.version` defaults to **2**, while version 1 remains readable.
`Melody.syllables` is a list of `{line_id, word_index, text, start, end}`; `Note.syllable` is its
index, or null for legacy/free notes. Existing note, key, tuning and display fields remain.
Schemas and generated TypeScript types are updated. Stage method is **5**, track cache version
is **3**. Old caches recompute. A failed CREPE run still falls back to pyin; its single contour
is represented by Gaussian pitch bumps, which cannot recover alternatives pyin already discarded.
No path contract changes or extra dependencies are needed. Torch remains in the isolated stage.

Video ASS bars split at the explicit syllables, and the rail displays their pieces. Zero-support
notes are faint even with pyin. The web roll marks syllable starts, shows the count, and computes
missing spans against syllables so trimmed rests are not labelled missing singing. MIDI and
contour API shapes remain compatible. The separate `bg-pictures` Video overlay consumes the
preserved note fields and was also checked in an isolated combined build.

## Measurements

11 real jobs; eight songs with non-null octave offsets for the original pitch mean; all 40
vocadito clips. A jump is ≥7 semitones between notes at most 0.5 s apart. “Wrong” below means the
reference moves ≤4 semitones: **a reference-disagreement classification, not a listening verdict**.

| Measure | Method 4 baseline | Method 5 syllable candidate |
|---|---:|---:|
| Large jumps / adjacent pairs | 230 / 4,126 (5.6%) | 73 / 4,861 (1.5%) |
| Jumps inside reference spans | 151 | 48 |
| Both endpoints match reference | 17 | 9 |
| Reference also leaps, pitches differ | 30 | 6 |
| Reference moves ≤4 semitones (“wrong”) | 92 | 29 |
| Unjudgeable endpoint | 12 | 4 |
| Reference notes found | 93.9% | 90.6% |
| Exact semitone agreement | 67.6% | 60.4% |
| Timing-tolerant agreement | 76.1% | 72.1% |
| Within one semitone | 74.7% | 67.3% |
| Octave disagreement | 1.2% | 0.9% |
| Onset F, 150 ms | 67.3% | 61.0% |
| Words with no overlapping note | 92 / 2,810 (3.3%) | 0 / 2,810 |
| Inferred syllables with no linked note | not represented | 0 / 3,796 |
| Note-seconds outside union of word spans | 274.989 / 1,373.81 (20.0%) | 0 / 1,210.975 |
| Syllable invariant violations | not represented | 0 |

Exact agreement for each song, baseline → candidate: Billie Jean 52.3 → 50.9; Bohemian Rhapsody
49.2 → 43.6; Deu Onda 85.7 → 78.6; Imagine 77.7 → 76.4; Mas Que Nada 78.6 → 67.9; Nothing
Compares 2 U 77.6 → 66.5; Respect 54.9 → 50.0; Tudo Que Você Podia Ser 64.7 → 49.6 (%).

The corrected free decoder with the constrained prior, **before** enforcing syllables, gives
162 jumps, 72 “wrong”, 64.2% exact and 76.7% tolerant agreement. A weaker constrained prior
(`weight=1, change=2, on=off=.5`) gives 155 jumps, 82 “wrong”, 61.4% exact, 75.4% tolerant,
and 57.6% onset F. No tested setting establishes an across-the-board improvement.

Clipping the **unchanged baseline notes** to the words alone reduces exact agreement to 63.5%
and found coverage to 88.8%. Thus strict word containment and the existing timings account for
part of the trade-off, but not all of it. This diagnostic does not meet the syllable guarantee.

| Vocadito, reference A1 | Baseline | New free mode |
|---|---:|---:|
| Found | 97.1% | 92.3% |
| Fractional pitch within 50 cents | 80.9% | 84.2% |
| Semitone label within 50 cents | 77.1% | 79.3% |
| Onset F, 50 ms | 75.0% | 72.6% |
| Note F, onset + pitch | 67.0% | 64.4% |
| Estimated notes / reference notes | 0.81 | 1.04 |

This gains overlapping pitch agreement but loses coverage and event matching. On reference
**A2**, pitch is 90.0 → 90.2%, label 85.3 → 84.4%, note F 66.8 → 57.5%. The larger event loss
against A2 is consistent with over-segmentation; a single favourable A1 pitch score would conceal it.

## Critical audit of the references

The local label “verified to line up” means numerical alignment checks, **not listening**.
None of the cached full-song references was manually auditioned during its original setup.
The reference authors themselves checked their transcriptions; our mapping to different audio
masters has not had the same note-by-note review.

- [Rock Corpus's methodology](https://rockcorpus.midside.com/melodic_transcriptions.html) confirms
  one transcriber per song, no direct note offsets, and a forced choice of one vocal line in
  ambiguous multi-voice passages. Our importer invents ends from the next onset, capped at two
  seconds. That makes durations and overlap windows uncertain. Its
  [timing data](https://rockcorpus.midside.com/timing_data.html) are tied to particular CD releases;
  these jobs use YouTube masters with fitted offsets/time scale.
- [Hooktheory/Sheet Sage](https://github.com/chrisdonahue/sheetsage#hooktheory-dataset) supplies
  user and refined beat alignments, not a guarantee of performance-exact timing. The local set
  already overrides a roughly half-second alignment error on Mas Que Nada. Its octave estimate
  is marked weak, yet the legacy mean treats every non-null offset as known. Ai Se Eu Te Pego's
  octave is unknown and it is already excluded from the eight-song exact mean.
- [The revised Rock Corpus release](https://rockcorpus.midside.com/lyrics_stress.html) explicitly
  mentions pitch/rhythm corrections and includes syllable/melisma annotations. I downloaded only
  these text annotations to temporary storage and compared shared metrical onsets with our old
  timed-note lists. Of 1,140 shared onsets, **19 pitches differ**: Billie Jean 9/468, Imagine 2/174,
  Nothing Compares 2 U 5/269, Respect 3/229. Another 37 revised syllable onsets have no old match.
  Differences establish reference-version uncertainty; they do not prove the newer label fits
  our particular master better. No local reference was rewritten.
- [Vocadito's authors](https://arxiv.org/abs/2110.05580) explicitly study subjectivity in vocal
  notes, including whether ornaments form separate events and whether an onset includes a
  consonant. Our own A2-vs-A1 scoring gives 79.8% overlapping pitch agreement and 73.96% note F.
  These are inter-annotator agreements, **not ceilings** on model accuracy. “Exact human notes”
  was an overstatement in the old benchmark description.

Additional checks, with identical selection for baseline and candidate:

| Exact agreement sensitivity | Baseline | Candidate |
|---|---:|---:|
| Legacy scoring, all eight | 67.58% | 60.42% |
| Sum covered time per pitch, invariant to splitting a held note | 67.63% | 60.93% |
| Same aggregate scoring, omit weak-octave Mas Que Nada | 66.07% | 59.94% |
| Same aggregate scoring, five Rock Corpus songs | 62.41% | 57.64% |
| Same aggregate scoring, reference windows overlapping lyric words | 72.05% | 67.40% |

The legacy scorer chooses the **single** overlapping note with greatest overlap. Artificially
splitting a held note into syllables can change its answer. `audit_melody_gold.aggregate_pitch`
unions intervals by pitch before summing support; the original scorer remains unchanged for
comparability. Removing that bias raises the candidate by 0.51 points, not enough to explain
the whole difference.

Fixed ±100 ms shifts show substantial sensitivity on short excerpts: Deu Onda baseline exact
agreement ranges from 45.2% to 90.5%, versus 85.7% at the stored alignment. These are diagnostics,
not permission to select the best lag for each algorithm. A low stem level likewise flags an
inspection case; it cannot distinguish missing vocals from separation failure. Tudo has 41 of
119 reference onset windows with no overlapping lyric word and 29 with stem level below −45 dB.

On the **same shared-onset subset**, changing old labels to official revised labels yields:

| Song | Baseline old → revised | Candidate old → revised |
|---|---:|---:|
| Billie Jean | 51.5 → 51.5% | 51.1 → 50.0% |
| Imagine | 79.9 → 81.0% | 81.6 → 82.8% |
| Nothing Compares 2 U | 75.8 → 74.7% | 70.3 → 69.1% |
| Respect | 55.5 → 56.3% | 51.5 → 52.4% |

This audit supports treating the references critically; it **does not reverse the overall
comparison**. A listening review should distinguish tracker errors, alternative vocal lines,
incorrect word alignment, and annotation ambiguity, with the algorithm identity hidden.
No note-level listening adjudication is claimed here.

The revised stress files also provide an independent check on the English splitter: 971/998
word tokens (97.3%) have matching syllable counts across these four songs. A later rendered-preview check exposed an English contraction error (silent e in “there’s”);
that rule was corrected and the comparison rerun, with the same count agreement. This is a
count comparison, not validation of cut times or of PT-BR pronunciation. The guarantee is coverage of inferred syllables, not that every syllable
boundary matches the performance. Note-onset F is not a direct syllable-timing accuracy score.

## Reproduction and verification

From the `melody-syllables` worktree (job outputs are isolated copies; audio was initially
symlinked; the preview job's audio was copied for API containment checks):

```sh
uv run python spikes/melody_jumps.py --melody melody.before.json
uv run python spikes/melody_decode.py --score before
uv run python spikes/melody_decode.py --mode syllables --tag final --json /tmp/melody-metrics.json
uv run python spikes/audit_melody_gold.py --tags before final --json /tmp/reference-audit.json
# Optional: unpack the official 2018 stress zip locally, then add --revised-dir <directory>.
uv run pytest
uv run ruff check karaoke tests spikes
(cd web && npm test && npm run typecheck && npm run build)
```

Tests cover pitch alternatives, the rest bypass, long-silence backtracking, short/empty audio,
text preservation, melismas, unheard and tiny syllables, invalid word timings, all coverage
invariants, cache reuse, pyin fallback, v1 reading, API/MIDI export and both render paths.
The real CREPE melody stage also completed on the synthetic demo and Imagine. No fetched
lyrics, audio, annotations, model caches or per-job outputs belong in the public PR.

Final validation: 402 Python tests (11 slow tests deselected), 45 web tests, Ruff, TypeScript
type checking and the production build pass. An eight-second 720p libass render was inspected;
the syllable rail and pitch contour render correctly. Browser checks cover the melody editor
and the separate Video overlay in a temporary combined build.

The next step is the [local A/B listening review](melody-listening.md), with 36 diagnostic
excerpts and saved first preferences before optional reference playback. No listening results
have been entered for that session yet.
