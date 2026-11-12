# Local melody listening review

This review compares the method-4 baseline with the method-5 syllable candidate from
[the decoder report](melody-decoder.md). It is a diagnostic listening sample, not a new
overall accuracy score. No listening judgments have been supplied by the agent.

## Open the prepared review

From the `melody-syllables` worktree:

```sh
uv run python spikes/melody_listening.py serve --pack data/listening/melody-ab-ready --port 8776
```

Open <http://127.0.0.1:8776>. The server listens on loopback only. No external assets,
accounts, API calls or uploads are used. The original studio does not need to be running.

1. Play the original mix; switch to isolated vocals if the lead voice is hard to follow.
2. Switch A/B during playback. Both guides use the same instrument and fixed note gain,
   without confidence-dependent volume. They share the recording's playback clock. Use
   **Melody alone** to isolate the guide, or **Recording only** to return to the source.
3. Save which guide follows the lead vocal more closely, including “both right”, “both wrong”,
   or “I can't tell”. The labels are randomly assigned per excerpt.
4. Optionally inspect the reference after saving this first choice. It may be wrong. Some
   clips also offer **Revised pitches** from the official later annotations. This replaces
   only pitches at matched metrical onsets; it is not a full revised transcription or a
   timing correction. Reference note ends may be inferred.
5. Note pitch, timing, syllable or competing-voice issues, then **Save & next excerpt**.
   You can stop and return later. Only explicitly saved answers persist; unsaved edits are
   marked and navigation asks before discarding them.

Space plays/pauses; A and B switch guides; R returns to the recording. Shortcuts are ignored
while typing or using form controls. Eight-second excerpts loop by default. Edge-cut syllables
are not extraction errors, and these short contexts may not resolve every ambiguity.

After all excerpts are finished, **Reveal versions & results** asks for confirmation before
showing assignments and counts by selection group. The results use the first saved choices;
later changes and reference exposure remain in the export. A first choice can still be revised,
but its original value is preserved. Returning to an exposed excerpt is not a fresh blinded trial.

## What was selected

The prepared pack contains 36 non-overlapping, eight-second excerpts across all 11 real jobs,
three or four per song. The public review hides the selection reasons until completion:

| Selection cue | Excerpts |
|---|---:|
| Baseline large jump without a nearby candidate large jump | 7 |
| Retained or newly introduced candidate large jump | 6 |
| Different pitch/coverage at a word midpoint | 6 |
| Old versus revised reference pitches differ at a shared onset | 6 |
| Reference timing/lyric mismatch or weak octave assignment | 6 |
| Same pitch and coverage around the focus | 5 |

Selection uses a fixed seed, rotates categories and prefers underrepresented songs. A and B
assignments are separately randomised and balanced 18/18 across the pack. The private key
records the selection cues, mappings, JSON input hashes, builder hash and audio-excerpt hashes.
The control cue checks semitones and coverage on a 100 ms grid for three seconds around the
focus; it does not guarantee identical audio over the whole eight-second clip.

All synths use semitone labels plus each algorithm's stored tuning, without a display octave
shift. Contiguous same-pitch notes are merged before synthesis, so artificial syllable splits
do not give one method extra attacks. The recording and vocal stem are downmixed/resampled
to 22.05 kHz and level adjusted once per excerpt, identically under both guides. Timbre and
gain are fixed for reference playback as well. Isolated stems can contain separation artifacts;
the original mix remains the primary listening source.

This is label blinding, not a guarantee that an experienced listener cannot infer which
algorithm produced a contour. The sample deliberately overrepresents difficult passages and
uses songs already involved in tuning. It cannot replace an untouched-song evaluation.

## Answers and reproducibility

Generated files live under ignored `data/listening/`; they must not be committed or published.

- `manifest.json`: neutral case metadata and excerpt words.
- `audio/`: short recording, vocal and anonymous guide WAVs; reference WAVs when available.
- `key.json`: private assignments and sampling provenance. Do not inspect it before reviewing.
- `answers.json`: saved answers, first preferences, reference-exposure history and reveal status.

The server serves an explicit route allowlist; it does not expose the pack directory or key
as static files. Reference audio is gated behind the first saved choice. Writes are atomic,
and stale tabs are rejected rather than silently overwriting newer answers. Downloading answers
before revealing omits the key; downloading after revealing includes it. Keep this as a
single local reviewer session, not a multi-user deployment.

To rebuild a fresh pack from the isolated jobs (never overwrite an existing review):

```sh
uv run python spikes/melody_listening.py build \
  --out data/listening/another-review \
  --revised-dir /path/to/official-2018-stress-files
```

Inputs are `melody.before.json`, `melody.final.json`, existing word timings, audio and stems.
`--jobs`, `--gold`, `--count` and `--seed` can select another local corpus. The optional revised
directory supplies `.str` annotations described in the decoder report. Without it that category
is absent and other categories fill the pack; inspect the printed counts. Building never changes
the source jobs or golden annotations. Assignments differ in a new pack even with the same
selection seed; retain the original key when analysing a completed review.

Validation covers synthesis equivalence for split held notes, silence, selection and clip-edge
overlap, reference gating, hidden assignments, durable first choices, stale writes, export/reveal,
and source preservation. Browser QA uses a separate two-excerpt copy; its dummy answers never
enter the real 36-excerpt session.

After listening, classify the disagreements before retuning: pitch decoding, incorrect word
timing, syllable inference, competing vocal lines, incorrect reference, or unresolved ambiguity.
Then compare the syllable constraint with a weaker continuity prior on untouched songs.
