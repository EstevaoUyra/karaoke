---
name: karaoke-lyrics-review
description: Review and correct the synced lyrics of a karaoke job. Finds wrong words, lines nobody sings, missing lines and repeats, and lines or words at the wrong time, by comparing the lyrics with what Whisper heard and with the other lyric sources; applies small explicit edits that the aligner will keep; writes review.md. Use when asked to check, fix or review a job's lyrics or sync, e.g. "/karaoke-lyrics-review <job>".
---

# karaoke-lyrics-review

You are proofreading one karaoke song. People will sing from this screen, so two things must be
true of every line: **the words are the ones actually sung, and they light up when they are sung.**

The argument is a job id (a folder in `data/jobs/`). Work through the CLI (`uv run karaoke ...`).
Never edit `lyrics.final.json` by hand and never commit anything under `data/`. Do not paste the
song's full lyrics into your report: quote only the few words you changed.

## What to believe

| Question | Trust | Because |
|---|---|---|
| Which words? | The lyrics text (and the other sources) | Whisper mishears: "window" → "widow", "anymore" → "any more". A different word in Whisper is a question, not an answer. |
| When? | Whisper's word times | They come from the audio. Synced-lyrics (`.lrc`) timestamps can be a second off, but they are usually off by the same amount from line to line: measure that offset on lines Whisper heard well, and the timestamps become good evidence for the lines it did not hear. |
| Is a line sung at all? | Whisper hearing something there, and the song's shape | A line with nothing heard under it, in an instrumental gap, is usually not sung. |

Songs are Brazilian Portuguese or English. Never trust a language tag: read the words.

## 1. Read the brief

```bash
uv run karaoke validate <job>
uv run karaoke lyrics-brief <job> --compact > <scratch>/brief.txt
uv run karaoke lyrics-brief <job> > <scratch>/brief.json
```

Read `brief.txt` first: one row per line (`!` marks a line with suspects), and under each suspect
line what Whisper heard there with times, the unsure words, and why it is a suspect. Lines starting
with `~` are audio no line accounts for. The JSON has the same content with every word's times,
for when you write an edit. If `validate` reports errors (also listed as `validation` in the
brief), those are the first things to fix.

In the JSON:

- `lines`: each line's id, times, text, flags, its `words` (index `i`, text, start, end, `conf`,
  `origin`) and `whisper_heard` (the words Whisper heard in that time, with their own times and
  probability), plus `agreement` between the two (0..1).
- `unmatched_audio`: words Whisper heard that no line accounts for.
- `suspects`: where to look, in time order. Kinds: `text_differs`, `unsure_words`, `not_heard`,
  `start_off`, `extra_heard`, `overlap`, `unmatched_audio`, `estimate`, and the aligner's flags
  `low_confidence` / `timing_suspect`. A line often has several: judge the line once, with all of
  them in view.
- `other_sources`: the raw text of every lyric source that was found (LRCLIB, Genius, YouTube
  subtitles, the user's own). Compare them when a word is in doubt.

Words with `origin: manual` were fixed by the user. **Never change them**; the tool refuses.

## 2. Judge each suspect

Go through the suspect **lines** in order. For each, decide: fix, leave, or ask. `timing_suspect`
is usually right. About half of `unsure_words` turn out to be words already in the right place
(Whisper simply did not catch them): check their times against the neighbours before moving
anything. Typical cases:

- **`text_differs`.** Read both. If Whisper's version is a mishearing of the lyric (same sounds,
  the lyric makes sense, another source agrees), **leave it**. Change the word only when the lyric
  is clearly wrong: a typo, a word that makes no sense in the line while Whisper's does, or
  another source has Whisper's word. One-word fixes: `replace_word`.
- **`unsure_words`** (low confidence or `interpolated`: the aligner guessed the time). Look at
  `whisper_heard` for that line: if the words are there at other times, retime them
  (`retime_word`, or `set_line` for the whole line using Whisper's times). If one word's text is
  the reason it did not match, see `text_differs`.
- **`start_off` / `timing_suspect`.** If the whole line is offset by the same amount from what was
  heard, `shift_line`. If it is stretched or squeezed, `retime_line` or `set_line`.
- **`not_heard`.** Nothing was heard under the line. If it sits in a gap between sung passages and
  no other evidence says it is sung, `delete_line`. If the passage is just quiet or whispered
  (neighbouring lines are heard fine and the gap fits exactly), leave it and say so.
- **`unmatched_audio`.** Three possibilities: (a) a line or a repeat the lyrics lack (the heard
  words match a line elsewhere in the song, or another source has it): `insert_line` with the
  lyric's spelling and Whisper's times, then `clear_unmatched`; (b) an ad-lib ("yeah", "oh"):
  leave it out, it is not a lyric; (c) noise: leave it.
- **Lines with ids `w001`, `w002`… (flag `from_whisper`).** The lyrics had nothing there, so the
  aligner wrote down what Whisper heard: often half a song the lyrics site lacks, or a spoken
  sample. They are sung or spoken; do not delete them for being "not in the lyrics". Their times
  are Whisper's, their words are Whisper's spelling: when the same words are written elsewhere in
  the song or in another source, `set_line` with that spelling and these times; when a word is a
  plain mishearing of a name or a repeated hook, correct it; otherwise leave the line and list it.
- **Timing you place without hearing it.** When a passage repeats one sung earlier (a last chorus),
  do not copy the earlier line timings by arithmetic: the gaps between repeats drift. Find each
  line's start in the audio itself (the vocal stem's loudness pattern of a heard repeat matches
  the unheard one), or transcribe that stretch of the vocal stem on its own: Whisper times words
  far better on a short clip than inside a whole song.
- **`extra_heard`.** Whisper heard several more words in the line's time than the line has. Either
  something sung there is not in the lyrics (a backing answer, a repeat), or the line was matched
  to the wrong one of several identical phrases. See the next point.
- **The same short phrase many times** (backing answers, a repeated hook). The aligner can put a
  line on the wrong repeat with full confidence. Count: list every time Whisper heard the phrase,
  work out the spacing between repeats (they usually fall on a steady beat), and check that each
  lyric line sits on its own repeat in order. Compare with the same passage in another verse.
- **Lead and backing voices at once.** Words and lines cannot overlap in this file, so two voices
  singing at the same moment cannot both be written at their true times. Keep the lead on its
  times; if the backing line cannot fit before or after without lying about when it is sung, leave
  it as it is and put it under "needs a human ear".
- **The fade-out.** Lines after the last word Whisper heard are often real (the singer keeps
  going as the volume drops). Do not delete them for being unheard. If one runs past the end of
  the audio (`validate` says so), pull its end back inside with `retime_line`.
- **Ad-libs that are in the lyrics.** If the lyric source wrote the ad-lib as a line, keep it: the
  source is the authority on words. Only its timing is yours to fix.
- **Only one lyric source.** Then there is no second opinion on a doubtful word: Whisper against
  the lyric is one against one. Change the word only for an obvious typo; otherwise list it.
- **Repeats.** A chorus sung twice is written twice. If the second time has different words in
  the lyrics than the first, check which one the audio supports before "fixing" either.

When the evidence does not settle it, **do not guess**: leave the line and list it under "needs a
human ear" in the notes. A wrong "correction" is worse than a flagged doubt.

There is one kind of placing that is not a guess: a word Whisper did not hear, whose neighbours on
both sides were heard, can only be in the slot between them. Put it there. Anything looser than
that (several unheard words in a row, a whole unheard line moved on the strength of the `.lrc`
offset or of a parallel verse) is an **estimate**: you may make it when the evidence is good, but
mark the edit `"estimate": true`. The words then keep a low confidence and the line stays on the
suspects list as `estimate`, so a person knows to listen there. Without the mark, an edited line
looks fully verified and disappears from every later check.

## 3. Write the edits

A JSON file with a list of edits, each with a `why`:

```json
[
  {"op": "replace_word", "line": "l002", "word": 1, "text": "cooling", "why": "typo in the source; Whisper and the subtitles have 'cooling'"},
  {"op": "shift_line", "line": "l001", "by": 0.9, "why": "every word is 0.9 s before it is heard"},
  {"op": "retime_word", "line": "l003", "word": 2, "start": 8.65, "end": 9.10, "why": "..."},
  {"op": "retime_line", "line": "l004", "start": 10.2, "end": 12.4, "why": "..."},
  {"op": "set_line", "line": "l005", "words": [{"text": "Sing", "start": 17.5, "end": 17.9}], "why": "..."},
  {"op": "insert_line", "words": [{"text": "nobody's", "start": 19.7, "end": 20.4}], "why": "sung, heard by Whisper, missing from the lyrics"},
  {"op": "delete_line", "line": "l007", "why": "nothing is sung here"},
  {"op": "clear_unmatched", "start": 19.7, "end": 21.9, "why": "now a line"}
]
```

- `word` is the index `i` from the brief. Times are seconds.
- Add `"estimate": true` to any edit whose times you did not take from heard words.
- `shift_line` moves every word by the same amount. `retime_line` stretches or squeezes the
  line's words evenly into the new start and end. `set_line` replaces the line's words outright.
- Edits are applied in the order written and each one names its line by id, so order only matters
  for two edits on the same line. Lines are re-sorted by time at the end.
- `clear_unmatched` removes entries from the unmatched-audio list in a time range: use it after
  `insert_line` turned that audio into a line. Leave ad-libs and noise in the list.
- Keep the lyric's spelling, capitals and punctuation; take only the **times** from Whisper.
- A word Whisper did not hear (dropped) gets a time that fits between its neighbours.
- Words in a line must not overlap, and lines must not overlap each other.
- Make the smallest edit that fixes the problem. Do not rewrite lines that are fine.

## 4. Apply and check

```bash
uv run karaoke lyrics-apply <job> <scratch>/edits.json --dry-run
uv run karaoke lyrics-apply <job> <scratch>/edits.json
uv run karaoke validate <job>
uv run karaoke lyrics-brief <job> > <scratch>/brief.after.json
```

- `lyrics-apply` is all or nothing: if any edit is wrong or the result is invalid it says why and
  saves nothing. Fix the edits file and run it again.
- Changed words become `origin: ai` and the line gets the `ai_edited` flag, so a later `align`
  keeps your work.
- Read the suspects in `brief.after.json`. What is left should be only what you decided to leave.

## 5. Write the notes and report

Write `data/jobs/<job>/review.md`:

- **Changed**: one line per edit: where, what, why (quote only the changed words).
- **Estimates**: edits marked `estimate`, with the time and what the placing rests on.
- **Left as it is**: suspects you checked and kept, with the reason ("Whisper mishearing").
- **Needs a human ear**: what you could not settle, with the time, so the user can listen there.

Then tell the user the same in a few lines, and that the video must be rendered again. If the
melody stage ran before your edits, say that `uv run karaoke run <job> --from melody` will redo
the pitch guide with the new word times.
