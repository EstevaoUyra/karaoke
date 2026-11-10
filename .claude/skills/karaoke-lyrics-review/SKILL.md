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
| When? | Whisper's word times | They come from the audio. Synced-lyrics timestamps are often a second off. |
| Is a line sung at all? | Whisper hearing something there, and the song's shape | A line with nothing heard under it, in an instrumental gap, is usually not sung. |

Songs are Brazilian Portuguese or English. Never trust a language tag: read the words.

## 1. Read the brief

```bash
uv run karaoke lyrics-brief <job> > <scratch>/brief.json
```

- `lines`: each line's id, times, text, flags, its `words` (index `i`, text, start, end, `conf`,
  `origin`) and `whisper_heard` (the words Whisper heard in that time, with their own times and
  probability), plus `agreement` between the two (0..1).
- `unmatched_audio`: words Whisper heard that no line accounts for.
- `suspects`: where to look, in time order. Kinds: `text_differs`, `unsure_words`, `not_heard`,
  `start_off`, `overlap`, `unmatched_audio`, and the aligner's flags `low_confidence` /
  `timing_suspect`.
- `other_sources`: the raw text of every lyric source that was found (LRCLIB, Genius, YouTube
  subtitles, the user's own). Compare them when a word is in doubt.

Words with `origin: manual` were fixed by the user. **Never change them**; the tool refuses.

## 2. Judge each suspect

Go through `suspects` one by one. For each, decide: fix, leave, or ask. Typical cases:

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
- **Repeats.** A chorus sung twice is written twice. If the second time has different words in
  the lyrics than the first, check which one the audio supports before "fixing" either.

When the evidence does not settle it, **do not guess**: leave the line and list it under "needs a
human ear" in the notes. A wrong "correction" is worse than a flagged doubt.

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
- **Left as it is**: suspects you checked and kept, with the reason ("Whisper mishearing").
- **Needs a human ear**: what you could not settle, with the time, so the user can listen there.

Then tell the user the same in a few lines, and that the video must be rendered again. If the
melody stage ran before your edits, say that `uv run karaoke run <job> --from melody` will redo
the pitch guide with the new word times.
