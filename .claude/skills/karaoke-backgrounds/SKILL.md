---
name: karaoke-backgrounds
description: Art-direct the background of a karaoke job. Reads the song (lyrics, sections, loudness), writes a short art direction, composes background.json from the materials at hand (generated motion, the cover, the source video, pictures with a free licence), and checks the result on a contact sheet. Use when asked to make, improve or change a job's background, e.g. "/karaoke-backgrounds <job>" or "make the chorus warmer".
---

# karaoke-backgrounds

You are the art director of one karaoke video. The singer reads lyrics off this screen, so the
background has one job before any other: **the lyrics stay easy to read**. After that: it should
feel like this song, and it should change when the song changes.

The argument is a job id (a folder in `data/jobs/`). Everything goes through the CLI
(`uv run karaoke ...`) and two files in the job folder: `background.json` and `art_direction.md`.
Never commit anything under `data/`.

## 1. Look at the song

```bash
uv run karaoke bg-brief <job>
```

It prints JSON: title, artist, duration, tempo; the **sections** (start, end, a label, loudness
0..1, and the lines sung in each); `line_gaps` (silences of 1 s or more before, between and after
the sung lines); the **materials**; and the `current` scenes, with which ones are `locked`.

- If `has_structure` is false, run `uv run karaoke run <job> --only structure` first (10–30 s).
- Section labels are a guess from repeated lines and loudness. Trust the loudness and the lines
  more than the word "chorus".
- Read the lines. What is the song about, what does it feel like, where does it turn?

Save a copy of the current background before you change anything: locked scenes are the user's.

```bash
cp data/jobs/<job>/background.json <scratch>/background.before.json
```

## 2. Write the art direction

Write `data/jobs/<job>/art_direction.md`, short (under 25 lines):

- **Concept**: one or two sentences. What the viewer should feel, and the one visual idea.
- **Palette**: 3–5 hex colours and why (the cover's colours in the brief are a starting point; a
  grey or sepia cover deserves colours chosen from the song's mood instead).
- **Pacing**: where the scenes change and why (which section boundaries matter).
- **Per section**: one line each: what is on screen.

Decide before you build. If the user gave a direction ("warmer", "use the video"), it leads.

## 3. Choose materials

In rough order of how well they work on this machine:

| Material | Source type | Good for | Watch out |
|---|---|---|---|
| Generated motion | `generated`, value `"<style>:<hex>,<hex>,..."` | Any song. Follows loudness, beats and sections by itself. | Pick the colours yourself. Bright, saturated colours are fine: the generator caps the brightness. |
| The cover | `image`, `assets/bg/cover.jpg` | Songs whose cover is a real picture. | Needs `blur` ≥ 20 and `dim` 0.4–0.55. A small cover, or one with large lettering, stays readable as text even then, and blurred further it turns to mush: open it first, and if it is mostly a title, take its colours instead (`cover_palette`). `bg-check` does not catch readable lettering. |
| The source video | `video`, `source.mp4` | Real music videos. | One scene only, covering the whole song (a video scene starts at the file's first frame). If the brief says `source_video_is_still: true`, the video is just the cover: skip `video` and `stills`. A lyric video is no use either (its text fights the lyrics): open two of `assets/bg/still-*.jpg` after running the `stills` preset if unsure. |
| Frames of the video | `image`, from the `stills` preset | Music videos, one frame per section. | Same as the cover. |
| Pictures with a free licence | `image`, via `bg-image` | A clear subject in the lyrics (a place, the sea, a city at night). | See "Pictures" below. Two or three good ones beat ten. |
| Gradient / colour | `gradient` `"#top,#bottom"`, `color` | Calm passages, intros. | Keep it dark: top under 20 % brightness, bottom under 50 %. |

A photograph is less saturated than generated colour and gets darker with `dim`, so next to
generated scenes it can look like another video. Either give it neighbours in its own colours, or
use pictures for a whole stretch of the song rather than one scene among generated ones.

A generated value takes 2 to 5 colours; they are blended in the order given, round and round, so
put colours next to each other that you want to see meet. `pulse` is the darkest style (a glow on
a dark field) and `aurora` the fullest: at the same loudness a `pulse` scene looks darker than an
`aurora` one. Brightness itself is set by the song's loudness, not by you.

Generated styles: `aurora` (drifting colour bands, the default choice), `pulse` (a glow from below
that breathes with the beat: rhythmic songs, choruses), `drift` (soft lights floating up: ballads,
quiet verses). To start from a preset and then edit: `uv run karaoke background <job> <preset>`
(`gradient`, `palette`, `cover`, `stills`, `video`, `aurora`); add `--replace-locked` **only** if
the user said so.

### Pictures

No API keys exist. Use Wikimedia Commons, which needs none:

1. Search (the `filetype:bitmap` keeps scanned books and PDFs out):
   `https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch=filetype:bitmap+<words>&gsrnamespace=6&gsrlimit=8&prop=imageinfo&iiprop=url|extmetadata|size&iiurlwidth=1920&format=json`
2. Keep only pictures whose licence (`extmetadata.LicenseShortName`) is Public domain, CC0, CC BY
   or CC BY-SA, at least 1600 px wide, landscape, with no recognisable private person and no
   lettering that would still read after the blur (a caption, a date stamp, a sign filling the
   frame; a small number plate in the distance is fine).
3. Look before you keep: download the `thumburl` at a small width into your scratch folder
   (`curl -s -o <scratch>/cand1.jpg "<thumburl with 1920px replaced by 640px>"`) and open it.
4. Fetch the one you chose with the tool, which records the credit. Use the `thumburl` (1920 px)
   without its `?utm_...` query string:

```bash
uv run karaoke bg-image <job> "<thumburl or url>" --credit "<Artist>, <title>" --license "<LicenseShortName>" --page "<descriptionurl>"
```

It prints the path to put in the scene's `source.path`. Copy the same credit and licence into the
scene's `source.credit` / `source.license`, put the description page in `source.url`, and set
`source.origin` to `"wikimedia"`. A picture you fetched and then decided against:
`uv run karaoke bg-image-rm <job> <file name>` (it also removes the credit entry).

Never use pictures from anywhere else without asking the user, and never pictures of the artist
from a search engine.

## 4. Write background.json

Edit `data/jobs/<job>/background.json` (schema: `docs/schema/background.schema.json`).

```json
{
  "version": 1, "resolution": [1920, 1080], "fps": 30, "style": "ai",
  "scenes": [
    {"id": "s1", "start": 0.0, "end": 24.085,
     "source": {"type": "generated", "value": "drift:#1f3a5f,#4f7cac,#c0e0de", "origin": "generated"},
     "motion": "none", "transition": {"type": "cut", "duration": 0.0},
     "dim": 0.0, "blur": 0.0, "locked": false, "reason": "quiet opening: slow lights, cold colours"}
  ]
}
```

A picture scene uses `path`, not `value`:

```json
{"id": "s3", "start": 63.55, "end": 88.81,
 "source": {"type": "image", "path": "assets/bg/img-7db685f9f5.jpg", "origin": "wikimedia",
            "url": "https://commons.wikimedia.org/wiki/File:...", "credit": "<Artist>, <title>", "license": "CC BY-SA 2.5"},
 "motion": "kenburns-in", "transition": {"type": "fade", "duration": 2.0},
 "dim": 0.45, "blur": 10.0, "locked": false, "reason": "the skies of the lyrics"}
```

- `dim` is 0 to 1: how much darker (0.45 leaves 55 % of the brightness; the colours keep their
  strength). `blur` is in pixels at 1080p; 8–12 softens a photograph, 20+ hides lettering.
- `origin` is free text for the record: `generated`, `thumbnail` (the cover), `video-frame`,
  `source-video`, `wikimedia`, `upload`.

Rules:

- Scenes cover `0` to the song's `duration` with no gaps or overlaps; ids are unique.
- **Locked scenes stay exactly as they are**: same id, times, source, everything. Build around them.
- A scene starts at a section boundary from the brief, or inside a `line_gap`. Never in the middle
  of a sung line.
- Hold each scene at least 8 s; most songs want 4–10 scenes, not 20.
- Transition into a scene: `fade` of 1–2 s by default. `cut` only where the loudness jumps between
  two sections (say by 0.3 or more in the brief), which is the nearest thing to a "hit" you can
  see. The first scene is always `cut`.
- `motion` (`kenburns-in`, `kenburns-out`, `pan-left`, `pan-right`) is for still images: alternate
  it between neighbouring scenes. Generated and video scenes use `none`.
- Louder sections may be brighter and more saturated; quiet ones darker and cooler. With generated
  scenes do this with the colours, not with `dim`.
- Every scene gets a `reason`: one short line the user will read in the UI.
- Set `"style": "ai"`.

Two generated scenes with different colours cross-fade cleanly (the motion is continuous in song
time), so "same style, new palette at the chorus" is a strong, cheap move.

## 5. Check, look, fix

```bash
uv run karaoke bg-check <job> --before <scratch>/background.before.json
uv run karaoke bg-sheet <job>
```

- `bg-check` must end with `0 error(s)`. Fix every error. Read each warning: "too bright" or "too
  much detail behind the lyrics" means raise `dim` or `blur` on that scene; "in the middle of a
  sung line" means move the boundary.
- `bg-check` warns when the picture behind the lyrics averages brighter than 0.42 (of 1), has
  highlights above 0.78, or is too detailed. If a photograph cannot get under the brightness limit
  without going dull, it is the wrong picture for a lyric screen: choose a darker one.
- `bg-sheet` writes `output/bg_sheet.jpg`: every scene with the lyrics over it, each at one moment
  near its middle, labelled with the section's guessed name. To compare an experiment without
  overwriting it: `bg-sheet <job> --out <scratch>/try.jpg`. **Open the picture and look at it.** Can you read every lyric line at a glance? Do neighbouring scenes look like
  they belong to one video? Does the loud section look louder?
- Change what is wrong and run both again. Two or three rounds is normal.

For a moving check of one passage, render a preview (about half the song's length in time):
set the job's render preset to `preview` in the UI's Video tab, or tell the user to press Render.
Do not start a full render unless asked.

## 6. Report

Tell the user, briefly: the concept in one sentence, the scenes (time, what, why) as a short
table, where the pictures came from (credits), anything you could not do, and that the contact
sheet is at `data/jobs/<job>/output/bg_sheet.jpg`. Say plainly that the video has to be rendered
again to show the new background.
