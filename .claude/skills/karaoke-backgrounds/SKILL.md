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

It prints JSON: title, artist, duration, tempo; `recording` (what the source says this recording
is: its full title, who posted it, tags, the start of the description); the **sections** (start,
end, a label, loudness 0..1, and the lines sung in each); `line_gaps` (silences of 1 s or more before, between and after
the sung lines); the **materials**; and the `current` scenes, with which ones are `locked`.

- If `has_structure` is false, run `uv run karaoke run <job> --only structure` first (10–30 s).
- Section labels are a guess from repeated lines and loudness. Trust the loudness and the lines
  more than the word "chorus".
- Read the lines. What is the song about, what does it feel like, where does it turn?

### Read the recording, then propose

The same song can be several different videos. Before any colour or picture, work out what
*this* recording is and what could be on screen because of it. Reason from the brief and from
what you know; there is no list of cases to match.

- **Who is singing here, and why does this version exist?** Read `recording` (title, who posted
  it, tags, description). If it names something you do not know (a person, a place, another
  work), look it up before you go on.
- **What sets this version apart from the song in general?** The arrangement, the occasion, the
  people, where it was heard. Trust what you hear in the brief (loudness, tempo, how much is
  sung) over the song's reputation.
- **What would someone who chose this version expect to see, and what would feel wrong to them?**
  Showing people who are not the ones singing is the commonest way to get it wrong.
- **What can actually be shown?** The materials are free pictures, the cover, the source video and
  generated motion. Some things no free source has. Know which before you promise them.

From this, write down **two or three strategies that are really different**: they differ in what
leads the pictures (for instance the words taken at their word, the world this version belongs
to, the people performing, or mood alone with no pictures), not in details. For each: one
sentence on what the viewer sees, what makes it right for this recording, and what it cannot do
or risks.

**Then ask the user to choose**, with your recommendation first and why. Use the question tool
if you have one, plain text if not. In the same round, ask at most two more things, and only
ones that change the result and that you cannot work out: for example whether they have pictures
of their own to add. Do not ask what the brief already answers.

Do not ask when the user's request already settles the strategy, or when nobody is there to
answer (you were told not to ask, or the run is unattended): take your recommendation and say so
in the report. Either way, the strategies you set aside go in the art direction, one line each,
so the choice can be revisited.

Save a copy of the current background before you change anything: locked scenes are the user's.

```bash
cp data/jobs/<job>/background.json <scratch>/background.before.json
```

## 2. Write the art direction

Write `data/jobs/<job>/art_direction.md`, short (under 30 lines):

- **Reading**: what this recording is, the strategy chosen and who chose it (the user, or you
  unasked), and the strategies set aside, one line each.
- **Concept**: one or two sentences. What the viewer should feel, and the one visual idea.
- **Palette**: 3–5 hex colours and why (the cover's colours in the brief are a starting point; a
  grey or sepia cover deserves colours chosen from the song's mood instead).
- **Pacing**: where the scenes change and why (which section boundaries matter).
- **Per section**: one line each: what is on screen.

Decide before you build, and decide: one clear idea carried through beats a little of everything.
If a picture, a colour or a scene does not serve the concept, it is out, however good it looks.
If the user gave a direction ("warmer", "use the video"), it leads.

## 3. Choose materials

In rough order of how well they work on this machine:

| Material | Source type | Good for | Watch out |
|---|---|---|---|
| Generated motion | `generated`, value `"<style>:<hex>,<hex>,..."` | Any song. Follows loudness, beats and sections by itself. | Pick the colours yourself. Bright, saturated colours are fine: the generator caps the brightness. |
| The cover | `image`, `assets/bg/cover.jpg` | Songs whose cover is a real picture. | Needs `blur` ≥ 20 and `dim` 0.4–0.55. A small cover, or one with large lettering, stays readable as text even then, and blurred further it turns to mush: open it first, and if it is mostly a title, take its colours instead (`cover_palette`). `bg-check` does not catch readable lettering. |
| The source video | `video`, `source.mp4` | Real music videos. | One scene only, covering the whole song (a video scene starts at the file's first frame). If the brief says `source_video_is_still: true`, the video is just the cover: skip `video` and `stills`. A lyric video is no use either (its text fights the lyrics): open two of `assets/bg/still-*.jpg` after running the `stills` preset if unsure. |
| Frames of the video | `image`, from the `stills` preset | Music videos, one frame per section. | Same as the cover. |
| Pictures with a free licence | `image`, via `bg-search` and `bg-image` | The artist on stage, what the song is about (a place, the sea, a city at night), the world around it (a crowd, an instrument, an era). | See "Pictures" below. Six well-chosen ones beat twenty. |
| Gradient / colour | `gradient` `"#top,#bottom"`, `color` | Calm passages, intros. | Keep it dark: top under 20 % brightness, bottom under 50 %. |

A photograph is less saturated than generated colour and gets darker with `dim`, so one picture
among generated scenes looks like another video. Decide in the art direction which kind of
background this is:

- **Picture-led**: pictures carry the song (all of it, or a long stretch such as every verse).
  Give them one treatment (the same `blur`, `dim` within 0.1 of each other) so they read as one
  video, and use generated scenes only for passages with no singing, in colours taken from the
  pictures next to them (`bg-search` reports each candidate's average `color`).
- **Motion-led**: generated scenes carry the song. Use no pictures, or the cover once.

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

`bg-brief` lists the sources that are switched on under `materials.picture_sources`. Wikimedia
Commons always is. Unsplash and Pexels (stock photographs) need a key in `.env`; if they are off,
say so in your report and work with Commons. Commons itself refuses clients that do not say who
runs them: if a search's `notes` say so, `KARAOKE_CONTACT` is missing from `.env`. Tell the user
and go on without pictures; do not try another way in.

**What to look for, and where.** Write down three to six picture ideas, each for a section of the
song, before you search:

| Idea | Source | Queries that work |
|---|---|---|
| **The people performing**: the band on stage, the singer at a microphone. Only when they are who is singing on this recording. | `--source wikimedia` only | The performer's name, then `<name> concert`, `<name> live`, `<name> <year>`. |
| **What the song is about**: a place, the sea, a road, a room, rain | Every source. Commons for places and things with a name (`Pelourinho`, `Copacabana beach`), stock for subjects and moods. | Subject plus light: `empty highway night`, `rain window city`, `beach dusk silhouette`. |
| **The world around it**: a crowd, stage lights, an instrument, a dance floor, the city or decade the song comes from | Stock first. | `concert crowd lights`, `vinyl record close`, `dance hall 1970s`. |

- Stock sites have no real artists: a stock "band" is four strangers. Never show strangers as if
  they were the artist. Pictures of the artist come from Commons, and the title or page must say
  it is them. Many artists have no free picture: then do without.
- Queries are 2 to 4 words, in English (Commons also answers Portuguese for Brazilian subjects).
  Dark pictures are the easy ones behind white lyrics: add `night`, `dusk`, `dark` or `silhouette`.
- Each search costs one request per source, and Unsplash allows 50 an hour: plan the queries, do
  not fish. Six to ten searches are plenty for a song.

**Search and look.**

```bash
uv run karaoke bg-search <job> "<query>" [--source wikimedia] [--n 8] [--name <short-name>]
```

It prints the candidates and draws their thumbnails, numbered, into one picture (`sheet`), each
cropped the way the video will crop it. **Open the sheet.** For each candidate it also gives:
`dim_needed` (the `dim` that brings it under the brightness limit behind the lyrics; above 0.6 the
picture is too bright and would go dull), `detail` (how busy it is where the lyrics go; lower is
easier) and `color` (its average colour). `notes` says which sources gave nothing and why. A
candidate with `error` could not be shown: skip it. Add `--any-shape` only when a portrait
picture is worth losing its top and bottom.

**Choose.** Keep a picture when:

- the lower half is calm (the lyrics sit there): sky, water, a floor, darkness, a blur of lights.
  The subject belongs in the upper half or at the sides;
- it has no lettering that would still read after the blur (a sign filling the frame, a caption);
- it shows no private person you could recognise. For people, prefer backs, silhouettes, hands,
  crowds. The artist on stage is the exception;
- its colour sits with the palette and with the pictures next to it.

**Fetch** the chosen ones by their `pick`. The credit and licence the source gave are recorded for
you, and the command prints the scene's whole `source` object: paste it as it is.

```bash
uv run karaoke bg-image <job> --pick <search-name>:<number>
```

A picture you fetched and then decided against: `uv run karaoke bg-image-rm <job> <file name>`
(it also removes the credit entry).

**Compose with them.**

- Give the parts of the song different jobs and say which in the art direction: for instance
  one kind of picture on the verses, another on the chorus, a third where nobody sings. Or one
  subject seen three ways.
- Search for the things themselves, not for a title: a free source knows `cabin forest`, not the
  name of the work the cabin is in.
- Free sources do not have everything. When the strategy needs a picture they cannot give, use
  the nearest honest thing, say so in the report, and name the scenes where the user's own
  picture ("Upload an image" in the UI) would go.
- Follow the words when they name something you can show ("the bluest skies", "the rain"), and
  put it on screen while it is sung, not a section later.
- A long section can hold two or three pictures of one subject and one colour, changing in the
  gaps between lines (`line_gaps`), each held 8 s or more, with fades of 1.5 to 2 s.
- When a section comes back (the chorus), bring its picture back, with another `motion`.
- Start from `dim` = `dim_needed` + 0.1, never under 0.35, and `blur` 8 to 12; then let
  `bg-check` and the contact sheet decide.

Pictures from anywhere else (a URL the user gives, a file they upload) go through
`bg-image <job> <https url> --credit ... --license ... [--page ...]`. Never fetch pictures from a
search engine or from a page you found yourself.

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
  `source-video`, `wikimedia`, `unsplash`, `pexels`, `upload`.

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
  it between neighbouring scenes. Generated and video scenes use `none`. Every picture should
  move: a still photograph looks like a slide.
- How fast pictures move is one setting for the whole video, `motion_strength` at the top of the
  file (`subtle`, `normal`, `strong`; the speed is the same in a short scene and a long one). It
  is the user's taste, set in the UI: keep the value you find. In a new file use `normal`.
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

Tell the user, briefly: the strategy and who chose it, the concept in one sentence, the scenes (time, what, why) as a short
table, the pictures' credits (from `assets/bg/credits.json`: author, source, licence), anything
you could not do (a source that was off, an artist with no free picture), and that the contact
sheet is at `data/jobs/<job>/output/bg_sheet.jpg`. If any picture is CC BY or CC BY-SA, say that
its credit has to go wherever the video is shared. Say plainly that the video has to be rendered
again to show the new background.
