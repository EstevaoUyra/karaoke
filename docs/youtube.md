# Publishing on YouTube

`karaoke publish <job>` puts a finished video (`output/karaoke.mp4`, rendered with the final
preset) on your YouTube channel through the YouTube Data API v3.

## One-time setup (Google Cloud)

1. Go to <https://console.cloud.google.com/> signed in as the account that owns the channel and
   create a project (any name, e.g. "karaoke uploads").
2. **APIs & Services → Library**: enable **YouTube Data API v3**.
3. **APIs & Services → OAuth consent screen** (Google Auth Platform): user type **External**,
   an app name, your email as support and developer contact. Under **Audience → Test users**, add
   the account that owns the channel.
   - While the app is in *Testing*, Google asks you to sign in again every 7 days. Pressing
     **Publish app** (no verification needed for your own use) stops that; the sign-in page then
     shows a "Google hasn't verified this app" warning, which you click through.
4. **Clients → Create client → Desktop app**. Download the JSON and save it as
   `~/.config/karaoke/client_secret.json` (or point `KARAOKE_YOUTUBE_CLIENT_SECRET` at it).
   Never put it in the repo.

The first upload opens a browser: sign in, choose the channel, allow access. The token is saved in
`~/.config/karaoke/youtube-token.json` (`KARAOKE_YOUTUBE_TOKEN`) and reused after that.

## Each song

```bash
uv run karaoke publish <job>              # writes output/youtube.json and output/thumbnail.jpg
# read and edit output/youtube.json: title, description, tags, playlist, privacy
uv run karaoke publish <job> --upload     # uploads, private unless youtube.json says otherwise
# a day later, check the Copyright column in YouTube Studio, then:
uv run karaoke publish <job> --update --privacy public
```

- The draft is never overwritten unless you pass `--redraft` (which keeps the uploaded video's id).
- `--update` sends the title, description, tags and privacy of `youtube.json` to the uploaded
  video. YouTube cannot replace a video's file: after a new render, `--upload --again` uploads a
  second copy and the old one is deleted by hand in Studio.
- `warnings` in `youtube.json` lists what the draft could not settle (guessed chapter names, a
  picture licence that needs a look, the Japanese title). They are not sent to YouTube.

## What the draft contains

- **Title**: `Artist - Song (version) (feat.) | Karaokê (Playback com Letra)`, shortened to fit 100
  characters. Portuguese songs get Portuguese text, the others English.
- **Description**: the original video, chapters from the song's sections (YouTube needs the first
  at 0:00, at least three, each 10 s or longer, or it shows none), the rights line, and the credits
  of every picture on screen: title, author, licence with its link, source page and how it was
  changed (cropped, blurred, darkened, set in motion), as CC BY asks. With a CC BY-SA picture the
  description also says the changed picture is shared under the same licence. Hashtags at the end.
- **Tags**: artist and song with "karaokê/karaoke/playback", and general karaoke terms, up to
  YouTube's 500 characters (a tag with a space counts 2 more, for its quotes).
- **Settings**: category Music (10), not made for kids, no altered or synthetic content (the
  backgrounds are abstract animation and real photographs), the song's language, private, the
  playlist "Karaokê" or "Karaoke" (created private if you have none by that name).
- **Thumbnail**: the title card frame, 1280×720. Custom thumbnails need a phone-verified channel
  (<https://www.youtube.com/verify>); without it the upload still works and YouTube picks a frame.

## Copyright

The backing track is the original recording with the voice removed, and the lyrics belong to their
authors, so Content ID will usually match the video. Most labels then *claim* it (the video stays
up and the rights holder takes any ad money; it may be blocked in some countries), but some block
it or send a takedown, which is a copyright strike. That is why uploads start private: check the
claims in Studio before making a video public.

## Limits

Uploads have their own quota: 100 a day per Google Cloud project (since June 2026). Videos from an
unverified project are no longer locked to private (since 2 November 2026), so `privacy` in
`youtube.json` is what you get.
