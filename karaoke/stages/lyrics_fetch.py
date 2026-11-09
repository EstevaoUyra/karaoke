"""Stage 3 — lyrics_fetch. Owner: workstream C.

Inputs: ``meta.json``, optional ``lyrics/sources/user.txt``, optional ``lyrics/sources/youtube.vtt``.
Outputs:
- ``lyrics/sources/lrclib.lrc`` / ``lrclib.txt`` (if found), ``genius.txt`` (if a token is set and it is found).
- ``lyrics/chosen.json`` (models.LyricsChoice), with every source found and the chosen one.
- ``lyrics/chosen.txt``: the canonical text for alignment. One sung line per line, with blank lines
  between stanzas. Section headers become ``[chorus]``-style lines. ``[Chorus x2]``/"(repeat)"
  markers are EXPANDED.

Priority: ``user`` > ``lrclib`` (synced, then plain) > ``genius`` > ``youtube`` (manual subs) > ``whisper``.
``job.settings.lyrics_source`` forces a source.
- LRCLIB lookup: ``GET https://lrclib.net/api/get`` with artist/title/duration, then fall back to
  ``/api/search``. Send a User-Agent of ``karaoke (https://github.com/EstevaoUyra/karaoke)``.
- ``use_line_timings`` is True only for synced LRC with ``|duration - meta.duration| <= 2 s``.
- Language comes from ``job.settings.language``, then from lingua detection on the text (pt/en only).
  NEVER use ``meta.youtube_language``.
- If nothing is found, choose ``whisper``: write an empty chosen.txt. Alignment then uses the
  transcript text and flags every line for review.
"""

from karaoke.pipeline import StageContext


def run(ctx: StageContext) -> None:
    raise NotImplementedError("lyrics_fetch: Phase 1, workstream C")
