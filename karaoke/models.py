"""Data contracts for every file in a job folder.

These models are FROZEN after Phase 0: workstreams depend on them. Changes go through the
main session (bump ``version`` fields when the on-disk shape changes).
All times are seconds (float) from the start of ``audio.wav``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def now() -> datetime:
    return datetime.now(UTC)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


# --------------------------------------------------------------------------- job.json


class StageName(StrEnum):
    ingest = "ingest"
    separate = "separate"
    lyrics_fetch = "lyrics_fetch"
    transcribe = "transcribe"
    align = "align"
    melody = "melody"  # sung melody + key for the pitch guide (after align: uses word timings)
    backgrounds = "backgrounds"
    render = "render"


class StageStatus(StrEnum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    skipped = "skipped"  # cached: inputs unchanged since last success


class StageState(_Model):
    status: StageStatus = StageStatus.pending
    input_hash: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_s: float | None = None
    progress: float = 0.0  # 0..1
    message: str = ""
    error: str | None = None


class Quality(StrEnum):
    fast = "fast"
    hq = "hq"


class RenderSettings(_Model):
    preset: Literal["preview", "final"] = "final"
    font: str = "Arial Rounded MT Bold"
    font_size: int = 64  # at 1080p; scaled for other resolutions
    color_sung: str = "#4FC3F7"  # fill colour of sung syllables
    color_unsung: str = "#FFFFFF"
    color_outline: str = "#000000"
    guide_vocals_db: float | None = None  # e.g. -18 mixes vocals back in quietly; None = off
    lead_in_s: float = 1.5  # show a line this long before it starts
    countdown_gap_s: float = 4.0  # show countdown before lines after gaps longer than this
    show_title_card: bool = True
    # pitch guide: the sung melody (melody.json) as bars on a treble staff at the top of the screen
    show_pitch: bool = False
    pitch_lookahead_s: float = 4.0  # notes enter on the right this long before they are sung
    pitch_trail_s: float = 1.0  # ... and stay visible this long left of the playhead
    pitch_min_conf: float = 0.0  # hide notes whose confidence is below this


class JobSettings(_Model):
    quality: Quality = Quality.fast
    separator_model: str | None = None  # None → config default for the quality
    language: Literal["pt", "en", "ja"] | None = None  # None → detect (pt/en only; "ja" must be set)
    lyrics_source: str | None = None  # force a source name (see LyricsSourceName)
    render: RenderSettings = Field(default_factory=RenderSettings)


class Job(_Model):
    version: Literal[1] = 1
    id: str
    url: str | None = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    settings: JobSettings = Field(default_factory=JobSettings)
    stages: dict[StageName, StageState] = Field(
        default_factory=lambda: {s: StageState() for s in StageName}
    )

    @field_validator("stages")
    @classmethod
    def _all_stages(cls, v: dict[StageName, StageState]) -> dict[StageName, StageState]:
        """Jobs created before a stage was added get a pending state for it."""
        return {s: v.get(s, StageState()) for s in StageName}


# --------------------------------------------------------------------------- meta.json


class Meta(_Model):
    version: Literal[1] = 1
    url: str | None = None
    video_id: str | None = None
    title: str  # raw YouTube title
    uploader: str | None = None
    artist: str | None = None  # yt-dlp `artist`/`creator` if present
    track: str | None = None  # yt-dlp `track` if present
    album: str | None = None
    duration: float  # seconds, of audio.wav
    thumbnail: str | None = None  # URL
    youtube_language: str | None = None  # informational only; NOT trusted
    has_manual_subs: list[str] = Field(default_factory=list)  # language codes
    clean_artist: str | None = None  # from titleclean; used for lyrics lookup
    clean_title: str | None = None
    # Qualifiers stripped from the title that help the lyrics search find the right recording,
    # e.g. ["Captain Fantastic"] for "Sweet Child O Mine - Captain fantastic soundtrack Lyrics".
    clean_extra: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- transcript.whisper.json


class TWord(_Model):
    text: str  # as emitted by whisper (may include leading space)
    start: float
    end: float
    prob: float


class TSegment(_Model):
    start: float
    end: float
    text: str
    words: list[TWord] = Field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


class Transcript(_Model):
    version: Literal[1] = 1
    model: str
    language: str
    language_probability: float | None = None
    audio: Literal["vocals", "mix"] = "vocals"
    initial_prompt: str | None = None
    segments: list[TSegment]


# --------------------------------------------------------------------------- lyrics/


class LyricsSourceName(StrEnum):
    user = "user"  # lyrics/sources/user.txt — always wins
    lrclib = "lrclib"  # lrclib.lrc (synced) and/or lrclib.txt (plain)
    genius = "genius"
    youtube = "youtube"  # manual subs only (youtube.vtt)
    whisper = "whisper"  # fallback: transcript text


class SourceInfo(_Model):
    name: LyricsSourceName
    file: str  # relative to job dir
    synced: bool = False
    language: str | None = None
    duration: float | None = None  # duration reported by the source (LRCLIB)
    duration_match: bool | None = None  # |source.duration - meta.duration| <= 2s
    url: str | None = None
    note: str = ""


class LyricsChoice(_Model):
    """lyrics/chosen.json — what lyrics_fetch found and which text alignment should use."""

    version: Literal[1] = 1
    chosen: LyricsSourceName
    language: str | None = None
    use_line_timings: bool = False  # True only for a synced source whose duration matches
    sources: list[SourceInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- lyrics.final.json


class WordOrigin(StrEnum):
    matched = "matched"  # exact match to a whisper word; whisper timing
    fuzzy = "fuzzy"  # fuzzy match; whisper timing
    interpolated = "interpolated"  # no match; timing interpolated between neighbours
    manual = "manual"  # edited by the user in the UI — never overwritten by `align`
    ai = "ai"  # edited by an AI skill — never overwritten by `align`


class Word(_Model):
    text: str  # display text, original casing/punctuation/accents
    start: float
    end: float
    conf: float = 1.0  # 0..1
    origin: WordOrigin = WordOrigin.matched


class Line(_Model):
    id: str  # stable, e.g. "l012"
    section: str | None = None  # "verse", "chorus", "bridge", ... (from source headers)
    singer: str | None = None
    start: float
    end: float
    words: list[Word]
    flags: list[str] = Field(default_factory=list)  # e.g. low_confidence, timing_suspect, ai_edited


class UnmatchedAudio(_Model):
    """Whisper heard words here that match no lyric word (ad-lib? missing repetition?)."""

    start: float
    end: float
    text: str


class LyricsFinal(_Model):
    version: Literal[1] = 1
    language: str | None = None
    text_source: LyricsSourceName
    lines: list[Line]
    unmatched_audio: list[UnmatchedAudio] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=now)


# --------------------------------------------------------------------------- background.json


class BgSource(_Model):
    type: Literal["color", "gradient", "image", "video"]
    value: str | None = None  # color: "#101018"; gradient: "#000000,#203040" (top→bottom)
    path: str | None = None  # image/video: relative to job dir, e.g. "assets/bg/s1.jpg"
    origin: str | None = None  # "upload", "wikimedia", "pexels", "unsplash", "web", "thumbnail", ...
    url: str | None = None  # where it was fetched from
    credit: str | None = None
    license: str | None = None


class Transition(_Model):
    type: Literal["cut", "fade"] = "fade"
    duration: float = 1.0


class Scene(_Model):
    id: str
    start: float
    end: float
    source: BgSource
    motion: Literal["none", "kenburns-in", "kenburns-out", "pan-left", "pan-right"] = "none"
    transition: Transition = Field(default_factory=Transition)  # transition INTO this scene
    dim: float = 0.0  # 0..1 black overlay, keeps lyrics legible
    blur: float = 0.0  # gaussian sigma in px at 1080p
    locked: bool = False  # set by the user; AI skills must not change locked scenes
    reason: str = ""  # why this scene (written by the AI skill)


class Background(_Model):
    version: Literal[1] = 1
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 30
    style: str = "default"
    scenes: list[Scene]


# --------------------------------------------------------------------------- melody.json


class Note(_Model):
    start: float
    end: float
    midi: int  # sounding pitch, MIDI number (60 = C4)
    pitch: float  # median f0 of the note in fractional MIDI (intonation, before rounding)
    conf: float = 1.0  # 0..1: mean pyin voicing probability over the note


class KeyEstimate(_Model):
    tonic: str  # spelled tonic, e.g. "F#", "Bb"
    tonic_pc: int  # pitch class, 0 = C
    mode: Literal["major", "minor"]
    fifths: int  # key signature: >0 sharps, <0 flats
    accidentals: list[str] = Field(default_factory=list)  # e.g. ["F#", "C#"] in signature order
    confidence: float = 0.0  # Krumhansl-Schmuckler correlation of the best key (-1..1)
    margin: float = 0.0  # best correlation minus the runner-up's
    runner_up: str | None = None  # second-best key, e.g. "E minor"
    method: str = "krumhansl-schmuckler"


class Melody(_Model):
    """melody.json: the sung melody of the vocal stem as notes, plus the song's key."""

    version: Literal[1] = 1
    hop_s: float  # pitch-tracker hop the notes were segmented from
    notes: list[Note]
    key: KeyEstimate | None = None
    median_midi: float | None = None
    tuning: float = 0.0  # semitones the singing sits off A440 equal temperament (taken out before rounding)
    # octaves the guide shifts the notes by to fit a treble staff (+1: written an octave above
    # the sounding pitch, shown as "8vb" under the clef; -1: an octave below, "8va" above it)
    display_octave_shift: int = 0
    params: dict[str, float | int | str | bool] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=now)


# --------------------------------------------------------------------------- export


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "job": Job,
    "meta": Meta,
    "transcript": Transcript,
    "lyrics_choice": LyricsChoice,
    "lyrics_final": LyricsFinal,
    "background": Background,
    "melody": Melody,
}
