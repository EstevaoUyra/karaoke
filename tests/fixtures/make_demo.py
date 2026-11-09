"""Build the synthetic `demo` job: a 24 s fake song with known ground truth.

    uv run python tests/fixtures/make_demo.py data/jobs     # create data/jobs/demo for UI work

The song (original lyrics, EN) has two chorus repetitions, a 5 s instrumental gap before the
second chorus (countdown case) and an ad-lib in the gap. ``TRUTH`` is the ground-truth word
timing; every other file is derived from it:

- transcript.whisper.json: truth with jitter and planted ASR mistakes (see ``WHISPER_ERRORS``)
- lyrics/sources/lrclib.lrc: line starts from truth; lrclib.txt: the plain text with a
  "[Chorus x2]"-style repeat marker, to test expansion; youtube.vtt: manual subs
- lyrics/chosen.json + chosen.txt: what lyrics_fetch should produce
- lyrics.final.json: what align should produce (equal to the truth)
- background.json: the default single gradient scene
- audio.wav / stems/vocals.wav / stems/instrumental.wav: tones on each word, plus a pad and kick
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from karaoke.models import (  # noqa: E402
    Background,
    BgSource,
    Job,
    Line,
    LyricsChoice,
    LyricsFinal,
    LyricsSourceName,
    Meta,
    Scene,
    SourceInfo,
    Transcript,
    TSegment,
    TWord,
    UnmatchedAudio,
    Word,
)
from karaoke.store import JobPaths, write_model  # noqa: E402

DURATION = 24.0
SR = 44100

# (section, [(word, start, end), ...]) — ground truth
TRUTH: list[tuple[str, list[tuple[str, float, float]]]] = [
    ("verse", [("Morning", 2.00, 2.55), ("light", 2.60, 3.00), ("is", 3.05, 3.20), ("on", 3.25, 3.45),
               ("the", 3.50, 3.65), ("window,", 3.70, 4.60)]),
    ("verse", [("coffee", 4.80, 5.30), ("cooling", 5.35, 5.90), ("by", 5.95, 6.15), ("the", 6.20, 6.35),
               ("door", 6.40, 7.00)]),
    ("chorus", [("Sing", 8.00, 8.40), ("it", 8.45, 8.60), ("loud,", 8.65, 9.10), ("sing", 9.15, 9.50),
                ("it", 9.55, 9.70), ("loud", 9.75, 10.00)]),
    ("chorus", [("nobody's", 10.20, 10.90), ("listening", 10.95, 11.60), ("anymore", 11.65, 12.40)]),
    ("chorus", [("Sing", 17.50, 17.90), ("it", 17.95, 18.10), ("loud,", 18.15, 18.60), ("sing", 18.65, 19.00),
                ("it", 19.05, 19.20), ("loud", 19.25, 19.50)]),
    ("chorus", [("nobody's", 19.70, 20.40), ("listening", 20.45, 21.10), ("anymore", 21.15, 21.90)]),
]

# Planted ASR mistakes: what the fake whisper transcript contains instead of the truth.
WHISPER_ERRORS = {
    (0, 5): ["widow"],  # window → widow (substitution)
    (2, 5): ["out"],  # loud → out (substitution)
    (3, 0): ["nobody"],  # nobody's → nobody (fuzzy)
    (3, 2): ["any", "more"],  # anymore → "any more" (split)
    (4, 0): [],  # first "Sing" of chorus 2 dropped (deletion → must be interpolated)
}
ADLIB = ("yeah", 14.0, 14.6)  # whisper hears an ad-lib in the gap → unmatched_audio

LRC_TEXT_HEADER = "[ar:Demo Band]\n[ti:Sing It Loud]\n[length:00:24.00]\n"


def _line_text(words: list[tuple[str, float, float]]) -> str:
    return " ".join(w for w, _, _ in words)


def truth_lyrics() -> LyricsFinal:
    lines = []
    for i, (section, words) in enumerate(TRUTH, start=1):
        lines.append(Line(
            id=f"l{i:03d}", section=section, start=words[0][1], end=words[-1][2],
            words=[Word(text=w, start=s, end=e, conf=1.0) for w, s, e in words],
        ))
    return LyricsFinal(language="en", text_source=LyricsSourceName.lrclib, lines=lines,
                       unmatched_audio=[UnmatchedAudio(start=ADLIB[1], end=ADLIB[2], text=ADLIB[0])])


def fake_transcript(seed: int = 7) -> Transcript:
    rng = np.random.default_rng(seed)
    segments = []
    for li, (_, words) in enumerate(TRUTH):
        tw: list[TWord] = []
        for wi, (w, s, e) in enumerate(words):
            heard = WHISPER_ERRORS.get((li, wi), [w.strip(",.").lower() if wi else w.strip(",.")])
            if not heard:
                continue
            span = (e - s) / len(heard)
            for k, h in enumerate(heard):
                js, je = rng.normal(0, 0.03, 2)
                tw.append(TWord(text=" " + h, start=round(s + k * span + js, 3),
                                end=round(s + (k + 1) * span + je, 3), prob=round(float(rng.uniform(0.6, 0.99)), 3)))
        segments.append(TSegment(start=tw[0].start, end=tw[-1].end, text="".join(w.text for w in tw), words=tw))
        if li == 3:  # ad-lib after the first chorus
            segments.append(TSegment(start=ADLIB[1], end=ADLIB[2], text=" " + ADLIB[0],
                                     words=[TWord(text=" " + ADLIB[0], start=ADLIB[1], end=ADLIB[2], prob=0.41)]))
    return Transcript(model="fake", language="en", language_probability=0.99, audio="vocals",
                      segments=segments)


def _fmt_lrc(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def _fmt_vtt(t: float) -> str:
    return f"00:{int(t // 60):02d}:{t % 60:06.3f}"


def write_texts(p: JobPaths) -> None:
    p.lyrics_sources_dir.mkdir(parents=True, exist_ok=True)
    lrc = LRC_TEXT_HEADER + "".join(f"[{_fmt_lrc(w[0][1])}]{_line_text(w)}\n" for _, w in TRUTH)
    p.lyrics_source("lrclib.lrc").write_text(lrc)
    # Plain text as lyrics sites often have it: chorus written once, then a repeat marker.
    plain = (
        "[Verse]\n" + _line_text(TRUTH[0][1]) + "\n" + _line_text(TRUTH[1][1]) + "\n\n"
        "[Chorus]\n" + _line_text(TRUTH[2][1]) + "\n" + _line_text(TRUTH[3][1]) + "\n\n"
        "[Chorus x1]\n"
    )
    p.lyrics_source("lrclib.txt").write_text(plain)
    vtt = "WEBVTT\n\n" + "".join(
        f"{_fmt_vtt(w[0][1])} --> {_fmt_vtt(w[-1][2])}\n♪ {_line_text(w)} ♪\n\n" for _, w in TRUTH
    )
    p.lyrics_source("youtube.vtt").write_text(vtt)
    # Canonical text for alignment: expanded, one sung line per line, section headers kept.
    chosen = []
    prev = None
    for section, words in TRUTH:
        if section != prev:
            if chosen:
                chosen.append("")
            chosen.append(f"[{section}]")
            prev = section
        chosen.append(_line_text(words))
    p.lyrics_chosen_text.write_text("\n".join(chosen) + "\n")


def write_audio(p: JobPaths) -> None:
    t = np.arange(int(DURATION * SR)) / SR
    vocals = np.zeros_like(t)
    notes = [220.0, 246.9, 261.6, 293.7, 329.6]
    for li, (_, words) in enumerate(TRUTH):
        for wi, (_, s, e) in enumerate(words):
            idx = (t >= s) & (t < e)
            tt = t[idx] - s
            env = np.minimum(1, tt / 0.03) * np.minimum(1, (e - s - tt) / 0.05)
            f0 = notes[(li + wi) % len(notes)]
            vocals[idx] += env * sum(np.sin(2 * np.pi * f0 * k * tt) / k for k in (1, 2, 3)) * 0.25
    pad = sum(np.sin(2 * np.pi * f * t) for f in (110.0, 138.6, 164.8)) * 0.06
    kick = np.zeros_like(t)
    for beat in np.arange(0, DURATION, 0.5):
        idx = (t >= beat) & (t < beat + 0.12)
        tt = t[idx] - beat
        kick[idx] += np.sin(2 * np.pi * (60 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 25) * 0.5
    instrumental = pad + kick
    p.stems_dir.mkdir(parents=True, exist_ok=True)
    for path, sig in ((p.vocals, vocals), (p.instrumental, instrumental), (p.audio, vocals + instrumental)):
        stereo = np.stack([sig, sig], axis=1).astype(np.float32)
        sf.write(path, np.clip(stereo, -1, 1), SR, subtype="PCM_16")


def build(jobs_dir: Path, job_id: str = "demo", audio: bool = True) -> JobPaths:
    p = JobPaths(Path(jobs_dir) / job_id)
    p.ensure_dirs()
    write_model(p.job, Job(id=job_id, url="https://www.youtube.com/watch?v=DEMO0000000"))
    write_model(p.meta, Meta(
        url="https://www.youtube.com/watch?v=DEMO0000000", video_id="DEMO0000000",
        title="Demo Band - Sing It Loud (Official Video) [HD]", uploader="Demo Band VEVO",
        duration=DURATION, youtube_language="fr", has_manual_subs=["en"],
        clean_artist="Demo Band", clean_title="Sing It Loud",
    ))
    write_texts(p)
    write_model(p.lyrics_choice, LyricsChoice(
        chosen=LyricsSourceName.lrclib, language="en", use_line_timings=True,
        sources=[
            SourceInfo(name=LyricsSourceName.lrclib, file="lyrics/sources/lrclib.lrc", synced=True,
                       language="en", duration=DURATION, duration_match=True),
            SourceInfo(name=LyricsSourceName.youtube, file="lyrics/sources/youtube.vtt", synced=True,
                       language="en"),
        ],
    ))
    write_model(p.transcript, fake_transcript())
    write_model(p.lyrics_final, truth_lyrics())
    write_model(p.background, Background(scenes=[Scene(
        id="s1", start=0, end=DURATION, source=BgSource(type="gradient", value="#0b0b1a,#2a1f4a"),
        transition={"type": "cut", "duration": 0}, reason="default",
    )]))
    if audio:
        write_audio(p)
    return p


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "jobs"
    out = build(target)
    print(json.dumps({"job": str(out.root)}))
