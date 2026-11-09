"""transcribe stage. Fast tests mock the whisper model (faster_whisper is never imported here);
the slow test runs the real model on the demo job through run_job (child process)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pytest

from karaoke import audio
from karaoke.config import get_settings
from karaoke.models import LyricsChoice, LyricsSourceName, StageName, StageStatus, Transcript
from karaoke.pipeline import StageContext, run_job
from karaoke.stages import transcribe
from karaoke.store import read_model, write_model


@dataclass
class W:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class Seg:
    start: float
    end: float
    text: str
    words: list[W] = field(default_factory=list)
    avg_logprob: float | None = -0.3
    no_speech_prob: float | None = 0.01


def _healthy():
    return [
        Seg(2.0, 4.6, " Morning light", [W(" Morning", 2.0, 2.5, 0.9), W(" light", 2.6, 3.0, 0.8)]),
        Seg(8.0, 9.1, " Sing it", [W(" Sing", 8.0, 8.4, 0.95), W(" it", 8.45, 8.4, 0.7), W(" ", 8.5, 8.5, 0.1)]),
        Seg(9.5, 9.5, " ", []),
        Seg(10.2, 12.4, " nobody's listening anymore", [W(" nobody's", 10.2, 10.9, 0.9),
                                                       W(" listening", 10.95, 11.6, 0.9), W(" anymore", 11.65, 12.4, 0.9)]),
        Seg(17.5, 19.5, " Sing it loud", [W(" Sing", 17.5, 17.9, 0.9), W(" it", 17.95, 18.1, 0.9),
                                          W(" loud", 18.15, 18.6, 0.9)]),
    ]


def _loop(start=2.0, n=60):
    words = [W(" sweet", start + i * 0.1, start + i * 0.1 + 0.1, 0.9) for i in range(n)]
    return [Seg(start, start + n * 0.1, " sweet" * n, words)]


class FakeModel:
    """Both the WhisperModel and (via the patched _pipeline) the batched pipeline.

    ``results`` is consumed one list of segments per transcribe() call; when it runs out, every
    call returns the healthy transcript.
    """

    def __init__(self, lang_probs=None, results=None):
        self.calls: list[dict] = []
        self.detect_calls = 0
        self.lang_probs = lang_probs or [("en", 0.9), ("pt", 0.05)]
        self.results = list(results or [])

    def detect_language(self, audio=None, language_detection_segments=1, **kw):
        self.detect_calls += 1
        return self.lang_probs[0][0], self.lang_probs[0][1], self.lang_probs

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        segs = self.results.pop(0) if self.results else _healthy()
        return iter(segs), None


@pytest.fixture
def fake(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(transcribe, "_load_model", lambda settings: model)
    monkeypatch.setattr(transcribe, "_pipeline", lambda m: m)  # the fake is its own batched pipeline
    monkeypatch.setattr(transcribe, "_decode_audio", lambda path: np.zeros(24 * 16000, dtype=np.float32))
    monkeypatch.delenv("KARAOKE_WHISPER_PROMPT", raising=False)
    return model


def _ctx(demo, store):
    events = []
    c = StageContext(store.load(demo.id), demo, store, get_settings(), StageName.transcribe,
                     logging.getLogger("test.transcribe"), _emit=events.append)
    c.events = events  # type: ignore[attr-defined]
    return c


def test_passes_voice_clips_and_no_vad(demo, store, fake):
    ctx = _ctx(demo, store)
    transcribe.run(ctx)
    (kw,) = fake.calls
    regions = audio.voice_regions(demo.vocals, max_region_s=transcribe.CLIP_MAX_LEN)
    clips = audio.pack_clips(regions, transcribe.CLIP_MAX_LEN, transcribe.CLIP_MAX_GAP)
    assert kw["clip_timestamps"] == transcribe.clip_dicts(clips)
    assert kw["clip_timestamps"][0]["start"] < 2.0 and kw["clip_timestamps"][-1]["end"] > 21.9
    assert all(c["end"] - c["start"] <= transcribe.CLIP_MAX_LEN for c in kw["clip_timestamps"])
    assert kw["vad_filter"] is False
    assert kw["word_timestamps"] is True
    assert kw["condition_on_previous_text"] is False
    assert kw["without_timestamps"] is True
    assert "initial_prompt" not in kw and "hotwords" not in kw  # default prompt mode is "none"
    progress = [e.progress for e in ctx.events]
    assert progress == sorted(progress) and progress[-1] == 1.0


def test_writes_a_valid_transcript(demo, store, fake):
    transcribe.run(_ctx(demo, store))
    t = read_model(demo.transcript, Transcript)
    assert t.audio == "vocals"
    assert t.model == get_settings().whisper_model
    assert t.language == "en" and t.language_probability == 1.0  # from chosen.json
    assert len(t.segments) == 4  # the empty segment is dropped
    words = [w.text for s in t.segments for w in s.words]
    assert words[:4] == [" Morning", " light", " Sing", " it"]  # blank word dropped
    assert len(words) == 10
    it = t.segments[1].words[1]
    assert it.end >= it.start  # end < start from whisper is clamped
    assert t.initial_prompt is None


def test_language_choice(demo, store, fake):
    # job setting wins over chosen.json
    ctx = _ctx(demo, store)
    ctx.job.settings.language = "pt"
    transcribe.run(ctx)
    assert fake.calls[-1]["language"] == "pt" and fake.detect_calls == 0
    # no setting, no chosen.json: detect, restricted to pt/en
    demo.lyrics_choice.unlink()
    fake.lang_probs = [("es", 0.6), ("gl", 0.2), ("pt", 0.15), ("en", 0.05)]
    ctx = _ctx(demo, store)
    ctx.job.settings.language = None
    transcribe.run(ctx)
    assert fake.calls[-1]["language"] == "pt" and fake.detect_calls >= 1
    t = read_model(demo.transcript, Transcript)
    assert t.language == "pt" and t.language_probability == pytest.approx(0.15)


def test_choose_language_helper():
    choice = LyricsChoice(chosen=LyricsSourceName.lrclib, language="pt-BR")
    assert transcribe.choose_language(None, choice) == "pt"
    assert transcribe.choose_language("en", choice) == "en"
    assert transcribe.choose_language(None, LyricsChoice(chosen=LyricsSourceName.whisper)) is None
    assert transcribe.choose_language(None, None) is None
    assert transcribe.pick_language([[("es", 0.5), ("en", 0.3)], [("pt", 0.6), ("en", 0.2)]]) == ("pt", 0.3)


@pytest.mark.parametrize("mode,key", [("initial", "initial_prompt"), ("hotwords", "hotwords")])
def test_prompt_modes(demo, store, fake, monkeypatch, mode, key):
    monkeypatch.setenv("KARAOKE_WHISPER_PROMPT", mode)
    transcribe.run(_ctx(demo, store))
    kw = fake.calls[-1]
    assert kw[key].startswith("Morning light is on the window, coffee")
    assert "[" not in kw[key]  # section headers skipped
    assert read_model(demo.transcript, Transcript).initial_prompt == kw[key]


def test_prompt_mode_without_lyrics(demo, store, fake, monkeypatch):
    monkeypatch.setenv("KARAOKE_WHISPER_PROMPT", "initial")
    demo.lyrics_chosen_text.write_text("[Chorus]\n\n")
    transcribe.run(_ctx(demo, store))
    assert "initial_prompt" not in fake.calls[-1]


def test_bad_prompt_mode(monkeypatch):
    monkeypatch.setenv("KARAOKE_WHISPER_PROMPT", "always")
    with pytest.raises(ValueError):
        transcribe.prompt_mode()


def test_build_prompt():
    text = "[Verse 1]\none two three\n(Chorus)\nfour five\n\nsix"
    assert transcribe.prompt_text(text) == "one two three four five six"
    by_words = lambda s: len(s.split())  # noqa: E731
    assert transcribe.build_prompt(text, max_tokens=4, count_tokens=by_words) == "one two three four"
    assert transcribe.build_prompt(text, max_tokens=100, count_tokens=by_words) == "one two three four five six"
    assert transcribe.build_prompt("[Intro]\n", max_tokens=100) is None
    long = "\n".join(f"word{i} other{i}" for i in range(500))
    p = transcribe.build_prompt(long)  # default ~1.5 tokens per word
    assert 120 <= len(p.split()) <= 134


def test_no_voice_writes_empty_transcript(demo, store, fake, monkeypatch):
    monkeypatch.setattr(audio, "voice_regions", lambda path, **kw: [])
    transcribe.run(_ctx(demo, store))
    assert fake.calls == []
    assert read_model(demo.transcript, Transcript).segments == []


def test_loop_clip_is_redecoded_sequentially(demo, store, fake):
    fake.results = [_loop()]  # the batched pass loops; the retry gets the healthy transcript
    transcribe.run(_ctx(demo, store))
    assert len(fake.calls) == 2
    batched, retry = fake.calls
    assert "batch_size" in batched and "batch_size" not in retry
    assert retry["clip_timestamps"] == [batched["clip_timestamps"][0]["start"], batched["clip_timestamps"][0]["end"]]
    assert retry["vad_filter"] is False and retry["condition_on_previous_text"] is False
    t = read_model(demo.transcript, Transcript)
    assert "sweet" not in " ".join(w.text for s in t.segments for w in s.words)


def test_retry_that_also_loops_keeps_the_less_repetitive(demo, store, fake):
    fake.results = [_loop(n=60), _loop(n=40)]
    transcribe.run(_ctx(demo, store))
    t = read_model(demo.transcript, Transcript)
    assert sum(len(s.words) for s in t.segments) == 40


def test_dropout_clip_is_redecoded(demo, store, fake):
    sparse = [Seg(20.0, 21.0, " anymore", [W(" anymore", 20.0, 21.0, 0.9)])]
    fake.results = [sparse]
    transcribe.run(_ctx(demo, store))
    assert len(fake.calls) == 2
    assert sum(len(s.words) for s in read_model(demo.transcript, Transcript).segments) == 10


def test_clip_helpers():
    clips = [(0.0, 10.0), (20.0, 30.0)]
    a, b, c = Seg(1, 2, " a"), Seg(21, 22, " b"), Seg(14, 14.5, " c")
    assert transcribe.assign_to_clips([a, b, c], clips) == [[a, c], [b]]
    assert transcribe.compression_ratio("sweet " * 50) > transcribe.LOOP_COMPRESSION
    assert transcribe.compression_ratio("She has a smile that seems to me") < transcribe.LOOP_COMPRESSION
    assert transcribe.clip_problem(_loop(), (0, 10)) == "loop"
    assert transcribe.clip_problem([], (0, 20)) == "dropout"
    assert transcribe.clip_problem([], (0, 10)) is None  # too short to judge
    assert transcribe.clip_problem(_healthy(), (0, 20)) is None
    # words only at the end of a long voiced clip: whisper skipped the start
    late = [Seg(18, 20, " x y", [W(" x", 18, 19, 0.9), W(" y", 19, 20, 0.9)])]
    assert transcribe.clip_problem(late, (0, 20), [(0, 20)]) == "dropout"
    assert transcribe.clip_problem(late, (0, 20), [(0, 3), (17, 20)]) is None  # the start is silent
    assert transcribe.wordless_voice(late, (0, 20), [(0, 20)]) == pytest.approx(17.5)


def test_alignment_guard():
    class Base:
        def __init__(self, exc=None):
            self.exc = exc

        def find_alignment(self, tokenizer, text_tokens, encoder_output, num_frames, median_filter_width=7):
            if self.exc:
                raise self.exc
            return "aligned"

    class Guarded(transcribe.AlignmentGuard, Base):
        pass

    assert Guarded().find_alignment(None, [[1], [2]], None, 1) == [[], []]
    assert Guarded().find_alignment(None, [[1]], None, 0) == [[]]
    assert Guarded().find_alignment(None, [[1]], None, 100) == "aligned"
    assert Guarded().find_alignment(None, [[1]], None, [1, 100]) == "aligned"  # batched: passed through
    err = IndexError("boolean index did not match indexed array along axis 0")
    with pytest.raises(IndexError, match="num_frames=50"):
        Guarded(err).find_alignment(None, [[1]], None, 50)


def test_missing_vocals(demo, store, fake):
    demo.vocals.unlink()
    with pytest.raises(FileNotFoundError):
        transcribe.run(_ctx(demo, store))


@pytest.mark.slow
def test_transcribe_real_model_on_demo(demo, store):
    demo.transcript.unlink()
    write_model(demo.lyrics_choice, LyricsChoice(chosen=LyricsSourceName.lrclib, language="en"))
    job = run_job(demo.id, only=[StageName.transcribe], store=store)
    assert job.stages[StageName.transcribe].status == StageStatus.done
    t = read_model(demo.transcript, Transcript)  # the demo vocals are tones: few or no words
    assert t.language == "en" and t.audio == "vocals"
    for s in t.segments:
        assert 0 <= s.start <= s.end <= 24.5
