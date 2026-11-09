"""Stage 4 — transcribe. Owner: workstream B.

Input:  ``stems/vocals.wav`` (soft inputs: ``lyrics/chosen.json`` for the language,
``lyrics/chosen.txt`` for the lyrics prompt).
Output: ``transcript.whisper.json`` (models.Transcript, ``audio="vocals"``).

faster-whisper ``WhisperModel(settings.whisper_model, device="cpu",
compute_type=settings.whisper_compute_type, cpu_threads=settings.whisper_threads)`` with
``word_timestamps=True``, ``condition_on_previous_text=False`` and **``vad_filter=False``**.
The Silero VAD dropped half of a song in Phase 0. Voice regions come from the RMS energy of
vocals.wav instead (``karaoke.audio.voice_regions``), split at quiet points and packed into
≤ 30 s clips, and passed as ``clip_timestamps`` (see docs/benchmarks.md "Transcription").

The clips go through faster-whisper's ``BatchedInferencePipeline`` (over the same model, no
timestamp tokens, temperature 0): exactly one encoder pass per clip. On this CPU the turbo encoder
costs ~8 s per 30 s window and dominates. The sequential ``WhisperModel.transcribe`` re-encodes
the tail of every clip after its last word and re-decodes hallucination loops at up to 6
temperatures, which made it 2-4× slower for no accuracy gain. Its temperature fallback is still
useful, so a clip whose batched result looks broken (a repetition loop, or a long voiced stretch
with no word: the batched pass sometimes emits only the last phrase of a dense clip) is re-decoded
sequentially and the better result kept. Measurements: spikes/check_transcribe.py's docstring.

Language: ``job.settings.language``, else ``chosen.json.language``, else whisper's detection on
the voiced audio, restricted to the languages we support (PT-BR and EN).

Lyrics prompt (``KARAOKE_WHISPER_PROMPT`` env var, default ``none``, i.e. opt-in): ``initial``
passes about the first 200 tokens of chosen.txt (section headers skipped) as ``initial_prompt``,
which the batched pipeline applies to EVERY clip; ``hotwords`` passes the same text as hotwords.
On the EN cover the prompt HURT badly (44 % of lyric words matched with ``initial``, 50 % with
``hotwords``, vs 76 % without): whisper dropped or garbled whole clips that did not match the
start of the lyrics.

The model subclass applies the empty-window ``find_alignment`` guard: a residual window of
< 2 mel frames has no alignable word, and CTranslate2's empty alignment crashes NumPy masking.
Progress is segment.end / duration.
"""

from __future__ import annotations

import os
import re
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from karaoke.models import LyricsChoice, Transcript, TSegment, TWord
from karaoke.pipeline import StageContext
from karaoke.store import read_model_opt, write_model

SAMPLE_RATE = 16000
SUPPORTED_LANGUAGES = ("pt", "en")
PROMPT_MAX_TOKENS = 200
PROMPT_MODES = ("none", "initial", "hotwords")
DEFAULT_PROMPT_MODE = "none"
# Whisper windows built from the voice regions (see karaoke.audio.pack_clips).
CLIP_MAX_LEN = 30.0
CLIP_MAX_GAP = 8.0
BATCH_SIZE = 1  # batching does not speed up the CPU encoder; 1 gives per-clip progress
LANG_DETECT_WINDOWS = 3
LANG_DETECT_CONFIDENT = 0.8

_SECTION = re.compile(r"^\s*[\[(][^\])]*[\])]\s*$")


# --------------------------------------------------------------------------- pure helpers


def choose_language(job_language: str | None, choice: LyricsChoice | None) -> str | None:
    """Forced job language, then the fetched lyrics' language; None means detect."""
    if job_language:
        return job_language
    if choice and choice.language:
        lang = choice.language.lower().split("-")[0]
        return lang or None
    return None


def prompt_text(lyrics: str) -> str:
    """Lyric lines joined into one prompt string; section headers like ``[Chorus]`` are skipped."""
    lines = [ln.strip() for ln in lyrics.splitlines()]
    return " ".join(ln for ln in lines if ln and not _SECTION.match(ln))


def build_prompt(lyrics: str, max_tokens: int = PROMPT_MAX_TOKENS,
                 count_tokens: Callable[[str], int] | None = None) -> str | None:
    """About the first ``max_tokens`` tokens of the lyrics, cut at a word boundary.

    ``count_tokens`` is the whisper tokenizer when available; otherwise a word is assumed to be
    about 1.5 tokens (close for EN and PT lyrics with the multilingual tokenizer).
    """
    words = prompt_text(lyrics).split()
    if not words:
        return None
    count = count_tokens or (lambda s: int(len(s.split()) * 1.5 + 0.5))
    if count(" ".join(words)) <= max_tokens:
        return " ".join(words)
    lo, hi = 0, len(words)  # largest prefix that fits
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(" ".join(words[:mid])) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo]) or None


def clip_dicts(clips: list[tuple[float, float]]) -> list[dict[str, float]]:
    """``clip_timestamps`` for faster-whisper's BatchedInferencePipeline (seconds)."""
    return [{"start": round(s, 3), "end": round(e, 3)} for s, e in clips]


def to_transcript(segments: list[Any], *, model: str, language: str, language_probability: float | None,
                  initial_prompt: str | None) -> Transcript:
    out: list[TSegment] = []
    for seg in segments:
        words = []
        for w in seg.words or []:
            if not w.word.strip():
                continue
            start, end = float(w.start), float(w.end)
            words.append(TWord(text=w.word, start=round(start, 3), end=round(max(start, end), 3),
                               prob=round(float(w.probability), 4)))
        if not words and not seg.text.strip():
            continue
        out.append(TSegment(
            start=round(float(seg.start), 3), end=round(max(float(seg.start), float(seg.end)), 3),
            text=seg.text, words=words,
            avg_logprob=None if seg.avg_logprob is None else round(float(seg.avg_logprob), 4),
            no_speech_prob=None if seg.no_speech_prob is None else round(float(seg.no_speech_prob), 4),
        ))
    return Transcript(model=model, language=language, language_probability=language_probability,
                      audio="vocals", initial_prompt=initial_prompt, segments=out)


def pick_language(window_probs: list[list[tuple[str, float]]],
                  allowed: tuple[str, ...] = SUPPORTED_LANGUAGES) -> tuple[str, float]:
    """Best allowed language over several detection windows (mean probability per window)."""
    totals = {lang: 0.0 for lang in allowed}
    for probs in window_probs:
        d = dict(probs)
        for lang in allowed:
            totals[lang] += d.get(lang, 0.0)
    n = max(1, len(window_probs))
    best = max(allowed, key=lambda lang: totals[lang])
    return best, round(totals[best] / n, 4)


# A clip's whole text compressing this well is a hallucination loop. (Whisper's per-window 2.4
# flags legitimately repetitive lyrics, e.g. "sweet, sweet, sweet ..." scores 2.8-3.0; real
# loops of one word scored 20+.)
LOOP_COMPRESSION = 4.0
# Voiced audio this long with no word in it: whisper skipped part of the clip. The batched
# pipeline sometimes emits only the last phrase of a dense clip (PT-BR rap: 19 s skipped). The
# longest legitimate stretch seen was 10 s (sustained, overlapping "sweet"s in the EN cover).
MAX_WORDLESS_VOICE_S = 12.0
WORD_REACH_S = 0.5  # a word "covers" this much voice on each side


# Gap fill: whisper sometimes skips a short sung phrase. In Sacal the 30 s clip limit forced a cut
# into fast rap at 29.9 s (the only near-silence was a breath right before a phrase), and whisper
# dropped the phrase at the start of the next clip, as it often does with words at a clip's very
# start. Moving or overlapping clips only moved the problem (whisper re-decoded other clips worse),
# so gaps are fixed afterwards instead: gaps between words at least GAP_MIN_S long, with at least
# GAP_VOICE_S of loud vocal stem AND at least GAP_MIN_ONSETS syllable onsets in them, are re-decoded
# with GAP_CONTEXT_S of audio on each side; only the new words that fall inside the gap are kept.
# The onsets matter: a loud gap without them is noise or an echo tail, where whisper invents words.
GAP_MIN_S = 0.8
GAP_VOICE_S = 0.5
GAP_LOUD_DB = 30.0  # "loud" = within this of the stem's 95th-percentile level (as audio.ON_DB)
GAP_CONTEXT_S = 1.0
GAP_MAX = 40
GAP_MIN_ONSETS = 2
GAP_MIN_LOGPROB = -1.0
GAP_MAX_NO_SPEECH = 0.6
GAP_MIN_WORD_PROB = 0.2


@dataclass
class FillSegment:
    """Words recovered by the gap fill, shaped like a faster-whisper segment for to_transcript."""
    start: float
    end: float
    text: str
    words: list[Any]
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


def wordless_gaps(spans: list[tuple[float, float]], loud: Any, hop_s: float, onsets: Any,
                  min_gap: float = GAP_MIN_S, min_voice: float = GAP_VOICE_S,
                  min_onsets: int = GAP_MIN_ONSETS) -> list[tuple[float, float]]:
    """Gaps between consecutive words (sorted ``spans``) that have singing in them.

    ``loud`` is a boolean array of "voice is loud" frames, ``hop_s`` apart; ``onsets`` are syllable
    onset times (``karaoke.align.voicing.Envelope.onsets``).
    """
    import numpy as np

    on = np.asarray(onsets, dtype=float)
    out = []
    for (_, e0), (s1, _) in zip(spans, spans[1:], strict=False):
        if s1 - e0 < min_gap:
            continue
        a, b = int((e0 + 0.1) / hop_s), int((s1 - 0.1) / hop_s)
        n_on = int(np.sum((on > e0 + 0.05) & (on < s1 - 0.05)))
        if b > a and float(loud[a:b].sum()) * hop_s >= min_voice and n_on >= min_onsets:
            out.append((round(e0, 3), round(s1, 3)))
    return out


def gap_words(segments: list[Any], gap: tuple[float, float]) -> list[Any]:
    """The words of a gap re-decode that belong in the gap (not the context on either side)."""
    g0, g1 = gap
    keep = []
    for sg in segments:
        if (sg.avg_logprob is not None and sg.avg_logprob < GAP_MIN_LOGPROB) or \
                (sg.no_speech_prob is not None and sg.no_speech_prob > GAP_MAX_NO_SPEECH) or \
                compression_ratio(sg.text.strip()) > 2.4:
            continue
        for w in sg.words or []:
            mid = (w.start + w.end) / 2
            if w.word.strip() and g0 + 0.05 < mid < g1 - 0.05 and w.start >= g0 - 0.15 \
                    and w.probability >= GAP_MIN_WORD_PROB:
                keep.append(w)
    return keep


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def _n_words(segs: list[Any]) -> int:
    return sum(len([w for w in (sg.words or []) if w.word.strip()]) for sg in segs)


def clip_compression(segs: list[Any]) -> float:
    return compression_ratio("".join(sg.text for sg in segs).strip())


def assign_to_clips(segments: list[Any], clips: list[tuple[float, float]]) -> list[list[Any]]:
    """Group segments by the clip containing their midpoint (nearest clip if none does)."""
    out: list[list[Any]] = [[] for _ in clips]
    for sg in segments:
        mid = (sg.start + sg.end) / 2
        best = min(range(len(clips)), key=lambda i: 0.0 if clips[i][0] <= mid <= clips[i][1]
                   else min(abs(mid - clips[i][0]), abs(mid - clips[i][1])))
        out[best].append(sg)
    return out


def wordless_voice(segs: list[Any], clip: tuple[float, float], regions: list[tuple[float, float]]) -> float:
    """Longest stretch of voiced audio inside ``clip`` that no word comes near (seconds of voice)."""
    marks = sorted((w.start - WORD_REACH_S, w.end + WORD_REACH_S)
                   for sg in segs for w in (sg.words or []) if w.word.strip())
    gaps, t = [], clip[0]
    for s0, e0 in marks:
        if s0 > t:
            gaps.append((t, min(s0, clip[1])))
        t = max(t, e0)
    if t < clip[1]:
        gaps.append((t, clip[1]))
    best = 0.0
    for g0, g1 in gaps:
        voiced = sum(max(0.0, min(g1, r1) - max(g0, r0)) for r0, r1 in regions)
        best = max(best, voiced)
    return best


def clip_problem(segs: list[Any], clip: tuple[float, float], regions: list[tuple[float, float]] | None = None
                 ) -> str | None:
    """'loop' or 'dropout' when a clip's batched result looks broken, else None."""
    if clip_compression(segs) > LOOP_COMPRESSION:
        return "loop"
    if wordless_voice(segs, clip, regions if regions is not None else [clip]) > MAX_WORDLESS_VOICE_S:
        return "dropout"
    return None


def better_clip_result(old: list[Any], new: list[Any], clip: tuple[float, float]) -> list[Any]:
    """Prefer a result without a loop; among equals, the one with more words (ties keep ``old``)."""
    old_bad, new_bad = clip_problem(old, clip) == "loop", clip_problem(new, clip) == "loop"
    if old_bad != new_bad:
        return old if new_bad else new
    if old_bad:  # both loop: the less repetitive one
        return new if clip_compression(new) < clip_compression(old) else old
    return new if _n_words(new) > _n_words(old) else old


# --------------------------------------------------------------------------- model


class AlignmentGuard:
    """Mixin for WhisperModel that survives the empty residual window in word alignment.

    Splitting audio into 30 s windows can leave a residual window of one 10 ms frame (clip
    boundaries make this more likely). CTranslate2 returns no alignment for it, and
    faster-whisper's boolean mask over the empty array raises IndexError. Under two frames there
    is no word to align, so the window gets empty word lists. (Same idea as
    exercito/debate-live/whisper_runtime.py.) Kept free of faster_whisper imports so it can be
    tested without loading ctranslate2.
    """

    def find_alignment(self, tokenizer, text_tokens, encoder_output, num_frames, median_filter_width=7):
        if not isinstance(num_frames, (list, tuple)) and num_frames < 2:
            return [[] for _ in text_tokens]
        try:
            return super().find_alignment(tokenizer, text_tokens, encoder_output, num_frames,  # type: ignore[misc]
                                          median_filter_width)
        except IndexError as exc:
            if "boolean index did not match" in str(exc):
                raise IndexError(f"{exc}; alignment num_frames={num_frames}") from exc
            raise


def _guarded_model_class():
    from faster_whisper import WhisperModel

    return type("GuardedWhisperModel", (AlignmentGuard, WhisperModel), {})


def _load_model(settings):
    cls = _guarded_model_class()
    return cls(settings.whisper_model, device="cpu", compute_type=settings.whisper_compute_type,
               cpu_threads=settings.whisper_threads)


def _pipeline(model):
    from faster_whisper import BatchedInferencePipeline

    return BatchedInferencePipeline(model)


def _decode_audio(path) -> Any:
    from faster_whisper import decode_audio

    return decode_audio(str(path), sampling_rate=SAMPLE_RATE)


def _detect_language(model, audio16k, clips: list[tuple[float, float]]) -> tuple[str, float]:
    """Whisper language detection on 30 s windows of voiced audio (one encoder pass each).

    Stops at the first window where the best supported language reaches LANG_DETECT_CONFIDENT;
    otherwise averages up to LANG_DETECT_WINDOWS windows.
    """
    import numpy as np

    pieces = [audio16k[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)] for s, e in clips]
    voiced = np.concatenate(pieces) if pieces else audio16k
    win = 30 * SAMPLE_RATE
    probs = []
    for i in range(LANG_DETECT_WINDOWS):
        chunk = voiced[i * win:(i + 1) * win]
        if len(chunk) < 2 * SAMPLE_RATE:
            break
        _, _, all_probs = model.detect_language(audio=chunk, language_detection_segments=1)
        probs.append(list(all_probs))
        if pick_language(probs)[1] >= LANG_DETECT_CONFIDENT:
            break
    if not probs:
        _, _, all_probs = model.detect_language(audio=audio16k[:win], language_detection_segments=1)
        probs.append(list(all_probs))
    return pick_language(probs)


def prompt_mode() -> str:
    mode = os.environ.get("KARAOKE_WHISPER_PROMPT", DEFAULT_PROMPT_MODE).strip().lower()
    if mode not in PROMPT_MODES:
        raise ValueError(f"KARAOKE_WHISPER_PROMPT must be one of {PROMPT_MODES}, got {mode!r}")
    return mode


# --------------------------------------------------------------------------- stage


def _fill_gaps(ctx: StageContext, model, audio16k, segments: list[Any], common: dict[str, Any],
               dur: float) -> list[Any]:
    """Re-decode sung gaps between words (see ``wordless_gaps``) and add the words found."""
    import numpy as np

    from karaoke import audio

    signal, sr = audio.read_mono(ctx.paths.vocals)
    db = audio.rms_db(signal, sr)
    loud = db >= float(np.percentile(db, 95)) - GAP_LOUD_DB if len(db) else np.zeros(0, bool)
    spans = sorted((float(w.start), float(w.end)) for sg in segments for w in (sg.words or []) if w.word.strip())
    from karaoke.align.voicing import Envelope

    gaps = wordless_gaps(spans, loud, audio.HOP_S, Envelope(db).onsets())[:GAP_MAX]
    if not gaps:
        return segments
    t0 = time.monotonic()
    added = []
    for k, (g0, g1) in enumerate(gaps):
        ctx.progress(0.9 + 0.09 * k / len(gaps), f"re-decoding a sung gap at {g0:.0f}s")
        w0, w1 = max(0.0, g0 - GAP_CONTEXT_S), min(dur, g1 + GAP_CONTEXT_S)
        it, _ = model.transcribe(audio16k, **common, clip_timestamps=[w0, w1])
        words = gap_words(list(it), (g0, g1))
        ctx.log.info("gap %.2f-%.2f s: %d words recovered%s", g0, g1, len(words),
                     f": {' '.join(w.word.strip() for w in words)}" if words else "")
        if words:
            added.append(FillSegment(start=words[0].start, end=words[-1].end,
                                     text="".join(w.word for w in words), words=words))
    ctx.log.info("gap fill: %d gaps re-decoded, %d words added in %.1f s", len(gaps),
                 sum(len(f.words) for f in added), time.monotonic() - t0)
    return sorted(segments + added, key=lambda sg: sg.start)


def run(ctx: StageContext) -> None:
    from karaoke import audio

    p = ctx.paths
    if not p.vocals.exists():
        raise FileNotFoundError(f"{p.rel(p.vocals)} is missing: run the separate stage first")
    dur = audio.duration(p.vocals)

    ctx.progress(0.0, "finding voice regions")
    regions = audio.voice_regions(p.vocals, max_region_s=CLIP_MAX_LEN)
    clips = audio.pack_clips(regions, max_len=CLIP_MAX_LEN, max_gap=CLIP_MAX_GAP)
    ctx.log.info("voice regions: %d covering %.1f s of %.1f s; %d whisper clips covering %.1f s",
                 len(regions), audio.coverage(regions), dur, len(clips), audio.coverage(clips))
    if not clips:
        ctx.log.warning("no voice found in %s; writing an empty transcript", p.rel(p.vocals))

    language = choose_language(ctx.job.settings.language, read_model_opt(p.lyrics_choice, LyricsChoice))

    mode = prompt_mode()
    lyrics = p.lyrics_chosen_text.read_text(encoding="utf-8") if p.lyrics_chosen_text.exists() else ""

    ctx.progress(0.01, f"loading {ctx.settings.whisper_model}")
    model = _load_model(ctx.settings)
    tokenizer = getattr(model, "hf_tokenizer", None)
    count = (lambda s: len(tokenizer.encode(" " + s).ids)) if tokenizer is not None else None
    prompt = build_prompt(lyrics, count_tokens=count) if mode != "none" and lyrics.strip() else None

    audio16k = _decode_audio(p.vocals)
    language_probability: float | None = None
    if language is None and clips:
        ctx.progress(0.02, "detecting language")
        language, language_probability = _detect_language(model, audio16k, clips)
        ctx.log.info("detected language %s (p=%.2f, restricted to %s)", language, language_probability,
                     "/".join(SUPPORTED_LANGUAGES))
    elif language is not None:
        language_probability = 1.0
    language = language or SUPPORTED_LANGUAGES[0]

    segments: list[Any] = []
    if clips:
        common: dict[str, Any] = dict(language=language, task="transcribe", word_timestamps=True,
                                      vad_filter=False, condition_on_previous_text=False)
        if prompt and mode == "initial":
            common["initial_prompt"] = prompt
        elif prompt and mode == "hotwords":
            common["hotwords"] = prompt
        ctx.log.info("transcribing: language=%s prompt=%s (%d chars)", language, mode if prompt else "none",
                     len(prompt or ""))
        t0 = time.monotonic()
        seg_iter, _info = _pipeline(model).transcribe(
            audio16k, **common,
            # Batched default: no timestamp tokens (words still get timestamps from alignment).
            # With timestamp tokens the pass was ~20 % slower and produced more loops; the rare
            # clip it would have saved (a dropout) is caught below and re-decoded.
            without_timestamps=True,
            clip_timestamps=clip_dicts(clips), batch_size=BATCH_SIZE,
        )
        n_words = 0
        for seg in seg_iter:
            segments.append(seg)
            n_words += len(seg.words or [])
            ctx.progress(0.03 + 0.85 * min(1.0, seg.end / dur if dur else 1.0),
                         f"{seg.end:.0f}/{dur:.0f}s, {n_words} words")
        ctx.log.info("batched pass: %d segments, %d words in %.1f s", len(segments), n_words,
                     time.monotonic() - t0)

        # The batched pipeline decodes once at temperature 0, so a clip can come back as a
        # hallucination loop (or nearly empty). Re-run only those clips through the sequential
        # decoder, which has whisper's temperature fallback, and keep the better result.
        per_clip = assign_to_clips(segments, clips)
        bad = [i for i, (clip, segs) in enumerate(zip(clips, per_clip, strict=True))
               if clip_problem(segs, clip, regions)]
        for k, i in enumerate(bad):
            clip, old = clips[i], per_clip[i]
            ctx.progress(0.88 + 0.11 * k / len(bad),
                         f"re-decoding clip {clip[0]:.0f}-{clip[1]:.0f}s ({clip_problem(old, clip, regions)})")
            retry_iter, _ = model.transcribe(audio16k, **common, clip_timestamps=[clip[0], clip[1]])
            new = [sg for sg in retry_iter if clip[0] - 1.0 <= (sg.start + sg.end) / 2 <= clip[1] + 1.0]
            keep = better_clip_result(old, new, clip)
            ctx.log.info("clip %.1f-%.1f s: %s; batched %d words (cr %.1f), retry %d words (cr %.1f) -> %s",
                         clip[0], clip[1], clip_problem(old, clip, regions), _n_words(old), clip_compression(old),
                         _n_words(new), clip_compression(new), "retry" if keep is new else "batched")
            per_clip[i] = keep
        segments = [sg for segs in per_clip for sg in segs]
        ctx.log.info("transcribed %d segments, %d words in %.1f s (%d clips re-decoded)", len(segments),
                     sum(_n_words([sg]) for sg in segments), time.monotonic() - t0, len(bad))
        segments = _fill_gaps(ctx, model, audio16k, segments, common, dur)

    transcript = to_transcript(segments, model=ctx.settings.whisper_model, language=language,
                               language_probability=language_probability, initial_prompt=prompt)
    write_model(p.transcript, transcript)
    ctx.progress(1.0, f"{sum(len(s.words) for s in transcript.segments)} words ({language})")
