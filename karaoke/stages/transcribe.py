"""Stage 4 — transcribe. Owner: workstream B.

Input:  ``stems/vocals.wav`` (soft inputs: ``lyrics/chosen.json`` for the language,
``lyrics/chosen.txt`` for ``initial_prompt``).
Output: ``transcript.whisper.json`` (models.Transcript, ``audio="vocals"``).

faster-whisper ``WhisperModel(settings.whisper_model, device="cpu",
compute_type=settings.whisper_compute_type, cpu_threads=settings.whisper_threads)``.
Use ``word_timestamps=True``, ``condition_on_previous_text=False`` and **``vad_filter=False``**.
The Silero VAD dropped half of a song in Phase 0. Compute voice regions from the RMS energy of
vocals.wav instead, and pass them as ``clip_timestamps`` (see docs/benchmarks.md "Transcription").
Acceptance on Zdh2hot8rjU: ≥ 75 % of LRC lyric words matched, in ≤ 2× the VAD-on time (about 40 s).
The language is ``job.settings.language`` or ``chosen.json.language``; otherwise whisper detects it.
``initial_prompt`` is roughly the first 200 tokens of chosen.txt (if any).
Apply the empty-window ``find_alignment`` guard (see the reference in PLAN.md §4).
Report progress as segment.end / duration.
"""

from karaoke.pipeline import StageContext


def run(ctx: StageContext) -> None:
    raise NotImplementedError("transcribe: Phase 1, workstream B")
