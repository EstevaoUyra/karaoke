"""Stage 2 — separate. Owner: workstream B.

Input:  ``audio.wav``.
Output: ``stems/vocals.wav``, ``stems/instrumental.wav`` (44.1 kHz stereo).

Uses ``audio_separator.separator.Separator`` with
``model_file_dir=ctx.settings.separator_models_dir``. The model is
``job.settings.separator_model`` or, by default, ``settings.separator_model`` (fast) or
``settings.separator_model_hq`` (quality == "hq"). Rename the library's output files to the contract
names. Demucs-type models give 4 stems: instrumental = the sum of drums, bass and other.
"""

from karaoke.pipeline import StageContext


def run(ctx: StageContext) -> None:
    raise NotImplementedError("separate: Phase 1, workstream B")
