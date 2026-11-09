"""Pipeline stages. Each module exposes ``run(ctx: karaoke.pipeline.StageContext) -> None``.

Rules for every stage:
- Read inputs / write outputs ONLY via ``ctx.paths`` (see karaoke/store.py) and the models in
  karaoke/models.py (``write_model`` writes atomically).
- Report progress with ``ctx.progress(fraction, message)``; log details with ``ctx.log``.
- Raise on failure; the runner records the error in job.json.
- Heavy imports (torch, faster_whisper, audio_separator, librosa) go INSIDE ``run``.
"""
