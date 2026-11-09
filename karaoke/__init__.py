"""Karaoke video generator."""

import os

# torch (separation) and ctranslate2 (faster-whisper) each bundle Intel's libiomp5 on macOS x86_64.
# Loading both aborts with "OMP: Error #15" unless duplicates are allowed, and even then torch
# DEADLOCKS once both have run OpenMP work in one process. Rule: separation and transcription
# never run in the same process (the pipeline runs heavy stages in child processes).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
