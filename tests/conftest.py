from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

import make_demo  # noqa: E402

from karaoke.store import JobPaths, JobStore  # noqa: E402


@pytest.fixture
def jobs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "jobs"
    d.mkdir()
    return d


@pytest.fixture
def store(jobs_dir: Path) -> JobStore:
    return JobStore(jobs_dir)


@pytest.fixture
def demo(jobs_dir: Path) -> JobPaths:
    """A complete synthetic job (all files incl. audio). See tests/fixtures/make_demo.py."""
    return make_demo.build(jobs_dir)


@pytest.fixture
def demo_no_audio(jobs_dir: Path) -> JobPaths:
    return make_demo.build(jobs_dir, audio=False)
