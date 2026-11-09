from __future__ import annotations

from karaoke.models import Background, BgSource, Scene, StageName
from karaoke.pipeline import run_job
from karaoke.store import JobStore, read_model, write_model
from karaoke.validate import validate_background


def test_default_background_written_when_missing(demo_no_audio):
    p = demo_no_audio
    p.background.unlink()
    run_job("demo", only=[StageName.backgrounds], store=JobStore(p.root.parent))
    bg = read_model(p.background, Background)
    assert bg.style == "default"
    assert len(bg.scenes) == 1
    s = bg.scenes[0]
    assert (s.start, s.end) == (0.0, 24.0)
    assert s.source.type == "gradient" and s.source.value == "#0b0b1a,#2a1f4a"
    assert s.transition.type == "cut"
    assert validate_background(bg, p, 24.0) == []


def test_existing_background_is_never_touched(demo_no_audio):
    p = demo_no_audio
    custom = Background(style="ai", scenes=[
        Scene(id="mine", start=0, end=24, source=BgSource(type="color", value="#ff0000"), locked=True),
    ])
    write_model(p.background, custom)
    before = p.background.read_bytes()
    run_job("demo", only=[StageName.backgrounds], store=JobStore(p.root.parent))
    assert p.background.read_bytes() == before
