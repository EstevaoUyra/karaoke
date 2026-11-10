"""Art-direction tools: the brief, the checker, the contact sheet and the picture downloader."""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from karaoke import bg_tools
from karaoke.models import Background, BgSource, LyricsFinal, Scene, Section, Structure
from karaoke.render import generated
from karaoke.store import read_model, write_model


def scene(i, a, b, src, **kw):
    return Scene(id=f"s{i}", start=a, end=b, source=src, **kw)


GEN = BgSource(type="generated", value="aurora:#27496d,#f2a65a", origin="generated")


@pytest.fixture
def job(demo_no_audio):
    p = demo_no_audio
    sections = [Section(start=0.0, end=12.0, label="verse", energy=0.3, vocal=0.8),
                Section(start=12.0, end=24.0, label="chorus", energy=0.9, vocal=0.9)]
    write_model(p.structure, Structure(duration=24.0, beats=[float(b) for b in range(1, 24)], energy=[0.5] * 48,
                                       sections=sections))
    return p


def test_brief_has_the_song_its_sections_and_materials(job):
    b = bg_tools.brief(job)
    assert b["duration"] == 24.0 and b["has_structure"]
    assert [s["label"] for s in b["sections"]] == ["verse", "chorus"]
    assert sum(len(s["lines"]) for s in b["sections"]) > 0 and all(isinstance(x, str) for x in b["sections"][0]["lines"])
    assert b["materials"]["generated_styles"] == list(generated.STYLES)
    assert b["materials"]["presets"]["video"]["available"] is False
    assert b["current"]["scenes"][0]["locked"] is False
    json.dumps(b)  # plain data


def test_check_protects_locked_scenes_and_sung_lines(job):
    before = Background(scenes=[scene(1, 0, 14, BgSource(type="color", value="#101018"), locked=True),
                                scene(2, 14, 24, GEN)])  # 14 s: the gap between two lines
    line = read_model(job.lyrics_final, LyricsFinal).lines[0]
    mid = round((line.start + line.end) / 2, 2)
    write_model(job.background, Background(scenes=[
        scene(1, 0, mid, BgSource(type="color", value="#202020"), locked=True), scene(2, mid, 24, GEN)]))
    found = bg_tools.check(job, None, before)
    assert any(f.severity == "error" and "locked" in f.message for f in found)
    assert any("middle of a sung line" in f.message for f in found)
    write_model(job.background, before)
    assert bg_tools.check(job, None, before) == []


def test_add_image_saves_the_picture_and_its_credit(job):
    buf = io.BytesIO()
    Image.new("RGB", (64, 36), (10, 80, 160)).save(buf, "PNG")
    dest = bg_tools.add_image(job, "https://example.invalid/sky.png", "A. Person, Sky", "CC BY 4.0",
                              "https://example.invalid/page", download=lambda url: buf.getvalue())
    assert dest.parent == job.bg_assets_dir and dest.suffix == ".png"
    assert json.loads(job.bg_credits.read_text())[dest.name]["license"] == "CC BY 4.0"
    with pytest.raises(ValueError, match="https"):
        bg_tools.add_image(job, "http://example.invalid/x.png", "x", "x", download=lambda url: b"")
    with pytest.raises(ValueError, match="did not return a picture"):
        bg_tools.add_image(job, "https://example.invalid/x.png", "x", "x", download=lambda url: b"<html>")


def test_legibility_tells_a_dark_soft_picture_from_a_bright_busy_one():
    import numpy as np

    dark = Image.new("RGB", (320, 180), (30, 30, 50))
    rng = np.random.default_rng(0)
    busy = Image.fromarray(rng.integers(120, 255, (180, 320, 3), dtype=np.uint8), "RGB")
    a, b = bg_tools.legibility(dark), bg_tools.legibility(busy)
    assert a["mean"] < bg_tools.MAX_MEAN_LUMA and a["detail"] < bg_tools.MAX_DETAIL * 10
    assert b["mean"] > bg_tools.MAX_MEAN_LUMA and b["detail"] > bg_tools.MAX_DETAIL * 10


def test_every_generated_style_draws_and_stays_dark_enough():
    st = Structure(duration=24.0, beats=[float(b) for b in range(1, 24)], energy=[1.0] * 48, sections=[
        Section(start=0.0, end=24.0, label="chorus", energy=1.0, vocal=1.0)])
    seen = []
    for style in generated.STYLES:
        frame = next(generated.frames(f"{style}:#ff4040,#40a0ff,#ffe060", 10, 100, 101, st, 24.0))
        assert frame.shape == (180, 320, 3)
        leg = bg_tools.legibility(Image.fromarray(frame))
        assert leg["mean"] < bg_tools.MAX_MEAN_LUMA + 0.1, style  # even at full loudness
        seen.append(frame.tobytes())
    assert len(set(seen)) == len(generated.STYLES)


def test_contact_sheet_and_legibility_check_on_real_frames(job, tmp_path):
    from karaoke.config import get_settings
    from karaoke.render.ffmpeg import has_libass

    ffmpeg = get_settings().ffmpeg_bin
    if not has_libass(ffmpeg):
        pytest.skip("no ffmpeg with libass")
    write_model(job.background, Background(scenes=[
        scene(1, 0, 12, GEN), scene(2, 12, 24, BgSource(type="color", value="#f4f4f4"))]))
    info = bg_tools.contact_sheet(job, tmp_path / "sheet.jpg", ffmpeg)
    assert [r["scene"] for r in info] == ["s1", "s2"] and info[1]["section"] == "chorus"
    with Image.open(tmp_path / "sheet.jpg") as im:
        assert im.size == (2 * bg_tools.SHEET_CELL[0], bg_tools.SHEET_CELL[1])
    found = bg_tools.check(job, ffmpeg)
    assert [f.where for f in found if "too bright" in f.message] == ["scene s2"]
