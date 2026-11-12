"""Song structure (sections, loudness) and the instant background presets built on it."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from karaoke import bg_presets, structure
from karaoke.models import Background, BgSource, Line, Meta, Scene, Section, StageName, Structure, Word
from karaoke.pipeline import ORDER, run_job
from karaoke.store import JobStore, read_model, write_model
from karaoke.validate import validate_background


def line(k: int, start: float, end: float, text: str) -> Line:
    words = text.split()
    step = (end - start) / len(words)
    return Line(id=f"l{k:03d}", start=start, end=end,
                words=[Word(text=w, start=start + i * step, end=start + (i + 1) * step) for i, w in enumerate(words)])


def test_repeated_lines_ignore_case_accents_and_close_neighbours():
    lines = [line(0, 0, 2, "Não vou embora"), line(1, 2, 4, "something else here"), line(2, 4, 6, "one more line"),
             line(3, 6, 8, "nao vou embora!"), line(4, 8, 10, "yeah"), line(5, 10, 12, "yeah"), line(6, 12, 14, "yeah")]
    assert structure.repeated_lines(lines) == [True, False, False, True, False, False, False]


def test_long_pauses_between_lines_are_boundaries():
    lines = [line(0, 12, 20, "a b"), line(1, 21, 30, "c d"), line(2, 45, 50, "e f")]
    assert structure.lyric_bounds(lines, 70.0) == [12.0, 30.0, 45.0, 50.0]
    assert structure.snap_to_lines(14.0, lines) == 12.0 and structure.snap_to_lines(29.0, lines) == 30.0
    assert structure.snap_to_lines(35.0, lines) == 35.0  # between lines: left alone


def test_audio_candidates_only_split_where_they_fit():
    lines = [line(0, 10, 100, "a b")]
    strong, weak = (40.0, 9.0), (70.0, 1.0)
    cands = [strong, weak, *[(float(t), 0.5) for t in (3, 5, 7)]]
    # 40 s is inside the line: it moves to the nearer edge (10 s), which is too close to 0 and to
    # nothing else, so it is dropped; the long section is still split by a weaker candidate
    out = structure.pick_bounds([], cands, [], 100.0)
    assert 40.0 in out
    assert all(b - a >= structure.MIN_GAP_S for a, b in zip([0.0, *out], [*out, 100.0], strict=True))
    assert structure.pick_bounds([], [strong], lines, 100.0) == [10.0]


def test_short_sections_merge_and_labels_follow_lyrics_and_loudness():
    energy = [0.1] * 20 + [0.5] * 60 + [0.9] * 60 + [0.1] * 20  # 0.5 s hop: 80 s
    assert structure.drop_short([10.0, 13.0, 40.0], energy, 0.5, 80.0) == [10.0, 40.0]
    lines = [line(0, 10, 20, "first verse line"), line(1, 20, 30, "second verse line"), line(2, 30, 39, "third thing"),
             line(3, 40, 50, "the big chorus"), line(4, 50, 60, "everybody sing along"),
             line(5, 60, 65, "the big chorus"), line(6, 65, 70, "everybody sing along")]
    secs = structure.label_sections([10.0, 40.0, 70.0], lines, energy, 0.5, 80.0)
    assert [s.label for s in secs] == ["intro", "verse", "chorus", "outro"]
    assert secs[2].energy > secs[1].energy > secs[0].energy
    assert secs[0].vocal == 0.0 and secs[2].vocal == 1.0


def test_structure_stage_runs_on_the_demo(demo):
    store = JobStore(demo.root.parent)
    assert ORDER.index(StageName.structure) == ORDER.index(StageName.backgrounds) - 1
    run_job("demo", only=[StageName.structure], store=store)
    st = read_model(demo.structure, Structure)
    assert st.sections[0].start == 0.0 and st.sections[-1].end == 24.0
    assert all(a.end == b.start for a, b in zip(st.sections, st.sections[1:], strict=False))
    assert len(st.energy) in (47, 48) and all(0.0 <= v <= 1.0 for v in st.energy)


# --------------------------------------------------------------------------- presets


def cover_bytes() -> bytes:
    im = Image.new("RGB", (160, 90), "black")  # a 4:3 picture with black bars
    im.paste(Image.new("RGB", (60, 90), (200, 60, 40)), (20, 0))
    im.paste(Image.new("RGB", (60, 90), (30, 90, 180)), (80, 0))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


STRUCT = Structure(duration=24.0, sections=[
    Section(start=0.0, end=10.0, label="verse", energy=0.2, vocal=0.8),
    Section(start=10.0, end=24.0, label="chorus", energy=0.9, vocal=0.9),
])


@pytest.fixture
def job(demo_no_audio):
    p = demo_no_audio
    meta = read_model(p.meta, Meta)
    meta.thumbnail = "https://example.invalid/thumb.jpg"
    write_model(p.meta, meta)
    write_model(p.structure, STRUCT)
    return p


def apply(p, name, **kw):
    return bg_presets.apply_preset(name, p, "ffmpeg", download=lambda url: cover_bytes(), **kw)


def test_palette_follows_the_sections_and_the_cover(job):
    bg = apply(job, "palette")
    assert bg.style == "preset:palette" and [(s.start, s.end) for s in bg.scenes] == [(0.0, 10.0), (10.0, 24.0)]
    assert all(s.source.type == "gradient" for s in bg.scenes)
    assert bg.scenes[0].source.value != bg.scenes[1].source.value
    assert bg.scenes[0].transition.type == "cut" and bg.scenes[1].transition.type == "fade"
    for s in bg.scenes:  # dark enough for white lyrics
        assert all(max(int(c[i:i + 2], 16) for i in (1, 3, 5)) <= 140 for c in s.source.value.split(","))
    assert read_model(job.background, Background) == bg and validate_background(bg, job, 24.0) == []
    with Image.open(job.bg_assets_dir / bg_presets.COVER_NAME) as im:
        assert im.size == (120, 90)  # the black bars are cut off


def test_cover_is_blurred_darkened_and_moves(job):
    bg = apply(job, "cover")
    assert [s.source.path for s in bg.scenes] == ["assets/bg/cover.jpg"] * 2
    assert all(s.blur >= 20 and 0.35 <= s.dim <= 0.6 and s.motion != "none" for s in bg.scenes)
    assert bg.scenes[1].dim < bg.scenes[0].dim  # the loud section is brighter
    assert bg.scenes[0].motion != bg.scenes[1].motion


def test_without_structure_a_preset_is_one_scene(job):
    job.structure.unlink()
    bg = apply(job, "cover")
    assert [(s.start, s.end) for s in bg.scenes] == [(0.0, 24.0)]


def test_locked_scenes_are_kept_unless_confirmed(job):
    write_model(job.background, Background(scenes=[
        Scene(id="mine", start=0, end=24, source=BgSource(type="color", value="#ff0000"), locked=True)]))
    before = job.background.read_bytes()
    with pytest.raises(bg_presets.PresetError, match="locked"):
        apply(job, "gradient")
    assert job.background.read_bytes() == before
    assert apply(job, "gradient", replace_locked=True).style == "default"


def test_video_presets_need_the_video_and_unknown_names_fail(job):
    assert bg_presets.available(job, read_model(job.meta, Meta)) == {
        "gradient": True, "palette": True, "cover": True, "stills": False, "video": False, "aurora": True}
    for name in ("video", "stills"):
        with pytest.raises(bg_presets.PresetError, match="source.mp4"):
            apply(job, name)
    with pytest.raises(bg_presets.PresetError, match="unknown preset"):
        apply(job, "nope")


def test_api_lists_and_applies_presets(job, monkeypatch):
    from fastapi.testclient import TestClient

    from karaoke.api.app import create_app

    monkeypatch.setattr(bg_presets, "_download", lambda url: cover_bytes())
    client = TestClient(create_app(JobStore(job.root.parent)))
    listed = client.get("/api/jobs/demo/background/presets").json()
    assert listed["has_structure"] and [p["name"] for p in listed["presets"]] == list(bg_presets.PRESETS)
    r = client.post("/api/jobs/demo/background/preset", json={"name": "palette"})
    assert r.status_code == 200 and len(r.json()["background"]["scenes"]) == 2
    assert client.post("/api/jobs/demo/background/preset", json={"name": "video"}).status_code == 422
    assert [s["label"] for s in client.get("/api/jobs/demo/structure").json()["sections"]] == ["verse", "chorus"]


# --------------------------------------------------------------------------- generated


def test_generated_frames_follow_loudness_beats_and_sections():
    import numpy as np

    from karaoke.render import generated

    st = Structure(duration=24.0, beats=[float(b) for b in range(1, 24)], energy=[0.0] * 20 + [1.0] * 28,
                   sections=STRUCT.sections)
    quiet, loud = (next(generated.frames("aurora:#ff0000,#0000ff", 10, f, f + 1, st, 24.0)) for f in (30, 200))
    assert quiet.shape == (180, 320, 3) and quiet.dtype == np.uint8
    assert loud.mean() > 1.3 * quiet.mean()  # louder is brighter
    assert loud.max() <= 190  # ... but never bright enough to hide white lyrics
    on_beat, off_beat = (next(generated.frames("aurora", 100, f, f + 1, st, 24.0)) for f in (2000, 2090))
    assert on_beat.mean() > off_beat.mean()
    tl = generated.Timeline(st, 24.0)
    assert tl.shift(2.0) == 0.0 and abs(tl.shift(20.0) - generated.SECTION_SHIFT) < 1e-9
    assert tl.phase(20.0) - tl.phase(19.0) > 2 * (tl.phase(5.0) - tl.phase(4.0))  # and moves faster
    # the same absolute time gives the same picture, whatever the clip
    a = list(generated.frames("aurora", 10, 50, 53, st, 24.0))[2]
    assert np.array_equal(a, next(generated.frames("aurora", 10, 52, 53, st, 24.0)))
    with pytest.raises(ValueError, match="unknown generated style"):
        generated.parse("lava:#000000")


def test_aurora_preset_validates_and_needs_no_cover(job):
    meta = read_model(job.meta, Meta)
    meta.thumbnail = None
    write_model(job.meta, meta)
    bg = apply(job, "aurora")
    assert [s.source.type for s in bg.scenes] == ["generated"] and bg.scenes[0].source.value == "aurora:"
    assert validate_background(bg, job, 24.0) == []
    bad = Background(scenes=[Scene(id="s1", start=0, end=24, source=BgSource(type="generated", value="lava"))])
    assert [i.severity for i in validate_background(bad, job, 24.0)] == ["error"]


def test_generated_scene_renders(job, tmp_path):
    from karaoke.config import get_settings
    from karaoke.render.compositor import render_background
    from karaoke.render.ffmpeg import has_libass

    ffmpeg = get_settings().ffmpeg_bin
    if not has_libass(ffmpeg):
        pytest.skip("no ffmpeg")
    bg = Background(fps=10, scenes=[
        Scene(id="s1", start=0, end=2, source=BgSource(type="color", value="#000000")),
        Scene(id="s2", start=2, end=4, source=BgSource(type="generated", value="aurora:#ff0000,#00ff00")),
    ])
    g = render_background(bg, job.root, tmp_path / "bg.mp4", ffmpeg, size=(320, 180), duration=4.0)
    assert g.frames == 40 and (tmp_path / "bg.mp4").stat().st_size > 1000
