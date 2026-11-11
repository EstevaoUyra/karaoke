from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from karaoke.config import get_settings
from karaoke.models import Background, BgSource, Scene, StageName, Transition
from karaoke.pipeline import run_job
from karaoke.render.compositor import build_graph, render_background
from karaoke.render.ffmpeg import extract_frame, ffprobe, has_filter, has_libass
from karaoke.stages.render import ENCODES, build_command
from karaoke.store import JobStore

FFMPEG = get_settings().ffmpeg_bin
FFPROBE = get_settings().ffprobe_bin
needs_compositor_ffmpeg = pytest.mark.skipif(
    not (has_filter(FFMPEG, "xfade") and has_filter(FFMPEG, "perspective")),
    reason=f"{FFMPEG} lacks xfade/perspective",
)
needs_libass = pytest.mark.skipif(not has_libass(FFMPEG), reason=f"{FFMPEG} has no libass ('ass' filter)")


def make_test_image(path: Path, size=(640, 400)) -> Path:
    """A grid with a white disc: motion and scaling are easy to see."""
    im = Image.new("RGB", size, (30, 80, 40))
    d = ImageDraw.Draw(im)
    for x in range(0, size[0], 40):
        d.line([(x, 0), (x, size[1])], fill=(220, 220, 60), width=3)
    for y in range(0, size[1], 40):
        d.line([(0, y), (size[0], y)], fill=(220, 60, 60), width=3)
    cx, cy = size[0] // 2, size[1] // 2
    d.ellipse([cx - 50, cy - 50, cx + 50, cy + 50], fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path


def scene(id: str, start: float, end: float, src: BgSource, **kw) -> Scene:
    return Scene(id=id, start=start, end=end, source=src, **kw)


CUT = Transition(type="cut", duration=0)


# --------------------------------------------------------------------------- filtergraph strings


def test_graph_color_scene(tmp_path):
    bg = Background(scenes=[scene("s1", 0, 5, BgSource(type="color", value="#101018"), transition=CUT)])
    g = build_graph(bg, tmp_path, tmp_path, size=(1280, 720))
    assert g.inputs == []
    assert g.filter.startswith("color=c=0x101018:s=1280x720:r=30:")
    assert "trim=end_frame=150" in g.filter and g.filter.endswith("[bg]")
    assert g.frames == 150 and g.duration == 5.0


def test_graph_gradient_scene_makes_png(tmp_path):
    bg = Background(scenes=[scene("s1", 0, 4, BgSource(type="gradient", value="#000000,#ff0000"), transition=CUT)])
    g = build_graph(bg, tmp_path, tmp_path / "w", size=(640, 360))
    assert len(g.inputs) == 1 and g.inputs[0][:2] == ["-loop", "1"]
    png = Path(g.inputs[0][-1])
    im = Image.open(png)
    assert im.size == (640, 360)
    top, bottom = im.getpixel((10, 0)), im.getpixel((10, 359))
    assert top[0] <= 2 and bottom[0] >= 252  # black at the top, red at the bottom
    assert g.filter.startswith("[0:v]fps=30")


def test_graph_image_static_and_motion(tmp_path):
    make_test_image(tmp_path / "assets/bg/a.jpg")
    for motion, expect in [
        ("none", None),
        ("kenburns-in", "perspective=x0=W*(1-1/(1+0.12*on/89))/2:y0=H*(1-1/(1+0.12*on/89))/2:"),
        ("kenburns-out", "x0=W*(1-1/(1+0.12*(1-on/89)))/2"),
        ("pan-left", "x0=W*(1-1/1.15)*(1-on/89):"),
        ("pan-right", "x0=W*(1-1/1.15)*on/89:"),
    ]:
        bg = Background(scenes=[scene("s1", 0, 3, BgSource(type="image", path="assets/bg/a.jpg"),
                                      motion=motion, transition=CUT)])
        g = build_graph(bg, tmp_path, tmp_path / "w", size=(1280, 720))
        assert Image.open(g.inputs[0][-1]).size == (1280, 720)  # cover-cropped once with Pillow
        if expect is None:
            assert "perspective" not in g.filter
        else:
            assert expect in g.filter and "interpolation=cubic:eval=frame" in g.filter


def test_graph_video_dim_blur(tmp_path):
    (tmp_path / "assets/bg").mkdir(parents=True)
    (tmp_path / "assets/bg/v.mp4").write_bytes(b"")
    bg = Background(scenes=[scene("s1", 0, 3, BgSource(type="video", path="assets/bg/v.mp4"),
                                  transition=CUT, dim=0.45, blur=6)])
    g = build_graph(bg, tmp_path, tmp_path / "w", size=(1280, 720))
    assert g.inputs[0][:2] == ["-stream_loop", "-1"]
    assert "scale=1280:720:force_original_aspect_ratio=increase" in g.filter and "crop=1280:720" in g.filter
    assert "gblur=sigma=4.00" in g.filter  # 6 px at 1080p → 4 px at 720p
    assert "colorchannelmixer=rr=0.550:gg=0.550:bb=0.550" in g.filter


def test_graph_missing_image_is_a_clear_error(tmp_path):
    bg = Background(scenes=[scene("s1", 0, 3, BgSource(type="image", path="assets/bg/nope.jpg"))])
    with pytest.raises(FileNotFoundError, match="nope.jpg"):
        build_graph(bg, tmp_path, tmp_path, size=(1280, 720))


def test_graph_transitions_keep_exact_duration(tmp_path):
    bg = Background(scenes=[
        scene("s1", 0, 10, BgSource(type="color", value="#110000")),
        scene("s2", 10, 20, BgSource(type="color", value="#001100"), transition=Transition(type="fade", duration=1.0)),
        scene("s3", 20, 25, BgSource(type="color", value="#000011"), transition=CUT),
        scene("s4", 25, 30.5, BgSource(type="color", value="#111111"), transition=Transition(type="fade", duration=2)),
    ])
    g = build_graph(bg, tmp_path, tmp_path, size=(640, 360), duration=30.0)  # audio is 30 s: last scene clipped
    assert g.frames == 900
    lens = [int(m) for m in re.findall(r"trim=end_frame=(\d+),setpts=PTS-STARTPTS\[c\d\]", g.filter)]
    # fades are centred on the boundaries: s1 ends 0.5 s late, s2 starts 0.5 s early, s3 +1 s, s4 -1 s
    assert lens == [315, 315, 180, 180]
    xf = re.findall(r"xfade=transition=fade:duration=([\d.]+):offset=([\d.]+)", g.filter)
    assert [(float(d), float(o)) for d, o in xf] == [(1.0, 9.5), (2.0, 24.0)]
    assert "[x1][c2]concat=n=2:v=1:a=0,fps=30[x2]" in g.filter
    assert sum(lens) - 30 - 60 == 900


# --------------------------------------------------------------------------- compositor with real ffmpeg


@needs_compositor_ffmpeg
def test_compositor_renders_three_scenes(tmp_path):
    make_test_image(tmp_path / "assets/bg/t.jpg")
    fade = Transition(type="fade", duration=0.5)
    bg = Background(scenes=[
        scene("s1", 0, 2, BgSource(type="color", value="#203050"), transition=CUT),
        scene("s2", 2, 4, BgSource(type="gradient", value="#0b0b1a,#2a1f4a"), transition=fade),
        scene("s3", 4, 6, BgSource(type="image", path="assets/bg/t.jpg"), motion="kenburns-in", transition=fade,
              dim=0.2),
    ])
    out = tmp_path / "bg.mp4"
    g = render_background(bg, tmp_path, out, FFMPEG, size=(640, 360))
    info = ffprobe(out, FFPROBE)
    v = info["streams"][0]
    assert (v["width"], v["height"]) == (640, 360)
    assert abs(float(info["format"]["duration"]) - 6.0) < 0.05
    assert int(v["nb_frames"]) == g.frames == 180
    frames = {}
    for t in (1.0, 3.0, 5.9):
        extract_frame(FFMPEG, out, t, tmp_path / f"f{t}.png")
        frames[t] = Image.open(tmp_path / f"f{t}.png").convert("RGB")
    r, gg, b = frames[1.0].getpixel((320, 180))
    assert abs(r - 0x20) < 12 and abs(gg - 0x30) < 12 and abs(b - 0x50) < 12  # the colour scene
    centre = frames[5.9].getpixel((320, 180))
    assert min(centre) > 170  # the white disc (dimmed 20 %) is in the middle of the image scene


# --------------------------------------------------------------------------- render stage


def test_build_command_mixes_guide_vocals(tmp_path):
    args = build_command(
        bg_inputs=["-loop", "1", "-i", "g.png"], bg_filter="[0:v]null[bg]", frames=720, fps=30,
        ass_name="karaoke.ass", instrumental=tmp_path / "i.wav", vocals=tmp_path / "v.wav", guide_db=-18,
        duration=24.0, encode=ENCODES["final"], out=tmp_path / "o.mp4",
    )
    fc = args[args.index("-filter_complex") + 1]
    assert "[bg]ass=filename=karaoke.ass[v]" in fc
    assert "[2:a]" in fc and "volume=-18.0dB" in fc and "amix=inputs=2" in fc
    assert args[args.index("-crf") + 1] == "22" and args[args.index("-preset") + 1] == "medium"
    assert args[args.index("-b:a") + 1] == "192k"
    no_guide = build_command(
        bg_inputs=[], bg_filter="color=c=black[bg]", frames=720, fps=30, ass_name="karaoke.ass",
        instrumental=tmp_path / "i.wav", vocals=None, guide_db=None, duration=24.0,
        encode=ENCODES["preview"], out=tmp_path / "o.mp4",
    )
    fc = no_guide[no_guide.index("-filter_complex") + 1]
    assert "[0:a]" in fc and "amix" not in fc


def test_render_fails_clearly_without_libass(demo_no_audio, monkeypatch):
    import karaoke.render.ffmpeg as ff

    monkeypatch.setattr(ff, "has_libass", lambda b: False)
    with pytest.raises(RuntimeError, match="libass"):
        run_job("demo", only=[StageName.render], store=JobStore(demo_no_audio.root.parent))


def _region_has(im: Image.Image, box, rgb, tol=40, min_px=30) -> bool:
    crop = im.crop(box)
    n = sum(1 for px in crop.getdata() if all(abs(a - b) <= tol for a, b in zip(px, rgb, strict=False)))
    return n >= min_px


@needs_libass
def test_render_demo_preview(demo, tmp_path):
    store = JobStore(demo.root.parent)
    job = store.load("demo")
    job.settings.render.preset = "preview"
    store.save(job)
    run_job("demo", only=[StageName.render], store=store)

    assert demo.ass.exists() and demo.preview_video.exists()
    info = ffprobe(demo.preview_video, FFPROBE)
    assert abs(float(info["format"]["duration"]) - 24.0) <= 0.1
    v = [s for s in info["streams"] if s["codec_type"] == "video"]
    a = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert len(v) == 1 and (v[0]["width"], v[0]["height"]) == (1280, 720)
    assert len(a) == 1

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = {}
    for t in (1.0, 3.0, 14.5, 18.5):
        png = frames_dir / f"t{t:05.1f}.png"
        extract_frame(FFMPEG, demo.preview_video, t, png)
        frames[t] = Image.open(png).convert("RGB")
    if os.environ.get("KARAOKE_TEST_FRAMES_DIR"):  # keep the frames for a human (or Claude) to look at
        shutil.copytree(frames_dir, os.environ["KARAOKE_TEST_FRAMES_DIR"], dirs_exist_ok=True)

    sung, white = (0x4F, 0xC3, 0xF7), (255, 255, 255)
    top_row = (0, 470, 1280, 540)  # centre y = 0.70 * 720 = 504
    dots = (560, 435, 720, 475)  # countdown dots, above the top row
    # 1.0 s: line 1 is up (lead-in, nothing sung yet)
    assert _region_has(frames[1.0], top_row, white) and not _region_has(frames[1.0], top_row, sung)
    # 3.0 s: mid line 1, partly filled
    assert _region_has(frames[3.0], top_row, sung) and _region_has(frames[3.0], top_row, white)
    # 14.5 s: instrumental gap, countdown dots visible
    assert _region_has(frames[14.5], dots, sung, min_px=20)
    assert not _region_has(frames[14.5], top_row, sung)
    # 18.5 s: second chorus being sung on the top row
    assert _region_has(frames[18.5], top_row, sung)


def test_a_cut_followed_by_a_fade_renders(tmp_path):
    """A hard cut and then a cross-fade: concat's timebase used to break the xfade after it."""
    def sc(i, a, b, colour, kind):
        return Scene(id=f"s{i}", start=a, end=b, source=BgSource(type="color", value=colour),
                     transition=Transition(type=kind, duration=0.0 if kind == "cut" else 0.5))

    bg = Background(fps=10, scenes=[sc(1, 0, 2, "#200000", "cut"), sc(2, 2, 4, "#002000", "cut"),
                                    sc(3, 4, 6, "#000020", "fade"), sc(4, 6, 8, "#202000", "cut"),
                                    sc(5, 8, 10, "#002020", "fade")])
    out = tmp_path / "bg.mp4"
    g = render_background(bg, tmp_path, out, FFMPEG, size=(320, 180))
    assert g.frames == 100 and out.stat().st_size > 1000
