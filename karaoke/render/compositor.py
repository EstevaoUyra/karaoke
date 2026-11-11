"""background.json → one ffmpeg filtergraph that outputs the background video as ``[bg]``.

Each scene becomes a clip of an exact number of frames:

- ``color``: a ``color=`` source.
- ``gradient``: a PNG made with Pillow (top → bottom, 2+ stops, lightly dithered), looped.
- ``image``: pre-scaled with Pillow to cover the frame (once, not per frame). Static for motion
  ``none``. Ken Burns / pan move at a fixed speed set by the background's ``motion_strength``,
  whatever the scene's length. They use the ``perspective`` filter (per-frame corner expressions, cubic
  sub-pixel sampling) rather than ``zoompan``: zoompan crops on whole, chroma-aligned pixels and
  visibly jitters even on a 2x pre-upscaled image (measured: per-frame motion std 0.023 vs 0.006,
  with backward steps). perspective costs ~40 % more CPU than zoompan.
- ``video``: looped (``-stream_loop -1``), scaled and cropped to cover (motion works too).
- ``generated``: moving colour that follows the music (``render.generated``), drawn small into the
  workdir for exactly the scene's frames, then scaled up and softened. It reads the job's
  ``structure.json`` if there is one.

Then ``gblur`` (sigma given at 1080p, scaled to the output height) and the darkening: R, G and B
are multiplied by ``1 - dim``, which keeps the colours' saturation.
A ``fade`` transition into scene *i* is an ``xfade`` centred on the scene
boundary: clip *i* starts ``d/2`` early and clip *i-1* ends ``d/2`` late, so every boundary stays
where background.json puts it and the total length stays exact. ``cut`` is ``concat``.
All boundaries are snapped to whole frames first, so the arithmetic is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from karaoke.models import Background, BgSource, Scene

# How fast a still picture moves, per second: the zoom a Ken Burns gains (0.008 = 0.8 %), and the
# share of the picture's width a pan travels. "normal" over a 15 s scene is a 12 % zoom.
MOTION_SPEED = {"subtle": 0.004, "normal": 0.008, "strong": 0.014}
MAX_ZOOM = 0.25  # a long scene stops gaining here: more would show the picture's pixels
MAX_TRAVEL = 0.20


@dataclass
class BgGraph:
    inputs: list[list[str]]  # ffmpeg input args, one list per input, in input-index order
    filter: str  # filter_complex fragment; its output label is ``[label]``
    frames: int
    size: tuple[int, int]
    fps: int
    label: str = "bg"
    files: list[Path] = field(default_factory=list)  # generated files (in the workdir)

    @property
    def input_args(self) -> list[str]:
        return [a for inp in self.inputs for a in inp]

    @property
    def duration(self) -> float:
        return self.frames / self.fps


@dataclass
class _Clip:
    scene: Scene
    lo: int  # first frame (output timeline)
    hi: int  # one past the last frame
    fade_in: int  # xfade frames shared with the previous clip (0 = cut / first)

    @property
    def frames(self) -> int:
        return self.hi - self.lo


# --------------------------------------------------------------------------- helpers


def _ff_color(value: str) -> str:
    v = value.strip()
    return "0x" + v[1:] if v.startswith("#") else v


def _parse_hex(value: str) -> tuple[int, int, int]:
    v = value.strip().lstrip("#")
    if len(v) not in (6, 8):
        raise ValueError(f"not a hex colour: {value!r}")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def make_gradient_png(value: str, size: tuple[int, int], out: Path) -> Path:
    """Vertical gradient through the comma-separated hex stops in ``value`` (top → bottom)."""
    import numpy as np
    from PIL import Image

    stops = [_parse_hex(c) for c in value.split(",") if c.strip()]
    if not stops:
        raise ValueError(f"empty gradient: {value!r}")
    if len(stops) == 1:
        stops = stops * 2
    W, H = size
    y = np.linspace(0.0, 1.0, H)
    pos = np.linspace(0.0, 1.0, len(stops))
    cols = np.stack([np.interp(y, pos, [s[c] for s in stops]) for c in range(3)], axis=1)  # H x 3
    img = np.repeat(cols[:, None, :], W, axis=1)
    rng = np.random.default_rng(0)
    img = img + rng.uniform(-0.5, 0.5, img.shape)  # dither: hides banding in dark gradients
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.round(img), 0, 255).astype("uint8"), "RGB").save(out)
    return out


def prepare_image(src: Path, size: tuple[int, int], out: Path) -> Path:
    """Scale+centre-crop ``src`` to cover exactly ``size`` (EXIF orientation applied) → PNG."""
    from PIL import Image, ImageOps

    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        fitted = ImageOps.fit(im, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    out.parent.mkdir(parents=True, exist_ok=True)
    fitted.save(out)
    return out


def _snap_clips(scenes: list[Scene], fps: int, total: int) -> list[_Clip]:
    starts = [0] + [min(total, max(0, round(s.start * fps))) for s in scenes[1:]]
    ends = starts[1:] + [total]
    keep = [i for i in range(len(scenes)) if ends[i] > starts[i]]
    scenes = [scenes[i] for i in keep]
    starts, ends = [starts[i] for i in keep], [ends[i] for i in keep]
    clips: list[_Clip] = []
    fades = [0] * len(scenes)
    for i in range(1, len(scenes)):
        t = scenes[i].transition
        if t.type == "fade" and t.duration > 0:
            longest = min(ends[i - 1] - starts[i - 1], ends[i] - starts[i])
            fades[i] = max(0, min(round(t.duration * fps), longest))
    for i, sc in enumerate(scenes):
        lo = starts[i] - fades[i] // 2
        nxt = fades[i + 1] if i + 1 < len(scenes) else 0
        hi = ends[i] + (nxt - nxt // 2)
        clips.append(_Clip(sc, lo, hi, fades[i]))
    return clips


def _motion_filter(motion: str, frames: int, fps: int = 30, strength: str = "normal") -> str:
    """A ``perspective`` filter that moves a crop window (left, top, width, height) frame by frame.

    ``W``/``H`` are the frame size and ``on`` the output frame number (perspective's variables).
    The window moves at ``MOTION_SPEED[strength]`` per second, so a short scene moves as fast as
    a long one and simply gets less far.
    """
    n1 = max(frames - 1, 1)
    p = f"on/{n1}"  # 0 → 1 over the clip
    reach = MOTION_SPEED[strength] * frames / fps
    if motion in ("kenburns-in", "kenburns-out"):
        zoom = f"{min(MAX_ZOOM, reach):.4f}"  # zooms 1.0 → 1 + zoom (or back)
        z = f"(1+{zoom}*{p})" if motion == "kenburns-in" else f"(1+{zoom}*(1-{p}))"
        left, top = f"W*(1-1/{z})/2", f"H*(1-1/{z})/2"
        cw, ch = f"W/{z}", f"H/{z}"
    elif motion in ("pan-left", "pan-right"):  # pan-left: the view slides from the right edge to the left
        travel = f"{min(MAX_TRAVEL, reach):.4f}"  # the window is this much narrower than the picture
        frac = f"(1-{p})" if motion == "pan-left" else p
        left, top = f"W*{travel}*{frac}", f"H*{travel}/2"
        cw, ch = f"W*(1-{travel})", f"H*(1-{travel})"
    else:
        raise ValueError(f"unknown motion {motion!r}")
    right, bottom = f"{left}+{cw}", f"{top}+{ch}"
    return (f"perspective=x0={left}:y0={top}:x1={right}:y1={top}:x2={left}:y2={bottom}:"
            f"x3={right}:y3={bottom}:interpolation=cubic:eval=frame")


def _cover(size: tuple[int, int]) -> str:
    W, H = size
    return f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=bicubic,crop={W}:{H}"


def _effects(scene: Scene, size: tuple[int, int]) -> list[str]:
    out = []
    if scene.blur > 0:
        out.append(f"gblur=sigma={scene.blur * size[1] / 1080:.2f}")
    if scene.dim > 0:
        # Scale R, G and B alike: the colours keep their saturation. (A black box drawn over the
        # YUV frame, the first version, lifted the blacks and left pictures grey.)
        k = 1.0 - min(scene.dim, 1.0)
        out.append(f"format=rgb24,colorchannelmixer=rr={k:.3f}:gg={k:.3f}:bb={k:.3f}")
    return out


# --------------------------------------------------------------------------- graph


def build_graph(
    bg: Background,
    root: Path,
    workdir: Path,
    *,
    size: tuple[int, int] | None = None,
    duration: float | None = None,
    fps: int | None = None,
    input_offset: int = 0,
    ffmpeg_bin: str | None = None,
) -> BgGraph:
    """Build the background filtergraph.

    ``root`` is the job folder (scene paths are relative to it); generated PNGs go to ``workdir``.
    ``size``/``fps`` default to background.json's; ``duration`` defaults to the last scene's end.
    The scene timeline is clipped/extended to exactly ``duration``. Input indices start at
    ``input_offset`` (for callers that put other inputs first).
    """
    W, H = size or tuple(bg.resolution)
    W, H = W - W % 2, H - H % 2
    size = (W, H)
    fps = fps or bg.fps
    if not bg.scenes:
        raise ValueError("background.json has no scenes")
    duration = duration if duration is not None else bg.scenes[-1].end
    total = max(1, round(duration * fps))
    scenes = sorted(bg.scenes, key=lambda s: s.start)
    clips = _snap_clips(scenes, fps, total)

    structure = None  # read once, only if a generated scene needs it
    inputs: list[list[str]] = []
    files: list[Path] = []
    chains: list[str] = []
    norm = f"fps={fps},setsar=1,format=yuv420p"

    def add_input(args: list[str]) -> int:
        inputs.append(args)
        return input_offset + len(inputs) - 1

    for i, c in enumerate(clips):
        sc, src = c.scene, c.scene.source
        n = c.frames
        secs = f"{n / fps + 1:.3f}"  # a little slack; trim cuts the exact frame count
        steps: list[str]
        head: str
        if src.type == "color":
            head = f"color=c={_ff_color(_need(src, 'value', sc))}:s={W}x{H}:r={fps}:d={secs}"
            steps = []
        elif src.type == "gradient":
            png = make_gradient_png(_need(src, "value", sc), size, workdir / f"bg_{i:02d}_{sc.id}_gradient.png")
            files.append(png)
            k = add_input(["-loop", "1", "-framerate", str(fps), "-t", secs, "-i", str(png)])
            head, steps = f"[{k}:v]", []
        elif src.type == "image":
            path = _source_path(src, sc, root)
            png = prepare_image(path, size, workdir / f"bg_{i:02d}_{sc.id}.png")
            steps = [] if sc.motion == "none" else [_motion_filter(sc.motion, n, fps, bg.motion_strength)]
            files.append(png)
            k = add_input(["-loop", "1", "-framerate", str(fps), "-t", secs, "-i", str(png)])
            head = f"[{k}:v]"
        elif src.type == "video":
            path = _source_path(src, sc, root)
            k = add_input(["-stream_loop", "-1", "-t", secs, "-i", str(path)])
            head = f"[{k}:v]"
            steps = [f"fps={fps}", _cover(size)]
            if sc.motion != "none":
                steps.append(_motion_filter(sc.motion, n, fps, bg.motion_strength))
            steps.append(f"tpad=stop_mode=clone:stop_duration={secs}")
        elif src.type == "generated":
            from karaoke.render.generated import render_clip

            if structure is None:
                from karaoke.models import Structure
                from karaoke.store import JobPaths, read_model_opt

                structure = read_model_opt(JobPaths(root).structure, Structure) or False
            if ffmpeg_bin is None:
                from karaoke.config import get_settings

                ffmpeg_bin = get_settings().ffmpeg_bin
            clip = render_clip(_need(src, "value", sc), fps, c.lo, c.hi, structure or None, total / fps,
                               workdir / f"bg_{i:02d}_{sc.id}_generated.mp4", ffmpeg_bin)
            files.append(clip)
            k = add_input(["-i", str(clip)])
            head = f"[{k}:v]"
            # bicubic upscale of a soft picture, then a blur that removes the scaling steps
            # ... and fine noise: a dark, smooth field in 8 bits shows its steps as bands otherwise
            steps = [f"scale={W}:{H}:flags=bicubic", f"gblur=sigma={H / 60:.2f}", "noise=alls=3:allf=t",
                     f"tpad=stop_mode=clone:stop_duration={secs}"]
        else:  # pragma: no cover — the model's Literal forbids it
            raise ValueError(f"unknown source type {src.type!r}")
        steps += _effects(sc, size)
        steps += [norm, f"trim=end_frame={n}", "setpts=PTS-STARTPTS"]
        sep = "," if src.type == "color" else ""  # a source filter is chained; an input label is not
        chains.append(f"{head}{sep}{','.join(steps)}[c{i}]")

    acc, acc_frames = "c0", clips[0].frames
    for i in range(1, len(clips)):
        c = clips[i]
        out = f"x{i}"
        if c.fade_in > 0:
            offset = (acc_frames - c.fade_in) / fps
            chains.append(f"[{acc}][c{i}]xfade=transition=fade:duration={c.fade_in / fps:.6f}:"
                          f"offset={offset:.6f}[{out}]")
            acc_frames += c.frames - c.fade_in
        else:
            # concat hands on a 1/1000000 timebase; a later xfade needs both inputs on 1/fps
            chains.append(f"[{acc}][c{i}]concat=n=2:v=1:a=0,fps={fps}[{out}]")
            acc_frames += c.frames
        acc = out
    assert acc_frames == total, (acc_frames, total)
    # Safety net against off-by-one rounding inside xfade: pad with the last frame, cut exactly.
    chains.append(f"[{acc}]tpad=stop_mode=clone:stop_duration=1,trim=end_frame={total},"
                  f"setpts=PTS-STARTPTS,format=yuv420p[bg]")
    return BgGraph(inputs=inputs, filter=";".join(chains), frames=total, size=size, fps=fps, files=files)


def _need(src: BgSource, attr: str, sc: Scene) -> str:
    v = getattr(src, attr)
    if not v:
        raise ValueError(f"scene {sc.id}: {src.type} source needs '{attr}'")
    return v


def _source_path(src: BgSource, sc: Scene, root: Path) -> Path:
    p = Path(_need(src, "path", sc))
    p = p if p.is_absolute() else root / p
    if not p.exists():
        raise FileNotFoundError(f"scene {sc.id}: {src.type} file not found: {p}")
    return p


# --------------------------------------------------------------------------- background-only render


def render_background(
    bg: Background,
    root: Path,
    out: Path,
    ffmpeg_bin: str,
    *,
    size: tuple[int, int] | None = None,
    duration: float | None = None,
    preset: str = "ultrafast",
    crf: int = 28,
    on_progress=None,
    log=None,
) -> BgGraph:
    """Encode just the background (no lyrics, no audio) to ``out``. For previews and tests."""
    import tempfile

    from karaoke.render.ffmpeg import run_ffmpeg

    with tempfile.TemporaryDirectory(prefix="karaoke-bg-") as tmp:
        g = build_graph(bg, root, Path(tmp), size=size, duration=duration, ffmpeg_bin=ffmpeg_bin)
        args = ["-y", *g.input_args, "-filter_complex", g.filter, "-map", f"[{g.label}]",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-frames:v", str(g.frames), "-r", str(g.fps), str(out)]
        run_ffmpeg(ffmpeg_bin, args, on_progress=on_progress, log=log)
    return g
