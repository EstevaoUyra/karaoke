"""Instant background presets: a whole ``background.json`` from what the job already has.

No AI and no image search: the song's own cover (the YouTube thumbnail), its own video and its
structure (``structure.json``; without it a preset is one scene for the whole song).

- ``gradient``: the plain default.
- ``palette``: gradients in the cover's colours, one per section, brighter where the song is louder.
- ``cover``: the cover, blurred and darkened, with slow motion that changes at each section.
- ``stills``: one frame of the source video per section, softened, with slow motion.
- ``video``: the source video itself, blurred and darkened, in sync with the song.
- ``aurora``: generated moving colour in the cover's colours; faster and brighter where the song
  is louder, swelling on the beat, changing colours at each section (``render.generated``).

Every preset keeps the lyrics legible: images and video are always darkened and blurred.
"""

from __future__ import annotations

import colorsys
import logging
from collections.abc import Callable
from pathlib import Path

from karaoke.models import Background, BgSource, Meta, Scene, Section, Structure, Transition
from karaoke.store import JobPaths

log = logging.getLogger(__name__)

PRESETS: dict[str, str] = {
    "gradient": "Plain dark gradient",
    "palette": "Gradients in the cover's colours, changing with the sections",
    "cover": "The cover, blurred and darkened, moving slowly",
    "stills": "Frames of the original video, one per section",
    "video": "The original video, blurred and darkened",
    "aurora": "Moving colours that follow the music",
}
FADE_S = 1.5
MOTIONS = ("kenburns-in", "pan-left", "kenburns-out", "pan-right")
COVER_NAME = "cover.jpg"
PROXY_NAME = "source-h264.mp4"


class PresetError(ValueError):
    """The preset cannot be made for this job (e.g. no source video)."""


def _sections(structure: Structure | None, duration: float) -> list[Section]:
    """The structure's sections, stretched to cover exactly ``0..duration``."""
    secs = [s.model_copy() for s in structure.sections] if structure else []
    secs = [s for s in secs if s.start < duration - 0.5]
    if not secs:
        return [Section(start=0.0, end=round(duration, 3), label="verse", energy=0.5, vocal=1.0)]
    secs[0].start = 0.0
    secs[-1].end = round(duration, 3)
    return secs


def _scene(k: int, sec: Section, source: BgSource, **kw) -> Scene:
    fade = min(FADE_S, (sec.end - sec.start) / 2)
    return Scene(
        id=f"s{k + 1}", start=sec.start, end=sec.end, source=source,
        transition=Transition(type="fade", duration=round(fade, 3)) if k else Transition(type="cut", duration=0.0),
        reason=f"{sec.label}, loudness {sec.energy:.0%}", **kw,
    )


# --------------------------------------------------------------------------- cover and frames


def decodable(ffmpeg_bin: str, video: Path) -> bool:
    """Can this ffmpeg decode the video's first frame?"""
    import subprocess

    r = subprocess.run([ffmpeg_bin, "-v", "error", "-nostdin", "-i", str(video), "-frames:v", "1", "-f", "null",
                        "-"], capture_output=True, check=False)
    return r.returncode == 0


def usable_video(p: JobPaths, ffmpeg_bin: str) -> Path:
    """The source video in a form the render ffmpeg can decode.

    The minimal render build has no software AV1 decoder. If another ffmpeg on the PATH can read
    the file, it is converted once to H.264 at 720p (``assets/bg/source-h264.mp4``): the presets
    blur the video anyway."""
    import shutil
    import subprocess

    if not p.source_video.exists():
        raise PresetError("no source.mp4: this preset needs the original video")
    proxy = p.bg_assets_dir / PROXY_NAME
    if proxy.exists() and proxy.stat().st_mtime >= p.source_video.stat().st_mtime:
        return proxy
    if decodable(ffmpeg_bin, p.source_video):
        return p.source_video
    other = shutil.which("ffmpeg")
    if not other or Path(other).resolve() == Path(ffmpeg_bin).resolve() or not decodable(other, p.source_video):
        raise PresetError("ffmpeg cannot decode source.mp4 (probably AV1): install an ffmpeg with an AV1 "
                          "decoder on the PATH, or download the video again")
    p.bg_assets_dir.mkdir(parents=True, exist_ok=True)
    part = proxy.with_name(f".{proxy.name}.part.mp4")
    try:
        subprocess.run([other, "-v", "error", "-nostdin", "-y", "-i", str(p.source_video), "-an",
                        "-vf", "scale=-2:'min(720,ih)'", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-pix_fmt", "yuv420p", str(part)], check=True, capture_output=True)
        part.replace(proxy)
    except subprocess.CalledProcessError as exc:
        raise PresetError(f"could not convert source.mp4: {exc.stderr.decode(errors='replace')[-300:]}") from exc
    finally:
        part.unlink(missing_ok=True)
    return proxy


def trim_borders(im):
    """Cut black bars (a 4:3 video in a 16:9 frame) off a PIL image, so they are not stretched
    over the screen. Leaves the image alone unless the bars are clear and the rest is large."""
    w, h = im.size
    box = im.convert("L").point(lambda v: 255 if v > 24 else 0).getbbox()
    if not box:
        return im
    bw, bh = box[2] - box[0], box[3] - box[1]
    if (bw > 0.96 * w and bh > 0.96 * h) or bw * bh < 0.4 * w * h:
        return im
    return im.crop(box)


def grab_frame(ffmpeg_bin: str, video: Path, t: float, out: Path) -> Path:
    """One frame of ``video`` at ``t`` seconds, saved as JPEG."""
    from PIL import Image

    from karaoke.render.ffmpeg import run_ffmpeg

    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_name(f".{out.stem}.png")
    try:
        run_ffmpeg(ffmpeg_bin, ["-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", str(png)])
        with Image.open(png) as im:
            trim_borders(im.convert("RGB")).save(out, quality=90)
    finally:
        png.unlink(missing_ok=True)
    return out


def _download(url: str) -> bytes:
    import httpx

    r = httpx.get(url, timeout=15.0, follow_redirects=True)
    r.raise_for_status()
    return r.content


def ensure_cover(p: JobPaths, meta: Meta, ffmpeg_bin: str,
                 download: Callable[[str], bytes] | None = None) -> tuple[Path, BgSource]:
    """``assets/bg/cover.jpg``: the YouTube thumbnail, or a frame of the source video if the
    thumbnail cannot be fetched. Returns the file and an image source describing it."""
    import io

    from PIL import Image

    dest = p.bg_assets_dir / COVER_NAME
    src = BgSource(type="image", path=p.rel(dest), origin="thumbnail", url=meta.thumbnail,
                   credit="video thumbnail")
    if dest.exists():
        return dest, src
    p.bg_assets_dir.mkdir(parents=True, exist_ok=True)
    if meta.thumbnail:
        try:
            with Image.open(io.BytesIO((download or _download)(meta.thumbnail))) as im:
                trim_borders(im.convert("RGB")).save(dest, quality=92)
            return dest, src
        except Exception as exc:  # noqa: BLE001 — fall back to a video frame
            log.warning("could not fetch the thumbnail (%s); using a video frame", exc)
    if p.source_video.exists():
        grab_frame(ffmpeg_bin, usable_video(p, ffmpeg_bin), meta.duration * 0.3, dest)
        return dest, src.model_copy(update={"origin": "video-frame", "url": None, "credit": "frame of the video"})
    raise PresetError("no cover: the video has no thumbnail and source.mp4 is missing")


def cover_palette(image: Path, n: int = 6) -> list[tuple[float, float, float]]:
    """The cover's main colours as (hue, saturation, share), most colourful and common first."""
    from PIL import Image

    with Image.open(image) as im:
        small = im.convert("RGB").resize((96, 54))
    q = small.quantize(n, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()[: 3 * n]
    counts = sorted(q.getcolors(), reverse=True)
    total = sum(c for c, _ in counts)
    out = []
    for count, idx in counts:
        r, g, b = (v / 255 for v in pal[3 * idx: 3 * idx + 3])
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        out.append((h, s if v > 0.12 else 0.0, count / total))  # near-black has no real hue
    return sorted(out, key=lambda c: -(c[1] * 0.7 + c[2] * 0.3))


def _hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def palette_gradient(palette: list[tuple[float, float, float]], k: int, energy: float) -> str:
    """A dark top → brighter bottom gradient in two of the cover's colours. Louder sections are
    brighter and more saturated; consecutive sections take different colours. The brightness is
    capped so white lyrics always stand out."""
    a = palette[k % len(palette)]
    b = palette[(k + 1) % len(palette)]
    sat = lambda c: 0.35 + 0.5 * c[1]  # noqa: E731 — grey covers still get some colour
    top = _hex(a[0], sat(a), 0.10 + 0.10 * energy)
    bottom = _hex(b[0], sat(b) * (0.8 + 0.2 * energy), 0.26 + 0.24 * energy)
    return f"{top},{bottom}"


def aurora_colors(palette: list[tuple[float, float, float]], n: int = 4) -> list[str]:
    """The cover's hues as full, mid-bright colours (the generator sets the brightness itself).
    Hues too close to one already taken are skipped; a grey cover gets its tint strengthened."""
    hues: list[tuple[float, float]] = []
    for h, s, _ in palette:
        if all(min(abs(h - o), 1 - abs(h - o)) > 0.06 for o, _ in hues):
            hues.append((h, s))
    if len(hues) == 1:  # one hue: add its neighbours so the field still has bands
        h, s = hues[0]
        hues += [((h + 0.08) % 1.0, s), ((h - 0.1) % 1.0, s)]
    return [_hex(h, 0.45 + 0.45 * s, 0.95 if k % 2 == 0 else 0.6) for k, (h, s) in enumerate(hues[:n])]


# --------------------------------------------------------------------------- presets


def build_preset(name: str, p: JobPaths, meta: Meta, structure: Structure | None, ffmpeg_bin: str,
                 download: Callable[[str], bytes] | None = None) -> Background:
    """The ``Background`` of preset ``name`` for this job. Writes images to ``assets/bg`` but not
    ``background.json``."""
    from karaoke.stages.backgrounds import default_background

    if name not in PRESETS:
        raise PresetError(f"unknown preset {name!r}: choose one of {', '.join(PRESETS)}")
    duration = meta.duration
    secs = _sections(structure, duration)
    if name == "gradient":
        bg = default_background(duration)
    elif name == "palette":
        cover, _ = ensure_cover(p, meta, ffmpeg_bin, download)
        pal = cover_palette(cover)
        bg = Background(scenes=[
            _scene(k, s, BgSource(type="gradient", value=palette_gradient(pal, k, s.energy), origin="thumbnail"))
            for k, s in enumerate(secs)
        ])
    elif name == "cover":
        _, src = ensure_cover(p, meta, ffmpeg_bin, download)
        bg = Background(scenes=[
            _scene(k, s, src.model_copy(), motion=MOTIONS[k % len(MOTIONS)], blur=24.0,
                   dim=round(0.55 - 0.15 * s.energy, 2))
            for k, s in enumerate(secs)
        ])
    elif name == "stills":
        video = usable_video(p, ffmpeg_bin)
        scenes = []
        for k, s in enumerate(secs):
            t = (s.start + s.end) / 2
            still = grab_frame(ffmpeg_bin, video, t, p.bg_assets_dir / f"still-{round(t * 1000):07d}.jpg")
            src = BgSource(type="image", path=p.rel(still), origin="video-frame", url=meta.url,
                           credit=f"frame of the video at {t:.0f}s")
            scenes.append(_scene(k, s, src, motion=MOTIONS[k % len(MOTIONS)], blur=8.0,
                                 dim=round(0.55 - 0.15 * s.energy, 2)))
        bg = Background(scenes=scenes)
    elif name == "aurora":
        try:
            cover, _ = ensure_cover(p, meta, ffmpeg_bin, download)
            colors = aurora_colors(cover_palette(cover))
        except PresetError:
            colors = []  # no cover: the generator's own colours
        whole = Section(start=0.0, end=round(duration, 3), label="verse", energy=0.5, vocal=1.0)
        src = BgSource(type="generated", value="aurora:" + ",".join(colors), origin="generated")
        bg = Background(scenes=[_scene(0, whole, src)])
        bg.scenes[0].reason = "generated: follows the song's loudness, beats and sections"
    else:  # video
        video = usable_video(p, ffmpeg_bin)
        whole = Section(start=0.0, end=round(duration, 3), label="verse", energy=0.5, vocal=1.0)
        src = BgSource(type="video", path=p.rel(video), origin="source-video", url=meta.url,
                       credit="the original video")
        # one scene: a video scene always starts at the file's first frame, so only a single
        # scene stays in sync with the song
        bg = Background(scenes=[_scene(0, whole, src, blur=20.0, dim=0.45)])
        bg.scenes[0].reason = "the original video, in sync with the song"
    bg.style = "default" if name == "gradient" else f"preset:{name}"
    return bg


def apply_preset(name: str, p: JobPaths, ffmpeg_bin: str, *, replace_locked: bool = False,
                 download: Callable[[str], bytes] | None = None) -> Background:
    """Build preset ``name`` and write it as the job's ``background.json``. Refuses to replace
    scenes the user locked unless ``replace_locked``."""
    from karaoke.store import read_model, read_model_opt, write_model
    from karaoke.validate import validate_background

    if not p.meta.exists():
        raise PresetError("no meta.json yet: run the download step first")
    meta = read_model(p.meta, Meta)
    old = read_model_opt(p.background, Background)
    locked = [s.id for s in old.scenes if s.locked] if old else []
    if locked and not replace_locked:
        raise PresetError(f"locked scene(s) {', '.join(locked)} would be replaced: unlock them or confirm")
    bg = build_preset(name, p, meta, read_model_opt(p.structure, Structure), ffmpeg_bin, download)
    errors = [i for i in validate_background(bg, p, meta.duration) if i.severity == "error"]
    if errors:  # pragma: no cover — a bug in the preset, not in the job
        raise PresetError(f"preset {name} made an invalid background: {errors[0]}")
    write_model(p.background, bg)
    return bg


def available(p: JobPaths, meta: Meta | None) -> dict[str, bool]:
    video = p.source_video.exists()
    cover = video or bool(meta and meta.thumbnail) or (p.bg_assets_dir / COVER_NAME).exists()
    return {"gradient": True, "palette": cover, "cover": cover, "stills": video, "video": video, "aurora": True}
