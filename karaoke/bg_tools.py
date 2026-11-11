"""Tools for art-directing a background (the ``karaoke-backgrounds`` skill, or a person at the CLI).

- ``brief``: everything needed to decide: the song, its sections with the lines sung in each, the
  materials at hand, the cover's colours, the current scenes (and which are locked).
- ``scene_frame`` / ``contact_sheet``: what each scene looks like under the lyrics, in one picture.
- ``check``: the rules a background must meet: the contract (``validate_background``), locked
  scenes untouched, scene changes where the song changes, and lyrics that stay readable.
- ``add_image``: download a picture into ``assets/bg`` and record where it came from.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from karaoke.models import Background, LyricsFinal, Meta, Scene, Structure
from karaoke.store import JobPaths, read_model, read_model_opt

MIN_SCENE_S = 6.0  # shorter scenes flicker
SNAP_S = 1.5  # a scene change should be this close to a section boundary or a gap between lines
LYRIC_BAND = (0.50, 0.92)  # share of the frame's height where the lyric lines are drawn
MAX_MEAN_LUMA = 0.42  # behind the lyrics: brighter than this and white text loses contrast
MAX_P95_LUMA = 0.78
MAX_DETAIL = 0.060  # mean local contrast behind the lyrics: more than this and a picture fights the text
IMAGE_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
MAX_IMAGE_BYTES = 40 * 1024 * 1024
SHEET_CELL = (640, 360)


@dataclass(frozen=True)
class Finding:
    severity: str  # "error" must be fixed; "warning" should be looked at
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity}: {self.where}: {self.message}"


# --------------------------------------------------------------------------- brief


def video_is_still(p: JobPaths, duration: float, ffmpeg_bin: str) -> bool | None:
    """Does the source video show the same picture a fifth, half and four fifths of the way in?"""
    import tempfile

    import numpy as np
    from PIL import Image

    from karaoke.bg_presets import PresetError, grab_frame, usable_video

    if not p.source_video.exists():
        return None
    try:
        video = usable_video(p, ffmpeg_bin)
        with tempfile.TemporaryDirectory(prefix="karaoke-still-") as tmp:
            shots = []
            for k, share in enumerate((0.2, 0.5, 0.8)):
                f = grab_frame(ffmpeg_bin, video, duration * share, Path(tmp) / f"f{k}.jpg")
                with Image.open(f) as im:
                    shots.append(np.asarray(im.convert("L").resize((64, 36)), dtype=float) / 255.0)
    except (PresetError, OSError, RuntimeError, ValueError):
        return None
    return bool(max(np.abs(shots[0] - s).mean() for s in shots[1:]) < 0.02)


def remove_image(p: JobPaths, name: str) -> None:
    """Delete a downloaded picture from ``assets/bg`` and its entry in ``credits.json``. Refuses
    if a scene of the current background still uses it."""
    target = p.bg_assets_dir / Path(name).name
    bg = read_model_opt(p.background, Background)
    if bg and any(s.source.path == p.rel(target) for s in bg.scenes):
        raise ValueError(f"{p.rel(target)} is used by a scene: change the scene first")
    if not target.exists():
        raise ValueError(f"no such picture: {p.rel(target)}")
    target.unlink()
    if p.bg_credits.exists():
        credits = json.loads(p.bg_credits.read_text())
        if credits.pop(target.name, None) is not None:
            p.bg_credits.write_text(json.dumps(credits, indent=2, ensure_ascii=False) + "\n")


def brief(p: JobPaths, ffmpeg_bin: str | None = None) -> dict:
    """What an art director needs to know about the job, as plain data. With ``ffmpeg_bin`` the
    source video is sampled to see whether it is a moving picture at all."""
    from karaoke.bg_presets import COVER_NAME, PRESETS, available, cover_palette
    from karaoke.render.generated import STYLES

    meta = read_model(p.meta, Meta)
    lyrics = read_model_opt(p.lyrics_final, LyricsFinal)
    structure = read_model_opt(p.structure, Structure)
    bg = read_model_opt(p.background, Background)
    lines = lyrics.lines if lyrics else []

    def text(a: float, b: float) -> list[str]:
        return [" ".join(w.text for w in ln.words) for ln in lines if a <= (ln.start + ln.end) / 2 < b]

    sections = [
        {"start": s.start, "end": s.end, "label": s.label, "loudness": s.energy, "sung": s.vocal,
         "lines": text(s.start, s.end)}
        for s in (structure.sections if structure else [])
    ]
    gaps = []
    prev = 0.0
    for ln in lines:
        if ln.start - prev >= 1.0:
            gaps.append([round(prev, 2), round(ln.start, 2)])
        prev = max(prev, ln.end)
    if lines and meta.duration - prev >= 1.0:
        gaps.append([round(prev, 2), round(meta.duration, 2)])  # after the last sung line
    cover = p.bg_assets_dir / COVER_NAME
    palette = []
    if cover.exists():
        import colorsys

        for h, s, share in cover_palette(cover):
            r, g, b = colorsys.hsv_to_rgb(h, max(s, 0.05), 0.8)
            palette.append({"hex": f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}",
                            "saturation": round(s, 2), "share": round(share, 2)})
    images = sorted(p.rel(f) for f in p.bg_assets_dir.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    return {
        "job": p.id,
        "title": meta.clean_title or meta.title,
        "artist": meta.clean_artist or meta.artist or meta.uploader,
        "duration": meta.duration,
        "language": lyrics.language if lyrics else None,
        "tempo_bpm": structure.tempo_bpm if structure else None,
        "sections": sections,
        "has_structure": structure is not None,
        # silences of 1 s or more before, between and after the sung lines: safe places for a scene change
        "line_gaps": gaps,
        "materials": {
            "presets": {name: {"what": what, "available": ok}
                        for (name, what), ok in zip(PRESETS.items(), available(p, meta).values(), strict=True)},
            "generated_styles": list(STYLES),
            "cover": p.rel(cover) if cover.exists() else None,
            "cover_palette": palette,
            "source_video": p.rel(p.source_video) if p.source_video.exists() else None,
            # True: the video is one still picture (a cover) for the whole song, so `video` and
            # `stills` add nothing over `cover`. None: unknown (no video, or it could not be read).
            "source_video_is_still": video_is_still(p, meta.duration, ffmpeg_bin) if ffmpeg_bin else None,
            "images": images,
        },
        "current": None if bg is None else {
            "style": bg.style,
            "scenes": [{"id": s.id, "start": s.start, "end": s.end, "type": s.source.type,
                        "value": s.source.value, "path": s.source.path, "locked": s.locked} for s in bg.scenes],
        },
    }


# --------------------------------------------------------------------------- frames


def scene_frame(bg: Background, scene: Scene, p: JobPaths, t: float, size: tuple[int, int], ffmpeg_bin: str):
    """The background alone at song time ``t`` (inside ``scene``), as a PIL image. Transitions are
    not drawn: this is the scene as it looks once it has faded in."""
    import subprocess
    import tempfile

    from PIL import Image

    from karaoke.render.compositor import build_graph

    structure = read_model_opt(p.structure, Structure)
    if scene.source.type == "generated":
        from karaoke.render.generated import frames

        frame = next(frames(scene.source.value or "", bg.fps, int(t * bg.fps), int(t * bg.fps) + 1, structure,
                            max(t + 1.0, bg.scenes[-1].end)))
        im = Image.fromarray(frame).resize(size, Image.Resampling.BICUBIC)
        if scene.dim > 0:
            im = Image.blend(im, Image.new("RGB", size, "black"), min(scene.dim, 1.0))
        return im
    span = 4.0
    cut = scene.transition.model_copy(update={"type": "cut", "duration": 0.0})
    one = Background(fps=10, scenes=[scene.model_copy(update={"start": 0.0, "end": span, "transition": cut})])
    with tempfile.TemporaryDirectory(prefix="karaoke-frame-") as tmp:
        work = Path(tmp)
        g = build_graph(one, p.root, work, size=size, duration=span, ffmpeg_bin=ffmpeg_bin)
        out = work / "frame.png"
        pre = []
        if scene.source.type == "video":  # a video scene plays from its first frame: look at the same moment
            k = g.input_args.index("-i")
            pre = ["-ss", f"{max(0.0, t - scene.start):.3f}"]
            args = [*g.input_args[:k], *pre, *g.input_args[k:]]
        else:
            args = g.input_args
        subprocess.run([ffmpeg_bin, "-v", "error", "-nostdin", "-y", *args, "-filter_complex",
                        f"{g.filter};[bg]select='gte(n,{0 if pre else 20})'[v]", "-map", "[v]", "-frames:v", "1", str(out)],
                       check=True, capture_output=True)
        with Image.open(out) as im:
            return im.convert("RGB")


def legibility(im) -> dict[str, float]:
    """How hard the picture makes white lyrics to read, in the band where they are drawn."""
    import numpy as np

    a = np.asarray(im.convert("L"), dtype=float) / 255.0
    h = a.shape[0]
    band = a[int(LYRIC_BAND[0] * h):int(LYRIC_BAND[1] * h)]
    small = band[::2, ::2]
    detail = (np.abs(np.diff(small, axis=0)).mean() + np.abs(np.diff(small, axis=1)).mean()) / 2
    return {"mean": round(float(band.mean()), 3), "p95": round(float(np.percentile(band, 95)), 3),
            "detail": round(float(detail * 10), 3)}


def _mid(scene: Scene) -> float:
    return (scene.start + scene.end) / 2


def contact_sheet(p: JobPaths, out: Path, ffmpeg_bin: str, with_lyrics: bool = True) -> list[dict]:
    """One picture with every scene at its middle (lyrics drawn over it), and the scenes' numbers.

    Returns one dict per scene: id, time shown, section label there, and the legibility measures
    of the background alone."""
    import subprocess
    import tempfile

    from PIL import Image, ImageDraw

    from karaoke.models import RenderSettings
    from karaoke.render.ass import build_ass

    bg = read_model(p.background, Background)
    meta = read_model(p.meta, Meta)
    lyrics = read_model_opt(p.lyrics_final, LyricsFinal)
    structure = read_model_opt(p.structure, Structure)
    lines = lyrics.lines if lyrics else []
    cols = min(4, max(1, len(bg.scenes)))
    rows = -(-len(bg.scenes) // cols)
    w, h = SHEET_CELL
    sheet = Image.new("RGB", (cols * w, rows * h), "black")
    info = []
    with tempfile.TemporaryDirectory(prefix="karaoke-sheet-") as tmp:
        work = Path(tmp)
        if with_lyrics and lyrics:
            (work / "sub.ass").write_text(build_ass(lyrics, meta, RenderSettings(), (w, h), 0.5), encoding="utf-8")
        for k, sc in enumerate(bg.scenes):
            # prefer a moment with lyrics on screen: the middle of the line nearest the scene's middle
            inside = [ln for ln in lines if sc.start + 0.5 <= (ln.start + ln.end) / 2 <= sc.end - 0.5]
            t = _mid(sc) if not inside else min(((ln.start + ln.end) / 2 for ln in inside), key=lambda x: abs(x - _mid(sc)))
            im = scene_frame(bg, sc, p, t, (w, h), ffmpeg_bin)
            leg = legibility(im)
            if with_lyrics and lyrics:
                im.save(work / "bg.png")
                subprocess.run([ffmpeg_bin, "-v", "error", "-nostdin", "-y", "-loop", "1", "-framerate", "10", "-t", "1",
                                "-i", "bg.png", "-vf", f"setpts=PTS+{t:.3f}/TB,ass=filename=sub.ass", "-frames:v", "1",
                                "out.png"], cwd=work, check=True, capture_output=True)
                with Image.open(work / "out.png") as o:
                    im = o.convert("RGB")
            label = next((s.label for s in (structure.sections if structure else []) if s.start <= t < s.end), "")
            ImageDraw.Draw(im).text((8, 6), f"{sc.id}  {sc.start:.0f}-{sc.end:.0f}s  {label}", fill="white",
                                    stroke_width=2, stroke_fill="black")
            sheet.paste(im, ((k % cols) * w, (k // cols) * h))
            info.append({"scene": sc.id, "shown_at": round(t, 2), "section": label, **leg})
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return info


# --------------------------------------------------------------------------- check


def _locked_changed(old: Background | None, new: Background) -> list[str]:
    if old is None:
        return []
    now = {s.id: s for s in new.scenes}
    return [s.id for s in old.scenes if s.locked and now.get(s.id) != s]


def check(p: JobPaths, ffmpeg_bin: str | None = None, before: Background | None = None) -> list[Finding]:
    """Everything wrong or doubtful about the job's ``background.json``.

    ``before``: the background as it was when the work started; its locked scenes must be
    unchanged. ``ffmpeg_bin``: given, each scene's frame is measured for legibility (slower)."""
    from karaoke.validate import validate_background

    bg = read_model(p.background, Background)
    meta = read_model(p.meta, Meta)
    out = [Finding(i.severity, i.where, i.message) for i in validate_background(bg, p, meta.duration)]
    for sid in _locked_changed(before, bg):
        out.append(Finding("error", f"scene {sid}", "this scene was locked by the user and has been changed or removed"))
    structure = read_model_opt(p.structure, Structure)
    lyrics = read_model_opt(p.lyrics_final, LyricsFinal)
    lines = lyrics.lines if lyrics else []
    anchors = [s.start for s in structure.sections[1:]] if structure else []
    for sc in bg.scenes:
        where = f"scene {sc.id}"
        if sc.end - sc.start < MIN_SCENE_S and len(bg.scenes) > 1:
            out.append(Finding("warning", where,
                               f"only {sc.end - sc.start:.1f} s long (under {MIN_SCENE_S:.0f} s flickers)"))
        if sc.start > 0:
            cut = next((ln for ln in lines if ln.start + 0.15 < sc.start < ln.end - 0.15), None)
            if cut is not None:
                out.append(Finding("warning", where, f"starts at {sc.start:.2f} s, in the middle of a sung line "
                                                     f"({cut.start:.2f}-{cut.end:.2f} s)"))
            elif anchors and min(abs(sc.start - a) for a in anchors) > SNAP_S and sc.source.type != "generated":
                near = min(anchors, key=lambda a: abs(sc.start - a))
                out.append(Finding("warning", where, f"starts at {sc.start:.2f} s, not at a section boundary "
                                                     f"(nearest: {near:.2f} s)"))
        if sc.source.type in ("image", "video") and not (sc.source.credit or sc.source.origin):
            out.append(Finding("warning", where, "no credit or origin recorded for this picture"))
    if ffmpeg_bin:
        for sc in bg.scenes:
            try:
                leg = legibility(scene_frame(bg, sc, p, _mid(sc), SHEET_CELL, ffmpeg_bin))
            except Exception as exc:  # noqa: BLE001 — a scene that cannot be drawn is the finding
                out.append(Finding("error", f"scene {sc.id}", f"cannot be drawn: {exc}"))
                continue
            if leg["mean"] > MAX_MEAN_LUMA or leg["p95"] > MAX_P95_LUMA:
                out.append(Finding("warning", f"scene {sc.id}", f"too bright behind the lyrics (mean {leg['mean']:.2f}, "
                                                                f"brightest {leg['p95']:.2f}): raise dim"))
            if leg["detail"] > MAX_DETAIL * 10:
                out.append(Finding("warning", f"scene {sc.id}", f"too much detail behind the lyrics ({leg['detail']:.2f}): "
                                                                "raise blur"))
    return out


# --------------------------------------------------------------------------- images


def add_image(p: JobPaths, url: str, credit: str, license_: str, source_page: str | None = None,
              download=None) -> Path:
    """Download a picture (JPEG, PNG or WebP) into ``assets/bg`` and record its credit in
    ``assets/bg/credits.json``. Returns the saved file."""
    from PIL import Image, UnidentifiedImageError

    if not url.lower().startswith("https://"):
        raise ValueError("only https:// pictures are fetched")
    if download is None:
        import httpx

        def download(u: str) -> bytes:
            with httpx.stream("GET", u, timeout=30.0, follow_redirects=True,
                              headers={"User-Agent": "karaoke-backgrounds/1.0"}) as r:
                r.raise_for_status()
                data = b""
                for chunk in r.iter_bytes():
                    data += chunk
                    if len(data) > MAX_IMAGE_BYTES:
                        raise ValueError("picture larger than 40 MB")
                return data

    data = download(url)
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("the URL did not return a picture (JPEG, PNG or WebP)") from exc
    if fmt not in IMAGE_FORMATS:
        raise ValueError(f"unsupported picture format {fmt}")
    p.bg_assets_dir.mkdir(parents=True, exist_ok=True)
    dest = p.bg_assets_dir / f"img-{hashlib.sha1(data).hexdigest()[:10]}{IMAGE_FORMATS[fmt]}"
    dest.write_bytes(data)
    credits = json.loads(p.bg_credits.read_text()) if p.bg_credits.exists() else {}
    credits[dest.name] = {"url": url, "page": source_page, "credit": credit, "license": license_}
    p.bg_credits.write_text(json.dumps(credits, indent=2, ensure_ascii=False) + "\n")
    return dest
