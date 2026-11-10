"""Generated backgrounds: moving colour that follows the music. No image, no model.

A ``generated`` scene source has ``value = "<style>:<hex>,<hex>,..."`` (the colours are optional).
The only style so far is ``aurora``: a slow field of drifting colour bands.

- It moves faster and is brighter where the song is louder (``structure.json``'s loudness curve).
- It swells a little on every beat.
- Its colours shift at each section boundary.

Without a structure file it still moves, at a constant pace.

Frames are drawn with numpy at ``GEN_SIZE`` (the picture has no detail: the compositor scales it up
and blurs it) and piped to ffmpeg. Everything is a function of the absolute song time, so clips
that overlap in a cross-fade show the same picture.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from karaoke.models import Structure

if TYPE_CHECKING:
    import numpy as np

STYLES = ("aurora",)
DEFAULT_COLORS = ("#3a1c71", "#d76d77", "#1f6f8b", "#ffaf7b")
GEN_SIZE = (320, 180)
BASE_SPEED = 0.10  # field phase per second in silence ...
ENERGY_SPEED = 0.45  # ... plus this much at full loudness
BASE_LEVEL = 0.34  # brightness in silence ...
ENERGY_LEVEL = 0.30  # ... plus this much at full loudness: white lyrics must stay readable
BEAT_LEVEL = 0.07  # extra brightness right on a beat (times the loudness)
BEAT_DECAY_S = 0.22
SECTION_SHIFT = 0.37  # the palette turns this far (of a full cycle) at each section
SECTION_BLEND_S = 3.0
SMOOTH_S = 2.0  # the loudness curve is smoothed over this long
# (x cycles, y cycles, speed factor, phase, weight) of the waves that make the field
WAVES = ((0.9, 0.35, 1.0, 0.0, 1.0), (-0.45, 0.8, 0.7, 1.7, 0.8), (1.3, -0.6, -0.5, 3.1, 0.5),
         (0.25, 1.4, 0.35, 4.4, 0.4))


def parse(value: str) -> tuple[str, list[tuple[int, int, int]]]:
    """``"aurora:#112233,#445566"`` → ("aurora", [(17, 34, 51), (68, 85, 102)])."""
    style, _, rest = value.partition(":")
    style = style.strip()
    if style not in STYLES:
        raise ValueError(f"unknown generated style {style!r}: use one of {', '.join(STYLES)}")
    cols = [c.strip().lstrip("#") for c in rest.split(",") if c.strip()] or [c.lstrip("#") for c in DEFAULT_COLORS]
    out = []
    for c in cols:
        if len(c) != 6:
            raise ValueError(f"not a hex colour: #{c}")
        out.append((int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)))
    if len(out) < 2:
        out = out * 2
    return style, out


class Timeline:
    """The song as the generator needs it: loudness, flow phase, beat pulse and palette shift at
    any time."""

    def __init__(self, structure: Structure | None, duration: float):
        import numpy as np

        hop = structure.energy_hop_s if structure and structure.energy else 0.5
        n = max(2, int(np.ceil(duration / hop)) + 2)
        e = np.full(n, 0.5)
        if structure and structure.energy:
            src = np.asarray(structure.energy, dtype=float)
            e[: min(n, len(src))] = src[:n]
            e[len(src):] = src[-1]
            k = max(1, int(round(SMOOTH_S / hop)))
            e = np.convolve(np.pad(e, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
        self.hop = hop
        self.energy_curve = e
        self.phase_curve = np.concatenate([[0.0], np.cumsum((BASE_SPEED + ENERGY_SPEED * e) * hop)])[:n]
        self.beats = np.asarray(structure.beats if structure else [], dtype=float)
        self.bounds = [s.start for s in structure.sections[1:]] if structure else []

    def energy(self, t: float) -> float:
        import numpy as np

        return float(np.interp(t / self.hop, np.arange(len(self.energy_curve)), self.energy_curve))

    def phase(self, t: float) -> float:
        import numpy as np

        return float(np.interp(t / self.hop, np.arange(len(self.phase_curve)), self.phase_curve))

    def beat(self, t: float) -> float:
        """1 right on a beat, decaying to 0 before the next one."""
        import numpy as np

        if not len(self.beats):
            return 0.0
        k = int(np.searchsorted(self.beats, t, side="right")) - 1
        return float(np.exp(-(t - self.beats[k]) / BEAT_DECAY_S)) if k >= 0 else 0.0

    def shift(self, t: float) -> float:
        """How far the palette has turned: one step per section passed, eased around the boundary."""
        total = 0.0
        for b in self.bounds:
            x = (t - b) / SECTION_BLEND_S + 0.5
            x = min(1.0, max(0.0, x))
            total += x * x * (3 - 2 * x)
        return total * SECTION_SHIFT


def frames(value: str, fps: int, lo: int, hi: int, structure: Structure | None, duration: float,
           size: tuple[int, int] = GEN_SIZE) -> Iterator[np.ndarray]:
    """RGB frames (H x W x 3, uint8) of output frames ``lo..hi-1``."""
    import numpy as np

    _, colors = parse(value)
    tl = Timeline(structure, max(duration, hi / fps))
    w, h = size
    x, y = np.meshgrid(np.linspace(0.0, 1.0, w), np.linspace(0.0, 1.0, h))
    spatial = [2 * np.pi * (kx * x + ky * y) + ph for kx, ky, _, ph, _ in WAVES]
    weight = sum(wt for *_, wt in WAVES)
    # a cyclic palette, as a lookup table
    pal = np.asarray([*colors, colors[0]], dtype=float)
    pos = np.linspace(0.0, 1.0, len(pal))
    lut = np.stack([np.interp(np.linspace(0.0, 1.0, 512, endpoint=False), pos, pal[:, c]) for c in range(3)], axis=1)
    # darker towards the top and the edges: depth, and calmer where the pitch guide sits
    shade = (0.72 + 0.28 * y) * (1.0 - 0.25 * (2 * x - 1) ** 2)
    for f in range(lo, hi):
        t = f / fps
        e, ph = tl.energy(t), tl.phase(t)
        field = sum(wt * np.sin(sp + 2 * np.pi * ph * speed)
                    for sp, (_, _, speed, _, wt) in zip(spatial, WAVES, strict=True)) / weight
        idx = ((field * 0.5 + 0.5) * 0.6 + tl.shift(t)) % 1.0
        level = BASE_LEVEL + ENERGY_LEVEL * e + BEAT_LEVEL * e * tl.beat(t)
        rgb = lut[(idx * 512).astype(int) % 512] * (shade * level)[..., None]
        yield np.clip(rgb, 0, 255).astype(np.uint8)


def render_clip(value: str, fps: int, lo: int, hi: int, structure: Structure | None, duration: float,
                out: Path, ffmpeg_bin: str, size: tuple[int, int] = GEN_SIZE) -> Path:
    """Encode output frames ``lo..hi-1`` of a generated source to ``out`` (H.264, small)."""
    parse(value)  # fail before starting ffmpeg
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_bin, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames(value, fps, lo, hi, structure, duration, size):
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed encoding a generated background: {err[-400:]}")
    except BrokenPipeError as exc:
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
        raise RuntimeError(f"ffmpeg failed encoding a generated background: {err[-400:]}") from exc
    finally:
        if proc.poll() is None:
            proc.kill()
    return out
