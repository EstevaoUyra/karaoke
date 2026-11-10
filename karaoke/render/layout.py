"""Text measurement and display lines for the ASS renderer.

The lyrics font size never changes. A lyric line too wide for one row is split into several
*display lines* at its best break (``karaoke.linebreak``), each sung in its own row.

Widths are measured with the real font (found with fontconfig's ``fc-match``, which is also how
libass finds it), scaled the way libass sizes fonts: an ASS font size is the font's full height
(ascender + descender), not its em size. Without the font file, a per-character estimate is used.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from karaoke.linebreak import W, split_to_fit
from karaoke.models import Line

CHAR_WIDTH_EM = 0.58  # fallback: rough average advance of a bold sans font, in font sizes
_REF_SIZE = 100


@functools.lru_cache(maxsize=16)
def font_file(family: str) -> Path | None:
    exe = shutil.which("fc-match")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-f", "%{file}", family], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    p = Path(out.stdout.strip())
    return p if out.returncode == 0 and p.is_file() else None


@functools.lru_cache(maxsize=16)
def _pil_font(path: str):
    from PIL import ImageFont

    return ImageFont.truetype(path, _REF_SIZE)


def text_measurer(family: str, font_size: float, outline: float = 0.0) -> Callable[[str], float]:
    """``width(text)`` in pixels for text in ``family`` at ASS size ``font_size``."""
    path = font_file(family)
    if path is not None:
        try:
            font = _pil_font(str(path))
            asc, desc = font.getmetrics()
            k = font_size / (asc + desc)
            return lambda text: font.getlength(text) * k + 2 * outline
        except Exception:  # noqa: BLE001 — unreadable font: fall back to the estimate
            pass
    return lambda text: len(text) * CHAR_WIDTH_EM * font_size + 2 * outline


def display_lines(lines: list[Line], width: Callable[[str], float], max_width: float,
                  soft_width: float | None = None) -> list[Line]:
    """Lines as they are shown: lines wider than ``max_width`` are split (same id, fewer words),
    and so are lines wider than ``soft_width`` that have a strong break (punctuation, parentheses)."""
    out: list[Line] = []
    for ln in lines:
        if not ln.words:
            continue
        ws = [W(w.text.strip(), w.start, w.end) for w in ln.words]
        parts = split_to_fit(ws, width, max_width, soft_width)
        if len(parts) == 1:
            out.append(ln)
            continue
        for part in parts:
            words = [ln.words[i] for i in part]
            out.append(ln.model_copy(update={"words": words, "start": words[0].start,
                                             "end": max(w.end for w in words)}))
    return out
