"""Sanity checks for job files. Used by `karaoke validate`, the API and the AI skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from karaoke.models import SCHEMA_MODELS, Background, LyricsFinal
from karaoke.store import JobPaths

EPS = 0.02  # 20 ms tolerance


@dataclass
class Issue:
    severity: Literal["error", "warning"]
    file: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.file} {self.where}: {self.message}"


def validate_lyrics(lf: LyricsFinal, duration: float | None = None) -> list[Issue]:
    out: list[Issue] = []
    f = "lyrics.final.json"

    def err(where: str, msg: str, sev: Literal["error", "warning"] = "error") -> None:
        out.append(Issue(sev, f, where, msg))

    seen: set[str] = set()
    prev_start = -1.0
    prev_end = 0.0
    for line in lf.lines:
        w = f"line {line.id}"
        if line.id in seen:
            err(w, "duplicate line id")
        seen.add(line.id)
        if not line.words:
            err(w, "line has no words")
            continue
        if line.end < line.start - EPS:
            err(w, f"end {line.end:.2f} < start {line.start:.2f}")
        if line.start < prev_start - EPS:
            err(w, f"lines out of order (starts {line.start:.2f} before previous {prev_start:.2f})")
        if line.start < prev_end - 0.25:
            err(w, f"overlaps previous line by {prev_end - line.start:.2f}s", "warning")
        if duration is not None and (line.start < -EPS or line.end > duration + EPS):
            err(w, f"outside audio duration 0..{duration:.2f}")
        last = line.start - EPS
        for i, word in enumerate(line.words):
            ww = f"{w} word {i} {word.text!r}"
            if not word.text.strip():
                err(ww, "empty word text")
            if word.end < word.start - EPS:
                err(ww, f"end {word.end:.2f} < start {word.start:.2f}")
            if word.start < last - EPS:
                err(ww, "word starts before previous word")
            if not (line.start - EPS <= word.start <= line.end + EPS
                    and line.start - EPS <= word.end <= line.end + EPS):
                err(ww, "word outside its line bounds")
            if word.end - word.start > 5.0:
                err(ww, f"suspiciously long word ({word.end - word.start:.1f}s)", "warning")
            last = word.start
        prev_start, prev_end = line.start, max(prev_end, line.end)
    return out


def validate_background(bg: Background, paths: JobPaths, duration: float | None = None) -> list[Issue]:
    out: list[Issue] = []
    f = "background.json"
    if not bg.scenes:
        return [Issue("error", f, "scenes", "no scenes")]
    t = 0.0
    seen: set[str] = set()
    for s in bg.scenes:
        w = f"scene {s.id}"
        if s.id in seen:
            out.append(Issue("error", f, w, "duplicate scene id"))
        seen.add(s.id)
        if abs(s.start - t) > EPS:
            out.append(Issue("error", f, w, f"starts at {s.start:.2f}, expected {t:.2f} (gap or overlap)"))
        if s.end <= s.start:
            out.append(Issue("error", f, w, "end <= start"))
        if s.transition.type == "fade" and s.transition.duration > (s.end - s.start) / 2:
            out.append(Issue("warning", f, w, "transition longer than half the scene"))
        if not 0 <= s.dim <= 1:
            out.append(Issue("error", f, w, "dim must be within 0..1"))
        src = s.source
        if src.type in ("image", "video"):
            if not src.path:
                out.append(Issue("error", f, w, f"{src.type} scene without path"))
            elif not (paths.root / src.path).exists():
                out.append(Issue("error", f, w, f"missing file {src.path}"))
        elif src.type in ("color", "gradient") and not src.value:
            out.append(Issue("error", f, w, f"{src.type} scene without value"))
        t = s.end
    if duration is not None and abs(t - duration) > 0.5:
        out.append(Issue("error", f, "scenes", f"scenes end at {t:.2f}, audio is {duration:.2f}s"))
    return out


_FILES = {
    "job": "job.json",
    "meta": "meta.json",
    "transcript": "transcript.whisper.json",
    "lyrics_choice": "lyrics/chosen.json",
    "lyrics_final": "lyrics.final.json",
    "background": "background.json",
}


def validate_job(paths: JobPaths) -> list[Issue]:
    """Parse every existing contract file, then run the semantic checks."""
    out: list[Issue] = []
    parsed: dict[str, object] = {}
    for key, rel in _FILES.items():
        p = paths.root / rel
        if not p.exists():
            continue
        try:
            parsed[key] = SCHEMA_MODELS[key].model_validate_json(p.read_text(encoding="utf-8"))
        except ValidationError as e:
            for er in e.errors():
                loc = ".".join(str(x) for x in er["loc"])
                out.append(Issue("error", rel, loc, er["msg"]))
    duration = getattr(parsed.get("meta"), "duration", None)
    if isinstance(parsed.get("lyrics_final"), LyricsFinal):
        out += validate_lyrics(parsed["lyrics_final"], duration)  # type: ignore[arg-type]
    if isinstance(parsed.get("background"), Background):
        out += validate_background(parsed["background"], paths, duration)  # type: ignore[arg-type]
    return out
