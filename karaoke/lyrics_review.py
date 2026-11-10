"""Tools for reviewing synced lyrics (the ``karaoke-lyrics-review`` skill, or a person at the CLI).

- ``brief``: every line with its words, what Whisper heard at that time, the flags, the audio no
  line accounts for, and a list of suspects: the places worth a look.
- ``apply_edits``: a list of small, explicit edits applied safely: words the user edited by hand
  are never touched, changed words become ``origin: ai`` (so ``align`` keeps them), and nothing is
  saved unless the result passes ``validate_lyrics``.
- ``compare``: how close a ``lyrics.final.json`` is to a known truth (for the seeded-error job).

The lyrics text is the source of truth for the WORDS and Whisper for the TIMING. Whisper mishears
(``window`` → ``widow``); a difference between the two is a question, not an answer.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from karaoke.models import Line, LyricsFinal, Meta, Transcript, UnmatchedAudio, Word, WordOrigin, now
from karaoke.store import JobPaths, read_model, read_model_opt, write_model

AI_FLAG = "ai_edited"
LOW_CONF = 0.5
START_OFF_S = 0.4  # a line starting this far from the first word Whisper heard in it is a suspect
_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


class EditError(ValueError):
    """An edit cannot be applied (bad reference, a hand-edited word, an invalid result)."""


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(_PUNCT.sub(" ", t).split())


def _heard(transcript: Transcript | None, start: float, end: float, pad: float = 0.25) -> list[dict]:
    if transcript is None:
        return []
    out = []
    for seg in transcript.segments:
        for w in seg.words:
            mid = (w.start + w.end) / 2
            if start - pad <= mid <= end + pad:
                out.append({"text": w.text.strip(), "start": w.start, "end": w.end, "prob": round(w.prob, 2)})
    return out


def brief(p: JobPaths) -> dict:
    """The lyrics and the evidence around them, as plain data."""
    lf = read_model(p.lyrics_final, LyricsFinal)
    meta = read_model_opt(p.meta, Meta)
    transcript = read_model_opt(p.transcript, Transcript)
    lines = []
    suspects: list[dict] = []
    prev_end = 0.0
    for ln in lf.lines:
        heard = _heard(transcript, ln.start, ln.end)
        text = " ".join(w.text for w in ln.words)
        heard_text = " ".join(h["text"] for h in heard)
        agree = SequenceMatcher(None, norm(text).split(), norm(heard_text).split(), autojunk=False).ratio()
        lines.append({
            "id": ln.id, "start": ln.start, "end": ln.end, "text": text, "flags": ln.flags,
            "words": [{"i": i, "text": w.text, "start": w.start, "end": w.end, "conf": round(w.conf, 2),
                       "origin": w.origin.value} for i, w in enumerate(ln.words)],
            "whisper_heard": heard, "agreement": round(agree, 2),
        })

        def suspect(kind: str, note: str, line: Line = ln) -> None:
            suspects.append({"line": line.id, "at": round(line.start, 2), "kind": kind, "note": note})

        if any(w.origin == WordOrigin.manual for w in ln.words):
            continue  # the user's own work: not a suspect
        if all(w.origin == WordOrigin.ai for w in ln.words):
            continue  # already reviewed word by word
        if not heard:
            suspect("not_heard", "Whisper heard nothing during this line: is it sung at all?")
        elif agree < 0.6:
            suspect("text_differs", f"lyrics: “{text}” / Whisper heard: “{heard_text}”")
        low = [w.text for w in ln.words if w.conf < LOW_CONF or w.origin == WordOrigin.interpolated]
        if low and heard:
            suspect("unsure_words", "low confidence or interpolated timing: " + ", ".join(low))
        if heard and abs(heard[0]["start"] - ln.start) > START_OFF_S:
            suspect("start_off", f"line starts at {ln.start:.2f} s but the first word heard near it is at "
                                 f"{heard[0]['start']:.2f} s")
        if ln.start < prev_end - 0.05:
            suspect("overlap", f"starts before the previous line ends ({prev_end:.2f} s)")
        for f in ln.flags:
            if f != AI_FLAG:
                suspect(f, "flag set by the aligner")
        prev_end = max(prev_end, ln.end)
    for k, u in enumerate(lf.unmatched_audio):
        suspects.append({"line": None, "at": round(u.start, 2), "kind": "unmatched_audio",
                         "note": f"Whisper heard “{u.text}” at {u.start:.2f}-{u.end:.2f} s and no line has it "
                                 f"(unmatched #{k}): a missing line or repeat, or just an ad-lib",
                         "whisper_heard": _heard(transcript, u.start, u.end, pad=0.05)})
    sources = {}
    for f in sorted(p.lyrics_sources_dir.glob("*")) if p.lyrics_sources_dir.exists() else []:
        if f.suffix in (".txt", ".lrc", ".vtt") and f.stat().st_size < 200_000:
            sources[f.name] = f.read_text(encoding="utf-8", errors="replace")
    return {
        "job": p.id,
        "title": (meta.clean_title or meta.title) if meta else None,
        "artist": (meta.clean_artist or meta.artist) if meta else None,
        "duration": meta.duration if meta else None,
        "language": lf.language,
        "text_source": lf.text_source.value,
        "lines": lines,
        "unmatched_audio": [u.model_dump() for u in lf.unmatched_audio],
        "suspects": sorted(suspects, key=lambda s: s["at"]),
        "other_sources": sources,
    }


# --------------------------------------------------------------------------- edits


def _line(lf: LyricsFinal, line_id: str) -> Line:
    for ln in lf.lines:
        if ln.id == line_id:
            return ln
    raise EditError(f"no line {line_id!r}")


def _guard(words: list[Word], what: str) -> None:
    if any(w.origin == WordOrigin.manual for w in words):
        raise EditError(f"{what}: it has words the user edited by hand; leave those alone")


def _mark(ln: Line, words: list[Word]) -> None:
    for w in words:
        w.origin = WordOrigin.ai
        w.conf = max(w.conf, 0.9)
    from karaoke.align.confidence import COMPUTED_FLAGS

    ln.flags = [f for f in ln.flags if f not in COMPUTED_FLAGS]  # the aligner's doubts were about the old words
    if AI_FLAG not in ln.flags:
        ln.flags.append(AI_FLAG)


def _words(spec: list[dict]) -> list[Word]:
    if not spec:
        raise EditError("a line needs at least one word")
    try:
        return [Word(text=str(w["text"]), start=float(w["start"]), end=float(w["end"]), conf=0.9, origin=WordOrigin.ai)
                for w in spec]
    except (KeyError, TypeError, ValueError) as exc:
        raise EditError(f"each word needs text, start and end: {exc}") from exc


def _new_id(lf: LyricsFinal) -> str:
    used = {ln.id for ln in lf.lines}
    n = len(lf.lines) + 1
    while f"l{n:03d}" in used:
        n += 1
    return f"l{n:03d}"


def apply_edit(lf: LyricsFinal, e: dict) -> str:
    """Apply one edit in place; returns a one-line description."""
    op = e.get("op")
    if op == "replace_word":
        ln = _line(lf, e["line"])
        w = ln.words[int(e["word"])]
        _guard([w], f"replace_word {ln.id}[{e['word']}]")
        old, w.text = w.text, str(e["text"])
        _mark(ln, [w])
        return f"{ln.id}: “{old}” → “{w.text}”"
    if op == "retime_word":
        ln = _line(lf, e["line"])
        w = ln.words[int(e["word"])]
        _guard([w], f"retime_word {ln.id}[{e['word']}]")
        w.start, w.end = float(e["start"]), float(e["end"])
        _mark(ln, [w])
        return f"{ln.id}: “{w.text}” now {w.start:.2f}-{w.end:.2f} s"
    if op == "shift_line":
        ln = _line(lf, e["line"])
        _guard(ln.words, f"shift_line {ln.id}")
        by = float(e["by"])
        for w in ln.words:
            w.start, w.end = round(w.start + by, 3), round(w.end + by, 3)
        _mark(ln, ln.words)
        return f"{ln.id}: moved by {by:+.2f} s"
    if op == "retime_line":
        ln = _line(lf, e["line"])
        _guard(ln.words, f"retime_line {ln.id}")
        a0, b0, a1, b1 = ln.words[0].start, ln.words[-1].end, float(e["start"]), float(e["end"])
        if b1 <= a1:
            raise EditError(f"retime_line {ln.id}: end must be after start")
        k = (b1 - a1) / max(b0 - a0, 1e-6)
        for w in ln.words:
            w.start, w.end = round(a1 + (w.start - a0) * k, 3), round(a1 + (w.end - a0) * k, 3)
        _mark(ln, ln.words)
        return f"{ln.id}: now {a1:.2f}-{b1:.2f} s"
    if op == "set_line":
        ln = _line(lf, e["line"])
        _guard(ln.words, f"set_line {ln.id}")
        ln.words = _words(e["words"])
        _mark(ln, ln.words)
        return f"{ln.id}: rewritten as “{' '.join(w.text for w in ln.words)}”"
    if op == "insert_line":
        words = _words(e["words"])
        ln = Line(id=_new_id(lf), start=words[0].start, end=words[-1].end, words=words, flags=[AI_FLAG])
        lf.lines.append(ln)
        return f"{ln.id}: new line “{' '.join(w.text for w in words)}” at {ln.start:.2f} s"
    if op == "delete_line":
        ln = _line(lf, e["line"])
        _guard(ln.words, f"delete_line {ln.id}")
        lf.lines.remove(ln)
        return f"{ln.id}: deleted (“{' '.join(w.text for w in ln.words)}”)"
    if op == "clear_unmatched":
        a, b = float(e["start"]), float(e["end"])
        before = len(lf.unmatched_audio)
        lf.unmatched_audio = [u for u in lf.unmatched_audio if not (a - 0.05 <= u.start and u.end <= b + 0.05)]
        return f"unmatched audio {a:.2f}-{b:.2f} s: {before - len(lf.unmatched_audio)} entr(y/ies) cleared"
    raise EditError(f"unknown op {op!r}")


def apply_edits(p: JobPaths, edits: list[dict], dry_run: bool = False) -> list[str]:
    """Apply ``edits`` to the job's lyrics.final.json. All or nothing: any bad edit, or a result
    that fails validation, raises ``EditError`` and leaves the file as it was."""
    from karaoke.validate import validate_lyrics

    lf = read_model(p.lyrics_final, LyricsFinal)
    meta = read_model_opt(p.meta, Meta)
    log = []
    for k, e in enumerate(edits):
        try:
            log.append(apply_edit(lf, e) + (f"  ({e['why']})" if e.get("why") else ""))
        except (KeyError, IndexError) as exc:
            raise EditError(f"edit {k} ({e.get('op')}): bad reference {exc}") from exc
    lf.lines.sort(key=lambda ln: ln.words[0].start)
    for ln in lf.lines:
        ln.start, ln.end = ln.words[0].start, ln.words[-1].end
    errors = [str(i) for i in validate_lyrics(lf, meta.duration if meta else None) if i.severity == "error"]
    for a, b in zip(lf.lines, lf.lines[1:], strict=False):  # stricter than the contract: no new overlaps
        if b.start < a.end - 0.05:
            errors.append(f"{b.id} starts at {b.start:.2f} s, before {a.id} ends ({a.end:.2f} s)")
    if errors:
        raise EditError("the result is not valid, nothing was saved: " + "; ".join(errors[:5]))
    if not dry_run:
        lf.updated_at = now()
        write_model(p.lyrics_final, lf)
    return log


# --------------------------------------------------------------------------- scoring against a truth


def compare(final: LyricsFinal, truth: LyricsFinal, tol_s: float = 0.15) -> dict:
    """Word-level distance from ``final`` to ``truth``: wrong, missing and extra words, and the
    timing of the words they share."""
    a = [(norm(w.text), w.start) for ln in truth.lines for w in ln.words]
    b = [(norm(w.text), w.start) for ln in final.lines for w in ln.words]
    sm = SequenceMatcher(None, [x[0] for x in a], [x[0] for x in b], autojunk=False)
    wrong = missing = extra = 0
    errs: list[float] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            errs += [abs(a[i][1] - b[j][1]) for i, j in zip(range(i1, i2), range(j1, j2), strict=True)]
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            wrong += n
            missing += (i2 - i1) - n
            extra += (j2 - j1) - n
        elif tag == "delete":
            missing += i2 - i1
        else:
            extra += j2 - j1
    late = sum(e > tol_s for e in errs)
    return {
        "truth_words": len(a), "wrong_words": wrong, "missing_words": missing, "extra_words": extra,
        "mistimed_words": late, "worst_timing_s": round(max(errs), 2) if errs else 0.0,
        "lines": len(final.lines), "truth_lines": len(truth.lines),
        "total_errors": wrong + missing + extra + late,
    }


__all__ = ["EditError", "apply_edit", "apply_edits", "brief", "compare", "norm", "UnmatchedAudio"]
