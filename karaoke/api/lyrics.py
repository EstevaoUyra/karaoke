"""lyrics.final.json: read, validate, save (atomically, with every edited word marked ``manual``)."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

from karaoke.api.deps import issue_dict, job_paths, state
from karaoke.models import LyricsFinal, Meta, StageName, StageStatus, WordOrigin, now
from karaoke.store import read_model_opt, write_model
from karaoke.validate import validate_lyrics

router = APIRouter(prefix="/api")

PROTECTED = (WordOrigin.manual, WordOrigin.ai)


def _key(text: str, start: float, end: float) -> tuple[str, float, float]:
    return (text, round(start, 3), round(end, 3))


def mark_manual_edits(old: LyricsFinal | None, new: LyricsFinal) -> int:
    """Safety net for the UI's own marking: any word that does not exist unchanged (same text and
    timing) in the file on disk was edited, so it gets ``origin: manual`` and align never touches it.
    Returns how many words were re-marked."""
    known: Counter = Counter()
    if old is not None:
        known.update(_key(w.text, w.start, w.end) for ln in old.lines for w in ln.words)
    changed = 0
    for ln in new.lines:
        for w in ln.words:
            k = _key(w.text, w.start, w.end)
            if known[k] > 0:
                known[k] -= 1
                continue
            if w.origin not in PROTECTED:
                w.origin = WordOrigin.manual
                changed += 1
    return changed


def _duration(request: Request, job_id: str) -> float | None:
    meta = read_model_opt(state(request).store.paths(job_id).meta, Meta)
    return meta.duration if meta else None


def _same_time(a: datetime, b: datetime) -> bool:
    return abs((a - b).total_seconds()) < 1e-3


@router.get("/jobs/{job_id}/lyrics")
def get_lyrics(request: Request, job_id: str) -> dict:
    p = job_paths(request, job_id)
    lf = read_model_opt(p.lyrics_final, LyricsFinal)
    if lf is None:
        raise HTTPException(404, "no lyrics.final.json yet: run the pipeline up to align")
    return lf.model_dump(mode="json")


@router.post("/jobs/{job_id}/lyrics/validate")
def validate_route(request: Request, job_id: str, body: LyricsFinal) -> dict:
    job_paths(request, job_id)
    return {"issues": [issue_dict(i) for i in validate_lyrics(body, _duration(request, job_id))]}


@router.put("/jobs/{job_id}/lyrics")
def put_lyrics(request: Request, job_id: str, body: LyricsFinal, base: datetime | None = None) -> dict:
    """Save the editor's lyrics.

    ``base`` is the ``updated_at`` the editor loaded. If the file changed on disk since (a Claude
    session or a re-run of align wrote it), the save is refused with 409 so nothing is lost.
    Errors from ``validate_lyrics`` refuse the save with 422; warnings are returned.
    """
    p = job_paths(request, job_id)
    st = state(request)
    if st.queue.state_for(job_id) == "running":
        job = st.store.load(job_id)
        if job.stages[StageName.align].status == StageStatus.running:
            raise HTTPException(409, "align is running and will rewrite lyrics.final.json: wait for it")
    old = read_model_opt(p.lyrics_final, LyricsFinal)
    if base is not None and old is not None and not _same_time(old.updated_at, base):
        raise HTTPException(409, {"message": "lyrics.final.json changed on disk since you loaded it",
                                  "current_updated_at": old.updated_at.isoformat()})
    remarked = mark_manual_edits(old, body)
    issues = validate_lyrics(body, _duration(request, job_id))
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise HTTPException(422, {"message": f"{len(errors)} problem(s) must be fixed before saving",
                                  "issues": [issue_dict(i) for i in issues]})
    body.updated_at = now()
    write_model(p.lyrics_final, body)
    return {"lyrics": body.model_dump(mode="json"), "issues": [issue_dict(i) for i in issues],
            "remarked": remarked}
