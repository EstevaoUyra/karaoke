"""The pitch guide: melody.json, the raw pitch contour behind it, and a MIDI file of the notes."""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from karaoke.api.deps import job_paths
from karaoke.api.midi import melody_to_midi
from karaoke.models import Melody, Meta
from karaoke.store import read_model_opt

router = APIRouter(prefix="/api")

NO_MELODY = "no melody.json yet: run the melody step"


def _melody(request: Request, job_id: str) -> Melody:
    p = job_paths(request, job_id)
    m = read_model_opt(p.melody, Melody)
    if m is None:
        raise HTTPException(404, NO_MELODY)
    return m


@router.get("/jobs/{job_id}/melody")
def get_melody(request: Request, job_id: str) -> dict:
    return _melody(request, job_id).model_dump(mode="json")


def downsample_contour(f0, prob, hop_s: float, max_points: int) -> dict:
    """Blocks of ``k`` frames → one point each: the median voiced f0 (Hz) and the max voicing
    probability. Unvoiced blocks get ``None``. Times are block centres."""
    import numpy as np

    f0 = np.asarray(f0, dtype=float)
    prob = np.nan_to_num(np.asarray(prob, dtype=float))
    n = len(f0)
    k = max(1, math.ceil(n / max(1, max_points)))
    t, hz, conf = [], [], []
    for a in range(0, n, k):
        block = f0[a:a + k]
        voiced = block[np.isfinite(block) & (block > 0)]
        t.append(round((a + (len(block) - 1) / 2) * hop_s, 3))
        hz.append(round(float(np.median(voiced)), 2) if len(voiced) else None)
        conf.append(round(float(prob[a:a + k].max()), 3) if len(block) else 0.0)
    return {"hop_s": hop_s, "step_s": round(k * hop_s, 4), "t": t, "hz": hz, "conf": conf}


@router.get("/jobs/{job_id}/melody/contour")
def melody_contour(request: Request, job_id: str, max_points: int = 6000) -> dict:
    """The raw pitch contour (cache/melody.f0.npz), downsampled, for a diagnostic view.

    The confidence is ``conf`` (CREPE periodicity or pyin voicing, melody stage v2+) or ``prob``
    (pyin voicing, v1 caches)."""
    import numpy as np

    from karaoke.store import file_hash

    p = job_paths(request, job_id)
    if not p.melody_f0.exists():
        raise HTTPException(404, "no pitch contour yet: run the melody step")
    try:
        with np.load(p.melody_f0) as z:
            conf_key = "conf" if "conf" in z.files else "prob"
            f0, prob, hop_s = z["f0"], z[conf_key], float(z["hop_s"])
            vhash = str(z["vocals_hash"]) if "vocals_hash" in z else None
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"unreadable pitch contour: {exc}") from exc
    out = downsample_contour(f0, prob, hop_s, max(100, min(max_points, 50_000)))
    out["stale"] = bool(vhash and p.vocals.exists() and file_hash(p.vocals) != vhash)
    return out


@router.get("/jobs/{job_id}/melody.mid")
def melody_midi(request: Request, job_id: str) -> Response:
    m = _melody(request, job_id)
    meta = read_model_opt(job_paths(request, job_id).meta, Meta)
    title = (meta.clean_title or meta.title) if meta else job_id
    return Response(melody_to_midi(m, title=f"{title} (melody)"), media_type="audio/midi",
                    headers={"Content-Disposition": f'attachment; filename="{job_id}-melody.mid"',
                             "Cache-Control": "no-cache"})
