"""background.json: read, save, and replace a scene's image with an upload."""

from __future__ import annotations

import hashlib
import io
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from karaoke.api.deps import issue_dict, job_paths
from karaoke.models import Background, BgSource, Meta
from karaoke.store import JobPaths, read_model_opt, write_model
from karaoke.validate import validate_background

router = APIRouter(prefix="/api")

IMAGE_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
MAX_IMAGE_BYTES = 40 * 1024 * 1024


def _check(bg: Background, p: JobPaths) -> list[dict]:
    meta = read_model_opt(p.meta, Meta)
    issues = validate_background(bg, p, meta.duration if meta else None)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise HTTPException(422, {"message": f"{len(errors)} problem(s) must be fixed before saving",
                                  "issues": [issue_dict(i) for i in issues]})
    return [issue_dict(i) for i in issues]


@router.get("/jobs/{job_id}/background")
def get_background(request: Request, job_id: str) -> dict:
    p = job_paths(request, job_id)
    bg = read_model_opt(p.background, Background)
    if bg is None:
        raise HTTPException(404, "no background.json yet: run the backgrounds stage")
    return bg.model_dump(mode="json")


@router.put("/jobs/{job_id}/background")
def put_background(request: Request, job_id: str, body: Background) -> dict:
    p = job_paths(request, job_id)
    issues = _check(body, p)
    write_model(p.background, body)
    return {"background": body.model_dump(mode="json"), "issues": issues}


@router.post("/jobs/{job_id}/background/scenes/{scene_id}/image")
def upload_scene_image(request: Request, job_id: str, scene_id: str,
                       file: Annotated[UploadFile, File()]) -> dict:
    """Replace a scene's source with an uploaded image (JPEG/PNG/WebP). The scene is locked, so the
    AI art direction keeps the user's choice."""
    from PIL import Image, UnidentifiedImageError

    p = job_paths(request, job_id)
    bg = read_model_opt(p.background, Background)
    if bg is None:
        raise HTTPException(404, "no background.json yet")
    scene = next((s for s in bg.scenes if s.id == scene_id), None)
    if scene is None:
        raise HTTPException(404, f"no scene {scene_id!r}")
    data = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image larger than 40 MB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(422, "not an image (use JPEG, PNG or WebP)") from exc
    if fmt not in IMAGE_FORMATS:
        raise HTTPException(422, f"unsupported image format {fmt}: use JPEG, PNG or WebP")
    safe_id = "".join(c for c in scene_id if c.isalnum() or c in "-_") or "scene"
    name = f"upload-{safe_id}-{hashlib.sha1(data).hexdigest()[:8]}{IMAGE_FORMATS[fmt]}"
    dest = p.bg_assets_dir / name
    p.bg_assets_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    scene.source = BgSource(type="image", path=p.rel(dest), origin="upload",
                            credit=f"uploaded: {file.filename}" if file.filename else "uploaded")
    scene.locked = True
    issues = _check(bg, p)
    write_model(p.background, bg)
    return {"background": bg.model_dump(mode="json"), "issues": issues}
