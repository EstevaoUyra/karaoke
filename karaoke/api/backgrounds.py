"""background.json: read, save, and replace a scene's image with an upload or a picture found
by ``karaoke.bg_search``."""

from __future__ import annotations

import hashlib
import io
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

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


@router.get("/jobs/{job_id}/structure")
def get_structure(request: Request, job_id: str) -> dict:
    from karaoke.models import Structure

    st = read_model_opt(job_paths(request, job_id).structure, Structure)
    if st is None:
        raise HTTPException(404, "no structure.json yet: run the structure stage")
    return st.model_dump(mode="json")


def _picture_keys(request: Request) -> dict[str, str | None]:
    from karaoke.api.deps import state

    s = state(request).settings
    return {"unsplash": s.unsplash_access_key, "pexels": s.pexels_api_key}


def _scene(p: JobPaths, scene_id: str):
    bg = read_model_opt(p.background, Background)
    if bg is None:
        raise HTTPException(404, "no background.json yet")
    scene = next((s for s in bg.scenes if s.id == scene_id), None)
    if scene is None:
        raise HTTPException(404, f"no scene {scene_id!r}")
    return bg, scene


class ApplyPreset(BaseModel):
    name: str
    replace_locked: bool = False


@router.get("/jobs/{job_id}/background/presets")
def list_presets(request: Request, job_id: str) -> dict:
    """The instant presets, whether each can be made for this job, and whether the song's
    structure is known (without it a preset is a single scene)."""
    from karaoke.bg_presets import PRESETS, available
    from karaoke.bg_search import enabled

    p = job_paths(request, job_id)
    ok = available(p, read_model_opt(p.meta, Meta))
    return {"presets": [{"name": n, "label": label, "available": ok[n]} for n, label in PRESETS.items()],
            "has_structure": p.structure.exists(),
            # where pictures can be searched: Commons always, the others when their key is in .env
            "picture_sources": enabled(_picture_keys(request))}


@router.post("/jobs/{job_id}/background/preset")
def apply_background_preset(request: Request, job_id: str, body: ApplyPreset) -> dict:
    """Replace background.json with a preset. 409 if that would replace locked scenes."""
    from karaoke.api.deps import state
    from karaoke.bg_presets import PresetError, apply_preset

    p = job_paths(request, job_id)
    try:
        bg = apply_preset(body.name, p, state(request).settings.ffmpeg_bin, replace_locked=body.replace_locked)
    except PresetError as exc:
        raise HTTPException(409 if "locked" in str(exc) else 422, str(exc)) from exc
    return {"background": bg.model_dump(mode="json"), "issues": _check(bg, p)}


@router.post("/jobs/{job_id}/background/scenes/{scene_id}/image")
def upload_scene_image(request: Request, job_id: str, scene_id: str,
                       file: Annotated[UploadFile, File()]) -> dict:
    """Replace a scene's source with an uploaded image (JPEG/PNG/WebP). The scene is locked, so the
    AI art direction keeps the user's choice."""
    from PIL import Image, UnidentifiedImageError

    p = job_paths(request, job_id)
    bg, scene = _scene(p, scene_id)
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


class UsePicture(BaseModel):
    pick: str  # "<search name>:<number>", as the search returned it


@router.get("/jobs/{job_id}/background/pictures")
def search_pictures(request: Request, job_id: str, q: Annotated[str, Query(min_length=2, max_length=80)],
                    source: Annotated[list[str] | None, Query()] = None,
                    n: Annotated[int, Query(ge=1, le=20)] = 8) -> dict:
    """Search free pictures (Wikimedia Commons, and Unsplash and Pexels when their keys are set).
    Each candidate has a ``pick`` to hand to the endpoint below. ``notes`` says which sources gave
    nothing and why."""
    from karaoke import bg_search as bs

    p = job_paths(request, job_id)
    try:
        cands, notes = bs.search(q, _picture_keys(request), source or None, n)
    except bs.SearchError as exc:
        raise HTTPException(422, str(exc)) from exc
    name = bs.slug(q)
    if cands:
        bs.save_search(p, name, q, cands)
    return {"query": q, "notes": notes,
            "candidates": [{"pick": f"{name}:{k + 1}", "source": c.provider, "title": c.title, "thumb": c.thumb_url,
                            "size": [c.width, c.height], "license": c.license, "credit": c.credit, "page": c.page}
                           for k, c in enumerate(cands)]}


@router.post("/jobs/{job_id}/background/scenes/{scene_id}/picture")
def use_picture(request: Request, job_id: str, scene_id: str, body: UsePicture) -> dict:
    """Download a picture the search found and make it the scene's source, with its credit. The
    scene is locked, as with an upload. A scene that was not a picture before also gets a
    picture's usual treatment (slow zoom, darkened, softened), for the user to adjust."""
    from karaoke import bg_search as bs

    p = job_paths(request, job_id)
    bg, scene = _scene(p, scene_id)
    try:
        cand = bs.load_pick(p, body.pick)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        _, source = bs.add_candidate(p, cand, _picture_keys(request))
    except Exception as exc:  # noqa: BLE001 — the source refused or sent something that is no picture
        raise HTTPException(502, f"could not fetch the picture ({type(exc).__name__}: {str(exc)[:120]})"
                                 + bs._refused_hint(cand.provider, exc)) from exc
    if scene.source.type != "image":
        scene.motion, scene.dim, scene.blur = "kenburns-in", max(scene.dim, 0.45), max(scene.blur, 8.0)
    scene.source = BgSource(**source)
    scene.locked = True
    issues = _check(bg, p)
    write_model(p.background, bg)
    return {"background": bg.model_dump(mode="json"), "issues": issues}
