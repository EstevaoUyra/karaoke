"""Find pictures for a background (the ``karaoke-backgrounds`` skill, or a person at the CLI).

Three sources, all free to use: Wikimedia Commons (no key; the only one with pictures of real
artists and places by name), Unsplash and Pexels (stock photographs; each needs a key in ``.env``).

- ``search``: ask the sources, keep the pictures that are large enough, landscape and free to use.
- ``candidate_sheet``: the candidates' thumbnails in one numbered picture, each measured for how
  much it would have to be darkened to sit behind the lyrics.
- ``save_search`` / ``load_pick``: a search is kept in the job's cache, so a candidate is fetched
  by its number with the credit the source gave, never retyped.
- ``add_candidate``: download the picked one into ``assets/bg`` and record its credit.
"""

from __future__ import annotations

import html
import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from karaoke.store import JobPaths

CLIENT = "karaoke-backgrounds/1.0"
APP_NAME = "karaoke"  # Unsplash asks for it on links back (utm_source)
PROVIDERS = ("wikimedia", "unsplash", "pexels")
KEY_NAMES = {"unsplash": "UNSPLASH_ACCESS_KEY", "pexels": "PEXELS_API_KEY"}
MIN_WIDTH = 1600  # narrower than this and a 1080p background goes soft
MIN_ASPECT = 1.2  # width / height: a portrait picture loses most of itself in a 16:9 crop
IMAGE_WIDTH = 1920
THUMB_WIDTH = 500  # Commons serves only a few fixed thumbnail widths; 500 is one, 640 is not
MAX_THUMB_BYTES = 5 * 1024 * 1024
CELL = (480, 270)
SHEET_COLS = 4


@dataclass(frozen=True)
class Candidate:
    provider: str
    id: str
    title: str  # what the source says the picture shows
    width: int
    height: int
    thumb_url: str  # small, to look at
    image_url: str  # about 1920 px wide, to use
    page: str  # the picture's page at the source
    credit: str
    license: str
    author_url: str | None = None
    track_url: str | None = None  # Unsplash: to be requested once when the picture is used


class SearchError(Exception):
    pass


def user_agent() -> str:
    """Who is asking. Wikimedia's robot policy wants a contact in it (``KARAOKE_CONTACT`` in .env);
    without one it refuses most searches and pictures (HTTP 403)."""
    import httpx

    from karaoke.config import get_settings

    contact = get_settings().contact
    return f"{CLIENT} ({contact}) python-httpx/{httpx.__version__}" if contact else CLIENT


def _with_params(url: str, **params: str | int) -> str:
    """``url`` with the query parameters set (the ones already there are kept)."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update({k: str(v) for k, v in params.items()})
    return urlunsplit(parts._replace(query=urlencode(query)))


def _plain(text: str | None) -> str:
    """The text of a snippet of HTML, on one line."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text or "")).split())


def _fits(width: int, height: int, landscape: bool) -> bool:
    return width >= MIN_WIDTH and (not landscape or width / max(1, height) >= MIN_ASPECT)


# --------------------------------------------------------------------------- the sources


def free_licence(name: str) -> bool:
    """Is this Commons licence one that lets the picture be used with a credit (or with none)?"""
    n = name.lower().replace("-", " ")
    if re.search(r"\b(nc|nd)\b", n):
        return False
    return n.startswith(("public domain", "pd", "cc0", "cc by", "no restrictions"))


def wikimedia_request(query: str, n: int) -> tuple[str, dict, dict]:
    return ("https://commons.wikimedia.org/w/api.php", {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",  # bitmap: no scanned books, no PDFs
        "gsrnamespace": 6, "gsrlimit": min(50, n * 3),  # many are dropped for size, shape or licence
        "prop": "imageinfo", "iiprop": "url|extmetadata|size|mime", "iiurlwidth": IMAGE_WIDTH,
        "iiextmetadatafilter": "LicenseShortName|Artist|ObjectName",
    }, {})


def wikimedia_parse(data: dict, n: int, landscape: bool = True) -> list[Candidate]:
    out = []
    pages = sorted((data.get("query") or {}).get("pages", {}).values(), key=lambda pg: pg.get("index", 0))
    for pg in pages:
        info = (pg.get("imageinfo") or [{}])[0]
        meta = {k: _plain(v.get("value")) for k, v in (info.get("extmetadata") or {}).items()}
        licence = meta.get("LicenseShortName", "")
        if info.get("mime") not in ("image/jpeg", "image/png", "image/webp") or not free_licence(licence):
            continue
        if not _fits(info.get("width", 0), info.get("height", 0), landscape) or not info.get("thumburl"):
            continue
        image = info["thumburl"].split("?")[0]  # the query string is only their click tracking
        marker = f"/{IMAGE_WIDTH}px-"
        title = meta.get("ObjectName") or re.sub(r"^File:|\.\w+$", "", pg.get("title", ""))
        out.append(Candidate(
            provider="wikimedia", id=str(pg.get("pageid", pg.get("title", ""))), title=title,
            width=info["width"], height=info["height"],
            thumb_url=image.replace(marker, f"/{THUMB_WIDTH}px-") if marker in image else image,
            image_url=image, page=info.get("descriptionurl", ""),
            credit=f"{meta.get('Artist') or 'Unknown author'}, {title}", license=licence,
        ))
        if len(out) == n:
            break
    return out


def unsplash_request(query: str, n: int, key: str, landscape: bool = True) -> tuple[str, dict, dict]:
    params = {"query": query, "per_page": min(30, n), "content_filter": "high"}
    if landscape:
        params["orientation"] = "landscape"
    return ("https://api.unsplash.com/search/photos", params,
            {"Authorization": f"Client-ID {key}", "Accept-Version": "v1"})


def unsplash_parse(data: dict, n: int, landscape: bool = True) -> list[Candidate]:
    out = []
    referral = {"utm_source": APP_NAME, "utm_medium": "referral"}
    for ph in data.get("results", []):
        raw = (ph.get("urls") or {}).get("raw")
        if not raw or not _fits(ph.get("width", 0), ph.get("height", 0), landscape):
            continue
        user, links = ph.get("user") or {}, ph.get("links") or {}
        profile = (user.get("links") or {}).get("html")
        out.append(Candidate(
            provider="unsplash", id=str(ph.get("id", "")),
            title=ph.get("alt_description") or ph.get("description") or "",
            width=ph["width"], height=ph["height"],
            # their picture server resizes on request; the parameters already in `raw` must stay
            thumb_url=_with_params(raw, w=THUMB_WIDTH, fit="max", fm="jpg", q=70),
            image_url=_with_params(raw, w=IMAGE_WIDTH, fit="max", fm="jpg", q=85),
            page=_with_params(links["html"], **referral) if links.get("html") else "",
            credit=f"Photo by {user.get('name') or 'an Unsplash photographer'} on Unsplash",
            license="Unsplash License",
            author_url=_with_params(profile, **referral) if profile else None,
            track_url=links.get("download_location"),
        ))
        if len(out) == n:
            break
    return out


def pexels_request(query: str, n: int, key: str, landscape: bool = True) -> tuple[str, dict, dict]:
    params = {"query": query, "per_page": min(80, n), "size": "small"}  # "small" is 4 megapixels or more
    if landscape:
        params["orientation"] = "landscape"
    return ("https://api.pexels.com/v1/search", params, {"Authorization": key})


def pexels_parse(data: dict, n: int, landscape: bool = True) -> list[Candidate]:
    out = []
    for ph in data.get("photos", []):
        src = ph.get("src") or {}
        if not src.get("large2x") or not _fits(ph.get("width", 0), ph.get("height", 0), landscape):
            continue
        out.append(Candidate(
            provider="pexels", id=str(ph.get("id", "")), title=ph.get("alt") or "",
            width=ph["width"], height=ph["height"],
            thumb_url=src.get("medium") or src["large2x"],
            image_url=src["large2x"],  # at most 1880 x 1300: the largest size the API documents
            page=ph.get("url", ""),
            credit=f"Photo by {ph.get('photographer') or 'a Pexels photographer'} on Pexels",
            license="Pexels License", author_url=ph.get("photographer_url"),
        ))
        if len(out) == n:
            break
    return out


# --------------------------------------------------------------------------- search

NO_CONTACT = ("Wikimedia refuses automated clients that do not say who runs them: "
              "set KARAOKE_CONTACT (an email or a URL of yours) in .env")


def _refused_hint(name: str, exc: Exception) -> str:
    from karaoke.config import get_settings

    refused = "403" in str(exc) or "429" in str(exc)
    return f". {NO_CONTACT}" if name == "wikimedia" and refused and not get_settings().contact else ""



def _get_json(url: str, params: dict, headers: dict) -> dict:
    import httpx

    r = httpx.get(url, params=params, headers={"User-Agent": user_agent(), **headers}, timeout=30.0,
                  follow_redirects=True)
    r.raise_for_status()
    return r.json()


def enabled(keys: dict[str, str | None]) -> dict[str, bool]:
    """Which sources can be asked: Commons always, the others when their key is set."""
    return {name: name == "wikimedia" or bool(keys.get(name)) for name in PROVIDERS}


def search(query: str, keys: dict[str, str | None], providers: list[str] | None = None, n: int = 8,
           landscape: bool = True, get_json=None) -> tuple[list[Candidate], list[str]]:
    """Up to ``n`` candidates from each source, mixed so that no source comes last, and notes
    about the sources that gave nothing (no key, an error, no result)."""
    get_json = get_json or _get_json
    on = enabled(keys)
    asked = providers or [name for name in PROVIDERS if on[name]]
    notes = [f"{name}: off (set {KEY_NAMES[name]} in .env)" for name in PROVIDERS if not on[name] and not providers]
    found: list[list[Candidate]] = []
    for name in asked:
        if name not in PROVIDERS:
            raise SearchError(f"unknown source {name!r} (known: {', '.join(PROVIDERS)})")
        if not on[name]:
            raise SearchError(f"{name} needs {KEY_NAMES[name]} in .env")
        try:
            if name == "wikimedia":
                got = wikimedia_parse(get_json(*wikimedia_request(query, n)), n, landscape)
            elif name == "unsplash":
                got = unsplash_parse(get_json(*unsplash_request(query, n, keys[name], landscape)), n, landscape)
            else:
                got = pexels_parse(get_json(*pexels_request(query, n, keys[name], landscape)), n, landscape)
        except Exception as exc:  # noqa: BLE001 — one source failing must not lose the others
            hint = _refused_hint(name, exc)
            notes.append(f"{name}: {hint[2:]}" if hint else f"{name}: failed ({type(exc).__name__}: {str(exc)[:120]})")
            continue
        if not got:
            notes.append(f"{name}: nothing usable for this query")
        found.append(got)
    mixed = [c for k in range(max((len(g) for g in found), default=0)) for g in found if k < len(g) for c in [g[k]]]
    return mixed, notes


# --------------------------------------------------------------------------- looking at them


def _download(url: str, limit: int) -> bytes:
    import httpx

    if not url.lower().startswith("https://"):
        raise ValueError("only https:// pictures are fetched")
    with httpx.stream("GET", url, timeout=30.0, follow_redirects=True, headers={"User-Agent": user_agent()}) as r:
        r.raise_for_status()
        data = b""
        for chunk in r.iter_bytes():
            data += chunk
            if len(data) > limit:
                raise ValueError("picture too large")
        return data


def candidate_sheet(cands: list[Candidate], out: Path, download=None) -> list[dict]:
    """One picture with every candidate's thumbnail, cropped as the video would crop it and
    numbered from 1. Returns one dict per candidate: how bright and busy it is where the lyrics
    go, the ``dim`` it would need, and its average colour."""
    from PIL import Image, ImageDraw, ImageOps

    from karaoke.bg_tools import MAX_MEAN_LUMA, MAX_P95_LUMA, legibility

    download = download or (lambda url: _download(url, MAX_THUMB_BYTES))
    w, h = CELL
    cols = min(SHEET_COLS, max(1, len(cands)))
    sheet = Image.new("RGB", (cols * w, -(-len(cands) // cols) * h), "black")
    info = []
    for k, c in enumerate(cands):
        try:
            with Image.open(io.BytesIO(download(c.thumb_url))) as raw:
                im = ImageOps.fit(raw.convert("RGB"), CELL)
        except Exception as exc:  # noqa: BLE001 — a thumbnail that will not load is reported, not fatal
            im = Image.new("RGB", CELL, "#303030")
            info.append({"error": f"thumbnail failed ({type(exc).__name__}: {str(exc)[:60]})"
                                  + _refused_hint(c.provider, exc)})
        else:
            leg = legibility(im)
            r, g, b = im.resize((1, 1)).getpixel((0, 0))
            dim = max(0.0, 1 - MAX_MEAN_LUMA / max(leg["mean"], 1e-6), 1 - MAX_P95_LUMA / max(leg["p95"], 1e-6))
            info.append({"mean": leg["mean"], "detail": leg["detail"], "dim_needed": round(dim, 2),
                         "color": f"#{r:02x}{g:02x}{b:02x}"})
        ImageDraw.Draw(im).text((8, 6), f"{k + 1}  {c.provider}", fill="white", stroke_width=2,
                                stroke_fill="black", font_size=22)
        sheet.paste(im, ((k % cols) * w, (k // cols) * h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return info


# --------------------------------------------------------------------------- keeping and picking


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "search"


def save_search(p: JobPaths, name: str, query: str, cands: list[Candidate]) -> Path:
    p.bg_search_dir.mkdir(parents=True, exist_ok=True)
    dest = p.bg_search_dir / f"{slug(name)}.json"
    dest.write_text(json.dumps({"query": query, "candidates": [asdict(c) for c in cands]}, indent=2,
                               ensure_ascii=False) + "\n")
    return dest


def load_pick(p: JobPaths, pick: str) -> Candidate:
    """The candidate named ``<search>:<number>``, as ``bg-search`` printed it."""
    name, _, number = pick.rpartition(":")
    path = p.bg_search_dir / f"{slug(name)}.json"
    if not name or not number.isdigit() or not path.exists():
        raise ValueError(f"no such pick {pick!r}: use <search name>:<number> from bg-search")
    cands = json.loads(path.read_text())["candidates"]
    if not 1 <= int(number) <= len(cands):
        raise ValueError(f"{name} has candidates 1 to {len(cands)}")
    return Candidate(**cands[int(number) - 1])


def add_candidate(p: JobPaths, c: Candidate, keys: dict[str, str | None], download=None, ping=None) -> tuple[Path, dict]:
    """Download a candidate into ``assets/bg`` with its credit. Returns the file and the
    ``source`` object for its scene in background.json."""
    from karaoke.bg_tools import add_image

    dest = add_image(p, c.image_url, c.credit, c.license, c.page, download=download)
    credits = json.loads(p.bg_credits.read_text())
    credits[dest.name].update({"provider": c.provider, "author_url": c.author_url, "title": c.title})
    p.bg_credits.write_text(json.dumps(credits, indent=2, ensure_ascii=False) + "\n")
    if c.track_url and keys.get("unsplash"):
        # Unsplash's API terms: tell them when one of their pictures is put to use
        try:
            (ping or _get_json)(c.track_url, {}, {"Authorization": f"Client-ID {keys['unsplash']}"})
        except Exception:  # noqa: BLE001, S110 — a courtesy count; the picture is already here
            pass
    return dest, {"type": "image", "path": p.rel(dest), "origin": c.provider, "url": c.page,
                  "credit": c.credit, "license": c.license}
