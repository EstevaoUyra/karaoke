"""Picture search for backgrounds: what each source's answer becomes, the sheet, and picking."""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from karaoke import bg_search as bs

KEYS = {"unsplash": "u-key", "pexels": "p-key"}


def commons_page(index, title, width, height, licence, mime="image/jpeg", artist='<a href="//x">Ana  Lima</a>'):
    name = title.replace(" ", "_")
    return {
        "pageid": 100 + index, "title": f"File:{title}.jpg", "index": index,
        "imageinfo": [{
            "width": width, "height": height, "mime": mime,
            "thumburl": f"https://thumb.example.invalid/thumb/a/ab/{name}.jpg/1920px-{name}.jpg?utm_source=x",
            "descriptionurl": f"https://commons.example.invalid/wiki/File:{name}.jpg",
            "extmetadata": {"LicenseShortName": {"value": licence}, "Artist": {"value": artist},
                            "ObjectName": {"value": title}},
        }],
    }


COMMONS = {"query": {"pages": {
    "3": commons_page(3, "Third by rank", 3000, 2000, "CC BY-SA 4.0"),
    "1": commons_page(1, "Sea at dusk", 4000, 2600, "CC BY 2.0"),
    "2": commons_page(2, "Too small", 800, 600, "CC0"),
    "4": commons_page(4, "A tall one", 2000, 3000, "CC0"),
    "5": commons_page(5, "Not free", 4000, 2600, "CC BY-NC 2.0"),
    "6": commons_page(6, "A scan", 4000, 2600, "Public domain", mime="application/pdf"),
}}}

UNSPLASH = {"results": [{
    "id": "abc", "alt_description": "a street at night", "description": None, "width": 6000, "height": 4000,
    "urls": {"raw": "https://images.example.invalid/photo-1?ixid=XYZ&ixlib=rb"},
    "links": {"html": "https://unsplash.example.invalid/photos/abc",
              "download_location": "https://api.example.invalid/photos/abc/download?ixid=XYZ"},
    "user": {"name": "Rui Sato", "links": {"html": "https://unsplash.example.invalid/@rui"}},
}, {"id": "small", "width": 900, "height": 600, "urls": {"raw": "https://images.example.invalid/photo-2?ixid=Q"}}]}

PEXELS = {"photos": [{
    "id": 77, "width": 5000, "height": 3300, "url": "https://pexels.example.invalid/photo/77/",
    "photographer": "Lia Moreau", "photographer_url": "https://pexels.example.invalid/@lia", "alt": "a crowd at a concert",
    "src": {"large2x": "https://images.example.invalid/77.jpeg?auto=compress&dpr=2&h=650&w=940",
            "medium": "https://images.example.invalid/77.jpeg?auto=compress&h=350"},
}]}


def fake_get(url, params, headers):
    if "commons" in url:
        return COMMONS
    if "unsplash" in url:
        assert headers["Authorization"] == "Client-ID u-key" and params["orientation"] == "landscape"
        return UNSPLASH
    assert headers["Authorization"] == "p-key"
    return PEXELS


def png(colour, size=(500, 300)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "PNG")
    return buf.getvalue()


def test_commons_keeps_large_landscape_free_pictures_in_rank_order():
    got = bs.wikimedia_parse(COMMONS, 8)
    assert [c.title for c in got] == ["Sea at dusk", "Third by rank"]
    c = got[0]
    assert c.credit == "Ana Lima, Sea at dusk" and c.license == "CC BY 2.0"
    assert c.image_url.endswith("/1920px-Sea_at_dusk.jpg") and c.thumb_url.endswith("/500px-Sea_at_dusk.jpg")
    assert "A tall one" in [c.title for c in bs.wikimedia_parse(COMMONS, 8, landscape=False)]


@pytest.mark.parametrize(("name", "free"), [
    ("CC BY-SA 4.0", True), ("CC0", True), ("Public domain", True), ("PD-USGov", True),
    ("CC BY-NC 2.0", False), ("CC BY-ND 3.0", False), ("Fair use", False), ("", False),
])
def test_only_licences_that_allow_use_pass(name, free):
    assert bs.free_licence(name) is free


def test_unsplash_keeps_its_tracking_parameters_and_credits_the_photographer():
    (c,) = bs.unsplash_parse(UNSPLASH, 8)
    assert c.image_url.startswith("https://images.example.invalid/photo-1?") and "ixid=XYZ" in c.image_url
    assert "w=1920" in c.image_url and "w=500" in c.thumb_url
    assert c.credit == "Photo by Rui Sato on Unsplash" and c.license == "Unsplash License"
    assert "utm_source=karaoke" in c.page and "utm_medium=referral" in c.author_url
    assert c.track_url.endswith("/download?ixid=XYZ")


def test_pexels_gives_the_documented_large_size():
    (c,) = bs.pexels_parse(PEXELS, 8)
    assert "dpr=2" in c.image_url and c.credit == "Photo by Lia Moreau on Pexels" and c.title == "a crowd at a concert"


def test_search_mixes_the_sources_and_says_which_are_off():
    cands, notes = bs.search("night", KEYS, get_json=fake_get)
    assert [c.provider for c in cands] == ["wikimedia", "unsplash", "pexels", "wikimedia"] and notes == []
    cands, notes = bs.search("night", {}, get_json=fake_get)
    assert {c.provider for c in cands} == {"wikimedia"}
    assert any("UNSPLASH_ACCESS_KEY" in n for n in notes) and any("PEXELS_API_KEY" in n for n in notes)
    with pytest.raises(bs.SearchError, match="PEXELS_API_KEY"):
        bs.search("night", {}, ["pexels"], get_json=fake_get)


def test_one_source_failing_keeps_the_others():
    def flaky(url, params, headers):
        if "pexels" in url:
            raise RuntimeError("429 Too Many Requests")
        return fake_get(url, params, headers)

    cands, notes = bs.search("night", KEYS, get_json=flaky)
    assert {c.provider for c in cands} == {"wikimedia", "unsplash"}
    assert len(notes) == 1 and notes[0].startswith("pexels: failed")


def test_sheet_numbers_the_candidates_and_says_how_much_to_darken(tmp_path):
    cands, _ = bs.search("night", KEYS, get_json=fake_get)
    pictures = iter([png((230, 230, 230)), png((20, 30, 60)), b"<html>refused</html>", png((90, 90, 90))])
    info = bs.candidate_sheet(cands, tmp_path / "sheet.jpg", download=lambda url: next(pictures))
    with Image.open(tmp_path / "sheet.jpg") as sheet:
        assert sheet.size == (4 * bs.CELL[0], bs.CELL[1])
    assert info[0]["dim_needed"] > 0.5 and info[1]["dim_needed"] == 0.0  # a white picture, a dark one
    assert "thumbnail failed" in info[2]["error"] and info[1]["color"] == "#141e3c"


def test_a_pick_is_fetched_with_the_credit_the_source_gave(demo_no_audio):
    p = demo_no_audio
    cands, _ = bs.search("night", KEYS, get_json=fake_get)
    bs.save_search(p, "Night City!", "night", cands)
    c = bs.load_pick(p, "night-city:2")
    assert c.provider == "unsplash"
    pinged = []
    dest, source = bs.add_candidate(p, c, KEYS, download=lambda url: png((10, 20, 30)),
                                    ping=lambda url, params, headers: pinged.append((url, headers)))
    assert dest.exists() and source["path"] == p.rel(dest)
    assert source == {"type": "image", "path": p.rel(dest), "origin": "unsplash", "url": c.page,
                      "credit": "Photo by Rui Sato on Unsplash", "license": "Unsplash License"}
    entry = json.loads(p.bg_credits.read_text())[dest.name]
    assert entry["provider"] == "unsplash" and entry["author_url"] == c.author_url
    assert pinged == [(c.track_url, {"Authorization": "Client-ID u-key"})]  # Unsplash is told once
    for bad in ("night-city:9", "nothing:1", "night-city"):
        with pytest.raises(ValueError):
            bs.load_pick(p, bad)
