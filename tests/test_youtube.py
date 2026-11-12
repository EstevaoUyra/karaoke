"""karaoke publish: the draft (title, description, credits, tags, chapters) and the upload, with a
fake YouTube client: nothing here touches the network."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("googleapiclient")

from karaoke import youtube as yt  # noqa: E402
from karaoke.models import (  # noqa: E402
    Background,
    BgSource,
    Job,
    JobSettings,
    Meta,
    Scene,
    Section,
    Structure,
    YouTubeUpload,
)
from karaoke.store import read_model, write_model  # noqa: E402


@pytest.fixture
def job(demo_no_audio):
    p = demo_no_audio
    p.final_video.parent.mkdir(parents=True, exist_ok=True)
    p.final_video.write_bytes(b"not really a video")
    write_model(p.meta, Meta(url="https://www.youtube.com/watch?v=DEMO0000000", video_id="DEMO0000000",
                             title="DEMO BAND - SING IT LOUD (com MC Um, MC Dois)", duration=60.0,
                             clean_artist="DEMO BAND", clean_title="SING IT LOUD",
                             clean_extra=["Com Mc Um, Mc Dois"]))
    write_model(p.job, Job(id=p.id, settings=JobSettings(language="pt")))
    write_model(p.structure, Structure(duration=60.0, sections=[
        Section(start=0.0, end=6.0, label="intro", energy=0.2, vocal=0.0),
        Section(start=6.0, end=20.0, label="verse", energy=0.5, vocal=0.9),
        Section(start=20.0, end=34.0, label="chorus", energy=0.9, vocal=1.0),
        Section(start=34.0, end=40.0, label="chorus", energy=0.9, vocal=1.0),
        Section(start=40.0, end=52.0, label="verse", energy=0.5, vocal=0.9),
        Section(start=52.0, end=60.0, label="outro", energy=0.2, vocal=0.0),
    ]))
    p.bg_assets_dir.mkdir(parents=True, exist_ok=True)
    p.bg_credits.write_text(json.dumps({
        "img-a.jpg": {"page": "https://commons.wikimedia.org/wiki/File:Sea.jpg", "credit": "Ana Lima, Sea at dusk",
                      "license": "CC BY-SA 4.0", "provider": "wikimedia", "title": "Sea at dusk"},
        "img-b.jpg": {"page": "https://unsplash.com/photos/x", "credit": "Photo by Bo on Unsplash",
                      "license": "Unsplash License", "provider": "unsplash"},
        "img-unused.jpg": {"credit": "Nobody", "license": "CC BY 4.0", "provider": "wikimedia"},
    }))
    scenes = [
        Scene(id="s1", start=0, end=20, source=BgSource(type="generated", value="aurora:#000000,#112233")),
        Scene(id="s2", start=20, end=40, source=BgSource(type="image", path="assets/bg/img-a.jpg",
                                                         origin="wikimedia"), dim=0.4, blur=10, motion="kenburns-in"),
        Scene(id="s3", start=40, end=50, source=BgSource(type="image", path="assets/bg/img-b.jpg", origin="unsplash")),
        Scene(id="s4", start=50, end=60, source=BgSource(type="image", path="assets/bg/img-a.jpg",
                                                         origin="wikimedia"), motion="pan-left"),
    ]
    write_model(p.background, Background(scenes=scenes))
    return p


def test_draft_title_credits_chapters_and_tags(job):
    y, written = yt.draft(job)
    assert written and job.youtube.exists()
    assert y.title == "Demo Band - Sing It Loud (part. MC Um, MC Dois) | Karaokê (Playback com Letra)"
    d = y.description
    assert d.splitlines()[1] == "Versão original: https://www.youtube.com/watch?v=DEMO0000000"
    # the short intro joins the first verse; the two chorus sections merge; the outro is too short alone
    assert "0:00 Verso 1\n0:20 Refrão\n0:40 Verso 2" in d
    # only the pictures on screen, each once, with author, licence link, source and what was changed
    assert '"Sea at dusk" – Ana Lima, CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)' in d
    assert "via Wikimedia Commons: https://commons.wikimedia.org/wiki/File:Sea.jpg" in d
    assert "recortada, desfocada, escurecida, em movimento" in d
    assert d.count("Sea at dusk") == 1 and "Nobody" not in d
    assert "Photo by Bo on Unsplash, Unsplash License (https://unsplash.com/license)" in d
    assert "mesma licença" in d  # share-alike
    assert yt.tags_length(y.tags) <= yt.TAGS_MAX and "demo band sing it loud karaokê" in y.tags
    assert y.privacy == "private" and y.language == "pt-BR" and not y.made_for_kids
    assert yt.check(y) == []


def test_draft_keeps_user_edits_unless_redrafted(job):
    y, _ = yt.draft(job)
    write_model(job.youtube, y.model_copy(update={"title": "My own title", "video_id": "abc123def45"}))
    assert yt.draft(job)[0].title == "My own title"
    redone, written = yt.draft(job, redo=True)
    assert written and redone.title != "My own title" and redone.video_id == "abc123def45"


def test_licences():
    assert yt.licence_url("CC BY 3.0") == "https://creativecommons.org/licenses/by/3.0/"
    assert yt.licence_url("CC BY-SA 2.0") == "https://creativecommons.org/licenses/by-sa/2.0/"
    assert yt.licence_url("CC0") == "https://creativecommons.org/publicdomain/zero/1.0/"
    assert yt._licence_problem("CC BY-ND 4.0") and yt._licence_problem("CC BY-NC-SA 4.0")
    assert yt._licence_problem("CC BY 4.0") is None


def test_check_finds_what_youtube_refuses():
    y = YouTubeUpload(title="x" * 101, description="a <b>", tags=["t" * 300, "u" * 300])
    errs = " ".join(yt.check(y))
    assert "title is 101" in errs and '"<" or ">"' in errs and "tags add up" in errs
    assert yt.tags_length(["a b", "c"]) == 3 + 2 + 1 + 1


class _Status:
    def __init__(self, f):
        self.f = f

    def progress(self):
        return self.f


class _Exec:
    def __init__(self, fn):
        self.fn = fn

    def execute(self):
        return self.fn()


class FakeYouTube:
    """Just enough of googleapiclient's YouTube resource."""

    def __init__(self, fail_chunks=0):
        self.calls: list[tuple] = []
        self.fail_chunks = fail_chunks
        self.playlists_mine: list[dict] = []

    def videos(self):
        return self

    def thumbnails(self):
        fake = self

        class T:
            def set(self, videoId, media_body):
                fake.calls.append(("thumbnail", videoId))
                return _Exec(lambda: {})
        return T()

    def playlists(self):
        fake = self

        class P:
            def list(self, **kw):
                return _Exec(lambda: {"items": fake.playlists_mine})

            def list_next(self, req, res):
                return None

            def insert(self, part, body):
                fake.calls.append(("playlist", body["snippet"]["title"]))
                return _Exec(lambda: {"id": "PL1"})
        return P()

    def playlistItems(self):  # noqa: N802 — the API's name
        fake = self

        class I:  # noqa: E742
            def insert(self, part, body):
                fake.calls.append(("item", body["snippet"]["playlistId"], body["snippet"]["resourceId"]["videoId"]))
                return _Exec(lambda: {})
        return I()

    def insert(self, part, body, media_body):
        self.calls.append(("insert", body))
        fake = self

        class R:
            n = 0

            def next_chunk(self):
                if fake.fail_chunks:
                    fake.fail_chunks -= 1
                    raise OSError("connection reset")
                R.n += 1
                if R.n < 3:
                    return _Status(R.n / 3), None
                return None, {"id": "VID00000001", "status": {"uploadStatus": "uploaded"}}
        return R()

    def update(self, part, body):
        self.calls.append(("update", body))
        return _Exec(lambda: body)


def test_upload_saves_the_id_then_thumbnail_and_playlist(job):
    yt.draft(job)
    fake = FakeYouTube(fail_chunks=2)
    logs: list[str] = []
    y = yt.upload(job, fake, log=logs.append, sleep=lambda s: None)
    assert y.video_id == "VID00000001" and read_model(job.youtube, YouTubeUpload).video_id == "VID00000001"
    body = fake.calls[0][1]
    assert body["status"] == {"privacyStatus": "private", "selfDeclaredMadeForKids": False,
                              "containsSyntheticMedia": False}
    assert body["snippet"]["categoryId"] == "10" and body["snippet"]["defaultLanguage"] == "pt-BR"
    assert ("thumbnail", "VID00000001") in fake.calls or not y.thumbnail  # no ffmpeg frame from a fake video
    assert ("playlist", "Karaokê") in fake.calls and ("item", "PL1", "VID00000001") in fake.calls
    assert sum("retrying" in m for m in logs) == 2
    with pytest.raises(ValueError, match="already uploaded"):
        yt.upload(job, fake, sleep=lambda s: None)


def test_upload_refuses_a_draft_youtube_would_refuse(job):
    y, _ = yt.draft(job)
    write_model(job.youtube, y.model_copy(update={"title": "x" * 120}))
    with pytest.raises(ValueError, match="title is 120"):
        yt.upload(job, FakeYouTube())


def test_update_sends_edits_and_privacy(job):
    y, _ = yt.draft(job)
    write_model(job.youtube, y.model_copy(update={"video_id": "VID00000001", "title": "New title"}))
    fake = FakeYouTube()
    out = yt.update(job, fake, privacy="public", log=lambda m: None)
    body = fake.calls[0][1]
    assert body["id"] == "VID00000001" and body["snippet"]["title"] == "New title"
    assert body["status"]["privacyStatus"] == "public" and out.privacy == "public"
    assert read_model(job.youtube, YouTubeUpload).privacy == "public"
