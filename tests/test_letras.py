from __future__ import annotations

import json

import httpx

from karaoke.lyrics import letras

DOCS = {"response": {"docs": [
    {"id": "art1", "art": "Banda Demo", "dns": "banda-demo", "txt": "Banda Demo"},
    {"id": "mus11", "art": "Banda Demo", "dns": "banda-demo", "txt": "Canção do Teste", "url": "11"},
    {"id": "mus12", "art": "Outra Banda", "dns": "outra-banda", "txt": "Canção do Teste", "url": "12"},
    {"id": "mus13", "art": "Banda Demo", "dns": "banda-demo", "txt": "Outra Música", "url": "13"},
]}}


def test_parse_jsonp_and_pick_the_matching_song():
    docs = letras.parse_suggestions("LetrasSug(" + json.dumps(DOCS) + ")")
    hit = letras.best_hit(docs, "Cancao do teste", "Banda Demo")
    assert hit is not None and hit.url == "https://www.letras.mus.br/banda-demo/11/"
    assert letras.best_hit(docs, "Uma Canção Inexistente", "Banda Demo") is None
    assert letras.best_hit(docs, "Canção do Teste", "Ninguém") is None


def test_find_queries_the_suggestion_endpoint(monkeypatch):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["q"] = req.url.params["q"]
        seen["ua"] = req.headers["user-agent"]
        return httpx.Response(200, text="LetrasSug(" + json.dumps(DOCS) + ")")

    monkeypatch.setattr(letras, "TRANSPORT", httpx.MockTransport(handler))
    hit = letras.find("Canção do Teste", "Banda Demo")
    assert hit is not None and hit.title == "Canção do Teste"
    assert seen["q"] == "Banda Demo Canção do Teste" and "karaoke" in seen["ua"]


def test_lyrics_fetch_saves_the_letras_pointer(demo, store, monkeypatch):
    import logging

    from karaoke.config import get_settings
    from karaoke.lyrics import lrclib
    from karaoke.models import Meta, StageName
    from karaoke.pipeline import StageContext
    from karaoke.stages import lyrics_fetch
    from karaoke.store import write_model

    monkeypatch.setattr(lrclib, "TRANSPORT", httpx.MockTransport(lambda r: httpx.Response(404, json={})))
    monkeypatch.setattr(letras, "TRANSPORT", httpx.MockTransport(
        lambda r: httpx.Response(200, text="LetrasSug(" + json.dumps(DOCS) + ")")))
    write_model(demo.meta, Meta(title="Banda Demo - Canção do Teste", duration=24.0,
                                clean_title="Canção do Teste", clean_artist="Banda Demo"))
    demo.lyrics_source("user.txt").unlink(missing_ok=True)
    ctx = StageContext(store.load(demo.id), demo, store, get_settings(), StageName.lyrics_fetch,
                       logging.getLogger("test"))
    lyrics_fetch.run(ctx)
    saved = json.loads(demo.lyrics_source("letras.json").read_text())
    assert saved["url"] == "https://www.letras.mus.br/banda-demo/11/"
