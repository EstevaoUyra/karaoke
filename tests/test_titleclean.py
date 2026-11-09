from __future__ import annotations

import pytest

from karaoke.lyrics.titleclean import CleanTitle, clean

# (title, uploader, artist, track) → (artist, title, extra). Apart from the first two (real titles
# of the Phase 0 test videos, no lyrics), every band and song here is invented.
CASES: list[tuple[tuple[str, str | None, str | None, str | None], tuple[str | None, str, list[str]]]] = [
    # --- required cases
    (("Sweet Child O Mine - Captain fantastic soundtrack Lyrics", "Tomotion Picture", None, None),
     (None, "Sweet Child O Mine", ["Captain Fantastic"])),
    (("Sacal - Vão Querer Me Internar (Tropical Paradise Riddim)", "Sacal Prod", None, None),
     ("Sacal", "Vão Querer Me Internar", ["Tropical Paradise Riddim"])),
    (("Demo Band - Sing It Loud (Official Video) [HD]", "Demo Band VEVO", None, None),
     ("Demo Band", "Sing It Loud", [])),
    # --- yt-dlp music fields win
    (("Sing It Loud", "Demo Band - Topic", "Demo Band", "Sing It Loud"), ("Demo Band", "Sing It Loud", [])),
    (("Demo Band - Sing It Loud (Official Audio)", "Random Uploads", "Demo Band", "Sing It Loud (Live)"),
     ("Demo Band", "Sing It Loud", ["Live"])),
    (("Demo Band - Sing It Loud", "Somebody", None, "Sing It Loud"), ("Demo Band", "Sing It Loud", [])),
    # --- uploader as artist only for artist channels
    (("Sing It Loud", "Demo Band - Topic", None, None), ("Demo Band", "Sing It Loud", [])),
    (("Sing It Loud (Lyrics)", "Best Lyrics Hub", None, None), (None, "Sing It Loud", [])),
    (("Sing It Loud - Demo Band (Lyrics)", "Demo Band", None, None), ("Demo Band", "Sing It Loud", [])),
    # --- EN noise
    (("Demo Band - Sing It Loud (Official Music Video) ft. Guest Singer", None, None, None),
     ("Demo Band", "Sing It Loud", [])),
    (("Demo Band feat. Guest Singer - Sing It Loud [Official Lyric Video] #pop #2024", None, None, None),
     ("Demo Band", "Sing It Loud", [])),
    (("【MV】Demo Band - Sing It Loud", None, None, None), ("Demo Band", "Sing It Loud", [])),
    (("Demo Band - Sing It Loud [Remastered 2011]", None, None, None), ("Demo Band", "Sing It Loud", [])),
    (("Demo Band - Sing It Loud (Prod. DJ Nobody)", None, None, None), ("Demo Band", "Sing It Loud", [])),
    (("Demo Band - Sing It Loud | Official Video", "Demo Band", None, None), ("Demo Band", "Sing It Loud", [])),
    (('Demo Band "Sing It Loud" (Official Video)', None, None, None), ("Demo Band", "Sing It Loud", [])),
    (("Demo Band - Home Video", None, None, None), ("Demo Band", "Home Video", [])),
    (("Demo Band - Dancing with Shadows", None, None, None), ("Demo Band", "Dancing with Shadows", [])),
    # --- EN qualifiers kept
    (("Demo Band – Sing It Loud (Acoustic Cover)", None, None, None), ("Demo Band", "Sing It Loud", ["Acoustic"])),
    (("Demo Band - Sing It Loud (Live at Example Hall) [4K]", None, None, None),
     ("Demo Band", "Sing It Loud", ["Live at Example Hall"])),
    (("Demo Band - Sing It Loud (Remix) - Official Audio", None, None, None),
     ("Demo Band", "Sing It Loud", ["Remix"])),
    (('Demo Band - Sing It Loud (From "Invented Movie")', None, None, None),
     ("Demo Band", "Sing It Loud", ["Invented Movie"])),
    (("Demo Band - Live Forever Tonight", None, None, None), ("Demo Band", "Live Forever Tonight", [])),
    # --- PT-BR
    (("Banda Inventada - Canção do Mar (Clipe Oficial)", None, None, None), ("Banda Inventada", "Canção do Mar", [])),
    (("Cantora Fictícia - Noite Clara (Ao Vivo)", None, None, None), ("Cantora Fictícia", "Noite Clara", ["Ao Vivo"])),
    (("Dupla Exemplo - Meu Sertão | Áudio Oficial", None, None, None), ("Dupla Exemplo", "Meu Sertão", [])),
    (("Fulano part. Beltrano - Samba Novo (Letra)", None, None, None), ("Fulano", "Samba Novo", [])),
    (("MC Teste - Batidão (Lyric Video) #funk", None, None, None), ("MC Teste", "Batidão", [])),
    (("Grupo Modelo - Amor de Verão (DVD ao vivo em Recife)", None, None, None),
     ("Grupo Modelo", "Amor de Verão", ["DVD Ao Vivo em Recife"])),
    (("Canção Inventada - Trilha Sonora da Novela Exemplo", "Canal de Letras", None, None),
     (None, "Canção Inventada", ["Novela Exemplo"])),
    (("Banda Inventada - Canção do Mar (Tradução / Legendado)", None, None, None),
     ("Banda Inventada", "Canção do Mar", [])),
    (("Banda Inventada - Canção do Mar - Vídeo Oficial", "Banda Inventada Oficial", None, None),
     ("Banda Inventada", "Canção do Mar", [])),
]


@pytest.mark.parametrize(("args", "expected"), CASES, ids=[c[0][0] for c in CASES])
def test_clean(args, expected):
    artist, title, extra = expected
    assert clean(*args) == CleanTitle(artist, title, extra)


def test_lyrics_channel_is_never_the_artist():
    r = clean("Sing It Loud", uploader="Lyrics Daily")
    assert r.artist is None


def test_empty_title_does_not_crash():
    assert clean("").title == ""
