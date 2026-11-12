"""Japanese: whisper hears kana/kanji, the lyrics are romaji (invented lines, not a real song)."""

from __future__ import annotations

import pytest

pytest.importorskip("fugashi")

from karaoke.align.core import align_lyrics  # noqa: E402
from karaoke.align.japanese import is_japanese, romaji_key, romaji_tokens  # noqa: E402
from karaoke.align.tokens import parse_chosen_text  # noqa: E402
from karaoke.models import Transcript, TSegment, TWord, WordOrigin  # noqa: E402


def ja_transcript(chunks: list[tuple[str, float, float]]) -> Transcript:
    """Whisper-like Japanese words: one to three characters each, one segment."""
    words = [TWord(text=t, start=s, end=e, prob=0.9) for t, s, e in chunks]
    return Transcript(model="test", language="ja", segments=[
        TSegment(start=words[0].start, end=words[-1].end, text="".join(w.text for w in words), words=words)])


# 空に なって 歩いて ゆく / 光を 探して いる  ("I become the sky and walk on / I am looking for the light")
HEARD = [("空", 1.0, 1.4), ("に", 1.4, 1.6), ("なっ", 1.7, 2.0), ("て", 2.0, 2.2), ("歩", 2.4, 2.7),
         ("いて", 2.7, 3.0), ("ゆく", 3.1, 3.6),
         ("光", 5.0, 5.4), ("を", 5.4, 5.6), ("探", 5.7, 6.0), ("して", 6.0, 6.3), ("いる", 6.4, 6.9)]
LYRICS = "Sora ni natte aruite yuku\nHikari wo sagashite iru"


def test_romaji_words_keep_whisper_times():
    toks = romaji_tokens(ja_transcript(HEARD))
    assert [t.key for t in toks] == ["sora", "ni", "natte", "aruite", "yuku", "hikari", "o", "sagashite", "iru"]
    natte = toks[2]
    assert (natte.start, natte.end) == pytest.approx((1.7, 2.2))  # "なっ" + "て" joined, small tsu kept
    assert toks[3].start == pytest.approx(2.4) and toks[3].end == pytest.approx(3.0)


def test_romaji_lyrics_align_word_by_word():
    lf, stats = align_lyrics(parse_chosen_text(LYRICS), ja_transcript(HEARD), language="ja")
    words = [w for ln in lf.lines for w in ln.words]
    assert [w.text for w in words] == LYRICS.split()  # the lyrics' own spelling is displayed
    assert all(w.origin in (WordOrigin.matched, WordOrigin.fuzzy) for w in words)
    assert words[2].start == pytest.approx(1.7) and words[5].start == pytest.approx(5.0)
    assert lf.unmatched_audio == []


def test_romaji_spelling_variants_fold():
    assert romaji_key("toukyou") == romaji_key("tokyo") == "tokyo"
    assert romaji_key("wo") == "o" and romaji_key("yuu") == "yu"
    assert is_japanese("花になって") and not is_japanese("hana ni natte")
