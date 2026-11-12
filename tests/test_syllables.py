"""Text fidelity, inferred syllable timing, coverage and melisma guarantees."""
import numpy as np
import pytest

from karaoke.lyrics.syllables import split_syllables
from karaoke.melody.decode import activation_from_contour
from karaoke.melody.pipeline import decode_job
from karaoke.models import Line, LyricsFinal, Melody, Word
from karaoke.validate import validate_melody


@pytest.mark.parametrize('text,lang,want', [
    ('coração', 'pt', ['co', 'ra', 'ção']), ('Olha', 'pt-BR', ['O', 'lha']),
    ('querer', 'pt', ['que', 'rer']), ('saída', 'pt', ['sa', 'í', 'da']),
    ('carro', 'pt', ['car', 'ro']), ('chuva', 'pt', ['chu', 'va']),
    ('love', 'en', ['love']), ('little', 'en', ['lit', 'tle']),
    ('wanted', 'en', ['wan', 'ted']), ('(Hee-hee)', 'en', ['(Hee-', 'hee)']),
    ('R-E-S-P-E-C-T', 'en', ['R-E-S-P-E-C-T']), ('15', 'pt', ['15']),
    ('مرحبا', 'pt', ['مرحبا']), ("don't", 'en', ["don't"]),
    ('mi-mi-mine', 'en', ['mi-', 'mi-', 'mine']), ('', 'en', ['']),
    ("there's", 'en', ["there's"]), ('(there’s)', 'en', ['(there’s)']),
    ("we've", 'en', ["we've"]), ("Imagine's", 'en', ['I', 'ma', "gine's"]),
])
def test_split(text, lang, want):
    parts = split_syllables(text, lang)
    assert parts == want
    assert ''.join(parts) == text


def lyrics(words, lang='en'):
    ws = [Word(text=t, start=s, end=e) for t, s, e in words]
    return LyricsFinal(text_source='user', language=lang,
                       lines=[Line(id='l1', start=min(w.start for w in ws), end=max(w.end for w in ws), words=ws)])


def decode(parts, lf):
    midi = np.concatenate([np.full(round(s/0.01), m) for s, m in parts])
    f0 = 440 * 2 ** ((midi-69)/12)
    act, bin0 = activation_from_contour(f0, np.where(np.isfinite(midi), 0.9, 0))
    res = decode_job(act, bin0, 0.01, None, lf, syllables=True, tuning=0)
    mel = Melody(hop_s=0.01, notes=res.notes, syllables=res.syllables, params={'syllables': True})
    assert validate_melody(mel, lf) == []
    return mel


def test_syllables_split_held_pitch_and_have_no_notes_outside_words():
    lf = lyrics([('coração', 0.2, 1.4), ('vai', 1.6, 1.9)], 'pt')
    mel = decode([(2, 64)], lf)
    assert [s.text for s in mel.syllables] == ['co', 'ra', 'ção', 'vai']
    assert {n.midi for n in mel.notes} == {64}
    assert all(0.2 <= n.start < n.end <= 1.4 or 1.6 <= n.start < n.end <= 1.9 for n in mel.notes)


def test_unheard_word_uses_neighbor_with_zero_confidence():
    lf = lyrics([('love', 0.2, 0.6), ('you', 0.6, 0.9), ('too', 0.9, 1.3)])
    mel = decode([(0.6, 67), (0.3, np.nan), (0.5, 67)], lf)
    silent = [n for n in mel.notes if n.syllable == 1]
    assert silent and {n.midi for n in silent} == {67}
    assert max(n.conf for n in silent) == 0


def test_melisma_and_trailing_rest():
    lf = lyrics([('love', 0, 1.8)])
    mel = decode([(0.6, 60), (0.6, 64), (0.6, np.nan)], lf)
    assert len(mel.syllables) == 1
    assert {n.midi for n in mel.notes} == {60, 64}
    assert max(n.end for n in mel.notes) <= 1.22


def test_tiny_and_out_of_audio_words_keep_every_piece():
    lf = lyrics([('coração', 0.011, 0.019), ('Olha', 2, 2.1)], 'pt')
    mel = decode([(0.1, 65)], lf)
    assert len(mel.syllables) == 5 and len(mel.notes) == 5
    assert all(n.end > n.start for n in mel.notes)


def test_silent_recording_has_explicit_uncertain_placeholder():
    mel = decode([(1, np.nan)], lyrics([('little', 0.1, 0.8)]))
    assert len(mel.syllables) == 2
    assert all(n.conf == 0 and n.midi == 60 for n in mel.notes)


def test_zero_duration_word_fails_instead_of_claiming_coverage():
    with pytest.raises(ValueError, match='positive time bounds'):
        decode([(1, 60)], lyrics([('hi', 0.5, 0.5)]))


def test_validator_catches_missing_links_gaps_and_outside_words():
    lf = lyrics([('love', 0, 1)])
    mel = decode([(1, 60)], lf)
    mel.notes[0].start = -0.1
    assert validate_melody(mel, lf)
    mel.notes[0].start = 0.1
    assert validate_melody(mel, lf)
    mel.notes[0].syllable = None
    assert validate_melody(mel, lf)
    mel.notes = []
    assert validate_melody(mel, lf)


def test_empty_lyrics_forbid_all_notes_and_empty_audio_is_supported():
    lf = LyricsFinal(text_source='user', lines=[])
    act, bin0 = activation_from_contour(np.full(100, 440.), np.ones(100))
    assert decode_job(act, bin0, .01, None, lf, syllables=True).notes == []
    empty = decode_job(act[:0], bin0, .01, None, lyrics([('hello', 0, 1)]), syllables=True)
    assert len(empty.syllables) == 2 and all(n.conf == 0 for n in empty.notes)


def test_v1_documents_remain_readable():
    old = Melody.model_validate({'version': 1, 'hop_s': .01, 'notes': [
        {'start': 1, 'end': 2, 'midi': 60, 'pitch': 60., 'conf': .8}]})
    assert old.syllables == [] and old.notes[0].syllable is None
