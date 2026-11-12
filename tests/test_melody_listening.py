"""A/B blinding, durable verdicts and fair audio generation for the local review tool."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


@pytest.fixture
def review(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'spikes'))
    return importlib.import_module('melody_listening')


@pytest.fixture
def pack(tmp_path, review):
    pack = tmp_path/'review'
    pack.mkdir()
    review.write(pack/'manifest.json', dict(id='pack1', version=1, cases=[dict(id='c01', has_reference=True)]))
    review.write(pack/'key.json', dict(cases={'c01': dict(mapping={'A': 'before', 'B': 'final'}, has_revised=False)}))
    review.write(pack/'answers.json', dict(pack_id='pack1', revision=0, answers={}, history=[], unblinded=False))
    (pack/'audio').mkdir()
    for label in ['recording', 'vocals', 'A', 'B', 'R']:
        sf.write(pack/'audio'/f'c01-{label}.wav', np.zeros(800), 8000)
    return pack


def test_blinding_reference_gate_and_first_choice_survive_changes(pack, review):
    client = TestClient(review.create_app(pack))
    initial = client.get('/api/session').json()
    assert 'mapping' not in json.dumps(initial)
    assert client.get('/key.json').status_code == 404
    assert client.get('/static/../key.json').status_code == 404
    assert client.get('/audio/c01/R').status_code == 403
    assert client.post('/api/reference/c01').status_code == 409
    assert client.post('/api/unblind').status_code == 409
    assert client.get('/audio/c01/A').status_code == 200
    v = dict(revision=0, preference='A', confidence='high', finished=False)
    s = client.post('/api/answers/c01', json=v).json()
    assert s['revision'] == 1
    assert client.post('/api/answers/c01', json=v).status_code == 409  # stale tab cannot overwrite
    r = client.post('/api/reference/c01').json()
    assert r['state']['answers']['c01']['reference_exposed']
    assert client.get('/audio/c01/R').status_code == 200
    assert 'mapping' not in json.dumps(client.get('/api/export').json())
    v.update(revision=r['state']['revision'], preference='B', reference='differs', finished=True)
    s = client.post('/api/answers/c01', json=v).json()
    assert s['answers']['c01']['initial_preference'] == 'A'
    assert s['answers']['c01']['preference'] == 'B'
    assert len(s['history']) == 3
    revealed = client.post('/api/unblind').json()
    assert revealed['key']['cases']['c01']['mapping']['B'] == 'final'
    assert 'key' in client.get('/api/export').json()
    reopened = TestClient(review.create_app(pack)).get('/api/session').json()
    assert reopened['state']['unblinded']
    assert reopened['state']['answers']['c01']['preference'] == 'B'


def test_api_rejects_unknown_cases_invalid_verdicts_and_cross_origin_writes(pack, review):
    client = TestClient(review.create_app(pack))
    v = dict(revision=0, preference='A')
    assert client.post('/api/answers/other', json=v).status_code == 404
    assert client.post('/api/answers/c01', json={**v, 'preference': 'baseline'}).status_code == 422
    assert client.post('/api/answers/c01', json=v, headers={'Origin': 'https://example.org'}).status_code == 403
    assert client.get('/api/session', headers={'Host': 'evil.example'}).status_code == 403
    assert client.get('/audio/c01/key').status_code == 404
    assert 'frame-ancestors' in client.get('/').headers['content-security-policy']


def test_synth_treats_split_held_notes_identically_and_preserves_silence(review):
    held = review.synth([(1., 2., 60)], 0, 3)
    split = review.synth([(1., 1.4, 60), (1.4, 2., 60)], 0, 3)
    assert np.array_equal(held, split)
    assert np.max(np.abs(held)) > .1
    assert not np.any(held[:review.SR])
    assert not np.any(held[2*review.SR:])
    # Pitch and tuning are audible, not just onset differences.
    sharp = review.synth([(1., 2., 61)], 0, 3)
    assert not np.array_equal(held, sharp)


def test_selection_is_reproducible_stratified_and_nonoverlapping(review):
    candidates = [dict(job=f'song{i%4}', group=g, time=i*11., detail='test')
                  for g in review.GROUPS for i in range(40)]
    selected = review.select(candidates, 36)
    assert selected == review.select(candidates, 36)
    assert {g: sum(c['group'] == g for c in selected) for g in review.GROUPS} == dict.fromkeys(review.GROUPS, 6)
    for i, a in enumerate(selected):
        for b in selected[i+1:]:
            assert a['job'] != b['job'] or abs(a['time']-b['time']) >= 9


def test_build_copies_only_excerpt_audio_and_refuses_to_overwrite(tmp_path, review):
    jobs, gold = tmp_path/'jobs', tmp_path/'gold'
    job = jobs/'song'
    (job/'stems').mkdir(parents=True)
    gold.mkdir()
    signal = .1*np.sin(np.arange(12*16000)*2*np.pi*220/16000)
    sf.write(job/'audio.wav', signal, 16000)
    sf.write(job/'stems/vocals.wav', signal, 16000)
    review.write(job/'meta.json', dict(title='Synthetic song', artist='Fixture'))
    review.write(job/'lyrics.final.json', dict(lines=[dict(words=[dict(text='hello', start=2, end=3)])]))
    old = dict(tuning=0, notes=[dict(start=1, end=2, midi=60), dict(start=2, end=4, midi=72)])
    new = dict(tuning=0, notes=[dict(start=1, end=4, midi=60)])
    review.write(job/'melody.before.json', old)
    review.write(job/'melody.final.json', new)
    out = tmp_path/'pack'
    review.build(out, jobs, gold, count=1)
    public = review.read(out/'manifest.json')
    assert len(public['cases']) == 1
    assert 'mapping' not in json.dumps(public) and 'removed_jump' not in json.dumps(public)
    audio = list((out/'audio').glob('*.wav'))
    assert len(audio) == 4
    assert {sf.info(p).frames for p in audio} == {8*review.SR}
    assert review.read(job/'melody.before.json') == old
    with pytest.raises(ValueError, match='preserve reviews'):
        review.build(out, jobs, gold, count=1)


def test_selection_checks_clamped_excerpt_edges(review):
    candidates = [dict(job='short', group='control', time=t, start=s, duration=8)
                  for t, s in [(0, 0), (9, 5.5)]]
    assert len(review.select(candidates, 2)) == 1
