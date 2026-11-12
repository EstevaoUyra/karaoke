"""Build and serve a local, label-blinded listening pack. No production job is modified.

    uv run python spikes/melody_listening.py build --out data/listening/review
    uv run python spikes/melody_listening.py serve --pack data/listening/review

The generated audio, lyrics, assignment key and answers stay in the ignored data directory.
Selection is diagnostic and deliberately stratified, not a population accuracy estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import secrets
import shutil
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from scipy.signal import resample_poly

from karaoke.store import JobPaths

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).with_name('listening')
SR = 22050
GROUPS = ('removed_jump', 'retained_jump', 'pitch_difference', 'reference_revision', 'reference_timing', 'control')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def notes(doc):
    return [(n['start'], n['end'], n['midi']) for n in doc['notes']]


def jumps(ns):
    return [b[0] for a, b in zip(ns, ns[1:], strict=False) if b[0]-a[1] <= .5 and abs(b[2]-a[2]) >= 7]


def at(ns, t):
    return next((p for a, b, p in ns if a <= t < b), None)


def collect(jobs_dir: Path, gold_dir: Path, revised_dir: Path | None):
    """Selection cues never assume that a reference mismatch is an algorithm error."""
    songs, candidates = {}, []
    refs = {}
    for file in sorted(gold_dir.glob('*/reference.json')):
        r = read(file)
        if job := (r.get('audio') or {}).get('job_id'):
            refs[job] = (file, r)
    for file in sorted(jobs_dir.glob('*/melody.final.json')):
        p = JobPaths(file.parent)
        before = p.root / 'melody.before.json'
        if not before.exists() or not p.audio.exists() or not p.vocals.exists() or not p.lyrics_final.exists():
            continue
        old, new, meta, lyrics = read(before), read(file), read(p.meta), read(p.lyrics_final)
        ns, ms = notes(old), notes(new)
        duration = min(sf.info(p.audio).duration, sf.info(p.vocals).duration)
        words = [w for line in lyrics['lines'] for w in line['words']]
        song = dict(paths=p, before=old, final=new, duration=duration, words=words,
                    title=meta.get('clean_title') or meta.get('title') or p.id,
                    artist=meta.get('clean_artist') or meta.get('artist') or '', reference=[], revised=[],
                    reference_status='No reference with a known octave', sources=[before, file, p.lyrics_final])
        songs[p.id] = song

        def add(group, t, detail, clip_duration=duration, job_id=p.id):
            if 0 <= t <= clip_duration:
                candidates.append(dict(job=job_id, group=group, time=round(t, 4), detail=detail,
                                       start=max(0., min(t-3.5, clip_duration-8.)), duration=min(8., clip_duration)))

        js, ks = jumps(ns), jumps(ms)
        for t in js:
            retained = any(abs(t-k) < .6 for k in ks)
            add('retained_jump' if retained else 'removed_jump', t, 'Baseline jump with/without nearby candidate jump')
        for t in ks:
            if not any(abs(t-j) < .6 for j in js):
                add('retained_jump', t, 'Candidate jump with no nearby baseline jump')
        for w in words:
            t = (w['start']+w['end'])/2
            a, b = at(ns, t), at(ms, t)
            if a != b:
                add('pitch_difference', t, 'Different pitch or coverage at a lyric word midpoint')
            elif a is not None:
                probes = np.arange(max(0, t-1.5), min(duration, t+1.5), .1)
                if all(at(ns, x) == at(ms, x) for x in probes):
                    add('control', t, 'Same semitone/coverage on a 100 ms grid for three seconds')
        if p.id not in refs:
            continue
        rf, r = refs[p.id]
        song['sources'].append(rf)
        song['reference_status'] = r.get('status', 'Unspecified')
        if r.get('midi_offset') is None:
            continue
        scale, offset, shift = r.get('time_scale') or 1., r.get('offset_s') or 0., r['midi_offset']
        rn = [(n['start']*scale+offset, n['end']*scale+offset, round(n['midi'])+shift) for n in r['notes']]
        song['reference'] = rn
        for a, b, _ in rn:
            if not any(min(a+.2, b, w['end']) > max(a, w['start']) for w in words):
                add('reference_timing', a, 'Reference onset window has no overlapping lyric word')
            elif rf.parent.name == 'mas-que-nada':
                add('reference_timing', a, 'Octave marked weak; audio alignment previously corrected')
        revised = revised_dir / (rf.parent.name.replace('-', '_')+'.str') if revised_dir else None
        if revised and revised.exists() and all('measure' in n for n in r['notes']):
            song['sources'].append(revised)
            changed = list(rn)
            for line in revised.read_text().splitlines():
                cols = line.split()
                if len(cols) < 6:
                    continue
                measure, pitch = float(cols[1]), int(cols[2])+shift
                i = min(range(len(rn)), key=lambda k: abs(r['notes'][k]['measure']-measure))
                if abs(r['notes'][i]['measure']-measure) <= .006 and rn[i][2] != pitch:
                    changed[i] = (*rn[i][:2], pitch)
                    add('reference_revision', rn[i][0], 'Old and revised reference pitches differ at a shared onset')
            song['revised'] = changed
    return songs, candidates


def select(candidates, count=36, seed=20261112, duration=8.):
    """Round-robin strata with balanced songs and no overlapping excerpts within a song."""
    rng = random.Random(seed)
    pools = {g: [dict(c) for c in candidates if c['group'] == g] for g in GROUPS}
    for pool in pools.values():
        rng.shuffle(pool)
    chosen, song_counts = [], Counter()

    def disjoint(a, b):
        a0, b0 = a.get('start', max(0, a['time']-3.5)), b.get('start', max(0, b['time']-3.5))
        return a['job'] != b['job'] or (a0 >= b0+b.get('duration', duration)+1 or
                                       b0 >= a0+a.get('duration', duration)+1)

    while len(chosen) < count:
        progressed = False
        for group in GROUPS:
            eligible = [c for c in pools[group] if all(disjoint(c, x) for x in chosen)]
            if not eligible:
                continue
            c = min(eligible, key=lambda x: song_counts[x['job']])
            chosen.append(c)
            song_counts[c['job']] += 1
            progressed = True
            if len(chosen) == count:
                break
        if not progressed:
            break
    rng.shuffle(chosen)
    return chosen


def excerpt(path, start, duration, sr=SR):
    with sf.SoundFile(path) as f:
        f.seek(round(start*f.samplerate))
        y = f.read(round(duration*f.samplerate), dtype='float32', always_2d=True).mean(axis=1)
        source_sr = f.samplerate
    gcd = math.gcd(source_sr, sr)
    y = resample_poly(y, sr//gcd, source_sr//gcd)
    n = round(duration*sr)
    y = np.pad(y[:n], (0, max(0, n-len(y))))
    # Same recording buffer under A and B; gain never depends on the chosen method.
    rms = float(np.sqrt(np.mean(y*y)))
    peak = float(np.max(np.abs(y)))
    gain = min(.12/max(rms, 1e-8), .85/max(peak, 1e-8), 8.)
    return y * gain


def synth(ns, start, duration, tuning=0., sr=SR):
    """Same tone/velocity for every method; merge exact unison splits to avoid articulation cues."""
    merged = []
    for a, b, pitch in sorted(ns):
        if merged and pitch == merged[-1][2] and abs(a-merged[-1][1]) < 1e-7:
            merged[-1] = (merged[-1][0], b, pitch)
        else:
            merged.append((a, b, pitch))
    y = np.zeros(round(duration*sr), dtype=np.float32)
    for a, b, pitch in merged:
        lo, hi = max(0, round((a-start)*sr)), min(len(y), round((b-start)*sr))
        if hi <= lo:
            continue
        t = np.arange(hi-lo)/sr
        frequency = 440*2**((pitch+tuning-69)/12)
        phase = 2*np.pi*frequency*t
        tone = .16*(np.sin(phase)+.2*np.sin(2*phase))
        ramp = min(round(.008*sr), (hi-lo)//2)
        if ramp:
            tone[:ramp] *= np.linspace(0, 1, ramp)
            tone[-ramp:] *= np.linspace(1, 0, ramp)
        y[lo:hi] += tone
    return np.clip(y, -.95, .95)


def build(out, jobs_dir, gold_dir, revised_dir=None, count=36, seed=20261112):
    if out.exists():
        raise ValueError(f'{out} already exists; choose a new directory to preserve reviews')
    songs, candidates = collect(jobs_dir, gold_dir, revised_dir)
    chosen = select(candidates, count, seed)
    if len(chosen) < count:
        raise ValueError(f'Only {len(chosen)} non-overlapping excerpts available for requested {count}')
    temp = out.with_name(out.name+'.building')
    temp.mkdir(parents=True, exist_ok=False)
    try:
        (temp/'audio').mkdir()
        labels = ['before']*(count//2) + ['final']*(count-count//2)
        secrets.SystemRandom().shuffle(labels)
        public, private = [], {}
        for i, (c, a_method) in enumerate(zip(chosen, labels, strict=True), 1):
            song = songs[c['job']]
            duration = min(8., song['duration'])
            start = max(0., min(c['time']-3.5, song['duration']-duration))
            cid = f'c{i:02d}'
            sources = {'recording': excerpt(song['paths'].audio, start, duration),
                       'vocals': excerpt(song['paths'].vocals, start, duration)}
            mapping = {'A': a_method, 'B': 'final' if a_method == 'before' else 'before'}
            for label, method in mapping.items():
                sources[label] = synth(notes(song[method]), start, duration, song[method].get('tuning', 0.))
            ref = [n for n in song['reference'] if min(n[1], start+duration) > max(n[0], start)]
            revised = [n for n in song['revised'] if min(n[1], start+duration) > max(n[0], start)]
            if ref:
                sources['R'] = synth(ref, start, duration)
            if revised and revised != ref:
                sources['R2'] = synth(revised, start, duration)
            for label, data in sources.items():
                sf.write(temp/'audio'/f'{cid}-{label}.wav', data, SR, subtype='PCM_16')
            words = [dict(text=w['text'], start=max(0, w['start']-start), end=min(duration, w['end']-start))
                     for w in song['words'] if min(w['end'], start+duration) > max(w['start'], start)]
            public.append(dict(id=cid, title=song['title'], artist=song['artist'], start=start,
                               duration=duration, focus=c['time']-start, words=words, has_reference=bool(ref)))
            private[cid] = dict(**c, mapping=mapping, reference_status=song['reference_status'],
                                has_revised='R2' in sources)
            print(f'Built {i}/{count}', flush=True)
        manifest = dict(version=1, id=secrets.token_hex(12), created=datetime.now(UTC).isoformat(), cases=public)
        hashes = {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for job in {c['job'] for c in chosen} for p in songs[job]['sources']}
        write(temp/'manifest.json', manifest)
        write(temp/'key.json', dict(cases=private, selection_seed=seed, hashes=hashes,
                                   builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                   audio_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                 for p in sorted((temp/'audio').glob('*.wav'))},
                                   strata=dict(Counter(c['group'] for c in chosen))))
        write(temp/'answers.json', dict(pack_id=manifest['id'], revision=0, answers={}, history=[], unblinded=False))
        temp.rename(out)
    except Exception:
        shutil.rmtree(temp)
        raise
    print(f'Review pack: {out}\nStrata: {dict(Counter(c["group"] for c in chosen))}')


class Verdict(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: int = Field(ge=0)
    preference: Literal['A', 'B', 'equal_good', 'equal_bad', 'unsure']
    confidence: Literal['low', 'medium', 'high'] = 'medium'
    issues: list[Literal['pitch', 'timing', 'syllables', 'alternate_voice', 'unclear_audio']] = Field(default_factory=list)
    note: str = Field(default='', max_length=4000)
    reference: Literal['not_assessed', 'matches', 'differs', 'ambiguous', 'unsure'] = 'not_assessed'
    finished: bool = False


def create_app(pack: Path):
    pack = pack.resolve()
    manifest, key = read(pack/'manifest.json'), read(pack/'key.json')
    ids = {c['id'] for c in manifest['cases']}
    lock = threading.RLock()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        from urllib.parse import urlsplit

        host = request.headers.get('host', '').split(':')[0]
        if host not in {'127.0.0.1', 'localhost', 'testserver'}:
            return JSONResponse({'detail': 'Loopback host required'}, 403)
        origin = request.headers.get('origin')
        if origin and urlsplit(origin).netloc != request.headers.get('host'):
            return JSONResponse({'detail': 'Same-origin requests required'}, 403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            "img-src 'self' data:; media-src 'self' blob:; frame-ancestors 'none'"
        )
        return response

    def state():
        return read(pack/'answers.json')

    def save(s):
        s['revision'] += 1
        write(pack/'answers.json', s)

    def check_id(cid):
        if cid not in ids:
            raise HTTPException(404, 'Unknown excerpt')

    @app.get('/')
    def index():
        return FileResponse(STATIC/'index.html')

    @app.get('/static/{name}')
    def static(name: str):
        if name not in {'app.js', 'style.css'}:
            raise HTTPException(404)
        return FileResponse(STATIC/name)

    @app.get('/api/session')
    def session():
        with lock:
            return dict(manifest=manifest, state=state())

    @app.post('/api/answers/{cid}')
    def answer(cid: str, verdict: Verdict):
        check_id(cid)
        with lock:
            s = state()
            if verdict.revision != s['revision']:
                raise HTTPException(409, 'Review changed in another tab. Reload before saving.')
            previous = s['answers'].get(cid, {})
            v = verdict.model_dump(exclude={'revision'})
            v['initial_preference'] = previous.get('initial_preference', v['preference'])
            v['reference_exposed'] = previous.get('reference_exposed', False)
            v['updated'] = datetime.now(UTC).isoformat()
            s['answers'][cid] = v
            s['history'].append(dict(case=cid, event='verdict', value=v.copy(), after_unblind=s['unblinded']))
            save(s)
            return s

    @app.post('/api/reference/{cid}')
    def reference(cid: str):
        check_id(cid)
        with lock:
            s = state()
            if cid not in s['answers']:
                raise HTTPException(409, 'Save your A/B choice before consulting the reference')
            if not s['answers'][cid].get('reference_exposed'):
                s['answers'][cid]['reference_exposed'] = True
                s['history'].append(dict(case=cid, event='reference_exposed', at=datetime.now(UTC).isoformat()))
                save(s)
            return dict(state=s, has_revised=key['cases'][cid]['has_revised'])

    @app.get('/audio/{cid}/{label}')
    def audio(cid: str, label: str):
        check_id(cid)
        if label not in {'recording', 'vocals', 'A', 'B', 'R', 'R2'}:
            raise HTTPException(404)
        with lock:
            if label in {'R', 'R2'} and not state()['answers'].get(cid, {}).get('reference_exposed'):
                raise HTTPException(403, 'Reference not yet requested')
        path = pack/'audio'/f'{cid}-{label}.wav'
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path, media_type='audio/wav')

    @app.get('/api/export')
    def export():
        with lock:
            s = state()
            result = dict(manifest=manifest, state=s)
            if s['unblinded']:
                result['key'] = key
            return JSONResponse(result, headers={'Content-Disposition': 'attachment; filename="listening-review.json"'})

    @app.post('/api/unblind')
    def unblind():
        with lock:
            s = state()
            if not all(s['answers'].get(cid, {}).get('finished') for cid in ids):
                raise HTTPException(409, 'Finish every excerpt before revealing the assignments')
            if not s['unblinded']:
                s['unblinded'] = True
                save(s)
            return dict(state=s, key=key)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--jobs', type=Path, default=ROOT/'data/jobs')
    b.add_argument('--gold', type=Path, default=ROOT/'data/gold')
    b.add_argument('--revised-dir', type=Path)
    b.add_argument('--count', type=int, default=36)
    b.add_argument('--seed', type=int, default=20261112)
    s = sub.add_parser('serve')
    s.add_argument('--pack', type=Path, required=True)
    s.add_argument('--port', type=int, default=8775)
    args = ap.parse_args()
    if args.command == 'build':
        if not 1 <= args.count <= 200:
            ap.error('--count must be between 1 and 200')
        build(args.out.resolve(), args.jobs, args.gold, args.revised_dir, args.count, args.seed)
    else:
        import uvicorn

        uvicorn.run(create_app(args.pack), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
