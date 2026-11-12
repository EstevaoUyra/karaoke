"""Audit reference/metric uncertainty without changing the annotations or choosing new lags.

Outputs aggregate statistics only. --tags compares both algorithms on identical subsets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import score_gold as sg
import score_vocadito as sv
from melody_decode import job_inputs


def aggregate_pitch(ref, ours, lag=0.0):
    """Pitch with greatest total covered time, invariant to splitting a held pitch into syllables.

    Union intervals per pitch so overlapping predictions cannot get extra votes.
    """
    a = ref.start + sg.PITCH_SKIP_S
    b = ref.start + max(.08, min(ref.end-ref.start, sg.PITCH_WINDOW_S))
    by_pitch = {}
    for n in ours:
        s, e = max(a, n.start+lag), min(b, n.end+lag)
        if e > s:
            by_pitch.setdefault(n.midi, []).append((s, e))
    votes = {}
    for pitch, spans in by_pitch.items():
        total, end = 0., -np.inf
        for s, e in sorted(spans):
            total += max(0, e-max(end, s))
            end = max(end, e)
        votes[pitch] = total
    return max(votes, key=votes.get) if votes else None


def audit(tags):
    out = {'songs': {}, 'vocadito_annotator_agreement': sv.score(lambda i: (sv.reference(i, 'A2'),)*2)}
    metrics = {}
    for directory in sorted(sg.GOLD.iterdir()):
        path = directory/'reference.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        job = r.get('audio', {}).get('job_id')
        if not job or not all((sg.JOBS/job/f'melody.{tag}.json').exists() for tag in tags):
            continue
        refs, spans, known = sg.load_reference(r)
        refs = [n for n in refs if sg.in_spans(n.start, spans)]
        *_, hop, db, lyrics = job_inputs(job)
        words = [(w.start, w.end) for line in lyrics.lines for w in line.words]
        in_words = [any(min(n.start+.2, n.end, e) > max(n.start, a) for a, e in words) for n in refs]
        quiet = [float(np.median(db[max(0, round(n.start/hop)):round((n.start+.15)/hop)])) < -45 for n in refs]
        info = {'notes': len(refs), 'status': r['status'], 'midi_offset': r['midi_offset'],
                'reference_windows_without_word': sum(not x for x in in_words),
                'quiet_stem_windows_below_minus45db': sum(quiet), 'scores': {}}
        for tag in tags:
            ours = [sg.N(n['start'], n['end'], n['midi'])
                    for n in json.loads((sg.JOBS/job/f'melody.{tag}.json').read_text())['notes']]
            got = [aggregate_pitch(n, ours) for n in refs]
            row = {'legacy_exact': sg.pitch_scores(refs, ours)['pitch'],
                   'aggregate_exact': float(np.mean([p == n.midi for n, p in zip(refs, got, strict=True)])),
                   'aggregate_pc': float(np.mean([p is not None and (p-n.midi) % 12 == 0
                                                 for n, p in zip(refs, got, strict=True)])),
                   'with_words_exact': float(np.mean([p == n.midi
                       for n, p, ok in zip(refs, got, in_words, strict=True) if ok])),
                   'legacy_at_fixed_lags': {str(lag): sg.pitch_scores(refs, ours, lag)['pitch']
                                            for lag in [-.1, 0, .1]}}
            info['scores'][tag] = row
            if known:
                metrics.setdefault(tag, []).append((directory.name, row))
        out['songs'][directory.name] = info
    out['means'] = {}
    for tag, rows in metrics.items():
        out['means'][tag] = {}
        for name, selected in [('all_eight', rows), ('exclude_weak_octave', [r for r in rows if r[0] != 'mas-que-nada']),
                               ('rock_corpus', [r for r in rows
                                   if r[0] not in {'mas-que-nada', 'deu-onda', 'tudo-que-voce-podia-ser'}])]:
            out['means'][tag][name] = {k: float(np.mean([r[k] for _, r in selected]))
                                      for k in ['legacy_exact', 'aggregate_exact', 'aggregate_pc', 'with_words_exact']}
    return out


def revised_annotations(directory: Path, tags: list[str]) -> dict:
    """Official 2018 stress annotations, optional local download; never replace our references.

    They contain syllable onsets, omitting non-initial melisma notes. Compare only shared
    metrical onsets (old source rounded to .01 bars), retaining the old audio alignment.
    """
    from karaoke.lyrics.syllables import split_syllables

    out = {}
    for path in sorted(directory.glob('*.str')):
        ref = json.loads((sg.GOLD/path.stem.replace('_', '-')/'reference.json').read_text())
        old, _, _ = sg.load_reference(ref)
        pairs, unmatched, words = [], 0, []
        for line in path.read_text().splitlines():
            cols = line.split()
            if len(cols) < 6:
                continue
            measure, pitch = float(cols[1]), int(cols[2])
            word, index = cols[5].split('[')
            index = int(index.rstrip(']'))
            if index == 1:
                words.append([word, index])
            elif words:
                words[-1][1] = max(words[-1][1], index)
            idx = min(range(len(old)), key=lambda k: abs(ref['notes'][k]['measure']-measure))
            if abs(ref['notes'][idx]['measure']-measure) <= .006:
                previous = old[idx]
                revised = sg.N(previous.start, previous.end, pitch + (ref['midi_offset'] or 0))
                pairs.append((previous, revised))
            else:
                unmatched += 1
        row = {'shared_onsets': len(pairs), 'changed_pitches': sum(a.midi != b.midi for a, b in pairs),
               'unmatched_revised_onsets': unmatched, 'word_tokens': len(words),
               'matching_syllable_counts': sum(len(split_syllables(w, 'en')) == n for w, n in words), 'scores': {}}
        job = ref['audio']['job_id']
        for tag in tags:
            notes = [sg.N(n['start'], n['end'], n['midi'])
                     for n in json.loads((sg.JOBS/job/f'melody.{tag}.json').read_text())['notes']]
            row['scores'][tag] = {key: float(np.mean([aggregate_pitch(pair[i], notes) == pair[i].midi for pair in pairs]))
                                  for i, key in enumerate(['old_labels', 'revised_labels'])}
        out[path.stem] = row
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tags', nargs='+', default=['before', 'final'])
    ap.add_argument('--json', type=Path)
    ap.add_argument('--revised-dir', type=Path, help='directory with official 2018 .str files')
    args = ap.parse_args()
    result = audit(args.tags)
    if args.revised_dir:
        result["revised_annotations"] = revised_annotations(args.revised_dir, args.tags)
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2)+'\n')
