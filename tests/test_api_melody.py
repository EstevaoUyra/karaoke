"""Pitch-guide endpoints and the MIDI writer, on the demo job with a small melody.json fixture."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from karaoke.api.app import create_app
from karaoke.api.midi import BPM, PPQ, melody_to_midi, seconds_to_ticks, vlq
from karaoke.models import KeyEstimate, Melody, Note
from karaoke.store import file_hash, write_model

# ---------------------------------------------------------------- a minimal SMF reader (for the tests)


def read_vlq(b: bytes, i: int) -> tuple[int, int]:
    n = 0
    while True:
        c = b[i]
        i += 1
        n = (n << 7) | (c & 0x7F)
        if not c & 0x80:
            return n, i


def parse_smf(b: bytes) -> dict:
    assert b[:4] == b"MThd"
    hlen, fmt, ntracks, ppq = struct.unpack(">IHHH", b[4:14])
    assert hlen == 6
    i = 14
    tracks = []
    for _ in range(ntracks):
        assert b[i:i + 4] == b"MTrk"
        (tlen,) = struct.unpack(">I", b[i + 4:i + 8])
        data = b[i + 8:i + 8 + tlen]
        i += 8 + tlen
        j, tick, events, status = 0, 0, [], 0
        while j < len(data):
            delta, j = read_vlq(data, j)
            tick += delta
            if data[j] == 0xFF:
                kind = data[j + 1]
                ln, j = read_vlq(data, j + 2)
                events.append((tick, "meta", kind, data[j:j + ln]))
                j += ln
                continue
            if data[j] & 0x80:
                status = data[j]
                j += 1
            hi = status & 0xF0
            n = 1 if hi in (0xC0, 0xD0) else 2
            events.append((tick, "midi", status, data[j:j + n]))
            j += n
        tracks.append(events)
    assert i == len(b), "trailing bytes"
    return {"format": fmt, "ppq": ppq, "tracks": tracks}


def notes_of(track: list, ppq: int, bpm: float) -> list[tuple[int, float, float, int]]:
    """(key, start s, end s, velocity) from note on/off pairs."""
    sec = 60.0 / (bpm * ppq)
    open_: dict[int, tuple[int, int]] = {}
    out = []
    for tick, kind, status, data in track:
        if kind != "midi":
            continue
        if status & 0xF0 == 0x90 and data[1] > 0:
            open_[data[0]] = (tick, data[1])
        elif status & 0xF0 in (0x80, 0x90):
            on, vel = open_.pop(data[0])
            out.append((data[0], on * sec, tick * sec, vel))
    assert not open_, "unterminated notes"
    return sorted(out, key=lambda n: n[1])


def demo_melody() -> Melody:
    # notes under the first verse line of the demo song; "window," (3.70-4.60) and the whole second
    # line are left without notes (the "missing places")
    notes = [Note(start=2.0, end=2.55, midi=64, pitch=64.1, conf=0.9),
             Note(start=2.6, end=3.0, midi=66, pitch=65.9, conf=0.8),
             Note(start=3.05, end=3.65, midi=67, pitch=67.2, conf=0.5),
             Note(start=8.0, end=9.1, midi=71, pitch=71.0, conf=1.0)]
    key = KeyEstimate(tonic="E", tonic_pc=4, mode="minor", fifths=1, accidentals=["F#"], confidence=0.71,
                      margin=0.08, runner_up="G major")
    return Melody(hop_s=0.04, notes=notes, key=key, median_midi=66.0, tuning=0.25)


# ---------------------------------------------------------------- MIDI writer


def test_vlq():
    assert vlq(0) == b"\x00"
    assert vlq(0x7F) == b"\x7f"
    assert vlq(0x80) == b"\x81\x00"
    assert vlq(0x0FFFFFFF) == b"\xff\xff\xff\x7f"
    assert read_vlq(vlq(123456), 0) == (123456, 3)


def test_midi_roundtrip():
    m = demo_melody()
    smf = parse_smf(melody_to_midi(m, title="Sing It Loud"))
    assert smf["format"] == 0 and smf["ppq"] == PPQ and len(smf["tracks"]) == 1
    tr = smf["tracks"][0]
    metas = {kind: data for _, k, kind, data in tr if k == "meta"}
    assert metas[0x03] == b"Sing It Loud"
    assert int.from_bytes(metas[0x51], "big") == round(60_000_000 / BPM)  # tempo
    assert metas[0x58][:2] == bytes([4, 2])  # 4/4
    assert struct.unpack(">bB", metas[0x59]) == (1, 1)  # one sharp, minor: E minor
    assert 0x2F in metas  # end of track
    assert tr[-1][2] == 0x2F
    program = [d for _, k, s, d in tr if k == "midi" and s & 0xF0 == 0xC0]
    assert program == [bytes([73])]
    bend = [d for _, k, s, d in tr if k == "midi" and s & 0xF0 == 0xE0][0]
    assert (bend[0] | bend[1] << 7) - 8192 == pytest.approx(0.25 / 2 * 8191, abs=1)  # +25 cents
    got = notes_of(tr, PPQ, BPM)
    assert [n[0] for n in got] == [64, 66, 67, 71]
    for (_key, s, e, vel), n in zip(got, m.notes, strict=True):
        assert s == pytest.approx(n.start, abs=1e-3) and e == pytest.approx(n.end, abs=1e-3)
        assert vel == round(40 + 80 * n.conf)


def test_midi_adjacent_same_pitch_and_no_key():
    m = Melody(hop_s=0.04, notes=[Note(start=1.0, end=1.5, midi=60, pitch=60, conf=1),
                                  Note(start=1.5, end=2.0, midi=60, pitch=60, conf=1)])
    tr = parse_smf(melody_to_midi(m))["tracks"][0]
    assert not [1 for _, k, kind, _ in tr if k == "meta" and kind == 0x59]  # no key → no key signature
    assert [(round(s, 3), round(e, 3)) for _, s, e, _ in notes_of(tr, PPQ, BPM)] == [(1.0, 1.5), (1.5, 2.0)]
    assert seconds_to_ticks(1.5) == 1440


# ---------------------------------------------------------------- endpoints


@pytest.fixture
def client(store, demo):
    with TestClient(create_app(store, watch=False, web_dist=None)) as c:
        yield c


def test_melody_404_then_served(client, demo):
    r = client.get("/api/jobs/demo/melody")
    assert r.status_code == 404 and "melody step" in r.json()["detail"]
    assert client.get("/api/jobs/demo/melody.mid").status_code == 404
    assert client.get("/api/jobs/demo/melody/contour").status_code == 404
    assert client.get("/api/jobs/demo").json()["files"]["melody"] is None

    write_model(demo.melody, demo_melody())
    d = client.get("/api/jobs/demo/melody").json()
    assert len(d["notes"]) == 4 and d["key"]["tonic"] == "E" and d["tuning"] == 0.25
    assert client.get("/api/jobs/demo").json()["files"]["melody"]["size"] > 0

    mid = client.get("/api/jobs/demo/melody.mid")
    assert mid.status_code == 200 and mid.headers["content-type"] == "audio/midi"
    assert 'filename="demo-melody.mid"' in mid.headers["content-disposition"]
    assert [n[0] for n in notes_of(parse_smf(mid.content)["tracks"][0], PPQ, BPM)] == [64, 66, 67, 71]
    assert parse_smf(mid.content)["tracks"][0][0][3] == b"Sing It Loud (melody)"


def test_contour_is_downsampled(client, demo):
    n = 600  # 24 s at 40 ms
    f0 = np.full(n, np.nan, dtype=np.float32)
    f0[50:100] = 329.63  # E4 from 2.0 s to 4.0 s
    prob = np.where(np.isfinite(f0), 0.9, 0.05).astype(np.float32)
    demo.melody_f0.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(demo.melody_f0, f0=f0, prob=prob, hop_s=0.04, vocals_hash=file_hash(demo.vocals), version=1)
    r = client.get("/api/jobs/demo/melody/contour?max_points=150").json()
    assert len(r["t"]) == len(r["hz"]) == len(r["conf"]) == 150
    assert r["step_s"] == pytest.approx(0.16) and not r["stale"]
    voiced = [(t, hz) for t, hz in zip(r["t"], r["hz"], strict=True) if hz is not None]
    assert voiced and all(hz == pytest.approx(329.63, abs=0.01) for _, hz in voiced)
    assert 1.9 <= voiced[0][0] <= 2.2 and 3.8 <= voiced[-1][0] <= 4.1
    assert r["conf"][0] == pytest.approx(0.05)
    demo.vocals.write_bytes(demo.vocals.read_bytes()[:-2])  # the stem changed: the contour is stale
    assert client.get("/api/jobs/demo/melody/contour").json()["stale"]
