"""melody.json → a Standard MIDI File (format 0), written by hand: the format is small.

One track, in order: track name, tempo, time signature, key signature (from ``Melody.key``),
program change (flute: a plain tone that is easy to compare with a voice), a pitch bend for the
song's tuning (so the file sounds in tune with the recording), then the notes. Times map to ticks
at a fixed tempo, so a note at 12.5 s plays at 12.5 s in any player. Velocity follows the note's
confidence.
"""

from __future__ import annotations

import struct

from karaoke.models import Melody

PPQ = 480  # ticks per quarter note
BPM = 120.0  # no beat tracking: a fixed tempo makes ticks a plain time scale
PROGRAM = 73  # General MIDI flute (0-based)
CHANNEL = 0
BEND_RANGE = 2.0  # semitones, the General MIDI default


def vlq(n: int) -> bytes:
    """MIDI variable-length quantity."""
    if n < 0:
        raise ValueError("negative delta time")
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def seconds_to_ticks(t: float, bpm: float = BPM, ppq: int = PPQ) -> int:
    return max(0, round(t * ppq * bpm / 60.0))


def _meta(kind: int, data: bytes) -> bytes:
    return bytes([0xFF, kind]) + vlq(len(data)) + data


def melody_to_midi(melody: Melody, title: str = "melody", bpm: float = BPM, ppq: int = PPQ) -> bytes:
    events: list[tuple[int, int, bytes]] = []  # (tick, order, bytes): order puts offs before ons

    def at0(data: bytes) -> None:
        events.append((0, len(events) - 1000, data))  # keep header events first, in insertion order

    at0(_meta(0x03, title.encode("utf-8")[:120]))
    at0(_meta(0x51, struct.pack(">I", round(60_000_000 / bpm))[1:]))  # microseconds per quarter
    at0(_meta(0x58, bytes([4, 2, 24, 8])))  # 4/4
    if melody.key is not None:
        sf = max(-7, min(7, melody.key.fifths))
        at0(_meta(0x59, struct.pack(">bB", sf, 1 if melody.key.mode == "minor" else 0)))
    at0(bytes([0xC0 | CHANNEL, PROGRAM]))
    if abs(melody.tuning) > 1e-3:
        bend = 8192 + round(max(-1.0, min(1.0, melody.tuning / BEND_RANGE)) * 8191)
        bend = max(0, min(16383, bend))
        at0(bytes([0xE0 | CHANNEL, bend & 0x7F, bend >> 7]))

    for n in melody.notes:
        key = max(0, min(127, n.midi))
        on = seconds_to_ticks(n.start, bpm, ppq)
        off = max(on + 1, seconds_to_ticks(n.end, bpm, ppq))
        vel = max(1, min(127, round(40 + 80 * max(0.0, min(1.0, n.conf)))))
        events.append((on, 1, bytes([0x90 | CHANNEL, key, vel])))
        events.append((off, 0, bytes([0x80 | CHANNEL, key, 0])))

    events.sort(key=lambda e: (e[0], e[1]))
    track = bytearray()
    last = 0
    for tick, _, data in events:
        track += vlq(tick - last) + data
        last = tick
    track += vlq(0) + _meta(0x2F, b"")  # end of track
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq)
    return header + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
