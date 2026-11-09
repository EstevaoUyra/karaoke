"""Shared audio helpers (numpy + soundfile only, so they are cheap to import anywhere).

``voice_regions`` is our voice-activity detector for the *separated vocal stem*: once the
instrumental is removed, frame energy is a reliable VAD (the Silero VAD dropped half a song in
Phase 0, see docs/benchmarks.md). ``pack_clips`` turns the regions into the windows that are
passed to faster-whisper as ``clip_timestamps``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

Region = tuple[float, float]

# Defaults tuned on the full-song Kim_Vocal_2 stem of the EN cover with spikes/check_transcribe.py.
# Leakage in a separated vocal stem sits roughly 30-45 dB below the singing, so thresholds are
# relative to the track's own loud level, not absolute. The padding matters most for whisper:
# 0.2-0.3 s cut words at clip edges (69 % of lyric words matched), 0.4-0.6 s gave 75-76 %.
FRAME_S = 0.05  # RMS window
HOP_S = 0.01  # RMS hop
REF_PERCENTILE = 95.0  # "loud" reference level of the track
ON_DB = 30.0  # a region starts when energy rises above ref - ON_DB ...
OFF_DB = 38.0  # ... and ends when it falls below ref - OFF_DB (hysteresis)
FLOOR_DBFS = -65.0  # never call anything below this voice, however quiet the track is
MIN_REGION_S = 0.12  # drop blips shorter than this (clicks, breaths)
MERGE_GAP_S = 1.0  # join regions separated by less than this
SPLIT_SMOOTH_S = 0.3  # cuts go in the quietest stretch this long
SPLIT_MIN_FRAC = 0.5  # long regions are cut at their quietest point after this fraction of max_len
PAD_S = 0.4  # widen each region by this on both sides (whisper needs context at clip edges)


def duration(path: Path | str) -> float:
    """Duration of an audio file in seconds (header only, no decoding)."""
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def read_mono(path: Path | str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def rms_db(signal: np.ndarray, sr: int, frame_s: float = FRAME_S, hop_s: float = HOP_S) -> np.ndarray:
    """Frame RMS in dBFS. Frame ``i`` is centred on ``i * hop_s`` seconds."""
    hop = max(1, int(round(hop_s * sr)))
    frame = max(hop, int(round(frame_s * sr)))
    n = len(signal)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    sq = np.concatenate(([0.0], np.cumsum(signal.astype(np.float64) ** 2)))
    centres = np.arange(0, n, hop)
    lo = np.clip(centres - frame // 2, 0, n)
    hi = np.clip(centres + frame // 2, 0, n)
    energy = (sq[hi] - sq[lo]) / np.maximum(hi - lo, 1)
    return 10.0 * np.log10(np.maximum(energy, 1e-12))


def _hysteresis(db: np.ndarray, on: float, off: float) -> list[tuple[int, int]]:
    """Frame index runs [start, end) that rise above ``on`` and last until they drop below ``off``."""
    runs: list[tuple[int, int]] = []
    active = False
    start = 0
    above_on = db >= on
    below_off = db < off
    for i in range(len(db)):
        if not active and above_on[i]:
            active, start = True, i
        elif active and below_off[i]:
            active = False
            runs.append((start, i))
    if active:
        runs.append((start, len(db)))
    return runs


def merge_regions(regions: list[Region], gap: float) -> list[Region]:
    out: list[Region] = []
    for s, e in sorted(regions):
        if out and s - out[-1][1] < gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def regions_from_db(
    db: np.ndarray,
    hop_s: float,
    total: float,
    *,
    ref_percentile: float = REF_PERCENTILE,
    on_db: float = ON_DB,
    off_db: float = OFF_DB,
    floor_dbfs: float = FLOOR_DBFS,
    min_region_s: float = MIN_REGION_S,
    merge_gap_s: float = MERGE_GAP_S,
    pad_s: float = PAD_S,
    max_region_s: float | None = None,
) -> list[Region]:
    if len(db) == 0:
        return []
    ref = float(np.percentile(db, ref_percentile))
    on = max(ref - on_db, floor_dbfs)
    off = on - max(0.0, off_db - on_db)  # the hysteresis width stays the same at the floor
    raw = [(s * hop_s, e * hop_s) for s, e in _hysteresis(db, on, off)]
    raw = [r for r in raw if r[1] - r[0] >= min_region_s]
    merged = merge_regions(raw, merge_gap_s)
    padded = [(max(0.0, s - pad_s), min(total, e + pad_s)) for s, e in merged]
    # padding can make neighbours touch again
    regions = merge_regions(padded, 0.0)
    if max_region_s:
        regions = [piece for r in regions for piece in split_region(r, db, hop_s, max_region_s)]
    return [(round(s, 3), round(e, 3)) for s, e in regions]


def split_region(region: Region, db: np.ndarray, hop_s: float, max_len: float,
                 min_frac: float = SPLIT_MIN_FRAC) -> list[Region]:
    """Cut a region longer than ``max_len`` at its quietest frames (breaths between phrases).

    Each cut falls in the last ``1 - min_frac`` of the next ``max_len`` stretch, so pieces are
    between ``min_frac * max_len`` and ``max_len`` long (the last one may be shorter).
    """
    s, e = region
    out: list[Region] = []
    # the quietest 0.3 s stretch, not the quietest frame: a real pause between phrases rather
    # than a breath right before a word (a cut there made whisper drop the phrase after it)
    k = max(1, int(round(SPLIT_SMOOTH_S / hop_s)))
    smooth = np.convolve(db, np.ones(k) / k, mode="same") if len(db) >= k else db
    while e - s > max_len:
        lo = int((s + max_len * min_frac) / hop_s)
        hi = min(int((s + max_len) / hop_s), len(db))
        cut = (lo + int(np.argmin(smooth[lo:hi]))) * hop_s if hi > lo else s + max_len
        out.append((s, cut))
        s = cut
    out.append((s, e))
    return out


def voice_regions(
    vocals_path: Path | str,
    *,
    frame_s: float = FRAME_S,
    hop_s: float = HOP_S,
    ref_percentile: float = REF_PERCENTILE,
    on_db: float = ON_DB,
    off_db: float = OFF_DB,
    floor_dbfs: float = FLOOR_DBFS,
    min_region_s: float = MIN_REGION_S,
    merge_gap_s: float = MERGE_GAP_S,
    pad_s: float = PAD_S,
    max_region_s: float | None = None,
) -> list[Region]:
    """Where someone is singing in a separated vocal stem, as sorted ``(start, end)`` seconds.

    Frame RMS (``frame_s`` window, ``hop_s`` hop) in dB is compared against the track's
    ``ref_percentile`` loudness: a region opens above ``ref - on_db`` and closes below
    ``ref - off_db`` (hysteresis, so sustained notes that dip are not cut). Nothing below
    ``floor_dbfs`` counts. Regions shorter than ``min_region_s`` are dropped, gaps shorter than
    ``merge_gap_s`` are merged, and each region is padded by ``pad_s``. With ``max_region_s``, longer regions are
    split at their quietest point (see ``split_region``).
    """
    signal, sr = read_mono(vocals_path)
    total = len(signal) / sr
    db = rms_db(signal, sr, frame_s, hop_s)
    return regions_from_db(
        db, hop_s, total, ref_percentile=ref_percentile, on_db=on_db, off_db=off_db,
        floor_dbfs=floor_dbfs, min_region_s=min_region_s, merge_gap_s=merge_gap_s, pad_s=pad_s,
        max_region_s=max_region_s,
    )


def pack_clips(regions: list[Region], max_len: float = 30.0, max_gap: float = 2.0) -> list[Region]:
    """Group voice regions into whisper windows.

    Whisper encodes a padded 30 s window per clip whatever its length, so many short clips cost
    many encoder passes. Neighbouring regions are joined into one clip while the gap between them
    is at most ``max_gap`` and the clip stays within ``max_len``. Longer regions are kept whole
    (whisper walks through them in 30 s steps itself).
    """
    clips: list[Region] = []
    for s, e in sorted(regions):
        if clips and s - clips[-1][1] <= max_gap and e - clips[-1][0] <= max_len:
            clips[-1] = (clips[-1][0], e)
        else:
            clips.append((s, e))
    return clips


def coverage(regions: list[Region]) -> float:
    return float(sum(e - s for s, e in regions))
