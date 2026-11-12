"""F0 trackers for the vocal stem: (f0 Hz, confidence 0..1, hop seconds).

- ``crepe_track``: torchcrepe (CREPE, MIT; weights bundled with the package) with viterbi
  decoding. Its confidence is CREPE's periodicity. It always returns a pitch, so voicing is decided
  downstream from the confidence. **It imports torch**: never call it in a process that has loaded
  ctranslate2 (faster-whisper); the melody stage is ``isolated=True`` for that reason.
- ``pyin_track``: librosa pyin with ``karaoke.align.voicing``'s parameters (the fallback when
  torch is unavailable). NaN where pyin says unvoiced; confidence is its voicing probability.

Both run in ``CHUNK_S`` chunks (progress, bounded memory) with ``CHUNK_PAD_S`` of context.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

SR = 16000
FMIN, FMAX = 70.0, 900.0  # as karaoke.align.voicing.voicing
PYIN_FRAME = 1024
CHUNK_S = 30.0
CHUNK_PAD_S = 1.0

Progress = Callable[[float], None] | None


def _resample(signal: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    return librosa.resample(signal, orig_sr=sr, target_sr=SR) if sr != SR else signal


def _chunks(n_frames: int, hop_s: float):
    chunk, pad = int(CHUNK_S / hop_s), int(CHUNK_PAD_S / hop_s)
    for c0 in range(0, n_frames, chunk):
        c1 = min(n_frames, c0 + chunk)
        yield c0, c1, max(0, c0 - pad), pad


def pyin_track(signal: np.ndarray, sr: int, progress: Progress = None) -> tuple[np.ndarray, np.ndarray, float]:
    import librosa

    from karaoke.align.voicing import PITCH_HOP_S

    y = _resample(signal, sr)
    hop = int(PITCH_HOP_S * SR)
    n_frames = 1 + len(y) // hop
    f0 = np.full(n_frames, np.nan, dtype=np.float32)
    prob = np.zeros(n_frames, dtype=np.float32)
    for c0, c1, a, pad in _chunks(n_frames, PITCH_HOP_S):
        seg = y[a * hop: (c1 + pad) * hop]
        if len(seg) < PYIN_FRAME:
            continue
        f, _, p = librosa.pyin(seg, fmin=FMIN, fmax=FMAX, sr=SR, frame_length=PYIN_FRAME, hop_length=hop, center=True)
        k0 = c0 - a
        k1 = min(len(f), k0 + (c1 - c0))
        f0[c0:c0 + k1 - k0] = f[k0:k1]
        prob[c0:c0 + k1 - k0] = np.nan_to_num(p[k0:k1])
        if progress:
            progress(c1 / n_frames)
    return f0, prob, PITCH_HOP_S


CREPE_BINS = 360
CREPE_BIN_ST = 0.2  # CREPE's pitch bins are 20 cents apart
CREPE_BIN0_MIDI = 69.0 + (1997.3794084376191 - 1200.0 * np.log2(44.0)) / 100.0  # bin 0 in MIDI (23.46)


def bin_to_midi(b):
    """Fractional MIDI of CREPE pitch bin ``b`` (may be fractional or an array)."""
    return CREPE_BIN0_MIDI + CREPE_BIN_ST * np.asarray(b, dtype=float)


def midi_to_bin(m):
    return (np.asarray(m, dtype=float) - CREPE_BIN0_MIDI) / CREPE_BIN_ST


def crepe_activation(signal: np.ndarray, sr: int, model: str = "full", hop_s: float = 0.01,
                     progress: Progress = None, batch_size: int = 512) -> tuple[np.ndarray, int, float]:
    """CREPE's whole output, before any pitch is chosen: ``(act, bin0, hop seconds)``.

    ``act[t, k]`` (float32, 0..1) is the network's activation for pitch bin ``bin0 + k`` in frame
    ``t`` (centred on ``t * hop``): one independent sigmoid per bin, not a distribution that sums
    to one. Only the bins between ``FMIN`` and ``FMAX`` are kept. ``bin_to_midi`` gives a bin's
    pitch. Imports torch (see the module docstring)."""
    import torch
    import torchcrepe

    y = _resample(signal, sr).astype(np.float32)
    hop = int(round(hop_s * SR))
    lo = int(np.floor(midi_to_bin(69.0 + 12.0 * np.log2(FMIN / 440.0))))
    hi = int(np.ceil(midi_to_bin(69.0 + 12.0 * np.log2(FMAX / 440.0)))) + 1
    lo, hi = max(0, lo), min(CREPE_BINS, hi)
    n_frames = 1 + len(y) // hop
    act = np.zeros((n_frames, hi - lo), dtype=np.float32)
    done = 0
    with torch.inference_mode():
        for frames in torchcrepe.preprocess(torch.from_numpy(y)[None], SR, hop, batch_size, "cpu", True):
            out = torchcrepe.infer(frames, model, "cpu").numpy()
            k = min(len(out), n_frames - done)
            act[done:done + k] = out[:k, lo:hi]
            done += k
            if progress:
                progress(done / n_frames)
    return act, lo, hop / SR


def crepe_track(signal: np.ndarray, sr: int, model: str = "full", hop_s: float = 0.01,
                progress: Progress = None, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray, float]:
    import torch
    import torchcrepe

    y = _resample(signal, sr).astype(np.float32)
    hop = int(round(hop_s * SR))
    n_frames = 1 + len(y) // hop
    f0 = np.full(n_frames, np.nan, dtype=np.float32)
    conf = np.zeros(n_frames, dtype=np.float32)
    with torch.inference_mode():
        for c0, c1, a, pad in _chunks(n_frames, hop / SR):
            seg = y[a * hop: (c1 + pad) * hop]
            if len(seg) < 1024:
                continue
            f, p = torchcrepe.predict(torch.from_numpy(seg)[None], SR, hop, FMIN, FMAX, model,
                                      decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                                      batch_size=batch_size, device="cpu", pad=True)
            f, p = f[0].numpy(), p[0].numpy()
            k0 = c0 - a
            k1 = min(len(f), k0 + (c1 - c0))
            f0[c0:c0 + k1 - k0] = f[k0:k1]
            conf[c0:c0 + k1 - k0] = p[k0:k1]
            if progress:
                progress(c1 / n_frames)
    return f0, conf, hop / SR
