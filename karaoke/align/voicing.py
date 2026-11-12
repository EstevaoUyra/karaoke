"""Snap word timings to the pauses the vocal stem actually has.

Whisper's word timestamps are back to back: when the singer pauses, the silence is folded into
the word before or after it. Rendered as karaoke, the fill then crawls over that word during the
pause. The separated vocal stem shows where the pause is, but not as absolute silence (reverb and
leakage keep a stem at -30..-45 dB between phrases). So a pause is measured *relative to the word
itself*: a stretch of at least ``MIN_PAUSE_S`` that stays ``DROP_DB`` below the word's own loud
level (its ``LOUD_PERCENTILE`` frame).

For each word (``trim_words``):
- a quiet stretch of at least ``LEAD_MIN_PAUSE_S`` at the start moves the start to just before
  the voice onset (shorter ones are left: soft consonants look like pauses to an energy detector);
- the longest quiet stretch after that ends the word (plus a short release);
- whatever is loud after that pause is the next word's onset when that word starts within
  ``ONSET_PULL_S``, so the next word starts there instead.

Loudness is not enough, though: a stem can stay loud through a pause (in Sacal, 16-20 s is
-25 dB of breath, noise and leakage with no singing). So the stem's *pitch* is tracked too
(pyin voicing probability): a stretch of at least ``PITCH_PAUSE_S`` with no pitch is a pause
candidate. Raspy or shouted singing has no pitch either, so a candidate only counts if it is
clearly quieter than the singing around it, or if whisper stretched a word across it. Its edges
are then moved to where the loudness really falls and rises (the pitch tracker misses consonants).
Whisper puts the words around such a pause in one of two wrong places:
the word after it starts at the pause (the fill crawls over it until the singing resumes), or
the word before it runs to the end of the pause. ``retime_across_pauses`` moves the words that
are inside a pause to just after it, squeezing them into the time before the next word, and ends
the words before it at the pause. It runs before the loudness trim.

A singer can also hold the last word of a phrase while whisper already starts the next phrase's
words inside that held note (Sacal 22-27 s: "boy" held at one pitch, then "Eles vão…" starting
with a sharp rise at 26.8 s). Pitch can't see that; syllable *onsets* can: no new syllable
starts during a held note. ``retime_before_onsets`` moves words that sit in a stretch with no
onset, when whisper stretched them (at least ``SQUEEZE_RATIO`` times their natural length) and
the word after them starts right at the next onset. The word before them keeps the held note
until it fades (a smooth fade is an echo or reverb tail, shown as a pause).
Smooth (legato) singing also goes long without sharp onsets, which is why all three conditions
are required.

Pauses are only ever cut out of words, never added. Words the caller marks as fixed (manual and
AI edits) are not touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from karaoke.audio import HOP_S, read_mono, rms_db

MIN_PAUSE_S = 0.3  # shorter dips are consonants and breaths
LEAD_MIN_PAUSE_S = 0.6  # a word's start only moves past a longer pause (sibilants like "s", "sh" are quiet)
DROP_DB = 12.0  # a pause is this far below the word's loud level ...
LOUD_PERCENTILE = 90.0
FLOOR_DB = -60.0  # ... or simply below this (true silence, whatever the word's level)
LEAD_PAD_S = 0.15  # a moved start sits this much before the loud onset (the consonant before the vowel)
TAIL_PAD_S = 0.10  # a trimmed end keeps this much of the release
ONSET_PULL_S = 0.6  # the next word may start at most this much earlier than whisper said
MIN_WORD_S = 0.15
MIN_LOUD_S = 0.08  # a loud part shorter than this before a pause is a click, not the word

PITCH_HOP_S = 0.04
PITCH_SR = 16000
UNVOICED_PROB = 0.1  # pyin voicing probability below this is "no singing"
VOICED_PROB = 0.3  # ... and above this is singing
SUNG_BEFORE_S = 0.1  # a word with this much pitch before a pause was sung before it
PAUSE_QUIET_DB = 12.0  # a pitch pause this much quieter than the singing around it is a pause ...
STRETCH_S = 1.5  # ... and so is one that whisper stretched a word across (this much of it, and
#     at least SQUEEZE_RATIO times the word's natural length)
ONSET_RISE_DB = 10.0  # a syllable onset: the level rises this much within ONSET_WIN_S ...
ONSET_WIN_S = 0.15
ONSET_MIN_DB = -35.0  # ... to at least this level
ONSET_GAP_S = 1.5  # a stretch this long with no onset is a held note or a pause
SQUEEZE_RATIO = 3.0  # words whisper stretched to this many times their natural length ...
SQUEEZE_MIN_S = 2.0  # ... and at least this long in total
NEXT_AT_ONSET_S = 0.6  # the word after them starts within this of the next onset
ONSET_OWN_S = 0.15  # a word starting this soon after an onset is the onset's own word
EDGE_DROP_DB = 12.0  # a pause edge is where the voice falls this far below its level ...
EDGE_FLOOR_DB = 4.0  # ... or to within this of the pause's own background level
OFFSET_MAX_S = 0.8
PITCH_PAUSE_S = 1.0  # rap has unpitched stretches of up to ~0.6 s (consonant clusters); pauses are longer
# A chopped, processed hook has no pitch and sits quieter than the singing around it, but a
# syllable starts on every beat; a pause has none. A pitch pause with at least two onsets, one
# per this many seconds, is singing.
SUNG_ONSET_EVERY_S = 0.8
START_TOL_S = 0.3  # a word starting this close before a pause starts is the first word after it
SQUEEZED_WORD_S = 0.12  # shortest word when words are moved out of a pause


@dataclass
class Envelope:
    db: np.ndarray
    hop_s: float = HOP_S
    voiced_prob: np.ndarray | None = None  # pyin voicing probability, one value per PITCH_HOP_S
    pitch_hop_s: float = PITCH_HOP_S

    @classmethod
    def from_file(cls, vocals: Path | str, pitch: bool = True) -> Envelope:
        signal, sr = read_mono(vocals)
        return cls(rms_db(signal, sr), voiced_prob=voicing(signal, sr) if pitch else None)

    def voiced_time(self, s: float, e: float) -> float:
        """Seconds of [s, e] with a sung pitch (0 without pitch data)."""
        if self.voiced_prob is None or e <= s:
            return 0.0
        h = self.pitch_hop_s
        a, b = max(0, int(s / h)), min(len(self.voiced_prob), int(np.ceil(e / h)))
        return float(np.sum(self.voiced_prob[a:b] >= VOICED_PROB)) * h

    def pitched(self, t: float) -> bool:
        if self.voiced_prob is None:
            return False
        i = int(t / self.pitch_hop_s)
        return 0 <= i < len(self.voiced_prob) and self.voiced_prob[i] >= VOICED_PROB

    def _edge_level(self, ref: float, floor: float) -> float:
        return max(ref - EDGE_DROP_DB, floor + EDGE_FLOOR_DB)

    def offset_after(self, t: float, limit: float, floor: float = -120.0) -> float:
        """Where the voice that is loud just before ``t`` actually stops (pitch trackers lose
        unpitched syllables like "-nar"): scan forward while the loudness stays above the edge
        level (see ``_edge_level``), for at most ``limit`` seconds."""
        a0, a1 = self.frames(t - 0.4, t)
        if a1 - a0 < 3:
            return t
        level = self._edge_level(float(np.percentile(self.db[a0:a1], 80)), floor)
        i, end = a1, self.frames(t + limit, t + limit)[0]
        last_loud, quiet = a1, 0
        while i < end:
            if self.db[i] >= level:
                last_loud, quiet = i, 0
            else:
                quiet += 1
                if quiet * self.hop_s >= 0.08:
                    break
            i += 1
        return last_loud * self.hop_s

    def onset_before(self, t: float, limit: float, floor: float = -120.0) -> float:
        """Where the voice that is loud just after ``t`` actually starts (consonants before the
        first pitched vowel): scan backward while the loudness stays above the edge level, for at
        most ``limit`` seconds."""
        a0, a1 = self.frames(t, t + 0.4)
        if a1 - a0 < 3:
            return t
        level = self._edge_level(float(np.percentile(self.db[a0:a1], 80)), floor)
        i, stop = a0 - 1, self.frames(t - limit, t - limit)[0]
        first_loud, quiet = a0, 0
        while i >= stop:
            if self.db[i] >= level:
                first_loud, quiet = i, 0
            else:
                quiet += 1
                if quiet * self.hop_s >= 0.08:
                    break
            i -= 1
        return first_loud * self.hop_s

    def pause_edges(self, a: float, b: float) -> tuple[float, float]:
        """A pitch pause narrowed to where the voice really stops and starts again."""
        limit = min(OFFSET_MAX_S, (b - a) / 2)
        i0, i1 = self.frames(a, b)
        floor = float(np.median(self.db[i0:i1])) if i1 > i0 else -120.0
        return self.offset_after(a, limit, floor), self.onset_before(b, limit, floor)

    def onsets(self) -> np.ndarray:
        """Times of syllable onsets: sharp rises in loudness (see ``ONSET_RISE_DB``)."""
        db = self.db
        win = max(1, int(round(ONSET_WIN_S / self.hop_s)))
        if len(db) <= win:
            return np.zeros(0)
        # minimum over the previous ``win`` frames, vectorised
        from numpy.lib.stride_tricks import sliding_window_view

        prev_min = sliding_window_view(db[:-1], win).min(axis=1)  # frames [i-win, i) for i = win..n-1
        cur = db[win:]
        hit = np.flatnonzero((cur >= ONSET_MIN_DB) & (cur - prev_min >= ONSET_RISE_DB)) + win
        out, last = [], -10**9
        for i in hit:
            if i - last > 10:  # one onset per rise
                out.append(i * self.hop_s)
            last = i
        return np.asarray(out)

    def pitched_until(self, t: float, limit: float) -> float:
        """End of the pitched stretch that is going on at ``t`` (or ``t`` if none), up to ``limit``."""
        if self.voiced_prob is None:
            return t
        h = self.pitch_hop_s
        i, end = int(t / h), min(len(self.voiced_prob), int(limit / h))
        last = None
        quiet = 0
        while i < end:
            if self.voiced_prob[i] >= VOICED_PROB:
                last, quiet = i, 0
            else:
                quiet += 1
                if quiet * h > 0.2:
                    break
            i += 1
        return (last + 1) * h if last is not None else t

    def quieter_than_around(self, a: float, b: float) -> bool:
        """Is [a, b] at least ``PAUSE_QUIET_DB`` quieter than the 1.5 s on either side?
        (Raspy or shouted singing has no pitch for pyin but stays as loud as the rest.)"""
        i0, i1 = self.frames(a, b)
        l0, _ = self.frames(a - 1.5, a)
        _, r1 = self.frames(b, b + 1.5)
        around = np.r_[self.db[l0:i0], self.db[i1:r1]]
        if i1 <= i0 or len(around) < 3:
            return False
        return float(np.median(self.db[i0:i1])) <= float(np.percentile(around, 80)) - PAUSE_QUIET_DB

    def pitch_pauses(self, min_len: float = PITCH_PAUSE_S) -> list[tuple[float, float]]:
        """Stretches of at least ``min_len`` with no sung pitch, as (start, end) seconds."""
        if self.voiced_prob is None or len(self.voiced_prob) == 0:
            return []
        vp = np.asarray(self.voiced_prob, dtype=float)
        if len(vp) >= 3:  # one stray voiced frame does not end a pause
            vp = np.minimum(vp, np.maximum(np.r_[vp[1:], 0.0], np.r_[0.0, vp[:-1]]))
        quiet = vp < UNVOICED_PROB
        runs, i, h = [], 0, self.pitch_hop_s
        while i < len(quiet):
            if quiet[i]:
                j = i
                while j < len(quiet) and quiet[j]:
                    j += 1
                if (j - i) * h >= min_len:
                    runs.append((round(i * h, 3), round(j * h, 3)))
                i = j
            else:
                i += 1
        return runs

    def frames(self, s: float, e: float) -> tuple[int, int]:
        n = len(self.db)
        return min(n, max(0, int(round(s / self.hop_s)))), min(n, max(0, int(round(e / self.hop_s))))

    def quiet_runs(self, s: float, e: float) -> list[tuple[float, float]]:
        """Pauses inside [s, e] relative to that span's own loudness, as (start, end) seconds."""
        a, b = self.frames(s, e)
        seg = self.db[a:b]
        if len(seg) < 3:
            return []
        thr = max(float(np.percentile(seg, LOUD_PERCENTILE)) - DROP_DB, min(FLOOR_DB, float(seg.max()) - DROP_DB))
        quiet = seg < thr
        if self.voiced_prob is not None:  # a fading held note is quiet but still sung
            t = (a + np.arange(len(seg))) * self.hop_s
            idx = np.clip((t / self.pitch_hop_s).astype(int), 0, len(self.voiced_prob) - 1)
            quiet &= self.voiced_prob[idx] < VOICED_PROB
        runs = []
        i = 0
        while i < len(quiet):
            if quiet[i]:
                j = i
                while j < len(quiet) and quiet[j]:
                    j += 1
                if (j - i) * self.hop_s >= MIN_PAUSE_S:
                    runs.append(((a + i) * self.hop_s, (a + j) * self.hop_s))
                i = j
            else:
                i += 1
        return runs


def voicing(signal: np.ndarray, sr: int) -> np.ndarray:
    """pyin voicing probability of a mono signal, one value per ``PITCH_HOP_S`` (~0.2x real time)."""
    import librosa

    y = librosa.resample(signal, orig_sr=sr, target_sr=PITCH_SR) if sr != PITCH_SR else signal
    _, _, prob = librosa.pyin(y, fmin=70, fmax=900, sr=PITCH_SR, frame_length=1024,
                              hop_length=int(PITCH_HOP_S * PITCH_SR), center=True)
    return np.nan_to_num(prob).astype(np.float32)


def _weights(texts: list[str] | None, idx: list[int]) -> list[float]:
    if texts is None:
        return [1.0] * len(idx)
    return [len(texts[i].strip()) + 2.0 for i in idx]


def retime_across_pauses(env: Envelope, times: list[list[float]], fixed: list[bool] | None = None,
                         texts: list[str] | None = None) -> int:
    """Move words out of pitch pauses (see the module docstring), in place. Returns words changed."""
    from karaoke.align.interpolate import natural_duration

    pauses = [env.pause_edges(a, b) for a, b in env.pitch_pauses()]
    onsets = env.onsets()
    n = len(times)
    fixed = fixed or [False] * n
    changed: set[int] = set()
    for a, b in pauses:
        if b - a < PITCH_PAUSE_S:
            continue
        n_on = int(np.count_nonzero((onsets > a) & (onsets < b)))
        if n_on >= 2 and n_on >= (b - a) / SUNG_ONSET_EVERY_S:
            continue  # syllables keep starting: unpitched singing (a chopped hook), not a pause
        inside = [i for i in range(n) if times[i][0] < b - 0.05 and times[i][1] > a + 0.05]
        if not inside or any(fixed[i] for i in inside):
            continue
        stretched = any(
            min(times[i][1], b) - max(times[i][0], a) >= STRETCH_S
            and (texts is None or times[i][1] - times[i][0] >= SQUEEZE_RATIO * natural_duration(texts[i]))
            for i in inside)
        if not (stretched or env.quieter_than_around(a, b)):
            continue  # unpitched but loud, and no word stretched over it: raspy singing, not a pause
        # the first word after the pause: the first one with no singing before the pause
        m = inside[-1] + 1
        for i in inside:
            sung_before = times[i][0] < a - START_TOL_S or env.voiced_time(times[i][0], a) >= SUNG_BEFORE_S
            if not sung_before:
                m = i
                break
        for i in inside:
            if i < m and times[i][1] > a:  # sung before the pause: end it there
                ne = max(times[i][0] + MIN_WORD_S, a + TAIL_PAD_S)
                if ne < times[i][1]:
                    times[i][1] = round(ne, 3)
                    changed.add(i)
        moved = [i for i in inside if i >= m and times[i][0] < b]
        if not moved:
            continue
        k = moved[-1]
        start = max(b - LEAD_PAD_S / 2, times[m - 1][1] if m > 0 else 0.0)
        end = times[k][1]
        if k + 1 < n:
            end = min(end, times[k + 1][0]) if times[k + 1][0] > start else end
        need = SQUEEZED_WORD_S * len(moved)
        end = max(end, start + need)
        w = _weights(texts, moved)
        t = start
        for i, wi in zip(moved, w, strict=True):
            d = (end - start) * wi / sum(w)
            times[i][0], times[i][1] = round(t, 3), round(t + d, 3)
            changed.add(i)
            t += d
        # the squeeze may push into the following words: keep the order
        prev_end = times[k][1]
        for j in range(k + 1, n):
            if times[j][0] >= prev_end or fixed[j]:
                break
            dur = max(times[j][1] - times[j][0], SQUEEZED_WORD_S)
            times[j][0] = round(prev_end, 3)
            times[j][1] = round(max(times[j][1], prev_end + min(dur, SQUEEZED_WORD_S)), 3)
            changed.add(j)
            prev_end = times[j][1]
    return len(changed)


def retime_before_onsets(env: Envelope, times: list[list[float]], fixed: list[bool] | None = None,
                         texts: list[str] | None = None) -> int:
    """Move words whisper put inside a held note or a pause to the next syllable onset (see the
    module docstring), in place. Returns words changed."""
    from karaoke.align.interpolate import natural_duration

    if texts is None:
        return 0
    n = len(times)
    fixed = fixed or [False] * n
    on = env.onsets()
    changed: set[int] = set()
    for g0, g1 in zip(on, on[1:], strict=False):
        if g1 - g0 < ONSET_GAP_S:
            continue
        inside = [i for i in range(n) if g0 + ONSET_OWN_S < times[i][0] < g1 - 0.05]
        if len(inside) < 2 or any(fixed[i] for i in inside) or inside[-1] + 1 >= n:
            continue
        k = inside[-1]
        nxt = times[k + 1][0]
        if not (g1 - 0.1 <= nxt <= g1 + NEXT_AT_ONSET_S) or fixed[k + 1]:
            continue
        span = sum(times[i][1] - times[i][0] for i in inside)
        natural = sum(natural_duration(texts[i]) for i in inside)
        if span < SQUEEZE_MIN_S or span < SQUEEZE_RATIO * natural:
            continue
        m = inside[0]
        start = max(g1 - LEAD_PAD_S / 2, times[m - 1][1] if m > 0 else 0.0)
        end = max(nxt, start + SQUEEZED_WORD_S * len(inside))
        w = _weights(texts, inside)
        t = start
        for i, wi in zip(inside, w, strict=True):
            d = (end - start) * wi / sum(w)
            times[i][0], times[i][1] = round(t, 3), round(t + d, 3)
            changed.add(i)
            t += d
        if times[k + 1][0] < end:
            times[k + 1][0] = round(end, 3)
            times[k + 1][1] = round(max(times[k + 1][1], end + SQUEEZED_WORD_S), 3)
            changed.add(k + 1)
        if m > 0 and not fixed[m - 1]:  # the word before holds its note while it is still sung:
            # pitched, and not yet faded away (a smooth fade is an echo or reverb tail)
            t0 = times[m - 1][1]
            held = min(env.pitched_until(t0, start), env.offset_after(t0, start - t0))
            if held > times[m - 1][1]:
                times[m - 1][1] = round(min(held + TAIL_PAD_S, start), 3)
                changed.add(m - 1)
    return len(changed)


def trim_span(env: Envelope, s: float, e: float) -> tuple[float, float, float | None]:
    """(start, end, onset) of a word without its pauses; ``onset`` is where the voice comes back
    after the word's pause (the next word's real start), or None."""
    if e - s <= MIN_WORD_S:
        return s, e, None
    runs = env.quiet_runs(s, e)
    if not runs:
        return s, e, None
    ns, ne, onset = s, e, None
    q0, q1 = runs[0]
    if q0 - s < MIN_LOUD_S:  # starts in a pause: move the start to the onset
        if q1 - q0 >= LEAD_MIN_PAUSE_S:
            ns = max(s, q1 - LEAD_PAD_S)
        runs = runs[1:]
    if runs:
        q0, q1 = max(runs, key=lambda r: r[1] - r[0])
        if q0 - ns >= MIN_LOUD_S:
            ne = min(e, q0 + TAIL_PAD_S)
            onset = q1 if e - q1 >= 0.02 else None
    if ne - ns < MIN_WORD_S:
        return s, e, None
    return round(ns, 3), round(ne, 3), (round(onset, 3) if onset is not None else None)


def trim_words(env: Envelope, times: list[list[float]], fixed: list[bool] | None = None,
               texts: list[str] | None = None) -> int:
    """Take pauses out of word timings in place (``times`` = sorted [start, end] pairs): words in
    held notes (``retime_before_onsets``), then pitch pauses (``retime_across_pauses``), then each
    word's quiet edges (``trim_span``).

    Returns how many word timings were changed.
    """
    fixed = fixed or [False] * len(times)
    changed = retime_before_onsets(env, times, fixed, texts)
    changed += retime_across_pauses(env, times, fixed, texts)
    for i, (s, e) in enumerate(times):
        if fixed[i]:
            continue
        ns, ne, onset = trim_span(env, s, e)
        if (ns, ne) != (s, e):
            times[i][0], times[i][1] = ns, ne
            changed += 1
        if onset is not None and i + 1 < len(times) and not fixed[i + 1]:
            nxt = times[i + 1]
            if ne <= onset < nxt[0] and nxt[0] - onset <= ONSET_PULL_S:
                nxt[0] = round(max(ne, onset - LEAD_PAD_S), 3)
    return changed
