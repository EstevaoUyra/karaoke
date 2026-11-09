"""Global alignment (Needleman-Wunsch with affine gaps) of lyric tokens to whisper tokens.

Why global: choruses repeat, so a greedy or local matcher happily jumps to the wrong repetition
(Phase 0 found 18 s and 26 s errors that way). A global alignment has to account for every
lyric word and every whisper word in order, so a jump costs a long gap and is not taken.

Moves (lyric i, whisper j):
- ``match``: 1 lyric word : 1 whisper word, exact or fuzzy (rapidfuzz ``ratio`` above a threshold).
  Mismatches are not allowed as pairs; they become one lyric gap plus one whisper gap, and the
  caller decides afterwards whether that whisper word timed the lyric word (see ``core``).
- ``split``: 1 lyric word : 2 whisper words ("anymore" vs "any more").
- ``merge``: 2 lyric words : 1 whisper word ("going to" vs "gonna").
- ``lgap``: a lyric word whisper did not hear. ``wgap``: a whisper word not in the lyrics.

Three-state Gotoh DP (pairs / lyric gap / whisper gap) filled row by row with numpy. The
whisper-gap state depends on its left neighbour, which is solved with a cumulative max, so every
row is a handful of vectorized operations: O(n·m) time and memory, ~0.1 s for 600 x 600.
An optional ``penalty`` matrix (n x m, <= 0) is added to every pair score; ``core`` uses it for
the LRC line windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.process import cdist

NEG = -1e18


@dataclass(frozen=True)
class DPParams:
    exact: float = 1.6  # exact key match, plus exact_prob * whisper prob
    exact_prob: float = 0.4
    fuzzy_lo: float = 0.4  # fuzzy match at the threshold ...
    fuzzy_hi: float = 1.4  # ... up to this just below exact
    fuzzy_prob: float = 0.2
    thr_long: float = 0.72  # rapidfuzz ratio / 100 needed for a fuzzy match
    thr_short: float = 0.80  # when either word has <= 3 characters
    merge_penalty: float = 0.25  # splits/merges cost a little more than plain matches
    merge_margin: float = 0.08  # a split must beat matching either half on its own by this much
    lgap_open: float = 1.0
    lgap_ext: float = 0.35
    wgap_open: float = 1.0
    wgap_ext: float = 0.3
    end_gap: float = 0.15  # whisper words before the first / after the last lyric word


@dataclass(frozen=True)
class Op:
    kind: str  # match | split | merge | lgap | wgap
    li: tuple[int, ...]
    wi: tuple[int, ...]
    sim: float = 0.0


def _sim(a: list[str], b: list[str]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return cdist(a, b, scorer=fuzz.ratio, dtype=np.float32, workers=1).astype(np.float64) / 100.0


def _pair_scores(sim: np.ndarray, la: np.ndarray, lb: np.ndarray, prob: np.ndarray, p: DPParams) -> np.ndarray:
    """Score of pairing each lyric (row) with each whisper item (column); NEG where not allowed."""
    thr = np.where(np.minimum(la[:, None], lb[None, :]) <= 3, p.thr_short, p.thr_long)
    exact = sim >= 0.999
    fuzzy = (sim >= thr) & ~exact
    out = np.full(sim.shape, NEG)
    pr = np.broadcast_to(prob[None, :], sim.shape)
    out[exact] = p.exact + p.exact_prob * pr[exact]
    frac = np.clip((sim - thr) / np.maximum(1e-9, 1 - thr), 0, 1)
    out[fuzzy] = p.fuzzy_lo + (p.fuzzy_hi - p.fuzzy_lo) * frac[fuzzy] + p.fuzzy_prob * pr[fuzzy]
    return out


@dataclass
class Scores:
    sim: np.ndarray  # n x m
    sub: np.ndarray  # n x m
    split: np.ndarray  # n x (m-1): lyric i vs whisper (j, j+1)
    split_sim: np.ndarray
    merge: np.ndarray  # (n-1) x m: lyric (i, i+1) vs whisper j
    merge_sim: np.ndarray


def pair_scores(a: list[str], b: list[str], prob: list[float] | np.ndarray, penalty: np.ndarray | None,
                p: DPParams, quality: np.ndarray | None = None) -> Scores:
    n, m = len(a), len(b)
    prob = np.asarray(prob, dtype=np.float64) if len(prob) else np.zeros(m)
    q = np.ones(m) if quality is None else np.asarray(quality, dtype=np.float64)
    la = np.array([len(x) for x in a], dtype=np.int64)
    lb = np.array([len(x) for x in b], dtype=np.int64)
    sim = _sim(a, b)
    sub = _pair_scores(sim, la, lb, prob, p)
    sub = np.where(sub > NEG / 2, sub * q[None, :], NEG)
    pen = penalty if penalty is not None else np.zeros((n, m))
    sub = np.where(sub > NEG / 2, sub + pen, NEG)

    # 1 lyric : 2 whisper
    if m >= 2:
        b2 = [b[j] + b[j + 1] for j in range(m - 1)]
        s12 = _sim(a, b2)
        lb2 = np.array([len(x) for x in b2], dtype=np.int64)
        pr2 = (prob[:-1] + prob[1:]) / 2
        sc12 = _pair_scores(s12, la, lb2, pr2, p) * ((q[:-1] + q[1:]) / 2)[None, :] - p.merge_penalty
        best_half = np.maximum(sim[:, :-1], sim[:, 1:])
        ok = s12 >= best_half + p.merge_margin
        pen12 = (pen[:, :-1] + pen[:, 1:]) / 2
        sc12 = np.where(ok & (sc12 > NEG / 2), sc12 + pen12, NEG)
    else:
        s12 = np.zeros((n, 0))
        sc12 = np.zeros((n, 0))
    # 2 lyric : 1 whisper
    if n >= 2:
        a2 = [a[i] + a[i + 1] for i in range(n - 1)]
        s21 = _sim(a2, b)
        la2 = np.array([len(x) for x in a2], dtype=np.int64)
        sc21 = _pair_scores(s21, la2, lb, prob, p) * q[None, :] - p.merge_penalty
        best_half = np.maximum(sim[:-1, :], sim[1:, :])
        ok = s21 >= best_half + p.merge_margin
        pen21 = (pen[:-1, :] + pen[1:, :]) / 2
        sc21 = np.where(ok & (sc21 > NEG / 2), sc21 + pen21, NEG)
    else:
        s21 = np.zeros((0, m))
        sc21 = np.zeros((0, m))
    return Scores(sim, sub, sc12, s12, sc21, s21)


def align(a: list[str], b: list[str], prob: list[float] | np.ndarray | None = None,
          penalty: np.ndarray | None = None, params: DPParams | None = None,
          lgap_extra: np.ndarray | None = None, quality: np.ndarray | None = None,
          wgap_scale: np.ndarray | None = None) -> list[Op]:
    """Align lyric keys ``a`` to whisper keys ``b``. Returns ops in order (covering every index once).

    ``lgap_extra`` (length m+1, >= 0): extra cost per lyric gap placed after whisper word j-1 and
    before whisper word j. ``core`` makes it high where whisper words are back to back (no room
    for unheard lyric words) and zero across long silences, so missing lines land where time is.
    ``quality`` (length m, 0..1) scales the match score of each whisper word, and ``wgap_scale``
    (length m) scales the cost of skipping it: unreliable whisper words (zero-length, low
    probability) attract fewer lyric words and are cheap to leave unmatched.
    """
    p = params or DPParams()
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return [Op("wgap", (), (j,)) for j in range(m)] + [Op("lgap", (i,), ()) for i in range(n)]
    sc = pair_scores(a, b, prob if prob is not None else np.ones(m), penalty, p, quality)
    gx = np.zeros(m + 1) if lgap_extra is None else np.asarray(lgap_extra, dtype=np.float64)
    ws = np.ones(m) if wgap_scale is None else np.asarray(wgap_scale, dtype=np.float64)
    wopen = p.wgap_open * ws
    wext = p.wgap_ext * ws
    endg = np.minimum(p.end_gap, wext)
    S = np.concatenate([[0.0], np.cumsum(wext)])  # S[j] = sum of wext[:j]
    E = np.concatenate([[0.0], np.cumsum(endg)])

    M = np.full((n + 1, m + 1), NEG)
    X = np.full((n + 1, m + 1), NEG)
    Y = np.full((n + 1, m + 1), NEG)
    M[0, 0] = 0.0
    Y[0, 1:] = -E[1:]
    X[1:, 0] = -(p.lgap_open + p.lgap_ext * np.arange(0, n)) - gx[0] * np.arange(1, n + 1)
    B = np.maximum(np.maximum(M, X), Y)
    for i in range(1, n + 1):
        bp = B[i - 1]
        mi = np.full(m + 1, NEG)
        mi[1:] = bp[:-1] + sc.sub[i - 1]
        if m >= 2:
            mi[2:] = np.maximum(mi[2:], bp[:-2] + sc.split[i - 1])
        if i >= 2:
            mi[1:] = np.maximum(mi[1:], B[i - 2][:-1] + sc.merge[i - 2])
        mi[0] = NEG
        xi = np.maximum(np.maximum(M[i - 1], Y[i - 1]) - p.lgap_open, X[i - 1] - p.lgap_ext) - gx
        o = np.maximum(mi, xi)[:-1] - wopen + S[1:]  # open a whisper gap at token k, plus S[k+1]
        yi = np.full(m + 1, NEG)
        yi[1:] = np.maximum.accumulate(o) - S[1:]
        M[i], X[i], Y[i] = mi, xi, yi
        B[i] = np.maximum(np.maximum(mi, xi), yi)

    end = B[n] - (E[m] - E)
    j = int(np.argmax(end))
    ops: list[Op] = [Op("wgap", (), (k,)) for k in range(m - 1, j - 1, -1)]

    def close(x: float, y: float) -> bool:
        return abs(x - y) <= 1e-7 * max(1.0, abs(x))

    def best_state(ii: int, jj: int) -> str:
        v = B[ii, jj]
        for s, mat in (("Y", Y), ("M", M), ("X", X)):  # tie → prefer whisper gaps (earliest copy)
            if close(mat[ii, jj], v):
                return s
        raise AssertionError("no state")

    i = n
    state = best_state(i, j)
    while i > 0 and j > 0:
        if state == "M":
            v = M[i, j]
            if close(v, B[i - 1, j - 1] + sc.sub[i - 1, j - 1]):
                ops.append(Op("match", (i - 1,), (j - 1,), float(sc.sim[i - 1, j - 1])))
                i, j = i - 1, j - 1
            elif j >= 2 and close(v, B[i - 1, j - 2] + sc.split[i - 1, j - 2]):
                ops.append(Op("split", (i - 1,), (j - 2, j - 1), float(sc.split_sim[i - 1, j - 2])))
                i, j = i - 1, j - 2
            elif i >= 2 and close(v, B[i - 2, j - 1] + sc.merge[i - 2, j - 1]):
                ops.append(Op("merge", (i - 2, i - 1), (j - 1,), float(sc.merge_sim[i - 2, j - 1])))
                i, j = i - 2, j - 1
            else:
                raise AssertionError(f"traceback failed at M[{i},{j}]")
            state = best_state(i, j) if i > 0 and j > 0 else "M"
        elif state == "X":
            v = X[i, j]
            ops.append(Op("lgap", (i - 1,), ()))
            if i >= 2 and close(v, X[i - 1, j] - p.lgap_ext - gx[j]):
                state = "X"
            elif close(v, Y[i - 1, j] - p.lgap_open - gx[j]):
                state = "Y"
            else:
                state = "M"
            i -= 1
        else:
            v = Y[i, j]
            ops.append(Op("wgap", (), (j - 1,)))
            if j >= 2 and close(v, Y[i, j - 1] - wext[j - 1]):
                state = "Y"
            elif close(v, M[i, j - 1] - wopen[j - 1]):
                state = "M"
            else:
                state = "X"
            j -= 1
    ops += [Op("lgap", (k,), ()) for k in range(i - 1, -1, -1)]
    ops += [Op("wgap", (), (k,)) for k in range(j - 1, -1, -1)]
    ops.reverse()
    return ops
