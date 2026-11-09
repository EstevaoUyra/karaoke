"""Where to break a lyric line: shared by the whisper fallback (``align.fallback``) and the
renderer (``render.layout``), which splits lines that are too wide for one row.

A break between word ``k - 1`` and word ``k`` costs less when it falls:
- after punctuation (``.!?`` most, then ``,;:``), around parentheses (backing vocals);
- before a capitalised word (a new sentence) or a conjunction;
- at a pause in the singing;
- near the middle (balanced halves);
and costs more when it leaves a function word ("the", "de", "me", …) dangling at the end of a line.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

# words that usually start a clause: a good place to start a new line
CONJUNCTIONS = {
    # EN
    "and", "but", "or", "so", "'cause", "cause", "because", "when", "while", "if", "that", "like", "then",
    "till", "until", "where", "who", "as",
    # PT
    "e", "mas", "ou", "que", "porque", "pois", "quando", "se", "enquanto", "onde", "então", "pra", "sem",
    "até", "como",
}
# words that should not end a line (they belong to what follows)
CLINGY = {
    # EN
    "the", "a", "an", "my", "your", "his", "her", "our", "their", "to", "of", "in", "on", "at", "for", "with",
    "i", "i'm", "i'd", "i'll", "you're", "and", "but", "or",
    # PT
    "o", "os", "as", "um", "uma", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "me",
    "te", "se", "lhe", "meu", "minha", "seu", "sua", "pra", "por", "com", "e", "mas", "que", "eu", "tu", "não",
}
I_FORMS = {"i", "i'm", "i'd", "i'll", "i've"}
SENTENCE_END = (".", "!", "?", "…")
CLAUSE_END = (",", ";", ":", "—", "-")


@dataclass(frozen=True)
class W:
    text: str
    start: float = 0.0
    end: float = 0.0


def bare(text: str) -> str:
    return text.strip().strip("\"'“”‘’()[]¿¡").lower()


def starts_sentence(text: str) -> bool:
    """A capitalised word that is not just the English "I"."""
    t = text.strip().lstrip("\"'“‘(¿¡")
    return bool(t) and t[0].isupper() and bare(text) not in I_FORMS


def break_cost(words: Sequence[W], k: int, widths: Sequence[float]) -> float:
    """Cost of starting a new line at word ``k`` (0 < k < len(words)); lower is better."""
    prev, nxt = words[k - 1].text.strip(), words[k].text.strip()
    left, right = sum(widths[:k]), sum(widths[k:])
    cost = abs(left - right) / max(left + right, 1e-6)  # 0 (balanced) .. 1
    if prev.endswith(SENTENCE_END):
        cost -= 1.0
    elif prev.endswith(CLAUSE_END):
        cost -= 0.6
    if prev.endswith(")") or nxt.startswith("("):
        cost -= 1.0
    before = " ".join(w.text for w in words[:k])
    if before.count("(") > before.count(")"):
        cost += 0.6  # keep a parenthesised part (backing vocals) together
    if starts_sentence(nxt):
        cost -= 0.5
    if bare(nxt) in CONJUNCTIONS:
        cost -= 0.3
    gap = words[k].start - words[k - 1].end
    if gap > 0:
        cost -= 0.8 * min(gap, 1.0)
    if bare(prev) in CLINGY and not prev.endswith(SENTENCE_END + CLAUSE_END):
        cost += 0.5
    # a single word alone on a line reads badly
    if k == 1 or k == len(words) - 1:
        cost += 0.4
    return cost


SOFT_BREAK_COST = -0.6  # a "strong" break: after punctuation, around parentheses


def split_to_fit(words: Sequence[W], width: Callable[[str], float], max_width: float,
                 soft_width: float | None = None) -> list[list[int]]:
    """Split ``words`` into lines that each fit ``max_width``, as lists of word indices.

    Recursive best split: the cheapest break whose halves both fit, or (if none does) the
    cheapest break overall, then each half again. A line that fits but is wider than
    ``soft_width`` is split too, but only at a strong break (cost <= ``SOFT_BREAK_COST``).
    """
    idx = list(range(len(words)))
    return _split(words, idx, width, max_width, soft_width)


def _line_width(words: Sequence[W], idx: list[int], width: Callable[[str], float]) -> float:
    return width(" ".join(words[i].text.strip() for i in idx))


def _split(words: Sequence[W], idx: list[int], width: Callable[[str], float], max_width: float,
           soft_width: float | None) -> list[list[int]]:
    if len(idx) <= 1:
        return [idx]
    total = _line_width(words, idx, width)
    soft = total <= max_width
    if soft and (soft_width is None or total <= soft_width):
        return [idx]
    sub = [words[i] for i in idx]
    widths = [width(w.text.strip() + " ") for w in sub]
    best_fit, best_any = None, None
    for k in range(1, len(sub)):
        c = break_cost(sub, k, widths)
        fits = (_line_width(words, idx[:k], width) <= max_width and _line_width(words, idx[k:], width) <= max_width)
        if best_any is None or c < best_any[0]:
            best_any = (c, k)
        if fits and (best_fit is None or c < best_fit[0]):
            best_fit = (c, k)
    if soft:
        if best_fit is None or best_fit[0] > SOFT_BREAK_COST:
            return [idx]
        k = best_fit[1]
    else:
        k = (best_fit or best_any)[1]  # type: ignore[index]
    return (_split(words, idx[:k], width, max_width, soft_width)
            + _split(words, idx[k:], width, max_width, soft_width))
