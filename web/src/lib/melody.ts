// Where the pitch guide has gaps: time where words are sung but no note was found.
import type { LyricsDoc } from "../types";

export interface Span {
  start: number;
  end: number;
}

/** Union of intervals, sorted and merged (touching or overlapping by up to `join` seconds). */
export function union(spans: Span[], join = 0): Span[] {
  const s = spans.filter((x) => x.end > x.start).sort((a, b) => a.start - b.start);
  const out: Span[] = [];
  for (const x of s) {
    const last = out[out.length - 1];
    if (last && x.start <= last.end + join) last.end = Math.max(last.end, x.end);
    else out.push({ ...x });
  }
  return out;
}

/** a minus b (both unions). */
export function subtract(a: Span[], b: Span[]): Span[] {
  const out: Span[] = [];
  let j = 0;
  for (const x of a) {
    let cur = x.start;
    while (j < b.length && b[j].end <= cur) j++;
    let k = j;
    while (k < b.length && b[k].start < x.end) {
      if (b[k].start > cur) out.push({ start: cur, end: b[k].start });
      cur = Math.max(cur, b[k].end);
      k++;
    }
    if (cur < x.end) out.push({ start: cur, end: x.end });
  }
  return out;
}

/**
 * Sung time with no note: words (joined across small gaps) minus notes, keeping pieces of at least
 * `minS`. These are the "missing places" of the pitch guide.
 */
export function missingSpans(lyrics: LyricsDoc | null, notes: Span[], minS = 0.25): Span[] {
  if (!lyrics) return [];
  const words = union(lyrics.lines.flatMap((l) => l.words.map((w) => ({ start: w.start, end: w.end }))), 0.12);
  return subtract(words, union(notes)).filter((s) => s.end - s.start >= minS);
}

export const total = (spans: Span[]) => spans.reduce((a, s) => a + s.end - s.start, 0);
