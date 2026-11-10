// What the play-along stage shows at time t: the same rules as the rendered video
// (current line plus the next one, lines appear `leadIn` seconds early, a countdown after long gaps).
import type { LineDoc, WordDoc } from "../types";

export interface StageView {
  current: number | null; // line index being sung (or about to be)
  next: number | null;
  countdown: number | null; // 3, 2, 1 before a line that follows a long gap
}

/** First line that has not finished at time t (binary search; lines are in time order). */
export function firstUnfinished(lines: LineDoc[], t: number): number {
  let lo = 0;
  let hi = lines.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (lines[mid].end <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function stageView(lines: LineDoc[], t: number, leadIn = 1.5, countdownGap = 4): StageView {
  const i = firstUnfinished(lines, t);
  if (i >= lines.length) return { current: null, next: null, countdown: null };
  const line = lines[i];
  const prevEnd = i > 0 ? lines[i - 1].end : 0;
  const untilStart = line.start - t;
  let countdown: number | null = null;
  if (line.start - prevEnd > countdownGap && untilStart > 0 && untilStart <= 3) countdown = Math.ceil(untilStart);
  if (untilStart > leadIn) {
    // in a gap: show the coming line dimmed only if it's close
    return { current: null, next: untilStart <= leadIn + 1.5 ? i : null, countdown };
  }
  return { current: i, next: i + 1 < lines.length ? i + 1 : null, countdown };
}

/** 0..1: how much of the word has been sung. */
export function wordFill(w: WordDoc, t: number): number {
  if (t <= w.start) return 0;
  if (t >= w.end) return 1;
  return (t - w.start) / Math.max(w.end - w.start, 1e-3);
}

/** Index of the word being sung in the line at t (the last started one), or -1 before the line. */
export function activeWord(line: LineDoc, t: number): number {
  let k = -1;
  for (let i = 0; i < line.words.length; i++) {
    if (line.words[i].start <= t) k = i;
    else break;
  }
  return k;
}

/** Index of the line being sung at t: the last line that has started (−1 before the first). */
export function activeLine(lines: LineDoc[], t: number): number {
  let lo = 0;
  let hi = lines.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (lines[mid].start <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo - 1;
}
