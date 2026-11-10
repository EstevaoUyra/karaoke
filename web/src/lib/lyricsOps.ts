// Pure edits of lyrics.final.json. Every word an edit changes gets origin "manual" (align never
// overwrites those) and conf 1 (a person checked it). Lines are refitted to their words afterwards,
// so a word never falls outside its line.
import type { LineDoc, LyricsDoc, WordDoc } from "../types";
import { round3 } from "./time";

export const NUDGE = 0.05;
export const MIN_WORD = 0.04;

export const isProtected = (w: WordDoc) => w.origin === "manual" || w.origin === "ai";

export function touch(w: WordDoc): WordDoc {
  return { ...w, start: round3(w.start), end: round3(w.end), origin: "manual", conf: 1 };
}

/** Line start/end = its words' extent. */
export function fitLine(line: LineDoc): LineDoc {
  if (!line.words.length) return line;
  const start = Math.min(...line.words.map((w) => w.start));
  const end = Math.max(...line.words.map((w) => w.end));
  return { ...line, start: round3(start), end: round3(end) };
}

function withLine(doc: LyricsDoc, li: number, fn: (line: LineDoc) => LineDoc | null): LyricsDoc {
  const lines = doc.lines.slice();
  const next = fn(lines[li]);
  if (next === null || !next.words.length) lines.splice(li, 1);
  else lines[li] = fitLine(next);
  return { ...doc, lines };
}

function withWords(doc: LyricsDoc, li: number, fn: (words: WordDoc[]) => WordDoc[]): LyricsDoc {
  return withLine(doc, li, (line) => ({ ...line, words: fn(line.words.slice()) }));
}

/** Split [start, end] into consecutive spans weighted by text length. */
export function distribute(texts: string[], start: number, end: number): WordDoc[] {
  const weights = texts.map((t) => Math.max(1, t.replace(/[^\p{L}\p{N}]/gu, "").length));
  const total = weights.reduce((a, b) => a + b, 0);
  const span = Math.max(end - start, MIN_WORD * texts.length);
  let t = start;
  return texts.map((text, i) => {
    const d = (span * weights[i]) / total;
    const w: WordDoc = touch({ text, start: t, end: t + d, conf: 1, origin: "manual" });
    t += d;
    return w;
  });
}

export const tokenize = (text: string) => text.split(/\s+/).filter(Boolean);

// ------------------------------------------------------------------------------ words

export function setWordText(doc: LyricsDoc, li: number, wi: number, text: string): LyricsDoc {
  const tokens = tokenize(text);
  if (tokens.length === 0) return deleteWord(doc, li, wi);
  if (tokens.length > 1) return splitWord(doc, li, wi, tokens);
  return withWords(doc, li, (ws) => {
    ws[wi] = touch({ ...ws[wi], text: tokens[0] });
    return ws;
  });
}

export function setWordTiming(doc: LyricsDoc, li: number, wi: number, start: number, end: number): LyricsDoc {
  return withWords(doc, li, (ws) => {
    const s = Math.max(0, start);
    ws[wi] = touch({ ...ws[wi], start: s, end: Math.max(end, s + MIN_WORD) });
    return ws;
  });
}

/**
 * Move a word's start, end, or both by `delta` seconds. Moving into a neighbour moves the shared
 * boundary (the neighbour shrinks, but never below MIN_WORD).
 */
export function nudgeWord(doc: LyricsDoc, li: number, wi: number, which: "start" | "end" | "both",
                          delta: number): LyricsDoc {
  return withWords(doc, li, (ws) => {
    const w = { ...ws[wi] };
    const prev = wi > 0 ? ws[wi - 1] : null;
    const next = wi < ws.length - 1 ? ws[wi + 1] : null;
    if (which === "both") {
      w.start += delta;
      w.end += delta;
    } else if (which === "start") w.start = Math.min(w.start + delta, w.end - MIN_WORD);
    else w.end = Math.max(w.end + delta, w.start + MIN_WORD);
    // keep the order: never start before the previous word starts, never end after the next ends
    if (prev && w.start < prev.start + MIN_WORD) {
      const d = prev.start + MIN_WORD - w.start;
      w.start += d;
      if (which === "both") w.end += d;
    }
    if (next && w.end > next.end - MIN_WORD) {
      const d = w.end - (next.end - MIN_WORD);
      w.end -= d;
      if (which === "both") w.start -= d;
    }
    w.start = Math.max(0, w.start);
    w.end = Math.max(w.end, w.start + MIN_WORD);
    if (prev && prev.end > w.start) ws[wi - 1] = touch({ ...prev, end: w.start });
    if (next && next.start < w.end) ws[wi + 1] = touch({ ...next, start: w.end });
    ws[wi] = touch(w);
    return ws;
  });
}

/** Replace one word by several, sharing its time by text length. */
export function splitWord(doc: LyricsDoc, li: number, wi: number, parts: string[]): LyricsDoc {
  const clean = parts.map((p) => p.trim()).filter(Boolean);
  if (clean.length < 2) return doc;
  return withWords(doc, li, (ws) => {
    const w = ws[wi];
    ws.splice(wi, 1, ...distribute(clean, w.start, w.end));
    return ws;
  });
}

/** Split a word's text at a character position ("anymore", 3 → "any" + "more"). */
export function splitWordAt(doc: LyricsDoc, li: number, wi: number, at: number): LyricsDoc {
  const t = doc.lines[li].words[wi].text;
  if (at <= 0 || at >= t.length) return doc;
  return splitWord(doc, li, wi, [t.slice(0, at), t.slice(at)]);
}

/** Join a word with the next one in its line, as one word without a space. */
export function mergeWithNext(doc: LyricsDoc, li: number, wi: number): LyricsDoc {
  const ws0 = doc.lines[li].words;
  if (wi >= ws0.length - 1) return doc;
  return withWords(doc, li, (ws) => {
    const a = ws[wi];
    const b = ws[wi + 1];
    ws.splice(wi, 2, touch({ ...a, text: a.text + b.text, end: Math.max(a.end, b.end) }));
    return ws;
  });
}

export function deleteWord(doc: LyricsDoc, li: number, wi: number): LyricsDoc {
  return withWords(doc, li, (ws) => {
    ws.splice(wi, 1);
    return ws;
  });
}

/** Insert a word at position `at` of the line, in the gap there or in half of a neighbour's time. */
export function insertWord(doc: LyricsDoc, li: number, at: number, text: string): LyricsDoc {
  const t = text.trim();
  if (!t) return doc;
  return withWords(doc, li, (ws) => {
    const prev = at > 0 ? ws[at - 1] : null;
    const next = at < ws.length ? ws[at] : null;
    let start: number;
    let end: number;
    const gapStart = prev ? prev.end : next!.start - 0.5;
    const gapEnd = next ? next.start : prev!.end + 0.5;
    if (gapEnd - gapStart >= 0.1) {
      start = gapStart;
      end = Math.min(gapEnd, gapStart + 0.6);
    } else if (prev) {
      const mid = (prev.start + prev.end) / 2;
      ws[at - 1] = touch({ ...prev, end: mid });
      start = mid;
      end = prev.end;
    } else {
      const mid = (next!.start + next!.end) / 2;
      ws[at] = touch({ ...next!, start: mid });
      start = next!.start;
      end = mid;
    }
    ws.splice(at, 0, touch({ text: t, start: Math.max(0, start), end, conf: 1, origin: "manual" }));
    return ws;
  });
}

// ------------------------------------------------------------------------------ lines

const norm = (s: string) => s.toLowerCase().normalize("NFD").replace(/[^\p{L}\p{N}]/gu, "");

/** Longest common subsequence of a and b (by normalised text): matched index pairs. */
function lcsPairs(a: string[], b: string[]): [number, number][] {
  const n = a.length;
  const m = b.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = norm(a[i]) === norm(b[j]) ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out: [number, number][] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (norm(a[i]) === norm(b[j])) {
      out.push([i, j]);
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) i++;
    else j++;
  }
  return out;
}

/**
 * Retype a whole line. Words that are still there keep their timing (and only get marked manual if
 * their text changed); new words share the time between the kept words around them.
 */
export function setLineText(doc: LyricsDoc, li: number, text: string): LyricsDoc {
  const tokens = tokenize(text);
  if (!tokens.length) return deleteLine(doc, li);
  return withLine(doc, li, (line) => {
    const old = line.words;
    if (tokens.length === old.length) {
      return { ...line, words: old.map((w, i) => (w.text === tokens[i] ? w : touch({ ...w, text: tokens[i] }))) };
    }
    const pairs = lcsPairs(old.map((w) => w.text), tokens);
    const keptAt = new Map(pairs.map(([i, j]) => [j, old[i]]));
    const words: (WordDoc | null)[] = tokens.map((tok, j) => {
      const w = keptAt.get(j);
      return w ? (w.text === tok ? w : touch({ ...w, text: tok })) : null;
    });
    // fill runs of new tokens between kept anchors
    let j = 0;
    while (j < tokens.length) {
      if (words[j]) {
        j++;
        continue;
      }
      let k = j;
      while (k < tokens.length && !words[k]) k++;
      const prev = j > 0 ? words[j - 1] : null;
      const next = k < tokens.length ? words[k] : null;
      let start = prev ? prev.end : line.start;
      let end = next ? next.start : line.end;
      if (end - start < MIN_WORD * (k - j)) {
        // no room: take the second half of the previous word (or the first half of the next)
        if (prev) {
          const mid = (prev.start + prev.end) / 2;
          words[j - 1] = touch({ ...prev, end: mid });
          start = mid;
          end = Math.max(end, prev.end);
        } else if (next) {
          const mid = (next.start + next.end) / 2;
          words[k] = touch({ ...next, start: mid });
          start = next.start;
          end = mid;
        } else end = start + 0.4 * (k - j);
      }
      distribute(tokens.slice(j, k), start, end).forEach((w, n) => (words[j + n] = w));
      j = k;
    }
    return { ...line, words: words as WordDoc[] };
  });
}

export function shiftLine(doc: LyricsDoc, li: number, delta: number): LyricsDoc {
  return withWords(doc, li, (ws) => {
    const d = Math.max(delta, -Math.min(...ws.map((w) => w.start)));
    return ws.map((w) => touch({ ...w, start: w.start + d, end: w.end + d }));
  });
}

export function setLineSection(doc: LyricsDoc, li: number, section: string | null): LyricsDoc {
  return withLine(doc, li, (line) => ({ ...line, section: section?.trim() || null }));
}

export function deleteLine(doc: LyricsDoc, li: number): LyricsDoc {
  return withLine(doc, li, () => null);
}

export function newLineId(doc: LyricsDoc, after: string | null): string {
  const ids = new Set(doc.lines.map((l) => l.id));
  const base = after ?? "l000";
  for (let n = 0; ; n++) {
    let suffix = "";
    let k = n;
    do {
      suffix = String.fromCharCode(97 + (k % 26)) + suffix;
      k = Math.floor(k / 26) - 1;
    } while (k >= 0);
    if (!ids.has(base + suffix)) return base + suffix;
  }
}

/** Add a line after line `li` (-1: before the first line), in the gap there. */
export function addLineAfter(doc: LyricsDoc, li: number, text: string, duration?: number): LyricsDoc {
  const tokens = tokenize(text);
  if (!tokens.length) return doc;
  const prev = li >= 0 ? doc.lines[li] : null;
  const next = li + 1 < doc.lines.length ? doc.lines[li + 1] : null;
  let start = prev ? prev.end + 0.1 : Math.max(0, (next?.start ?? 2.6) - 2.6);
  let end = next ? next.start - 0.1 : start + 2.5;
  if (duration !== undefined) end = Math.min(end, duration);
  end = Math.min(end, start + 4);
  if (end - start < 0.3) {
    start = prev ? prev.end : Math.max(0, start);
    end = start + Math.max(0.3, 0.25 * tokens.length);
  }
  const line: LineDoc = fitLine({
    id: newLineId(doc, prev?.id ?? null), section: prev?.section ?? next?.section ?? null, singer: null,
    start, end, words: distribute(tokens, start, end), flags: [],
  });
  const lines = doc.lines.slice();
  lines.splice(li + 1, 0, line);
  return { ...doc, lines };
}

/** Break a line in two before word `wi`. */
export function splitLineAt(doc: LyricsDoc, li: number, wi: number): LyricsDoc {
  const line = doc.lines[li];
  if (wi <= 0 || wi >= line.words.length) return doc;
  const a = fitLine({ ...line, words: line.words.slice(0, wi) });
  const b = fitLine({ ...line, id: newLineId(doc, line.id), words: line.words.slice(wi) });
  const lines = doc.lines.slice();
  lines.splice(li, 1, a, b);
  return { ...doc, lines };
}

export function mergeLineWithNext(doc: LyricsDoc, li: number): LyricsDoc {
  if (li >= doc.lines.length - 1) return doc;
  const a = doc.lines[li];
  const b = doc.lines[li + 1];
  const merged = fitLine({ ...a, words: [...a.words, ...b.words], flags: [...new Set([...a.flags, ...b.flags])] });
  const lines = doc.lines.slice();
  lines.splice(li, 2, merged);
  return { ...doc, lines };
}

const REVIEW_FLAGS = new Set(["low_confidence", "timing_suspect"]);

/** "This line is right": every word becomes manual (protected from align) and the doubt flags go. */
export function markLineReviewed(doc: LyricsDoc, li: number): LyricsDoc {
  return withLine(doc, li, (line) => ({
    ...line, words: line.words.map(touch), flags: line.flags.filter((f) => !REVIEW_FLAGS.has(f)),
  }));
}

// ------------------------------------------------------------------------------ queries

export type Doubt = "low" | "unsure" | null;

/** How much to doubt a word: low = likely wrong (interpolated or conf < 0.5), unsure = conf < 0.75. */
export function wordDoubt(w: WordDoc): Doubt {
  if (isProtected(w)) return null;
  if (w.origin === "interpolated" || w.conf < 0.5) return "low";
  if (w.conf < 0.75) return "unsure";
  return null;
}

export function lineNeedsReview(line: LineDoc): boolean {
  return line.flags.some((f) => REVIEW_FLAGS.has(f)) && !line.words.every(isProtected);
}

export const lineText = (line: LineDoc) => line.words.map((w) => w.text).join(" ");

export function countManual(doc: LyricsDoc): number {
  return doc.lines.reduce((n, l) => n + l.words.filter((w) => w.origin === "manual").length, 0);
}
