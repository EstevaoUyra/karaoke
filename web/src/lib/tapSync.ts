// Tap-to-sync: press a key on each word (or line) while the song plays.
//
// Word mode: key DOWN = the word starts; key UP = it ends. When the next word of the same line is
// pressed, the previous word is stretched to meet it (legato), so the hold length only matters
// for the last word of a line (a held note). Line mode: key DOWN moves the whole line so it
// starts there, keeping its word rhythm.
//
// Times are the player's clock minus `latency` (people press slightly late). Words not tapped yet
// are pushed later only when needed to keep the line in order; everything changed becomes manual.
import type { LyricsDoc } from "../types";
import { MIN_WORD, fitLine, touch } from "./lyricsOps";
import { round3 } from "./time";

export type TapMode = "word" | "line";

export interface TapTarget {
  li: number;
  wi: number;
}

export interface TapSession {
  mode: TapMode;
  targets: TapTarget[];
  index: number; // next target to tap
  down: boolean; // key held on targets[index]
  latency: number;
}

export function startTap(doc: LyricsDoc, mode: TapMode, fromLi: number, fromWi = 0, latency = 0): TapSession {
  const targets: TapTarget[] = [];
  doc.lines.forEach((line, li) => {
    if (li < fromLi) return;
    if (mode === "line") targets.push({ li, wi: 0 });
    else line.words.forEach((_, wi) => (li > fromLi || wi >= fromWi) && targets.push({ li, wi }));
  });
  return { mode, targets, index: 0, down: false, latency };
}

export const tapDone = (s: TapSession) => s.index >= s.targets.length;
export const currentTarget = (s: TapSession): TapTarget | null => s.targets[s.index] ?? null;

/** Push words after `wi` later if they now start before the previous word ends. */
function settleLine(doc: LyricsDoc, li: number, wi: number): LyricsDoc {
  const words = doc.lines[li].words.slice();
  for (let k = wi + 1; k < words.length; k++) {
    const prevEnd = words[k - 1].end;
    if (words[k].start >= prevEnd - 1e-6) break;
    const d = prevEnd - words[k].start;
    words[k] = touch({ ...words[k], start: words[k].start + d, end: words[k].end + d });
  }
  const lines = doc.lines.slice();
  lines[li] = fitLine({ ...lines[li], words });
  return settleLinesAfter({ ...doc, lines }, li);
}

/** Push following lines later while they would overlap the line before by more than 0.25 s. */
function settleLinesAfter(doc: LyricsDoc, li: number): LyricsDoc {
  let lines = doc.lines;
  for (let k = li + 1; k < lines.length; k++) {
    const prev = lines[k - 1];
    if (lines[k].start >= prev.end - 0.25 && lines[k].start >= prev.start) break;
    const d = prev.end - lines[k].start;
    if (lines === doc.lines) lines = lines.slice();
    lines[k] = fitLine({ ...lines[k], words: lines[k].words.map((w) => touch({ ...w, start: w.start + d, end: w.end + d })) });
  }
  return lines === doc.lines ? doc : { ...doc, lines };
}

function setWord(doc: LyricsDoc, li: number, wi: number, start: number, end: number): LyricsDoc {
  const lines = doc.lines.slice();
  const words = lines[li].words.slice();
  words[wi] = touch({ ...words[wi], start: round3(start), end: round3(Math.max(end, start + MIN_WORD)) });
  lines[li] = fitLine({ ...lines[li], words });
  return { ...doc, lines };
}

export function tapDown(doc: LyricsDoc, s: TapSession, clock: number): { doc: LyricsDoc; session: TapSession } {
  const target = currentTarget(s);
  if (!target || s.down) return { doc, session: s };
  const t = Math.max(0, clock - s.latency);
  const { li, wi } = target;
  if (s.mode === "line") {
    const line = doc.lines[li];
    const d = t - line.start;
    const lines = doc.lines.slice();
    lines[li] = fitLine({ ...line, words: line.words.map((w) => touch({ ...w, start: w.start + d, end: w.end + d })) });
    return { doc: settleLinesAfter({ ...doc, lines }, li), session: { ...s, index: s.index + 1 } };
  }
  let next = doc;
  let start = t;
  const prev = s.index > 0 ? s.targets[s.index - 1] : null;
  if (prev && prev.li === li) {
    // legato: the previous word lasts until this one starts
    const pw = next.lines[li].words[prev.wi];
    start = Math.max(t, pw.start + MIN_WORD);
    next = setWord(next, li, prev.wi, pw.start, start);
  }
  const w = next.lines[li].words[wi];
  const dur = Math.max(w.end - w.start, MIN_WORD);
  next = setWord(next, li, wi, start, start + dur); // provisional end until the key comes up
  return { doc: settleLine(next, li, wi), session: { ...s, down: true } };
}

export function tapUp(doc: LyricsDoc, s: TapSession, clock: number): { doc: LyricsDoc; session: TapSession } {
  const target = currentTarget(s);
  if (!target || !s.down || s.mode === "line") return { doc, session: { ...s, down: false } };
  const t = Math.max(0, clock - s.latency);
  const { li, wi } = target;
  const w = doc.lines[li].words[wi];
  const next = setWord(doc, li, wi, w.start, Math.max(t, w.start + MIN_WORD));
  return { doc: settleLine(next, li, wi), session: { ...s, down: false, index: s.index + 1 } };
}
