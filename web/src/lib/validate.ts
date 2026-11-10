// Mirror of karaoke.validate.validate_lyrics, for instant feedback while editing.
// The server runs the real check on save.
import type { Issue, LyricsDoc } from "../types";

const EPS = 0.02;

export interface LyricsIssue extends Issue {
  lineId: string;
  wordIndex: number | null;
}

export function validateLyrics(doc: LyricsDoc, duration?: number | null): LyricsIssue[] {
  const out: LyricsIssue[] = [];
  const add = (lineId: string, wordIndex: number | null, where: string, message: string,
               severity: Issue["severity"] = "error") =>
    out.push({ severity, file: "lyrics.final.json", where, message, lineId, wordIndex });
  const seen = new Set<string>();
  let prevStart = -1;
  let prevEnd = 0;
  for (const line of doc.lines) {
    const w = `line ${line.id}`;
    if (seen.has(line.id)) add(line.id, null, w, "duplicate line id");
    seen.add(line.id);
    if (!line.words.length) {
      add(line.id, null, w, "line has no words");
      continue;
    }
    if (line.end < line.start - EPS) add(line.id, null, w, `end ${line.end.toFixed(2)} < start ${line.start.toFixed(2)}`);
    if (line.start < prevStart - EPS)
      add(line.id, null, w, `lines out of order (starts ${line.start.toFixed(2)} before previous ${prevStart.toFixed(2)})`);
    if (line.start < prevEnd - 0.25)
      add(line.id, null, w, `overlaps previous line by ${(prevEnd - line.start).toFixed(2)}s`, "warning");
    if (duration != null && (line.start < -EPS || line.end > duration + EPS))
      add(line.id, null, w, `outside audio duration 0..${duration.toFixed(2)}`);
    let last = line.start - EPS;
    line.words.forEach((word, i) => {
      const ww = `${w} word ${i} '${word.text}'`;
      if (!word.text.trim()) add(line.id, i, ww, "empty word text");
      if (word.end < word.start - EPS) add(line.id, i, ww, `end ${word.end.toFixed(2)} < start ${word.start.toFixed(2)}`);
      if (word.start < last - EPS) add(line.id, i, ww, "word starts before previous word");
      const inside = (t: number) => line.start - EPS <= t && t <= line.end + EPS;
      if (!(inside(word.start) && inside(word.end))) add(line.id, i, ww, "word outside its line bounds");
      if (word.end - word.start > 5) add(line.id, i, ww, `suspiciously long word (${(word.end - word.start).toFixed(1)}s)`, "warning");
      last = word.start;
    });
    prevStart = line.start;
    prevEnd = Math.max(prevEnd, line.end);
  }
  return out;
}
