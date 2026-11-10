import { describe, expect, it } from "vitest";
import { demoDoc } from "./fixtures";
import { startTap, tapDone, tapDown, tapUp } from "./tapSync";
import { validateLyrics } from "./validate";
import { stageView, wordFill } from "./playalong";

function taps(mode: "word" | "line", fromLi: number, presses: [number, number][], latency = 0) {
  let doc = demoDoc();
  let session = startTap(doc, mode, fromLi, 0, latency);
  for (const [down, up] of presses) {
    ({ doc, session } = tapDown(doc, session, down));
    ({ doc, session } = tapUp(doc, session, up));
  }
  return { doc, session };
}

describe("tap-to-sync, word mode", () => {
  it("key down starts a word, the next key down ends it (legato)", () => {
    const { doc } = taps("word", 2, [[17.0, 17.1], [17.6, 17.7], [18.0, 18.9]]);
    const ws = doc.lines[2].words;
    expect(ws.map((x) => [x.start, x.end])).toEqual([[17.0, 17.6], [17.6, 18.0], [18.0, 18.9]]);
    expect(ws.every((x) => x.origin === "manual")).toBe(true);
    expect(doc.lines[2]).toMatchObject({ start: 17.0, end: 18.9 });
  });

  it("the last word of a line ends when the key comes up (a held note)", () => {
    const { doc } = taps("word", 2, [[17.5, 17.5], [17.9, 17.9], [18.1, 19.4]]);
    expect(doc.lines[2].words[2].end).toBe(19.4);
  });

  it("does not stretch the last word of a line to the next line", () => {
    const { doc } = taps("word", 0, [[2.0, 2.1], [2.6, 2.7], [3.0, 3.1], [3.2, 3.3], [3.5, 3.6], [3.7, 4.2], [4.9, 5.0]]);
    expect(doc.lines[0].words[5].end).toBe(4.2);
    expect(doc.lines[1].words[0].start).toBe(4.9);
  });

  it("subtracts the tap latency", () => {
    const { doc } = taps("word", 2, [[17.1, 17.2]], 0.1);
    expect(doc.lines[2].words[0].start).toBe(17.0);
  });

  it("tapping late pushes the untapped rest of the line, keeping it valid", () => {
    // "Sing" pressed 1 s late and held: "it" and "loud" must move after it
    const { doc } = taps("word", 2, [[18.5, 18.9]]);
    const ws = doc.lines[2].words;
    expect(ws[0]).toMatchObject({ start: 18.5, end: 18.9 });
    expect(ws[1].start).toBeGreaterThanOrEqual(18.9);
    expect(validateLyrics(doc, 24).filter((i) => i.severity === "error")).toEqual([]);
  });

  it("finishes after the last word", () => {
    const { session } = taps("word", 2, [[17, 17.1], [17.5, 17.6], [18, 18.1]]);
    expect(tapDone(session)).toBe(true);
  });
});

describe("tap-to-sync, line mode", () => {
  it("moves each tapped line to the tap, keeping its rhythm", () => {
    const { doc } = taps("line", 0, [[2.5, 2.6], [5.0, 5.1]]);
    expect(doc.lines[0].start).toBe(2.5);
    expect(doc.lines[0].words[1].start).toBeCloseTo(3.1, 6); // 0.6 s after the first word, as before
    expect(doc.lines[1].start).toBe(5.0);
    expect(doc.lines[2].words[0].origin).toBe("interpolated"); // not tapped, not touched
  });

  it("a late tap pushes an overlapping next line", () => {
    const { doc } = taps("line", 0, [[4.5, 4.6]]); // line 1 now ends at 7.1, line 2 started at 4.8
    expect(doc.lines[1].start).toBeGreaterThanOrEqual(doc.lines[0].end - 0.25);
    expect(validateLyrics(doc, 24).filter((i) => i.severity === "error")).toEqual([]);
  });
});

describe("play-along view", () => {
  const lines = demoDoc().lines;
  it("shows a line from its lead-in, with the next one", () => {
    expect(stageView(lines, 12.0, 1.5)).toEqual({ current: null, next: null, countdown: null });
    expect(stageView(lines, 0.0, 1.5)).toEqual({ current: null, next: 0, countdown: null }); // coming up, dimmed
    expect(stageView(lines, 0.8, 1.5)).toMatchObject({ current: 0, next: 1 });
    expect(stageView(lines, 4.7, 1.5)).toMatchObject({ current: 1, next: 2 });
  });

  it("counts down before a line after a long gap", () => {
    expect(stageView(lines, 15.2, 1.5, 4).countdown).toBe(3);
    expect(stageView(lines, 16.9, 1.5, 4)).toMatchObject({ current: 2, countdown: 1 });
  });

  it("fills words over their duration", () => {
    const word = lines[0].words[0];
    expect(wordFill(word, 1.9)).toBe(0);
    expect(wordFill(word, 2.275)).toBeCloseTo(0.5, 6);
    expect(wordFill(word, 3)).toBe(1);
  });
});
