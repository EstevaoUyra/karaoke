import { describe, expect, it } from "vitest";
import { demoDoc } from "./fixtures";
import {
  NUDGE, addLineAfter, deleteWord, insertWord, markLineReviewed, mergeLineWithNext, mergeWithNext, nudgeWord,
  setLineText, setWordText, shiftLine, splitLineAt, splitWordAt, wordDoubt,
} from "./lyricsOps";
import { validateLyrics } from "./validate";

const errors = (doc: ReturnType<typeof demoDoc>) => validateLyrics(doc, 24).filter((i) => i.severity === "error");

describe("edits mark words manual", () => {
  it("editing a word's text marks only that word", () => {
    const doc = setWordText(demoDoc(), 0, 5, "window,");
    const ws = doc.lines[0].words;
    expect(ws[5]).toMatchObject({ text: "window,", origin: "manual", conf: 1, start: 3.7, end: 4.6 });
    expect(ws.slice(0, 5).every((x) => x.origin === "matched")).toBe(true);
    expect(errors(doc)).toEqual([]);
  });

  it("the input document is never mutated", () => {
    const before = demoDoc();
    const snapshot = JSON.stringify(before);
    setWordText(before, 0, 0, "Evening");
    nudgeWord(before, 0, 1, "both", 0.05);
    expect(JSON.stringify(before)).toBe(snapshot);
  });

  it("typing a space into a word splits it, sharing the time", () => {
    const doc = setWordText(demoDoc(), 1, 0, "cof fee");
    const [a, b] = doc.lines[1].words;
    expect([a.text, b.text]).toEqual(["cof", "fee"]);
    expect(a.start).toBe(4.8);
    expect(b.end).toBe(5.3);
    expect(a.end).toBeCloseTo(b.start, 6);
    expect(a.origin).toBe("manual");
    expect(b.origin).toBe("manual");
  });

  it("split at a character and join back", () => {
    let doc = splitWordAt(demoDoc(), 1, 1, 4); // cooling → cool + ing
    expect(doc.lines[1].words.map((x) => x.text)).toEqual(["coffee", "cool", "ing", "by", "the", "door"]);
    doc = mergeWithNext(doc, 1, 1);
    expect(doc.lines[1].words[1]).toMatchObject({ text: "cooling", start: 5.35, end: 5.9, origin: "manual" });
  });

  it("deleting the last word of a line deletes the line", () => {
    let doc = demoDoc();
    for (let i = 0; i < 3; i++) doc = deleteWord(doc, 2, 0);
    expect(doc.lines.map((l) => l.id)).toEqual(["l001", "l002"]);
  });

  it("inserting a word into a gap-less spot takes half the neighbour", () => {
    const doc = insertWord(demoDoc(), 2, 3, "now");
    const ws = doc.lines[2].words;
    expect(ws.map((x) => x.text)).toEqual(["Sing", "it", "loud", "now"]);
    expect(ws[3].start).toBeGreaterThanOrEqual(ws[2].end);
    expect(errors(doc)).toEqual([]);
  });
});

describe("nudges", () => {
  it("moves a word by ±50 ms and refits the line", () => {
    const doc = nudgeWord(demoDoc(), 0, 0, "both", -NUDGE);
    expect(doc.lines[0].words[0]).toMatchObject({ start: 1.95, end: 2.5, origin: "manual" });
    expect(doc.lines[0].start).toBe(1.95);
  });

  it("pushing into a neighbour moves the shared boundary", () => {
    let doc = demoDoc();
    for (let i = 0; i < 3; i++) doc = nudgeWord(doc, 0, 0, "end", NUDGE); // 2.55 → 2.70, past "light" (2.60)
    const [a, b] = doc.lines[0].words;
    expect(a.end).toBeCloseTo(2.7, 6);
    expect(b.start).toBeCloseTo(2.7, 6);
    expect(b.origin).toBe("manual");
    expect(errors(doc)).toEqual([]);
  });

  it("a word cannot be nudged before the previous one starts", () => {
    let doc = demoDoc();
    for (let i = 0; i < 40; i++) doc = nudgeWord(doc, 0, 1, "both", -NUDGE);
    expect(doc.lines[0].words[1].start).toBeGreaterThan(doc.lines[0].words[0].start);
    expect(errors(doc)).toEqual([]);
  });

  it("shifting a line moves every word", () => {
    const doc = shiftLine(demoDoc(), 1, 0.25);
    expect(doc.lines[1].start).toBeCloseTo(5.05, 6);
    expect(doc.lines[1].words.every((x) => x.origin === "manual")).toBe(true);
  });
});

describe("lines", () => {
  it("retyping a line keeps the timing of words that stayed", () => {
    const doc = setLineText(demoDoc(), 0, "Morning light is on the window, again");
    const ws = doc.lines[0].words;
    expect(ws.map((x) => x.text)).toEqual(["Morning", "light", "is", "on", "the", "window,", "again"]);
    expect(ws[0]).toMatchObject({ start: 2.0, end: 2.55, origin: "matched" }); // untouched
    expect(ws[5].origin).toBe("manual"); // widow → window
    expect(ws[6].origin).toBe("manual");
    expect(errors(doc)).toEqual([]);
  });

  it("retyping with one word fewer", () => {
    const doc = setLineText(demoDoc(), 1, "coffee cooling by door");
    expect(doc.lines[1].words.map((x) => x.text)).toEqual(["coffee", "cooling", "by", "door"]);
    expect(doc.lines[1].words[3]).toMatchObject({ start: 6.4, end: 7.0, origin: "matched" });
  });

  it("add, split and merge lines stay valid", () => {
    let doc = addLineAfter(demoDoc(), 1, "yeah yeah");
    expect(doc.lines[2]).toMatchObject({ id: "l002a" });
    expect(doc.lines[2].start).toBeGreaterThan(doc.lines[1].end);
    expect(doc.lines[2].end).toBeLessThan(doc.lines[3].start);
    doc = splitLineAt(doc, 0, 3);
    expect(doc.lines[0].words.map((x) => x.text)).toEqual(["Morning", "light", "is"]);
    expect(doc.lines[1].id).toBe("l001a");
    expect(doc.lines[1].start).toBe(3.25);
    doc = mergeLineWithNext(doc, 0);
    expect(doc.lines[0].words).toHaveLength(6);
    expect(errors(doc)).toEqual([]);
  });

  it("marking a line reviewed protects it and clears the doubt", () => {
    const doc = markLineReviewed(demoDoc(), 0);
    expect(doc.lines[0].flags).toEqual([]);
    expect(doc.lines[0].words.every((x) => x.origin === "manual" && x.conf === 1)).toBe(true);
    expect(doc.lines[0].words.map(wordDoubt)).toEqual([null, null, null, null, null, null]);
  });
});

describe("doubt", () => {
  it("flags interpolated and low-confidence words", () => {
    const doc = demoDoc();
    expect(wordDoubt(doc.lines[0].words[5])).toBe("low");
    expect(wordDoubt(doc.lines[2].words[0])).toBe("low");
    expect(wordDoubt({ ...doc.lines[0].words[0], conf: 0.6 })).toBe("unsure");
    expect(wordDoubt(doc.lines[0].words[0])).toBe(null);
  });
});
