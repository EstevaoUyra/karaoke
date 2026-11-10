import { describe, expect, it } from "vitest";
import { demoDoc } from "./fixtures";
import { missingSpans, subtract, total, union } from "./melody";
import { ctxToSong, hzToMidi, noteFreq, noteName, planNotes, songToCtx, type SynthNote } from "./synth";

const notes: SynthNote[] = [
  { start: 1.0, end: 1.5, midi: 69, conf: 1 },
  { start: 1.5, end: 3.0, midi: 72, conf: 1 },
  { start: 4.0, end: 4.02, midi: 60, conf: 1 }, // too short to bother
  { start: 5.0, end: 6.0, midi: 57, conf: 0.5 },
];

describe("synth pitch", () => {
  it("A4 is 440 Hz, an octave doubles", () => {
    expect(noteFreq(69)).toBe(440);
    expect(noteFreq(81)).toBeCloseTo(880, 9);
    expect(noteFreq(60)).toBeCloseTo(261.626, 3);
  });

  it("adds the singer's tuning", () => {
    expect(noteFreq(69, 0.5)).toBeCloseTo(440 * 2 ** (0.5 / 12), 9); // a quarter tone sharp
  });

  it("slowed playback lowers the music, so the synth goes down by the same factor", () => {
    expect(noteFreq(69, 0, 0.5)).toBe(220); // half speed = an octave down, like the buffer
    expect(noteFreq(69, 0, 0.75)).toBeCloseTo(330, 9);
    expect(hzToMidi(noteFreq(64, 0, 0.75))).toBeCloseTo(64 + 12 * Math.log2(0.75), 9);
  });

  it("names notes", () => {
    expect(noteName(60)).toBe("C4");
    expect(noteName(69)).toBe("A4");
    expect(noteName(70)).toBe("B♭4");
  });
});

describe("synth scheduling", () => {
  const clock = { startedAt: 100, offset: 0, rate: 1 };

  it("maps song time to the audio clock and back, at any rate", () => {
    const slow = { startedAt: 100, offset: 10, rate: 0.5 };
    expect(songToCtx(12, slow)).toBe(104); // 2 s of song take 4 s at half speed
    expect(ctxToSong(104, slow)).toBe(12);
    expect(songToCtx(10, slow)).toBe(100);
  });

  it("plans the notes in the window, once each", () => {
    const done = new Set<number>();
    const first = planNotes(notes, 0, 2, clock, 0, done);
    expect(first.map((p) => p.index)).toEqual([0, 1]);
    expect(first[0]).toMatchObject({ when: 101, stop: 101.5, freq: 440 });
    first.forEach((p) => done.add(p.index));
    const second = planNotes(notes, 2, 7, clock, 0, done);
    expect(second.map((p) => p.index)).toEqual([3]); // note 1 already scheduled, note 2 too short
  });

  it("seeking into a long note starts it right away", () => {
    const seek = { startedAt: 200, offset: 2.0, rate: 1 };
    const plan = planNotes(notes, 2.0, 4, seek, 0, new Set());
    expect(plan).toHaveLength(1);
    expect(plan[0]).toMatchObject({ index: 1, when: 200, stop: 201 });
  });

  it("at half speed notes last twice as long and sound an octave lower", () => {
    const half = { startedAt: 0, offset: 0, rate: 0.5 };
    const [a, b] = planNotes(notes, 0, 3, half, 0, new Set());
    expect([a.when, a.stop, a.freq]).toEqual([2, 3, 220]);
    expect([b.when, b.stop]).toEqual([3, 6]);
  });

  it("applies the tuning to every planned note", () => {
    const [a] = planNotes(notes, 0, 1.2, clock, -0.25, new Set());
    expect(a.freq).toBeCloseTo(noteFreq(69, -0.25), 9);
  });
});

describe("gaps in the pitch guide", () => {
  it("union and subtract", () => {
    expect(union([{ start: 2, end: 3 }, { start: 0, end: 1 }, { start: 1.05, end: 1.5 }], 0.1)).toEqual([
      { start: 0, end: 1.5 }, { start: 2, end: 3 }]);
    expect(subtract([{ start: 0, end: 10 }], [{ start: 1, end: 2 }, { start: 5, end: 12 }])).toEqual([
      { start: 0, end: 1 }, { start: 2, end: 5 }]);
  });

  it("finds sung words with no note", () => {
    const doc = demoDoc(); // line 1: 2.0-4.6, line 2: 4.8-7.0, line 3: 17.5-18.6
    const ns = [{ start: 2.0, end: 3.65 }, { start: 4.8, end: 7.0 }];
    const gaps = missingSpans(doc, ns);
    expect(gaps).toEqual([{ start: 3.65, end: 4.6 }, { start: 17.5, end: 18.6 }]);
    expect(total(gaps)).toBeCloseTo(2.05, 9); // words are joined across gaps under 0.12 s
    expect(missingSpans(doc, [{ start: 0, end: 30 }])).toEqual([]);
  });
});
