import { describe, expect, it } from "vitest";
import type { MelodyDoc } from "../types";
import { guideGaps } from "./melody";
import { demoDoc } from "./fixtures";

const melody: MelodyDoc = {
  version: 2, hop_s: 0.01, key: null, median_midi: 60, tuning: 0, display_octave_shift: 0,
  params: {}, updated_at: "2026-11-11T00:00:00Z",
  syllables: [{ line_id: "l1", word_index: 0, text: "Sing", start: 1, end: 1.5 }],
  notes: [{ start: 1, end: 1.5, midi: 60, pitch: 60, conf: 0, syllable: 0 }],
};

describe("syllable coverage", () => {
  it("counts uncertain notes and does not report trimmed rests as missing syllables", () => {
    expect(guideGaps(demoDoc(), melody)).toEqual([]);
  });
  it("reports a syllable whose pitch is missing", () => {
    expect(guideGaps(null, { ...melody, notes: [] })).toEqual([{ start: 1, end: 1.5 }]);
  });
});
