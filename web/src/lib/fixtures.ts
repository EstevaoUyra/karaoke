// A small lyrics document for tests (the shape of tests/fixtures/make_demo.py's song).
import type { LineDoc, LyricsDoc, WordDoc } from "../types";

export function w(text: string, start: number, end: number, origin: WordDoc["origin"] = "matched", conf = 0.9): WordDoc {
  return { text, start, end, conf, origin };
}

export function line(id: string, words: WordDoc[], flags: string[] = []): LineDoc {
  return {
    id, section: "verse", singer: null, flags, words,
    start: Math.min(...words.map((x) => x.start)), end: Math.max(...words.map((x) => x.end)),
  };
}

export function demoDoc(): LyricsDoc {
  return {
    version: 1, language: "en", text_source: "lrclib", unmatched_audio: [], updated_at: "2026-11-09T00:00:00Z",
    lines: [
      line("l001", [w("Morning", 2.0, 2.55), w("light", 2.6, 3.0), w("is", 3.05, 3.2), w("on", 3.25, 3.45),
                    w("the", 3.5, 3.65), w("widow,", 3.7, 4.6, "fuzzy", 0.4)], ["low_confidence"]),
      line("l002", [w("coffee", 4.8, 5.3), w("cooling", 5.35, 5.9), w("by", 5.95, 6.15), w("the", 6.2, 6.35),
                    w("door", 6.4, 7.0)]),
      line("l003", [w("Sing", 17.5, 17.9, "interpolated", 0.3), w("it", 17.95, 18.1), w("loud", 18.15, 18.6)]),
    ],
  };
}
