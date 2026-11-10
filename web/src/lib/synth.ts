// The melody synth's math, kept pure so it can be tested: which notes to schedule, when (on the
// AudioContext clock), and at what frequency.
//
// Playback maps song time s to context time: ctx = startedAt + (s - offset) / rate.
// A slowed-down buffer (playbackRate < 1) also sounds lower by the same factor, so the synth
// multiplies its frequencies by `rate` to stay in tune with the music.

export interface SynthNote {
  start: number;
  end: number;
  midi: number;
  conf: number;
}

export interface Clock {
  startedAt: number; // context time at which song time `offset` plays
  offset: number;
  rate: number;
}

export interface Planned {
  index: number; // index into the notes array
  when: number; // context time the note starts
  stop: number; // context time the note ends (before the release tail)
  freq: number;
}

export const ATTACK = 0.012;
export const RELEASE = 0.08;
export const MIN_NOTE = 0.03;

/** Frequency of a MIDI note, with the singer's tuning (semitones off A440) and the playback rate. */
export function noteFreq(midi: number, tuning = 0, rate = 1): number {
  return 440 * 2 ** ((midi + tuning - 69) / 12) * rate;
}

export const songToCtx = (s: number, c: Clock) => c.startedAt + (s - c.offset) / c.rate;
export const ctxToSong = (t: number, c: Clock) => c.offset + (t - c.startedAt) * c.rate;

/**
 * Notes to schedule for song time [from, to): those that sound in that window and are not in
 * `done`. A note already under way at `from` starts right away (at `from`), so seeking into the
 * middle of a long note still plays it.
 */
export function planNotes(notes: SynthNote[], from: number, to: number, clock: Clock, tuning: number,
                          done: Set<number>): Planned[] {
  const out: Planned[] = [];
  for (let i = firstEndingAfter(notes, from); i < notes.length; i++) {
    const n = notes[i];
    if (n.start >= to) break;
    if (done.has(i) || n.end <= from) continue;
    const s = Math.max(n.start, from);
    if (n.end - s < MIN_NOTE) continue;
    out.push({ index: i, when: songToCtx(s, clock), stop: songToCtx(n.end, clock), freq: noteFreq(n.midi, tuning, clock.rate) });
  }
  return out;
}

/** First note whose end is after t (notes sorted by start and not overlapping much). */
function firstEndingAfter(notes: SynthNote[], t: number): number {
  let lo = 0;
  let hi = notes.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (notes[mid].end <= t) lo = mid + 1;
    else hi = mid;
  }
  // a long earlier note can still be sounding: step back over overlaps
  while (lo > 0 && notes[lo - 1].end > t) lo--;
  return lo;
}

/** Peak gain of a note: quieter when the tracker was unsure. */
export const noteGain = (conf: number) => 0.18 + 0.17 * Math.max(0, Math.min(1, conf));

const NAMES = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"];
export const noteName = (midi: number) => `${NAMES[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
export const hzToMidi = (hz: number) => 69 + 12 * Math.log2(hz / 440);
