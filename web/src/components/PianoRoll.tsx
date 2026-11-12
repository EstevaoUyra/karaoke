// A scrolling piano roll of the detected melody under the play-along: note bars over the raw pitch
// contour (faint), amber bands where words are sung but no note was found, and the playhead.
import { useEffect, useMemo, useRef, useState } from "react";
import type { StemPlayer } from "../lib/player";
import { hzToMidi, noteName } from "../lib/synth";
import type { Span } from "../lib/melody";
import { useClock } from "../lib/usePlayer";
import type { MelodyDoc, ContourDoc } from "../types";

const WINDOW_S = 10;
const PLAYHEAD = 0.25; // fraction of the width left of the playhead
const HEIGHT = 160;

/** Rows shown: the notes' 2nd to 98th percentile plus a margin, at least 18 semitones. Outliers
 *  (octave errors, a stray shout) are drawn clipped at the top or bottom edge. */
export function pitchRange(melody: MelodyDoc): [number, number] {
  const ms = melody.notes.map((n) => n.midi).sort((a, b) => a - b);
  if (!ms.length) return [55, 79];
  const q = (f: number) => ms[Math.min(ms.length - 1, Math.max(0, Math.round(f * (ms.length - 1))))];
  let lo = q(0.02) - 2;
  let hi = q(0.98) + 2;
  if (hi - lo < 18) {
    const mid = (hi + lo) / 2;
    lo = Math.floor(mid - 9);
    hi = Math.ceil(mid + 9);
  }
  return [lo, hi];
}

export function PianoRoll({ player, melody, contour, missing, onSeek }: {
  player: StemPlayer;
  melody: MelodyDoc;
  contour: ContourDoc | null;
  missing: Span[];
  onSeek: (t: number) => void;
}) {
  const t = useClock(player);
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(800);
  const [lo, hi] = useMemo(() => pitchRange(melody), [melody]);
  const contourPts = useMemo(() => {
    if (!contour) return [];
    const out: { t: number; m: number; c: number }[] = [];
    contour.t.forEach((tt, i) => {
      const hz = contour.hz[i];
      if (hz) out.push({ t: tt, m: hzToMidi(hz) - melody.tuning, c: contour.conf[i] });
    });
    return out;
  }, [contour, melody.tuning]);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(Math.max(200, Math.round(el.clientWidth))));
    ro.observe(el);
    setWidth(Math.max(200, Math.round(el.clientWidth)));
    return () => ro.disconnect();
  }, []);

  const pxPerS = width / WINDOW_S;
  const t0 = t - WINDOW_S * PLAYHEAD;
  const x = (s: number) => (s - t0) * pxPerS;
  const rowH = HEIGHT / (hi - lo + 1);
  const y = (m: number) => HEIGHT - (Math.max(lo, Math.min(hi, m)) - lo + 1) * rowH; // outliers clipped to the edges

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== width * dpr) {
      c.width = width * dpr;
      c.height = HEIGHT * dpr;
    }
    const g = c.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, HEIGHT);
    // octave lines (C) and labels
    g.font = "10px system-ui, sans-serif";
    for (let m = Math.ceil(lo); m <= hi; m++) {
      if (m % 12 !== 0) continue;
      g.fillStyle = "rgba(255,255,255,0.08)";
      g.fillRect(0, y(m) + rowH, width, 1);
      g.fillStyle = "rgba(255,255,255,0.35)";
      g.fillText(noteName(m), 4, y(m) + rowH - 2);
    }
    if (melody.key) {
      for (let m = Math.ceil(lo); m <= hi; m++) {
        if (((m - melody.key.tonic_pc) % 12 + 12) % 12 === 0) {
          g.fillStyle = "rgba(255,255,255,0.04)";
          g.fillRect(0, y(m), width, rowH);
        }
      }
    }
    const tEnd = t0 + WINDOW_S;
    // sung but no note
    for (const s of missing) {
      if (s.end < t0 || s.start > tEnd) continue;
      g.fillStyle = "rgba(242, 179, 61, 0.16)";
      g.fillRect(x(s.start), 0, (s.end - s.start) * pxPerS, HEIGHT);
      g.fillStyle = "rgba(242, 179, 61, 0.9)";
      g.fillRect(x(s.start), 0, (s.end - s.start) * pxPerS, 3);
    }
    // raw contour
    for (const p of contourPts) {
      if (p.t < t0 || p.t > tEnd) continue;
      g.fillStyle = `rgba(255,255,255,${(0.12 + 0.35 * p.c).toFixed(3)})`;
      g.fillRect(x(p.t) - 1, y(p.m) + rowH / 2 - 1, 2.2, 2.2);
    }
    // notes
    let current: (typeof melody.notes)[number] | null = null;
    for (const n of melody.notes) {
      if (n.end < t0 || n.start > tEnd) continue;
      const on = n.start <= t && t < n.end;
      if (on) current = n;
      const a = (on ? 1 : 0.55) * (0.2 + 0.8 * n.conf);
      g.fillStyle = on ? `rgba(255,255,255,${a.toFixed(3)})` : `rgba(79,195,247,${a.toFixed(3)})`;
      const w = Math.max(2, (n.end - n.start) * pxPerS - 1);
      const r = Math.min(3, rowH / 2, w / 2);
      g.beginPath();
      g.roundRect(x(n.start), y(n.midi) + 1, w, Math.max(2, rowH - 2), r);
      g.fill();
    }
    // Explicit syllable starts remain visible even when adjacent pitches agree.
    g.fillStyle = "rgba(255,255,255,0.6)";
    for (const sy of melody.syllables) {
      if (sy.start < t0 || sy.start > tEnd) continue;
      g.fillRect(x(sy.start), HEIGHT - 5, 1, 5);
    }
    // playhead
    const px = x(t);
    g.fillStyle = "rgba(255,255,255,0.85)";
    g.fillRect(px - 0.5, 0, 1.5, HEIGHT);
    if (current) {
      g.font = "600 12px system-ui, sans-serif";
      g.fillStyle = "#ffffff";
      g.fillText(noteName(current.midi), px + 6, Math.max(12, y(current.midi) - 3));
    }
  });

  return (
    <div ref={box} className="pianoroll">
      <canvas ref={canvas} style={{ width: "100%", height: HEIGHT }} role="img"
        aria-label="Detected melody: note bars, the raw pitch contour and gaps where words have no note"
        onClick={(e) => {
          const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
          const at = t0 + (e.clientX - r.left) / pxPerS;
          const m = lo - 1 + (HEIGHT - (e.clientY - r.top)) / rowH;
          const hit = melody.notes.find((n) => n.start - 0.05 <= at && at <= n.end + 0.05 && Math.abs(n.midi + 0.5 - m) < 2);
          onSeek(hit ? hit.start : at);
        }} />
    </div>
  );
}
