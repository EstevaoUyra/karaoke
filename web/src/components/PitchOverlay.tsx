// The pitch guide as it sits at the top of the video: upcoming notes scroll from the right to a
// playhead, sung notes fill, and the voice's own pitch runs over them as a thin line. A sketch of
// karaoke/render/roll.py for the live stage (no words under the notes, no note names).
import { useEffect, useMemo, useRef, useState } from "react";
import type { StemPlayer } from "../lib/player";
import { hzToMidi } from "../lib/synth";
import { useClock } from "../lib/usePlayer";
import type { ContourDoc, MelodyDoc, RenderSettingsDoc } from "../types";
import { pitchRange } from "./PianoRoll";

const CURVE = "#FFC94D";
const UNSURE_CONF = 0.4; // below this the tracker is right far less often: drawn faint, as in the video

export function PitchOverlay({ player, melody, contour, render }: {
  player: StemPlayer;
  melody: MelodyDoc;
  contour: ContourDoc | null;
  render: RenderSettingsDoc;
}) {
  const t = useClock(player);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<[number, number]>([600, 120]);
  const [lo, hi] = useMemo(() => pitchRange(melody), [melody]);
  const curve = useMemo(() => {
    const out: { t: number; m: number }[] = [];
    contour?.t.forEach((tt, i) => {
      const hz = contour.hz[i];
      if (hz) out.push({ t: tt, m: hzToMidi(hz) - melody.tuning });
    });
    return out;
  }, [contour, melody.tuning]);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const measure = () => setSize([Math.max(100, el.clientWidth), Math.max(40, el.clientHeight)]);
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = canvas.current;
    const g = c?.getContext("2d");
    if (!c || !g) return;
    const [w, h] = size;
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const pad = h * 0.08;
    const span = Math.max(0.5, render.pitch_trail_s + render.pitch_lookahead_s);
    const t0 = t - render.pitch_trail_s;
    const x = (s: number) => pad + ((s - t0) / span) * (w - 2 * pad);
    const rowH = (h - 2 * pad) / (hi - lo + 1);
    const y = (m: number) => h - pad - (Math.max(lo, Math.min(hi, m)) - lo + 1) * rowH;
    // the tonic's rows, faintly
    if (melody.key) {
      g.fillStyle = "rgba(255,255,255,0.14)";
      for (let m = Math.ceil(lo); m <= hi; m++) {
        if ((((m - melody.key.tonic_pc) % 12) + 12) % 12 === 0) g.fillRect(pad, y(m) + rowH - 1, w - 2 * pad, 1);
      }
    }
    const px = x(t);
    const bar = Math.max(2, Math.min(rowH * 0.9, h * 0.07));
    for (const n of melody.notes) {
      if (n.end < t0 || n.start > t0 + span) continue;
      const x0 = Math.max(pad, x(n.start));
      const x1 = Math.min(w - pad, x(n.end));
      if (x1 <= x0) continue;
      const top = y(n.midi) + (rowH - bar) / 2;
      g.globalAlpha = n.conf < UNSURE_CONF ? 0.35 : 0.95;
      g.fillStyle = render.color_unsung;
      g.beginPath();
      g.roundRect(x0, top, x1 - x0, bar, Math.min(bar / 2, 3));
      g.fill();
      if (px > x0) { // the part already sung
        g.fillStyle = render.color_sung;
        g.beginPath();
        g.roundRect(x0, top, Math.min(px, x1) - x0, bar, Math.min(bar / 2, 3));
        g.fill();
      }
    }
    g.globalAlpha = 0.9;
    g.fillStyle = CURVE;
    for (const p of curve) {
      if (p.t < t0 || p.t > t0 + span) continue;
      g.fillRect(x(p.t) - 0.75, y(p.m) + rowH / 2 - 0.75, 1.5, 1.5);
    }
    g.globalAlpha = 1;
    g.fillStyle = "rgba(255,255,255,0.9)";
    g.fillRect(px - 0.75, pad / 2, 1.5, h - pad);
  });

  return <canvas ref={canvas} className="stage-pitch" role="img" aria-label="Pitch guide: the melody's notes and the voice's pitch" />;
}
