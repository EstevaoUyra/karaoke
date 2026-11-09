// The whole song at a glance: one block per line, coloured by how sure we are of it. Click or drag to seek.
import { useRef } from "react";
import type { StemPlayer } from "../lib/player";
import { useClock } from "../lib/usePlayer";
import { isProtected, lineNeedsReview } from "../lib/lyricsOps";
import type { LineDoc } from "../types";

export function Timeline({ lines, player, duration, errorLines, onSeek }: {
  lines: LineDoc[];
  player: StemPlayer;
  duration: number;
  errorLines: Set<string>;
  onSeek: (t: number) => void;
}) {
  const t = useClock(player);
  const box = useRef<HTMLDivElement>(null);
  const pct = (x: number) => `${(100 * x) / Math.max(duration, 1)}%`;
  const at = (clientX: number) => {
    const r = box.current!.getBoundingClientRect();
    return Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * duration;
  };
  return (
    <div
      ref={box} className="timeline" role="slider" aria-label="Song position" tabIndex={-1}
      aria-valuemin={0} aria-valuemax={Math.round(duration)} aria-valuenow={Math.round(t)}
      onPointerDown={(e) => {
        (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
        onSeek(at(e.clientX));
      }}
      onPointerMove={(e) => {
        if (e.buttons === 1) onSeek(at(e.clientX));
      }}
    >
      {lines.map((l) => {
        const cls = errorLines.has(l.id) ? "err" : l.words.every(isProtected) ? "mine" : lineNeedsReview(l) ? "unsure" : "";
        return <i key={l.id} className={`seg ${cls}`} style={{ left: pct(l.start), width: pct(Math.max(l.end - l.start, 0.15)) }} />;
      })}
      <i className="head" style={{ left: pct(t) }} />
    </div>
  );
}

export function Clock({ player }: { player: StemPlayer }) {
  const t = useClock(player);
  const fmt = (x: number) => {
    const m = Math.floor(x / 60);
    return `${m}:${(x - m * 60).toFixed(2).padStart(5, "0")}`;
  };
  return <span className="clock">{fmt(t)} / {fmt(player.duration)}</span>;
}
