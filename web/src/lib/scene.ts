// What a background scene looks like at a moment of the song, as CSS: shared by the scene
// previews of the Background tab and the live stage of the Lyrics and Video tabs.
import type { CSSProperties } from "react";
import { fileUrl } from "../api";
import type { BackgroundDoc, SceneDoc } from "../types";

export type Strength = BackgroundDoc["motion_strength"];
// The same numbers as karaoke/render/compositor.py: movement per second, and where a long scene stops.
export const MOTION_SPEED: Record<Strength, number> = { subtle: 0.004, normal: 0.008, strong: 0.014 };
export const MAX_ZOOM = 0.25;
export const MAX_TRAVEL = 0.2;

/** How far a moving picture gets over its scene: the zoom gained, or the share of its width panned. */
export function reach(s: SceneDoc, strength: Strength): number {
  const r = MOTION_SPEED[strength] * Math.max(0, s.end - s.start);
  return Math.min(s.motion.startsWith("kenburns") ? MAX_ZOOM : MAX_TRAVEL, r);
}

/** The scene's source as CSS. Generated motion cannot be drawn here: its colours stand in for it. */
export function sceneFill(jobId: string, s: SceneDoc): CSSProperties {
  const src = s.source;
  if (src.type === "color") return { background: src.value ?? "#000" };
  if (src.type === "gradient") {
    const [a, b] = (src.value ?? "#000,#000").split(",");
    return { background: `linear-gradient(180deg, ${a}, ${b ?? a})` };
  }
  if (src.type === "generated") {
    const cols = (src.value ?? "").split(":")[1]?.split(",").filter(Boolean) ?? [];
    return { background: `linear-gradient(120deg, ${(cols.length > 1 ? cols : ["#3a1c71", "#d76d77", "#1f6f8b"]).join(", ")})` };
  }
  if (src.type === "image" && src.path) return { backgroundImage: `url("${fileUrl(jobId, src.path)}")` };
  return { background: "#222" };
}

/** How bright the scene is drawn: the video multiplies the colours by 1 - dim. */
export function sceneBrightness(s: SceneDoc): number {
  return (s.source.type === "generated" ? 0.55 : 1) * (1 - s.dim);
}

/** Where a moving picture is at song time `t`: the same movement as the video. */
export function sceneTransform(s: SceneDoc, strength: Strength, t: number): string | undefined {
  if (s.motion === "none" || s.source.type !== "image") return undefined;
  const p = Math.min(1, Math.max(0, (t - s.start) / Math.max(0.001, s.end - s.start)));
  const r = reach(s, strength);
  if (s.motion === "kenburns-in") return `scale(${(1 + r * p).toFixed(4)})`;
  if (s.motion === "kenburns-out") return `scale(${(1 + r * (1 - p)).toFixed(4)})`;
  // pan-left: the view slides from the picture's right edge to its left, so the picture moves right
  const x = (s.motion === "pan-left" ? p - 0.5 : 0.5 - p) * r * 100;
  return `scale(${(1 / (1 - r)).toFixed(4)}) translateX(${x.toFixed(3)}%)`;
}

/** The scenes on screen at `t`, bottom first, with their opacity: two while one fades into the next. */
export function scenesAt(bg: BackgroundDoc, t: number): { scene: SceneDoc; opacity: number }[] {
  const scenes = bg.scenes;
  if (!scenes.length) return [];
  let i = scenes.findIndex((s) => s.start <= t && t < s.end);
  if (i < 0) i = t < scenes[0].start ? 0 : scenes.length - 1;
  const fade = (k: number) => (k > 0 && k < scenes.length && scenes[k].transition.type === "fade" ? scenes[k].transition.duration : 0);
  // a fade is centred on the boundary: it starts half its length before the scene does
  const into = fade(i);
  if (into > 0 && t < scenes[i].start + into / 2) {
    return [{ scene: scenes[i - 1], opacity: 1 }, { scene: scenes[i], opacity: (t - scenes[i].start + into / 2) / into }];
  }
  const out = fade(i + 1);
  if (out > 0 && t >= scenes[i + 1].start - out / 2) {
    return [{ scene: scenes[i], opacity: 1 }, { scene: scenes[i + 1], opacity: (t - scenes[i + 1].start + out / 2) / out }];
  }
  return [{ scene: scenes[i], opacity: 1 }];
}
