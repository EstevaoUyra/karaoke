import { describe, expect, it } from "vitest";
import { reach, scenesAt, sceneTransform } from "./scene";
import type { BackgroundDoc, SceneDoc } from "../types";

const scene = (id: string, start: number, end: number, extra: Partial<SceneDoc> = {}): SceneDoc => ({
  id, start, end, source: { type: "image", path: `assets/bg/${id}.jpg` }, motion: "none",
  transition: { type: "cut", duration: 0 }, dim: 0, blur: 0, locked: false, reason: "", ...extra,
});
const bg = (scenes: SceneDoc[]): BackgroundDoc => ({
  version: 1, resolution: [1920, 1080], fps: 30, style: "ai", motion_strength: "normal", scenes,
});

describe("scene", () => {
  it("moves a picture at one speed, whatever the scene's length", () => {
    expect(reach(scene("a", 0, 15, { motion: "kenburns-in" }), "normal")).toBeCloseTo(0.12);
    expect(reach(scene("a", 0, 7.5, { motion: "kenburns-in" }), "normal")).toBeCloseTo(0.06);
    expect(reach(scene("a", 0, 120, { motion: "kenburns-in" }), "strong")).toBe(0.25);
    expect(reach(scene("a", 0, 120, { motion: "pan-left" }), "strong")).toBe(0.2);
  });

  it("puts the picture where the video has it at a given time", () => {
    const zoom = scene("a", 10, 25, { motion: "kenburns-in" });
    expect(sceneTransform(zoom, "normal", 10)).toBe("scale(1.0000)");
    expect(sceneTransform(zoom, "normal", 25)).toBe("scale(1.1200)");
    expect(sceneTransform(scene("a", 10, 25, { motion: "kenburns-out" }), "normal", 10)).toBe("scale(1.1200)");
    const pan = scene("a", 0, 10, { motion: "pan-left" });
    expect(sceneTransform(pan, "normal", 0)).toBe("scale(1.0870) translateX(-4.000%)");
    expect(sceneTransform(pan, "normal", 10)).toBe("scale(1.0870) translateX(4.000%)");
    expect(sceneTransform(scene("a", 0, 10), "normal", 5)).toBeUndefined();
  });

  it("shows two scenes while one fades into the next, centred on the boundary", () => {
    const b = bg([scene("a", 0, 10), scene("b", 10, 20, { transition: { type: "fade", duration: 2 } }), scene("c", 20, 30)]);
    expect(scenesAt(b, 5).map((x) => x.scene.id)).toEqual(["a"]);
    expect(scenesAt(b, 9.5)).toEqual([{ scene: b.scenes[0], opacity: 1 }, { scene: b.scenes[1], opacity: 0.25 }]);
    expect(scenesAt(b, 10.5)).toEqual([{ scene: b.scenes[0], opacity: 1 }, { scene: b.scenes[1], opacity: 0.75 }]);
    expect(scenesAt(b, 12).map((x) => x.scene.id)).toEqual(["b"]);
    expect(scenesAt(b, 19.9).map((x) => x.scene.id)).toEqual(["b"]); // a cut: no overlap
    expect(scenesAt(b, 99).map((x) => x.scene.id)).toEqual(["c"]);
  });
});
