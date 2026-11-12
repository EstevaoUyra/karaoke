// The background behind a live stage at song time `t`: the scene's picture or colours, moving,
// softened and darkened as in the video, and cross-fading into the next scene.
import type { CSSProperties } from "react";
import { sceneBrightness, sceneFill, scenesAt, sceneTransform } from "../lib/scene";
import type { BackgroundDoc } from "../types";

export function StageBackground({ jobId, bg, t }: { jobId: string; bg: BackgroundDoc | null; t: number }) {
  if (!bg) return null;
  return (
    <>
      {scenesAt(bg, t).map(({ scene, opacity }) => (
        <div key={scene.id} className="stage-bg" style={{
          ...sceneFill(jobId, scene), transform: sceneTransform(scene, bg.motion_strength, t), opacity,
          "--blur": scene.blur, "--bright": sceneBrightness(scene),
        } as CSSProperties} />
      ))}
    </>
  );
}
