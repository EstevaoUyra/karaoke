// The play-along stage: the lyric overlay of the video, drawn with HTML/CSS from the lyrics being
// edited, over the instrumental. No render needed to check an edit.
import type { CSSProperties } from "react";
import { fileUrl } from "../api";
import { stageView, wordFill } from "../lib/playalong";
import type { StemPlayer } from "../lib/player";
import { useClock } from "../lib/usePlayer";
import type { BackgroundDoc, LineDoc, RenderSettingsDoc } from "../types";

function sceneStyle(jobId: string, bg: BackgroundDoc | null, t: number): CSSProperties {
  const scene = bg?.scenes.find((s) => s.start <= t && t < s.end) ?? bg?.scenes[0];
  if (!scene) return {};
  const src = scene.source;
  const style: CSSProperties & Record<string, string | number> = { "--dim": scene.dim };
  if (src.type === "color" && src.value) style.background = src.value;
  else if (src.type === "gradient" && src.value) {
    const [a, b] = src.value.split(",");
    style.background = `linear-gradient(180deg, ${a}, ${b ?? a})`;
  } else if (src.type === "image" && src.path) style.backgroundImage = `url("${fileUrl(jobId, src.path)}")`;
  return style;
}

export function PlayAlong(props: {
  jobId: string;
  lines: LineDoc[];
  player: StemPlayer;
  render: RenderSettingsDoc;
  background: BackgroundDoc | null;
  tapLabel: string | null;
  tapDown: boolean;
  onWordClick: (li: number, wi: number) => void;
  idle: string | null;
}) {
  const { lines, player, render } = props;
  const t = useClock(player);
  const view = stageView(lines, t, render.lead_in_s, render.countdown_gap_s);
  const vars = {
    "--fill": render.color_sung,
    "--unfill": render.color_unsung,
    "--outline": render.color_outline,
    fontFamily: `"${render.font}", var(--round)`,
  } as CSSProperties;

  const lineEl = (li: number, kind: "current" | "next") => (
    <div key={`${kind}-${lines[li].id}`} className={`stage-line ${kind}`}>
      {lines[li].words.map((w, wi) => (
        <span key={wi} className="stage-word" style={{ "--p": `${(wordFill(w, t) * 100).toFixed(1)}%` } as CSSProperties}
          onClick={() => props.onWordClick(li, wi)}>
          {w.text}
        </span>
      ))}
    </div>
  );

  return (
    <div className="stagebox" style={{ ...sceneStyle(props.jobId, props.background, t), ...vars }}>
      {props.tapLabel && <div className={`stage-tap ${props.tapDown ? "down" : ""}`}>{props.tapLabel}</div>}
      {view.countdown !== null && (
        <div className="countdown" aria-label={`${view.countdown}`}>
          {Array.from({ length: view.countdown }, (_, i) => <i key={i} />)}
        </div>
      )}
      {props.idle ? (
        <div className="stage-idle">{props.idle}</div>
      ) : (
        <div className="stage-lines">
          {view.current !== null && lineEl(view.current, "current")}
          {view.next !== null && lineEl(view.next, "next")}
        </div>
      )}
    </div>
  );
}
