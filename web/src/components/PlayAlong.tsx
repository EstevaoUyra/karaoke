// The play-along stage: the lyric overlay of the video, drawn with HTML/CSS from the lyrics being
// edited, over the instrumental. No render needed to check an edit.
import type { CSSProperties, ReactNode } from "react";
import { stageView, wordFill } from "../lib/playalong";
import type { StemPlayer } from "../lib/player";
import { useClock } from "../lib/usePlayer";
import type { BackgroundDoc, LineDoc, RenderSettingsDoc } from "../types";
import { StageBackground } from "./StageBackground";

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
  /** "video": the video's proportions, with the lyrics where the video has them and its font size. */
  variant?: "strip" | "video";
  /** Drawn over the background and under the lyrics (the pitch guide). */
  children?: ReactNode;
}) {
  const { lines, player, render } = props;
  const t = useClock(player);
  const view = stageView(lines, t, render.lead_in_s, render.countdown_gap_s);
  const vars = {
    "--fill": render.color_sung,
    "--unfill": render.color_unsung,
    "--outline": render.color_outline,
    "--fs": render.font_size,
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
    <div className={`stagebox ${props.variant === "video" ? "video" : ""}`} style={vars}>
      <StageBackground jobId={props.jobId} bg={props.background} t={t} />
      {props.children}
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
