// The pitch guide under the play-along: key, gaps, the piano roll and the MIDI download.
import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, api } from "../api";
import { useServerEvent } from "../events";
import { missingSpans, total } from "../lib/melody";
import type { StemPlayer } from "../lib/player";
import { fmtTime } from "../lib/time";
import type { ContourDoc, LyricsDoc, MelodyDoc } from "../types";
import { PianoRoll } from "./PianoRoll";

export function useMelody(jobId: string) {
  const [melody, setMelody] = useState<MelodyDoc | null>(null);
  const [contour, setContour] = useState<ContourDoc | null>(null);
  const [missing, setMissing] = useState(false);
  const load = useCallback(() => {
    api.melody(jobId).then(
      (m) => {
        setMelody(m);
        setMissing(false);
      },
      (e) => {
        if (e instanceof ApiError && e.status === 404) {
          setMelody(null);
          setMissing(true);
        }
      },
    );
    api.melodyContour(jobId).then(setContour, () => setContour(null));
  }, [jobId]);
  useEffect(load, [load]);
  useServerEvent((e) => {
    if (e.type === "files" && e.data.job_id === jobId && e.data.paths.some((p) => p === "melody.json" || p.startsWith("cache/"))) load();
  });
  return { melody, contour, missing };
}

export function MelodyPanel({ jobId, player, melody, contour, missing, lyrics, running, exact, setExact }: {
  jobId: string;
  player: StemPlayer;
  melody: MelodyDoc | null;
  contour: ContourDoc | null;
  missing: boolean;
  lyrics: LyricsDoc;
  running: boolean;
  exact: boolean;
  setExact: (v: boolean) => void;
}) {
  const [queued, setQueued] = useState(false);
  const gaps = useMemo(() => missingSpans(lyrics, melody?.notes ?? []), [lyrics, melody]);
  const sung = useMemo(() => total(missingSpans(lyrics, [], 0)), [lyrics]);

  if (!melody) {
    return (
      <div className="melodybar">
        <span>{missing ? "No melody yet: the pitch guide needs the Melody step." : "Loading the melody…"}</span>
        {missing && (
          <button className="small" disabled={running || queued}
            onClick={() => api.run(jobId, { only: ["melody"] }).then(() => setQueued(true))}>
            {queued ? "Melody step queued" : "Run the melody step"}
          </button>
        )}
      </div>
    );
  }

  const gapS = total(gaps);
  const nextGap = () => {
    const t = player.time + 0.05;
    const g = gaps.find((s) => s.start > t) ?? gaps[0];
    if (g) player.seek(Math.max(0, g.start - 1));
  };
  const k = melody.key;
  return (
    <div className="melody">
      <div className="melodybar">
        <span title={k ? `Krumhansl-Schmuckler correlation ${k.confidence.toFixed(2)}, margin ${k.margin.toFixed(2)} over ${k.runner_up}` : ""}>
          {k ? <>Key <b>{k.tonic} {k.mode}</b> (confidence {k.confidence.toFixed(2)}{k.runner_up ? `, next best ${k.runner_up}` : ""})</> : "Key unknown"}
        </span>
        <span>{melody.notes.length} notes</span>
        <span className={gapS > 0 ? "gapinfo" : ""} title="Time where words are sung but no note was detected">
          {gapS > 0 ? `${fmtTime(gapS, 0)} of singing without notes (${Math.round((100 * gapS) / Math.max(sung, 1))}%)` : "Every sung word has a note"}
        </span>
        <span className="spacer" />
        {gaps.length > 0 && <button className="small" onClick={nextGap}>Next gap</button>}
        <label className="row" style={{ gap: "0.3rem" }}
          title="Play each note at the exact pitch the singer hit, instead of the rounded note">
          <input type="checkbox" checked={exact} onChange={(e) => setExact(e.target.checked)} /> Exact pitch
        </label>
        <a className="button small" href={api.midiUrl(jobId)} download>MIDI</a>
      </div>
      <PianoRoll player={player} melody={melody} contour={contour} missing={gaps} onSeek={(t) => player.seek(t)} />
      {contour?.stale && <p className="faint">The pitch contour is older than the vocal stem: run the Melody step again.</p>}
    </div>
  );
}
