// The Video tab's live stage: the background, the lyrics and the pitch guide as they are now, at
// the video's proportions and over the instrumental. Nothing is rendered: it follows the settings
// as they change, so a choice can be judged before the minutes of a render.
import { useEffect, useState } from "react";
import { api } from "../api";
import { useServerEvent } from "../events";
import { fmtTime } from "../lib/time";
import { useClock, usePlayer } from "../lib/usePlayer";
import type { StemPlayer } from "../lib/player";
import type { BackgroundDoc, JobDetail, LyricsDoc, RenderSettingsDoc } from "../types";
import { useMelody } from "./MelodyPanel";
import { PitchOverlay } from "./PitchOverlay";
import { PlayAlong } from "./PlayAlong";

function Transport({ player, disabled }: { player: StemPlayer; disabled: boolean }) {
  const t = useClock(player);
  const total = player.duration || 0;
  return (
    <div className="row">
      <button type="button" className="primary" style={{ minWidth: "5.5rem" }} disabled={disabled}
        onClick={() => player.toggle()}>{player.playing ? "Pause" : "Play"}</button>
      <input type="range" min={0} max={total} step={0.1} value={Math.min(t, total)} disabled={disabled} style={{ flex: 1 }}
        aria-label="Position in the song" onChange={(e) => player.seek(Number(e.target.value))} />
      <span className="faint" style={{ fontVariantNumeric: "tabular-nums" }}>{fmtTime(t, 0)} / {fmtTime(total, 0)}</span>
    </div>
  );
}

export function VideoPreview({ detail, render }: { detail: JobDetail; render: RenderSettingsDoc }) {
  const id = detail.job.id;
  const { player, loading, error } = usePlayer(id, detail.files.instrumental?.mtime ?? null);
  const [lyrics, setLyrics] = useState<LyricsDoc | null>(null);
  const [bg, setBg] = useState<BackgroundDoc | null>(null);
  const mel = useMelody(id);

  const load = () => {
    api.lyrics(id).then(setLyrics, () => setLyrics(null));
    api.background(id).then(setBg, () => setBg(null));
  };
  useEffect(load, [id]);
  useServerEvent((e) => {
    if (e.type === "files" && e.data.job_id === id
      && e.data.paths.some((p) => p === "background.json" || p === "lyrics.final.json")) load();
  });
  useEffect(() => () => player.pause(), [player]);

  const idle = loading ? "Loading the instrumental…" : error ?? (lyrics ? null : "No synced lyrics yet.");
  const pitch = render.show_pitch && mel.melody && mel.melody.notes.length > 0;
  return (
    <div className="videobox">
      <PlayAlong jobId={id} lines={lyrics?.lines ?? []} player={player} render={render} background={bg} variant="video"
        tapLabel={null} tapDown={false} onWordClick={() => {}} idle={idle}>
        {pitch && <PitchOverlay player={player} melody={mel.melody!} contour={mel.contour} render={render} />}
      </PlayAlong>
      <Transport player={player} disabled={loading || !!error} />
      <p className="faint">
        Live preview: the background, lyrics and pitch guide as they are now, changing as you change the settings.
        It is a sketch of the video (moving colours show as a still blend, and the lyric lines sit a little
        differently); the render is the real thing.
        {render.show_pitch && !pitch && " There is no melody yet for the pitch guide: run the Melody step."}
      </p>
    </div>
  );
}
