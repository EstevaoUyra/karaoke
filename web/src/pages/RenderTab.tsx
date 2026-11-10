import { useEffect, useState } from "react";
import { api, fileUrl } from "../api";
import type { JobDetail, RenderSettingsDoc } from "../types";
import { fmtDuration } from "../lib/time";

const FONTS = ["Arial Rounded MT Bold", "Arial Black", "Helvetica Neue", "Avenir Next", "Futura", "Verdana", "Georgia"];

export function RenderTab({ detail }: { detail: JobDetail }) {
  const id = detail.job.id;
  const [rs, setRs] = useState<RenderSettingsDoc>(detail.job.settings.render);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const st = detail.job.stages.render;
  const queue = detail.summary.queue;
  const rendering = st.status === "running";
  useEffect(() => {
    if (!rendering) setRs(detail.job.settings.render);
  }, [detail.job.settings.render]);

  const set = <K extends keyof RenderSettingsDoc>(k: K, v: RenderSettingsDoc[K]) => setRs({ ...rs, [k]: v });

  async function start() {
    setBusy(true);
    setError(null);
    try {
      await api.render(id, rs);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const want = rs.preset === "preview" ? "preview_video" : "final_video";
  const other = rs.preset === "preview" ? "final_video" : "preview_video";
  const shown = detail.files[want] ? want : detail.files[other] ? other : null;
  const path = shown === "preview_video" ? "output/preview.mp4" : "output/karaoke.mp4";
  const info = shown ? detail.files[shown]! : null;
  const missing = !detail.files.instrumental || !detail.files.lyrics_final;

  return (
    <div className="two">
      <form className="panel form" onSubmit={(e) => {
        e.preventDefault();
        void start();
      }}>
        <div className="presets" role="radiogroup" aria-label="Quality">
          {(["preview", "final"] as const).map((p) => (
            <button type="button" key={p} role="radio" aria-checked={rs.preset === p}
              className={`preset ${rs.preset === p ? "on" : ""}`} onClick={() => set("preset", p)}>
              {p === "preview" ? "Preview" : "Final"}
              <small>{p === "preview" ? "720p, fast, to check the timing" : "1080p, best quality, slower"}</small>
            </button>
          ))}
        </div>
        <div className="field">
          <label htmlFor="font">Font</label>
          <input id="font" list="fonts" value={rs.font} onChange={(e) => set("font", e.target.value)} />
          <datalist id="fonts">{FONTS.map((f) => <option key={f} value={f} />)}</datalist>
        </div>
        <div className="pair">
          <div className="field">
            <label htmlFor="fsize">Size at 1080p</label>
            <input id="fsize" type="number" min={24} max={160} value={rs.font_size} onChange={(e) => set("font_size", Number(e.target.value))} />
          </div>
          <div className="field">
            <label htmlFor="leadin">Show lines early (s)</label>
            <input id="leadin" type="number" min={0} max={5} step={0.1} value={rs.lead_in_s} onChange={(e) => set("lead_in_s", Number(e.target.value))} />
          </div>
        </div>
        <div className="row" style={{ gap: "1rem" }}>
          {([["color_sung", "Sung"], ["color_unsung", "Not sung yet"], ["color_outline", "Outline"]] as const).map(([k, label]) => (
            <label key={k} className="field">
              {label}
              <input type="color" value={rs[k]} onChange={(e) => set(k, e.target.value.toUpperCase())} />
            </label>
          ))}
        </div>
        <div className="field">
          <label className="row">
            <input type="checkbox" checked={rs.guide_vocals_db !== null}
              onChange={(e) => set("guide_vocals_db", e.target.checked ? -18 : null)} />
            Guide vocals, mixed in quietly
          </label>
          {rs.guide_vocals_db !== null && (
            <div className="row">
              <input type="range" min={-36} max={-3} step={1} value={rs.guide_vocals_db} style={{ flex: 1 }}
                aria-label="Guide vocals level" onChange={(e) => set("guide_vocals_db", Number(e.target.value))} />
              <span className="faint">{rs.guide_vocals_db} dB</span>
            </div>
          )}
        </div>
        <div className="pair">
          <div className="field">
            <label htmlFor="cgap">Countdown after a gap of (s)</label>
            <input id="cgap" type="number" min={1} max={30} step={0.5} value={rs.countdown_gap_s}
              onChange={(e) => set("countdown_gap_s", Number(e.target.value))} />
          </div>
          <label className="row" style={{ alignSelf: "end" }}>
            <input type="checkbox" checked={rs.show_title_card} onChange={(e) => set("show_title_card", e.target.checked)} />
            Title card in the intro
          </label>
        </div>
        <label className="row" title={detail.files.melody ? "" : "Run the Melody step first"}>
          <input type="checkbox" checked={rs.show_pitch} disabled={!detail.files.melody && !rs.show_pitch}
            onChange={(e) => set("show_pitch", e.target.checked)} />
          Pitch guide: the melody as bars on a staff at the top
        </label>
        <button className="primary" disabled={busy || rendering || queue !== null || missing}>
          {rendering ? `Rendering ${Math.round(st.progress * 100)}%` : queue === "queued" ? "Waiting in the queue" :
            `Render ${rs.preset === "preview" ? "preview" : "final video"}`}
        </button>
        {missing && <p className="faint">Rendering needs the instrumental and the synced lyrics: run the steps first.</p>}
        {rendering && <div className="bar"><i style={{ width: `${st.progress * 100}%` }} /></div>}
        {rendering && st.message && <p className="faint">{st.message}</p>}
        {st.status === "failed" && st.error && <p className="notice err">{st.error}</p>}
        {error && <p className="notice err">{error}</p>}
      </form>

      <div className="videobox">
        {info ? (
          <>
            <video key={`${path}-${info.mtime}`} controls preload="metadata" src={fileUrl(id, path, info.mtime)} />
            <div className="row">
              <span className="muted">
                {shown === "preview_video" ? "Preview" : "Final video"}, {(info.size / 1e6).toFixed(1)} MB,
                made {new Date(info.mtime * 1000).toLocaleString()}
                {st.duration_s && st.status === "done" ? ` in ${fmtDuration(st.duration_s)}` : ""}
              </span>
              <span className="spacer" />
              <a className="button" href={fileUrl(id, path, info.mtime, true)} download>Download</a>
            </div>
            {shown !== want && <p className="faint">There is no {rs.preset} render yet; this is the other one.</p>}
          </>
        ) : (
          <div className="emptybox">
            <div>
              <p>No video yet.</p>
              <p className="faint">A preview takes a minute or two.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
