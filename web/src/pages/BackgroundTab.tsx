import { useEffect, useState, type CSSProperties } from "react";
import { ApiError, api, fileUrl } from "../api";
import { useServerEvent } from "../events";
import { navigate, jobHref } from "../router";
import { fmtTime, parseTime } from "../lib/time";
import type { BackgroundDoc, Issue, JobDetail, SceneDoc } from "../types";

const MOTIONS: SceneDoc["motion"][] = ["none", "kenburns-in", "kenburns-out", "pan-left", "pan-right"];
const MOTION_LABEL: Record<SceneDoc["motion"], string> = {
  none: "Still", "kenburns-in": "Slow zoom in", "kenburns-out": "Slow zoom out", "pan-left": "Pan left", "pan-right": "Pan right",
};

function preview(id: string, s: SceneDoc): CSSProperties {
  const src = s.source;
  const st: Record<string, string | number> = { "--dim": s.dim };
  if (src.type === "color") st.background = src.value ?? "#000";
  else if (src.type === "gradient") {
    const [a, b] = (src.value ?? "#000,#000").split(",");
    st.background = `linear-gradient(180deg, ${a}, ${b ?? a})`;
  } else if (src.type === "image" && src.path) st.backgroundImage = `url("${fileUrl(id, src.path)}")`;
  else st.background = "#222";
  return st as CSSProperties;
}

/** Split scene k at its middle (the new scene copies it). */
function splitScene(bg: BackgroundDoc, k: number): BackgroundDoc {
  const s = bg.scenes[k];
  const mid = Math.round(((s.start + s.end) / 2) * 1000) / 1000;
  const ids = new Set(bg.scenes.map((x) => x.id));
  let n = bg.scenes.length + 1;
  while (ids.has(`s${n}`)) n++;
  const scenes = bg.scenes.slice();
  scenes.splice(k, 1, { ...s, end: mid }, { ...s, id: `s${n}`, start: mid, locked: false, reason: "" });
  return { ...bg, scenes };
}

/** Remove scene k; its time goes to the previous scene (or the next, for the first). */
function removeScene(bg: BackgroundDoc, k: number): BackgroundDoc {
  if (bg.scenes.length < 2) return bg;
  const scenes = bg.scenes.slice();
  const [gone] = scenes.splice(k, 1);
  if (k > 0) scenes[k - 1] = { ...scenes[k - 1], end: gone.end };
  else scenes[0] = { ...scenes[0], start: gone.start };
  return { ...bg, scenes };
}

export function BackgroundTab({ detail }: { detail: JobDetail }) {
  const id = detail.job.id;
  const [bg, setBg] = useState<BackgroundDoc | null>(null);
  const [savedJson, setSavedJson] = useState("");
  const [issues, setIssues] = useState<Issue[]>([]);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const [missing, setMissing] = useState(false);
  const [copied, setCopied] = useState(false);

  const load = () =>
    api.background(id).then((b) => {
      setBg(b);
      setSavedJson(JSON.stringify(b));
      setMissing(false);
    }, (e) => e instanceof ApiError && e.status === 404 && setMissing(true));
  useEffect(() => {
    void load();
  }, [id]);
  const dirty = bg !== null && JSON.stringify(bg) !== savedJson;
  useServerEvent((e) => {
    if (e.type === "files" && e.data.job_id === id && e.data.paths.includes("background.json") && !dirty) void load();
  });

  if (missing) return <p className="notice warn">No background yet: run the Background step.</p>;
  if (!bg) return <p className="muted">Loading…</p>;

  const setScene = (k: number, patch: Partial<SceneDoc>) =>
    setBg({ ...bg, scenes: bg.scenes.map((s, i) => (i === k ? { ...s, ...patch } : s)) });
  const setBoundary = (k: number, t: number) => {
    // the boundary between scene k-1 and k
    const scenes = bg.scenes.slice();
    scenes[k - 1] = { ...scenes[k - 1], end: t };
    scenes[k] = { ...scenes[k], start: t };
    setBg({ ...bg, scenes });
  };

  async function save(next = bg!) {
    setMsg(null);
    try {
      const r = await api.saveBackground(id, next);
      setBg(r.background);
      setSavedJson(JSON.stringify(r.background));
      setIssues(r.issues);
      setMsg({ kind: "ok", text: "Saved. Render again to see it in the video." });
    } catch (e) {
      setIssues(e instanceof ApiError ? e.issues : []);
      setMsg({ kind: "err", text: (e as Error).message });
    }
  }

  async function upload(k: number, file: File) {
    setMsg(null);
    try {
      if (dirty) await save();
      const r = await api.uploadSceneImage(id, bg!.scenes[k].id, file);
      setBg(r.background);
      setSavedJson(JSON.stringify(r.background));
      setMsg({ kind: "ok", text: "Image saved and the scene locked, so the AI art direction keeps it." });
    } catch (e) {
      setMsg({ kind: "err", text: (e as Error).message });
    }
  }

  const cmd = `/karaoke-backgrounds ${id}`;
  return (
    <div style={{ display: "grid", gap: "1rem", maxWidth: 1000 }}>
      <div className="row">
        <button className="primary" disabled={!dirty} onClick={() => save()}>Save background</button>
        <button disabled={dirty} onClick={() => navigate(jobHref(id, "render"))}>Render with it</button>
        {dirty && <span className="faint"><i className="dirty-dot" /> Not saved</span>}
        <span className="spacer" />
        <span className="faint">{bg.scenes.length} scene{bg.scenes.length > 1 ? "s" : ""}, style {bg.style}</span>
      </div>
      {msg && <p className={`notice ${msg.kind === "err" ? "err" : ""}`}>{msg.text}</p>}
      {issues.length > 0 && (
        <ul className="issues" style={{ margin: 0, paddingLeft: "1.2rem" }}>
          {issues.map((i, k) => <li key={k} className={i.severity === "error" ? "error-text" : "faint"}>{i.where}: {i.message}</li>)}
        </ul>
      )}
      <div className="scenes">
        {bg.scenes.map((s, k) => (
          <div key={s.id} className="panel scene">
            <div style={{ display: "grid", gap: "0.5rem", alignContent: "start" }}>
              <div className="scene-preview" style={preview(id, s)} />
              <label className="button" style={{ cursor: "pointer" }}>
                Replace with an image
                <input type="file" accept="image/jpeg,image/png,image/webp" hidden
                  onChange={(e) => e.target.files?.[0] && upload(k, e.target.files[0])} />
              </label>
            </div>
            <div className="scene-fields">
              <div className="row">
                <h3>Scene {s.id}</h3>
                <span className="faint">{fmtTime(s.start, 1)} to {fmtTime(s.end, 1)}</span>
                <span className="spacer" />
                <label className="row" title="The AI art direction never changes a locked scene">
                  <input type="checkbox" checked={s.locked} onChange={(e) => setScene(k, { locked: e.target.checked })} /> Locked
                </label>
              </div>
              {s.reason && <p className="faint">{s.reason}</p>}
              <div className="pair">
                {k > 0 && (
                  <label className="field">
                    Starts at
                    <input defaultValue={s.start.toFixed(2)} key={`b${s.id}${s.start}`} onBlur={(e) => {
                      const t = parseTime(e.target.value);
                      if (t !== null && t > bg.scenes[k - 1].start && t < s.end) setBoundary(k, t);
                    }} />
                  </label>
                )}
                <label className="field">
                  Type
                  <select value={s.source.type} onChange={(e) => {
                    const type = e.target.value as SceneDoc["source"]["type"];
                    const value = type === "color" ? "#101018" : type === "gradient" ? "#0b0b1a,#2a1f4a" : null;
                    setScene(k, { source: { ...s.source, type, value } });
                  }}>
                    <option value="gradient">Gradient</option>
                    <option value="color">Colour</option>
                    <option value="image" disabled={!s.source.path}>Image</option>
                  </select>
                </label>
                {s.source.type === "color" && (
                  <label className="field">
                    Colour
                    <input type="color" value={s.source.value ?? "#101018"} onChange={(e) => setScene(k, { source: { ...s.source, value: e.target.value } })} />
                  </label>
                )}
                {s.source.type === "gradient" && (
                  <label className="field">
                    Top and bottom
                    <span className="row">
                      {[0, 1].map((n) => {
                        const parts = (s.source.value ?? "#000000,#000000").split(",");
                        return <input key={n} type="color" value={parts[n] ?? parts[0]} onChange={(e) => {
                          parts[n] = e.target.value;
                          setScene(k, { source: { ...s.source, value: parts.slice(0, 2).join(",") } });
                        }} />;
                      })}
                    </span>
                  </label>
                )}
                <label className="field">
                  Motion
                  <select value={s.motion} onChange={(e) => setScene(k, { motion: e.target.value as SceneDoc["motion"] })}>
                    {MOTIONS.map((m) => <option key={m} value={m}>{MOTION_LABEL[m]}</option>)}
                  </select>
                </label>
                <label className="field">
                  Darken {Math.round(s.dim * 100)}%
                  <input type="range" min={0} max={0.9} step={0.05} value={s.dim} onChange={(e) => setScene(k, { dim: Number(e.target.value) })} />
                </label>
                <label className="field">
                  Blur {s.blur}px
                  <input type="range" min={0} max={30} step={1} value={s.blur} onChange={(e) => setScene(k, { blur: Number(e.target.value) })} />
                </label>
                <label className="field">
                  Into this scene
                  <select value={s.transition.type} onChange={(e) => setScene(k, {
                    transition: { type: e.target.value as "cut" | "fade", duration: e.target.value === "fade" ? 1 : 0 },
                  })}>
                    <option value="cut">Cut</option>
                    <option value="fade">Fade</option>
                  </select>
                </label>
              </div>
              {s.source.credit && <p className="faint">{s.source.credit}</p>}
              <div className="row">
                <button className="small" onClick={() => setBg(splitScene(bg, k))}>Split in two</button>
                <button className="small danger" disabled={bg.scenes.length < 2} onClick={() => setBg(removeScene(bg, k))}>Remove scene</button>
              </div>
            </div>
          </div>
        ))}
      </div>
      <div className="panel aibox">
        <h3>AI art direction</h3>
        <p className="muted" style={{ fontSize: "0.9rem" }}>
          Coming in the next phase: Claude picks images for each section and writes the scenes, leaving locked
          scenes as they are. It will run from this button or from a Claude Code session in this repo:
        </p>
        <div className="cmd">
          <code>{cmd}</code>
          <button className="small" onClick={() => navigator.clipboard?.writeText(cmd).then(() => setCopied(true))}>
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
        <div><button disabled title="Phase 3">Ask AI to art-direct</button></div>
      </div>
    </div>
  );
}
