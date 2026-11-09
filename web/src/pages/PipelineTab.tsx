import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { useServerEvent } from "../events";
import { useThrottled } from "../hooks";
import { STAGES, type JobDetail, type StageName } from "../types";
import { fmtDuration } from "../lib/time";
import { STAGE_HELP, STAGE_LABEL, STATUS_LABEL } from "../components/labels";

function Log({ id, stage }: { id: string; stage: StageName }) {
  const [text, setText] = useState<string | null>(null);
  const load = () => api.log(id, stage, 300).then((r) => setText(r.text), () => setText("(could not read the log)"));
  const later = useThrottled(() => void load(), 700);
  useEffect(() => {
    void load();
  }, [id, stage]);
  useServerEvent((e) => {
    if (e.type === "files" && e.data.job_id === id && e.data.paths.includes(`logs/${stage}.log`)) later();
  });
  const pre = useRef<HTMLPreElement>(null);
  useEffect(() => {
    if (pre.current) pre.current.scrollTop = pre.current.scrollHeight; // newest lines first in view
  }, [text]);
  return <pre ref={pre} className="log">{text === null ? "Loading…" : text || "This step has no log yet."}</pre>;
}

export function PipelineTab({ detail, reload }: { detail: JobDetail; reload: () => Promise<void> }) {
  const { job, summary } = detail;
  const [open, setOpen] = useState<StageName | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const queued = summary.queue;

  async function act(fn: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await fn();
      await reload();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ maxWidth: 860, display: "grid", gap: "1rem" }}>
      <div className="row">
        <button className="primary" disabled={busy || queued !== null} onClick={() => act(() => api.run(job.id))}>
          Run the missing steps
        </button>
        {queued && (
          <button className="danger" disabled={busy} onClick={() => act(() => api.cancel(job.id))}>
            {queued === "running" ? "Stop" : "Remove from the queue"}
          </button>
        )}
        <span className="faint">
          {queued === "running" ? "Running now." : queued === "queued" ? "Waiting for another song to finish." :
            "Steps whose inputs did not change are skipped."}
        </span>
      </div>
      {error && <p className="notice err">{error}</p>}
      {detail.issues.some((i) => i.severity === "error") && (
        <div className="notice warn">
          {detail.issues.filter((i) => i.severity === "error").length} problem(s) in the job files, for example:{" "}
          {detail.issues[0].file} {detail.issues[0].where}: {detail.issues[0].message}
        </div>
      )}
      <ol className="stages" style={{ listStyle: "none", margin: 0, padding: 0 }}>
        {STAGES.map((name, i) => {
          const st = job.stages[name];
          const running = st.status === "running";
          return (
            <li key={name} className={`stage ${st.status}`}>
              <span className="stage-num">{i + 1}</span>
              <div>
                <div className="stage-name">{STAGE_LABEL[name]}</div>
                <div className="stage-status">
                  {STATUS_LABEL[st.status]}
                  {running && ` ${Math.round(st.progress * 100)}%`}
                  {st.duration_s != null && !running && st.status !== "skipped" && `, ${fmtDuration(st.duration_s)}`}
                  <span className="faint"> {STAGE_HELP[name]}</span>
                </div>
              </div>
              <div className="row" style={{ justifyContent: "flex-end" }}>
                <button className="quiet small" aria-expanded={open === name} onClick={() => setOpen(open === name ? null : name)}>
                  {open === name ? "Hide log" : "Log"}
                </button>
                <button className="small" disabled={busy || queued === "running"}
                  title="Run this step again, then every later step whose inputs changed"
                  onClick={() => act(() => api.run(job.id, { from_stage: name }))}>
                  Re-run from here
                </button>
              </div>
              {running && <div className="bar"><i style={{ width: `${st.progress * 100}%` }} /></div>}
              {st.message && st.message !== "started" && <div className="stage-msg">{st.message}</div>}
              {st.error && <div className="stage-err">{st.error}</div>}
              {open === name && <Log id={job.id} stage={name} />}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
