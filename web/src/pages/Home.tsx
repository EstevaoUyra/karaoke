import { useEffect, useState, type FormEvent } from "react";
import { api } from "../api";
import { useServerEvent } from "../events";
import { useThrottled } from "../hooks";
import { jobHref, navigate } from "../router";
import { STAGES, type JobSummary } from "../types";
import { fmtTime } from "../lib/time";
import { STAGE_LABEL, STATUS_LABEL } from "../components/labels";

export function StageStrip({ job }: { job: JobSummary }) {
  return (
    <div className="strip" aria-hidden>
      {STAGES.map((s) => (
        <span key={s} className={job.stages[s]?.status} title={`${STAGE_LABEL[s]}: ${STATUS_LABEL[job.stages[s]?.status ?? "pending"]}`} />
      ))}
    </div>
  );
}

export function jobStatusText(job: JobSummary): string {
  if (job.queue === "queued") return "Waiting in the queue";
  const failed = STAGES.find((s) => job.stages[s]?.status === "failed");
  if (failed) return `${STAGE_LABEL[failed]} failed`;
  const running = STAGES.find((s) => job.stages[s]?.status === "running");
  if (running) return `${STAGE_LABEL[running]}, ${Math.round((job.stages[running].progress ?? 0) * 100)}%`;
  const done = STAGES.filter((s) => ["done", "skipped"].includes(job.stages[s]?.status));
  if (done.length === STAGES.length) return "Video ready";
  if (done.length === 0) return "Not started";
  return `${done.length} of ${STAGES.length} steps done`;
}

function NewSong({ onCreated }: { onCreated: () => void }) {
  const [url, setUrl] = useState("");
  const [lyrics, setLyrics] = useState("");
  const [showLyrics, setShowLyrics] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [artist, setArtist] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const text = showLyrics ? lyrics : "";
      const r = uploading && file
        ? await api.uploadAudio(file, title, artist, text)
        : await api.createJob(url, text);
      onCreated();
      navigate(jobHref(r.job.id));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="panel newsong" onSubmit={submit}>
      <h1>Make a karaoke video</h1>
      {!uploading && (
        <div className="urlrow">
          <input
            type="url" required placeholder="Paste a YouTube link" value={url} aria-label="YouTube link"
            onChange={(e) => setUrl(e.target.value)} autoFocus
          />
          <button className="primary" disabled={busy || !url.trim()}>{busy ? "Starting…" : "Start"}</button>
        </div>
      )}
      {uploading && (
        <div className="field" style={{ gap: "0.6rem" }}>
          <input type="file" accept="audio/*,video/*" aria-label="Audio file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          <div className="row">
            <input placeholder="Title" value={title} onChange={(e) => setTitle(e.target.value)} aria-label="Title" />
            <input placeholder="Artist" value={artist} onChange={(e) => setArtist(e.target.value)} aria-label="Artist" />
            <button className="primary" disabled={busy || !file}>{busy ? "Uploading…" : "Upload and start"}</button>
          </div>
        </div>
      )}
      <div className="disclosures">
        <button type="button" className="disclosure-toggle" aria-expanded={showLyrics}
          onClick={() => setShowLyrics(!showLyrics)}>
          {showLyrics ? "Find the lyrics automatically" : "Paste the lyrics yourself"}
        </button>
        <button type="button" className="disclosure-toggle" aria-expanded={uploading}
          onClick={() => setUploading(!uploading)}>
          {uploading ? "Use a YouTube link" : "Upload an audio file instead"}
        </button>
      </div>
      {showLyrics && (
        <div className="field">
          <label htmlFor="lyrics">Lyrics, one sung line per line. They win over every other source.</label>
          <textarea id="lyrics" rows={8} value={lyrics} onChange={(e) => setLyrics(e.target.value)} />
        </div>
      )}
      {uploading && !showLyrics && (
        <p className="faint">Without a link, lyrics come from LRCLIB by title and artist, or from Whisper.</p>
      )}
      {error && <p className="notice err">{error}</p>}
    </form>
  );
}

export function Home() {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = () => api.jobs().then(setJobs, (e: Error) => setError(e.message));
  const later = useThrottled(() => void load(), 800);
  useEffect(() => {
    void load();
  }, []);
  useServerEvent((e) => {
    if (e.type === "jobs" || e.type === "queue") later();
    if (e.type === "stage") {
      setJobs((js) => js && js.map((j) => (j.id !== e.data.job_id ? j : {
        ...j, stages: { ...j.stages, [e.data.stage]: { status: e.data.status, progress: e.data.progress } },
      })));
    }
  });

  return (
    <div className="home">
      <NewSong onCreated={load} />
      <section>
        <h2 style={{ marginBottom: "0.5rem" }}>Songs</h2>
        {error && <p className="notice err">Could not load the songs: {error}</p>}
        {jobs && jobs.length === 0 && <p className="muted">No songs yet. Paste a link above to make the first one.</p>}
        <div className="joblist">
          {jobs?.map((j) => (
            <a key={j.id} className="jobrow" href={jobHref(j.id, j.stages.align?.status === "done" ? "lyrics" : "pipeline")}>
              {j.thumbnail ? <img className="thumb" src={j.thumbnail} alt="" loading="lazy" /> : <div className="thumb" />}
              <div style={{ minWidth: 0 }}>
                <div className="jobtitle">{j.title ?? j.id}</div>
                <div className="jobmeta">
                  {[j.artist, j.duration ? fmtTime(j.duration, 0) : null].filter(Boolean).join(", ")}
                </div>
                <div className="jobmeta">{jobStatusText(j)}</div>
              </div>
              <StageStrip job={j} />
            </a>
          ))}
        </div>
      </section>
    </div>
  );
}
