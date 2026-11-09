// Paste the correct lyrics (lyrics/sources/user.txt beats every other source), with a pointer to
// the song's letras.mus.br page when lyrics_fetch found one.
import { useEffect, useState } from "react";
import { api } from "../api";
import { useServerEvent } from "../events";
import type { LyricsSources } from "../types";

export function PasteLyrics({ jobId, textSource, running }: { jobId: string; textSource: string; running: boolean }) {
  const [sources, setSources] = useState<LyricsSources | null>(null);
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [state, setState] = useState<"idle" | "saving" | "saved" | "queued">("idle");
  const [error, setError] = useState<string | null>(null);

  const load = () =>
    api.lyricsSources(jobId).then((s) => {
      setSources(s);
      setText((cur) => cur || s.files.find((f) => f.name === "user.txt")?.text || "");
    }, () => {});
  useEffect(() => {
    void load();
  }, [jobId]);
  useServerEvent((e) => {
    if (e.type === "files" && e.data.job_id === jobId && e.data.paths.some((p) => p.startsWith("lyrics/"))) void load();
  });

  const letras = sources?.letras;
  const hasUser = sources?.files.some((f) => f.name === "user.txt");

  async function save(run: boolean) {
    setState("saving");
    setError(null);
    try {
      await api.saveUserLyrics(jobId, text, run);
      setState(run ? "queued" : "saved");
      void load();
    } catch (e) {
      setError((e as Error).message);
      setState("idle");
    }
  }

  return (
    <div className="panel" style={{ padding: "0.8rem 1rem", display: "grid", gap: "0.6rem" }}>
      <div className="row">
        <span className="muted" style={{ fontSize: "0.9rem" }}>
          {textSource === "user"
            ? "These lyrics come from the text you pasted."
            : textSource === "whisper"
              ? "No lyrics were found, so these words are what Whisper heard."
              : `Words from ${textSource}.`}
        </span>
        <span className="spacer" />
        <button className="small" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? "Close" : hasUser ? "Edit pasted lyrics" : "Paste the correct lyrics"}
        </button>
      </div>
      {letras && (
        <div className="letras">
          <span>
            Found on letras.mus.br: {letras.title}
            {letras.artist ? ` – ${letras.artist}` : ""}
          </span>
          <a href={letras.url} target="_blank" rel="noreferrer noopener">open</a>
          <span className="faint">Copy the lyrics there and paste them here.</span>
        </div>
      )}
      {open && (
        <>
          <textarea rows={10} value={text} aria-label="Pasted lyrics" placeholder="One sung line per line."
            onChange={(e) => {
              setText(e.target.value);
              setState("idle");
            }} />
          <div className="row">
            <button className="primary" disabled={state === "saving" || running || (!text.trim() && !hasUser)} onClick={() => save(true)}>
              Save and re-sync
            </button>
            <button disabled={state === "saving"} onClick={() => save(false)}>Save only</button>
            <span className="faint">
              {state === "saved" && "Saved to lyrics/sources/user.txt. Re-run from Find lyrics to use it."}
              {state === "queued" && "Saved. Re-running from Find lyrics: Sync lyrics and Render follow."}
              {state === "idle" && "Re-syncing keeps the words you fixed by hand wherever they still fit."}
            </span>
          </div>
          {state === "saved" && (
            <div className="row">
              <button className="small" disabled={running} onClick={() => api.run(jobId, { from_stage: "lyrics_fetch" }).then(() => setState("queued"))}>
                Re-run from Find lyrics
              </button>
            </div>
          )}
          {error && <p className="notice err">{error}</p>}
        </>
      )}
    </div>
  );
}
