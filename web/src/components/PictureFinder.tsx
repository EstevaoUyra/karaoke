import { useState } from "react";
import { api, type PictureCandidate } from "../api";

const SOURCE_LABEL: Record<string, string> = { wikimedia: "Wikimedia Commons", unsplash: "Unsplash", pexels: "Pexels" };
const SOURCE_KEY: Record<string, string> = { unsplash: "UNSPLASH_ACCESS_KEY", pexels: "PEXELS_API_KEY" };

/** Search free pictures for one scene and put the chosen one in it. */
export function PictureFinder({ jobId, sources, onUse, onClose }: {
  jobId: string;
  sources: Record<string, boolean>;
  onUse: (pick: string) => Promise<void>;
  onClose: () => void;
}) {
  const [q, setQ] = useState("");
  const [on, setOn] = useState<Record<string, boolean>>(sources);
  const [found, setFound] = useState<{ candidates: PictureCandidate[]; notes: string[] } | null>(null);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  const chosen = Object.keys(sources).filter((s) => sources[s] && on[s]);
  // what went wrong at a source; a source that is simply switched off is shown by its checkbox
  const problems = (found?.notes ?? []).filter((n) => !n.includes(": off") && !n.includes("nothing usable"));

  async function search() {
    if (q.trim().length < 2 || !chosen.length) return;
    setBusy("search");
    setErr("");
    try {
      setFound(await api.searchPictures(jobId, q.trim(), chosen));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy("");
    }
  }

  async function use(pick: string) {
    setBusy(pick);
    setErr("");
    try {
      await onUse(pick);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="finder">
      <form className="row" onSubmit={(e) => { e.preventDefault(); void search(); }}>
        <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} style={{ flex: 1, minWidth: 180 }}
          placeholder="What should the picture show? For example: beach at dusk" aria-label="Search pictures" />
        <button className="primary" disabled={busy !== "" || q.trim().length < 2 || !chosen.length}>
          {busy === "search" ? "Searching…" : "Search"}
        </button>
        <button type="button" onClick={onClose}>Close</button>
      </form>
      <div className="row">
        {Object.keys(sources).map((s) => (
          <label key={s} className="row faint" title={sources[s] ? "" : `Off: add ${SOURCE_KEY[s]} to the .env file to switch it on`}>
            <input type="checkbox" disabled={!sources[s]} checked={sources[s] && on[s]}
              onChange={(e) => setOn({ ...on, [s]: e.target.checked })} />
            {SOURCE_LABEL[s] ?? s}{!sources[s] && " (off)"}
          </label>
        ))}
      </div>
      {err && <p className="notice err">{err}</p>}
      {problems.map((n) => <p key={n} className="notice warn">{n}</p>)}
      {found && found.candidates.length === 0 && problems.length === 0 && (
        <p className="faint">Nothing found. Try fewer or different words.</p>
      )}
      {found && found.candidates.length > 0 && (
        <div className="finder-grid">
          {found.candidates.map((c) => (
            <figure key={c.pick}>
              <img src={c.thumb} alt={c.title} loading="lazy" referrerPolicy="no-referrer" />
              <figcaption>
                <span className="finder-title" title={c.title}>{c.title || "Untitled"}</span>
                <span className="faint">
                  <a href={c.page} target="_blank" rel="noreferrer">{SOURCE_LABEL[c.source] ?? c.source}</a>
                  {" · "}{c.license}{" · "}{c.size[0]}×{c.size[1]}
                </span>
                <button className="small" disabled={busy !== ""} onClick={() => use(c.pick)}>
                  {busy === c.pick ? "Fetching…" : "Use this picture"}
                </button>
              </figcaption>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}
