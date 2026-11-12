import { useEffect, useState } from "react";
import { api } from "../api";
import { useServerEvent } from "../events";
import type { Privacy, YouTubeStatus } from "../types";

const PRIVACY: Record<Privacy, string> = { private: "Private", unlisted: "Unlisted", public: "Public" };

/** The Video tab's YouTube box: prepare the title, description and tags, upload, then change privacy. */
export function YouTubePanel({ id, hasFinal }: { id: string; hasFinal: boolean }) {
  const [st, setSt] = useState<YouTubeStatus | null>(null);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState("");
  const [privacy, setPrivacy] = useState<Privacy>("unlisted");
  const [playlist, setPlaylist] = useState("");
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  function show(s: YouTubeStatus) {
    setSt(s);
    if (s.draft) {
      setTitle(s.draft.title);
      setDescription(s.draft.description);
      setTags(s.draft.tags.join(", "));
      setPrivacy(s.draft.privacy);
      setPlaylist(s.draft.playlist ?? "");
      setDirty(false);
    }
  }

  const load = () => api.youtube(id).then(show, (e: Error) => setError(e.message));
  useEffect(() => {
    void load();
  }, [id, hasFinal]);
  useServerEvent((e) => {
    if (e.type !== "youtube" || e.data.job_id !== id) return;
    setSt((s) => (s ? { ...s, task: { status: e.data.status, message: e.data.message } } : s));
    if (e.data.status === "done" || e.data.status === "failed") void load();
  });

  async function act(f: () => Promise<YouTubeStatus>) {
    setBusy(true);
    setError(null);
    setNote(null);
    try {
      const s = await f();
      show(s);
      if (s.messages?.length) setNote(s.messages.join(" "));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const edited = () => ({
    title, description, privacy, playlist: playlist.trim() || null,
    tags: tags.split(",").map((t) => t.trim()).filter(Boolean),
  });
  const save = () => act(() => api.saveYoutube(id, edited()));
  // Save first when there are edits, so what goes out is what is on screen.
  const saveThen = (f: () => Promise<YouTubeStatus>) =>
    act(async () => {
      if (dirty) await api.saveYoutube(id, edited());
      return f();
    });

  const d = st?.draft ?? null;
  const task = st?.task;
  const working = busy || task?.status === "queued" || task?.status === "running";
  const uploaded = Boolean(d?.video_id);
  const tagChars = tags.split(",").map((t) => t.trim()).filter(Boolean)
    .reduce((n, t, i) => n + t.length + (t.includes(" ") ? 2 : 0) + (i ? 1 : 0), 0);
  const edit = <T,>(f: (v: T) => void) => (v: T) => {
    f(v);
    setDirty(true);
  };

  return (
    <div className="videobox youtube">
      <div className="row">
        <h3>YouTube</h3>
        <span className="spacer" />
        {uploaded && d && (
          <a className={`ytbadge ${d.privacy}`} href={`https://youtu.be/${d.video_id}`} target="_blank" rel="noreferrer">
            On YouTube · {PRIVACY[d.privacy]}
          </a>
        )}
      </div>

      {!st ? (
        <p className="faint">Loading…</p>
      ) : !st.video_ready ? (
        <p className="faint">Render the final video first; this is where it goes to YouTube.</p>
      ) : !d ? (
        <div className="row">
          <p className="muted" style={{ margin: 0 }}>
            Prepare the title, description (with the picture credits) and tags, read them, then upload.
          </p>
          <span className="spacer" />
          <button className="primary" disabled={working} onClick={() => void act(() => api.youtubeDraft(id))}>
            Prepare upload
          </button>
        </div>
      ) : (
        <form className="form" style={{ padding: 0 }} onSubmit={(e) => {
          e.preventDefault();
          void save();
        }}>
          <div className="field">
            <label htmlFor="yt-title">Title <span className="faint">{title.length}/100</span></label>
            <input id="yt-title" value={title} maxLength={100} onChange={(e) => edit(setTitle)(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="yt-desc">Description</label>
            <textarea id="yt-desc" rows={10} value={description} onChange={(e) => edit(setDescription)(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="yt-tags">Tags, separated by commas <span className="faint">{tagChars}/500</span></label>
            <textarea id="yt-tags" rows={3} value={tags} onChange={(e) => edit(setTags)(e.target.value)} />
          </div>
          <div className="pair">
            <div className="field">
              <label htmlFor="yt-privacy">Privacy</label>
              <select id="yt-privacy" value={privacy} onChange={(e) => edit(setPrivacy)(e.target.value as Privacy)}>
                {(Object.keys(PRIVACY) as Privacy[]).map((p) => <option key={p} value={p}>{PRIVACY[p]}</option>)}
              </select>
            </div>
            <div className="field">
              <label htmlFor="yt-playlist">Playlist</label>
              <input id="yt-playlist" value={playlist} placeholder="none" onChange={(e) => edit(setPlaylist)(e.target.value)} />
            </div>
          </div>

          {d.warnings.length > 0 && !uploaded && (
            <ul className="notice warn ytwarn">{d.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
          )}
          {st.errors.length > 0 && <p className="notice err">{st.errors.join("; ")}</p>}
          {!st.client_secret && (
            <p className="notice warn">No Google sign-in is set up: see docs/youtube.md.</p>
          )}
          {st.client_secret && !st.signed_in && !uploaded && (
            <p className="faint">The first upload opens Google's sign-in page in this computer's browser.</p>
          )}
          {uploaded && st.video_changed && (
            <p className="notice warn">
              The video was rendered again after the upload. YouTube cannot swap a video's file: upload a new copy and delete the old one in Studio.
            </p>
          )}

          <div className="row">
            <button type="submit" disabled={working || !dirty}>Save</button>
            <button type="button" disabled={working} title="Rebuild the draft from the song, losing your edits"
              onClick={() => {
                if (window.confirm("Rebuild the title, description and tags from the song? Your edits are lost.")) {
                  void act(() => api.youtubeDraft(id, true));
                }
              }}>
              Rebuild draft
            </button>
            <span className="spacer" />
            {uploaded ? (
              <>
                {st.video_changed && (
                  <button type="button" disabled={working} onClick={() => void saveThen(() => api.youtubeUpload(id, true))}>
                    Upload new copy
                  </button>
                )}
                <button type="button" className="primary" disabled={working}
                  onClick={() => void saveThen(() => api.youtubeUpdate(id))}>
                  Update on YouTube
                </button>
              </>
            ) : (
              <button type="button" className="primary" disabled={working || st.errors.length > 0 || !st.client_secret}
                onClick={() => void saveThen(() => api.youtubeUpload(id))}>
                Upload to YouTube
              </button>
            )}
          </div>
        </form>
      )}

      {task && (task.status !== "done" || !uploaded) && (
        <p className={`notice ${task.status === "failed" ? "err" : ""}`} role="status">
          {task.status === "queued" ? "Waiting: " : task.status === "running" ? "Uploading: " : task.status === "failed" ? "Upload failed: " : ""}
          {task.message}
        </p>
      )}
      {note && <p className="faint">{note}</p>}
      {error && <p className="notice err">{error}</p>}
    </div>
  );
}
