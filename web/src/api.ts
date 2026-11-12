import type {
  BackgroundDoc, ContourDoc, Issue, MelodyDoc, JobDetail, JobSettingsDoc, JobSummary, LyricsDoc, LyricsSources, QueueSnapshot,
  RenderSettingsDoc, RunRequest, StageName, Privacy, YouTubeStatus,
} from "./types";

import type { Structure } from "./types/generated/structure";

export class ApiError extends Error {
  status: number;
  issues: Issue[];
  constructor(status: number, message: string, issues: Issue[] = []) {
    super(message);
    this.status = status;
    this.issues = issues;
  }
}

/** A picture a search found; `pick` names it when asking the server to put it in a scene. */
export interface PictureCandidate {
  pick: string; source: string; title: string; thumb: string; size: [number, number];
  license: string; credit: string; page: string;
}

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: {} };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    (init.headers as Record<string, string>)["Content-Type"] = "application/json";
  }
  const r = await fetch(url, init);
  if (!r.ok) {
    let message = `${r.status} ${r.statusText}`;
    let issues: Issue[] = [];
    try {
      const j = await r.json();
      const d = j.detail;
      if (typeof d === "string") message = d;
      else if (d && typeof d === "object" && !Array.isArray(d)) {
        message = d.message ?? message;
        issues = d.issues ?? [];
      } else if (Array.isArray(d)) {
        message = d.map((e: { loc?: unknown[]; msg?: string }) => `${(e.loc ?? []).join(".")}: ${e.msg}`).join("; ");
      }
    } catch {
      /* not JSON */
    }
    throw new ApiError(r.status, message, issues);
  }
  return (await r.json()) as T;
}

const j = (id: string) => `/api/jobs/${encodeURIComponent(id)}`;

export const api = {
  jobs: () => req<JobSummary[]>("GET", "/api/jobs"),
  job: (id: string) => req<JobDetail>("GET", j(id)),
  createJob: (url: string, lyrics?: string) =>
    req<{ job: JobSummary; run: RunRequest | null }>("POST", "/api/jobs", { url, lyrics: lyrics || null, run: true }),
  uploadAudio: (file: File, title: string, artist: string, lyrics: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("title", title);
    fd.append("artist", artist);
    fd.append("lyrics", lyrics);
    return req<{ job: JobSummary; run: RunRequest | null }>("POST", "/api/jobs/upload", fd);
  },
  run: (id: string, opts: { from_stage?: StageName; only?: StageName[]; force?: boolean } = {}) =>
    req<RunRequest>("POST", `${j(id)}/run`, opts),
  cancel: (id: string) => req<{ cancelled: boolean }>("POST", `${j(id)}/cancel`),
  queue: () => req<QueueSnapshot>("GET", "/api/queue"),
  log: (id: string, stage: StageName, lines = 200) =>
    req<{ text: string; size: number }>("GET", `${j(id)}/logs/${stage}?lines=${lines}`),

  lyrics: (id: string) => req<LyricsDoc>("GET", `${j(id)}/lyrics`),
  saveLyrics: (id: string, doc: LyricsDoc, base?: string) =>
    req<{ lyrics: LyricsDoc; issues: Issue[]; remarked: number }>(
      "PUT", `${j(id)}/lyrics${base ? `?base=${encodeURIComponent(base)}` : ""}`, doc),
  lyricsSources: (id: string) => req<LyricsSources>("GET", `${j(id)}/lyrics/sources`),
  saveUserLyrics: (id: string, text: string, run: boolean) =>
    req<{ saved: boolean; run: RunRequest | null }>("PUT", `${j(id)}/lyrics/user`, { text, run }),

  melody: (id: string) => req<MelodyDoc>("GET", `${j(id)}/melody`),
  melodyContour: (id: string) => req<ContourDoc>("GET", `${j(id)}/melody/contour`),
  midiUrl: (id: string) => `${j(id)}/melody.mid`,

  settings: (id: string) => req<JobSettingsDoc>("GET", `${j(id)}/settings`),
  render: (id: string, render: RenderSettingsDoc) => req<RunRequest>("POST", `${j(id)}/render`, { render }),

  background: (id: string) => req<BackgroundDoc>("GET", `${j(id)}/background`),
  saveBackground: (id: string, bg: BackgroundDoc) =>
    req<{ background: BackgroundDoc; issues: Issue[] }>("PUT", `${j(id)}/background`, bg),
  structure: (id: string) => req<Structure>("GET", `${j(id)}/structure`),
  backgroundPresets: (id: string) =>
    req<{
      presets: { name: string; label: string; available: boolean }[]; has_structure: boolean;
      picture_sources: Record<string, boolean>;
    }>("GET", `${j(id)}/background/presets`),
  searchPictures: (id: string, q: string, sources: string[]) =>
    req<{ query: string; notes: string[]; candidates: PictureCandidate[] }>(
      "GET", `${j(id)}/background/pictures?q=${encodeURIComponent(q)}${sources.map((s) => `&source=${encodeURIComponent(s)}`).join("")}`),
  useScenePicture: (id: string, sceneId: string, pick: string) =>
    req<{ background: BackgroundDoc; issues: Issue[] }>(
      "POST", `${j(id)}/background/scenes/${encodeURIComponent(sceneId)}/picture`, { pick }),
  applyBackgroundPreset: (id: string, name: string, replaceLocked = false) =>
    req<{ background: BackgroundDoc; issues: Issue[] }>(
      "POST", `${j(id)}/background/preset`, { name, replace_locked: replaceLocked }),
  youtube: (id: string) => req<YouTubeStatus>("GET", `${j(id)}/youtube`),
  youtubeDraft: (id: string, redo = false) => req<YouTubeStatus>("POST", `${j(id)}/youtube/draft`, { redo }),
  saveYoutube: (id: string, d: { title: string; description: string; tags: string[]; privacy: Privacy; playlist: string | null }) =>
    req<YouTubeStatus>("PUT", `${j(id)}/youtube`, d),
  youtubeUpload: (id: string, again = false) => req<YouTubeStatus>("POST", `${j(id)}/youtube/upload`, { again }),
  youtubeUpdate: (id: string, privacy?: Privacy) =>
    req<YouTubeStatus>("POST", `${j(id)}/youtube/update`, { privacy: privacy ?? null }),
  uploadSceneImage: (id: string, sceneId: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<{ background: BackgroundDoc; issues: Issue[] }>(
      "POST", `${j(id)}/background/scenes/${encodeURIComponent(sceneId)}/image`, fd);
  },
};

/** URL of a media file in the job folder; `v` busts the browser cache when the file changes. */
export function fileUrl(id: string, path: string, v?: number | string, download = false): string {
  const q = new URLSearchParams();
  if (v !== undefined) q.set("v", String(v));
  if (download) q.set("download", "1");
  const qs = q.toString();
  return `${j(id)}/files/${path}${qs ? `?${qs}` : ""}`;
}
