// Job-file types come from the JSON Schemas (src/types/generated, `npm run gen:types`).
// Fields with a default are optional in the schema; the API always sends every field, so the
// "Doc" types below mark them required for data read from the server.
import type * as LF from "./generated/lyrics_final";
import type * as BG from "./generated/background";
import type * as JB from "./generated/job";
import type { Meta } from "./generated/meta";
import type { LyricsChoice } from "./generated/lyrics_choice";

type Full<T> = { [K in keyof T]-?: T[K] };

export type WordOrigin = LF.WordOrigin;
export type WordDoc = Full<LF.Word>;
export type LineDoc = Omit<Full<LF.Line>, "words"> & { words: WordDoc[] };
export type LyricsDoc = Omit<Full<LF.LyricsFinal>, "lines"> & { lines: LineDoc[] };
export type LyricsFinal = LF.LyricsFinal;

export type BgSource = BG.BgSource;
export type SceneDoc = Omit<Full<BG.Scene>, "transition"> & { transition: Full<BG.Transition> };
export type BackgroundDoc = Omit<Full<BG.Background>, "scenes" | "resolution"> & {
  scenes: SceneDoc[];
  resolution: [number, number];
};

export type StageStatus = JB.StageStatus;
export type StageStateDoc = Full<JB.StageState>;
export type RenderSettingsDoc = Full<JB.RenderSettings>;
export type JobSettingsDoc = Omit<Full<JB.JobSettings>, "render"> & { render: RenderSettingsDoc };
export type JobDoc = Omit<Full<JB.Job>, "settings" | "stages"> & {
  settings: JobSettingsDoc;
  stages: Record<StageName, StageStateDoc>;
};
export type { Meta, LyricsChoice };

export const STAGES = ["ingest", "separate", "lyrics_fetch", "transcribe", "align", "backgrounds", "render"] as const;
export type StageName = (typeof STAGES)[number];

// ---------------------------------------------------------------- API responses (karaoke/api)

export type QueueState = "queued" | "running" | null;

export interface JobSummary {
  id: string;
  url: string | null;
  created_at: string;
  updated_at: string;
  title: string | null;
  artist: string | null;
  duration: number | null;
  thumbnail: string | null;
  stages: Record<StageName, { status: StageStatus; progress: number }>;
  queue: QueueState;
}

export interface FileInfo {
  size: number;
  mtime: number;
}

export interface Issue {
  severity: "error" | "warning";
  file: string;
  where: string;
  message: string;
}

export interface JobDetail {
  job: JobDoc;
  meta: Meta | null;
  summary: JobSummary;
  files: Record<
    | "audio" | "instrumental" | "vocals" | "lyrics_final" | "background" | "preview_video" | "final_video"
    | "user_lyrics" | "review_notes",
    FileInfo | null
  >;
  issues: Issue[];
}

export interface RunRequest {
  id: string;
  job_id: string;
  from_stage: StageName | null;
  only: StageName[] | null;
  force: boolean;
  state: "queued" | "running" | "done" | "failed" | "cancelled";
  error: string | null;
  queued_at: number;
  started_at: number | null;
  finished_at: number | null;
}

export interface QueueSnapshot {
  running: RunRequest | null;
  queued: RunRequest[];
  history: RunRequest[];
}

export interface StageEventData {
  job_id: string;
  stage: StageName;
  status: StageStatus;
  progress: number;
  message: string;
  error: string | null;
}

export interface LyricsSources {
  chosen: LyricsChoice | null;
  chosen_text: string | null;
  files: { name: string; text: string }[];
  /** lyrics/sources/letras.json: the song's letras.mus.br page, for the user to copy lyrics from. */
  letras: { url: string; title: string; artist: string } | null;
}
