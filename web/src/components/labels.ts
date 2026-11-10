import type { StageName, StageStatus } from "../types";

export const STAGE_LABEL: Record<StageName, string> = {
  ingest: "Download",
  separate: "Separate vocals",
  lyrics_fetch: "Find lyrics",
  transcribe: "Transcribe",
  align: "Sync lyrics",
  melody: "Melody",
  structure: "Song structure",
  backgrounds: "Background",
  render: "Render video",
};

export const STAGE_HELP: Record<StageName, string> = {
  ingest: "Video, audio and title from YouTube",
  separate: "Split the song into vocals and instrumental",
  lyrics_fetch: "LRCLIB, Genius, YouTube subtitles or your pasted text",
  transcribe: "Whisper listens to the vocal stem for word timings",
  align: "Lyrics words with Whisper's timing → lyrics.final.json",
  melody: "The sung notes and the key, for the pitch guide",
  structure: "Sections, loudness and beats, for backgrounds that follow the song",
  backgrounds: "The default background, unless you or the AI made one",
  render: "Subtitles, background and instrumental into the video",
};

export const STATUS_LABEL: Record<StageStatus, string> = {
  pending: "Not run",
  running: "Running",
  done: "Done",
  skipped: "Up to date",
  failed: "Failed",
};
