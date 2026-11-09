import { useJob } from "../hooks";
import { TABS, jobHref, type Tab } from "../router";
import { fmtTime } from "../lib/time";
import { PipelineTab } from "./PipelineTab";
import { LyricsTab } from "./LyricsTab";
import { RenderTab } from "./RenderTab";
import { BackgroundTab } from "./BackgroundTab";

const TAB_LABEL: Record<Tab, string> = {
  pipeline: "Steps",
  lyrics: "Lyrics",
  background: "Background",
  render: "Video",
};

export function JobPage({ id, tab }: { id: string; tab: Tab }) {
  const { detail, error, reload } = useJob(id);
  if (error && !detail) {
    return (
      <div className="notice err">
        {error}. <a href="#/">Back to the songs</a>
      </div>
    );
  }
  if (!detail) return <p className="muted">Loading…</p>;
  const { meta, summary } = detail;
  return (
    <>
      <div className="jobhead">
        {summary.thumbnail ? <img className="thumb" src={summary.thumbnail} alt="" /> : <div className="thumb" />}
        <div style={{ minWidth: 0 }}>
          <h1>{summary.title ?? id}</h1>
          <p className="muted">
            {summary.artist}
            {meta?.duration ? `, ${fmtTime(meta.duration, 0)}` : ""}
            {detail.job.url && (
              <>
                {" "}
                <a href={detail.job.url} target="_blank" rel="noreferrer">on YouTube</a>
              </>
            )}
          </p>
        </div>
      </div>
      <nav className="tabs" aria-label="Song sections">
        {TABS.map((t) => (
          <a key={t} href={jobHref(id, t)} aria-current={t === tab ? "page" : undefined}>
            {TAB_LABEL[t]}
          </a>
        ))}
      </nav>
      {tab === "pipeline" && <PipelineTab detail={detail} reload={reload} />}
      {tab === "lyrics" && <LyricsTab detail={detail} />}
      {tab === "background" && <BackgroundTab detail={detail} />}
      {tab === "render" && <RenderTab detail={detail} />}
    </>
  );
}
