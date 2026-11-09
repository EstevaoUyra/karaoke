import { useEffect, useState } from "react";
import { api } from "./api";
import { useConnected, useServerEvent } from "./events";
import { Home } from "./pages/Home";
import { JobPage } from "./pages/JobPage";
import { jobHref, useRoute } from "./router";
import type { QueueSnapshot, StageEventData } from "./types";
import { STAGE_LABEL } from "./components/labels";

function Activity() {
  const connected = useConnected();
  const [queue, setQueue] = useState<QueueSnapshot | null>(null);
  const [last, setLast] = useState<StageEventData | null>(null);
  useEffect(() => {
    api.queue().then(setQueue).catch(() => {});
  }, []);
  useServerEvent((e) => {
    if (e.type === "queue") setQueue(e.data);
    if (e.type === "stage") setLast(e.data);
  });
  const run = queue?.running;
  const waiting = queue?.queued.length ?? 0;
  return (
    <div className="activity" aria-live="polite">
      {run && (
        <a href={jobHref(run.job_id)}>
          {last && last.job_id === run.job_id && last.status === "running"
            ? `${STAGE_LABEL[last.stage]} ${Math.round(last.progress * 100)}%`
            : "Running"}
          {` in ${run.job_id}`}
        </a>
      )}
      {waiting > 0 && <span>{waiting} waiting</span>}
      <span title={connected ? "Changes on disk show up live" : "Reconnecting to the server…"}>
        <i className={`livedot ${connected ? "on" : ""}`} />
      </span>
    </div>
  );
}

export function App() {
  const route = useRoute();
  return (
    <>
      <header className="topbar">
        <a className="wordmark" href="#/">karaoke</a>
        <div className="spacer" />
        <Activity />
      </header>
      <main>{route.page === "home" ? <Home /> : <JobPage key={route.id} id={route.id} tab={route.tab} />}</main>
    </>
  );
}
