import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { useServerEvent } from "./events";
import type { JobDetail } from "./types";

/** Call `fn` at most once per `ms`, trailing. */
export function useThrottled(fn: () => void, ms = 400): () => void {
  const timer = useRef<number | null>(null);
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => () => {
    if (timer.current) window.clearTimeout(timer.current);
  }, []);
  return useCallback(() => {
    if (timer.current) return;
    timer.current = window.setTimeout(() => {
      timer.current = null;
      ref.current();
    }, ms);
  }, [ms]);
}

/** A job's detail, kept fresh by server events (stage progress, queue changes, files on disk). */
export function useJob(id: string) {
  const [detail, setDetail] = useState<JobDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const reload = useCallback(
    () =>
      api.job(id).then(
        (d) => {
          setDetail(d);
          setError(null);
        },
        (e: Error) => setError(e.message),
      ),
    [id],
  );
  const later = useThrottled(() => void reload(), 500);
  useEffect(() => {
    void reload();
  }, [reload]);
  useServerEvent((e) => {
    if (e.type === "stage" && e.data.job_id === id) {
      const ev = e.data;
      setDetail((d) => {
        if (!d) return d;
        const st = { ...d.job.stages[ev.stage], status: ev.status, progress: ev.progress };
        if (ev.message && ev.message !== "started") st.message = ev.message;
        if (ev.error) st.error = ev.error;
        return { ...d, job: { ...d.job, stages: { ...d.job.stages, [ev.stage]: st } } };
      });
      if (ev.status !== "running") later();
    } else if ((e.type === "files" || e.type === "jobs") && e.data.job_id === id) later();
    else if (e.type === "queue") later();
  });
  return { detail, error, reload };
}
