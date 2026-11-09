// One EventSource per tab (GET /api/events), fanned out to components with useServerEvent.
import { useEffect, useRef, useSyncExternalStore } from "react";
import type { QueueSnapshot, StageEventData } from "./types";

export type ServerEvent =
  | { type: "stage"; data: StageEventData }
  | { type: "queue"; data: QueueSnapshot }
  | { type: "files"; data: { job_id: string; paths: string[] } }
  | { type: "jobs"; data: { job_id: string } }
  | { type: "hello"; data: Record<string, never> };

type Listener = (e: ServerEvent) => void;

const listeners = new Set<Listener>();
let source: EventSource | null = null;
let connected = false;
const connListeners = new Set<() => void>();

function setConnected(v: boolean) {
  if (connected !== v) {
    connected = v;
    connListeners.forEach((l) => l());
  }
}

function ensureSource() {
  if (source || typeof EventSource === "undefined") return;
  source = new EventSource("/api/events");
  source.onopen = () => setConnected(true);
  source.onerror = () => setConnected(false); // the browser reconnects by itself
  for (const type of ["stage", "queue", "files", "jobs", "hello"] as const) {
    source.addEventListener(type, (m) => {
      let data: unknown = {};
      try {
        data = JSON.parse((m as MessageEvent).data);
      } catch {
        return;
      }
      if (type === "hello") setConnected(true);
      const ev = { type, data } as ServerEvent;
      listeners.forEach((l) => l(ev));
    });
  }
}

/** Call `handler` for every server event. The latest handler is used, so closures stay fresh. */
export function useServerEvent(handler: Listener): void {
  const ref = useRef(handler);
  ref.current = handler;
  useEffect(() => {
    ensureSource();
    const l: Listener = (e) => ref.current(e);
    listeners.add(l);
    return () => {
      listeners.delete(l);
    };
  }, []);
}

export function useConnected(): boolean {
  return useSyncExternalStore(
    (cb) => {
      ensureSource();
      connListeners.add(cb);
      return () => connListeners.delete(cb);
    },
    () => connected,
  );
}
