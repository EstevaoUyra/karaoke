// A tiny hash router: #/ (home) and #/jobs/<id>/<tab>.
import { useSyncExternalStore } from "react";

export type Tab = "pipeline" | "lyrics" | "background" | "render";
export const TABS: Tab[] = ["pipeline", "lyrics", "background", "render"];

export type Route = { page: "home" } | { page: "job"; id: string; tab: Tab };

export function parseRoute(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  if (parts[0] === "jobs" && parts[1]) {
    const tab = (TABS as string[]).includes(parts[2]) ? (parts[2] as Tab) : "pipeline";
    return { page: "job", id: parts[1], tab };
  }
  return { page: "home" };
}

export const jobHref = (id: string, tab: Tab = "pipeline") => `#/jobs/${encodeURIComponent(id)}/${tab}`;

export function navigate(href: string) {
  window.location.hash = href.replace(/^#/, "");
}

export function useRoute(): Route {
  const hash = useSyncExternalStore(
    (cb) => {
      window.addEventListener("hashchange", cb);
      return () => window.removeEventListener("hashchange", cb);
    },
    () => window.location.hash,
  );
  return parseRoute(hash);
}
