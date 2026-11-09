import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { fileUrl } from "../api";
import { StemPlayer } from "./player";

export interface PlayerState {
  player: StemPlayer;
  loading: boolean;
  error: string | null;
  vocalsLoading: boolean;
}

/** A StemPlayer for a job's instrumental (vocals load on demand with `loadVocals`). */
export function usePlayer(jobId: string, instrumentalVersion: number | null) {
  const ref = useRef<StemPlayer | null>(null);
  if (!ref.current) ref.current = new StemPlayer();
  const player = ref.current;
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [vocalsLoading, setVocalsLoading] = useState(false);

  useEffect(() => () => player.dispose(), [player]);
  useEffect(() => {
    if (instrumentalVersion === null) {
      setLoading(false);
      setError("No instrumental yet: run the Separate vocals step.");
      return;
    }
    setLoading(true);
    setError(null);
    player.loadInstrumental(fileUrl(jobId, "stems/instrumental.wav", instrumentalVersion)).then(
      () => setLoading(false),
      (e: Error) => {
        setLoading(false);
        setError(e.message);
      },
    );
  }, [jobId, instrumentalVersion, player]);

  async function loadVocals(version: number) {
    if (player.hasVocals) return;
    setVocalsLoading(true);
    try {
      await player.loadVocals(fileUrl(jobId, "stems/vocals.wav", version));
    } finally {
      setVocalsLoading(false);
    }
  }

  // re-render on play/pause/rate/volume changes
  useSyncExternalStore(
    (cb) => player.subscribe(cb),
    () => `${player.playing}|${player.rate}|${player.vocals}|${player.duration}|${player.hasVocals}`,
  );
  return { player, loading, error, vocalsLoading, loadVocals };
}

/** The player's time, updated every animation frame while it plays. */
export function useClock(player: StemPlayer): number {
  const [t, setT] = useState(player.time);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      setT(player.time);
      if (player.playing) raf = requestAnimationFrame(tick);
    };
    const unsub = player.subscribe(() => {
      cancelAnimationFrame(raf);
      tick();
    });
    tick();
    return () => {
      cancelAnimationFrame(raf);
      unsub();
    };
  }, [player]);
  return t;
}

/** Like useClock, but only re-renders when `select(time)` changes. */
export function useClockSelect<T>(player: StemPlayer, select: (t: number) => T): T {
  const selRef = useRef(select);
  selRef.current = select;
  const [v, setV] = useState(() => select(player.time));
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const next = selRef.current(player.time);
      setV((prev) => (Object.is(prev, next) ? prev : next));
      if (player.playing) raf = requestAnimationFrame(tick);
    };
    const unsub = player.subscribe(() => {
      cancelAnimationFrame(raf);
      tick();
    });
    tick();
    return () => {
      cancelAnimationFrame(raf);
      unsub();
    };
  }, [player]);
  // recompute when the selector's inputs change (e.g. the lyrics were edited)
  const now = select(player.time);
  return Object.is(now, v) ? v : now;
}
