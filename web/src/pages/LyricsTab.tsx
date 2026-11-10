// The lyrics editor: play-along preview, the line/word list, the inspector, tap-to-sync.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api } from "../api";
import { useServerEvent } from "../events";
import { Inspector } from "../components/Inspector";
import { LineRow } from "../components/LineRow";
import { MelodyPanel, useMelody } from "../components/MelodyPanel";
import { Mixer } from "../components/Mixer";
import { PasteLyrics } from "../components/PasteLyrics";
import { PlayAlong } from "../components/PlayAlong";
import { Clock, Timeline } from "../components/Timeline";
import * as ops from "../lib/lyricsOps";
import { activeLine, activeWord } from "../lib/playalong";
import { currentTarget, startTap, tapDone, tapDown, tapUp, type TapMode, type TapSession } from "../lib/tapSync";
import { useClockSelect, usePlayer } from "../lib/usePlayer";
import { validateLyrics } from "../lib/validate";
import type { BackgroundDoc, Issue, JobDetail, LyricsDoc } from "../types";

type Sel = { li: number; wi: number } | null;
type Notice = { kind: "ok" | "warn" | "err"; text: string } | null;

const isTyping = (el: EventTarget | null) =>
  el instanceof HTMLElement && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);

export function LyricsTab({ detail }: { detail: JobDetail }) {
  const id = detail.job.id;
  const duration = detail.meta?.duration;
  const render = detail.job.settings.render;
  const running = detail.summary.queue === "running";

  const [doc, setDocState] = useState<LyricsDoc | null>(null);
  const [saved, setSaved] = useState<LyricsDoc | null>(null);
  const [missing, setMissing] = useState(false);
  const [past, setPast] = useState<LyricsDoc[]>([]);
  const [future, setFuture] = useState<LyricsDoc[]>([]);
  const [sel, setSel] = useState<Sel>(null);
  const [editingLine, setEditingLine] = useState<number | null>(null);
  const [tap, setTapState] = useState<TapSession | null>(null);
  const [latencyMs, setLatencyMs] = useState(60);
  const [notice, setNotice] = useState<Notice>(null);
  const [diskChanged, setDiskChanged] = useState(false);
  const [serverIssues, setServerIssues] = useState<Issue[]>([]);
  const [saving, setSaving] = useState(false);
  const [bg, setBg] = useState<BackgroundDoc | null>(null);
  const [follow, setFollow] = useState(true);
  const [exact, setExact] = useState(false);

  const docRef = useRef(doc);
  const tapRef = useRef(tap);
  const setDoc = (d: LyricsDoc | null) => {
    docRef.current = d;
    setDocState(d);
  };
  const setTap = (s: TapSession | null) => {
    tapRef.current = s;
    setTapState(s);
  };

  const { player, loading, error: audioError, vocalsLoading, loadVocals } = usePlayer(id, detail.files.instrumental?.mtime ?? null);
  const mel = useMelody(id);
  useEffect(() => {
    const m = mel.melody;
    if (!m) return player.setNotes([]);
    // exact: the pitch the singer hit (tuning included); otherwise the rounded note in the song's tuning
    player.setNotes(m.notes.map((n) => ({ start: n.start, end: n.end, conf: n.conf, midi: exact ? n.pitch : n.midi })),
      exact ? 0 : m.tuning);
  }, [mel.melody, exact, player]);
  const ensureVocals = async () => {
    if (detail.files.vocals) await loadVocals(detail.files.vocals.mtime);
  };

  // ------------------------------------------------------------------ loading and live updates
  const load = useCallback(() => {
    api.lyrics(id).then(
      (d) => {
        setDoc(d);
        setSaved(d);
        setPast([]);
        setFuture([]);
        setDiskChanged(false);
        setMissing(false);
        setServerIssues([]);
      },
      (e) => {
        if (e instanceof ApiError && e.status === 404) setMissing(true);
        else setNotice({ kind: "err", text: (e as Error).message });
      },
    );
  }, [id]);
  useEffect(load, [load]);
  useEffect(() => {
    api.background(id).then(setBg, () => setBg(null));
  }, [id]);

  const dirty = doc !== null && doc !== saved;
  useServerEvent((e) => {
    if (e.type !== "files" || e.data.job_id !== id) return;
    if (e.data.paths.includes("background.json")) api.background(id).then(setBg, () => {});
    if (!e.data.paths.includes("lyrics.final.json")) return;
    api.lyrics(id).then((d) => {
      if (saved && d.updated_at === saved.updated_at) return; // our own save
      if (!dirty && !tapRef.current) {
        load();
        setNotice({ kind: "ok", text: "lyrics.final.json changed on disk and was reloaded." });
      } else setDiskChanged(true);
    }, () => {});
  });

  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => {
      if (dirty) e.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  // ------------------------------------------------------------------ editing
  const apply = useCallback((next: LyricsDoc, select?: Sel) => {
    const cur = docRef.current;
    if (!cur || next === cur) return;
    setPast((p) => [...p.slice(-199), cur]);
    setFuture([]);
    setDoc(next);
    if (select !== undefined) {
      const ok = select && next.lines[select.li]?.words[select.wi] ? select : null;
      setSel(ok);
    }
  }, []);

  const undo = () => {
    if (!past.length || !doc) return;
    setFuture((f) => [doc, ...f]);
    setDoc(past[past.length - 1]);
    setPast(past.slice(0, -1));
  };
  const redo = () => {
    if (!future.length || !doc) return;
    setPast((p) => [...p, doc]);
    setDoc(future[0]);
    setFuture(future.slice(1));
  };

  const issues = useMemo(() => (doc ? validateLyrics(doc, duration) : []), [doc, duration]);
  const errors = issues.filter((i) => i.severity === "error");
  const errorLines = useMemo(() => new Set(errors.map((i) => i.lineId)), [errors]);
  const errorWords = useMemo(() => {
    const m = new Map<string, string[]>();
    for (const i of errors) m.set(i.lineId, [...(m.get(i.lineId) ?? []), i.wordIndex === null ? "L" : String(i.wordIndex)]);
    return m;
  }, [errors]);

  async function save(force = false) {
    if (!doc || errors.length) return;
    setSaving(true);
    setNotice(null);
    try {
      const r = await api.saveLyrics(id, doc, force ? undefined : saved?.updated_at);
      setDoc(r.lyrics);
      setSaved(r.lyrics);
      setDiskChanged(false);
      setServerIssues([]);
      const manual = ops.countManual(r.lyrics);
      setNotice({ kind: "ok", text: `Saved. ${manual} word${manual === 1 ? " is" : "s are"} marked as edited by you, so a re-sync keeps them.` });
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setDiskChanged(true);
        setNotice({ kind: "warn", text: e.message });
      } else if (e instanceof ApiError && e.issues.length) {
        setServerIssues(e.issues);
        setNotice({ kind: "err", text: e.message });
      } else setNotice({ kind: "err", text: (e as Error).message });
    } finally {
      setSaving(false);
    }
  }

  // ------------------------------------------------------------------ playback position
  const lines = doc?.lines ?? [];
  const pos = useClockSelect(player, (t) => {
    const li = activeLine(lines, t);
    if (li < 0 || t > lines[li].end + 0.4) return "-1:-1";
    return `${li}:${activeWord(lines[li], t)}`;
  });
  const [activeLi, activeWi] = pos.split(":").map(Number);

  const listRef = useRef<HTMLDivElement>(null);
  const rows = useRef(new Map<number, HTMLDivElement>());
  const rowRef = useCallback((li: number, el: HTMLDivElement | null) => {
    if (el) rows.current.set(li, el);
    else rows.current.delete(li);
  }, []);
  /** Centre a line in the list; `reveal` also scrolls the page so the row is on screen. */
  const scrollToLine = (li: number, reveal = false) => {
    const box = listRef.current;
    const row = rows.current.get(li);
    if (!box || !row) return;
    box.scrollTo({ top: row.offsetTop - box.clientHeight / 2 + row.clientHeight / 2, behavior: reveal ? "auto" : "smooth" });
    if (reveal) row.scrollIntoView({ block: "nearest" });
  };
  useEffect(() => {
    if (follow && player.playing && activeLi >= 0) scrollToLine(activeLi);
  }, [activeLi, follow]);

  const playFrom = (t: number) => player.play(Math.max(0, t));
  const onWord = useCallback((li: number, wi: number) => {
    const w = docRef.current?.lines[li]?.words[wi];
    if (!w) return;
    setSel({ li, wi });
    setEditingLine(null);
    player.seek(w.start);
  }, [player]);
  const onSeekLine = useCallback((li: number) => {
    const l = docRef.current?.lines[li];
    if (l) player.play(Math.max(0, l.start - 0.3));
  }, [player]);
  const onCommitLine = useCallback((li: number, text: string | null) => {
    setEditingLine(null);
    const cur = docRef.current;
    if (text === null || !cur) return;
    apply(ops.setLineText(cur, li, text), { li, wi: 0 });
  }, [apply]);

  function nextToCheck() {
    if (!doc) return;
    const from = (sel?.li ?? activeLi) + 1;
    for (let k = 0; k < doc.lines.length; k++) {
      const li = (from + k) % doc.lines.length;
      const line = doc.lines[li];
      if (errorLines.has(line.id) || ops.lineNeedsReview(line)) {
        const wi = Math.max(0, line.words.findIndex((w) => ops.wordDoubt(w) === "low"));
        setSel({ li, wi });
        player.seek(line.start);
        scrollToLine(li, true);
        return;
      }
    }
    setNotice({ kind: "ok", text: "Nothing left to check." });
  }

  // ------------------------------------------------------------------ tap-to-sync
  function beginTap(mode: TapMode) {
    const cur = docRef.current;
    if (!cur || !cur.lines.length) return;
    const fromLi = sel?.li ?? Math.max(0, activeLi);
    const s = startTap(cur, mode, fromLi, mode === "word" ? sel?.wi ?? 0 : 0, latencyMs / 1000);
    const first = currentTarget(s);
    if (!first) return;
    setPast((p) => [...p.slice(-199), cur]); // the whole session is one undo step
    setFuture([]);
    setTap(s);
    setEditingLine(null);
    (document.activeElement as HTMLElement | null)?.blur();
    const w = cur.lines[first.li].words[first.wi];
    player.play(Math.max(0, w.start - 2.5 * player.rate));
  }
  function endTap(message?: string) {
    const s = tapRef.current;
    setTap(null);
    player.pause();
    if (s && s.index > 0) {
      const last = s.targets[Math.min(s.index, s.targets.length) - 1];
      setSel(last);
      setNotice({ kind: "ok", text: message ?? `Tapped ${s.index} ${s.mode === "word" ? "word" : "line"}${s.index === 1 ? "" : "s"}. Save to keep them, or undo.` });
    }
  }

  // ------------------------------------------------------------------ keyboard
  const keys = useRef<(e: KeyboardEvent) => void>(() => {});
  keys.current = (e: KeyboardEvent) => {
    const cur = docRef.current;
    if (!cur) return;
    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.key.toLowerCase() === "s" && e.type === "keydown") {
      e.preventDefault();
      void save();
      return;
    }
    if (isTyping(e.target)) return;
    const s = tapRef.current;
    if (e.code === "Space") {
      e.preventDefault();
      if (s) {
        if (e.repeat) return;
        const r = (e.type === "keydown" ? tapDown : tapUp)(cur, s, player.time);
        if (r.doc !== cur) setDoc(r.doc);
        setTap(r.session);
        if (tapDone(r.session)) endTap();
      } else if (e.type === "keydown" && !e.repeat) player.toggle();
      return;
    }
    if (e.type !== "keydown") return;
    if (e.key === "Escape") {
      if (s) endTap();
      else setSel(null);
      return;
    }
    if (mod && e.key.toLowerCase() === "z") {
      e.preventDefault();
      if (e.shiftKey) redo();
      else undo();
      return;
    }
    if (s) return;
    if (e.key === "n") return nextToCheck();
    if (!sel) return;
    const line = cur.lines[sel.li];
    const step = (d: number) => {
      let { li, wi } = sel;
      wi += d;
      if (wi < 0 && li > 0) {
        li -= 1;
        wi = cur.lines[li].words.length - 1;
      } else if (wi >= line.words.length && li < cur.lines.length - 1) {
        li += 1;
        wi = 0;
      }
      if (cur.lines[li]?.words[wi]) {
        e.preventDefault();
        onWord(li, wi);
        scrollToLine(li, true);
      }
    };
    const nudge = (which: "start" | "end" | "both", d: number) => {
      e.preventDefault();
      apply(ops.nudgeWord(cur, sel.li, sel.wi, which, d), sel);
    };
    switch (e.key) {
      case "ArrowLeft": return step(-1);
      case "ArrowRight": return step(1);
      case ",": return nudge("both", -ops.NUDGE);
      case ".": return nudge("both", ops.NUDGE);
      case "[": return nudge("start", -ops.NUDGE);
      case "]": return nudge("start", ops.NUDGE);
      case "{": return nudge("end", -ops.NUDGE);
      case "}": return nudge("end", ops.NUDGE);
      case "Enter":
        e.preventDefault();
        return playFrom(line.words[sel.wi].start - 1);
      case "e":
        e.preventDefault();
        return setEditingLine(sel.li);
    }
  };
  useEffect(() => {
    const h = (e: KeyboardEvent) => keys.current(e);
    window.addEventListener("keydown", h);
    window.addEventListener("keyup", h);
    return () => {
      window.removeEventListener("keydown", h);
      window.removeEventListener("keyup", h);
    };
  }, []);
  useEffect(() => () => player.pause(), [player]);

  // ------------------------------------------------------------------ render
  if (missing) {
    return (
      <div style={{ display: "grid", gap: "1rem", maxWidth: 760 }}>
        <p className="notice warn">No synced lyrics yet. Run the steps up to Sync lyrics, or paste the lyrics below.</p>
        <PasteLyrics jobId={id} textSource="none" running={running} />
      </div>
    );
  }
  if (!doc) return <p className="muted">Loading the lyrics…</p>;

  const tapTarget = tap ? currentTarget(tap) : null;
  const tapWord = tapTarget ? doc.lines[tapTarget.li]?.words[tapTarget.wi] : null;
  const tapLabel = tap && tapWord
    ? tap.mode === "word" ? `Tap: ${tapWord.text}` : `Tap line: ${doc.lines[tapTarget!.li].words.map((w) => w.text).join(" ").slice(0, 40)}`
    : null;
  const toCheck = doc.lines.filter((l) => ops.lineNeedsReview(l)).length;
  const shownIssues = serverIssues.length ? serverIssues : issues;

  return (
    <div className="editor">
      <div className="editor-main">
        <PasteLyrics jobId={id} textSource={doc.text_source} running={running} />
        {diskChanged && (
          <div className="notice warn row">
            <span>lyrics.final.json changed on disk (a re-sync or a Claude session) while you were editing.</span>
            <span className="spacer" />
            <button className="small" onClick={load}>Load the disk version</button>
            <button className="small" onClick={() => save(true)} disabled={!!errors.length}>Keep mine and save</button>
          </div>
        )}
        <PlayAlong jobId={id} lines={doc.lines} player={player} render={render} background={bg} tapLabel={tapLabel}
          tapDown={!!tap?.down} onWordClick={onWord}
          idle={loading ? "Loading the instrumental…" : audioError} />
        <MelodyPanel jobId={id} player={player} melody={mel.melody} contour={mel.contour} missing={mel.missing}
          lyrics={doc} running={running} exact={exact} setExact={setExact} />
        <div className="transport">
          <div className="row">
            <button className="primary" style={{ minWidth: "5.5rem" }} disabled={loading || !!audioError}
              onClick={() => player.toggle()} aria-label={player.playing ? "Pause" : "Play"}>
              {player.playing ? "Pause" : "Play"}
            </button>
            <Clock player={player} />
            <button className="small quiet" onClick={() => player.seek(player.time - 5)} aria-label="Back 5 seconds">−5 s</button>
            <button className="small quiet" onClick={() => player.seek(player.time + 5)} aria-label="Forward 5 seconds">+5 s</button>
            <select aria-label="Speed" value={player.rate} onChange={(e) => (player.rate = Number(e.target.value))}>
              <option value={1}>1× speed</option>
              <option value={0.75}>0.75×</option>
              <option value={0.5}>0.5×</option>
            </select>
            <span className="spacer" />
            <label className="row" style={{ gap: "0.3rem" }}>
              <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> Follow
            </label>
            {tap ? (
              <button className="small on" onClick={() => endTap()}>Stop tapping</button>
            ) : (
              <>
                <button className="small" onClick={() => beginTap("word")} disabled={loading || !!audioError}>Tap words</button>
                <button className="small" onClick={() => beginTap("line")} disabled={loading || !!audioError}>Tap lines</button>
              </>
            )}
          </div>
          <Mixer player={player} canVocals={!!detail.files.vocals} canMelody={!!mel.melody?.notes.length}
            vocalsLoading={vocalsLoading} ensureVocals={ensureVocals} />
          <Timeline lines={doc.lines} player={player} duration={duration ?? player.duration} errorLines={errorLines}
            onSeek={(t) => player.seek(t)} />
        </div>
        <div ref={listRef} className="panel lines" style={{ maxHeight: "56vh", overflow: "auto", position: "relative" }}>
          {doc.lines.map((line, li) => (
            <LineRow key={line.id} line={line} li={li}
              selectedWi={sel?.li === li ? sel.wi : null}
              activeWi={activeLi === li ? activeWi : null}
              tapWi={tapTarget?.li === li ? tapTarget.wi : null}
              errorWords={(errorWords.get(line.id) ?? []).join(",")}
              editing={editingLine === li}
              onWord={onWord} onSeekLine={onSeekLine} onCommitLine={onCommitLine} rowRef={rowRef} />
          ))}
          {doc.lines.length === 0 && (
            <div style={{ padding: "1rem" }} className="row">
              <span className="muted">No lines.</span>
              <button className="small" onClick={() => {
                const text = window.prompt("Text of the first line:");
                if (text?.trim()) apply(ops.addLineAfter(doc, -1, text, duration), { li: 0, wi: 0 });
              }}>Add a line</button>
            </div>
          )}
        </div>
      </div>

      <aside className="side">
        <div className="panel">
          <div className="savebar">
            <button className="primary" onClick={() => save()} disabled={!dirty || saving || errors.length > 0 || !!tap}>
              {saving ? "Saving…" : "Save"}
            </button>
            <button className="quiet small" onClick={undo} disabled={!past.length || !!tap} aria-label="Undo">Undo</button>
            <button className="quiet small" onClick={redo} disabled={!future.length || !!tap} aria-label="Redo">Redo</button>
            <span className="spacer" />
            {dirty && <span className="faint"><i className="dirty-dot" /> Not saved</span>}
          </div>
          {notice && <p className={`notice ${notice.kind === "ok" ? "" : notice.kind}`}>{notice.text}</p>}
          {errors.length > 0 && <p className="error-text" style={{ fontSize: "0.88rem" }}>Fix {errors.length} problem{errors.length > 1 ? "s" : ""} to save:</p>}
          {shownIssues.length > 0 && (
            <div className="issues">
              {shownIssues.slice(0, 30).map((i, k) => (
                <button key={k} className={`quiet small ${i.severity === "error" ? "danger" : ""}`} onClick={() => {
                  const li = doc.lines.findIndex((l) => i.where.startsWith(`line ${l.id}`));
                  if (li >= 0) {
                    const m = /word (\d+)/.exec(i.where);
                    onWord(li, m ? Number(m[1]) : 0);
                    scrollToLine(li, true);
                  }
                }}>
                  {i.where}: {i.message}
                </button>
              ))}
            </div>
          )}
          <div className="row">
            <button className="small" onClick={nextToCheck}>Next line to check</button>
            <span className="faint">{toCheck ? `${toCheck} of ${doc.lines.length} lines flagged` : "No flagged lines"}</span>
          </div>
        </div>

        {tap && (
          <div className="panel">
            <h3>Tap to sync</h3>
            <p className="muted" style={{ fontSize: "0.9rem" }}>
              {tap.mode === "word"
                ? <>Press <kbd>Space</kbd> as each word starts and release it as the word ends. The next press also ends the word before.</>
                : <>Press <kbd>Space</kbd> as each line starts. The whole line moves, keeping its rhythm.</>}{" "}
              <kbd>Esc</kbd> stops.
            </p>
            <label className="row">
              Tap delay
              <input type="number" min={0} max={300} step={10} value={latencyMs} style={{ width: "5rem" }}
                onChange={(e) => setLatencyMs(Number(e.target.value))} /> ms
            </label>
            <p className="faint">{tap.index} of {tap.targets.length} tapped. Slow the song down if it goes too fast.</p>
          </div>
        )}

        {sel && doc.lines[sel.li]?.words[sel.wi] && !tap ? (
          <Inspector doc={doc} li={sel.li} wi={sel.wi} apply={apply} playFrom={playFrom} duration={duration}
            editLine={(li) => setEditingLine(li)} startTap={beginTap} />
        ) : !tap && (
          <div className="panel">
            <h3>Fix the lyrics</h3>
            <div className="legend">
              <span>Click a word to select it and jump there.</span>
              <span><button className="word low" tabIndex={-1}>word</button> probably wrong: not heard, or low confidence</span>
              <span><button className="word unsure" tabIndex={-1}>word</button> not sure</span>
              <span><button className="word mine" tabIndex={-1}>word</button> edited by you: a re-sync keeps it</span>
            </div>
            <div className="legend">
              <span><kbd>Space</kbd> play or pause, <kbd>←</kbd> <kbd>→</kbd> previous or next word</span>
              <span><kbd>,</kbd> <kbd>.</kbd> move the word 50 ms, <kbd>[</kbd> <kbd>]</kbd> its start, <kbd>{"{"}</kbd> <kbd>{"}"}</kbd> its end</span>
              <span><kbd>Enter</kbd> play from the word, <kbd>e</kbd> edit the line, <kbd>n</kbd> next line to check</span>
              <span><kbd>⌘Z</kbd> undo, <kbd>⌘S</kbd> save</span>
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}
