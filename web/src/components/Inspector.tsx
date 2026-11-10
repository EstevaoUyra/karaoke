// Edit the selected word and its line.
import { useEffect, useRef, useState } from "react";
import * as ops from "../lib/lyricsOps";
import { NUDGE } from "../lib/lyricsOps";
import { parseTime } from "../lib/time";
import type { LyricsDoc } from "../types";

const ORIGIN_TEXT: Record<string, string> = {
  matched: "heard by Whisper",
  fuzzy: "roughly heard by Whisper",
  interpolated: "not heard, time guessed between neighbours",
  manual: "edited by you",
  ai: "edited by the AI review",
};

function TimeInput({ value, onCommit, label }: { value: number; onCommit: (v: number) => void; label: string }) {
  const [text, setText] = useState(value.toFixed(2));
  useEffect(() => setText(value.toFixed(2)), [value]);
  const commit = () => {
    const v = parseTime(text);
    if (v === null || Math.abs(v - value) < 1e-4) setText(value.toFixed(2));
    else onCommit(v);
  };
  return (
    <input aria-label={label} value={text} inputMode="decimal" onChange={(e) => setText(e.target.value)} onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") setText(value.toFixed(2));
      }} />
  );
}

export function Inspector(props: {
  doc: LyricsDoc;
  li: number;
  wi: number;
  apply: (doc: LyricsDoc, select?: { li: number; wi: number } | null) => void;
  playFrom: (t: number) => void;
  editLine: (li: number) => void;
  startTap: (mode: "word" | "line") => void;
  duration: number | undefined;
}) {
  const { doc, li, wi, apply } = props;
  const line = doc.lines[li];
  const w = line?.words[wi];
  const [text, setText] = useState(w?.text ?? "");
  const [newWord, setNewWord] = useState("");
  const textRef = useRef<HTMLInputElement>(null);
  useEffect(() => setText(w?.text ?? ""), [w?.text, li, wi]);
  if (!line || !w) return null;

  const commitText = () => {
    if (text.trim() === w.text) return;
    apply(ops.setWordText(doc, li, wi, text), { li, wi });
  };
  const nudge = (which: "start" | "end" | "both", d: number) => apply(ops.nudgeWord(doc, li, wi, which, d), { li, wi });

  return (
    <>
      <div className="panel" aria-label="Selected word">
        <div>
          <div className="inspector-word">{w.text}</div>
          <p className="faint">
            {ORIGIN_TEXT[w.origin] ?? w.origin}
            {!ops.isProtected(w) && `, confidence ${w.conf.toFixed(2)}`}
          </p>
        </div>
        <form className="row" onSubmit={(e) => {
          e.preventDefault();
          commitText();
        }}>
          <input ref={textRef} value={text} onChange={(e) => setText(e.target.value)} onBlur={commitText}
            aria-label="Word text" style={{ flex: 1, minWidth: 0 }} />
        </form>
        <p className="faint">Type a space to split the word in two.</p>
        <div className="timegrid">
          <label>Start</label>
          <TimeInput label="Word start (seconds)" value={w.start} onCommit={(v) => apply(ops.setWordTiming(doc, li, wi, v, Math.max(w.end, v + ops.MIN_WORD)), { li, wi })} />
          <button className="small" aria-label="Start 50 ms earlier" onClick={() => nudge("start", -NUDGE)}>−50</button>
          <button className="small" aria-label="Start 50 ms later" onClick={() => nudge("start", NUDGE)}>+50</button>
          <label>End</label>
          <TimeInput label="Word end (seconds)" value={w.end} onCommit={(v) => apply(ops.setWordTiming(doc, li, wi, w.start, v), { li, wi })} />
          <button className="small" aria-label="End 50 ms earlier" onClick={() => nudge("end", -NUDGE)}>−50</button>
          <button className="small" aria-label="End 50 ms later" onClick={() => nudge("end", NUDGE)}>+50</button>
          <label>Word</label>
          <span className="faint">move by 50 ms</span>
          <button className="small" aria-label="Move word 50 ms earlier" onClick={() => nudge("both", -NUDGE)}>−50</button>
          <button className="small" aria-label="Move word 50 ms later" onClick={() => nudge("both", NUDGE)}>+50</button>
        </div>
        <div className="btngrid">
          <button className="small" onClick={() => props.playFrom(w.start - 1)}>Play from here</button>
          <button className="small" disabled={wi >= line.words.length - 1}
            onClick={() => apply(ops.mergeWithNext(doc, li, wi), { li, wi })}>Join with next word</button>
          <button className="small danger" onClick={() => {
            const next = ops.deleteWord(doc, li, wi);
            const stillThere = next.lines[li]?.id === line.id;
            apply(next, stillThere ? { li, wi: Math.max(0, Math.min(wi, next.lines[li].words.length - 1)) } : null);
          }}>Delete word</button>
        </div>
        <form className="row" onSubmit={(e) => {
          e.preventDefault();
          if (!newWord.trim()) return;
          apply(ops.insertWord(doc, li, wi + 1, newWord), { li, wi: wi + 1 });
          setNewWord("");
        }}>
          <input placeholder="New word after this one" value={newWord} onChange={(e) => setNewWord(e.target.value)}
            aria-label="New word" style={{ flex: 1, minWidth: 0 }} />
          <button className="small" disabled={!newWord.trim()}>Add</button>
        </form>
      </div>

      <div className="panel" aria-label="Selected line">
        <div className="row">
          <h3>Line {line.id}</h3>
          <input aria-label="Section" placeholder="section" defaultValue={line.section ?? ""} key={`${line.id}-${line.section}`}
            style={{ width: "7.5rem" }} onBlur={(e) => e.target.value !== (line.section ?? "") &&
              apply(ops.setLineSection(doc, li, e.target.value), { li, wi })} />
        </div>
        <div className="btngrid">
          <button className="small" onClick={() => props.editLine(li)}>Edit line text</button>
          <button className="small" onClick={() => apply(ops.markLineReviewed(doc, li), { li, wi })}
            title="Keep this line as it is: protects it from a re-run of Sync lyrics">Mark line correct</button>
          <button className="small" aria-label="Move line 50 ms earlier" onClick={() => apply(ops.shiftLine(doc, li, -NUDGE), { li, wi })}>Line −50</button>
          <button className="small" aria-label="Move line 50 ms later" onClick={() => apply(ops.shiftLine(doc, li, NUDGE), { li, wi })}>Line +50</button>
          <button className="small" disabled={wi === 0} onClick={() => apply(ops.splitLineAt(doc, li, wi), { li: li + 1, wi: 0 })}>
            New line from this word
          </button>
          <button className="small" disabled={li >= doc.lines.length - 1}
            onClick={() => apply(ops.mergeLineWithNext(doc, li), { li, wi })}>Join with next line</button>
          <button className="small" onClick={() => {
            const text = window.prompt("Text of the new line (added after this one):");
            if (text?.trim()) apply(ops.addLineAfter(doc, li, text, props.duration), { li: li + 1, wi: 0 });
          }}>Add line after</button>
          <button className="small danger" onClick={() => apply(ops.deleteLine(doc, li), null)}>Delete line</button>
        </div>
        <div className="row">
          <button className="small" onClick={() => props.startTap("word")}>Tap words from here</button>
          <button className="small" onClick={() => props.startTap("line")}>Tap lines from here</button>
        </div>
      </div>
    </>
  );
}
