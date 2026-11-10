import { memo, useEffect, useRef, useState } from "react";
import { isProtected, lineNeedsReview, lineText, wordDoubt } from "../lib/lyricsOps";
import { fmtTime } from "../lib/time";
import type { LineDoc } from "../types";

export interface LineRowProps {
  line: LineDoc;
  li: number;
  selectedWi: number | null; // selection in this line
  activeWi: number | null; // word being sung (this line is active) or null
  tapWi: number | null; // next word to tap in this line
  errorWords: string; // comma-joined word indexes with errors ("" none; "L" for the line)
  editing: boolean;
  onWord: (li: number, wi: number) => void;
  onSeekLine: (li: number) => void;
  onCommitLine: (li: number, text: string | null) => void;
  rowRef: (li: number, el: HTMLDivElement | null) => void;
}

function LineRowImpl(p: LineRowProps) {
  const { line, li } = p;
  const errs = new Set(p.errorWords ? p.errorWords.split(",") : []);
  const [text, setText] = useState(() => lineText(line));
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (p.editing) {
      setText(lineText(line));
      requestAnimationFrame(() => {
        input.current?.focus();
        input.current?.select();
      });
    }
  }, [p.editing, line]);

  const review = lineNeedsReview(line);
  const reviewed = line.words.every(isProtected);
  const cls = ["lrow", p.activeWi !== null && "active", p.selectedWi !== null && "selected", errs.size && "has-error"]
    .filter(Boolean).join(" ");
  return (
    <div className={cls} ref={(el) => p.rowRef(li, el)} data-line={line.id}>
      <button className="ltime" title="Play from this line" onClick={() => p.onSeekLine(li)}>
        {fmtTime(line.start, 1)}
      </button>
      {p.editing ? (
        <form className="lineedit" onSubmit={(e) => {
          e.preventDefault();
          p.onCommitLine(li, text);
        }}>
          <input ref={input} value={text} aria-label={`Text of line ${line.id}`} onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Escape" && p.onCommitLine(li, null)} />
          <button className="primary small">Apply</button>
          <button type="button" className="small" onClick={() => p.onCommitLine(li, null)}>Cancel</button>
        </form>
      ) : (
        <>
          <div className="lwords">
            {line.section && <span className="lsection">{line.section}</span>}
            {line.words.map((w, wi) => {
              const doubt = wordDoubt(w);
              const c = [
                "word", doubt, w.origin === "manual" || w.origin === "ai" ? "mine" : null,
                p.activeWi !== null && wi < p.activeWi && "sung", p.activeWi === wi && "sung",
                p.selectedWi === wi && "selected", errs.has(String(wi)) && "err", p.tapWi === wi && "tapnext",
              ].filter(Boolean).join(" ");
              return (
                <button key={wi} className={c} onClick={() => p.onWord(li, wi)} aria-label={w.text}
                  title={`${fmtTime(w.start)} to ${fmtTime(w.end)}, ${w.origin}${isProtected(w) ? "" : `, confidence ${w.conf.toFixed(2)}`}`}>
                  {w.text}
                </button>
              );
            })}
          </div>
          <div className="lflags">
            {review && <span className="flag" title={line.flags.join(", ")}>check</span>}
            {reviewed && <span className="flag ok">edited</span>}
          </div>
        </>
      )}
    </div>
  );
}

export const LineRow = memo(LineRowImpl);
