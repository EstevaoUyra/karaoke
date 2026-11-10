/** 83.456 → "1:23.46" (or "1:23" with digits = 0). */
export function fmtTime(t: number | null | undefined, digits = 2): string {
  if (t === null || t === undefined || !Number.isFinite(t)) return "–";
  const sign = t < 0 ? "-" : "";
  const a = Math.abs(t);
  const m = Math.floor(a / 60);
  let s = (a - m * 60).toFixed(digits);
  if (Number(s) >= 60) return fmtTime((t < 0 ? -1 : 1) * (m + 1) * 60, digits);
  if (s.indexOf(".") === 1 || (digits === 0 && s.length === 1)) s = "0" + s;
  return `${sign}${m}:${s}`;
}

/** "1:23.46" or "83.46" → seconds; null if it doesn't parse. */
export function parseTime(text: string): number | null {
  const s = text.trim();
  const m = /^(-)?(?:(\d+):)?(\d+(?:\.\d*)?)$/.exec(s);
  if (!m) return null;
  const v = (m[2] ? Number(m[2]) * 60 : 0) + Number(m[3]);
  return m[1] ? -v : v;
}

export function fmtDuration(s: number | null | undefined): string {
  if (s === null || s === undefined) return "";
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

export const round3 = (x: number) => Math.round(x * 1000) / 1000;
