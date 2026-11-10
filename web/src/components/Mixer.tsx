// Three channels: the instrumental, the vocal stem and the melody synth. "Check pitch" puts the
// synth next to the real voice, with the music low, to hear where the detected melody is wrong.
import { CHANNELS, type Channel, type Mix, type StemPlayer } from "../lib/player";

const LABEL: Record<Channel, string> = { inst: "Music", voc: "Vocals", synth: "Melody" };

export const PRESETS: Record<"music" | "pitch" | "sing", { label: string; title: string; mix: Mix }> = {
  music: {
    label: "Music only",
    title: "The instrumental, as in the video",
    mix: { inst: { volume: 1, muted: false }, voc: { volume: 0.5, muted: true }, synth: { volume: 0.6, muted: true } },
  },
  pitch: {
    label: "Check pitch",
    title: "The real voice and the detected melody together, music low: wrong notes clash",
    mix: { inst: { volume: 0.15, muted: false }, voc: { volume: 0.85, muted: false }, synth: { volume: 0.55, muted: false } },
  },
  sing: {
    label: "Melody guide",
    title: "The instrumental with the detected melody on top",
    mix: { inst: { volume: 0.9, muted: false }, voc: { volume: 0.5, muted: true }, synth: { volume: 0.6, muted: false } },
  },
};

export function Mixer({ player, canVocals, canMelody, vocalsLoading, ensureVocals }: {
  player: StemPlayer;
  canVocals: boolean;
  canMelody: boolean;
  vocalsLoading: boolean;
  ensureVocals: () => Promise<void>;
}) {
  const mix = player.mix;
  const enabled = (ch: Channel) => (ch === "voc" ? canVocals : ch === "synth" ? canMelody : true);
  const apply = async (m: Mix) => {
    const want = { ...m };
    if (!canVocals) want.voc = { ...want.voc, muted: true };
    if (!canMelody) want.synth = { ...want.synth, muted: true };
    if (!want.voc.muted) await ensureVocals();
    player.setMix(want);
  };
  const active = (Object.keys(PRESETS) as (keyof typeof PRESETS)[]).find((k) =>
    CHANNELS.every((ch) => {
      const a = PRESETS[k].mix[ch];
      const b = mix[ch];
      return a.muted === b.muted && (a.muted || Math.abs(a.volume - b.volume) < 0.01);
    }));

  return (
    <div className="mixer" role="group" aria-label="Mixer">
      {CHANNELS.map((ch) => {
        const c = mix[ch];
        return (
          <div key={ch} className={`strip-ch ${c.muted || !enabled(ch) ? "off" : ""}`}>
            <button className={`small ${c.muted ? "" : "on"}`} disabled={!enabled(ch) || (ch === "voc" && vocalsLoading)}
              aria-pressed={!c.muted} title={c.muted ? `Unmute ${LABEL[ch].toLowerCase()}` : `Mute ${LABEL[ch].toLowerCase()}`}
              onClick={async () => {
                if (ch === "voc" && c.muted) await ensureVocals();
                player.setChannel(ch, { muted: !c.muted });
              }}>
              {ch === "voc" && vocalsLoading ? "Loading…" : LABEL[ch]}
            </button>
            <input type="range" min={0} max={1} step={0.05} value={c.volume} disabled={!enabled(ch)}
              aria-label={`${LABEL[ch]} volume`} onChange={(e) => player.setChannel(ch, { volume: Number(e.target.value) })} />
          </div>
        );
      })}
      <span className="mixer-presets">
        {(Object.keys(PRESETS) as (keyof typeof PRESETS)[]).map((k) => (
          <button key={k} className={`small ${active === k ? "on" : ""}`} title={PRESETS[k].title}
            disabled={(k !== "music" && !canMelody) || (k === "pitch" && !canVocals)}
            onClick={() => void apply(PRESETS[k].mix)}>
            {PRESETS[k].label}
          </button>
        ))}
      </span>
    </div>
  );
}
