// Sample-accurate playback with the Web Audio API, as a three-channel mixer: the instrumental, the
// vocal stem (guide vocals) and a synth playing the detected melody. All three run on the
// AudioContext clock, so seeking, pausing and changing speed keep them together.
// `time` is what you hear right now (output latency subtracted), which is what tap-to-sync needs.
import { ATTACK, RELEASE, noteGain, planNotes, type Clock, type SynthNote } from "./synth";

type Listener = () => void;
export type Channel = "inst" | "voc" | "synth";
export const CHANNELS: Channel[] = ["inst", "voc", "synth"];

export interface ChannelState {
  volume: number; // 0..1
  muted: boolean;
}
export type Mix = Record<Channel, ChannelState>;

export const DEFAULT_MIX: Mix = {
  inst: { volume: 1, muted: false },
  voc: { volume: 0.5, muted: true },
  synth: { volume: 0.6, muted: true },
};

const LOOKAHEAD_S = 1.2; // context seconds of synth notes scheduled ahead
const TICK_MS = 120;

export class StemPlayer {
  private ctx: AudioContext | null = null;
  private inst: AudioBuffer | null = null;
  private voc: AudioBuffer | null = null;
  private sources: AudioBufferSourceNode[] = [];
  private gains: Partial<Record<Channel, GainNode>> = {};
  private startedAt = 0; // ctx time when playback (re)started
  private offset = 0; // song time at startedAt
  private _rate = 1;
  private _mix: Mix = structuredClone(DEFAULT_MIX);
  private listeners = new Set<Listener>();
  // synth
  private notes: SynthNote[] = [];
  private tuning = 0;
  private scheduled = new Set<number>();
  private horizon = 0; // song time scheduled up to
  private voices = new Set<OscillatorNode>();
  private timer: number | null = null;
  playing = false;
  duration = 0;

  private audio(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext({ latencyHint: "interactive" });
      for (const ch of CHANNELS) {
        const g = this.ctx.createGain();
        g.gain.value = this.level(ch);
        g.connect(this.ctx.destination);
        this.gains[ch] = g;
      }
    }
    return this.ctx;
  }

  private level(ch: Channel): number {
    const c = this._mix[ch];
    return c.muted ? 0 : c.volume;
  }

  private async decode(url: string): Promise<AudioBuffer> {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`could not load ${url}: ${r.status}`);
    return await this.audio().decodeAudioData(await r.arrayBuffer());
  }

  async loadInstrumental(url: string): Promise<void> {
    this.inst = await this.decode(url);
    this.duration = this.inst.duration;
    this.emit();
  }

  async loadVocals(url: string): Promise<void> {
    this.voc = await this.decode(url);
    if (this.playing) this.play(this.time);
    this.emit();
  }

  get hasVocals(): boolean {
    return this.voc !== null;
  }

  /** The melody to synthesise. `tuning` (semitones off A440) is added to every note. */
  setNotes(notes: SynthNote[], tuning = 0): void {
    this.notes = notes.slice().sort((a, b) => a.start - b.start);
    this.tuning = tuning;
    if (this.playing) this.play(this.time);
  }

  get time(): number {
    if (!this.playing || !this.ctx) return this.offset;
    const latency = (this.ctx.outputLatency || 0) + (this.ctx.baseLatency || 0);
    const t = this.offset + (this.ctx.currentTime - this.startedAt - latency) * this._rate;
    return Math.max(this.offset - 0.001, Math.min(t, this.duration));
  }

  get rate(): number {
    return this._rate;
  }

  set rate(r: number) {
    const t = this.time;
    this._rate = r;
    if (this.playing) this.play(t); // re-schedules the synth at the new rate (and pitch)
    this.emit();
  }

  get mix(): Mix {
    return this._mix;
  }

  setChannel(ch: Channel, patch: Partial<ChannelState>): void {
    this._mix = { ...this._mix, [ch]: { ...this._mix[ch], ...patch } };
    const g = this.gains[ch];
    if (g && this.ctx) g.gain.setTargetAtTime(this.level(ch), this.ctx.currentTime, 0.015);
    this.emit();
  }

  setMix(mix: Mix): void {
    for (const ch of CHANNELS) this.setChannel(ch, mix[ch]);
  }

  play(from?: number): void {
    if (!this.inst) return;
    const ctx = this.audio();
    void ctx.resume();
    this.stopSources();
    const start = Math.max(0, Math.min(from ?? this.offset, this.duration - 0.01));
    const when = ctx.currentTime + 0.03;
    const make = (buf: AudioBuffer, gain: GainNode) => {
      const s = ctx.createBufferSource();
      s.buffer = buf;
      s.playbackRate.value = this._rate;
      s.connect(gain);
      s.start(when, start);
      return s;
    };
    this.sources = [make(this.inst, this.gains.inst!)];
    if (this.voc) this.sources.push(make(this.voc, this.gains.voc!));
    const main = this.sources[0];
    main.onended = () => {
      if (this.sources[0] === main) {
        this.playing = false;
        this.offset = this.duration;
        this.stopSynth();
        this.emit();
      }
    };
    this.startedAt = when;
    this.offset = start;
    this.playing = true;
    this.horizon = start;
    this.scheduled.clear();
    this.scheduleSynth();
    this.timer = window.setInterval(() => this.scheduleSynth(), TICK_MS);
    this.emit();
  }

  private clock(): Clock {
    return { startedAt: this.startedAt, offset: this.offset, rate: this._rate };
  }

  private scheduleSynth(): void {
    const ctx = this.ctx;
    if (!ctx || !this.playing || !this.notes.length) return;
    const clock = this.clock();
    const to = clock.offset + (ctx.currentTime + LOOKAHEAD_S - clock.startedAt) * clock.rate;
    const plan = planNotes(this.notes, this.horizon, to, clock, this.tuning, this.scheduled);
    this.horizon = Math.max(this.horizon, to);
    for (const p of plan) {
      this.scheduled.add(p.index);
      const when = Math.max(p.when, ctx.currentTime);
      if (p.stop <= when) continue;
      const osc = ctx.createOscillator();
      osc.type = "triangle";
      osc.frequency.value = p.freq;
      const env = ctx.createGain();
      const peak = noteGain(this.notes[p.index].conf);
      env.gain.setValueAtTime(0, when);
      env.gain.linearRampToValueAtTime(peak, when + ATTACK);
      env.gain.setValueAtTime(peak, Math.max(when + ATTACK, p.stop));
      env.gain.linearRampToValueAtTime(0, Math.max(when + ATTACK, p.stop) + RELEASE);
      osc.connect(env);
      env.connect(this.gains.synth!);
      osc.start(when);
      osc.stop(Math.max(when + ATTACK, p.stop) + RELEASE + 0.02);
      this.voices.add(osc);
      osc.onended = () => {
        this.voices.delete(osc);
        env.disconnect();
      };
    }
  }

  private stopSynth(): void {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
    for (const o of this.voices) {
      o.onended = null;
      try {
        o.stop();
      } catch {
        /* not started */
      }
      o.disconnect();
    }
    this.voices.clear();
    this.scheduled.clear();
  }

  pause(): void {
    if (!this.playing) return;
    this.offset = this.time;
    this.playing = false;
    this.stopSources();
    this.emit();
  }

  toggle(): void {
    if (this.playing) this.pause();
    else this.play(this.offset >= this.duration - 0.05 ? 0 : this.offset);
  }

  seek(t: number): void {
    const to = Math.max(0, Math.min(t, this.duration));
    if (this.playing) this.play(to);
    else {
      this.offset = to;
      this.emit();
    }
  }

  private stopSources(): void {
    for (const s of this.sources) {
      s.onended = null;
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
      s.disconnect();
    }
    this.sources = [];
    this.stopSynth();
  }

  subscribe(l: Listener): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }

  private emit(): void {
    this.listeners.forEach((l) => l());
  }

  /** Stop and release the audio device (buffers are kept; playing again opens a new context). */
  dispose(): void {
    if (this.playing) this.offset = this.time;
    this.playing = false;
    this.stopSources();
    void this.ctx?.close();
    this.ctx = null;
    this.gains = {};
  }
}
