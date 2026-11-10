// Sample-accurate playback of the stems with the Web Audio API: the instrumental, plus the vocal
// stem as an optional guide track, both started on the same clock so they never drift apart.
// `time` is what you hear right now (output latency subtracted), which is what tap-to-sync needs.

type Listener = () => void;

export class StemPlayer {
  private ctx: AudioContext | null = null;
  private inst: AudioBuffer | null = null;
  private voc: AudioBuffer | null = null;
  private sources: AudioBufferSourceNode[] = [];
  private instGain: GainNode | null = null;
  private vocGain: GainNode | null = null;
  private startedAt = 0; // ctx time when playback (re)started
  private offset = 0; // song time at startedAt
  private _rate = 1;
  private _vocals = 0; // guide vocal gain 0..1
  private listeners = new Set<Listener>();
  playing = false;
  duration = 0;

  private audio(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext({ latencyHint: "interactive" });
      this.instGain = this.ctx.createGain();
      this.vocGain = this.ctx.createGain();
      this.vocGain.gain.value = this._vocals;
      this.instGain.connect(this.ctx.destination);
      this.vocGain.connect(this.ctx.destination);
    }
    return this.ctx;
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
    if (this.playing) this.restart();
    this.emit();
  }

  get hasVocals(): boolean {
    return this.voc !== null;
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
    if (this.playing) this.play(t);
    this.emit();
  }

  get vocals(): number {
    return this._vocals;
  }

  set vocals(g: number) {
    this._vocals = g;
    if (this.vocGain) this.vocGain.gain.value = g;
    this.emit();
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
    this.sources = [make(this.inst, this.instGain!)];
    if (this.voc) this.sources.push(make(this.voc, this.vocGain!));
    const main = this.sources[0];
    main.onended = () => {
      if (this.sources[0] === main) {
        this.playing = false;
        this.offset = this.duration;
        this.emit();
      }
    };
    this.startedAt = when;
    this.offset = start;
    this.playing = true;
    this.emit();
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

  private restart(): void {
    this.play(this.time);
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
    this.instGain = null;
    this.vocGain = null;
  }
}
