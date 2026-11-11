// WebAudio playback of a composed piece (a small synth, one timbre per voice) and offline rendering to WAV.
// Audio starts only after a click (browsers require a user gesture).
import { tempoMap, endTick } from './timing.js';

const WAVES = ['triangle', 'sine', 'square', 'sawtooth'];
const hz = p => 440 * 2 ** ((p - 69) / 12);

function events(piece) {
  const map = tempoMap(piece.tempos);
  return piece.notes.map(([track, , pitch, tick, dur, vel]) => {
    const t0 = map.toSec(tick);
    return { track, pitch, vel, t0, t1: Math.max(map.toSec(tick + dur), t0 + 0.03) };
  }).sort((a, b) => a.t0 - b.t0);
}

export const pieceSeconds = piece => tempoMap(piece.tempos).toSec(endTick(piece.notes));

function voice(ctx, out, e, at) {
  const osc = ctx.createOscillator(), g = ctx.createGain();
  osc.type = WAVES[e.track % WAVES.length];
  osc.frequency.value = hz(e.pitch);
  const peak = (e.vel / 127) * (osc.type === 'square' || osc.type === 'sawtooth' ? 0.05 : 0.12);
  const a = at + e.t0, b = at + e.t1;
  g.gain.setValueAtTime(0, a);
  g.gain.linearRampToValueAtTime(peak, a + 0.01);
  g.gain.setTargetAtTime(peak * 0.6, a + 0.01, 0.08);
  g.gain.setValueAtTime(peak * 0.6, Math.max(a + 0.012, b - 0.002));
  g.gain.linearRampToValueAtTime(0, b + 0.08);
  osc.connect(g).connect(out);
  osc.start(a); osc.stop(b + 0.1);
}

function chain(ctx) {
  const comp = ctx.createDynamicsCompressor(), master = ctx.createGain();
  master.gain.value = 0.8;
  comp.connect(master).connect(ctx.destination);
  return comp;
}

export class Player {
  constructor() { this.ctx = null; this.timer = null; this.onEnd = null; }

  get playing() { return !!this.timer; }

  // Play from `fromSec`; onTime(sec) is called while playing (for the playhead).
  async play(piece, onTime, fromSec = 0) {
    this.stop();
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    await this.ctx.resume();
    const ctx = this.ctx, out = chain(ctx), evs = events(piece), total = pieceSeconds(piece);
    this.out = out;
    const at = ctx.currentTime + 0.1 - fromSec;
    let k = evs.findIndex(e => e.t0 >= fromSec); if (k < 0) k = evs.length;
    const tick = () => {
      const now = ctx.currentTime - at;
      while (k < evs.length && evs[k].t0 < now + 0.4) voice(ctx, out, evs[k++], at);
      onTime && onTime(Math.min(now, total));
      if (now > total + 0.3) { this.stop(); this.onEnd && this.onEnd(); }
    };
    tick();
    this.timer = setInterval(tick, 60);
  }

  stop() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    if (this.out) { try { this.out.disconnect(); } catch (e) { /* already disconnected */ } this.out = null; }
  }
}

// Render the piece with the same synth into a 16-bit stereo WAV (Blob).
export async function renderWav(piece, rate = 44100) {
  const secs = pieceSeconds(piece) + 0.5;
  const ctx = new OfflineAudioContext(2, Math.ceil(secs * rate), rate);
  const out = chain(ctx);
  for (const e of events(piece)) voice(ctx, out, e, 0);
  const buf = await ctx.startRendering();
  return encodeWav(buf);
}

function encodeWav(buf) {
  const ch = buf.numberOfChannels, n = buf.length, data = new DataView(new ArrayBuffer(44 + n * ch * 2));
  const str = (o, s) => [...s].forEach((c, i) => data.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF'); data.setUint32(4, 36 + n * ch * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
  data.setUint32(16, 16, true); data.setUint16(20, 1, true); data.setUint16(22, ch, true);
  data.setUint32(24, buf.sampleRate, true); data.setUint32(28, buf.sampleRate * ch * 2, true);
  data.setUint16(32, ch * 2, true); data.setUint16(34, 16, true); str(36, 'data'); data.setUint32(40, n * ch * 2, true);
  const chans = [...Array(ch).keys()].map(c => buf.getChannelData(c));
  let o = 44;
  for (let i = 0; i < n; i++) for (let c = 0; c < ch; c++) {
    const v = Math.max(-1, Math.min(1, chans[c][i])); data.setInt16(o, v < 0 ? v * 0x8000 : v * 0x7fff, true); o += 2;
  }
  return new Blob([data], { type: 'audio/wav' });
}
