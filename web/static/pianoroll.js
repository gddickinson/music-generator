// Piano-roll view of a composed piece: one colour per voice, section bands coloured by emotion, the section tension
// curve on top, and a playhead.
import { TICKS } from './midi_writer.js';
import { endTick } from './timing.js';

export const VOICE_COLORS = ['#ffb347', '#5fd4ff', '#ff5fa2', '#9dff6b', '#c9a6ff', '#ffe066', '#4fe3c1', '#ff8a65'];
export const EMOTION_COLORS = { ethereal: '#b9a6ff', dark: '#ff6b6b', joyful: '#ffd84d', melancholic: '#6fa8ff',
  energetic: '#ff9f43', peaceful: '#5fe0b0' };

export class PianoRoll {
  constructor(canvas) { this.canvas = canvas; this.piece = null; this.zoom = 1; this.playTick = null; this.hidden = new Set(); }

  set(piece) { this.piece = piece; this.playTick = null; this.draw(); }

  layout() {
    const c = this.canvas, wrap = c.parentElement;
    const cssW = Math.max(300, Math.round(wrap.clientWidth * this.zoom)), cssH = Math.min(420, Math.max(260, wrap.clientWidth * 0.42));
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(cssW * dpr) || c.height !== Math.round(cssH * dpr)) {
      c.width = Math.round(cssW * dpr); c.height = Math.round(cssH * dpr);
      c.style.width = cssW + 'px'; c.style.height = cssH + 'px';
    }
    return { w: cssW, h: cssH, dpr };
  }

  draw() {
    const p = this.piece; if (!p) return;
    const { w, h, dpr } = this.layout(), ctx = this.canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#05060c'; ctx.fillRect(0, 0, w, h);
    const end = Math.max(endTick(p.notes), 1), top = 34, bottom = h - 18, left = 30;
    const pitches = p.notes.map(n => n[2]);
    const lo = Math.min(...pitches, 60) - 2, hi = Math.max(...pitches, 72) + 2;
    const X = t => left + (t / end) * (w - left - 6), Y = pitch => bottom - ((pitch - lo) / (hi - lo)) * (bottom - top);
    // section bands and tension curve
    const pts = [];
    p.sections.forEach((s, i) => {
      const a = X(s.start * TICKS), b = X((s.start + s.duration) * TICKS);
      ctx.fillStyle = EMOTION_COLORS[s.emotion] + (i % 2 ? '14' : '22');
      ctx.fillRect(a, top, Math.max(1, b - a), bottom - top);
      ctx.fillStyle = EMOTION_COLORS[s.emotion]; ctx.fillRect(a, top - 6, Math.max(1, b - a - 1), 4);
      pts.push([a, 26 - s.tension * 20]);
    });
    ctx.strokeStyle = 'rgba(255,255,255,.7)'; ctx.lineWidth = 1.5; ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
    ctx.fillStyle = '#94a3c6'; ctx.font = '10px ui-monospace, Menlo, monospace'; ctx.fillText('tension', 2, 12);
    // octave grid
    for (let pc = Math.ceil(lo / 12) * 12; pc <= hi; pc += 12) {
      ctx.strokeStyle = 'rgba(148,163,198,.18)'; ctx.beginPath(); ctx.moveTo(left, Y(pc)); ctx.lineTo(w, Y(pc)); ctx.stroke();
      ctx.fillStyle = '#94a3c6'; ctx.fillText('C' + (pc / 12 - 1), 2, Y(pc) + 3);
    }
    // notes
    const nh = Math.max(2, (bottom - top) / (hi - lo) - 1);
    for (const [track, , pitch, tick, dur, vel] of p.notes) {
      if (this.hidden.has(track)) continue;
      ctx.globalAlpha = 0.35 + 0.65 * vel / 127;
      ctx.fillStyle = VOICE_COLORS[track % VOICE_COLORS.length];
      ctx.fillRect(X(tick), Y(pitch) - nh / 2, Math.max(1.5, X(tick + dur) - X(tick)), nh);
    }
    ctx.globalAlpha = 1;
    // beat axis
    ctx.fillStyle = '#94a3c6';
    const beats = end / TICKS, step = niceStep(beats / 8);
    for (let b = 0; b <= beats; b += step) ctx.fillText(String(b), X(b * TICKS) + 2, h - 5);
    if (this.playTick != null) {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath();
      ctx.moveTo(X(this.playTick), top - 8); ctx.lineTo(X(this.playTick), bottom); ctx.stroke();
    }
  }
}

function niceStep(x) {
  const p = 10 ** Math.floor(Math.log10(Math.max(x, 1e-9)));
  for (const m of [1, 2, 5, 10]) if (m * p >= x) return m * p;
  return 10 * p;
}
