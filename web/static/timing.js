// Tick <-> seconds through the piece's tempo map (one tempo event per section, as in the MIDI file).
import { TICKS } from './midi_writer.js';

export function tempoMap(tempos) {
  // events in insertion order; at equal ticks the later event wins (it comes later in the MIDI stream)
  const ev = tempos.map(([tick, bpm], i) => ({ tick, bpm, i })).sort((a, b) => a.tick - b.tick || a.i - b.i);
  const seg = [];
  for (const e of ev) {
    if (seg.length && seg[seg.length - 1].tick === e.tick) seg[seg.length - 1].bpm = e.bpm;
    else seg.push({ tick: e.tick, bpm: e.bpm });
  }
  if (!seg.length || seg[0].tick > 0) seg.unshift({ tick: 0, bpm: 120 }); // MIDI default tempo
  let sec = 0;
  for (let k = 0; k < seg.length; k++) {
    seg[k].sec = sec;
    if (k + 1 < seg.length) sec += (seg[k + 1].tick - seg[k].tick) / TICKS * 60 / seg[k].bpm;
  }
  const find = (key, v) => { let k = 0; while (k + 1 < seg.length && seg[k + 1][key] <= v) k++; return seg[k]; };
  return {
    toSec: tick => { const s = find('tick', tick); return s.sec + (tick - s.tick) / TICKS * 60 / s.bpm; },
    toTick: sec => { const s = find('sec', sec); return s.tick + (sec - s.sec) * s.bpm / 60 * TICKS; },
  };
}

// End tick of the piece (last note-off).
export const endTick = notes => notes.reduce((m, n) => Math.max(m, n[3] + n[4]), 0);
