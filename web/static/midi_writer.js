// Port of the parts of MIDIUtil 1.2.1 (MIDIFile) that CreativeComposer uses: addNote, addTempo and writeFile with
// the defaults (format 1 with a tempo track, 960 ticks per quarter note, removeDuplicates and deinterleave on,
// adjust_origin off). Byte-for-byte the same file as MIDIUtil (checked by composer_engine.test.mjs).

export const TICKS = 960;
const SORT = { NoteOff: 2, NoteOn: 3, Tempo: 3 };

export class MidiFile {
  constructor(numTracks) {
    this.numTracks = numTracks + 1; // tracks[0] is the tempo track
    this.tracks = Array.from({ length: this.numTracks }, () => []);
    this.counter = 0;
  }

  static ticks(t) { return Math.trunc(t * TICKS); }

  addNote(track, channel, pitch, time, duration, volume) {
    const tick = MidiFile.ticks(time), dur = MidiFile.ticks(duration), ins = this.counter++;
    const list = this.tracks[track + 1];
    list.push({ type: 'NoteOn', channel, pitch, tick, duration: dur, volume, ins });
    list.push({ type: 'NoteOff', channel, pitch, tick: tick + dur, volume, ins });
  }

  addTempo(track, time, bpm) {
    this.tracks[0].push({ type: 'Tempo', tick: MidiFile.ticks(time), tempo: Math.trunc(60000000 / bpm), bpm,
      ins: this.counter++ });
  }

  // Notes as added (before MIDIUtil's clean-up): [track, channel, pitch, tick, duration ticks, velocity].
  notes() {
    const out = [];
    this.tracks.forEach((list, t) => {
      for (const e of list) if (e.type === 'NoteOn') out.push([t - 1, e.channel, e.pitch, e.tick, e.duration, e.volume]);
    });
    return out;
  }

  tempos() { return this.tracks[0].map(e => [e.tick, e.bpm]); }

  writeFile() {
    const bytes = [...ascii('MThd'), ...u32(6), ...u16(1), ...u16(this.numTracks), ...u16(TICKS)];
    for (const list of this.tracks) {
      const data = [];
      let prev = 0;
      for (const e of processTrack(list)) {
        data.push(...varLength(e.tick - prev));
        prev = e.tick;
        if (e.type === 'Tempo') data.push(0xff, 0x51, 0x03, (e.tempo >>> 16) & 0xff, (e.tempo >>> 8) & 0xff, e.tempo & 0xff);
        else data.push((e.type === 'NoteOn' ? 0x90 : 0x80) | e.channel, e.pitch, e.volume);
      }
      data.push(0x00, 0xff, 0x2f, 0x00);
      bytes.push(...ascii('MTrk'), ...u32(data.length));
      for (const b of data) bytes.push(b);
    }
    return new Uint8Array(bytes);
  }
}

const sortKey = (a, b) => (a.tick - b.tick) || (SORT[a.type] - SORT[b.type]) || (a.ins - b.ins);

// MIDITrack.closeTrack: removeDuplicates (set semantics: the first of equal events survives), sort, deInterleaveNotes.
function processTrack(list) {
  const seen = new Set(), events = [];
  for (const e of list) {
    const key = e.type === 'Tempo' ? `T${e.tick}:${e.tempo}` : `${e.type}${e.tick}:${e.pitch}:${e.channel}`;
    if (seen.has(key)) continue;
    seen.add(key);
    events.push({ ...e });
  }
  events.sort(sortKey);
  // deInterleaveNotes; the stack key is str(pitch) + str(channel), exactly as MIDIUtil builds it
  const stack = new Map();
  for (const e of events) {
    if (e.type !== 'NoteOn' && e.type !== 'NoteOff') continue;
    const key = String(e.pitch) + String(e.channel);
    if (e.type === 'NoteOn') {
      if (!stack.has(key)) stack.set(key, []);
      stack.get(key).push(e.tick);
    } else {
      const s = stack.get(key);
      if (!s) throw new Error(`MIDIUtil KeyError: '${key}'`);
      if (!s.length) throw new Error('MIDIUtil IndexError: pop from empty list');
      if (s.length > 1) e.tick = s.pop(); else s.pop();
    }
  }
  events.sort(sortKey);
  return events;
}

function varLength(i) {
  if (i === 0) return [0];
  const out = [];
  let hibit = 0;
  while (i > 0) { out.push((i & 0x7f) | hibit); i = Math.floor(i / 128); hibit = 0x80; }
  return out.reverse();
}

const ascii = s => [...s].map(ch => ch.charCodeAt(0));
const u16 = n => [(n >>> 8) & 0xff, n & 0xff];
const u32 = n => [(n >>> 24) & 0xff, (n >>> 16) & 0xff, (n >>> 8) & 0xff, n & 0xff];
