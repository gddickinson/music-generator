// Port of the motif layer of advanced_composer.py: Note/Motif records, MotifDevelopment (invert, retrograde,
// augment, diminish, fragment, phase_shift) and the per-voice texture handlers (create_*_voice).
// Lists below keep the Python enum order, because random.choice(list(Enum)) depends on it.

export const ARTICULATIONS = ['legato', 'staccato', 'marcato', 'tenuto', 'portato', 'spiccato', 'pizzicato'];
export const TEXTURES = ['monophonic', 'homophonic', 'polyphonic', 'heterophonic', 'contrapuntal'];
export const DEVELOPMENT_TYPES = ['augmentation', 'diminution', 'inversion', 'retrograde', 'transposition',
  'fragmentation', 'sequence', 'phase_shift'];

// Note(pitch, duration, velocity=80, articulation=LEGATO, performance=None)
export const note = (pitch, duration, velocity = 80, articulation = 'legato', performance = null) =>
  ({ pitch, duration, velocity, articulation, performance });

export const motif = (notes, emotion, history = []) => ({ notes, emotion, history });

const withNotes = (m, notes, step) => motif(notes, m.emotion, m.history.concat([step]));

export const Development = {
  invert(m) {
    if (!m.notes.length) return m;
    const c = m.notes[0].pitch;
    return withNotes(m, m.notes.map(n => note(c + (c - n.pitch), n.duration, n.velocity, n.articulation, n.performance)),
      'inversion');
  },
  retrograde(m) {
    if (!m.notes.length) return m;
    return withNotes(m, m.notes.slice().reverse(), 'retrograde');
  },
  augment(m, factor = 2.0) {
    if (!m.notes.length) return m;
    return withNotes(m, m.notes.map(n => note(n.pitch, n.duration * factor, n.velocity, n.articulation, n.performance)),
      'augmentation');
  },
  diminish(m, factor = 0.5) {
    if (!m.notes.length) return m;
    return withNotes(m, m.notes.map(n => note(n.pitch, n.duration * factor, n.velocity, n.articulation, n.performance)),
      'diminution');
  },
  fragment(m, start, length) {
    if (!m.notes.length) return m;
    if (start + length > m.notes.length) length = m.notes.length - start;
    return withNotes(m, pySlice(m.notes, start, start + length), 'fragmentation');
  },
  phase_shift(m, shift) {
    if (!m.notes.length) return m;
    const perf = () => ({ tempo_change: null, dynamic_change: null, articulation: null, expression: null, timing_offset: shift });
    return withNotes(m, m.notes.map(n => note(n.pitch, n.duration, n.velocity, n.articulation, perf())), 'phase_shift');
  },
};

// Python list slicing semantics for the (non-negative or negative) bounds used above.
function pySlice(arr, a, b) {
  const n = arr.length;
  const norm = i => (i < 0 ? Math.max(0, n + i) : Math.min(i, n));
  return arr.slice(norm(a), Math.max(norm(a), norm(b)));
}

const PERFECT = new Set([0, 4, 7]);

// The texture handlers. Each returns a motif for voice v or null (silent voice). py = PyRandom stream.
function homophonic(m, v) {
  if (v === 0) return m;
  if (v >= 4) return null;
  const iv = [-3, -7, -10][v - 1];
  return motif(m.notes.map(n => note(n.pitch + iv, n.duration, Math.max(40, n.velocity - 20), n.articulation, n.performance)),
    m.emotion, m.history);
}

function polyphonic(m, v) {
  if (v === 0) return m;
  if (v >= 4) return null;
  return motif(m.notes.map(n => {
    const p = n.pitch;
    const off = v % 2 === 0 ? (PERFECT.has(mod12(p)) ? 4 : 3) : (PERFECT.has(mod12(p)) ? -3 : -4);
    return note(p + off + v * 2, n.duration * (v === 2 ? 1.5 : 0.75), Math.max(40, n.velocity - 15), n.articulation,
      n.performance);
  }), m.emotion, m.history);
}

function heterophonic(m, v, py) {
  if (v === 0) return m;
  if (v >= 4) return null;
  const out = [];
  for (const n of m.notes) {
    if (py.random() < 0.3) {
      out.push(note(n.pitch + py.choice([-1, 1]), n.duration * 0.5, n.velocity - 10, n.articulation, n.performance));
      out.push(note(n.pitch, n.duration * 0.5, n.velocity - 10, n.articulation, n.performance));
    } else {
      out.push(note(n.pitch, n.duration, Math.max(40, n.velocity - 15), n.articulation, n.performance));
    }
  }
  return motif(out, m.emotion, m.history);
}

const CONSONANT = [3, 4, 7, 8, 9];
function contrapuntal(m, v, py) {
  if (v === 0) return m;
  if (v >= 4) return null;
  let cur = py.choice(CONSONANT);
  const out = m.notes.map((n, i) => {
    if (i > 0) {
      const prev = cur;
      const possible = CONSONANT.filter(x => Math.abs(x - prev) <= 4);
      if (possible.length) cur = py.choice(possible);
    }
    const off = i % 2 === 0 ? cur : -cur;
    return note(n.pitch + off + (v - 2) * 12, n.duration, Math.max(40, n.velocity - 12), n.articulation, n.performance);
  });
  return motif(out, m.emotion, m.history);
}

// Python's % on ints is floored (non-negative for a positive divisor).
const mod12 = p => ((p % 12) + 12) % 12;

const HANDLERS = {
  monophonic: (m, v) => (v === 0 ? m : null), homophonic, polyphonic, heterophonic, contrapuntal,
};

// transform_for_voice(motif, voice_idx, texture)
export function transformForVoice(m, v, texture, py) {
  if (!m) return null;
  const h = HANDLERS[texture];
  return h ? h(m, v, py) : null;
}
