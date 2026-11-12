// Port of CreativeComposer's form initialisers, part 1: sonata, minimalist, through-composed, aleatoric
// (advanced_composer.py). `c` is the Composer (composer_engine.js): c.py / c.np random streams, c.params, c.gen
// (the fractal generator's mode and current emotion) and c.createMotif(). Random draws happen in the same order as
// in the Python, which is what makes a seed reproduce the same piece.
import { generateMelody } from './fractal_engine.js';
import { Development as D, ARTICULATIONS, note, motif } from './motif_engine.js';

export const section = (motifs, key, tempo, texture, emotion, dynamics) =>
  ({ motifs, key, tempo, texture, emotion, dynamics });
export const form = (name, sections, transitions) => ({ name, sections, transitions });
export const directive = (o = {}) => ({ tempo_change: null, dynamic_change: null, articulation: null,
  expression: null, timing_offset: 0.0, ...o });

export const KEY_NUMBER = { C: 0, G: 7, D: 2, A: 9, E: 4, B: 11, 'F#': 6, F: 5, Bb: 10, Eb: 3, Ab: 8, Db: 1, Gb: 6 };
export const KEYS = Object.keys(KEY_NUMBER);
const NUMBER_KEY = Object.fromEntries(Object.entries(KEY_NUMBER).map(([k, v]) => [v, k])); // later keys win, as in Python
export const dominantKey = k => NUMBER_KEY[(KEY_NUMBER[k] + 7) % 12];
export const subdominantKey = k => NUMBER_KEY[(KEY_NUMBER[k] + 5) % 12];
export const relativeMinor = k => NUMBER_KEY[(KEY_NUMBER[k] + 9) % 12];
// musical_types.transposition: shortest move (-5..+6 semitones) between key centres
export const transposition = (from, to) => { const d = (((KEY_NUMBER[to] - KEY_NUMBER[from]) % 12) + 12) % 12; return d > 6 ? d - 12 : d; };
// Python's round() (halves to even) for the non-negative values used here
export const pyRound = x => { const f = Math.floor(x), r = x - f; return r > 0.5 || (r === 0.5 && f % 2 === 1) ? f + 1 : f; };

// melody * (8 // len(melody) + 1) when shorter than 8 notes
const repeatTo8 = a => (a.length < 8 ? Array.from({ length: Math.floor(8 / a.length) + 1 }, () => a).flat() : a);

function theme(c, emotion, duration, articulation) {
  c.gen.emotion = emotion;
  let [mel, , vel] = generateMelody(c.gen.mode, c.gen.emotion, c.np, c.gen.offset);
  mel = repeatTo8(mel); vel = repeatTo8(vel);
  return motif(mel.slice(0, 16).map((p, i) => note(p, duration, vel[i % vel.length], articulation)), emotion);
}

export function sonata(c) {
  const { py, params: P } = c, base = P.base_tempo, sections = [], transitions = [];
  const primary = theme(c, 'energetic', 0.25, 'marcato');
  const secondary = theme(c, 'peaceful', 0.3, 'legato');
  sections.push(section([primary], P.key, base, 'homophonic', 'energetic', 80));
  transitions.push(directive({ tempo_change: 1.1, dynamic_change: 1.1, expression: 'poco a poco crescendo' }));
  sections.push(section([secondary], dominantKey(P.key), Math.trunc(base * 0.95), 'homophonic', 'peaceful', 75));

  const development = [];
  for (const m of [primary, secondary]) {
    if (!m.notes.length) continue;
    let t = m;
    const available = ['augmentation'];
    if (t.notes.length > 1) available.push('inversion');
    if (t.notes.length >= 8) available.push('fragmentation');
    const steps = c.maxDevelopmentSteps();
    for (let k = py.randint(Math.max(1, steps - 1), steps); k > 0; k--) {
      const type = py.choice(available);
      if (type === 'fragmentation' && t.notes.length >= 8) {
        const start = py.randint(0, t.notes.length - 4);
        const length = py.randint(4, Math.min(8, t.notes.length - start));
        t = D.fragment(t, start, length);
      } else if (type === 'inversion') t = D.invert(t);
      else t = D.augment(t, 1.5);
      development.push(t);
    }
  }
  sections.push(section(development, relativeMinor(P.key), Math.trunc(base * 1.1), 'polyphonic', 'dark', 85));
  transitions.push(directive({ tempo_change: 0.9, dynamic_change: 1.2, expression: 'building tension' }));
  sections.push(section([primary], P.key, base, 'homophonic', 'energetic', 90));
  transitions.push(directive({ tempo_change: 1.0, dynamic_change: 0.9, expression: 'flowing' }));
  sections.push(section([secondary], P.key, Math.trunc(base * 0.95), 'homophonic', 'peaceful', 85));
  if (py.random() > 0.5) {
    sections.push(section([D.diminish(primary, 0.75), D.augment(secondary, 1.25)], P.key, Math.trunc(base * 1.05),
      'polyphonic', 'joyful', 95));
  }
  return form('sonata', sections, transitions);
}

export function minimalist(c) {
  const P = c.params, sections = [], transitions = [];
  const [mel] = generateMelody(c.gen.mode, c.gen.emotion, c.np, c.gen.offset);
  const base = motif(mel.slice(0, 8).map(p => note(p, 0.25)), P.emotional_journey[0]);
  const motifs = [base, ...[0.125, 0.25, 0.375].map(s => D.phase_shift(base, s))];
  P.emotional_journey.forEach((emotion, i) => {
    const active = motifs.slice(0, Math.min(i + 2, motifs.length));
    if (i < P.emotional_journey.length - 1) {
      transitions.push(directive({ tempo_change: 1.0 + i * 0.1, dynamic_change: 1.0 + i * 0.15 }));
    }
    sections.push(section(active, P.key, Math.trunc(P.base_tempo * (1 + i * 0.1)), 'polyphonic', emotion, 70 + i * 5));
  });
  return form('minimalist', sections, transitions);
}

export function throughComposed(c) {
  const { py, params: P } = c, sections = [], transitions = [];
  let previous = null;
  P.emotional_journey.forEach((emotion, i) => {
    let current;
    if (previous === null) {
      c.gen.emotion = emotion;
      current = [c.createMotif(emotion)];
    } else {
      current = previous.map(prev => {
        const type = py.choice(['augmentation', 'diminution', 'fragmentation']);
        if (type === 'augmentation') return D.augment(prev);
        if (type === 'diminution') return D.diminish(prev);
        const start = py.randint(0, prev.notes.length - 2);
        return D.fragment(prev, start, py.randint(2, 4));
      });
    }
    if (i < P.emotional_journey.length - 1) {
      transitions.push(directive({ tempo_change: 0.9 + i * 0.1, dynamic_change: 0.9 + i * 0.1 }));
    }
    const texture = py.choice(['homophonic', 'polyphonic', 'contrapuntal']);
    sections.push(section(current, P.key, P.base_tempo, texture, emotion, 75));
    previous = current;
  });
  return form('through_composed', sections, transitions);
}

export function aleatoric(c) {
  const { py, params: P } = c, sections = [], transitions = [], pool = [];
  for (const emotion of P.emotional_journey) {
    c.gen.emotion = emotion;
    const m = c.createMotif(emotion, py.choice([0.125, 0.25, 0.5]));
    pool.push(m, D.retrograde(m), D.augment(m), D.diminish(m));
  }
  P.emotional_journey.forEach((emotion, i) => {
    const motifs = py.sample(pool, Math.min(3, pool.length));
    if (i < P.emotional_journey.length - 1) {
      const tempo_change = py.uniform(0.8, 1.2), dynamic_change = py.uniform(0.7, 1.3);
      const articulation = py.choice(ARTICULATIONS), timing_offset = py.uniform(-0.1, 0.1);
      transitions.push(directive({ tempo_change, dynamic_change, articulation, timing_offset }));
    }
    const texture = py.choice(['polyphonic', 'heterophonic', 'contrapuntal']);
    const tempo = Math.trunc(P.base_tempo * py.uniform(0.8, 1.2));
    sections.push(section(motifs, P.key, tempo, texture, emotion, py.randint(60, 90)));
  });
  return form('aleatoric', sections, transitions);
}
