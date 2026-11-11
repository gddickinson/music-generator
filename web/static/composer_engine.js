// Port of CreativeComposer (advanced_composer.py): set-up, _create_motif_from_melody, the section duration and
// tension measures, generate_composition and add_motif_to_midi. compose(params) is the engine's entry point:
// with Python's random and NumPy's legacy generator both seeded with `seed`, it gives the same notes and the
// same MIDI bytes as web/compose.py running the Python code.
import { PyRandom, NumpyRandom } from './pyrandom.js';
import { generateMelody, MODES, EMOTIONS } from './fractal_engine.js';
import { ARTICULATIONS, TEXTURES, motif, note, transformForVoice } from './motif_engine.js';
import { FORM_INITIALISERS, mixed } from './forms_extra.js';
import { KEYS } from './forms_core.js';
import { MidiFile } from './midi_writer.js';

// MusicalForm in enum order; rondo, theme_and_variations and fugue fall back to the mixed form in the Python.
export const FORMS = ['sonata', 'rondo', 'theme_and_variations', 'fugue', 'minimalist', 'through_composed',
  'aleatoric', 'mixed', 'arch', 'moment', 'process'];
export { MODES, EMOTIONS, KEYS, TEXTURES, ARTICULATIONS };

// CreativeComposerParams defaults (the fields the composer reads).
export const DEFAULTS = {
  form: 'sonata', base_tempo: 120, key: 'C', mode: 'ionian',
  emotional_journey: ['peaceful', 'energetic', 'melancholic', 'joyful'], section_count: 4, voice_count: 4, seed: 0,
};

const EMOTION_TENSION = { peaceful: 0.2, ethereal: 0.3, joyful: 0.5, energetic: 0.7, melancholic: 0.6, dark: 0.8 };
const TEXTURE_TENSION = { monophonic: 0.3, homophonic: 0.4, heterophonic: 0.6, polyphonic: 0.7, contrapuntal: 0.8 };
// Python's built-in sum() of floats (3.12+): Neumaier-compensated summation.
export function pySum(xs) {
  let f = 0.0, c = 0.0;
  for (const x of xs) {
    const t = f + x;
    if (Math.abs(f) >= Math.abs(x)) c += (f - t) + x; else c += (x - t) + f;
    f = t;
  }
  return c && Number.isFinite(c) ? f + c : f;
}
const sum = pySum;

export class Composer {
  constructor(params) {
    this.params = { ...DEFAULTS, ...params };
    this.py = new PyRandom(this.params.seed);
    this.np = new NumpyRandom(this.params.seed);
    this.gen = { mode: this.params.mode, emotion: this.params.emotional_journey[0] };
    this.form = (FORM_INITIALISERS[this.params.form] || mixed)(this);
  }

  // _create_motif_from_melody(emotion, duration): up to 32 fractal-melody notes with varied lengths and articulation.
  createMotif(emotion, duration = 0.25) {
    let [mel, , vel] = generateMelody(this.gen.mode, this.gen.emotion, this.np);
    if (mel.length < 8) {
      const rep = (a, n) => Array.from({ length: n }, () => a).flat();
      const k = Math.floor(8 / mel.length) + 1, kv = Math.floor(8 / vel.length) + 1;
      mel = rep(mel, k); vel = rep(vel, kv);
    }
    const notes = mel.slice(0, 32).map((p, i) => {
      const d = duration * this.py.choice([1.0, 1.5, 2.0]);
      return note(p, d, vel[i % vel.length], this.py.choice(ARTICULATIONS));
    });
    return motif(notes, emotion);
  }

  // calculate_section_duration (the Python divides beats by tempo/60, so this is in "seconds" per its comment)
  sectionDuration(s) {
    let total = 0.0;
    const tf = s.tempo / 60.0;
    for (const m of s.motifs) {
      let d = sum(m.notes.map(n => n.duration));
      for (const h of m.history) { if (h === 'augmentation') d *= 2.0; else if (h === 'diminution') d *= 0.5; }
      total += d;
    }
    return total > 0 ? total / tf : 4.0;
  }

  // calculate_section_tension (no harmony progression is ever attached, so that term is 0)
  sectionTension(s) {
    const base = EMOTION_TENSION[s.emotion] ?? 0.5, tex = TEXTURE_TENSION[s.texture] ?? 0.5;
    let rhythmic = 0.0;
    for (const m of s.motifs) {
      const d = sum(m.notes.map(n => n.duration));
      rhythmic += 1.0 - Math.min(1.0, d / m.notes.length);
    }
    rhythmic /= s.motifs.length || 1;
    const dynamic = (s.dynamics - 40) / 80.0;
    const t = base * 0.3 + tex * 0.2 + 0.0 * 0.2 + rhythmic * 0.2 + dynamic * 0.1;
    return Math.min(1.0, Math.max(0.0, t));
  }

  // generate_composition: returns { midi, sections } where sections carry start (beats), duration and tension.
  generate() {
    const P = this.params, midi = new MidiFile(P.voice_count), sections = [];
    this.starts = new Set();
    let t = 0.0;
    for (const s of this.form.sections) {
      midi.addTempo(0, t, s.tempo);
      const duration = this.sectionDuration(s), tension = this.sectionTension(s);
      sections.push({ start: t, duration, tension, tempo: s.tempo, key: s.key, texture: s.texture, emotion: s.emotion,
        dynamics: s.dynamics, motifs: s.motifs.map(m => ({ notes: m.notes.length, history: m.history })) });
      for (const m of s.motifs) {
        for (let v = 0; v < P.voice_count; v++) {
          const tm = transformForVoice(m, v, s.texture, this.py);
          if (tm) this.addMotif(midi, tm, v, t);
        }
        t += duration;
      }
    }
    return { midi, sections };
  }

  // add_motif_to_midi (track = channel = voice; a second note with the same pitch and start tick is skipped)
  addMotif(midi, m, track, start) {
    let cur = start;
    const add = (pitch, time, dur, vel) => {
      const key = `${track}:${pitch}:${MidiFile.ticks(time)}`;
      if (this.starts.has(key)) return;
      this.starts.add(key);
      midi.addNote(track, track, pitch, time, dur, vel);
    };
    for (const n of m.notes) {
      const pitch = Math.max(0, Math.min(127, Math.trunc(n.pitch)));
      if (n.performance) {
        let d = n.duration, vel = n.velocity;
        const time = cur + n.performance.timing_offset;
        if (n.performance.tempo_change) d *= n.performance.tempo_change;
        if (n.performance.dynamic_change) vel = Math.trunc(vel * n.performance.dynamic_change);
        if (n.articulation === 'staccato') d *= 0.5; else if (n.articulation === 'legato') d *= 1.1;
        add(pitch, time, d, Math.max(0, Math.min(127, Math.trunc(vel))));
        cur += d;
      } else {
        add(pitch, cur, n.duration, Math.max(0, Math.min(127, Math.trunc(n.velocity))));
        cur += n.duration;
      }
    }
  }
}

// One call: params -> { notes, tempos, sections, midi (Uint8Array), form }.
export function compose(params) {
  const c = new Composer(params);
  const { midi, sections } = c.generate();
  return { params: c.params, form: c.form.name, sections, notes: midi.notes(), tempos: midi.tempos(),
    midi: midi.writeFile() };
}
