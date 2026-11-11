// Port of CreativeComposer's form initialisers, part 2: arch, moment, process, mixed and _develop_section
// (advanced_composer.py). See forms_core.js for the conventions.
import { EMOTIONS } from './fractal_engine.js';
import { Development as D, ARTICULATIONS, TEXTURES, DEVELOPMENT_TYPES } from './motif_engine.js';
import { section, form, directive, sonata, minimalist, throughComposed, aleatoric } from './forms_core.js';

export function arch(c) {
  const P = c.params, sections = [], transitions = [];
  const material = {};
  for (const e of ['peaceful', 'energetic', 'dark']) { c.gen.emotion = e; material[e] = [c.createMotif(e)]; }
  const plan = [
    [material.peaceful, 'homophonic', 'peaceful', 1.0], [material.energetic, 'polyphonic', 'energetic', 1.1],
    [material.dark, 'contrapuntal', 'dark', 1.2], [material.energetic, 'polyphonic', 'energetic', 1.1],
    [material.peaceful, 'homophonic', 'peaceful', 1.0],
  ];
  plan.forEach(([motifs, texture, emotion, tf], i) => {
    sections.push(section(motifs, P.key, Math.trunc(P.base_tempo * tf), texture, emotion,
      70 + (i < 3 ? i * 5 : (4 - i) * 5)));
    if (i < plan.length - 1) {
      transitions.push(directive({ tempo_change: tf, dynamic_change: 1.0 + 0.1 * (i < 3 ? i : 4 - i) }));
    }
  });
  return form('arch', sections, transitions);
}

export function moment(c) {
  const { py, params: P } = c, sections = [], transitions = [];
  for (let i = 0; i < P.section_count; i++) {
    const emotion = py.choice(EMOTIONS), texture = py.choice(TEXTURES), tf = py.uniform(0.8, 1.4);
    c.gen.emotion = emotion;
    const motifs = [c.createMotif(emotion)];
    if (motifs[0].notes.length >= 8) {
      const type = py.choice(DEVELOPMENT_TYPES);
      if (type === 'augmentation') motifs[0] = D.augment(motifs[0], py.uniform(1.5, 2.0));
      else if (type === 'diminution') motifs[0] = D.diminish(motifs[0], py.uniform(0.4, 0.7));
    }
    const key = py.choice(['C', 'G', 'D', 'A', 'E', 'F']);
    sections.push(section(motifs, key, Math.trunc(P.base_tempo * tf), texture, emotion, py.randint(60, 90)));
    if (i < P.section_count - 1) {
      const tempo_change = py.uniform(0.7, 1.4), dynamic_change = py.uniform(0.6, 1.5);
      transitions.push(directive({ tempo_change, dynamic_change, articulation: py.choice(ARTICULATIONS) }));
    }
  }
  return form('moment', sections, transitions);
}

export function processMusic(c) {
  const P = c.params, sections = [], transitions = [];
  c.gen.emotion = 'ethereal';
  const base = c.createMotif('ethereal');
  const phases = Math.min(8, P.section_count);
  for (let i = 0; i < phases; i++) {
    const motifs = [base, D.phase_shift(base, 0.125 * i)];
    if (i > 2) motifs.push(D.augment(base, 2.0));
    sections.push(section(motifs, P.key, P.base_tempo, 'polyphonic', 'ethereal', 70 + i * 3));
    if (i < phases - 1) {
      transitions.push(directive({ tempo_change: 1.0, dynamic_change: 1.05, expression: 'gradually building' }));
    }
  }
  return form('process', sections, transitions);
}

export function developSection(c, s) {
  const py = c.py, motifs = [];
  for (const m of s.motifs) {
    if (!m.notes.length) continue;
    let t = m;
    for (let k = py.randint(1, 3); k > 0; k--) {
      const available = ['augmentation', 'diminution'];
      if (t.notes.length > 1) available.push('retrograde');
      if (t.notes.length >= 8) available.push('fragmentation');
      const type = py.choice(available);
      if (type === 'augmentation') t = D.augment(t, py.uniform(1.5, 2.0));
      else if (type === 'diminution') t = D.diminish(t, py.uniform(0.4, 0.7));
      else if (type === 'retrograde') t = D.retrograde(t);
      else if (t.notes.length >= 8) {
        const start = py.randint(0, t.notes.length - 4);
        const length = py.randint(4, Math.min(8, t.notes.length - start));
        t = D.fragment(t, start, length);
      }
    }
    motifs.push(t);
  }
  if (py.random() < 0.4) motifs.push(c.createMotif(s.emotion));
  const tempo = Math.trunc(s.tempo * py.uniform(0.9, 1.1));
  return section(motifs, s.key, tempo, py.choice(TEXTURES), s.emotion, s.dynamics);
}

export function mixed(c) {
  const { py, params: P } = c, J = P.emotional_journey, sections = [], transitions = [];
  const styles = [minimalist, aleatoric, throughComposed, sonata];
  J.forEach((emotion, i) => {
    const style = py.choice(styles);
    const temp = style(c);
    if (!temp.sections.length) return;
    for (const s of temp.sections.slice(0, py.randint(1, 2))) {
      s.emotion = emotion;
      if (py.random() < 0.3) sections.push(developSection(c, s));
      sections.push(s);
      if (i < J.length - 1) {
        const tempo_change = py.uniform(0.8, 1.2), dynamic_change = py.uniform(0.9, 1.1);
        transitions.push(directive({ tempo_change, dynamic_change, articulation: py.choice(ARTICULATIONS),
          expression: `transitioning to ${J[i + 1]}` }));
      }
    }
  });
  while (sections.length < P.section_count) {
    sections.push(developSection(c, py.choice(sections)));
    const tempo_change = py.uniform(0.9, 1.1), dynamic_change = py.uniform(0.9, 1.1);
    transitions.push(directive({ tempo_change, dynamic_change }));
  }
  return form('mixed', sections.slice(0, P.section_count), transitions.slice(0, Math.max(0, P.section_count - 1)));
}

// initialize_form: rondo, theme_and_variations and fugue have no initialiser and fall back to mixed.
export const FORM_INITIALISERS = {
  sonata, minimalist, aleatoric, through_composed: throughComposed, mixed, arch, moment, process: processMusic,
};
