// Compose page: the composer's parameters, the piece as a piano roll, playback, MIDI/WAV download and the Python
// snippet that reproduces the same piece. Composition runs in the browser (composer_engine.js).
import { route, api, app, esc, list, slider, download, onLeave, $ } from './core.js';
import { compose, DEFAULTS } from './composer_engine.js';
import { PianoRoll, VOICE_COLORS } from './pianoroll.js';
import { Player, renderWav, pieceSeconds } from './player.js';
import { tempoMap } from './timing.js';

const QS = { form: 'form', mode: 'mode', key: 'key', tempo: 'base_tempo', sections: 'section_count',
  voices: 'voice_count', seed: 'seed' };
const label = s => s.replace(/_/g, ' ');

function fromQuery(q, limits) {
  const p = structuredClone(DEFAULTS);
  for (const [k, name] of Object.entries(QS)) {
    if (q[k] == null) continue;
    p[name] = ['form', 'mode', 'key'].includes(name) ? q[k] : Number(q[k]);
    if (limits[name]) p[name] = Math.min(limits[name][1], Math.max(limits[name][0], Math.trunc(p[name]) || 0));
  }
  if (q.journey) p.emotional_journey = q.journey.split(',').filter(Boolean).slice(0, 8);
  return p;
}

function toQuery(p) {
  const q = new URLSearchParams();
  for (const [k, name] of Object.entries(QS)) q.set(k, p[name]);
  q.set('journey', p.emotional_journey.join(','));
  return q.toString();
}

export function pythonSnippet(p) {
  const E = e => `EmotionalStyle.${e.toUpperCase()}`;
  return `import random
import numpy as np
from advanced_composer import CreativeComposer, CreativeComposerParams, MusicalForm
from fractal_melody_generator import MusicalMode, EmotionalStyle

random.seed(${p.seed}); np.random.seed(${p.seed})   # both generators, before the composer is built
params = CreativeComposerParams(
    form=MusicalForm.${p.form.toUpperCase()}, base_tempo=${p.base_tempo}, key="${p.key}",
    mode=MusicalMode.${p.mode.toUpperCase()},
    emotional_journey=[${p.emotional_journey.map(E).join(', ')}],
    section_count=${p.section_count}, voice_count=${p.voice_count},
    visualization_enabled=False, debug_mode=False)
composer = CreativeComposer(params)
midi, _ = composer.generate_composition()
composer.save_composition(midi, None, "piece")   # piece.mid, byte-identical to the download here`;
}

route('compose', async (args, query) => {
  const info = await api('/info');
  const p = fromQuery(query, info.limits);
  const opts = (items, cur, fmt = label) => list(items, v => `<option value="${esc(v)}"${v === cur ? ' selected' : ''}>${esc(fmt(v))}</option>`);
  app().innerHTML = `
  <h1 class="glow">Compose</h1>
  <p class="lead">Choose a form, a mode and an emotional journey; the composer builds the piece from fractal melodies,
  motif development and voice textures. Same settings and seed, same notes as the Python program.</p>
  <section class="panel">
    <div class="ctlgrid">
      <div><label for="f-form">Form</label><select id="f-form">${opts(info.forms.map(f => f.id), p.form)}</select></div>
      <div><label for="f-mode">Mode</label><select id="f-mode">${opts(info.modes, p.mode)}</select></div>
      <div><label for="f-key">Key (section labels)</label><select id="f-key">${opts(info.keys, p.key, v => v)}</select></div>
      <div><label for="f-seed">Seed</label><div class="seedrow"><input id="f-seed" type="number" min="0" max="4294967295" value="${p.seed}">
        <button class="alt" id="dice" title="Random seed" aria-label="Random seed">Roll</button></div></div>
      <div id="sl"></div>
    </div>
    <p class="small mute" id="formdesc"></p>
    <label>Emotional journey (one emotion per section for most forms)</label>
    <div class="journey" id="journey"></div>
    <div class="actions">
      <button id="play">&#9654; Play</button>
      <button class="alt" id="dl-midi">Download MIDI</button>
      <button class="alt" id="dl-wav">Download WAV</button>
      <span class="stats" id="stats"></span>
    </div>
  </section>
  <div class="legend" id="legend"></div>
  <div class="rollwrap"><canvas id="roll" role="img" aria-label="Piano roll of the composed piece"></canvas></div>
  <div class="row" style="max-width:420px" id="zoomhost"></div>
  <h2>Sections</h2>
  <div class="scroll"><table id="sections"></table></div>
  <h2>Reproduce in Python</h2>
  <p class="small mute">The MIDI download is byte-for-byte the file this writes (checked for the cases on the
  <a href="#/verify">Verify</a> page). WAV is rendered in your browser with a simple synth; the Python composer writes MIDI only.</p>
  <pre class="snippet" id="snippet"></pre>`;

  const sl = $('#sl'), L = info.limits;
  sl.style.display = 'contents';
  const mk = (id, lab, name, min, max) => slider(sl, { id, label: lab, min, max, value: p[name] }, v => { p[name] = v; update(); });
  mk('tempo', 'Base tempo (BPM)', 'base_tempo', L.base_tempo[0], L.base_tempo[1]);
  mk('sections', 'Sections', 'section_count', L.section_count[0], L.section_count[1]);
  mk('voices', 'Voices', 'voice_count', L.voice_count[0], L.voice_count[1]);

  const roll = new PianoRoll($('#roll')), player = new Player();
  slider($('#zoomhost'), { id: 'zoom', label: 'Zoom', min: 1, max: 8, step: 0.5, value: 1, fmt: v => v + 'x' },
    v => { roll.zoom = v; roll.draw(); });
  const onResize = () => roll.draw();
  onLeave(() => { player.stop(); window.removeEventListener('resize', onResize); });
  let piece = null, timer = 0;

  const bind = (id, name, num) => $(id).addEventListener('change', e => { p[name] = num ? Number(e.target.value) : e.target.value; update(); });
  bind('#f-form', 'form'); bind('#f-mode', 'mode'); bind('#f-key', 'key'); bind('#f-seed', 'seed', true);
  $('#dice').addEventListener('click', () => { p.seed = Math.floor(Math.random() * 1e6); $('#f-seed').value = p.seed; update(); });

  function drawJourney() {
    $('#journey').innerHTML = p.emotional_journey.map((e, i) => `<span class="chip emo-${e}">${esc(e)}
      <button data-rm="${i}" aria-label="Remove ${esc(e)}"${p.emotional_journey.length < 2 ? ' disabled' : ''}>&times;</button></span>`).join('') +
      (p.emotional_journey.length < 8 ? `<span class="addemo"><label class="small" for="addemo" style="margin:0">add</label>
      <select id="addemo">${opts(info.emotions, '')}</select><button class="alt" id="addbtn">Add</button></span>` : '');
    $('#journey').querySelectorAll('[data-rm]').forEach(b => b.addEventListener('click', () => {
      p.emotional_journey.splice(Number(b.dataset.rm), 1); drawJourney(); update();
    }));
    const add = $('#addbtn');
    if (add) add.addEventListener('click', () => { p.emotional_journey.push($('#addemo').value); drawJourney(); update(); });
  }

  function update() {
    clearTimeout(timer);
    timer = setTimeout(run, 120);
  }

  function run() {
    const wasPlaying = player.playing;
    player.stop(); $('#play').innerHTML = '&#9654; Play';
    const t0 = performance.now();
    piece = compose(p);
    const ms = performance.now() - t0;
    history.replaceState(null, '', '#/compose?' + toQuery(p));
    $('#formdesc').textContent = info.forms.find(f => f.id === p.form).description;
    roll.set(piece);
    const secs = pieceSeconds(piece);
    $('#stats').innerHTML = `<span><b>${piece.notes.length}</b> notes</span><span><b>${piece.sections.length}</b> sections</span>
      <span><b>${Math.floor(secs / 60)}:${String(Math.round(secs % 60)).padStart(2, '0')}</b> long</span>
      <span><b>${piece.midi.length}</b> MIDI bytes</span><span>composed in ${ms.toFixed(0)} ms</span>`;
    const voices = [...new Set(piece.notes.map(n => n[0]))].sort((a, b) => a - b);
    $('#legend').innerHTML = voices.map(v => `<button class="alt" data-v="${v}" aria-pressed="${!roll.hidden.has(v)}"
      style="margin:0;padding:3px 9px;font-size:11px"><i style="background:${VOICE_COLORS[v % 8]}"></i>voice ${v + 1}</button>`).join('') +
      '<span>Bands: section emotion. Line: section tension.</span>';
    $('#legend').querySelectorAll('[data-v]').forEach(b => b.addEventListener('click', () => {
      const v = Number(b.dataset.v); roll.hidden.has(v) ? roll.hidden.delete(v) : roll.hidden.add(v);
      b.setAttribute('aria-pressed', String(!roll.hidden.has(v))); roll.draw();
    }));
    $('#sections').innerHTML = `<tr><th>#</th><th>Start (beat)</th><th>Emotion</th><th>Texture</th><th>Tempo</th><th>Key</th>
      <th>Motifs (notes, development)</th><th>Tension</th></tr>` + piece.sections.map((s, i) => `<tr><td>${i + 1}</td>
      <td>${s.start.toFixed(1)}</td><td class="emo-${s.emotion}">${s.emotion}</td><td>${s.texture}</td><td>${s.tempo}</td><td>${esc(s.key)}</td>
      <td>${s.motifs.map(m => `${m.notes}${m.history.length ? ' (' + m.history.join(', ') + ')' : ''}`).join('; ')}</td>
      <td>${s.tension.toFixed(2)}</td></tr>`).join('');
    $('#snippet').textContent = pythonSnippet(p);
    if (wasPlaying) togglePlay();
  }

  const map = () => tempoMap(piece.tempos);
  function togglePlay() {
    if (player.playing) { player.stop(); $('#play').innerHTML = '&#9654; Play'; roll.playTick = null; roll.draw(); return; }
    const m = map();
    $('#play').innerHTML = '&#9632; Stop';
    player.onEnd = () => { $('#play').innerHTML = '&#9654; Play'; roll.playTick = null; roll.draw(); };
    player.play(piece, sec => { roll.playTick = m.toTick(sec); roll.draw(); });
  }
  $('#play').addEventListener('click', togglePlay);
  const base = () => `${p.form}_${p.mode}_seed${p.seed}`;
  $('#dl-midi').addEventListener('click', () => download(base() + '.mid', new Blob([piece.midi], { type: 'audio/midi' })));
  $('#dl-wav').addEventListener('click', async e => {
    const b = e.currentTarget; b.disabled = true; b.textContent = 'Rendering...';
    try { download(base() + '.wav', await renderWav(piece)); } finally { b.disabled = false; b.textContent = 'Download WAV'; }
  });
  window.addEventListener('resize', onResize);
  drawJourney();
  run();
});
