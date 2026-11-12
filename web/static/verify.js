// Verify page: pieces composed by the Python program (exported at build time, or run live on the local app) against
// the browser engine, compared as SHA-256 digests of the note list and of the MIDI file.
import { route, api, app, esc, isStatic, needsServer, $ } from './core.js';
import { compose } from './composer_engine.js';

async function sha256(bytes) {
  const d = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(d)].map(b => b.toString(16).padStart(2, '0')).join('');
}
const enc = s => new TextEncoder().encode(s);
const desc = p => `${p.form}, ${p.mode}, seed ${p.seed}, ${p.voice_count} voices, ${p.emotional_journey.length} emotions`;

async function check(params, py) {
  const js = compose(params);
  const notes = await sha256(enc(JSON.stringify(js.notes)));
  const midi = await sha256(js.midi);
  return { js, notesOk: notes === py.notes_sha256, midiOk: midi === py.midi_sha256 };
}

route('verify', async () => {
  const cases = await api('/verify');
  app().innerHTML = `
  <h1 class="glow">Verify</h1>
  <p class="lead">The Python program composed these pieces (when this site was built). Your browser now composes them
  again with the JavaScript engine and compares the complete note lists and the MIDI files, byte for byte.</p>
  <p id="sum" class="mute">Composing...</p>
  <div class="scroll"><table id="tbl"><tr><th>Settings</th><th>Notes (Python)</th><th>Notes match</th><th>MIDI bytes</th><th>MIDI match</th></tr></table></div>
  <h2>Run any settings in Python</h2>
  <div id="live"></div>
  <h2>What is compared</h2>
  <p class="small">Every note as the composer adds it: voice (track), channel, pitch, start tick, length in ticks and
  velocity, plus the whole MIDI file MIDIUtil writes (duplicate removal and overlap handling included). The node test
  <code>web/static/composer_engine.test.mjs</code> checks 28 more pieces and all 60,000 fractal scale degrees the same way.</p>`;
  let ok = 0;
  for (const c of cases) {
    const r = await check(c.params, c);
    ok += r.notesOk && r.midiOk;
    $('#tbl').insertAdjacentHTML('beforeend', `<tr><td>${esc(desc(c.params))}</td><td>${c.n_notes}</td>
      <td class="${r.notesOk ? 'ok' : 'err'}">${r.notesOk ? 'identical' : 'differs'}</td><td>${c.midi_bytes}</td>
      <td class="${r.midiOk ? 'ok' : 'err'}">${r.midiOk ? 'identical' : 'differs'}</td></tr>`);
  }
  $('#sum').innerHTML = `<b class="${ok === cases.length ? 'ok' : 'err'}">${ok} of ${cases.length}</b> pieces identical in notes and MIDI bytes.`;

  if (isStatic()) { $('#live').innerHTML = needsServer('Running the Python composer on new settings'); return; }
  $('#live').innerHTML = `<p class="small mute">Paste settings (the Compose page's link works too) and the local
    server runs the Python composer on them.</p>
    <label for="lq">Compose link or query</label><input id="lq" value="form=moment&seed=42&mode=dorian&voices=4&sections=6">
    <button id="lrun">Run in Python and compare</button><p id="lout" class="small"></p>`;
  $('#lrun').addEventListener('click', async () => {
    const q = Object.fromEntries(new URLSearchParams($('#lq').value.split('?').pop()));
    const map = { form: 'form', mode: 'mode', key: 'key', tempo: 'base_tempo', sections: 'section_count', voices: 'voice_count', seed: 'seed',
      complexity: 'development_complexity', density: 'texture_density', modulation: 'modulation_frequency' };
    const params = {};
    for (const [k, v] of Object.entries(q)) if (map[k]) params[map[k]] = ['form', 'mode', 'key'].includes(map[k]) ? v : Number(v);
    if (q.journey) params.emotional_journey = q.journey.split(',');
    $('#lout').textContent = 'Running the Python composer...';
    try {
      const py = await api('/compose', { body: params });
      const midi = Uint8Array.from(atob(py.midi_base64), ch => ch.charCodeAt(0));
      const ref = { notes_sha256: await sha256(enc(JSON.stringify(py.notes))), midi_sha256: await sha256(midi) };
      const r = await check(py.params, ref);
      $('#lout').innerHTML = `${esc(desc(py.params))}: Python ${py.notes.length} notes, JS ${r.js.notes.length};
        notes <b class="${r.notesOk ? 'ok' : 'err'}">${r.notesOk ? 'identical' : 'differ'}</b>, MIDI
        <b class="${r.midiOk ? 'ok' : 'err'}">${r.midiOk ? 'identical' : 'differs'}</b> (${midi.length} bytes).`;
    } catch (e) { $('#lout').innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  });
});
