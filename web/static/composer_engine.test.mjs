// node web/static/composer_engine.test.mjs: the JS composer against the Python CreativeComposer.
// Fixtures (tests/fixtures/, written by scripts/make_fixtures.py from the Python code):
//   fractal_degrees.json - every emotion's scale-degree grid (Mandelbrot + savgol) -> must be identical
//   compositions.json    - 28 pieces (every form, several seeds and settings) -> identical notes, sections and
//                          MIDI file bytes.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { scaleDegrees, EMOTIONS } from './fractal_engine.js';
import { compose } from './composer_engine.js';

const load = f => JSON.parse(readFileSync(new URL('../../tests/fixtures/' + f, import.meta.url)));

const degrees = load('fractal_degrees.json');
for (const e of EMOTIONS) {
  const text = scaleDegrees(e).flat().map(v => '0123456789abcd'[v]).join('');
  assert.equal(text, degrees[e], `fractal degrees differ for ${e}`);
}
console.log(`fractal: ${EMOTIONS.length} emotions x 10000 scale degrees identical to the Python generator`);

let notes = 0, bytes = 0;
for (const fx of load('compositions.json')) {
  const label = JSON.stringify(fx.case);
  const r = compose(fx.params);
  assert.equal(r.form, fx.form, `form name ${label}`);
  assert.deepEqual(r.notes.slice(0, 40), fx.first_notes, `first notes ${label}`);
  assert.equal(r.notes.length, fx.n_notes, `note count ${label}`);
  const sha = createHash('sha256').update(JSON.stringify(r.notes)).digest('hex');
  assert.equal(sha, fx.notes_sha256, `notes ${label}`);
  assert.deepEqual(r.sections, fx.sections, `sections ${label}`);
  const midi = Buffer.from(fx.midi_base64, 'base64');
  assert.ok(Buffer.from(r.midi).equals(midi), `MIDI bytes ${label}`);
  notes += r.notes.length; bytes += midi.length;
}
console.log(`composer: ${load('compositions.json').length} pieces, ${notes} notes and ${bytes} MIDI bytes identical to Python`);
