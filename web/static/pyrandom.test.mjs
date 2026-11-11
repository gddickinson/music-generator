// node web/static/pyrandom.test.mjs: the JS random streams against draws made by CPython and NumPy
// (tests/fixtures/random_streams.json, written by scripts/make_fixtures.py).
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { PyRandom, NumpyRandom } from './pyrandom.js';

const fx = JSON.parse(readFileSync(new URL('../../tests/fixtures/random_streams.json', import.meta.url)));
const range = n => [...Array(n).keys()];
let checks = 0;
for (const [seed, want] of Object.entries(fx)) {
  const r = new PyRandom(Number(seed));
  const got = {
    random: range(5).map(() => r.random()), randint: range(20).map(() => r.randint(0, 9)),
    choice: range(20).map(() => r.choice(range(7))), uniform: range(5).map(() => r.uniform(0.8, 1.2)),
    sample24: r.sample(range(24), 3), sample12: r.sample(range(12), 3),
  };
  for (const k of Object.keys(got)) { assert.deepEqual(got[k], want[k], `random.${k} seed ${seed}`); checks++; }
  if (want.np_randint) {
    const n = new NumpyRandom(Number(seed));
    const g = range(30).map(() => n.randint(40, 60)).concat(range(10).map(() => n.randint(0, 1000)));
    assert.deepEqual(g, want.np_randint, `np.random.randint seed ${seed}`); checks++;
  }
}
console.log(`pyrandom: ${checks} streams identical to CPython / NumPy`);
