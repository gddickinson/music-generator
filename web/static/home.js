// Home page: hero with a piano roll of the default piece, counts from the API, entry cards.
import { route, api, app, list, esc } from './core.js';
import { compose, DEFAULTS } from './composer_engine.js';
import { PianoRoll } from './pianoroll.js';

const DOORS = [
  ['#/compose', 'Compose', 'Pick a form, mode, voices and an emotional journey; hear the piece, see it as a piano roll, download the MIDI file.'],
  ['#/fractal', 'Fractal melodies', 'Where the notes come from: Mandelbrot escape times, smoothed and read off as scale degrees, per emotion.'],
  ['#/verify', 'Verify', 'The browser engine against the Python program: the same seed gives the same notes and the same MIDI bytes.'],
  ['#/about', 'About', 'How the composer works, what the web version covers, how to run the Python and the web app, credits.'],
];

route('home', async () => {
  const info = await api('/info');
  const own = info.forms.filter(f => !/falls back/.test(f.description)).length;
  app().innerHTML = `
  <section class="hero two herosplit">
    <div>
      <p class="kicker">Fractals · musical forms · voice textures</p>
      <h1 class="glow">Creative Music Generator</h1>
      <p class="lead">${esc(info.summary)} This site runs the composer in your browser: a JavaScript port of the
      Python code that writes the same notes, note for note, for the same settings and seed.</p>
      <a class="btn" href="#/compose">Compose a piece</a><a class="btn alt" href="#/about">How it works</a>
    </div>
    <figure class="fig"><div class="rollwrap"><canvas id="hero" role="img" aria-label="Piano roll of the default sonata, seed 0"></canvas></div>
      <figcaption class="small mute">The default settings (sonata form, Ionian, four voices, seed 0) as a piano roll:
      colours are voices, bands are sections coloured by emotion.</figcaption></figure>
  </section>
  <div class="tiles">
    <div class="tile"><b>${own}</b><span>musical forms with their own plan (of ${info.forms.length})</span></div>
    <div class="tile"><b>${info.modes.length}</b><span>modes</span></div>
    <div class="tile"><b>${info.emotions.length}</b><span>emotional styles</span></div>
    <div class="tile"><b>${info.textures.length}</b><span>voice textures</span></div>
    <div class="tile"><b>${info.limits.voice_count[1]}</b><span>voices at most</span></div>
    <div class="tile"><b>${info.verify_cases}</b><span>pieces re-checked against Python</span></div>
  </div>
  <div class="grid wide">${list(DOORS, ([href, t, d]) => `<a class="card" href="${href}"><b>${t}</b><span>${d}</span></a>`)}</div>
  <h2>Forms</h2>
  <div class="grid">${list(info.forms, f => `<a class="card" href="#/compose?form=${f.id}"><b>${esc(f.name)}</b><span>${esc(f.description)}</span></a>`)}</div>`;
  const roll = new PianoRoll(document.getElementById('hero'));
  roll.set(compose(DEFAULTS));
});
