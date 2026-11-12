// About page: what the composer does, what the web version covers, how to run things, credits (from the README).
import { route, app } from './core.js';

route('about', () => {
  app().innerHTML = `
  <h1 class="glow">About</h1>
  <p class="lead">The Creative Music Generator is a Python algorithmic composition system: fractal-based melody
  generation, musical form structures, motif development and several voice textures, written out as MIDI.</p>
  <h2>How a piece is made</h2>
  <ol>
    <li><b>Melody.</b> FractalMelodyGenerator reads a Mandelbrot image in an emotion-specific region, smooths each
      row and maps it to the chosen mode, keeping only consonant steps (see <a href="#/fractal">Fractal melodies</a>).</li>
    <li><b>Form.</b> CreativeComposer lays out sections for the chosen form (sonata, arch, moment, process, minimalist,
      aleatoric, through-composed or mixed) and builds motifs of 8 to 32 notes from the melodies.</li>
    <li><b>Development.</b> Motifs are inverted, reversed, augmented, diminished, fragmented or phase-shifted.</li>
    <li><b>Texture.</b> Each section's texture turns a motif into up to four voices: monophonic, homophonic (parallel
      voices below), polyphonic, heterophonic (ornamented copies) or contrapuntal (consonant counter-lines).</li>
    <li><b>Key and timing.</b> Melodies are built on the home key's tonic; a section in another key (the sonata's
      dominant, a modulation) is transposed there. All motifs of a section start together and the next section starts
      where its last note ends, all in beats.</li>
    <li><b>MIDI.</b> Every voice is a MIDI track and channel; tempo changes per section.</li>
  </ol>
  <h2>What the web version covers</h2>
  <p>The composer (<code>advanced_composer.py</code> with <code>fractal_melody_generator.py</code>) is ported to
  JavaScript, including CPython's <code>random</code> module, NumPy's legacy random generator and MIDIUtil's file
  writer, so the same settings and seed give the same notes and the same MIDI bytes as the Python. One table comes
  from the Python: 11 of the 60,000 fractal cells where SciPy's floating-point smoothing rounds just below a whole
  number (<code>fractal_quirks.js</code>). Playback and the WAV download use a small browser synth; the Python
  composer itself writes MIDI only. <code>main_composer.py</code> (genetic algorithms, harmony engine, WAV synthesis)
  and <code>multi_fractal_music.py</code> are not part of the web version.</p>
  <p>The three 0-to-1 settings: <b>development complexity</b> sets how many development steps a motif goes through
  (1 to 4), <b>texture density</b> the share of accompanying voices that play (0 = the melody alone), and
  <b>modulation frequency</b> the chance that a later section in the home key moves to the dominant, subdominant or
  relative minor. Rondo, theme and variations and fugue have no plan of their own yet and use the mixed form.</p>
  <h2>Run it</h2>
  <h3>Python composer</h3>
  <pre>pip install -r requirements.txt
python advanced_composer.py      # writes creative_composition.mid and two PNG figures</pre>
  <p class="small">To reproduce a piece from this site, use the snippet on the Compose page: it seeds
  <code>random</code> and <code>numpy.random</code> before building the composer (Python 3.12 or later, whose
  <code>sum()</code> the timing depends on).</p>
  <h3>Web app (local)</h3>
  <pre>pip install -r requirements-web.txt
python run_web.py                # http://127.0.0.1:8166</pre>
  <p class="small">The local app adds a live Python run on the Verify page. <code>python scripts/build_static.py</code>
  builds this static site.</p>
  <h2>Credits</h2>
  <p>Inspired by works of Steve Reich, Karlheinz Stockhausen and Gy&ouml;rgy Ligeti; built with NumPy, SciPy, MIDIUtil,
  Matplotlib and NetworkX. MIT licence. Built with AI assistance from Claude (Anthropic).</p>`;
});
