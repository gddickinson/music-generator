// Fractal melodies page: the Mandelbrot region each emotion uses, the sampled columns, and the melody read from it
// (FractalMelodyGenerator.generate_melody, ported in fractal_engine.js).
import { route, api, app, esc, list, $ } from './core.js';
import { mandelbrot, scaleDegrees, melodyPitches, currentScale, RESOLUTION, MAX_ITER } from './fractal_engine.js';
import { EMOTION_COLORS } from './pianoroll.js';

route('fractal', async (args, query) => {
  const info = await api('/info');
  let emotion = info.emotions.includes(query.emotion) ? query.emotion : 'peaceful', mode = 'ionian';
  app().innerHTML = `
  <h1 class="glow">Fractal melodies</h1>
  <p class="lead">Every melody starts as a ${RESOLUTION} x ${RESOLUTION} Mandelbrot image. Each emotion looks at its own
  region (zoom and offset), and each row of escape times becomes a phrase.</p>
  <div class="row" style="max-width:520px">
    <div><label for="emo">Emotion</label><select id="emo">${list(info.emotions, e => `<option${e === emotion ? ' selected' : ''}>${e}</option>`)}</select></div>
    <div><label for="mode">Mode</label><select id="mode">${list(info.modes, m => `<option${m === mode ? ' selected' : ''}>${m}</option>`)}</select></div>
  </div>
  <div class="two">
    <figure class="fig"><canvas id="mb" width="400" height="400" role="img" aria-label="Mandelbrot escape times for the selected emotion"></canvas>
      <figcaption class="small mute">Escape iteration (bright = escapes late, black = never within ${MAX_ITER}).
      Dots: the columns read (every 4th), coloured by scale degree for the first rows.</figcaption></figure>
    <div>
      <h3>From image to melody</h3>
      <ol class="small">
        <li>Escape-time row (200 values) smoothed with a Savitzky-Golay filter (window 15, cubic).</li>
        <li>Every 4th value, modulo 14, picks a degree of the mode over two octaves from middle C.</li>
        <li>A note is kept only if it is a consonant step from the previous one (unison, 3rds, 4th, 5th, 6ths) and no
          wider than an octave.</li>
        <li>Each kept note gets a velocity drawn from the emotion's range (NumPy's random stream).</li>
      </ol>
      <table id="eparams"></table>
      <p class="small" id="mstat"></p>
    </div>
  </div>
  <h2>First 64 notes</h2>
  <canvas id="mel" width="1000" height="180" style="width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#05060c" role="img" aria-label="The first 64 pitches of the melody"></canvas>`;

  function draw() {
    const div = mandelbrot(emotion), c = $('#mb'), ctx = c.getContext('2d');
    const img = ctx.createImageData(RESOLUTION, RESOLUTION), col = EMOTION_COLORS[emotion];
    const [r, g, b] = [1, 3, 5].map(i => parseInt(col.slice(i, i + 2), 16));
    for (let i = 0; i < div.length; i++) {
      const v = div[i] === MAX_ITER ? 0 : Math.sqrt(div[i] / MAX_ITER);
      img.data.set([r * v, g * v, b * v, 255], i * 4);
    }
    const off = document.createElement('canvas'); off.width = off.height = RESOLUTION;
    off.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(off, 0, 0, c.width, c.height);
    const deg = scaleDegrees(emotion), s = c.width / RESOLUTION;
    for (let row = 0; row < RESOLUTION; row += 10) deg[row].forEach((d, j) => {
      ctx.fillStyle = `hsl(${d * 25},90%,65%)`; ctx.fillRect(j * 4 * s, row * s, 2, 2);
    });
    const p = info.emotion_params[emotion];
    $('#eparams').innerHTML = `<tr><th>zoom</th><td>${p.fractal_params.zoom}</td><th>offset</th><td>${p.fractal_params.offset.join(', ')}</td></tr>
      <tr><th>velocity</th><td>${p.velocity_range[0]} to ${p.velocity_range[1] - 1}</td><th>duration factor</th><td>${p.duration_factor}</td></tr>`;
    const mel = melodyPitches(mode, emotion), scale = currentScale(mode);
    $('#mstat').innerHTML = `<b>${mel.length}</b> notes kept of ${RESOLUTION * RESOLUTION / 4} samples. Scale: ${esc(scale.join(' '))}.
      The composer uses the first 8 to 32 of them per motif.`;
    const m = $('#mel'), mc = m.getContext('2d'), first = mel.slice(0, 64);
    mc.fillStyle = '#05060c'; mc.fillRect(0, 0, m.width, m.height);
    const lo = scale[0] - 1, hi = scale[scale.length - 1] + 1, w = m.width / 64;
    first.forEach((p2, i) => {
      mc.fillStyle = col; mc.fillRect(i * w + 1, m.height - 8 - ((p2 - lo) / (hi - lo)) * (m.height - 16), w - 2, 7);
    });
  }
  $('#emo').addEventListener('change', e => { emotion = e.target.value; draw(); });
  $('#mode').addEventListener('change', e => { mode = e.target.value; draw(); });
  draw();
});
