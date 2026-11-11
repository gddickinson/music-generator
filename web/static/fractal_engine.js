// Port of fractal_melody_generator.py: modes, emotional parameters, the Mandelbrot escape-time grid and
// FractalMelodyGenerator.generate_melody (Mandelbrot rows -> Savitzky-Golay smoothing -> scale degrees -> melodic
// filter -> velocities from NumPy's random stream).
//
// Smoothing: scipy.signal.savgol_filter(row, 15, 3) is a fixed linear filter, so on the integer escape times it is
// computed here exactly in integer arithmetic. SciPy works in floating point, and where the exact value is a whole
// number its result can land one ulp below it, which changes the scale degree. Those few cells are listed in
// fractal_quirks.js (exported from the Python code by scripts/make_fixtures.py), so the degrees match SciPy exactly.
import { QUIRKS } from './fractal_quirks.js';

export const MODES = {
  ionian: [0, 2, 4, 5, 7, 9, 11], dorian: [0, 2, 3, 5, 7, 9, 10], phrygian: [0, 1, 3, 5, 7, 8, 10],
  lydian: [0, 2, 4, 6, 7, 9, 11], mixolydian: [0, 2, 4, 5, 7, 9, 10], aeolian: [0, 2, 3, 5, 7, 8, 10],
  locrian: [0, 1, 3, 5, 6, 8, 10],
};

// Enum order of EmotionalStyle (random.choice(list(EmotionalStyle)) depends on it).
export const EMOTIONS = ['ethereal', 'dark', 'joyful', 'melancholic', 'energetic', 'peaceful'];

export const EMOTION_PARAMS = {
  ethereal: { tempo_factor: 0.8, velocity_range: [60, 80], duration_factor: 1.5, zoom: 0.5, offset: [0.2, 0.3] },
  dark: { tempo_factor: 0.7, velocity_range: [70, 90], duration_factor: 1.2, zoom: 0.3, offset: [-0.5, 0.0] },
  joyful: { tempo_factor: 1.2, velocity_range: [80, 100], duration_factor: 0.8, zoom: 0.4, offset: [0.3, 0.4] },
  melancholic: { tempo_factor: 0.6, velocity_range: [50, 70], duration_factor: 1.3, zoom: 0.6, offset: [-0.2, 0.1] },
  energetic: { tempo_factor: 1.4, velocity_range: [90, 110], duration_factor: 0.6, zoom: 0.3, offset: [0.4, 0.5] },
  peaceful: { tempo_factor: 0.7, velocity_range: [40, 60], duration_factor: 1.4, zoom: 0.7, offset: [0.1, 0.2] },
};

// MelodyParams defaults used by CreativeComposer.
export const RESOLUTION = 200, MAX_ITER = 100, BASE_NOTE = 60, OCTAVE_RANGE = 2, NOTE_DURATION = 0.25;
const STEP = 4; // scaled_values[::4]

// np.ogrid[a:b:n*1j] -> arange(n) * step + a
const grid = (a, b, n) => Array.from({ length: n }, (_, i) => i * ((b - a) / (n - 1)) + a);

// FractalMelodyGenerator.mandelbrot(h, w, max_iter): escape iteration per pixel (max_iter where it never escapes).
export function mandelbrot(emotion, h = RESOLUTION, w = RESOLUTION, maxIter = MAX_ITER) {
  const { zoom, offset } = EMOTION_PARAMS[emotion];
  const ys = grid(-1.4, 1.4, h), xs = grid(-2, 0.8, w);
  const out = new Int32Array(h * w);
  for (let i = 0; i < h; i++) {
    const ci = ys[i] * zoom + offset[1];
    for (let j = 0; j < w; j++) {
      const cr = xs[j] * zoom + offset[0];
      let zr = cr, zi = ci, div = maxIter;
      for (let it = 0; it < maxIter; it++) {
        const nr = zr * zr - zi * zi + cr, ni = zr * zi + zi * zr + ci;
        zr = nr; zi = ni;
        if (zr * zr + zi * zi > 4) { if (div === maxIter) div = it; zr = 2; zi = 0; }
      }
      out[i * w + j] = div;
    }
  }
  return out;
}

// Savitzky-Golay (window 15, cubic) weights as integers over a common denominator: the centre weights, and the
// polynomial-fit rows SciPy's mode='interp' uses at the edges for the sampled columns 0 and 4 (left window 0..14)
// and 196 (right window 185..199, position 11).
const SG = {
  centre: [1105, [-78, -13, 42, 87, 122, 147, 162, 167, 162, 147, 122, 87, 42, -13, -78]],
  0: [3060, [2059, 1144, 484, 44, -211, -316, -306, -216, -81, 64, 184, 244, 209, 44, -286]],
  4: [278460, [-19201, 19604, 44294, 57004, 59869, 55024, 44604, 30744, 15579, 1244, -10126, -16396, -15431, -5096, 16744]],
  196: [139230, [11102, -1573, -8338, -10208, -8198, -3323, 3402, 10962, 18342, 24527, 28502, 29252, 25762, 17017, 2002]],
};

// int(savgol(row)[col] % n) for every sampled column, computed exactly.
function degreesOfRow(div, row, w, n) {
  const out = [];
  for (let col = 0; col < w; col += STEP) {
    const [den, wts] = SG[col] || SG.centre;
    const start = SG[col] ? (col < w / 2 ? 0 : w - 15) : col - 7;
    let s = 0;
    for (let k = 0; k < 15; k++) s += wts[k] * div[row * w + start + k];
    const m = ((s % (n * den)) + n * den) % (n * den);
    out.push(Math.floor(m / den));
  }
  return out;
}

const degreeCache = new Map();
// Scale-degree grid (rows x 50 sampled columns) for an emotion, as SciPy produces it.
export function scaleDegrees(emotion, n = 7 * OCTAVE_RANGE) {
  const key = emotion + ':' + n;
  if (degreeCache.has(key)) return degreeCache.get(key);
  const div = mandelbrot(emotion);
  const rows = [];
  for (let r = 0; r < RESOLUTION; r++) rows.push(degreesOfRow(div, r, RESOLUTION, n));
  if (n === 7 * OCTAVE_RANGE) for (const [row, col, v] of QUIRKS[emotion] || []) rows[row][col / STEP] = v;
  degreeCache.set(key, rows);
  return rows;
}

export function currentScale(mode) {
  const scale = [];
  for (let o = 0; o < OCTAVE_RANGE; o++) for (const iv of MODES[mode]) scale.push(BASE_NOTE + iv + 12 * o);
  return scale;
}

// is_valid_melodic_step: no leap above an octave, consonant intervals only.
const CONSONANT = new Set([0, 3, 4, 5, 7, 8, 9]);
export const validStep = (prev, next) => { const iv = Math.abs(next - prev); return iv <= 12 && CONSONANT.has(iv); };

const melodyCache = new Map();
// The pitch sequence of generate_melody does not depend on the random stream; cache it per mode and emotion.
export function melodyPitches(mode, emotion) {
  const key = mode + ':' + emotion;
  if (melodyCache.has(key)) return melodyCache.get(key);
  const scale = currentScale(mode), melody = [];
  for (const row of scaleDegrees(emotion, scale.length)) {
    for (const v of row) {
      const note = scale[v];
      if (!melody.length || validStep(melody[melody.length - 1], note)) melody.push(note);
    }
  }
  melodyCache.set(key, melody);
  return melody;
}

// FractalMelodyGenerator.generate_melody(): [melody, timing, velocities]; draws one np.random.randint per note.
export function generateMelody(mode, emotion, np) {
  const p = EMOTION_PARAMS[emotion];
  const melody = melodyPitches(mode, emotion).slice();
  const timing = [], velocities = [];
  let t = 0;
  for (let i = 0; i < melody.length; i++) {
    timing.push(t);
    velocities.push(np.randint(p.velocity_range[0], p.velocity_range[1]));
    t += NOTE_DURATION * p.duration_factor;
  }
  return [melody, timing, velocities];
}
