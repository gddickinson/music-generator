# Creative Music Generator -- Interface Map

## Module Overview

| File | Purpose |
|------|---------|
| `fractal_melody_generator.py` | Fractal-based melody generation with mode/emotion support |
| `harmony_engine.py` | Harmony generation with voice leading and chord progressions |
| `musical_genetics.py` | Genetic algorithms for evolving musical phrases |
| `musical_types.py` | Enums and dataclasses (MusicalForm, Note, Motif, Section, etc.); `advanced_composer.py` keeps its own copies |
| `motif_development.py` | Motif transformations (invert, retrograde, augment, fragment, etc.) |
| `voice_leading.py` | Voice leading optimization (parallel fifths detection, voicing generation) |
| `music_visualization.py` | Visualization: form graphs, tension plots, voice leading diagrams |
| `advanced_composer.py` | CreativeComposer and the forms; still defines its own copies of the types, MotifDevelopment, VoiceLeading and Visualization (these shadow the imported split modules, so fixes go in both places) |
| `main_composer.py` | Simpler composition entry point using fractal + harmony + genetics |
| `multi_fractal_music.py` | Multi-fractal composition (Mandelbrot, Koch, Sierpinski, Julia, L-system); `multi-fractal-music.py` is an identical older copy |
| `test_music_generator.py` | Unit tests for all modules |
| `tests/` | `test_composer_regressions.py` (bug fixes), `test_web.py`, `test_fixtures_current.py`, `fixtures/` (Python output the JS is checked against) |
| `run_web.py` | Starts the web app (FastAPI, port 8166) |

## Key Classes

- `FractalMelodyGenerator` (`fractal_melody_generator.py`): Generates melodies from Mandelbrot set patterns
- `HarmonyEngine` (`harmony_engine.py`): Creates harmonizations with style-aware chord progressions
- `MusicalGenetics` (`musical_genetics.py`): Evolves musical phrases via genetic algorithms
- `CreativeComposer` (`advanced_composer.py`): Full composition system supporting multiple musical forms
- `MotifDevelopment` (`motif_development.py`): Static methods for motif transformation
- `VoiceLeading` (`voice_leading.py`): Chord voicing optimization
- `Visualization` (`music_visualization.py`): Matplotlib/networkx visualizations

## Entry Points

- `advanced_composer.py` -- primary entry point for full compositions (sonata, arch, moment, process, etc.)
- `main_composer.py` -- simpler entry point for classical/jazz/ambient styles
- `multi_fractal_music.py` -- standalone multi-fractal composition

## Data Flow

```
fractal_melody_generator --> advanced_composer
harmony_engine          --> advanced_composer
musical_types           --> motif_development, voice_leading, music_visualization, advanced_composer
motif_development       --> advanced_composer
voice_leading           --> advanced_composer
music_visualization     --> advanced_composer
musical_genetics        --> main_composer
```

## Web interface (`web/`, `scripts/`)

| File | Purpose |
|------|---------|
| `web/app.py` | FastAPI app: `/api/info`, `/api/verify` (exported to the static site via `STATIC_ENDPOINTS`), POST `/api/compose` (local only) |
| `web/compose.py` | `compose(params)`: seeds `random` + `np.random`, runs `CreativeComposer`, returns notes, sections, MIDI (base64) |
| `web/fractal.py` | Python vs exact Savitzky-Golay scale-degree grids; `quirks()` (cells where SciPy rounds below a whole number) |
| `scripts/make_fixtures.py` | Writes `web/static/fractal_quirks.js` and `tests/fixtures/*.json` from the Python code |
| `scripts/build_static.py`, `scripts/screenshots.py` | Static GitHub Pages export; browser smoke test and README images |
| `.github/workflows/pages.yml` | Tests and publishes the static site on push to `main` |

JavaScript engine (`web/static/`, ES modules, no build step), mirroring the Python:

| File | Ports |
|------|-------|
| `pyrandom.js` | MT19937, CPython `random` (seed, random, randint, choice, uniform, sample), NumPy legacy `seed`/`randint` |
| `fractal_engine.js` | `fractal_melody_generator.py`: modes, emotion params, `mandelbrot`, exact savgol degrees (+ `fractal_quirks.js`), `generate_melody` |
| `motif_engine.js` | `MotifDevelopment` and the `create_*_voice` texture handlers |
| `forms_core.js` / `forms_extra.js` | Form initialisers: sonata, minimalist, through-composed, aleatoric / arch, moment, process, mixed, `_develop_section` |
| `composer_engine.js` | `CreativeComposer` (`compose(params)`), section duration/tension, `add_motif_to_midi`, Python `sum()` |
| `midi_writer.js` | MIDIUtil 1.2.1 `MIDIFile` (addNote, addTempo, writeFile with duplicate removal and de-interleaving) |
| `timing.js`, `pianoroll.js`, `player.js` | Tempo map, piano-roll canvas, WebAudio playback and WAV rendering |
| `home.js`, `compose.js`, `fractal.js`, `verify.js`, `about.js` | Pages (hash router in `core.js`, entry `main.js`, data via `static_api.js` on the static site) |
| `*.test.mjs` | Node tests against `tests/fixtures/` (run by `tests/test_web.py`) |
