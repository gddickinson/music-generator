# Session log

## 2026-11-11: web interface, static site, bug fixes (Claude, branch `web-interface`)

**Bug fixes** (each with a regression test in `tests/test_composer_regressions.py`, separate commits):
- MIDI export crashed (`IndexError: pop from empty list` in MIDIUtil) for the default sonata form and many other
  pieces: overlapping sections put two notes with the same pitch and start tick on one track. `add_motif_to_midi`
  now skips the second one. Pieces that already exported are byte-identical.
- `FractalMelodyGenerator` ignored later changes to `params.emotion`, so every theme used the first emotion's fractal
  region and velocities. `current_params` is now a property.
- Minimalist form built its base motif from the `(melody, timing, velocities)` tuple (three nonsense notes).
- `phase_shift` doubled note onsets (cumulative `timing_offset` added to the running time again). Fixed in both copies.
- Arch/moment/process forms were labelled MIXED; `generate_wav=True` crashed with AttributeError after composing (now a
  clear NotImplementedError up front).
- README/INTERFACE/ROADMAP claims corrected (placeholder URLs, missing examples dir and CONTRIBUTING.md, WAV output,
  ticked-but-undone roadmap items).

**Web interface**: `web/` (FastAPI, port 8166) + static GitHub Pages site. The composer is ported to JS and matches the
Python exactly: 28 fixture pieces (20,332 notes, all MIDI bytes) in the node test, 13 more on the Verify page. One
data table comes from Python: 11 fractal cells where SciPy's float savgol rounds below a whole number
(`fractal_quirks.js`); `tests/test_fixtures_current.py` skips (not fails) if another platform rounds those differently.

**Open questions for George**:
- Section timing: `calculate_section_duration` divides beats by tempo/60 ("seconds") but the result is used as MIDI
  beats, and time advances by the whole section length after *every* motif. Sections overlap (the cause of the crash
  above) or leave long gaps (sonata development). Fixing it would change every piece.
- `key` has no effect on pitches (melodies always start at middle C); `development_complexity`, `texture_density` and
  `modulation_frequency` are never read; rondo/theme-and-variations/fugue fall back to mixed.
- `multi-fractal-music.py` is an identical copy of `multi_fractal_music.py`; `advanced_composer.py` duplicates the
  split modules.

No Mac mini use.
