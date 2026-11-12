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

**Open questions for George** (resolved in the follow-up below).

No Mac mini use.


## 2026-11-11 (follow-up, approved by George via the lead)

- Home page overflowed 390 px by 145 px when first drawn at desktop width and then narrowed: grid children may now
  shrink, the roll scrolls in its own box and redraws on resize. Checked at 390 px on every route (fresh and resized).
- Duplicates removed: `advanced_composer.py` imports the split modules (about 490 lines removed, output byte-identical
  for 11 forms x 4 seeds); `multi-fractal-music.py` deleted (nothing imported it).
- Section timing in beats: motifs of a section start together; the next section starts where the last note ends.
  Every piece changes (accepted). Test: sections tile the timeline exactly for every form.
- `key`, `development_complexity`, `texture_density`, `modulation_frequency` now affect the notes (see
  musical_types.py); the polyphonic voice uses the home key's tonic triad. Tests for each.
- JS engine and fixtures updated: 31 pieces, 16,686 notes, all MIDI bytes identical to the Python.
