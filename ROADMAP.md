# Creative Music Generator — Roadmap

## Current State
A sophisticated algorithmic composition system with 6 Python files implementing fractal melody generation, harmony engine with voice leading, genetic algorithms for musical development, and an advanced composer orchestrating multiple musical forms (sonata, arch, moment, process, minimalist, aleatoric). Outputs MIDI files with optional visualization. Code is well-organized across focused modules but has no package structure, no tests, and no CLI interface. All files sit in the project root.

## Short-term Improvements
- [x] Add `requirements.txt` — needs `numpy`, `matplotlib`, `midiutil`, `scipy`, `networkx`, `tqdm`
- [ ] Add `__init__.py` and reorganize into a package: `music_generator/` with submodules
- [x] Rename `multi-fractal-music.py` to `multi_fractal_music.py` (hyphen prevents Python import)
- [ ] Add CLI interface: `python -m music_generator --form sonata --key C --tempo 120 --output my_piece.mid`
- [x] Add unit tests for `harmony_engine.py` — verify voice leading rules, chord voicing, consonance scoring
- [x] Add unit tests for `fractal_melody_generator.py` — test scale mapping, mode application
- [ ] Add input validation in `advanced_composer.py` for `CreativeComposerParams` (invalid keys, out-of-range tempos)
- [ ] Clean up `main_composer.py` vs `advanced_composer.py` — clarify which is the primary entry point

## Feature Enhancements
- [ ] Add real-time MIDI output using `python-rtmidi` for live playback during composition
- [ ] Add audio synthesis: generate WAV directly using `sounddevice` or `pydub` instead of MIDI-only
- [ ] Add interactive parameter exploration: Streamlit or Gradio web UI for tweaking composition parameters
- [ ] Add more fractal types in `fractal_melody_generator.py`: Julia set, Sierpinski, L-systems
- [ ] Add rhythmic pattern generation: Euclidean rhythms, polyrhythms, metric modulation
- [ ] Add counterpoint rules to `harmony_engine.py`: species counterpoint, canon generation
- [ ] Add MusicXML export for notation display in MuseScore/Sibelius
- [ ] Add `musical_genetics.py` fitness visualization: show evolution of motifs over generations

## Long-term Vision
- [ ] Add machine learning integration: train on MIDI corpora to guide genetic algorithm fitness
- [ ] Add real-time performance mode: generate music on-the-fly responding to user input
- [ ] Build web interface for non-programmers to create algorithmic compositions
- [ ] Add multi-instrument orchestration: assign voices to specific MIDI instruments with idiomatic ranges
- [ ] Package and publish to PyPI: `pip install creative-music-generator`
- [ ] Add MIDI input: analyze existing MIDI files and generate variations/continuations

## Technical Debt
- [ ] `main_composer.py` vs `advanced_composer.py` — unclear which is canonical; consolidate or document
- [x] `multi-fractal-music.py` hyphenated name breaks importability
- [ ] No package structure — all 6 files in project root
- [x] Generated output files (`creative_composition.mid`, `form_structure.png`, `tension_graph.png`) are in the repo — add to `.gitignore`
- [x] No tests whatsoever — musical output is hard to test but harmony rules and scale math are deterministic
- [ ] Import chain (`advanced_composer.py` imports from `fractal_melody_generator.py` and `harmony_engine.py`) is fragile without proper packaging
- [ ] `musical_genetics.py` population management may have performance issues with large populations — needs profiling
- [x] `advanced_composer.py` exceeded 500 lines — split into `musical_types.py`, `motif_development.py`, `voice_leading.py`, `music_visualization.py`
