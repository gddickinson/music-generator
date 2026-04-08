# Creative Music Generator -- Interface Map

## Module Overview

| File | Purpose |
|------|---------|
| `fractal_melody_generator.py` | Fractal-based melody generation with mode/emotion support |
| `harmony_engine.py` | Harmony generation with voice leading and chord progressions |
| `musical_genetics.py` | Genetic algorithms for evolving musical phrases |
| `musical_types.py` | Shared enums, dataclasses (MusicalForm, Note, Motif, Section, etc.) |
| `motif_development.py` | Motif transformations (invert, retrograde, augment, fragment, etc.) |
| `voice_leading.py` | Voice leading optimization (parallel fifths detection, voicing generation) |
| `music_visualization.py` | Visualization: form graphs, tension plots, voice leading diagrams |
| `advanced_composer.py` | Main CreativeComposer class orchestrating all modules |
| `main_composer.py` | Simpler composition entry point using fractal + harmony + genetics |
| `multi_fractal_music.py` | Multi-fractal composition (Mandelbrot, Koch, Sierpinski, Julia, L-system) |
| `test_music_generator.py` | Unit tests for all modules |

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
