# Creative Music Generator

A sophisticated algorithmic composition system that combines fractal patterns, musical genetics, and advanced form structures to create complex musical compositions.

## Overview

This project implements a creative music generation system that uses multiple algorithmic approaches to create musical compositions. It combines:

- Fractal-based melody generation
- Advanced musical form structures
- Genetic algorithms for musical development
- Sophisticated harmony generation
- Multiple texture types and voice leading
- Visualization tools for musical analysis

## Components

The project consists of several Python modules:

### advanced_composer.py
The main composition engine that orchestrates all components and implements various musical forms including:
- Sonata form
- Arch form (ABCBA structure)
- Moment form (Stockhausen-inspired)
- Process music (Reich/Glass-inspired)
- Minimalist
- Aleatoric
- Through-composed

### fractal_melody_generator.py
Generates melodies using fractal patterns (Mandelbrot set) with:
- Multiple musical modes
- Emotional styling
- Pitch-to-scale mapping
- Dynamic and articulation control

### harmony_engine.py
Handles harmonic progression with:
- Voice leading optimization
- Multiple harmony styles (Classical, Jazz, Modal)
- Chord voicing generation
- Consonance/dissonance control

### musical_genetics.py
Implements genetic algorithms for musical development:
- Motif evolution
- Fitness evaluation
- Musical crossover and mutation
- Population management

### multi-fractal-music.py
Extends fractal generation to multiple musical parameters:
- Multiple fractal types (Mandelbrot, Julia, Koch, etc.)
- Independent control of melody, harmony, and rhythm
- Sophisticated visualization tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/creative-music-generator.git
cd creative-music-generator
```

2. Install required dependencies:
```bash
pip install numpy matplotlib midiutil scipy networkx tqdm
```

## Usage

Basic usage:

```python
from advanced_composer import CreativeComposer, CreativeComposerParams, MusicalForm, MusicalMode, EmotionalStyle

# Create composer parameters
params = CreativeComposerParams(
    form=MusicalForm.PROCESS,  # Choose musical form
    base_tempo=100,
    key="C",
    mode=MusicalMode.IONIAN,
    emotional_journey=[
        EmotionalStyle.PEACEFUL,
        EmotionalStyle.ENERGETIC,
        EmotionalStyle.MELANCHOLIC,
        EmotionalStyle.DARK,
        EmotionalStyle.ETHEREAL,
        EmotionalStyle.JOYFUL
    ],
    section_count=8,
    voice_count=4,
    visualization_enabled=True
)

# Create composer and generate composition
composer = CreativeComposer(params)
midi, audio = composer.generate_composition()

# Save the composition
composer.save_composition(midi, audio, "my_composition")
```

## Musical Theory and Implementation

### Fractal Melody Generation
The system uses fractal patterns to generate melodies by:
1. Generating a Mandelbrot set
2. Mapping fractal values to musical scales
3. Applying emotional parameters
4. Creating coherent melodic contours

### Musical Forms
Implements various musical forms including:

#### Sonata Form
- Exposition (Primary and Secondary themes)
- Development
- Recapitulation
- Optional Coda

#### Arch Form (ABCBA)
- Palindromic structure
- Gradual build-up to climax
- Symmetric return

#### Process Music
- Gradual transformation
- Phase shifting
- Additive processes
- Textural evolution

#### Moment Form
- Independent sections
- Non-linear structure
- Contrasting materials
- Abrupt transitions

### Voice Leading and Harmony
- Implements traditional voice leading rules
- Avoids parallel fifths/octaves
- Maintains proper spacing
- Controls voice crossing

### Genetic Development
Uses genetic algorithms for:
- Motif development
- Phrase evolution
- Form generation
- Style adaptation

## Visualization

The system generates various visualizations:
- Form structure graphs
- Tension graphs
- Voice leading analysis
- Harmonic progression visualization

## Output Formats

- MIDI files (.mid)
- Optional WAV audio (.wav)
- Visualization images (.png)
- Musical analysis data

## Examples

The repository includes example compositions in the `examples` directory, demonstrating different forms and styles:
- Sonata form example
- Process music example
- Arch form example
- Moment form example

## Contributing

Contributions are welcome! Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- Special thanks to the music theory and algorithmic composition community
- Inspired by works of Steve Reich, Karlheinz Stockhausen, and György Ligeti
- Built using various Python scientific computing libraries

## Future Development

Planned features:
- Additional musical forms
- More sophisticated genetic algorithms
- Enhanced visualization tools
- Real-time MIDI output
- Machine learning integration
- Web interface for parameter control

## Citation

If you use this project in your research, please cite:
```
@software{creative_music_generator,
  title = {Creative Music Generator},
  author = {[Your Name]},
  year = {2024},
  url = {https://github.com/yourusername/creative-music-generator}
}
```