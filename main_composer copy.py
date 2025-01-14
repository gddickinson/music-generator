import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from midiutil import MIDIFile
from tqdm import tqdm
import time

# Import our modules with correct class names
from fractal_melody_generator import (
    MusicalMode, EmotionalStyle, MelodyParams, FractalMelodyGenerator
)
from harmony_engine import HarmonyParams, HarmonyEngine, HarmonyStyle
from musical_genetics import GeneticParams, MusicalGenetics, MusicalPhrase

class CompositionStyle(Enum):
    CLASSICAL = "classical"
    JAZZ = "jazz"
    AMBIENT = "ambient"
    EXPERIMENTAL = "experimental"
    MINIMALIST = "minimalist"

@dataclass
class CompositionParams:
    style: CompositionStyle = CompositionStyle.CLASSICAL
    tempo: int = 120
    duration: float = 240  # in seconds
    complexity: float = 0.5  # 0.0 to 1.0
    evolution_generations: int = 50
    fractal_resolution: int = 200
    num_voices: int = 4
    batch_size: int = 100  # For audio synthesis
    generate_wav: bool = False  # Option to skip WAV generation
    form: str = "AABA"  # Musical form
    num_variations: int = 3  # Number of variations for each theme
    section_length: int = 32  # Bars per section
    emotional_progression: List[EmotionalStyle] = None  # Emotional journey

    def __post_init__(self):
        if self.emotional_progression is None:
            self.emotional_progression = [
                EmotionalStyle.PEACEFUL,
                EmotionalStyle.ENERGETIC,
                EmotionalStyle.ETHEREAL,
                EmotionalStyle.JOYFUL
            ]

class IntegratedComposer:
    def __init__(self, params: CompositionParams):
        self.params = params
        self.setup_components()

    def setup_components(self):
        """Initialize all composition components with progress updates"""
        print("Initializing composition components...")

        # Setup fractal parameters based on composition style
        print("- Setting up fractal melody generator...")
        mode, emotion = self.create_fractal_params()
        melody_params = MelodyParams(
            mode=mode,
            emotion=emotion,
            tempo=self.params.tempo
        )
        self.fractal_generator = FractalMelodyGenerator(melody_params)

        # Setup harmony parameters
        print("- Setting up harmony engine...")
        harmony_params = self.create_harmony_params()
        self.harmony_engine = HarmonyEngine(harmony_params)

        # Setup genetic parameters
        print("- Setting up genetic evolution engine...")
        genetic_params = self.create_genetic_params()
        self.genetic_engine = MusicalGenetics(genetic_params)

        print("All components initialized successfully!")

    def create_fractal_params(self) -> Tuple[MusicalMode, EmotionalStyle]:
        """Create appropriate mode and emotion for style"""
        style_mapping = {
            CompositionStyle.CLASSICAL: (MusicalMode.IONIAN, EmotionalStyle.JOYFUL),
            CompositionStyle.JAZZ: (MusicalMode.DORIAN, EmotionalStyle.ENERGETIC),
            CompositionStyle.AMBIENT: (MusicalMode.LYDIAN, EmotionalStyle.PEACEFUL),
            CompositionStyle.EXPERIMENTAL: (MusicalMode.PHRYGIAN, EmotionalStyle.DARK),
            CompositionStyle.MINIMALIST: (MusicalMode.MIXOLYDIAN, EmotionalStyle.ETHEREAL)
        }
        return style_mapping[self.params.style]

    def create_harmony_params(self) -> HarmonyParams:
        """Create appropriate harmony parameters for style"""
        style_map = {
            CompositionStyle.CLASSICAL: HarmonyStyle.CLASSICAL,
            CompositionStyle.JAZZ: HarmonyStyle.JAZZ,
            CompositionStyle.AMBIENT: HarmonyStyle.MODAL,
            CompositionStyle.EXPERIMENTAL: HarmonyStyle.IMPRESSIONIST,
            CompositionStyle.MINIMALIST: HarmonyStyle.MODAL
        }

        return HarmonyParams(
            style=style_map[self.params.style],
            tempo=self.params.tempo,
            num_voices=self.params.num_voices,
            complexity=self.params.complexity
        )

    def create_genetic_params(self) -> GeneticParams:
        """Create appropriate genetic parameters for style"""
        return GeneticParams(
            generations=self.params.evolution_generations,
            target_complexity=self.params.complexity,
            target_consonance=0.7 if self.params.style in
                [CompositionStyle.CLASSICAL, CompositionStyle.MINIMALIST] else 0.5
        )

    def generate_structured_composition(self) -> Tuple[MIDIFile, Optional[np.ndarray]]:
        """Generate a longer, structured composition with multiple sections"""
        print("\nGenerating structured composition...")
        print(f"Form: {self.params.form}")
        print(f"Emotional progression: {[e.value for e in self.params.emotional_progression]}")

        sections = {}
        variations = {}
        waveforms = {}

        # Generate base themes
        print("\nGenerating base themes...")
        for section in set(self.params.form):
            print(f"\nCreating theme for section {section}")
            # Generate with corresponding emotion
            emotion_idx = ord(section) - ord('A')
            emotion = self.params.emotional_progression[emotion_idx % len(self.params.emotional_progression)]

            # Update fractal generator emotion
            self.fractal_generator.params.emotion = emotion

            # Generate theme
            melody, timing, velocities = self.fractal_generator.generate_melody()
            sections[section] = (melody, timing, velocities)

            # Generate variations
            variations[section] = []
            for i in range(self.params.num_variations):
                print(f"Creating variation {i+1} for section {section}")
                # Evolve variation using genetics
                var_melody = self.evolve_variation(melody, timing, velocities)
                variations[section].append(var_melody)

        # Combine sections according to form
        print("\nCombining sections into final composition...")
        final_midi = MIDIFile(2)  # Melody and harmony tracks
        final_midi.addTempo(0, 0, self.params.tempo)

        current_time = 0
        all_waveforms = []

        # Build full composition
        for section_idx, section in enumerate(self.params.form):
            print(f"\nProcessing section {section}")
            # Choose either original or variation
            if section_idx > 0 and section in sections:
                var_idx = (section_idx - 1) % (self.params.num_variations + 1)
                if var_idx == 0:
                    melody, timing, velocities = sections[section]
                else:
                    melody = variations[section][var_idx - 1]
                    timing = sections[section][1]
                    velocities = sections[section][2]
            else:
                melody, timing, velocities = sections[section]

            # Generate harmony for this section
            harmony = self.harmony_engine.generate_harmonic_progression()

            # Add to MIDI
            for note, t, vel in zip(melody, timing, velocities):
                final_midi.addNote(0, 0, note, current_time + t,
                                 self.params.duration, vel)

            # Add harmony
            for chord, t in zip(harmony, timing):
                for note in chord:
                    final_midi.addNote(1, 0, note, current_time + t,
                                     self.params.duration, 80)

            # Generate waveform if needed
            if self.params.generate_wav:
                waveform = self.generate_optimized_waveform(melody, timing, velocities)
                all_waveforms.append(waveform)

            current_time += max(timing) + self.params.duration

        # Combine waveforms if needed
        final_waveform = None
        if self.params.generate_wav and all_waveforms:
            print("\nCombining audio waveforms...")
            final_waveform = np.concatenate(all_waveforms)
            final_waveform = final_waveform / np.max(np.abs(final_waveform))

        return final_midi, final_waveform

    def evolve_variation(self, melody: List[int], timing: List[float],
                        velocities: List[int]) -> List[int]:
        """Evolve a variation of a given melody"""
        # Initialize genetic population with the original melody
        self.genetic_engine.initialize_population()
        phrase = self.convert_fractal_to_phrase(melody, timing, velocities, None)
        self.genetic_engine.population[0] = phrase

        # Evolve for fewer generations for variations
        for _ in range(self.params.evolution_generations // 2):
            self.genetic_engine.evolve()

        # Return evolved melody
        best_phrase = max(self.genetic_engine.population, key=lambda x: x.fitness)
        return best_phrase.melody
        """Generate complete musical composition with detailed progress tracking"""
        total_start_time = time.time()
        print("\nBeginning composition generation process...")

        # Step 1: Generate Fractal Melody
        print("\nStep 1/4: Generating fractal melody...")
        step_start = time.time()
        melody, timing, velocities = self.fractal_generator.generate_melody()
        step_time = time.time() - step_start
        print(f"✓ Melody generation completed in {step_time:.2f} seconds")
        print(f"✓ Generated {len(melody)} notes")

        # Step 2: Create MIDI representation
        print("\nStep 2/4: Creating MIDI representation...")
        step_start = time.time()
        midi = MIDIFile(1)
        midi.addTempo(0, 0, self.params.tempo)

        for note, t, vel in tqdm(zip(melody, timing, velocities),
                                total=len(melody),
                                desc="Processing notes"):
            midi.addNote(0, 0, note, t, self.params.duration, vel)

        step_time = time.time() - step_start
        print(f"✓ MIDI creation completed in {step_time:.2f} seconds")

        # Step 3: Generate Waveform
        print("\nStep 3/4: Generating waveform...")
        step_start = time.time()
        fractal_wave = self.generate_optimized_waveform(melody, timing, velocities)
        step_time = time.time() - step_start
        print(f"✓ Waveform generation completed in {step_time:.2f} seconds")

        # Step 4: Evolve Melody
        print("\nStep 4/4: Evolving melodic variations...")
        step_start = time.time()

        # Initialize genetic population
        print("Initializing genetic population...")
        self.genetic_engine.initialize_population()

        # Add fractal melody to population
        fractal_phrase = self.convert_fractal_to_phrase(
            melody=melody,
            timing=timing,
            velocities=velocities,
            fractal_wave=fractal_wave
        )
        self.genetic_engine.population[0] = fractal_phrase

        # Evolve for specified generations with progress tracking
        best_fitness = float('-inf')
        for generation in tqdm(range(self.params.evolution_generations),
                             desc="Evolving melody"):
            self.genetic_engine.evolve()
            current_best = max(self.genetic_engine.population,
                             key=lambda x: x.fitness).fitness

            if current_best > best_fitness:
                best_fitness = current_best
                print(f"\n→ Generation {generation}: New best fitness = {best_fitness:.4f}")

        # Get best evolved melody
        best_phrase = max(self.genetic_engine.population, key=lambda x: x.fitness)
        step_time = time.time() - step_start
        print(f"✓ Evolution completed in {step_time:.2f} seconds")
        print(f"✓ Final best fitness: {best_phrase.fitness:.4f}")

        # Generate harmony
        print("\nGenerating harmonic accompaniment...")
        harmony_start = time.time()
        voiced_progression, final_midi = self.harmony_engine.generate_accompaniment(
            best_phrase.melody
        )
        harmony_time = time.time() - harmony_start
        print(f"✓ Harmony generation completed in {harmony_time:.2f} seconds")

        # Create final composition
        print("\nCreating final composition...")
        final_start = time.time()
        final_midi, final_audio = self.create_final_composition(
            best_phrase, voiced_progression
        )
        final_time = time.time() - final_start
        print(f"✓ Final composition completed in {final_time:.2f} seconds")

        # Report total time
        total_time = time.time() - total_start_time
        print(f"\nTotal composition time: {total_time:.2f} seconds")

        return final_midi, final_audio

    def generate_optimized_waveform(self, melody: List[int],
                                  timing: List[float],
                                  velocities: List[int]) -> np.ndarray:
        """Generate audio waveform with optimized batch processing"""
        # Initialize waveform
        sample_rate = 44100
        duration = max(timing) + self.params.duration
        total_samples = int(sample_rate * duration)
        waveform = np.zeros(total_samples)

        # Pre-calculate common values
        note_samples = int(self.params.duration * sample_rate)
        t_note = np.linspace(0, self.params.duration, note_samples)
        envelope = np.exp(-3 * t_note / self.params.duration)

        # Process notes in batches
        for i in tqdm(range(0, len(melody), self.params.batch_size),
                     desc="Synthesizing audio"):
            batch_end = min(i + self.params.batch_size, len(melody))
            batch_slice = slice(i, batch_end)

            # Process each note in the batch
            for note, start_time, vel in zip(
                melody[batch_slice],
                timing[batch_slice],
                velocities[batch_slice]
            ):
                freq = 440 * (2 ** ((note - 69) / 12))
                wave = np.sin(2 * np.pi * freq * t_note) * (vel / 127) * envelope

                idx_start = int(start_time * sample_rate)
                idx_end = idx_start + note_samples
                if idx_end <= total_samples:
                    waveform[idx_start:idx_end] += wave

        # Normalize waveform
        waveform = waveform / np.max(np.abs(waveform))
        return waveform

    def convert_fractal_to_phrase(self, melody: List[int],
                                timing: List[float],
                                velocities: List[int],
                                fractal_wave: np.ndarray) -> MusicalPhrase:
        """Convert fractal output to musical phrase"""
        print("Converting fractal melody to musical phrase...")

        # Generate basic harmony
        harmony = self.harmony_engine.generate_harmonic_progression()

        return MusicalPhrase(melody, harmony, timing, velocities)

    def create_final_composition(self, phrase: MusicalPhrase,
                               harmony: List[List[int]]) -> Tuple[MIDIFile, np.ndarray]:
        """Create final composition combining all elements"""
        print("Combining elements into final composition...")

        # Create MIDI with multiple tracks
        midi = MIDIFile(3)  # Melody, harmony, and effects tracks

        # Add melody track
        track = 0
        time = 0
        midi.addTrackName(track, time, "Evolved Melody")
        midi.addTempo(track, time, self.params.tempo)

        for note, duration, velocity in tqdm(
            zip(phrase.melody, phrase.rhythm, phrase.dynamics),
            desc="Processing melody track"
        ):
            midi.addNote(track, 0, note, time, duration, velocity)
            time += duration

        # Add harmony tracks
        track = 1
        time = 0
        midi.addTrackName(track, time, "Harmony")

        for chord, duration in tqdm(
            zip(harmony, phrase.rhythm),
            desc="Processing harmony track"
        ):
            for note in chord:
                midi.addNote(track, 1, note, time, duration, 80)
            time += duration

        # Generate final audio
        print("Generating final audio...")
        sample_rate = 44100
        duration = sum(phrase.rhythm)
        waveform = self.generate_optimized_waveform(
            phrase.melody,
            np.cumsum([0] + phrase.rhythm[:-1]),  # Convert durations to timings
            phrase.dynamics
        )

        return midi, waveform

    def save_composition(self, midi: MIDIFile, audio: Optional[np.ndarray],
                        base_filename: str = "composition"):
        """Save composition as MIDI and optionally WAV files"""
        print(f"\nSaving composition to {base_filename}.*...")

        # Save MIDI
        with open(f"{base_filename}.mid", "wb") as f:
            midi.writeFile(f)
        print(f"✓ Saved MIDI file: {base_filename}.mid")

        # Save WAV only if audio data is provided
        if audio is not None:
            from scipy.io import wavfile
            wavfile.write(f"{base_filename}.wav", 44100,
                         (audio * 32767).astype(np.int16))
            print(f"✓ Saved WAV file: {base_filename}.wav")

    def visualize_composition(self, phrase: MusicalPhrase,
                            harmony: List[List[int]],
                            filename: str = "composition_analysis.png"):
        """Create detailed visualization of the composition"""
        print(f"\nCreating visualization: {filename}")
        plt.figure(figsize=(15, 12))

        # Plot melody contour
        plt.subplot(411)
        plt.plot(phrase.melody, 'b-', label='Melody', alpha=0.7)
        plt.fill_between(range(len(phrase.melody)), phrase.melody, alpha=0.2)
        plt.title('Melodic Contour')
        plt.xlabel('Time')
        plt.ylabel('MIDI Note')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot harmony heat map
        plt.subplot(412)
        harmony_grid = np.zeros((128, len(harmony)))
        for t, chord in enumerate(harmony):
            for note in chord:
                harmony_grid[note, t] = 1
        plt.imshow(harmony_grid, aspect='auto', origin='lower',
                  cmap='Blues', interpolation='nearest')
        plt.title('Harmonic Progression')
        plt.xlabel('Time')
        plt.ylabel('MIDI Note')
        plt.colorbar(label='Note Intensity')

        # Plot rhythm/duration pattern
        plt.subplot(413)
        plt.plot(phrase.rhythm, 'g-', label='Rhythm', alpha=0.7)
        plt.fill_between(range(len(phrase.rhythm)), phrase.rhythm, alpha=0.2)
        plt.title('Rhythmic Pattern')
        plt.xlabel('Time')
        plt.ylabel('Duration (s)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot dynamics
        plt.subplot(414)
        plt.plot(phrase.dynamics, 'r-', label='Dynamics', alpha=0.7)
        plt.fill_between(range(len(phrase.dynamics)), phrase.dynamics, alpha=0.2)
        plt.title('Dynamic Profile')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved visualization: {filename}")


if __name__ == "__main__":
    print("Integrated Composer - Starting composition process...")

    # Create composer with desired style and form
    params = CompositionParams(
        style=CompositionStyle.AMBIENT,
        tempo=120,
        complexity=0.6,
        evolution_generations=50,
        batch_size=100,
        generate_wav=False,  # Set to True if you want WAV output
        form="ABACA",  # Rondo form
        num_variations=3,
        section_length=32,
        emotional_progression=[
            EmotionalStyle.PEACEFUL,    # A sections - main theme
            EmotionalStyle.ENERGETIC,   # B section - contrast
            EmotionalStyle.ETHEREAL,    # C section - development
        ]
    )

    # Initialize composer
    composer = IntegratedComposer(params)

    # Generate structured composition
    print("\nGenerating composition...")
    midi, audio = composer.generate_structured_composition()

    # Save outputs
    print("\nSaving composition files...")
    composer.save_composition(midi, audio if params.generate_wav else None)

    print("\nGenerated files:")
    print("- composition.mid (MIDI file)")
    if params.generate_wav:
        print("- composition.wav (Audio file)")

    print("\nComposition process complete!")
