import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from midiutil import MIDIFile
from tqdm import tqdm
import time
import random
from collections import defaultdict

# Import existing modules
from fractal_melody_generator import (
    MusicalMode, EmotionalStyle, MelodyParams, FractalMelodyGenerator
)
from harmony_engine import HarmonyParams, HarmonyEngine, HarmonyStyle
from musical_genetics import GeneticParams, MusicalGenetics, MusicalPhrase

class MusicalForm(Enum):
    SONATA = "sonata"
    RONDO = "rondo"
    THEME_AND_VARIATIONS = "theme_and_variations"
    FUGUE = "fugue"
    THROUGH_COMPOSED = "through_composed"

class Articulation(Enum):
    LEGATO = "legato"
    STACCATO = "staccato"
    MARCATO = "marcato"
    TENUTO = "tenuto"
    PORTATO = "portato"

@dataclass
class Motif:
    notes: List[int]
    rhythm: List[float]
    articulations: List[Articulation]
    dynamics: List[int]
    emotional_intent: EmotionalStyle

    def get_length(self) -> float:
        return sum(self.rhythm)

@dataclass
class Section:
    motifs: List[Motif]
    key: str
    tempo: int
    emotion: EmotionalStyle
    dynamics_base: int
    transitions: List[Tuple[float, float]]  # (time, value) for tempo/dynamic changes

@dataclass
class AdvancedCompositionParams:
    form: MusicalForm = MusicalForm.SONATA
    base_tempo: int = 120
    key: str = "C"
    mode: MusicalMode = MusicalMode.IONIAN
    emotional_journey: List[EmotionalStyle] = None
    development_complexity: float = 0.7
    contrapuntal_density: float = 0.6
    modulation_frequency: float = 0.3
    section_count: int = 4
    voice_count: int = 4
    generate_wav: bool = False

    def __post_init__(self):
        if self.emotional_journey is None:
            self.emotional_journey = [
                EmotionalStyle.PEACEFUL,
                EmotionalStyle.ENERGETIC,
                EmotionalStyle.MELANCHOLIC,
                EmotionalStyle.JOYFUL
            ]

class CounterpointRules:
    """Bach-style counterpoint rules"""

    @staticmethod
    def check_parallel_fifths(voice1: List[int], voice2: List[int]) -> bool:
        """Check for parallel fifths between voices"""
        for i in range(len(voice1) - 1):
            interval1 = abs(voice1[i] - voice2[i]) % 12
            interval2 = abs(voice1[i + 1] - voice2[i + 1]) % 12
            if interval1 == 7 and interval2 == 7:
                return False
        return True

    @staticmethod
    def check_parallel_octaves(voice1: List[int], voice2: List[int]) -> bool:
        """Check for parallel octaves between voices"""
        for i in range(len(voice1) - 1):
            interval1 = abs(voice1[i] - voice2[i]) % 12
            interval2 = abs(voice1[i + 1] - voice2[i + 1]) % 12
            if interval1 == 0 and interval2 == 0:
                return False
        return True

    @staticmethod
    def check_voice_crossing(voice1: List[int], voice2: List[int]) -> bool:
        """Check for voice crossing"""
        for n1, n2 in zip(voice1, voice2):
            if n1 < n2:  # Assuming voice1 should be higher than voice2
                return False
        return True

    @staticmethod
    def check_voice_leading(voice: List[int]) -> bool:
        """Check for proper voice leading"""
        for i in range(len(voice) - 1):
            interval = abs(voice[i] - voice[i + 1])
            if interval > 12:  # No jumps larger than an octave
                return False
        return True

class MotifDevelopment:
    """Techniques for developing motifs"""

    @staticmethod
    def invert(motif: Motif) -> Motif:
        """Invert the motif's pitch contour"""
        center = motif.notes[0]
        new_notes = [center + (center - note) for note in motif.notes]
        return Motif(
            notes=new_notes,
            rhythm=motif.rhythm.copy(),
            articulations=motif.articulations.copy(),
            dynamics=motif.dynamics.copy(),
            emotional_intent=motif.emotional_intent
        )

    @staticmethod
    def retrograde(motif: Motif) -> Motif:
        """Reverse the motif"""
        return Motif(
            notes=motif.notes[::-1],
            rhythm=motif.rhythm[::-1],
            articulations=motif.articulations[::-1],
            dynamics=motif.dynamics[::-1],
            emotional_intent=motif.emotional_intent
        )

    @staticmethod
    def augment(motif: Motif, factor: float = 2.0) -> Motif:
        """Augment the rhythm by a factor"""
        return Motif(
            notes=motif.notes.copy(),
            rhythm=[r * factor for r in motif.rhythm],
            articulations=motif.articulations.copy(),
            dynamics=motif.dynamics.copy(),
            emotional_intent=motif.emotional_intent
        )

    @staticmethod
    def diminish(motif: Motif, factor: float = 0.5) -> Motif:
        """Diminish the rhythm by a factor"""
        return Motif(
            notes=motif.notes.copy(),
            rhythm=[r * factor for r in motif.rhythm],
            articulations=motif.articulations.copy(),
            dynamics=motif.dynamics.copy(),
            emotional_intent=motif.emotional_intent
        )

    @staticmethod
    def transpose(motif: Motif, interval: int) -> Motif:
        """Transpose the motif by an interval"""
        return Motif(
            notes=[n + interval for n in motif.notes],
            rhythm=motif.rhythm.copy(),
            articulations=motif.articulations.copy(),
            dynamics=motif.dynamics.copy(),
            emotional_intent=motif.emotional_intent
        )

    @staticmethod
    def develop_sequence(motif: Motif, sequence_intervals: List[int]) -> List[Motif]:
        """Create a sequence of motif transpositions"""
        return [MotifDevelopment.transpose(motif, interval)
                for interval in sequence_intervals]

class VoiceLeading:
    """Advanced voice leading handler"""

    def __init__(self):
        self.consonant_intervals = {0, 3, 4, 7, 8, 9}  # Unison, thirds, fifths, sixths
        self.perfect_consonances = {0, 7}  # Unison and fifth

    def optimize_progression(self, current_chord: List[int],
                           next_chord: List[int]) -> List[int]:
        """Optimize voice leading between chords"""
        best_voicing = next_chord
        min_movement = float('inf')

        # Try different voicings of the next chord
        for voicing in self.generate_voicings(next_chord):
            total_movement = sum(abs(c - n) for c, n in zip(current_chord, voicing))
            if total_movement < min_movement:
                min_movement = total_movement
                best_voicing = voicing

        return best_voicing

    def generate_voicings(self, chord: List[int]) -> List[List[int]]:
        """Generate possible voicings for a chord"""
        voicings = []
        base_voicing = sorted(chord)

        # Add base voicing
        voicings.append(base_voicing)

        # Add octave displacements
        for i in range(len(chord)):
            up = base_voicing.copy()
            up[i] += 12
            down = base_voicing.copy()
            down[i] -= 12
            voicings.extend([up, down])

        return voicings

class Cadence:
    """Handles musical cadences"""

    AUTHENTIC = ([7, 11, 2], [0, 4, 7])      # V7-I
    PLAGAL = ([5, 9, 0], [0, 4, 7])          # IV-I
    DECEPTIVE = ([7, 11, 2], [9, 0, 4])      # V7-vi
    HALF = ([0, 4, 7], [7, 11, 2])           # I-V7

    @staticmethod
    def get_cadence(cadence_type: str, key: int) -> List[List[int]]:
        """Get cadence progression in specified key"""
        cadence = getattr(Cadence, cadence_type.upper())
        return [[note + key for note in chord] for chord in cadence]

class AdvancedComposer:
    def __init__(self, params: AdvancedCompositionParams):
        self.params = params
        self.counterpoint = CounterpointRules()
        self.voice_leading = VoiceLeading()
        self.motif_development = MotifDevelopment()
        self.sections = []  # Initialize empty sections list
        self.setup_components()
        self.initialize_sections()  # Now call this after setup_components

    def setup_components(self):
        """Initialize composition components"""
        print("Initializing advanced composition system...")

        # Initialize base systems
        self.fractal_generator = FractalMelodyGenerator(
            MelodyParams(
                mode=self.params.mode,
                emotion=self.params.emotional_journey[0]
            )
        )

        self.harmony_engine = HarmonyEngine(
            HarmonyParams(
                style=HarmonyStyle.CLASSICAL,
                complexity=self.params.development_complexity
            )
        )

        self.genetic_engine = MusicalGenetics(
            GeneticParams(
                target_complexity=self.params.development_complexity
            )
        )

    def initialize_sections(self):
        """Initialize musical sections based on form"""
        print(f"Initializing {self.params.form.value} form sections...")

        if self.params.form == MusicalForm.SONATA:
            self.sections = self.initialize_sonata_form()
        elif self.params.form == MusicalForm.RONDO:
            self.sections = self.initialize_rondo_form()
        elif self.params.form == MusicalForm.THEME_AND_VARIATIONS:
            self.sections = self.initialize_theme_and_variations()
        elif self.params.form == MusicalForm.FUGUE:
            self.sections = self.initialize_fugue()
        else:
            self.sections = self.initialize_through_composed()

        print(f"Initialized {len(self.sections)} sections")

    def initialize_sonata_form(self) -> List[Section]:
        """Initialize sonata form sections (Exposition-Development-Recapitulation)"""
        sections = []
        print("Initializing sonata form...")

        # Generate primary theme motifs first
        primary_motifs = self.generate_primary_theme()
        print(f"Generated {len(primary_motifs)} primary motifs")

        # Store these for later use in development and recapitulation
        self._primary_motifs = primary_motifs  # Save for reference

        # Exposition
        print("Creating exposition section...")
        sections.append(Section(
            motifs=primary_motifs,
            key=self.params.key,
            tempo=self.params.base_tempo,
            emotion=self.params.emotional_journey[0],
            dynamics_base=80,
            transitions=[]
        ))

        # Development
        print("Creating development section...")
        development_motifs = self.generate_development(primary_motifs)  # Pass motifs directly
        sections.append(Section(
            motifs=development_motifs,
            key=self.modulate_key(self.params.key, interval=4),  # Subdominant
            tempo=int(self.params.base_tempo * 1.1),
            emotion=self.params.emotional_journey[1],
            dynamics_base=90,
            transitions=[(0.5, 1.1)]  # Accelerando at midpoint
        ))

        # Recapitulation
        print("Creating recapitulation section...")
        recap_motifs = self.generate_recapitulation(primary_motifs)  # Pass motifs directly
        sections.append(Section(
            motifs=recap_motifs,
            key=self.params.key,
            tempo=self.params.base_tempo,
            emotion=self.params.emotional_journey[-1],
            dynamics_base=85,
            transitions=[(0.8, 0.9)]  # Slight ritardando near end
        ))

        return sections

    def generate_development(self, primary_motifs: List[Motif]) -> List[Motif]:
        """Generate development section motifs"""
        motifs = []
        sequence_intervals = [0, 2, 4, 5, 7]  # Sequence up the scale

        for motif in primary_motifs:  # Use passed motifs instead of accessing sections
            # Create sequences
            motifs.extend(
                self.motif_development.develop_sequence(motif, sequence_intervals)
            )

            # Add transformed versions
            motifs.append(self.motif_development.retrograde(motif))
            motifs.append(self.motif_development.diminish(motif, 0.5))

        return motifs

    def generate_recapitulation(self, primary_motifs: List[Motif]) -> List[Motif]:
        """Generate recapitulation motifs"""
        # Start with primary theme motifs
        motifs = primary_motifs.copy()

        # Add triumphant variations
        for motif in motifs[:]:
            motifs.append(
                Motif(
                    notes=motif.notes,
                    rhythm=motif.rhythm,
                    articulations=[Articulation.MARCATO] * len(motif.notes),
                    dynamics=[min(127, v + 10) for v in motif.dynamics],
                    emotional_intent=self.params.emotional_journey[-1]
                )
            )

        return motifs

    def generate_primary_theme(self) -> List[Motif]:
        """Generate primary theme motifs"""
        motifs = []

        # Generate base motif
        melody, timing, velocities = self.fractal_generator.generate_melody()
        base_motif = Motif(
            notes=melody[:16],  # Use first 16 notes as base motif
            rhythm=timing[:16],
            articulations=[Articulation.LEGATO] * 16,
            dynamics=velocities[:16],
            emotional_intent=self.params.emotional_journey[0]
        )
        motifs.append(base_motif)

        # Generate developments
        motifs.extend([
            self.motif_development.invert(base_motif),
            self.motif_development.augment(base_motif),
            self.motif_development.transpose(base_motif, 7)  # Up a fifth
        ])

        return motifs

    def generate_development(self, primary_motifs: List[Motif]) -> List[Motif]:
        """Generate development section motifs based on primary theme"""
        print(f"Developing {len(primary_motifs)} primary motifs...")
        motifs = []
        sequence_intervals = [0, 2, 4, 5, 7]  # Sequence up the scale

        for motif in primary_motifs:
            # Create sequences
            sequences = self.motif_development.develop_sequence(motif, sequence_intervals)
            motifs.extend(sequences)
            print(f"Generated {len(sequences)} sequences from motif")

            # Add transformed versions
            motifs.append(self.motif_development.retrograde(motif))
            motifs.append(self.motif_development.diminish(motif, 0.5))
            print("Added retrograde and diminished versions")

        print(f"Total development motifs generated: {len(motifs)}")
        return motifs

    def generate_recapitulation(self, primary_motifs: List[Motif]) -> List[Motif]:
        """Generate recapitulation motifs based on primary theme"""
        print("Generating recapitulation...")

        # Start with primary theme motifs
        motifs = primary_motifs.copy()
        print(f"Starting with {len(motifs)} primary motifs")

        # Add triumphant variations
        original_count = len(motifs)
        for i, motif in enumerate(motifs[:original_count]):
            motifs.append(
                Motif(
                    notes=motif.notes,
                    rhythm=motif.rhythm,
                    articulations=[Articulation.MARCATO] * len(motif.notes),
                    dynamics=[min(127, v + 10) for v in motif.dynamics],
                    emotional_intent=self.params.emotional_journey[-1]
                )
            )
            print(f"Added triumphant variation of motif {i + 1}")

        print(f"Total recapitulation motifs: {len(motifs)}")
        return motifs

    def modulate_key(self, base_key: str, interval: int) -> str:
        """Modulate to a new key"""
        key_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                  'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        rev_map = {v: k for k, v in key_map.items()}

        base_value = key_map[base_key]
        new_value = (base_value + interval) % 12
        return rev_map[new_value]

    def apply_voice_leading(self, voices: List[List[int]]) -> List[List[int]]:
        """Apply voice leading rules to all voices"""
        optimized_voices = []

        for voice_idx, voice in enumerate(voices):
            if voice_idx == 0:
                optimized_voices.append(voice)  # Keep melody unchanged
                continue

            # Optimize each subsequent voice
            optimized_voice = []
            for i in range(len(voice)):
                if i == 0:
                    optimized_voice.append(voice[i])
                    continue

                # Find best voice leading for this note
                current_chord = [v[i-1] for v in optimized_voices]
                next_options = self.voice_leading.generate_voicings([voice[i]])
                best_note = min(next_options,
                              key=lambda n: abs(n[0] - voice[i-1]))
                optimized_voice.append(best_note[0])

            optimized_voices.append(optimized_voice)

        return optimized_voices

    def add_cadence(self, motif: Motif, cadence_type: str) -> Motif:
        """Add a cadence to a motif"""
        cadence = Cadence.get_cadence(cadence_type, motif.notes[0] % 12)

        # Add cadence notes
        motif.notes.extend([note for chord in cadence for note in chord])
        motif.rhythm.extend([self.params.duration] * len(cadence) * 3)
        motif.articulations.extend([Articulation.TENUTO] * len(cadence) * 3)
        motif.dynamics.extend([90] * len(cadence) * 3)

        return motif

    def generate_composition(self) -> Tuple[MIDIFile, Optional[np.ndarray]]:
        """Generate complete musical composition"""
        print("\nGenerating advanced musical composition...")
        print(f"Form: {self.params.form.value}")

        midi = MIDIFile(self.params.voice_count)
        current_time = 0

        # Process each section
        for section_idx, section in enumerate(self.sections):
            print(f"\nProcessing {section_idx + 1}/{len(self.sections)}: "
                  f"Key: {section.key}, Tempo: {section.tempo}")

            # Set tempo for section
            midi.addTempo(0, current_time, section.tempo)

            # Process each motif in the section
            for motif_idx, motif in enumerate(section.motifs):
                print(f"Processing motif {motif_idx + 1}/{len(section.motifs)}")

                # Generate harmony for motif
                harmony = self.harmony_engine.generate_harmonic_progression()

                # Apply voice leading
                voices = [motif.notes]
                voices.extend([[note[i] for note in harmony]
                             for i in range(min(3, len(harmony[0])))])
                voices = self.apply_voice_leading(voices)

                # Add to MIDI file
                for voice_idx, voice in enumerate(voices):
                    track = voice_idx
                    midi.addTrackName(track, current_time, f"Voice {voice_idx + 1}")

                    for note_idx, (note, rhythm, dynamic, articulation) in enumerate(
                        zip(voice, motif.rhythm, motif.dynamics, motif.articulations)
                    ):
                        # Apply articulation
                        if articulation == Articulation.STACCATO:
                            duration = rhythm * 0.5
                        elif articulation == Articulation.LEGATO:
                            duration = rhythm * 1.1
                        else:
                            duration = rhythm

                        # Apply any tempo/dynamic transitions
                        for trans_time, trans_value in section.transitions:
                            if note_idx / len(voice) >= trans_time:
                                if isinstance(trans_value, float):  # Tempo change
                                    rhythm *= trans_value
                                else:  # Dynamic change
                                    dynamic = int(dynamic * trans_value)

                        midi.addNote(track, track, note, current_time,
                                   duration, dynamic)

                current_time += sum(motif.rhythm)

        # Generate waveform if requested
        waveform = None
        if self.params.generate_wav:
            print("\nGenerating audio waveform...")
            waveform = self.generate_waveform(midi)

        return midi, waveform

    def generate_waveform(self, midi: MIDIFile) -> np.ndarray:
        """Generate audio waveform from MIDI"""
        # Implementation similar to previous version
        # but with added support for articulations and dynamics
        pass  # Placeholder for actual implementation

    def save_composition(self, midi: MIDIFile, audio: Optional[np.ndarray],
                        base_filename: str = "advanced_composition"):
        """Save composition as MIDI and optionally WAV files"""
        print(f"\nSaving composition to {base_filename}.*...")

        # Save MIDI
        with open(f"{base_filename}.mid", "wb") as f:
            midi.writeFile(f)
        print(f"✓ Saved MIDI file: {base_filename}.mid")

        # Save WAV if available
        if audio is not None:
            from scipy.io import wavfile
            wavfile.write(f"{base_filename}.wav", 44100,
                         (audio * 32767).astype(np.int16))
            print(f"✓ Saved WAV file: {base_filename}.wav")


if __name__ == "__main__":
    print("Advanced Musical Composition System - Starting composition process...")

    # Create composer with sophisticated settings
    params = AdvancedCompositionParams(
        form=MusicalForm.SONATA,
        base_tempo=120,
        key="C",
        mode=MusicalMode.IONIAN,
        emotional_journey=[
            EmotionalStyle.PEACEFUL,      # Exposition
            EmotionalStyle.ENERGETIC,     # Development
            EmotionalStyle.MELANCHOLIC,   # Development climax
            EmotionalStyle.JOYFUL         # Recapitulation
        ],
        development_complexity=0.8,
        contrapuntal_density=0.7,
        modulation_frequency=0.4,
        section_count=3,
        voice_count=4,
        generate_wav=False
    )

    # Initialize composer
    composer = AdvancedComposer(params)

    # Generate composition
    print("\nGenerating composition...")
    midi, audio = composer.generate_composition()

    # Save outputs
    print("\nSaving files...")
    composer.save_composition(midi, audio)

    print("\nComposition complete!")
