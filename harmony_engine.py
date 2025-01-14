import numpy as np
import matplotlib.pyplot as plt
from midiutil import MIDIFile
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from scipy.io import wavfile
import colorsys

class HarmonyStyle(Enum):
    CLASSICAL = "classical"
    JAZZ = "jazz"
    MODAL = "modal"
    IMPRESSIONIST = "impressionist"
    ROMANTIC = "romantic"
    BAROQUE = "baroque"

class VoiceType(Enum):
    SOPRANO = "soprano"
    ALTO = "alto"
    TENOR = "tenor"
    BASS = "bass"

@dataclass
class VoiceRange:
    min_note: int
    max_note: int
    preferred_min: int
    preferred_max: int

@dataclass
class HarmonyParams:
    style: HarmonyStyle = HarmonyStyle.CLASSICAL
    tempo: int = 120
    key: str = 'C'
    mode: str = 'major'
    num_voices: int = 4
    complexity: float = 0.5  # 0.0 to 1.0
    tension: float = 0.5     # 0.0 to 1.0
    duration: float = 240    # in seconds

class HarmonyEngine:
    def __init__(self, params: HarmonyParams):
        self.params = params
        self.setup_theory()
        self.setup_voice_ranges()
        self.setup_progression_rules()
        self.setup_voice_leading_rules()

    def setup_theory(self):
        """Initialize music theory fundamentals"""
        # Scales and modes
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'natural_minor': [0, 2, 3, 5, 7, 8, 10],
            'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
            'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'phrygian': [0, 1, 3, 5, 7, 8, 10],
            'lydian': [0, 2, 4, 6, 7, 9, 11],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'locrian': [0, 1, 3, 5, 6, 8, 10],
            'whole_tone': [0, 2, 4, 6, 8, 10],
            'diminished': [0, 2, 3, 5, 6, 8, 9, 11]
        }

        # Chord types with root position and inversions
        self.chord_types = {
            'major': ([0, 4, 7], 'M'),
            'minor': ([0, 3, 7], 'm'),
            'diminished': ([0, 3, 6], 'dim'),
            'augmented': ([0, 4, 8], 'aug'),
            'major7': ([0, 4, 7, 11], 'M7'),
            'minor7': ([0, 3, 7, 10], 'm7'),
            'dominant7': ([0, 4, 7, 10], '7'),
            'half_diminished7': ([0, 3, 6, 10], 'ø7'),
            'diminished7': ([0, 3, 6, 9], 'o7'),
            'major9': ([0, 4, 7, 11, 14], 'M9'),
            'minor9': ([0, 3, 7, 10, 14], 'm9'),
            'dominant9': ([0, 4, 7, 10, 14], '9')
        }

        # Note to MIDI number mapping
        self.note_to_midi = {
            'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63,
            'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68,
            'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71
        }

    def setup_voice_ranges(self):
        """Set up vocal ranges for each voice type"""
        self.voice_ranges = {
            VoiceType.SOPRANO: VoiceRange(60, 79, 62, 77),  # Middle C to G5
            VoiceType.ALTO: VoiceRange(55, 74, 57, 72),     # G3 to D5
            VoiceType.TENOR: VoiceRange(48, 67, 50, 65),    # C3 to G4
            VoiceType.BASS: VoiceRange(40, 62, 42, 60)      # E2 to D4
        }

    def setup_progression_rules(self):
        """Initialize harmonic progression rules for different styles"""
        # Classical progression probabilities
        self.classical_progressions = {
            'I': {'IV': 0.3, 'V': 0.4, 'vi': 0.2, 'ii': 0.1},
            'ii': {'V': 0.7, 'IV': 0.2, 'vii°': 0.1},
            'iii': {'vi': 0.4, 'IV': 0.3, 'ii': 0.3},
            'IV': {'V': 0.4, 'I': 0.3, 'ii': 0.2, 'vii°': 0.1},
            'V': {'I': 0.6, 'vi': 0.2, 'iii': 0.1, 'IV': 0.1},
            'vi': {'ii': 0.3, 'IV': 0.3, 'V': 0.2, 'iii': 0.2},
            'vii°': {'I': 0.5, 'V': 0.5}
        }

        # Jazz progression probabilities
        self.jazz_progressions = {
            'I': {'iv': 0.2, 'V7': 0.3, 'ii7': 0.3, 'vi7': 0.2},
            'ii7': {'V7': 0.6, 'iii7': 0.2, 'vi7': 0.2},
            'iii7': {'vi7': 0.4, 'ii7': 0.3, 'V7': 0.3},
            'IV7': {'V7': 0.4, 'I': 0.3, 'ii7': 0.3},
            'V7': {'I': 0.5, 'iii7': 0.2, 'vi7': 0.3},
            'vi7': {'ii7': 0.4, 'V7': 0.3, 'iii7': 0.3},
            'vii°7': {'I': 0.6, 'V7': 0.4}
        }

        # Modal progression tendencies
        self.modal_progressions = {
            'dorian': ['i7', 'IV7', 'v7', 'VII'],
            'phrygian': ['i', 'II', 'vii', 'IV'],
            'lydian': ['I', 'II', 'vii', 'VII'],
            'mixolydian': ['I7', 'VII', 'v7', 'IV'],
            'aeolian': ['i', 'VI', 'VII', 'v'],
            'locrian': ['i°', 'II', 'VI', 'v']
        }

    def setup_voice_leading_rules(self):
        """Initialize voice leading rules"""
        self.voice_leading_rules = {
            'max_leap': 12,  # Maximum interval jump in semitones
            'preferred_max_leap': 7,  # Preferred maximum interval
            'contrary_motion_weight': 0.6,  # Preference for contrary motion
            'parallel_fifth_penalty': 1.0,  # Penalty for parallel fifths
            'parallel_octave_penalty': 1.0,  # Penalty for parallel octaves
            'voice_crossing_penalty': 0.8,  # Penalty for voice crossing
            'spacing_penalty': 0.5  # Penalty for large gaps between voices
        }

    def generate_harmonic_progression(self, melody: Optional[List[int]] = None) -> List[List[int]]:
        """Generate a harmonic progression based on style and optional melody"""
        if self.params.style == HarmonyStyle.CLASSICAL:
            progression = self._generate_classical_progression()
        elif self.params.style == HarmonyStyle.JAZZ:
            progression = self._generate_jazz_progression()
        elif self.params.style == HarmonyStyle.MODAL:
            progression = self._generate_modal_progression()
        else:
            progression = self._generate_classical_progression()  # Default

        # If melody is provided, adjust harmony to fit
        if melody is not None:
            progression = self._adjust_harmony_to_melody(progression, melody)

        return progression

    def _generate_classical_progression(self) -> List[List[int]]:
        """Generate a classical chord progression"""
        progression = []
        current_chord = 'I'
        length = 8  # Number of chords to generate

        for _ in range(length):
            # Get current chord notes
            chord_notes = self._get_chord_notes(current_chord)
            progression.append(chord_notes)

            # Choose next chord based on probabilities
            next_options = self.classical_progressions[current_chord]
            next_chord = np.random.choice(
                list(next_options.keys()),
                p=list(next_options.values())
            )
            current_chord = next_chord

        return progression

    def _generate_jazz_progression(self) -> List[List[int]]:
        """Generate a jazz chord progression"""
        progression = []
        current_chord = 'ii7'  # Start with ii-V-I
        length = 8

        for _ in range(length):
            # Get extended chord notes
            chord_notes = self._get_chord_notes(current_chord, extended=True)
            progression.append(chord_notes)

            # Choose next chord based on jazz progression probabilities
            next_options = self.jazz_progressions[current_chord]
            next_chord = np.random.choice(
                list(next_options.keys()),
                p=list(next_options.values())
            )
            current_chord = next_chord

        return progression

    def _generate_modal_progression(self) -> List[List[int]]:
        """Generate a modal progression"""
        mode = self.params.mode
        if mode not in self.modal_progressions:
            mode = 'dorian'  # Default to dorian if mode not found

        progression = []
        chord_sequence = self.modal_progressions[mode]

        # Repeat the modal pattern
        repeats = 2
        for _ in range(repeats):
            for chord in chord_sequence:
                chord_notes = self._get_chord_notes(chord, modal=True)
                progression.append(chord_notes)

        return progression

    def _get_chord_notes(self, chord_symbol: str, extended: bool = False,
                        modal: bool = False) -> List[int]:
        """Convert chord symbol to MIDI note numbers"""
        # Roman numeral to scale degree mapping
        roman_to_scale = {
            'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6,
            'i': 0, 'ii': 1, 'iii': 2, 'iv': 3, 'v': 4, 'vi': 5, 'vii': 6
        }

        # Parse chord symbol
        root_numeral = ''.join(c for c in chord_symbol if c.isalpha() or c in ['i', 'I', 'v', 'V'])
        quality = ''.join(c for c in chord_symbol if not (c.isalpha() or c in ['i', 'I', 'v', 'V']))

        # Get scale degree
        scale_degree = roman_to_scale[root_numeral]

        # Get base note from key and scale
        key_note = self.note_to_midi[self.params.key]
        scale = self.scales[self.params.mode]
        base_note = key_note + scale[scale_degree]

        # Determine chord type based on numeral case and quality
        if root_numeral.isupper():
            chord_type = 'major'
        else:
            chord_type = 'minor'

        # Override with specific qualities
        if '°' in quality or 'dim' in quality:
            chord_type = 'diminished'
        elif '+' in quality or 'aug' in quality:
            chord_type = 'augmented'
        elif '7' in quality:
            if root_numeral.isupper():
                chord_type = 'dominant7' if not extended else 'dominant9'
            else:
                chord_type = 'minor7' if not extended else 'minor9'

        intervals = self.chord_types[chord_type][0]
        return [base_note + interval for interval in intervals]

    def _adjust_harmony_to_melody(self, progression: List[List[int]],
                                melody: List[int]) -> List[List[int]]:
        """Adjust harmony to better fit the melody"""
        adjusted_progression = []

        for chord, melody_note in zip(progression, melody):
            # Find nearest chord voicing that includes melody note
            chord_root = chord[0] % 12
            melody_pitch_class = melody_note % 12

            if melody_pitch_class not in [note % 12 for note in chord]:
                # Adjust chord to include melody note
                new_chord = chord.copy()
                new_chord[-1] = melody_note  # Replace top note with melody
                adjusted_progression.append(new_chord)
            else:
                adjusted_progression.append(chord)

        return adjusted_progression

    def apply_voice_leading(self, progression: List[List[int]]) -> List[List[List[int]]]:
        """Apply voice leading rules to create smooth voice movement"""
        voice_led = []
        current_voicing = None

        for chord in progression:
            if current_voicing is None:
                # For first chord, just spread it across voices
                current_voicing = self._spread_chord(chord)
            else:
                # Find best voice leading to next chord
                current_voicing = self._optimize_voice_leading(current_voicing, chord)

            voice_led.append(current_voicing)

        return voice_led

    def _spread_chord(self, chord: List[int]) -> List[int]:
        """Spread chord tones across voice ranges"""
        num_voices = len(self.voice_ranges)
        if len(chord) < num_voices:
            # Double some notes if needed
            chord = chord + chord[:num_voices - len(chord)]

        # Sort notes from low to high
        chord = sorted(chord)

        # Assign to voices based on ranges
        voice_assignments = []
        for voice_type, voice_range in self.voice_ranges.items():
            # Find nearest note in chord that fits voice range
            suitable_notes = [note for note in chord
                            if voice_range.min_note <= note <= voice_range.max_note]
            if suitable_notes:
                note = min(suitable_notes, key=lambda x: abs(x - (voice_range.preferred_min +
                                                                voice_range.preferred_max) / 2))
                voice_assignments.append(note)
                chord.remove(note)
            else:
                # If no suitable note found, transpose until fit
                while chord[0] < voice_range.min_note:
                    chord[0] += 12
                while chord[0] > voice_range.max_note:
                    chord[0] -= 12
                voice_assignments.append(chord.pop(0))

        return voice_assignments

    def _optimize_voice_leading(self, current: List[int],
                              target_chord: List[int]) -> List[int]:
        """Find optimal voice leading between two chords"""
        possible_voicings = []
        scores = []

        # Generate possible voicings of target chord
        base_voicing = self._spread_chord(target_chord)
        possible_voicings.append(base_voicing)

        # Try octave displacements
        for voice in range(len(base_voicing)):
            up = base_voicing.copy()
            up[voice] += 12
            down = base_voicing.copy()
            down[voice] -= 12
            possible_voicings.extend([up, down])

        # Score each possibility
        for voicing in possible_voicings:
            score = self._score_voice_leading(current, voicing)
            scores.append(score)

        # Return voicing with best score
        best_idx = np.argmax(scores)
        return possible_voicings[best_idx]

    def _score_voice_leading(self, current: List[int], target: List[int]) -> float:
        """Score voice leading transition based on rules"""
        score = 0.0

        # Check voice crossing
        if any(t1 <= t2 for t1, t2 in zip(target[1:], target[:-1])):
            score -= self.voice_leading_rules['voice_crossing_penalty']

        # Calculate intervals and motions
        intervals = [abs(t - c) for c, t in zip(current, target)]

        # Penalize large leaps
        for interval in intervals:
            if interval > self.voice_leading_rules['max_leap']:
                score -= 1.0
            elif interval > self.voice_leading_rules['preferred_max_leap']:
                score -= 0.5

        # Check for parallel fifths and octaves
        for i in range(len(current) - 1):
            for j in range(i + 1, len(current)):
                current_interval = abs(current[i] - current[j]) % 12
                target_interval = abs(target[i] - target[j]) % 12

                if current_interval == target_interval:
                    if current_interval == 7:  # fifth
                        score -= self.voice_leading_rules['parallel_fifth_penalty']
                    elif current_interval == 0:  # octave
                        score -= self.voice_leading_rules['parallel_octave_penalty']

        # Reward contrary motion
        motions = [np.sign(t - c) for c, t in zip(current, target)]
        contrary_motions = sum(1 for m1, m2 in zip(motions[:-1], motions[1:])
                             if m1 != m2)
        score += contrary_motions * self.voice_leading_rules['contrary_motion_weight']

        return score

    def visualize_progression(self, progression: List[List[List[int]]],
                            filename: str = "harmony_visualization.png"):
        """Create visualization of harmonic progression and voice leading"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Voice leading graph
        times = np.arange(len(progression))
        colors = ['b', 'g', 'r', 'c']  # Different color for each voice

        for voice in range(len(progression[0])):
            voice_line = [chord[voice] for chord in progression]
            ax1.plot(times, voice_line, f'{colors[voice]}-o',
                    label=f'Voice {voice + 1}', alpha=0.7)

        ax1.set_title('Voice Leading Progression')
        ax1.set_xlabel('Chord Number')
        ax1.set_ylabel('MIDI Note')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Harmony heat map
        chord_grid = np.zeros((128, len(progression)))
        for t, chord in enumerate(progression):
            for note in chord:
                chord_grid[note, t] = 1

        ax2.imshow(chord_grid, aspect='auto', cmap='Blues',
                  origin='lower', interpolation='nearest')
        ax2.set_title('Harmony Heat Map')
        ax2.set_xlabel('Chord Number')
        ax2.set_ylabel('MIDI Note')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def create_midi(self, progression: List[List[List[int]]],
                   filename: str = "harmony.mid"):
        """Create MIDI file from progression"""
        midi = MIDIFile(len(progression[0]))  # One track per voice

        # Add tracks
        for voice in range(len(progression[0])):
            track = voice
            time = 0
            midi.addTrackName(track, time, f"Voice {voice + 1}")
            midi.addTempo(track, time, self.params.tempo)

            # Add notes for this voice
            for chord in progression:
                note = chord[voice]
                midi.addNote(track, channel=0, pitch=note,
                           time=time, duration=1.0, volume=100)
                time += 1

        # Save MIDI file
        with open(filename, "wb") as f:
            midi.writeFile(f)

    def generate_accompaniment(self, melody: List[int]) -> Tuple[List[List[List[int]]], MIDIFile]:
        """Generate full harmonic accompaniment for a melody"""
        # Generate basic progression
        raw_progression = self.generate_harmonic_progression(melody)

        # Apply voice leading
        voiced_progression = self.apply_voice_leading(raw_progression)

        # Create MIDI
        midi = MIDIFile(len(voiced_progression[0]) + 1)  # +1 for melody

        # Add melody track
        track = 0
        time = 0
        midi.addTrackName(track, time, "Melody")
        midi.addTempo(track, time, self.params.tempo)

        for note in melody:
            midi.addNote(track, channel=0, pitch=note,
                       time=time, duration=1.0, volume=100)
            time += 1

        # Add harmony tracks
        for voice in range(len(voiced_progression[0])):
            track = voice + 1
            time = 0
            midi.addTrackName(track, time, f"Harmony Voice {voice + 1}")

            for chord in voiced_progression:
                note = chord[voice]
                midi.addNote(track, channel=0, pitch=note,
                           time=time, duration=1.0, volume=80)
                time += 1

        return voiced_progression, midi

if __name__ == "__main__":
    # Example usage
    params = HarmonyParams(
        style=HarmonyStyle.CLASSICAL,
        key='C',
        mode='major'
    )

    engine = HarmonyEngine(params)

    # Generate progression without melody
    print("Generating harmonic progression...")
    progression = engine.generate_harmonic_progression()
    voiced_progression = engine.apply_voice_leading(progression)

    # Create visualizations and MIDI
    print("Creating visualizations and MIDI file...")
    engine.visualize_progression(voiced_progression)
    engine.create_midi(voiced_progression)

    print("\nGenerated files:")
    print("- harmony_visualization.png (Voice leading and harmony visualization)")
    print("- harmony.mid (MIDI file of progression)")

    # Example with melody
    print("\nGenerating accompaniment for sample melody...")
    sample_melody = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
    voiced_prog, midi = engine.generate_accompaniment(sample_melody)

    # Save accompaniment
    with open("accompaniment.mid", "wb") as f:
        midi.writeFile(f)

    engine.visualize_progression(voiced_prog, "accompaniment_visualization.png")

    print("\nGenerated additional files:")
    print("- accompaniment.mid (MIDI file with melody and harmony)")
    print("- accompaniment_visualization.png (Visualization of accompaniment)")
    print("\nComplete! You can now listen to the generated harmonies.")
