import numpy as np
import matplotlib.pyplot as plt
from midiutil import MIDIFile
from scipy.signal import savgol_filter
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import colorsys
from scipy.io import wavfile
import random
from enum import Enum

class FractalType(Enum):
    MANDELBROT = "mandelbrot"
    KOCH = "koch"
    SIERPINSKI = "sierpinski"
    JULIA = "julia"
    LSYSTEM = "lsystem"

@dataclass
class MultiFractalParams:
    melody_fractal: FractalType = FractalType.MANDELBROT
    harmony_fractal: FractalType = FractalType.KOCH
    rhythm_fractal: FractalType = FractalType.SIERPINSKI
    timbre_fractal: FractalType = FractalType.JULIA
    form_fractal: FractalType = FractalType.LSYSTEM
    base_note: int = 60  # Middle C
    tempo: int = 120
    resolution: int = 200
    duration: float = 0.25
    octave_range: int = 2

class MultiFractalMusic:
    def __init__(self, params: MultiFractalParams):
        self.params = params
        self.setup_scales()
        self.setup_harmonies()

    def setup_scales(self):
        """Initialize musical scales"""
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'pentatonic': [0, 2, 4, 7, 9],
            'whole_tone': [0, 2, 4, 6, 8, 10],
            'chromatic': list(range(12))
        }

    def setup_harmonies(self):
        """Initialize harmony patterns"""
        self.chord_types = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'diminished': [0, 3, 6],
            'augmented': [0, 4, 8],
            'major7': [0, 4, 7, 11],
            'minor7': [0, 3, 7, 10]
        }

    def generate_mandelbrot(self) -> np.ndarray:
        """Generate Mandelbrot set pattern"""
        h = w = self.params.resolution
        y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
        c = x + y*1j
        z = c
        divtime = np.zeros(z.shape, dtype=int)

        for i in range(100):  # max iterations
            z = z**2 + c
            diverge = z*np.conj(z) > 2**2
            div_now = diverge & (divtime == 0)
            divtime[div_now] = i
            z[diverge] = 2

        # Normalize to [0, 1]
        return (divtime - divtime.min()) / (divtime.max() - divtime.min())

    def generate_mandelbrot_melody(self) -> List[int]:
        """Generate melody using Mandelbrot set"""
        fractal = self.generate_mandelbrot()
        melody = self.map_to_scale(fractal, 'major')
        return melody

    def generate_koch_harmony(self) -> List[List[int]]:
        """Generate harmonic progression using Koch curve"""
        iterations = 4
        harmony = self.koch_curve(iterations)
        chord_progression = self.map_to_chords(harmony)
        return chord_progression

    def generate_sierpinski_rhythm(self) -> List[float]:
        """Generate rhythmic pattern using Sierpinski triangle"""
        depth = 5
        rhythm = self.sierpinski_pattern(depth)
        durations = self.map_to_durations(rhythm)
        return durations

    def generate_julia_timbre(self) -> List[float]:
        """Generate timbral evolution using Julia set"""
        timbre = self.julia_set()
        modulation = self.map_to_modulation(timbre)
        return modulation

    def generate_lsystem_form(self) -> List[str]:
        """Generate musical form using L-System"""
        rules = {
            'A': 'ABA',
            'B': 'BCB',
            'C': 'CDC'
        }
        iterations = 3
        form = self.l_system(rules, 'A', iterations)
        return form

    def koch_curve(self, iterations: int) -> np.ndarray:
        """Generate Koch curve pattern"""
        curve = np.array([[0, 0], [1, 0]])
        for _ in range(iterations):
            new_curve = []
            for i in range(len(curve) - 1):
                p1 = curve[i]
                p2 = curve[i + 1]
                v = p2 - p1
                p3 = p1 + v/3
                p4 = p3 + np.array([-v[1], v[0]]) * np.sqrt(3)/6
                p5 = p1 + 2*v/3
                new_curve.extend([p1, p3, p4, p5])
            new_curve.append(curve[-1])
            curve = np.array(new_curve)
        return curve

    def sierpinski_pattern(self, depth: int) -> List[List[bool]]:
        """Generate Sierpinski triangle pattern"""
        size = 2**depth
        pattern = np.zeros((size, size), dtype=bool)
        pattern[0, 0] = True

        for d in range(depth):
            step = 2**d
            for i in range(0, size, 2*step):
                for j in range(0, size, 2*step):
                    if pattern[i, j]:
                        pattern[i+step, j] = True
                        pattern[i, j+step] = True
        return pattern

    def julia_set(self, c: complex = -0.4 + 0.6j) -> np.ndarray:
        """Generate Julia set pattern"""
        x = np.linspace(-1.5, 1.5, self.params.resolution)
        y = np.linspace(-1.5, 1.5, self.params.resolution)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        output = np.zeros(Z.shape)
        for i in range(100):
            mask = np.abs(Z) <= 2
            output[mask] = i
            Z[mask] = Z[mask]**2 + c

        return output

    def l_system(self, rules: Dict[str, str], axiom: str, iterations: int) -> str:
        """Generate L-System pattern"""
        result = axiom
        for _ in range(iterations):
            new_result = ""
            for char in result:
                new_result += rules.get(char, char)
            result = new_result
        return result

    def map_to_scale(self, fractal: np.ndarray, scale_name: str) -> List[int]:
        """Map fractal values to musical scale"""
        scale = self.scales[scale_name]
        mapped = []

        for value in fractal.flatten():
            scale_index = int(value * len(scale)) % len(scale)
            octave = int(value * self.params.octave_range)
            note = self.params.base_note + scale[scale_index] + (12 * octave)
            mapped.append(note)

        return mapped

    def map_to_chords(self, pattern: np.ndarray) -> List[List[int]]:
        """Map pattern to chord progression"""
        progression = []

        for value in pattern:
            chord_type = random.choice(list(self.chord_types.keys()))
            root = int(value[0] * 12) + self.params.base_note
            chord = [root + interval for interval in self.chord_types[chord_type]]
            progression.append(chord)

        return progression

    def map_to_durations(self, pattern: np.ndarray) -> List[float]:
        """Map pattern to note durations"""
        base_duration = self.params.duration
        durations = []

        for row in pattern:
            for value in row:
                if value:
                    durations.append(base_duration)
                else:
                    durations.append(base_duration * 0.5)

        return durations

    def map_to_modulation(self, pattern: np.ndarray) -> List[float]:
        """Map pattern to timbral modulation"""
        return (pattern - pattern.min()) / (pattern.max() - pattern.min())

    def visualize_all_patterns(self, filename: str = "fractal_patterns.png"):
        """Create comprehensive visualization of all fractal patterns"""
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2, figure=fig)

        # 1. Mandelbrot Set (Melody)
        ax1 = fig.add_subplot(gs[0, 0])
        mandel = self.generate_mandelbrot()
        im1 = ax1.imshow(mandel, cmap='magma', extent=[-2, 0.8, -1.4, 1.4])
        ax1.set_title('Mandelbrot Set (Melody Generation)')
        plt.colorbar(im1, ax=ax1, label='Pitch Mapping')

        # 2. Koch Curve (Harmony)
        ax2 = fig.add_subplot(gs[0, 1])
        koch = self.koch_curve(4)
        ax2.plot(koch[:, 0], koch[:, 1], 'b-', linewidth=1)
        ax2.set_title('Koch Curve (Harmonic Progression)')
        ax2.set_aspect('equal')

        # 3. Sierpinski Triangle (Rhythm)
        ax3 = fig.add_subplot(gs[1, 0])
        sierp = self.sierpinski_pattern(7)
        ax3.imshow(sierp, cmap='Blues')
        ax3.set_title('Sierpinski Triangle (Rhythmic Pattern)')

        # 4. Julia Set (Timbre)
        ax4 = fig.add_subplot(gs[1, 1])
        julia = self.julia_set()
        im4 = ax4.imshow(julia, cmap='viridis', extent=[-1.5, 1.5, -1.5, 1.5])
        ax4.set_title('Julia Set (Timbral Evolution)')
        plt.colorbar(im4, ax=ax4, label='Timbre Modulation')

        # 5. L-System Visualization (Form)
        ax5 = fig.add_subplot(gs[2, :])
        self.visualize_lsystem(ax5)
        ax5.set_title('L-System (Musical Form)')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_lsystem(self, ax):
        """Visualize L-System as a tree structure"""
        form = self.generate_lsystem_form()

        def draw_branch(pos, angle, length, depth):
            if depth == 0:
                return

            # Calculate end point
            end_x = pos[0] + length * np.cos(np.radians(angle))
            end_y = pos[1] + length * np.sin(np.radians(angle))

            # Draw branch
            ax.plot([pos[0], end_x], [pos[1], end_y],
                   color=plt.cm.viridis(depth/5), linewidth=2-depth*0.3)

            # Recurse with branches
            new_pos = (end_x, end_y)
            new_length = length * 0.7

            draw_branch(new_pos, angle - 30, new_length, depth - 1)
            draw_branch(new_pos, angle + 30, new_length, depth - 1)

        # Start the recursive drawing
        ax.set_aspect('equal')
        draw_branch((0.5, 0), 90, 0.3, 5)
        ax.set_xticks([])
        ax.set_yticks([])

    def create_composition(self) -> Tuple[MIDIFile, np.ndarray]:
        """Create complete musical composition"""
        # Generate components using different fractals
        melody = self.generate_mandelbrot_melody()
        harmony = self.generate_koch_harmony()
        rhythm = self.generate_sierpinski_rhythm()
        timbre = self.generate_julia_timbre()
        form = self.generate_lsystem_form()

        # Create MIDI file
        midi = MIDIFile(2)  # 2 tracks - melody and harmony

        # Add melody
        track = 0
        time = 0
        midi.addTempo(track, time, self.params.tempo)

        for note, duration in zip(melody, rhythm):
            midi.addNote(track, 0, note, time, duration, 100)
            time += duration

        # Add harmony
        track = 1
        time = 0
        for chord, duration in zip(harmony, rhythm):
            for note in chord:
                midi.addNote(track, 0, note, time, duration, 80)
            time += duration

        # Generate audio waveform
        sample_rate = 44100
        duration = sum(rhythm)
        t = np.linspace(0, duration, int(sample_rate * duration))
        waveform = np.zeros_like(t)

        # Combine melody and harmony with timbral evolution
        current_time = 0
        for note, chord, dur, mod in zip(melody, harmony, rhythm, timbre.flatten()):
            freq = 440 * (2 ** ((note - 69) / 12))
            chord_freqs = [440 * (2 ** ((n - 69) / 12)) for n in chord]

            samples = int(dur * sample_rate)
            t_note = np.linspace(0, dur, samples)

            # Generate harmonically rich waveform
            wave = np.sin(2 * np.pi * freq * t_note)
            for i, f in enumerate(chord_freqs):
                wave += 0.5 * np.sin(2 * np.pi * f * t_note) * (mod * 0.5)

            # Apply envelope
            envelope = np.exp(-3 * t_note / dur)
            wave *= envelope

            idx = int(current_time * sample_rate)
            waveform[idx:idx+samples] += wave
            current_time += dur

        # Normalize waveform
        waveform = waveform / np.max(np.abs(waveform))

        # Create visualizations
        self.visualize_all_patterns("fractal_patterns.png")

        # Create waveform visualization
        plt.figure(figsize=(15, 5))

        # Plot waveform
        plt.subplot(211)
        time = np.linspace(0, len(waveform)/44100, len(waveform))
        plt.plot(time, waveform, 'b-', alpha=0.7)
        plt.title('Generated Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        # Plot spectrogram
        plt.subplot(212)
        plt.specgram(waveform, NFFT=2048, Fs=44100,
                    noverlap=1024, cmap='inferno')
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        plt.tight_layout()
        plt.savefig("waveform_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        return midi, waveform

if __name__ == "__main__":
    # Create generator with default parameters
    params = MultiFractalParams()
    generator = MultiFractalMusic(params)

    # Generate composition
    print("Generating fractal music composition...")
    midi, waveform = generator.create_composition()

    # Save files
    print("\nSaving files...")
    with open("multi_fractal_composition.mid", "wb") as f:
        midi.writeFile(f)

    wavfile.write("multi_fractal_composition.wav", 44100,
                 (waveform * 32767).astype(np.int16))

    print("\nGenerated files:")
    print("- multi_fractal_composition.mid (MIDI file)")
    print("- multi_fractal_composition.wav (Audio file)")
    print("- fractal_patterns.png (Visualization of fractal patterns)")
    print("- waveform_analysis.png (Waveform and spectrogram analysis)")
    print("\nComplete! You can now listen to the composition and examine the visualizations.")
