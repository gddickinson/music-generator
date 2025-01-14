import numpy as np
import matplotlib.pyplot as plt
from midiutil import MIDIFile
from scipy.signal import savgol_filter
from typing import List, Tuple, Dict, Optional
from enum import Enum
from dataclasses import dataclass
import colorsys
from scipy.io import wavfile
import random

class MusicalMode(Enum):
    IONIAN = "ionian"          # Traditional major scale
    DORIAN = "dorian"         # Minor with raised 6th
    PHRYGIAN = "phrygian"     # Minor with lowered 2nd
    LYDIAN = "lydian"         # Major with raised 4th
    MIXOLYDIAN = "mixolydian" # Major with lowered 7th
    AEOLIAN = "aeolian"       # Natural minor
    LOCRIAN = "locrian"       # Minor with lowered 2nd and 5th

class EmotionalStyle(Enum):
    ETHEREAL = "ethereal"
    DARK = "dark"
    JOYFUL = "joyful"
    MELANCHOLIC = "melancholic"
    ENERGETIC = "energetic"
    PEACEFUL = "peaceful"

@dataclass
class MelodyParams:
    mode: MusicalMode
    emotion: EmotionalStyle
    base_note: int = 60  # Middle C
    tempo: int = 120
    resolution: int = 200
    duration: float = 0.25
    octave_range: int = 2

class FractalMelodyGenerator:
    def __init__(self, params: MelodyParams):
        self.params = params
        self.setup_scales()
        self.setup_emotional_params()
    
    def setup_scales(self):
        """Initialize musical scales and modes"""
        # Define intervals for each mode (semitones from root)
        self.mode_intervals = {
            MusicalMode.IONIAN:     [0, 2, 4, 5, 7, 9, 11],
            MusicalMode.DORIAN:     [0, 2, 3, 5, 7, 9, 10],
            MusicalMode.PHRYGIAN:   [0, 1, 3, 5, 7, 8, 10],
            MusicalMode.LYDIAN:     [0, 2, 4, 6, 7, 9, 11],
            MusicalMode.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
            MusicalMode.AEOLIAN:    [0, 2, 3, 5, 7, 8, 10],
            MusicalMode.LOCRIAN:    [0, 1, 3, 5, 6, 8, 10]
        }
        
        # Generate full scale for current mode
        self.current_scale = []
        base_intervals = self.mode_intervals[self.params.mode]
        for octave in range(self.params.octave_range):
            self.current_scale.extend([self.params.base_note + interval + (12 * octave) 
                                     for interval in base_intervals])
    
    def setup_emotional_params(self):
        """Set parameters based on emotional style"""
        self.emotional_params = {
            EmotionalStyle.ETHEREAL: {
                'tempo_factor': 0.8,
                'velocity_range': (60, 80),
                'duration_factor': 1.5,
                'fractal_params': {'zoom': 0.5, 'offset': (0.2, 0.3)}
            },
            EmotionalStyle.DARK: {
                'tempo_factor': 0.7,
                'velocity_range': (70, 90),
                'duration_factor': 1.2,
                'fractal_params': {'zoom': 0.3, 'offset': (-0.5, 0.0)}
            },
            EmotionalStyle.JOYFUL: {
                'tempo_factor': 1.2,
                'velocity_range': (80, 100),
                'duration_factor': 0.8,
                'fractal_params': {'zoom': 0.4, 'offset': (0.3, 0.4)}
            },
            EmotionalStyle.MELANCHOLIC: {
                'tempo_factor': 0.6,
                'velocity_range': (50, 70),
                'duration_factor': 1.3,
                'fractal_params': {'zoom': 0.6, 'offset': (-0.2, 0.1)}
            },
            EmotionalStyle.ENERGETIC: {
                'tempo_factor': 1.4,
                'velocity_range': (90, 110),
                'duration_factor': 0.6,
                'fractal_params': {'zoom': 0.3, 'offset': (0.4, 0.5)}
            },
            EmotionalStyle.PEACEFUL: {
                'tempo_factor': 0.7,
                'velocity_range': (40, 60),
                'duration_factor': 1.4,
                'fractal_params': {'zoom': 0.7, 'offset': (0.1, 0.2)}
            }
        }
        
        self.current_params = self.emotional_params[self.params.emotion]
    
    def mandelbrot(self, h: int, w: int, max_iter: int) -> np.ndarray:
        """Generate Mandelbrot set with emotional parameters"""
        params = self.current_params['fractal_params']
        y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
        
        # Apply emotional zoom and offset
        x = x * params['zoom'] + params['offset'][0]
        y = y * params['zoom'] + params['offset'][1]
        
        c = x + y*1j
        z = c
        divtime = max_iter + np.zeros(z.shape, dtype=int)
        
        for i in range(max_iter):
            z = z**2 + c
            diverge = z*np.conj(z) > 2**2
            div_now = diverge & (divtime == max_iter)
            divtime[div_now] = i
            z[diverge] = 2
        
        return divtime
    
    def generate_melody(self) -> Tuple[List[int], List[float], List[int]]:
        """Generate melody using fractal patterns"""
        # Generate fractal pattern
        max_iter = 100
        fractal = self.mandelbrot(self.params.resolution, self.params.resolution, max_iter)
        
        # Extract musical path
        melody = []
        timing = []
        velocities = []
        current_time = 0
        
        # Get velocity range for current emotion
        vel_min, vel_max = self.current_params['velocity_range']
        
        # Extract melody from fractal pattern
        for i in range(self.params.resolution):
            row_values = fractal[i, :]
            # Smooth the values
            smoothed = savgol_filter(row_values, 15, 3)
            
            # Map to scale degrees
            scaled_values = (smoothed % len(self.current_scale)).astype(int)
            
            for value in scaled_values[::4]:  # Take every 4th note to reduce density
                note = self.current_scale[value]
                
                # Add note if it follows melodic rules
                if len(melody) == 0 or self.is_valid_melodic_step(melody[-1], note):
                    melody.append(note)
                    timing.append(current_time)
                    
                    # Generate velocity based on emotional style
                    velocity = np.random.randint(vel_min, vel_max)
                    velocities.append(velocity)
                    
                    current_time += self.params.duration * self.current_params['duration_factor']
        
        return melody, timing, velocities
    
    def is_valid_melodic_step(self, prev_note: int, next_note: int) -> bool:
        """Check if melodic step follows musical rules"""
        interval = abs(next_note - prev_note)
        
        # Avoid large jumps (more than an octave)
        if interval > 12:
            return False
        
        # Prefer consonant intervals
        consonant_intervals = [0, 3, 4, 5, 7, 8, 9]  # unison, thirds, fourth, fifth, sixths
        return interval in consonant_intervals
    
    def create_midi(self, melody: List[int], timing: List[float], 
                   velocities: List[int], filename: str = "fractal_melody.mid"):
        """Create MIDI file from melody"""
        midi = MIDIFile(1)
        track = 0
        time = 0
        
        # Set tempo based on emotional style
        tempo = int(self.params.tempo * self.current_params['tempo_factor'])
        midi.addTempo(track, time, tempo)
        
        # Add notes to MIDI file
        for note, time, velocity in zip(melody, timing, velocities):
            midi.addNote(track, 0, note, time, self.params.duration, velocity)
        
        # Save MIDI file
        with open(filename, "wb") as f:
            midi.writeFile(f)
    
    def visualize_melody(self, melody: List[int], timing: List[float], 
                        velocities: List[int], filename: str = "melody_visualization.png"):
        """Create visualization of the melody"""
        plt.figure(figsize=(15, 10))
        
        # Create color map based on emotional style
        colors = [self.get_note_color(note, velocity) for note, velocity in zip(melody, velocities)]
        
        # Plot melody
        plt.subplot(211)
        plt.scatter(timing, melody, c=colors, alpha=0.6, s=100)
        plt.plot(timing, melody, 'gray', alpha=0.3)
        plt.title(f'Fractal Melody - {self.params.mode.value} mode, {self.params.emotion.value} style')
        plt.ylabel('MIDI Note')
        
        # Plot velocity
        plt.subplot(212)
        plt.plot(timing, velocities, 'b-', alpha=0.5)
        plt.fill_between(timing, velocities, alpha=0.2)
        plt.title('Note Velocity Over Time')
        plt.ylabel('Velocity')
        plt.xlabel('Time')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_note_color(self, note: int, velocity: int) -> Tuple[float, float, float]:
        """Generate color based on note and emotional style"""
        # Map note to hue
        hue = (note % 12) / 12.0
        
        # Map velocity to saturation
        saturation = velocity / 127.0
        
        # Set value based on emotional style
        if self.params.emotion in [EmotionalStyle.ETHEREAL, EmotionalStyle.PEACEFUL]:
            value = 0.9
        elif self.params.emotion in [EmotionalStyle.DARK, EmotionalStyle.MELANCHOLIC]:
            value = 0.6
        else:
            value = 0.8
        
        return colorsys.hsv_to_rgb(hue, saturation, value)

if __name__ == "__main__":
    # Test different emotional styles and modes
    test_combinations = [
        (MusicalMode.IONIAN, EmotionalStyle.JOYFUL),
        (MusicalMode.DORIAN, EmotionalStyle.MELANCHOLIC),
        (MusicalMode.PHRYGIAN, EmotionalStyle.DARK),
        (MusicalMode.LYDIAN, EmotionalStyle.ETHEREAL),
        (MusicalMode.MIXOLYDIAN, EmotionalStyle.ENERGETIC),
        (MusicalMode.AEOLIAN, EmotionalStyle.PEACEFUL)
    ]
    
    for mode, emotion in test_combinations:
        print(f"\nGenerating {emotion.value} melody in {mode.value} mode...")
        
        params = MelodyParams(
            mode=mode,
            emotion=emotion,
            base_note=60,  # Middle C
            tempo=120,
            resolution=200,
            duration=0.25,
            octave_range=2
        )
        
        generator = FractalMelodyGenerator(params)
        melody, timing, velocities = generator.generate_melody()
        
        # Create output files
        output_base = f"fractal_melody_{mode.value}_{emotion.value}"
        generator.create_midi(melody, timing, velocities, f"{output_base}.mid")
        generator.visualize_melody(melody, timing, velocities, f"{output_base}.png")
        
        print(f"Generated files:")
        print(f"- {output_base}.mid (MIDI file)")
        print(f"- {output_base}.png (Visualization)")