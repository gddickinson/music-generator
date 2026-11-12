"""Musical type definitions: enums, dataclasses, and constants for the composition system."""

from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum, auto

from fractal_melody_generator import MusicalMode, EmotionalStyle


class MusicalForm(Enum):
    SONATA = "sonata"
    RONDO = "rondo"
    THEME_AND_VARIATIONS = "theme_and_variations"
    FUGUE = "fugue"
    MINIMALIST = "minimalist"
    THROUGH_COMPOSED = "through_composed"
    ALEATORIC = "aleatoric"
    MIXED = "mixed"
    ARCH = "arch"
    MOMENT = "moment"
    PROCESS = "process"


class Articulation(Enum):
    LEGATO = "legato"
    STACCATO = "staccato"
    MARCATO = "marcato"
    TENUTO = "tenuto"
    PORTATO = "portato"
    SPICCATO = "spiccato"
    PIZZICATO = "pizzicato"


class TextureType(Enum):
    MONOPHONIC = auto()
    HOMOPHONIC = auto()
    POLYPHONIC = auto()
    HETEROPHONIC = auto()
    CONTRAPUNTAL = auto()


class DevelopmentType(Enum):
    AUGMENTATION = "augmentation"
    DIMINUTION = "diminution"
    INVERSION = "inversion"
    RETROGRADE = "retrograde"
    TRANSPOSITION = "transposition"
    FRAGMENTATION = "fragmentation"
    SEQUENCE = "sequence"
    PHASE_SHIFT = "phase_shift"


@dataclass
class PerformanceDirective:
    tempo_change: Optional[float] = None
    dynamic_change: Optional[float] = None
    articulation: Optional[Articulation] = None
    expression: Optional[str] = None
    timing_offset: float = 0.0


@dataclass
class Note:
    pitch: int
    duration: float
    velocity: int = 80
    articulation: Articulation = Articulation.LEGATO
    performance: Optional[PerformanceDirective] = None


@dataclass
class Motif:
    notes: List[Note]
    emotional_intent: EmotionalStyle
    development_history: List[DevelopmentType] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(note.duration for note in self.notes)

    @property
    def pitch_range(self) -> Tuple[int, int]:
        pitches = [note.pitch for note in self.notes]
        return min(pitches), max(pitches)


@dataclass
class Section:
    motifs: List[Motif]
    key: str
    tempo: int
    texture: TextureType
    emotion: EmotionalStyle
    dynamics_base: int
    transitions: List[PerformanceDirective] = field(default_factory=list)
    harmony_progression: Optional[List[List[int]]] = None


@dataclass
class Form:
    name: MusicalForm
    sections: List[Section]
    transitions: List[PerformanceDirective]
    development_types: List[DevelopmentType]
    texture_progression: List[TextureType]


# Semitones above C of the keys CreativeComposer accepts (F# and Gb are the same pitch class).
KEY_SEMITONES = {
    'C': 0, 'G': 7, 'D': 2, 'A': 9, 'E': 4, 'B': 11, 'F#': 6,
    'F': 5, 'Bb': 10, 'Eb': 3, 'Ab': 8, 'Db': 1, 'Gb': 6
}


def transposition(from_key: str, to_key: str) -> int:
    """Shortest transposition in semitones (-5..+6) from one key centre to another."""
    d = (KEY_SEMITONES[to_key] - KEY_SEMITONES[from_key]) % 12
    return d - 12 if d > 6 else d


@dataclass
class CreativeComposerParams:
    form: MusicalForm = MusicalForm.SONATA
    base_tempo: int = 120
    # Home key: melodies are built on this tonic (the nearest one to middle C), and a section
    # in another key (dominant, relative minor, a modulation) is transposed to that key.
    key: str = "C"
    mode: MusicalMode = MusicalMode.IONIAN
    emotional_journey: List[EmotionalStyle] = field(default_factory=lambda: [
        EmotionalStyle.PEACEFUL,
        EmotionalStyle.ENERGETIC,
        EmotionalStyle.MELANCHOLIC,
        EmotionalStyle.JOYFUL
    ])
    # 0..1: how many development steps a motif goes through when developed (1 to 4).
    development_complexity: float = 0.7
    # 0..1: share of the accompanying voices that play (0 = the melody alone, 1 = every voice).
    texture_density: float = 0.6
    # 0..1: chance that a later section in the home key modulates to a related key
    # (dominant, subdominant or relative minor).
    modulation_frequency: float = 0.3
    section_count: int = 4
    voice_count: int = 4
    generate_wav: bool = False
    visualization_enabled: bool = True
    debug_mode: bool = True
