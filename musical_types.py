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


@dataclass
class CreativeComposerParams:
    form: MusicalForm = MusicalForm.SONATA
    base_tempo: int = 120
    key: str = "C"
    mode: MusicalMode = MusicalMode.IONIAN
    emotional_journey: List[EmotionalStyle] = field(default_factory=lambda: [
        EmotionalStyle.PEACEFUL,
        EmotionalStyle.ENERGETIC,
        EmotionalStyle.MELANCHOLIC,
        EmotionalStyle.JOYFUL
    ])
    development_complexity: float = 0.7
    texture_density: float = 0.6
    modulation_frequency: float = 0.3
    section_count: int = 4
    voice_count: int = 4
    generate_wav: bool = False
    visualization_enabled: bool = True
    debug_mode: bool = True
