import numpy as np
from typing import List, Tuple, Dict, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import matplotlib.pyplot as plt
from midiutil import MIDIFile
from tqdm import tqdm
import time
import random
from collections import defaultdict
import colorsys
import networkx as nx

# Import existing components
from fractal_melody_generator import (
    MusicalMode, EmotionalStyle, MelodyParams, FractalMelodyGenerator
)
from harmony_engine import HarmonyParams, HarmonyEngine, HarmonyStyle

# Types, motif development, voice leading and visualisation live in their own modules
# (re-exported here, so `from advanced_composer import Note, Motif, ...` keeps working).
from musical_types import (
    MusicalForm, Articulation, TextureType, DevelopmentType,
    PerformanceDirective, Note, Motif, Section, Form,
    CreativeComposerParams
)
from motif_development import MotifDevelopment
from voice_leading import VoiceLeading
from music_visualization import Visualization


class CreativeComposer:
    def __init__(self, params: CreativeComposerParams):
        if params.generate_wav:
            raise NotImplementedError(
                "WAV output is not implemented for CreativeComposer (it writes MIDI only); "
                "render the MIDI file, or use main_composer.py / multi_fractal_music.py, "
                "which synthesise WAV audio")
        self.params = params
        self.voice_leading = VoiceLeading()
        self.motif_development = MotifDevelopment()
        self.visualization = Visualization(debug_mode=params.debug_mode)

        # Initialize components
        self.setup_components()
        self.form = self.initialize_form()

        if self.params.debug_mode:
            print(f"Initialized {self.params.form.value} form with "
                  f"{len(self.form.sections)} sections")

    def setup_components(self):
        """Initialize all composition components"""
        # Initialize fractal melody generator
        self.fractal_generator = FractalMelodyGenerator(
            MelodyParams(
                mode=self.params.mode,
                emotion=self.params.emotional_journey[0]
            )
        )

        # Initialize harmony engine
        self.harmony_engine = HarmonyEngine(
            HarmonyParams(
                style=HarmonyStyle.CLASSICAL,
                complexity=self.params.development_complexity
            )
        )

    def calculate_section_tension(self, section: Section) -> float:
        """Calculate musical tension for a section based on various parameters"""
        # Base tension from emotional style
        emotion_tension = {
            EmotionalStyle.PEACEFUL: 0.2,
            EmotionalStyle.ETHEREAL: 0.3,
            EmotionalStyle.JOYFUL: 0.5,
            EmotionalStyle.ENERGETIC: 0.7,
            EmotionalStyle.MELANCHOLIC: 0.6,
            EmotionalStyle.DARK: 0.8
        }
        base_tension = emotion_tension.get(section.emotion, 0.5)

        # Add tension based on texture
        texture_tension = {
            TextureType.MONOPHONIC: 0.3,
            TextureType.HOMOPHONIC: 0.4,
            TextureType.HETEROPHONIC: 0.6,
            TextureType.POLYPHONIC: 0.7,
            TextureType.CONTRAPUNTAL: 0.8
        }
        texture_factor = texture_tension.get(section.texture, 0.5)

        # Calculate harmonic tension
        harmonic_tension = 0.0
        if section.harmony_progression:
            for chord in section.harmony_progression:
                # More complex chords add more tension
                harmonic_tension += len(chord) * 0.1
            harmonic_tension /= len(section.harmony_progression)

        # Calculate rhythmic tension
        rhythmic_tension = 0.0
        total_duration = 0.0
        for motif in section.motifs:
            duration = sum(note.duration for note in motif.notes)
            total_duration += duration
            # Shorter durations create more tension
            avg_duration = duration / len(motif.notes)
            rhythmic_tension += 1.0 - min(1.0, avg_duration)
        rhythmic_tension /= len(section.motifs) if section.motifs else 1

        # Calculate dynamic tension
        dynamic_tension = (section.dynamics_base - 40) / 80.0  # Normalize to 0-1 range

        # Combine all factors
        tension = (
            base_tension * 0.3 +
            texture_factor * 0.2 +
            harmonic_tension * 0.2 +
            rhythmic_tension * 0.2 +
            dynamic_tension * 0.1
        )

        return min(1.0, max(0.0, tension))

    def calculate_section_duration(self, section: Section) -> float:
        """Nominal length of a section in beats: its longest motif (motifs start together).

        Note durations already include augmentation/diminution. generate_composition uses the
        span of the notes actually written (voices can stretch a motif); this is the estimate,
        and the fallback for a section without notes.
        """
        longest = max((sum(note.duration for note in motif.notes) for motif in section.motifs),
                      default=0.0)
        return longest if longest > 0 else 4.0  # Default to 4 beats if no notes

    def initialize_sonata_form(self) -> Form:
        """Initialize sonata form with exposition, development, and recapitulation"""
        if self.params.debug_mode:
            print("\nInitializing sonata form...")

        sections = []
        transitions = []

        # Generate primary theme
        self.fractal_generator.params.emotion = EmotionalStyle.ENERGETIC
        primary_melody, primary_timing, primary_velocities = self.fractal_generator.generate_melody()

        # Ensure minimum length and create primary motif
        if len(primary_melody) < 8:
            primary_melody = primary_melody * (8 // len(primary_melody) + 1)
            primary_velocities = primary_velocities * (8 // len(primary_velocities) + 1)

        primary_motif = Motif(
            notes=[Note(
                pitch=pitch,
                duration=0.25,
                velocity=primary_velocities[i % len(primary_velocities)],
                articulation=Articulation.MARCATO
            ) for i, pitch in enumerate(primary_melody[:16])],
            emotional_intent=EmotionalStyle.ENERGETIC
        )

        # Generate secondary theme (usually more lyrical)
        self.fractal_generator.params.emotion = EmotionalStyle.PEACEFUL
        secondary_melody, secondary_timing, secondary_velocities = self.fractal_generator.generate_melody()

        # Ensure minimum length and create secondary motif
        if len(secondary_melody) < 8:
            secondary_melody = secondary_melody * (8 // len(secondary_melody) + 1)
            secondary_velocities = secondary_velocities * (8 // len(secondary_velocities) + 1)

        secondary_motif = Motif(
            notes=[Note(
                pitch=pitch,
                duration=0.3,
                velocity=secondary_velocities[i % len(secondary_velocities)],
                articulation=Articulation.LEGATO
            ) for i, pitch in enumerate(secondary_melody[:16])],
            emotional_intent=EmotionalStyle.PEACEFUL
        )

        # 1. Exposition
        # First subject (primary theme in tonic)
        sections.append(Section(
            motifs=[primary_motif],
            key=self.params.key,
            tempo=self.params.base_tempo,
            texture=TextureType.HOMOPHONIC,
            emotion=EmotionalStyle.ENERGETIC,
            dynamics_base=80
        ))

        # Bridge/Transition
        transitions.append(PerformanceDirective(
            tempo_change=1.1,
            dynamic_change=1.1,
            expression="poco a poco crescendo"
        ))

        # Second subject (secondary theme in dominant)
        dominant_key = self._get_dominant_key(self.params.key)
        sections.append(Section(
            motifs=[secondary_motif],
            key=dominant_key,
            tempo=int(self.params.base_tempo * 0.95),
            texture=TextureType.HOMOPHONIC,
            emotion=EmotionalStyle.PEACEFUL,
            dynamics_base=75
        ))

        # 2. Development
        # Create development motifs using various transformations
        development_motifs = []
        for motif in [primary_motif, secondary_motif]:
            if not motif.notes:  # Skip empty motifs
                continue

            # Apply multiple transformations
            transformed = motif
            available_transforms = [DevelopmentType.AUGMENTATION]  # Always safe

            if len(transformed.notes) > 1:
                available_transforms.append(DevelopmentType.INVERSION)

            if len(transformed.notes) >= 8:
                available_transforms.append(DevelopmentType.FRAGMENTATION)

            # Apply 2-3 transformations
            for _ in range(random.randint(2, 3)):
                transform_type = random.choice(available_transforms)

                if transform_type == DevelopmentType.FRAGMENTATION and len(transformed.notes) >= 8:
                    # Safe fragmentation
                    start = random.randint(0, len(transformed.notes) - 4)
                    length = random.randint(4, min(8, len(transformed.notes) - start))
                    transformed = self.motif_development.fragment(transformed, start, length)
                elif transform_type == DevelopmentType.INVERSION:
                    transformed = self.motif_development.invert(transformed)
                else:  # AUGMENTATION
                    transformed = self.motif_development.augment(transformed, 1.5)

                development_motifs.append(transformed)

        # Development section
        sections.append(Section(
            motifs=development_motifs,
            key=self._get_relative_minor(self.params.key),  # Often moves to relative minor
            tempo=int(self.params.base_tempo * 1.1),
            texture=TextureType.POLYPHONIC,
            emotion=EmotionalStyle.DARK,
            dynamics_base=85
        ))

        # Development transition
        transitions.append(PerformanceDirective(
            tempo_change=0.9,
            dynamic_change=1.2,
            expression="building tension"
        ))

        # 3. Recapitulation
        # Return of first subject in tonic
        sections.append(Section(
            motifs=[primary_motif],
            key=self.params.key,
            tempo=self.params.base_tempo,
            texture=TextureType.HOMOPHONIC,
            emotion=EmotionalStyle.ENERGETIC,
            dynamics_base=90
        ))

        # Modified bridge
        transitions.append(PerformanceDirective(
            tempo_change=1.0,
            dynamic_change=0.9,
            expression="flowing"
        ))

        # Second subject now in tonic
        sections.append(Section(
            motifs=[secondary_motif],
            key=self.params.key,  # Now in tonic instead of dominant
            tempo=int(self.params.base_tempo * 0.95),
            texture=TextureType.HOMOPHONIC,
            emotion=EmotionalStyle.PEACEFUL,
            dynamics_base=85
        ))

        # Optional: Add coda
        if random.random() > 0.5:
            # Create closing material combining both themes
            coda_motifs = [
                self.motif_development.diminish(primary_motif, 0.75),
                self.motif_development.augment(secondary_motif, 1.25)
            ]
            sections.append(Section(
                motifs=coda_motifs,
                key=self.params.key,
                tempo=int(self.params.base_tempo * 1.05),
                texture=TextureType.POLYPHONIC,
                emotion=EmotionalStyle.JOYFUL,
                dynamics_base=95
            ))

        return Form(
            name=MusicalForm.SONATA,
            sections=sections,
            transitions=transitions,
            development_types=[
                DevelopmentType.FRAGMENTATION,
                DevelopmentType.INVERSION,
                DevelopmentType.AUGMENTATION,
                DevelopmentType.DIMINUTION
            ],
            texture_progression=[s.texture for s in sections]
        )

    def _get_dominant_key(self, key: str) -> str:
        """Get the dominant key for a given key"""
        key_to_number = {
            'C': 0, 'G': 7, 'D': 2, 'A': 9, 'E': 4, 'B': 11, 'F#': 6,
            'F': 5, 'Bb': 10, 'Eb': 3, 'Ab': 8, 'Db': 1, 'Gb': 6
        }
        number_to_key = {v: k for k, v in key_to_number.items()}

        # Get the dominant (fifth up)
        current = key_to_number[key]
        dominant = (current + 7) % 12
        return number_to_key[dominant]

    def _get_relative_minor(self, key: str) -> str:
        """Get the relative minor key for a given major key"""
        key_to_number = {
            'C': 0, 'G': 7, 'D': 2, 'A': 9, 'E': 4, 'B': 11, 'F#': 6,
            'F': 5, 'Bb': 10, 'Eb': 3, 'Ab': 8, 'Db': 1, 'Gb': 6
        }
        number_to_key = {v: k for k, v in key_to_number.items()}

        # Get the relative minor (three semitones down)
        current = key_to_number[key]
        relative = (current + 9) % 12  # equivalent to -3 in modulo 12
        return number_to_key[relative]


    def initialize_minimalist_form(self) -> Form:
        """Initialize minimalist musical form using phase shifting and gradual processes"""
        if self.params.debug_mode:
            print("\nInitializing minimalist form...")

        sections = []
        transitions = []

        # Create base motif for phasing
        melody, _, _ = self.fractal_generator.generate_melody()
        base_motif = Motif(
            notes=[Note(pitch=p, duration=0.25) for p in melody[:8]],  # Keep it short for phasing
            emotional_intent=self.params.emotional_journey[0]
        )

        # Phase-shifted variations
        phase_shifts = [0.125, 0.25, 0.375]  # Different time shifts
        motifs = [base_motif]
        for shift in phase_shifts:
            motifs.append(self.motif_development.phase_shift(base_motif, shift))

        # Create sections with different textures and processes
        for i, emotion in enumerate(self.params.emotional_journey):
            # Gradually add voices and complexity
            active_motifs = motifs[:min(i+2, len(motifs))]

            # Create additive process
            if i < len(self.params.emotional_journey) - 1:
                directive = PerformanceDirective(
                    tempo_change=1.0 + (i * 0.1),
                    dynamic_change=1.0 + (i * 0.15)
                )
                transitions.append(directive)

            sections.append(Section(
                motifs=active_motifs,
                key=self.params.key,
                tempo=int(self.params.base_tempo * (1 + i * 0.1)),
                texture=TextureType.POLYPHONIC,
                emotion=emotion,
                dynamics_base=70 + (i * 5)
            ))

        return Form(
            name=MusicalForm.MINIMALIST,
            sections=sections,
            transitions=transitions,
            development_types=[DevelopmentType.PHASE_SHIFT],
            texture_progression=[TextureType.POLYPHONIC] * len(sections)
        )

    def _create_motif_from_melody(self, emotion: EmotionalStyle, duration: float = 0.25) -> Motif:
        """Helper function to create a motif from fractal melody with longer phrases"""
        melody_data, timing, velocities = self.fractal_generator.generate_melody()

        # Ensure we have enough notes (minimum 8, target 32)
        melody_length = len(melody_data)
        if melody_length < 8:
            # If we don't have enough notes, repeat the melody
            melody_data = melody_data * (8 // melody_length + 1)
            velocities = velocities * (8 // len(velocities) + 1)

        # Take up to 32 notes for longer phrases
        notes = []
        for i, pitch in enumerate(melody_data[:32]):
            notes.append(Note(
                pitch=pitch,
                duration=duration * random.choice([1.0, 1.5, 2.0]),  # More varied durations
                velocity=velocities[i % len(velocities)],  # Safely cycle through velocities
                articulation=random.choice(list(Articulation))  # More varied articulation
            ))

        return Motif(
            notes=notes,
            emotional_intent=emotion
        )

    def initialize_through_composed_form(self) -> Form:
        """Initialize through-composed form with continuous development"""
        if self.params.debug_mode:
            print("\nInitializing through-composed form...")

        sections = []
        transitions = []
        previous_motifs = None

        for i, emotion in enumerate(self.params.emotional_journey):
            # Generate new material while maintaining connections
            if previous_motifs is None:
                # Generate initial motifs
                self.fractal_generator.params.emotion = emotion
                current_motifs = [self._create_motif_from_melody(emotion)]
            else:
                # Develop from previous material
                current_motifs = []
                for prev_motif in previous_motifs:
                    transformed = prev_motif
                    # Apply development techniques
                    transform_type = random.choice([
                        DevelopmentType.AUGMENTATION,
                        DevelopmentType.DIMINUTION,
                        DevelopmentType.FRAGMENTATION
                    ])

                    if transform_type == DevelopmentType.AUGMENTATION:
                        transformed = self.motif_development.augment(transformed)
                    elif transform_type == DevelopmentType.DIMINUTION:
                        transformed = self.motif_development.diminish(transformed)
                    else:
                        transformed = self.motif_development.fragment(
                            transformed,
                            random.randint(0, len(transformed.notes)-2),
                            random.randint(2, 4)
                        )
                    current_motifs.append(transformed)

            # Create smoothing transition
            if i < len(self.params.emotional_journey) - 1:
                directive = PerformanceDirective(
                    tempo_change=0.9 + (i * 0.1),
                    dynamic_change=0.9 + (i * 0.1)
                )
                transitions.append(directive)

            # Vary texture for interest
            texture = random.choice([
                TextureType.HOMOPHONIC,
                TextureType.POLYPHONIC,
                TextureType.CONTRAPUNTAL
            ])

            sections.append(Section(
                motifs=current_motifs,
                key=self.params.key,
                tempo=self.params.base_tempo,
                texture=texture,
                emotion=emotion,
                dynamics_base=75
            ))

            previous_motifs = current_motifs

        return Form(
            name=MusicalForm.THROUGH_COMPOSED,
            sections=sections,
            transitions=transitions,
            development_types=list(DevelopmentType),
            texture_progression=[s.texture for s in sections]
        )

    def initialize_aleatoric_form(self) -> Form:
        """Initialize aleatoric form with controlled randomness"""
        if self.params.debug_mode:
            print("\nInitializing aleatoric form...")

        sections = []
        transitions = []

        # Generate pool of motifs using different fractal patterns
        motif_pool = []
        for emotion in self.params.emotional_journey:
            # Generate variations using different fractal parameters
            self.fractal_generator.params.emotion = emotion
            motif = self._create_motif_from_melody(emotion,
                                                 duration=random.choice([0.125, 0.25, 0.5]))
            motif_pool.append(motif)

            # Add transformed variations using safer transformations
            motif_pool.append(self.motif_development.retrograde(motif))
            motif_pool.append(self.motif_development.augment(motif))
            motif_pool.append(self.motif_development.diminish(motif))

        # Create sections with different chance operations
        for i, emotion in enumerate(self.params.emotional_journey):
            # Randomly select motifs for this section
            section_motifs = random.sample(motif_pool, min(3, len(motif_pool)))

            # Create aleatoric transition
            if i < len(self.params.emotional_journey) - 1:
                directive = PerformanceDirective(
                    tempo_change=random.uniform(0.8, 1.2),
                    dynamic_change=random.uniform(0.7, 1.3),
                    articulation=random.choice(list(Articulation)),
                    timing_offset=random.uniform(-0.1, 0.1)
                )
                transitions.append(directive)

            # Randomly select texture
            texture = random.choice([
                TextureType.POLYPHONIC,
                TextureType.HETEROPHONIC,
                TextureType.CONTRAPUNTAL
            ])

            sections.append(Section(
                motifs=section_motifs,
                key=self.params.key,
                tempo=int(self.params.base_tempo * random.uniform(0.8, 1.2)),
                texture=texture,
                emotion=emotion,
                dynamics_base=random.randint(60, 90)
            ))

        return Form(
            name=MusicalForm.ALEATORIC,
            sections=sections,
            transitions=transitions,
            development_types=[
                DevelopmentType.AUGMENTATION,
                DevelopmentType.DIMINUTION,
                DevelopmentType.RETROGRADE
            ],
            texture_progression=[s.texture for s in sections]
        )

    def initialize_arch_form(self) -> Form:
        """Initialize arch form (ABCBA structure) with palindromic design"""
        if self.params.debug_mode:
            print("\nInitializing arch form...")

        sections = []
        transitions = []

        # Generate A section material
        self.fractal_generator.params.emotion = EmotionalStyle.PEACEFUL
        a_motifs = [self._create_motif_from_melody(EmotionalStyle.PEACEFUL)]

        # Generate B section material (more intense)
        self.fractal_generator.params.emotion = EmotionalStyle.ENERGETIC
        b_motifs = [self._create_motif_from_melody(EmotionalStyle.ENERGETIC)]

        # Generate C section material (climax)
        self.fractal_generator.params.emotion = EmotionalStyle.DARK
        c_motifs = [self._create_motif_from_melody(EmotionalStyle.DARK)]

        # Create palindromic structure
        section_structure = [
            (a_motifs, TextureType.HOMOPHONIC, EmotionalStyle.PEACEFUL, 1.0),
            (b_motifs, TextureType.POLYPHONIC, EmotionalStyle.ENERGETIC, 1.1),
            (c_motifs, TextureType.CONTRAPUNTAL, EmotionalStyle.DARK, 1.2),
            (b_motifs, TextureType.POLYPHONIC, EmotionalStyle.ENERGETIC, 1.1),
            (a_motifs, TextureType.HOMOPHONIC, EmotionalStyle.PEACEFUL, 1.0)
        ]

        for i, (motifs, texture, emotion, tempo_factor) in enumerate(section_structure):
            sections.append(Section(
                motifs=motifs,
                key=self.params.key,
                tempo=int(self.params.base_tempo * tempo_factor),
                texture=texture,
                emotion=emotion,
                dynamics_base=70 + (i * 5 if i < 3 else (4-i) * 5)
            ))

            if i < len(section_structure) - 1:
                transitions.append(PerformanceDirective(
                    tempo_change=tempo_factor,
                    dynamic_change=1.0 + (0.1 * (i if i < 3 else 4-i))
                ))

        return Form(
            name=MusicalForm.ARCH,
            sections=sections,
            transitions=transitions,
            development_types=[DevelopmentType.AUGMENTATION, DevelopmentType.DIMINUTION],
            texture_progression=[s.texture for s in sections]
        )

    def initialize_moment_form(self) -> Form:
        """Initialize moment form (inspired by Stockhausen) with independent sections"""
        if self.params.debug_mode:
            print("\nInitializing moment form...")

        sections = []
        transitions = []

        # Create contrasting "moments"
        for i in range(self.params.section_count):
            # Randomly select parameters for this moment
            emotion = random.choice(list(EmotionalStyle))
            texture = random.choice(list(TextureType))
            tempo_factor = random.uniform(0.8, 1.4)

            # Generate material
            self.fractal_generator.params.emotion = emotion
            motifs = [self._create_motif_from_melody(emotion)]

            # Apply random transformations
            if len(motifs[0].notes) >= 8:
                transform_type = random.choice(list(DevelopmentType))
                if transform_type == DevelopmentType.AUGMENTATION:
                    motifs[0] = self.motif_development.augment(motifs[0], random.uniform(1.5, 2.0))
                elif transform_type == DevelopmentType.DIMINUTION:
                    motifs[0] = self.motif_development.diminish(motifs[0], random.uniform(0.4, 0.7))

            sections.append(Section(
                motifs=motifs,
                key=random.choice(['C', 'G', 'D', 'A', 'E', 'F']),  # Various key centers
                tempo=int(self.params.base_tempo * tempo_factor),
                texture=texture,
                emotion=emotion,
                dynamics_base=random.randint(60, 90)
            ))

            # Create abrupt transitions characteristic of moment form
            if i < self.params.section_count - 1:
                transitions.append(PerformanceDirective(
                    tempo_change=random.uniform(0.7, 1.4),
                    dynamic_change=random.uniform(0.6, 1.5),
                    articulation=random.choice(list(Articulation))
                ))

        return Form(
            name=MusicalForm.MOMENT,
            sections=sections,
            transitions=transitions,
            development_types=list(DevelopmentType),
            texture_progression=[s.texture for s in sections]
        )

    def initialize_process_music(self) -> Form:
        """Initialize process music (inspired by Reich/Glass) with gradual transformation"""
        if self.params.debug_mode:
            print("\nInitializing process music...")

        sections = []
        transitions = []

        # Generate base material
        self.fractal_generator.params.emotion = EmotionalStyle.ETHEREAL
        base_motif = self._create_motif_from_melody(EmotionalStyle.ETHEREAL)

        # Create phases of the process
        phase_shift = 0.125  # Eighth note shift
        num_phases = min(8, self.params.section_count)

        for i in range(num_phases):
            # Apply phase shifting
            shifted_motif = self.motif_development.phase_shift(base_motif, phase_shift * i)
            motifs = [base_motif, shifted_motif]

            # Add textural layers
            if i > 2:
                augmented = self.motif_development.augment(base_motif, 2.0)
                motifs.append(augmented)

            sections.append(Section(
                motifs=motifs,
                key=self.params.key,
                tempo=self.params.base_tempo,
                texture=TextureType.POLYPHONIC,
                emotion=EmotionalStyle.ETHEREAL,
                dynamics_base=70 + (i * 3)
            ))

            if i < num_phases - 1:
                transitions.append(PerformanceDirective(
                    tempo_change=1.0,
                    dynamic_change=1.05,
                    expression="gradually building"
                ))

        return Form(
            name=MusicalForm.PROCESS,
            sections=sections,
            transitions=transitions,
            development_types=[DevelopmentType.PHASE_SHIFT, DevelopmentType.AUGMENTATION],
            texture_progression=[TextureType.POLYPHONIC] * len(sections)
        )

    def initialize_form(self) -> Form:
        """Initialize musical form based on parameters"""
        if self.params.debug_mode:
            print(f"\nInitializing {self.params.form.value} form...")

        form_initializers = {
            MusicalForm.SONATA: self.initialize_sonata_form,
            MusicalForm.MINIMALIST: self.initialize_minimalist_form,
            MusicalForm.ALEATORIC: self.initialize_aleatoric_form,
            MusicalForm.THROUGH_COMPOSED: self.initialize_through_composed_form,
            MusicalForm.MIXED: self.initialize_mixed_form,
            MusicalForm.ARCH: self.initialize_arch_form,
            MusicalForm.MOMENT: self.initialize_moment_form,
            MusicalForm.PROCESS: self.initialize_process_music
        }

        # Get the appropriate initializer or default to mixed form
        initializer = form_initializers.get(self.params.form, self.initialize_mixed_form)
        return initializer()

    def initialize_mixed_form(self) -> Form:
        """Initialize mixed form with extended development"""
        if self.params.debug_mode:
            print("\nInitializing mixed form...")

        sections = []
        transitions = []

        # Use different compositional approaches for different sections
        compositional_styles = [
            (self.initialize_minimalist_form, TextureType.POLYPHONIC),
            (self.initialize_aleatoric_form, TextureType.HETEROPHONIC),
            (self.initialize_through_composed_form, TextureType.CONTRAPUNTAL),
            (self.initialize_sonata_form, TextureType.HOMOPHONIC)
        ]

        # Generate material for each emotional stage
        for i, emotion in enumerate(self.params.emotional_journey):
            # Choose a compositional style for this section
            style_func, texture = random.choice(compositional_styles)

            # Generate a form using the chosen style
            temp_form = style_func()

            if temp_form.sections:
                # Take one or more sections from the generated form
                num_sections = random.randint(1, 2)  # Sometimes use multiple sections
                for section in temp_form.sections[:num_sections]:
                    section.emotion = emotion  # Update emotion to match current journey

                    # Add development sections
                    if random.random() < 0.3:  # 30% chance of additional development
                        developed_section = self._develop_section(section)
                        sections.append(developed_section)

                    sections.append(section)

                    # Create transition to next section
                    if i < len(self.params.emotional_journey) - 1:
                        directive = PerformanceDirective(
                            tempo_change=random.uniform(0.8, 1.2),
                            dynamic_change=random.uniform(0.9, 1.1),
                            articulation=random.choice(list(Articulation)),
                            expression=f"transitioning to {self.params.emotional_journey[i+1].value}"
                        )
                        transitions.append(directive)

        # Ensure we have enough sections
        while len(sections) < self.params.section_count:
            source_section = random.choice(sections)
            new_section = self._develop_section(source_section)
            sections.append(new_section)

            # Add corresponding transition
            transitions.append(PerformanceDirective(
                tempo_change=random.uniform(0.9, 1.1),
                dynamic_change=random.uniform(0.9, 1.1)
            ))

        return Form(
            name=MusicalForm.MIXED,
            sections=sections[:self.params.section_count],  # Trim to desired length
            transitions=transitions[:self.params.section_count - 1],
            development_types=list(DevelopmentType),
            texture_progression=[s.texture for s in sections[:self.params.section_count]]
        )

    def _develop_section(self, section: Section) -> Section:
        """Create a developed variation of a section"""
        new_motifs = []
        for motif in section.motifs:
            if not motif.notes:  # Skip empty motifs
                continue

            # Apply multiple development techniques
            transformed = motif
            for _ in range(random.randint(1, 3)):
                # Choose appropriate transformation based on motif length
                available_transforms = [DevelopmentType.AUGMENTATION, DevelopmentType.DIMINUTION]

                if len(transformed.notes) > 1:
                    available_transforms.append(DevelopmentType.RETROGRADE)

                if len(transformed.notes) >= 8:
                    available_transforms.append(DevelopmentType.FRAGMENTATION)

                transform_type = random.choice(available_transforms)

                if transform_type == DevelopmentType.AUGMENTATION:
                    transformed = self.motif_development.augment(transformed,
                                                              factor=random.uniform(1.5, 2.0))
                elif transform_type == DevelopmentType.DIMINUTION:
                    transformed = self.motif_development.diminish(transformed,
                                                               factor=random.uniform(0.4, 0.7))
                elif transform_type == DevelopmentType.RETROGRADE:
                    transformed = self.motif_development.retrograde(transformed)
                elif transform_type == DevelopmentType.FRAGMENTATION and len(transformed.notes) >= 8:
                    # Safe fragmentation for longer motifs
                    start = random.randint(0, len(transformed.notes) - 4)
                    length = random.randint(4, min(8, len(transformed.notes) - start))
                    transformed = self.motif_development.fragment(transformed, start, length)
            new_motifs.append(transformed)

        # Sometimes add additional motifs
        if random.random() < 0.4:  # 40% chance
            new_motifs.append(self._create_motif_from_melody(section.emotion))

        return Section(
            motifs=new_motifs,
            key=section.key,
            tempo=int(section.tempo * random.uniform(0.9, 1.1)),
            texture=random.choice(list(TextureType)),
            emotion=section.emotion,
            dynamics_base=section.dynamics_base
        )

    def generate_composition(self) -> Tuple[MIDIFile, Optional[np.ndarray]]:
        """Generate complete musical composition"""
        if self.params.debug_mode:
            print("\nGenerating composition...")
            start_time = time.time()

        # Generate MIDI
        midi = MIDIFile(self.params.voice_count)
        current_time = 0.0
        # (track, pitch, start tick) of every note written so far; see add_motif_to_midi
        self._note_starts = set()

        # Calculate durations and tension values for visualization
        durations = []
        tension_values = []
        # (start, end) in beats of every section: each starts where the previous one's notes end
        self.section_spans = []

        # Process each section
        for section_idx, section in enumerate(tqdm(self.form.sections,
                                                 desc="Processing sections")):
            if self.params.debug_mode:
                print(f"\nProcessing section {section_idx + 1}:")
                print(f"Texture: {section.texture.value}")
                print(f"Emotion: {section.emotion.value}")

            # Set tempo for section
            midi.addTempo(0, current_time, section.tempo)

            section_tension = self.calculate_section_tension(section)
            tension_values.append(section_tension)
            section_end = current_time

            # Process motifs: all motifs of a section start together, at the section start
            for motif_idx, motif in enumerate(section.motifs):
                if self.params.debug_mode:
                    print(f"Processing motif {motif_idx + 1} "
                          f"({len(motif.notes)} notes)")

                # Process each voice
                for voice_idx in range(self.params.voice_count):
                    # Apply voice-specific transformations
                    transformed_motif = self.transform_for_voice(
                        motif, voice_idx, section.texture
                    )

                    # Add notes to MIDI
                    if transformed_motif:
                        end = self.add_motif_to_midi(
                            midi, transformed_motif, voice_idx, current_time
                        )
                        section_end = max(section_end, end)

            # The next section starts where this one's last note ends (all times in beats)
            if section_end <= current_time:
                section_end = current_time + self.calculate_section_duration(section)
            self.section_spans.append((current_time, section_end))
            durations.append(section_end - current_time)
            current_time = section_end

        # Generate waveform if requested
        waveform = None
        if self.params.generate_wav:
            waveform = self.generate_waveform(midi)

        # Create visualizations if enabled
        if self.params.visualization_enabled:
            self.visualization.create_form_graph(self.form)
            self.visualization.plot_tension_graph(
                self.form.sections,
                durations,
                tension_values,
                filename="tension_graph.png"
            )

        if self.params.debug_mode:
            end_time = time.time()
            print(f"\nComposition generated in {end_time - start_time:.2f} seconds")

        return midi, waveform

    def create_homophonic_voice(self, motif: Motif, voice_idx: int) -> Optional[Motif]:
        """Create homophonic voice part"""
        if voice_idx == 0:
            return motif  # Main melody unchanged
        elif voice_idx < 4:  # Create harmony parts
            new_notes = []
            harmony_intervals = [-3, -7, -10]  # Intervals for harmony voices below melody
            interval = harmony_intervals[voice_idx - 1]

            for note in motif.notes:
                # Handle pitch whether it's a list or single value
                original_pitch = note.pitch[0] if isinstance(note.pitch, (list, tuple)) else note.pitch

                new_notes.append(Note(
                    pitch=original_pitch + interval,
                    duration=note.duration,
                    velocity=max(40, note.velocity - 20),  # Slightly softer
                    articulation=note.articulation,
                    performance=note.performance
                ))

            return Motif(
                notes=new_notes,
                emotional_intent=motif.emotional_intent,
                development_history=motif.development_history
            )
        return None

    def create_polyphonic_voice(self, motif: Motif, voice_idx: int) -> Optional[Motif]:
        """Create polyphonic voice part with independent movement"""
        if voice_idx == 0:
            return motif  # Main melody unchanged
        elif voice_idx < 4:
            new_notes = []
            # Create contrasting rhythm and movement
            for note in motif.notes:
                # Handle pitch whether it's a list or single value
                original_pitch = note.pitch[0] if isinstance(note.pitch, (list, tuple)) else note.pitch

                # Alternate between parallel and contrary motion
                if voice_idx % 2 == 0:
                    pitch_offset = 4 if original_pitch % 12 in [0, 4, 7] else 3
                else:
                    pitch_offset = -3 if original_pitch % 12 in [0, 4, 7] else -4

                new_notes.append(Note(
                    pitch=original_pitch + pitch_offset + (voice_idx * 2),
                    duration=note.duration * (1.5 if voice_idx == 2 else 0.75),
                    velocity=max(40, note.velocity - 15),
                    articulation=note.articulation,
                    performance=note.performance
                ))

            return Motif(
                notes=new_notes,
                emotional_intent=motif.emotional_intent,
                development_history=motif.development_history
            )
        return None

    def create_heterophonic_voice(self, motif: Motif, voice_idx: int) -> Optional[Motif]:
        """Create heterophonic voice part with melodic variations"""
        if voice_idx == 0:
            return motif  # Main melody unchanged
        elif voice_idx < 4:
            new_notes = []
            for note in motif.notes:
                # Handle pitch whether it's a list or single value
                original_pitch = note.pitch[0] if isinstance(note.pitch, (list, tuple)) else note.pitch

                # Add ornaments and variations
                if random.random() < 0.3:  # 30% chance of ornament
                    # Add passing tones or neighbor tones
                    new_notes.extend([
                        Note(
                            pitch=original_pitch + random.choice([-1, 1]),
                            duration=note.duration * 0.5,
                            velocity=note.velocity - 10,
                            articulation=note.articulation,
                            performance=note.performance
                        ),
                        Note(
                            pitch=original_pitch,
                            duration=note.duration * 0.5,
                            velocity=note.velocity - 10,
                            articulation=note.articulation,
                            performance=note.performance
                        )
                    ])
                else:
                    new_notes.append(Note(
                        pitch=note.pitch,
                        duration=note.duration,
                        velocity=max(40, note.velocity - 15),
                        articulation=note.articulation,
                        performance=note.performance
                    ))

            return Motif(
                notes=new_notes,
                emotional_intent=motif.emotional_intent,
                development_history=motif.development_history
            )
        return None

    def create_contrapuntal_voice(self, motif: Motif, voice_idx: int) -> Optional[Motif]:
        """Create contrapuntal voice part using species counterpoint rules"""
        if voice_idx == 0:
            return motif  # Main melody unchanged
        elif voice_idx < 4:
            new_notes = []
            consonant_intervals = [3, 4, 7, 8, 9]  # Major/minor thirds, perfect fifth, sixths
            current_interval = random.choice(consonant_intervals)

            for i, note in enumerate(motif.notes):
                # Handle pitch whether it's a list or single value
                original_pitch = note.pitch[0] if isinstance(note.pitch, (list, tuple)) else note.pitch

                # Apply counterpoint rules
                if i > 0:
                    # Choose new interval based on voice leading rules
                    prev_interval = current_interval
                    possible_intervals = [i for i in consonant_intervals if abs(i - prev_interval) <= 4]
                    if possible_intervals:
                        current_interval = random.choice(possible_intervals)

                # Alternate between contrary and parallel motion
                if i % 2 == 0:
                    pitch_offset = current_interval
                else:
                    pitch_offset = -current_interval

                new_notes.append(Note(
                    pitch=original_pitch + pitch_offset + ((voice_idx - 2) * 12),
                    duration=note.duration,
                    velocity=max(40, note.velocity - 12),
                    articulation=note.articulation,
                    performance=note.performance
                ))

            return Motif(
                notes=new_notes,
                emotional_intent=motif.emotional_intent,
                development_history=motif.development_history
            )
        return None

    def transform_for_voice(self, motif: Motif, voice_idx: int, texture: TextureType) -> Optional[Motif]:
        """Transform motif based on voice and texture"""
        if motif is None:
            return None

        texture_handlers = {
            TextureType.MONOPHONIC: lambda m, v: m if v == 0 else None,
            TextureType.HOMOPHONIC: self.create_homophonic_voice,
            TextureType.POLYPHONIC: self.create_polyphonic_voice,
            TextureType.HETEROPHONIC: self.create_heterophonic_voice,
            TextureType.CONTRAPUNTAL: self.create_contrapuntal_voice
        }

        handler = texture_handlers.get(texture)
        if handler:
            return handler(motif, voice_idx)

        return None  # Default case if texture type not handled

    def add_motif_to_midi(self, midi: MIDIFile, motif: Optional[Motif],
                             voice_idx: int, start_time: float) -> float:
        """Add motif to MIDI file with performance directives.

        Returns the time (beats) at which the last written note ends (start_time if none).
        """
        if not motif:
            return start_time

        current_time = start_time
        end_time = start_time
        track = voice_idx

        for note in motif.notes:
            try:
                # Extract and validate pitch
                pitch = note.pitch[0] if isinstance(note.pitch, (list, tuple)) else note.pitch
                pitch = int(pitch)  # Convert to integer

                # Ensure pitch is in valid MIDI range (0-127)
                pitch = max(0, min(127, pitch))

                # Apply performance directives
                if note.performance:
                    duration = note.duration
                    velocity = note.velocity
                    timing = current_time + note.performance.timing_offset

                    if note.performance.tempo_change:
                        duration *= note.performance.tempo_change
                    if note.performance.dynamic_change:
                        velocity = int(velocity * note.performance.dynamic_change)

                    # Apply articulation
                    if note.articulation == Articulation.STACCATO:
                        duration *= 0.5
                    elif note.articulation == Articulation.LEGATO:
                        duration *= 1.1

                    # Ensure velocity is in valid MIDI range (0-127)
                    velocity = max(0, min(127, int(velocity)))

                    if self._is_new_note_start(midi, track, pitch, timing):
                        midi.addNote(track, track, pitch, timing,
                                   duration, velocity)
                        end_time = max(end_time, timing + duration)
                    current_time += duration
                else:
                    # Ensure velocity is in valid MIDI range (0-127)
                    velocity = max(0, min(127, int(note.velocity)))

                    if self._is_new_note_start(midi, track, pitch, current_time):
                        midi.addNote(track, track, pitch, current_time,
                                   note.duration, velocity)
                        end_time = max(end_time, current_time + note.duration)
                    current_time += note.duration

            except (TypeError, ValueError, IndexError) as e:
                if self.params.debug_mode:
                    print(f"Warning: Skipping invalid note in voice {voice_idx}: {str(e)}")
                continue
        return end_time

    def _is_new_note_start(self, midi: MIDIFile, track: int, pitch: int, time: float) -> bool:
        """True unless a note with this pitch already starts at this tick on this track.

        Overlapping sections can produce such duplicates. MIDIUtil's duplicate removal then
        drops one note-on but keeps both note-offs, and writeFile fails with IndexError.
        Skipping the second note keeps the first one, as MIDIUtil intends.
        """
        starts = getattr(self, '_note_starts', None)
        if starts is None:
            starts = self._note_starts = set()
        key = (track, pitch, midi.time_to_ticks(time))
        if key in starts:
            return False
        starts.add(key)
        return True

    def save_composition(self, midi: MIDIFile, audio: Optional[np.ndarray],
                        base_filename: str = "creative_composition"):
        """Save composition as MIDI and optionally WAV"""
        # Save MIDI
        with open(f"{base_filename}.mid", "wb") as f:
            midi.writeFile(f)

        # Save WAV if available
        if audio is not None:
            from scipy.io import wavfile
            wavfile.write(f"{base_filename}.wav", 44100,
                         (audio * 32767).astype(np.int16))

if __name__ == "__main__":
    # Create composer with extended settings
    params = CreativeComposerParams(
        form=MusicalForm.ARCH,
        base_tempo=10,  # Slightly slower tempo for more expansive development
        key="C",
        mode=MusicalMode.IONIAN,
        emotional_journey=[
            EmotionalStyle.PEACEFUL,
            EmotionalStyle.ENERGETIC,
            EmotionalStyle.MELANCHOLIC,
            EmotionalStyle.DARK,
            EmotionalStyle.ETHEREAL,
            EmotionalStyle.JOYFUL
        ],  # More emotional stages for longer development
        development_complexity=0.8,
        texture_density=0.7,
        modulation_frequency=0.4,
        section_count=20,  # Increased number of sections
        voice_count=4,
        generate_wav=False,
        visualization_enabled=True,
        debug_mode=True
    )

    # Create composer
    composer = CreativeComposer(params)

    # Generate composition
    print("\nGenerating creative composition...")
    midi, audio = composer.generate_composition()

    # Save outputs
    print("\nSaving files...")
    composer.save_composition(midi, audio)

    print("\nComposition process complete!")
