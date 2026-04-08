"""Motif development techniques: inversion, retrograde, augmentation, etc."""

from typing import List
from musical_types import (
    Motif, Note, PerformanceDirective, DevelopmentType
)


class MotifDevelopment:
    """Advanced motif development techniques."""

    @staticmethod
    def invert(motif: Motif) -> Motif:
        """Invert the motif's pitch contour."""
        if not motif.notes:
            return motif

        center = motif.notes[0].pitch
        new_notes = []

        for note in motif.notes:
            new_pitch = center + (center - note.pitch)
            new_notes.append(Note(
                pitch=new_pitch,
                duration=note.duration,
                velocity=note.velocity,
                articulation=note.articulation,
                performance=note.performance
            ))

        return Motif(
            notes=new_notes,
            emotional_intent=motif.emotional_intent,
            development_history=motif.development_history + [DevelopmentType.INVERSION]
        )

    @staticmethod
    def retrograde(motif: Motif) -> Motif:
        """Reverse the motif."""
        if not motif.notes:
            return motif

        return Motif(
            notes=list(reversed(motif.notes)),
            emotional_intent=motif.emotional_intent,
            development_history=motif.development_history + [DevelopmentType.RETROGRADE]
        )

    @staticmethod
    def augment(motif: Motif, factor: float = 2.0) -> Motif:
        """Augment the rhythm by a factor."""
        if not motif.notes:
            return motif

        new_notes = []
        for note in motif.notes:
            new_notes.append(Note(
                pitch=note.pitch,
                duration=note.duration * factor,
                velocity=note.velocity,
                articulation=note.articulation,
                performance=note.performance
            ))

        return Motif(
            notes=new_notes,
            emotional_intent=motif.emotional_intent,
            development_history=motif.development_history + [DevelopmentType.AUGMENTATION]
        )

    @staticmethod
    def diminish(motif: Motif, factor: float = 0.5) -> Motif:
        """Diminish the rhythm by a factor."""
        if not motif.notes:
            return motif

        new_notes = []
        for note in motif.notes:
            new_notes.append(Note(
                pitch=note.pitch,
                duration=note.duration * factor,
                velocity=note.velocity,
                articulation=note.articulation,
                performance=note.performance
            ))

        return Motif(
            notes=new_notes,
            emotional_intent=motif.emotional_intent,
            development_history=motif.development_history + [DevelopmentType.DIMINUTION]
        )

    @staticmethod
    def fragment(motif: Motif, start_idx: int, length: int) -> Motif:
        """Create a fragment of the motif."""
        if not motif.notes:
            return motif

        if start_idx + length > len(motif.notes):
            length = len(motif.notes) - start_idx

        return Motif(
            notes=motif.notes[start_idx:start_idx + length],
            emotional_intent=motif.emotional_intent,
            development_history=motif.development_history + [DevelopmentType.FRAGMENTATION]
        )

    @staticmethod
    def phase_shift(motif: Motif, shift: float) -> Motif:
        """Create a phase-shifted version for minimalist textures."""
        if not motif.notes:
            return motif

        new_notes = []
        current_time = shift

        for note in motif.notes:
            new_note = Note(
                pitch=note.pitch,
                duration=note.duration,
                velocity=note.velocity,
                articulation=note.articulation,
                performance=PerformanceDirective(
                    timing_offset=current_time
                )
            )
            new_notes.append(new_note)
            current_time += note.duration

        return Motif(
            notes=new_notes,
            emotional_intent=motif.emotional_intent,
            development_history=motif.development_history + [DevelopmentType.PHASE_SHIFT]
        )
