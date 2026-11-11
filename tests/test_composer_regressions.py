"""Regression tests for bugs fixed in the composer (see SESSION_LOG.md, 2026-11-11)."""
import contextlib
import io
import random

import numpy as np
import pytest

from advanced_composer import (CreativeComposer, CreativeComposerParams, MusicalForm,
                               MotifDevelopment, Motif, Note, EmotionalStyle)
from fractal_melody_generator import (FractalMelodyGenerator, MelodyParams, MusicalMode)


def compose(form, seed, **kw):
    random.seed(seed)
    np.random.seed(seed)
    params = CreativeComposerParams(form=form, visualization_enabled=False, debug_mode=False, **kw)
    with contextlib.redirect_stderr(io.StringIO()):
        composer = CreativeComposer(params)
        midi, _ = composer.generate_composition()
    return composer, midi

@pytest.mark.parametrize("form", list(MusicalForm))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_form_writes_a_midi_file(form, seed):
    """Overlapping sections used to put two notes with the same start on one track; MIDIUtil
    then dropped one note-on but kept its note-off and writeFile raised IndexError
    (the default sonata form failed for every seed)."""
    _, midi = compose(form, seed)
    buf = io.BytesIO()
    midi.writeFile(buf)
    assert buf.getvalue().startswith(b"MThd")


def test_melody_generator_follows_emotion_changes():
    """generate_melody used the emotion given at construction even after params.emotion
    changed, so every theme of a composition used the first emotion's fractal."""
    gen = FractalMelodyGenerator(MelodyParams(mode=MusicalMode.IONIAN, emotion=EmotionalStyle.PEACEFUL))
    np.random.seed(0)
    peaceful = gen.generate_melody()
    gen.params.emotion = EmotionalStyle.ENERGETIC
    np.random.seed(0)
    energetic = gen.generate_melody()
    fresh = FractalMelodyGenerator(MelodyParams(mode=MusicalMode.IONIAN, emotion=EmotionalStyle.ENERGETIC))
    np.random.seed(0)
    assert energetic == fresh.generate_melody()
    assert energetic != peaceful


def test_minimalist_form_uses_melody_pitches():
    """initialize_minimalist_form sliced the (melody, timing, velocities) tuple instead of the
    melody, so its 'notes' had lists as pitches (only 3 notes, pitch 0 among them)."""
    composer, _ = compose(MusicalForm.MINIMALIST, 0)
    base = composer.form.sections[0].motifs[0]
    assert len(base.notes) == 8
    assert all(isinstance(n.pitch, int) and 60 <= n.pitch < 84 for n in base.notes)


def test_phase_shift_offsets_by_shift_only():
    """phase_shift stored the running onset as timing_offset; add_motif_to_midi adds the
    running time again, so shifted notes landed at 2x their onset (gaps, not a phase)."""
    motif = Motif(notes=[Note(pitch=60 + i, duration=0.25) for i in range(4)],
                  emotional_intent=EmotionalStyle.PEACEFUL)
    shifted = MotifDevelopment.phase_shift(motif, 0.125)
    assert [n.performance.timing_offset for n in shifted.notes] == [0.125] * 4
