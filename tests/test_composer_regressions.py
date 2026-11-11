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
