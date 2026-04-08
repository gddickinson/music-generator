"""Smoke tests for music generator modules."""

import sys
import os
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestMusicalTypes(unittest.TestCase):
    """Test musical type definitions and dataclasses."""

    def test_imports(self):
        from musical_types import (
            MusicalForm, Articulation, TextureType, DevelopmentType,
            PerformanceDirective, Note, Motif, Section, Form,
            CreativeComposerParams
        )
        self.assertIsNotNone(MusicalForm.SONATA)
        self.assertIsNotNone(Articulation.LEGATO)

    def test_note_creation(self):
        from musical_types import Note, Articulation
        note = Note(pitch=60, duration=0.5, velocity=80)
        self.assertEqual(note.pitch, 60)
        self.assertEqual(note.duration, 0.5)
        self.assertEqual(note.articulation, Articulation.LEGATO)

    def test_motif_duration(self):
        from musical_types import Note, Motif
        from fractal_melody_generator import EmotionalStyle
        notes = [Note(pitch=60, duration=0.25), Note(pitch=62, duration=0.5)]
        motif = Motif(notes=notes, emotional_intent=EmotionalStyle.PEACEFUL)
        self.assertAlmostEqual(motif.duration, 0.75)

    def test_creative_composer_params_defaults(self):
        from musical_types import CreativeComposerParams, MusicalForm
        params = CreativeComposerParams()
        self.assertEqual(params.form, MusicalForm.SONATA)
        self.assertEqual(params.base_tempo, 120)
        self.assertEqual(params.voice_count, 4)


class TestMotifDevelopment(unittest.TestCase):
    """Test motif development techniques."""

    def _make_motif(self, pitches=None):
        from musical_types import Note, Motif, DevelopmentType
        from fractal_melody_generator import EmotionalStyle
        if pitches is None:
            pitches = [60, 62, 64, 65, 67]
        notes = [Note(pitch=p, duration=0.25) for p in pitches]
        return Motif(notes=notes, emotional_intent=EmotionalStyle.PEACEFUL)

    def test_invert(self):
        from motif_development import MotifDevelopment
        motif = self._make_motif([60, 64, 67])
        inverted = MotifDevelopment.invert(motif)
        # First note stays the same (center), others invert
        self.assertEqual(inverted.notes[0].pitch, 60)
        self.assertEqual(inverted.notes[1].pitch, 56)  # 60 - (64-60) = 56
        self.assertEqual(inverted.notes[2].pitch, 53)  # 60 - (67-60) = 53

    def test_retrograde(self):
        from motif_development import MotifDevelopment
        motif = self._make_motif([60, 62, 64])
        retro = MotifDevelopment.retrograde(motif)
        self.assertEqual([n.pitch for n in retro.notes], [64, 62, 60])

    def test_augment(self):
        from motif_development import MotifDevelopment
        motif = self._make_motif([60])
        augmented = MotifDevelopment.augment(motif, factor=2.0)
        self.assertAlmostEqual(augmented.notes[0].duration, 0.5)

    def test_diminish(self):
        from motif_development import MotifDevelopment
        motif = self._make_motif([60])
        diminished = MotifDevelopment.diminish(motif, factor=0.5)
        self.assertAlmostEqual(diminished.notes[0].duration, 0.125)

    def test_fragment(self):
        from motif_development import MotifDevelopment
        motif = self._make_motif([60, 62, 64, 65, 67])
        fragment = MotifDevelopment.fragment(motif, start_idx=1, length=3)
        self.assertEqual(len(fragment.notes), 3)
        self.assertEqual([n.pitch for n in fragment.notes], [62, 64, 65])

    def test_empty_motif_handling(self):
        from motif_development import MotifDevelopment
        from musical_types import Motif
        from fractal_melody_generator import EmotionalStyle
        empty = Motif(notes=[], emotional_intent=EmotionalStyle.PEACEFUL)
        self.assertEqual(len(MotifDevelopment.invert(empty).notes), 0)
        self.assertEqual(len(MotifDevelopment.retrograde(empty).notes), 0)


class TestVoiceLeading(unittest.TestCase):
    """Test voice leading optimization."""

    def test_voice_crossing_detection(self):
        from voice_leading import VoiceLeading
        vl = VoiceLeading()
        # No crossing: sorted ascending
        self.assertFalse(vl.check_voice_crossing([48, 55, 60, 67]))
        # Crossing: 60 > 55
        self.assertTrue(vl.check_voice_crossing([48, 60, 55, 67]))

    def test_parallel_motion_detection(self):
        from voice_leading import VoiceLeading
        vl = VoiceLeading()
        # Parallel fifths: both pairs 7 semitones apart
        self.assertTrue(vl.check_parallel_motion([60, 67], [62, 69]))
        # No parallel fifths
        self.assertFalse(vl.check_parallel_motion([60, 64], [62, 67]))

    def test_generate_voicings(self):
        from voice_leading import VoiceLeading
        vl = VoiceLeading()
        voicings = vl.generate_voicings([60, 64, 67])
        self.assertTrue(len(voicings) > 1)
        # Base voicing should be present
        self.assertIn([60, 64, 67], voicings)


class TestFractalMelodyGenerator(unittest.TestCase):
    """Test fractal melody generator."""

    def test_generator_creation(self):
        from fractal_melody_generator import (
            FractalMelodyGenerator, MelodyParams, MusicalMode, EmotionalStyle
        )
        params = MelodyParams(mode=MusicalMode.IONIAN, emotion=EmotionalStyle.PEACEFUL)
        gen = FractalMelodyGenerator(params)
        self.assertIsNotNone(gen)

    def test_melody_generation(self):
        from fractal_melody_generator import (
            FractalMelodyGenerator, MelodyParams, MusicalMode, EmotionalStyle
        )
        params = MelodyParams(
            mode=MusicalMode.DORIAN,
            emotion=EmotionalStyle.ENERGETIC,
            resolution=50
        )
        gen = FractalMelodyGenerator(params)
        result = gen.generate_melody()
        self.assertIsNotNone(result)
        # Should return melody, timing, velocities
        self.assertEqual(len(result), 3)


class TestHarmonyEngine(unittest.TestCase):
    """Test harmony engine."""

    def test_engine_creation(self):
        from harmony_engine import HarmonyEngine, HarmonyParams, HarmonyStyle
        params = HarmonyParams(style=HarmonyStyle.CLASSICAL)
        engine = HarmonyEngine(params)
        self.assertIsNotNone(engine)


class TestMultiFractalMusic(unittest.TestCase):
    """Test multi-fractal music (renamed from multi-fractal-music.py)."""

    def test_import_underscore_version(self):
        from multi_fractal_music import MultiFractalMusic, MultiFractalParams
        params = MultiFractalParams(resolution=20)
        gen = MultiFractalMusic(params)
        self.assertIsNotNone(gen)

    def test_mandelbrot_generation(self):
        from multi_fractal_music import MultiFractalMusic, MultiFractalParams
        params = MultiFractalParams(resolution=20)
        gen = MultiFractalMusic(params)
        result = gen.generate_mandelbrot()
        self.assertEqual(result.shape, (20, 20))


if __name__ == "__main__":
    unittest.main()
