"""Run the repo's CreativeComposer reproducibly and return what the web front end shows.

A piece is fixed by its parameters plus one seed: `random.seed(seed)` and `np.random.seed(seed)` before the
composer is built (the Python modules draw from both global generators). The browser engine
(static/composer_engine.js) consumes the same two streams in the same order, so it gives identical notes and MIDI.
"""
import base64
import contextlib
import io
import random
import threading

import numpy as np

from advanced_composer import CreativeComposer, CreativeComposerParams, MusicalForm
from fractal_melody_generator import EmotionalStyle, MusicalMode

DEFAULTS = {
    "form": "sonata", "base_tempo": 120, "key": "C", "mode": "ionian",
    "emotional_journey": ["peaceful", "energetic", "melancholic", "joyful"],
    "section_count": 4, "voice_count": 4, "seed": 0,
    "development_complexity": 0.7, "texture_density": 0.6, "modulation_frequency": 0.3,
}
FRACTIONS = ("development_complexity", "texture_density", "modulation_frequency")  # each 0..1
KEYS = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb']
LIMITS = {"base_tempo": (30, 240), "section_count": (1, 20), "voice_count": (1, 8), "seed": (0, 2**32 - 1)}
_lock = threading.Lock()  # the composer uses the global random generators


def normalise(params: dict) -> dict:
    """Merge with the defaults and validate (raises ValueError with a readable message)."""
    p = {**DEFAULTS, **{k: v for k, v in (params or {}).items() if k in DEFAULTS}}
    MusicalForm(p["form"])
    MusicalMode(p["mode"])
    if p["key"] not in KEYS:
        raise ValueError(f"unknown key {p['key']!r}")
    journey = [EmotionalStyle(e).value for e in p["emotional_journey"]]
    if not 1 <= len(journey) <= 8:
        raise ValueError("emotional_journey needs 1 to 8 emotions")
    p["emotional_journey"] = journey
    for name in FRACTIONS:
        p[name] = float(p[name])
        if not 0.0 <= p[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    for name, (lo, hi) in LIMITS.items():
        p[name] = int(p[name])
        if not lo <= p[name] <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}")
    return p


def compose(params: dict = None) -> dict:
    """Compose with the Python code. Returns params, form name, sections, notes (as added, before MIDIUtil's
    clean-up: [track, channel, pitch, tick, duration_ticks, velocity]), tempos [[tick, bpm]] and the MIDI file
    (base64)."""
    p = normalise(params)
    with _lock:
        random.seed(p["seed"])
        np.random.seed(p["seed"])
        cp = CreativeComposerParams(
            form=MusicalForm(p["form"]), base_tempo=p["base_tempo"], key=p["key"], mode=MusicalMode(p["mode"]),
            emotional_journey=[EmotionalStyle(e) for e in p["emotional_journey"]],
            section_count=p["section_count"], voice_count=p["voice_count"],
            **{name: p[name] for name in FRACTIONS},
            visualization_enabled=False, debug_mode=False)
        with contextlib.redirect_stderr(io.StringIO()):  # tqdm progress bar
            composer = CreativeComposer(cp)
            midi, _ = composer.generate_composition()
    notes = [[t - 1, e.channel, e.pitch, e.tick, e.duration, int(e.volume)]
             for t, track in enumerate(midi.tracks) for e in track.eventList if e.evtname == "NoteOn"]
    sections = []
    for s, (start, end) in zip(composer.form.sections, composer.section_spans):
        sections.append({
            "start": start, "duration": end - start, "tension": composer.calculate_section_tension(s),
            "tempo": s.tempo, "key": s.key, "texture": s.texture.name.lower(), "emotion": s.emotion.value,
            "dynamics": s.dynamics_base,
            "motifs": [{"notes": len(m.notes), "history": [h.value for h in m.development_history]}
                       for m in s.motifs]})
    buf = io.BytesIO()
    midi.writeFile(buf)
    return {"params": p, "form": composer.form.name.value, "sections": sections, "notes": notes,
            "midi_base64": base64.b64encode(buf.getvalue()).decode()}
