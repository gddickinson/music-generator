"""FastAPI web app for the Creative Music Generator: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8166, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below. The
composer itself runs in the browser (static/composer_engine.js, a port of advanced_composer.py); POST /compose runs
the Python code for the same settings so the Verify page can compare them (local app only).
"""
import base64
import hashlib
import json
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.compose import DEFAULTS, KEYS, LIMITS, compose

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Creative Music Generator")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# Form descriptions follow the README and the docstrings of the initialisers in advanced_composer.py.
FORMS = [
    ("sonata", "Sonata", "Exposition (energetic primary theme, peaceful secondary theme in the dominant), a dark "
     "polyphonic development, recapitulation in the tonic and, half the time, a coda."),
    ("rondo", "Rondo", "No rondo initialiser yet: the Python falls back to the mixed form."),
    ("theme_and_variations", "Theme and variations", "No initialiser yet: the Python falls back to the mixed form."),
    ("fugue", "Fugue", "No fugue initialiser yet: the Python falls back to the mixed form."),
    ("minimalist", "Minimalist", "An eight-note cell and phase-shifted copies, adding a voice per section while "
     "tempo and dynamics rise."),
    ("through_composed", "Through-composed", "One motif per emotion, each section developed from the previous "
     "one (augmentation, diminution, fragmentation)."),
    ("aleatoric", "Aleatoric", "A pool of motifs and their transformations; each section draws three by chance, "
     "with chance tempo, texture and dynamics."),
    ("mixed", "Mixed", "Each emotion borrows sections from a randomly chosen form, sometimes developed further."),
    ("arch", "Arch", "Palindromic ABCBA: peaceful, energetic, dark climax, energetic, peaceful."),
    ("moment", "Moment", "Stockhausen-inspired independent moments: random emotion, texture, tempo and key."),
    ("process", "Process", "Reich/Glass-inspired phasing: a motif against itself shifted by an extra eighth of a "
     "beat per section, with an augmented layer from the fourth phase."),
]
MODES = ["ionian", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"]
EMOTIONS = ["ethereal", "dark", "joyful", "melancholic", "energetic", "peaceful"]
TEXTURES = ["monophonic", "homophonic", "polyphonic", "heterophonic", "contrapuntal"]

# Pieces composed by the Python code and checked again in the browser on the Verify page.
VERIFY_CASES = [{"form": f, "seed": 1} for f, _, _ in FORMS] + [
    {"form": "mixed", "seed": 2026, "mode": "dorian", "section_count": 8, "voice_count": 6, "base_tempo": 96,
     "emotional_journey": ["dark", "ethereal", "joyful", "melancholic", "energetic", "peaceful"]},
    {"form": "moment", "seed": 31337, "mode": "lydian", "key": "Eb", "section_count": 10, "voice_count": 3},
]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary, the composer's parameters and their ranges (from the Python code)."""
    from fractal_melody_generator import FractalMelodyGenerator, MelodyParams, MusicalMode, EmotionalStyle
    gen = FractalMelodyGenerator(MelodyParams(mode=MusicalMode.IONIAN, emotion=EmotionalStyle.PEACEFUL))
    emotions = {e.value: {k: v for k, v in p.items()} for e, p in gen.emotional_params.items()}
    return {
        "title": "Creative Music Generator",
        "summary": "An algorithmic composition system that combines fractal patterns, musical genetics and "
                   "advanced form structures to create complex musical compositions.",
        "forms": [{"id": f, "name": n, "description": d} for f, n, d in FORMS],
        "modes": MODES, "emotions": EMOTIONS, "emotion_params": emotions, "textures": TEXTURES, "keys": KEYS,
        "mode_intervals": {m.value: iv for m, iv in gen.mode_intervals.items()},
        "defaults": DEFAULTS, "limits": LIMITS, "verify_cases": len(VERIFY_CASES),
    }


@app.get("/api/verify")
def verify():
    """VERIFY_CASES composed by the Python code: notes and MIDI as SHA-256 digests plus a few first notes."""
    out = []
    for case in VERIFY_CASES:
        r = compose(case)
        midi = base64.b64decode(r["midi_base64"])
        out.append({"params": r["params"], "form": r["form"], "n_notes": len(r["notes"]),
                    "n_sections": len(r["sections"]), "midi_bytes": len(midi),
                    "notes_sha256": hashlib.sha256(json.dumps(r["notes"], separators=(",", ":")).encode()).hexdigest(),
                    "midi_sha256": hashlib.sha256(midi).hexdigest(), "first_notes": r["notes"][:6]})
    return out


@app.post("/api/compose")
def compose_route(params: dict = Body(default={})):
    """Compose with the Python CreativeComposer (same parameters and seed as the browser engine)."""
    try:
        return compose(params)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e))


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/verify"]
