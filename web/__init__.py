"""Web interface for the Creative Music Generator.

Contents:
    app.py      FastAPI app: the front end in static/ and the JSON API under /api (STATIC_ENDPOINTS).
    compose.py  Runs the Python CreativeComposer with a seed and returns notes, sections and the MIDI file.
    fractal.py  The fractal scale-degree grids exactly as the Python generator computes them (for the JS checks).
    static/     The browser app, including the JavaScript port of the composer (see INTERFACE.md).
"""
