# Creative Music Generator

Algorithmic composition system using fractal melodies, harmony engines, and genetic algorithms.

## Quick Reference
@INTERFACE.md

## Running
- Main composition: `python advanced_composer.py`
- Simpler composition: `python main_composer.py`
- Multi-fractal: `python multi_fractal_music.py`
- Tests: `python -m pytest test_music_generator.py tests/ -q` (tests/ needs node for the JS engine tests)
- Web app: `python run_web.py` (http://127.0.0.1:8166); static site: `python scripts/build_static.py`
- The JS engine in `web/static/` mirrors `advanced_composer.py` + `fractal_melody_generator.py` exactly: after changing
  either, update the JS port and run `python scripts/make_fixtures.py` (see INTERFACE.md)

## Dependencies
See `requirements.txt` or `pyproject.toml`.
