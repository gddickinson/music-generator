"""The JS-vs-Python fixtures (scripts/make_fixtures.py) still match what the Python code produces.

The only platform-dependent step is SciPy's floating-point Savitzky-Golay smoothing: where the exact value is a whole
number, SciPy may land one ulp either side of it, and that can differ between macOS and Linux builds. Those cells are
the ones fractal_quirks.js records. If this machine rounds them differently, the composition checks are skipped
(with the reason) rather than failed; everything else must match exactly.
"""
import hashlib
import json
import re
from pathlib import Path

import pytest

from web.compose import compose
from web.fractal import near_integer_cells, python_degrees, quirks

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures"
EMOTIONS = ["ethereal", "dark", "joyful", "melancholic", "energetic", "peaceful"]


def recorded_quirks():
    text = (ROOT / "web/static/fractal_quirks.js").read_text()
    out = {}
    for e in EMOTIONS:
        m = re.search(rf"{e}: (\[.*?\]),\n", text)
        out[e] = [tuple(q) for q in json.loads(m.group(1))]
    return out


@pytest.mark.parametrize("emotion", EMOTIONS)
def test_quirks_are_only_ulp_rounding(emotion):
    near = near_integer_cells(emotion)
    assert all((r, c) in near for r, c, _ in recorded_quirks()[emotion])
    assert all((r, c) in near for r, c, _ in quirks(emotion))


def same_platform_rounding():
    return all(recorded_quirks()[e] == quirks(e) for e in EMOTIONS)


def test_fractal_degrees_current():
    if not same_platform_rounding():
        pytest.skip("SciPy rounds the near-integer smoothing cells differently on this platform")
    fx = json.loads((FX / "fractal_degrees.json").read_text())
    for e in EMOTIONS:
        assert "".join("0123456789abcd"[v] for row in python_degrees(e) for v in row) == fx[e]


def test_compositions_current():
    if not same_platform_rounding():
        pytest.skip("SciPy rounds the near-integer smoothing cells differently on this platform")
    for fx in json.loads((FX / "compositions.json").read_text()):
        r = compose(fx["case"])
        notes = json.dumps(r["notes"], separators=(",", ":"))
        assert hashlib.sha256(notes.encode()).hexdigest() == fx["notes_sha256"], fx["case"]
        assert r["midi_base64"] == fx["midi_base64"], fx["case"]
        assert len(r["sections"]) == len(fx["sections"]), fx["case"]
        for a, b in zip(r["sections"], fx["sections"]):
            assert a["duration"] == pytest.approx(b["duration"], rel=1e-9)
            assert a["tension"] == pytest.approx(b["tension"], rel=1e-9)
