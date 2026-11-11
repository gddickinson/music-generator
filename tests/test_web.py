"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific API tests ------------------------------------------------------------------------------------

def test_info_lists_the_composer_options():
    info = client.get("/api/info").json()
    assert [f["id"] for f in info["forms"]][:2] == ["sonata", "rondo"] and len(info["forms"]) == 11
    assert len(info["modes"]) == 7 and len(info["emotions"]) == 6
    assert info["defaults"]["form"] == "sonata"


def test_compose_runs_the_python_composer():
    r = client.post("/api/compose", json={"form": "arch", "seed": 4, "voice_count": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["form"] == "arch" and len(body["sections"]) == 5
    assert {n[0] for n in body["notes"]} <= {0, 1}
    import base64
    assert base64.b64decode(body["midi_base64"]).startswith(b"MThd")


def test_compose_is_reproducible():
    a = client.post("/api/compose", json={"form": "moment", "seed": 9}).json()
    b = client.post("/api/compose", json={"form": "moment", "seed": 9}).json()
    assert a["midi_base64"] == b["midi_base64"]


@pytest.mark.parametrize("bad", [{"form": "polka"}, {"mode": "blues"}, {"key": "H"}, {"voice_count": 0},
                                 {"emotional_journey": ["angry"]}, {"emotional_journey": []}])
def test_compose_rejects_bad_parameters(bad):
    assert client.post("/api/compose", json=bad).status_code == 422


def test_verify_cases_are_python_results():
    cases = client.get("/api/verify").json()
    assert len(cases) == 13 and all(len(c["midi_sha256"]) == 64 for c in cases)
