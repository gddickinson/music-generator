"""Start the Creative Music Generator web app on http://127.0.0.1:8166 (local only).

    pip install -r requirements.txt -r requirements-web.txt
    python run_web.py [--port 8166]
"""
import argparse

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8166)
    a = ap.parse_args()
    print(f"Creative Music Generator web app: http://127.0.0.1:{a.port}")
    uvicorn.run("web.app:app", host="127.0.0.1", port=a.port)
