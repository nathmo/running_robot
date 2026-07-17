"""Entry point:  .venv/Scripts/python.exe -m orchestrator.web [--port 8800]"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description="Local web UI for RL training runs")
    ap.add_argument("--port", type=int, default=8800)
    args = ap.parse_args()

    os.chdir(ROOT)                       # rl.train / framework paths are repo-root relative
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from orchestrator.web.app import app
    print(f"\nRL run orchestrator: http://127.0.0.1:{args.port}/  (runs in rl/runs/)")
    app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
