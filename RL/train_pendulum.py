"""Backward-compatible wrapper for the pendulum training entrypoint.

Use `python RL/train.py` for the full sim2real workflow.
"""

from train import main


if __name__ == "__main__":
    main()
