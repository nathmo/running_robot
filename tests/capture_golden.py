"""Capture golden reward/termination traces of Dash01Env for refactor-parity testing.

Run BEFORE any env.py refactor to record ground truth, then tests/test_env_parity.py
asserts the refactored env reproduces these traces bit-for-bit:

    .venv/Scripts/python.exe -m tests.capture_golden

One trace per (preset, seed): a fixed seeded action sequence is replayed open-loop and
every step's reward, per-term dict, termination flags and obs checksum are recorded.
Presets chosen to cover every reward branch: speed_mode (m2), command-tracking
(m2_walk), sprint+fourier macro-steps (m1_sprint_fourier), stand (m1_stand).
"""
import json
from pathlib import Path

import numpy as np

from rl.config import get_config
from rl.env import Dash01Env

GOLDEN_DIR = Path(__file__).parent / "golden"
PRESETS = ["m2", "m2_walk", "m1_sprint_fourier", "m1_stand"]
N_STEPS = 120          # env.step calls (fourier macro-steps cover ~1 cycle each)
SEED = 1234


def trace(preset: str) -> dict:
    cfg = get_config(preset)
    env = Dash01Env(cfg)
    obs, _ = env.reset(seed=SEED)
    rng = np.random.default_rng(SEED)
    rows = {"reward": [], "terminated": [], "truncated": [], "obs_sum": [], "terms": []}
    for _ in range(N_STEPS):
        a = rng.uniform(-1.0, 1.0, env.action_space.shape).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        rows["reward"].append(float(r))
        rows["terminated"].append(bool(term))
        rows["truncated"].append(bool(trunc))
        rows["obs_sum"].append(float(np.asarray(obs, dtype=np.float64).sum()))
        rows["terms"].append({k: float(v) for k, v in info["reward_terms"].items()})
        if term or trunc:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    return rows


def main():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for preset in PRESETS:
        out = GOLDEN_DIR / f"{preset}.json"
        data = trace(preset)
        out.write_text(json.dumps(data))
        print(f"[golden] {preset}: {len(data['reward'])} steps -> {out}")


if __name__ == "__main__":
    main()
