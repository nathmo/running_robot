"""Does the warm-start VecNormalize rejuvenation (train.rejuvenate_obs_rms: count cap + variance
floor) break a same-stage policy? Runs hook-free stochastic episodes of a checkpoint with its own
stats, then with the rejuvenated stats.

    python walk_mit/monitor/rejuv_probe.py RUN CKPT_STEM [--episodes 2] [--assist 0.8]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from evaluate import build  # noqa: E402
from train import rejuvenate_obs_rms  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("ckpt")
ap.add_argument("--episodes", type=int, default=2)
ap.add_argument("--assist", type=float, default=0.8)
ap.add_argument("--count-cap", type=float, default=100000.0)
ap.add_argument("--var-floor", type=float, default=0.01)
ap.add_argument("--greedy", action="store_true")
args = ap.parse_args()
run = Path(args.run)


def episodes(label, mutate):
    model, venv, raw = build(run, None, str(run / args.ckpt))
    raw.set_dr_scale(0.0)
    raw.set_pitch_assist(args.assist)
    if mutate:
        v = venv.obs_rms.var.copy()
        rejuvenate_obs_rms(venv, args.count_cap, args.var_floor)
        n_floored = int(np.sum(v < args.var_floor))
        print(f"   rejuvenated: {n_floored} of {v.size} obs dims had var < {args.var_floor} "
              f"(min var {v.min():.2e}); floored dims: {np.where(v < args.var_floor)[0].tolist()[:40]}")
    lens, dist = [], []
    for e in range(args.episodes):
        venv.seed(1000 + e)
        obs = venv.reset()
        n, sprint = 0, None
        while True:
            a, _ = model.predict(obs, deterministic=args.greedy)
            obs, r, d, info = venv.step(a)
            n += 1
            sprint = info[0].get("sprint", sprint)
            if d[0]:
                break
        lens.append(n)
        dist.append(np.nan if sprint is None else sprint["d"])
    print(f"[{label}] {'greedy' if args.greedy else 'stochastic'} ticks {lens}  x {np.round(dist, 1).tolist()} m")


episodes("own stats", False)
episodes("rejuvenated", True)
