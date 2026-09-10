"""Which domain-randomization axis makes MuJoCo go unstable? The v2_s1 controls logged 2-3
"simulation is unstable" warnings at 7 M steps (dr_scale 0) and 55-99 by 50 M (dr_scale 0.6-0.7).
Each is a MuJoCo auto-reset mid-episode: a corrupted transition in the rollout buffer.

    python walk_mit/monitor/dr_stability_probe.py --dr 1.0 --episodes 150
    python walk_mit/monitor/dr_stability_probe.py --dr 1.0 --set dr_loop_k=0.0        # ablate one axis
    python walk_mit/monitor/dr_stability_probe.py --dr 1.0 --set "dr_delay_ms_range=(12.0,12.0)"

Counts episodes in which MuJoCo's warning counters advance (BADQACC etc.), prints the first-warning
tick, and compares the applied DR draw of the failing episodes with the population mean per key
so the culprit axis stands out without a full ablation sweep.
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from config import get_config  # noqa: E402
from env import DashEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="v2_s1")
ap.add_argument("--dr", type=float, default=1.0)
ap.add_argument("--episodes", type=int, default=150)
ap.add_argument("--ticks", type=int, default=200)
ap.add_argument("--action", default="random", choices=["random", "zero"])
ap.add_argument("--set", nargs="*", default=[], help="FIELD=VALUE config overrides (python literals)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

cfg = get_config(args.preset)
for kv in args.set:
    k, v = kv.split("=", 1)
    assert hasattr(cfg, k), k
    setattr(cfg, k, eval(v))
env = DashEnv(cfg)
env.set_dr_scale(args.dr)
rng = np.random.default_rng(args.seed)


def warn_count():
    return int(sum(w.number for w in env.data.warning))


rows, fails, first_tick = [], [], []
t0 = time.time()
for e in range(args.episodes):
    env.reset(seed=args.seed * 100000 + e)
    w0 = warn_count()
    draw = {k: v for k, v in getattr(env._dr, "last", {}).items()}
    failed = False
    for t in range(args.ticks):
        if args.action == "random":
            a = rng.uniform(-1.0, 1.0, env.action_space.shape).astype(np.float32)
        else:
            a = np.zeros(env.action_space.shape, np.float32)
        _, _, term, trunc, _ = env.step(a)
        if warn_count() > w0:
            failed = True
            first_tick.append(t)
            break
        if term or trunc:
            break
    rows.append((draw, failed))
    if failed:
        fails.append(draw)

n_fail = len(fails)
print(f"[{args.preset} dr={args.dr} {args.action} set={args.set}] unstable episodes {n_fail}/{args.episodes}"
      f"  first-warning tick median {np.median(first_tick) if first_tick else '-'}  ({time.time() - t0:.0f} s)")
if n_fail:
    keys = sorted({k for d, _ in rows for k, v in d.items() if np.isscalar(v) and not isinstance(v, (str, bool))})
    print(f"   {'draw key':22s} {'mean(all)':>10s} {'mean(fail)':>11s} {'z':>6s}")
    for k in keys:
        allv = np.array([float(d[k]) for d, _ in rows if k in d])
        failv = np.array([float(d[k]) for d in fails if k in d])
        if len(failv) == 0 or allv.std() == 0:
            continue
        z = (failv.mean() - allv.mean()) / (allv.std() / np.sqrt(len(failv)))
        flag = " <--" if abs(z) > 2.5 else ""
        print(f"   {k:22s} {allv.mean():10.4f} {failv.mean():11.4f} {z:6.1f}{flag}")
    for d in fails[:3]:
        print("   failing draw:", {k: (round(float(v), 3) if np.isscalar(v) and not isinstance(v, (str, bool)) else v)
                                    for k, v in d.items()})
