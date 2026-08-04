"""Measure the foot workspace envelope a trained policy ACTUALLY uses, vs the measured reachable box
(cpg_foot_lut.npz). Tells us whether the one-legged parked leg leaves the real robot workspace and
where to set a workspace-kill termination threshold (so it catches the exploit but not a good gait).

Frame matches build_cpg_lut.measure_forward: toe (x,z) in the BASE frame, R_base.T @ (toe - base),
minus the LUT's nominal_toe -> (dx fore-aft, dz lift). dz>0 = foot lifted up (shorter leg).

  python training/diagnose_workspace.py --run training/runs/m6_slow_gait --episodes 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from evaluate import build     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--dx-max", type=float, default=0.30)
    ap.add_argument("--dz-max", type=float, default=0.12)   # lift ceiling (measured box is 0.10)
    ap.add_argument("--dz-min", type=float, default=-0.15)  # extension floor (generous)
    args = ap.parse_args()

    run = Path(args.run)
    model, venv, raw = build(run, args.preset, args.checkpoint)
    lut = np.load(str(PKG / "model" / "cpg_foot_lut.npz"), allow_pickle=True)
    ref = np.asarray(lut["nominal_toe"], float)             # (3,) base-frame nominal toe

    rec = {0: [], 1: []}

    def on_ctrl():
        base = raw.data.xpos[raw.base_id]
        R = raw.data.xmat[raw.base_id].reshape(3, 3)
        for fi, g in enumerate(raw.foot_gids):
            tb = R.T @ (raw.data.geom_xpos[g] - base)
            rec[fi].append((tb[0] - ref[0], tb[2] - ref[2]))
    raw.on_control_step = on_ctrl

    for _ in range(args.episodes):
        obs = venv.reset()
        done = [False]
        while not done[0]:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = venv.step(a)

    print(f"nominal_toe (base frame) = {np.round(ref, 4)}")
    print(f"measured reachable box: dx +-0.30 m, dz 0..0.10 m (folded branch past dz>0.40)")
    print(f"candidate kill box: |dx|<={args.dx_max}, {args.dz_min}<=dz<={args.dz_max}")
    for fi in (0, 1):
        arr = np.array(rec[fi])
        if not len(arr):
            continue
        dx, dz = arr[:, 0], arr[:, 1]

        def q(a):
            return (f"min {a.min():+.3f}  p5 {np.percentile(a,5):+.3f}  "
                    f"med {np.median(a):+.3f}  p95 {np.percentile(a,95):+.3f}  max {a.max():+.3f}")
        oob = np.mean((np.abs(dx) > args.dx_max) | (dz > args.dz_max) | (dz < args.dz_min))
        print(f" foot {'L' if fi==0 else 'R'} ({len(arr)} steps)")
        print(f"    dx: {q(dx)}")
        print(f"    dz: {q(dz)}")
        print(f"    frac OUTSIDE kill box = {oob:.3f}")


if __name__ == "__main__":
    main()
