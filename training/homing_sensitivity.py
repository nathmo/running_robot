"""How precise does homing have to be before the standing controller works?

The m3 stand-in-place result is knife-edge: with cfg.reset_joint_noise at its shipped 0.03 rad
(+-1.72 deg) the SAME stance, controller and plant give 12.0 s on three seeds out of eight and
under a second on the rest, and it does not reduce to the initial lean (seeds 3 and 7 start at
toe-CoM +11.1 vs +11.2 mm and give 12.00 s vs 0.82 s). Since the measured homing-zero uncertainty on
the real robot is ~5 deg, the obvious question is whether a better homing method buys a working
controller.

That is a survival-vs-jitter curve, so measure it. TWO rungs, because which one binds is the whole
answer:
  m3  x/z/pitch free, y+roll+yaw LOCKED  -- the sagittal problem this controller was built for
  m6  everything free                    -- the real robot

If m6 stays at 0 even with the jitter switched off, homing precision is not the binding constraint
and improving it buys nothing on its own.

    python training/homing_sensitivity.py --seeds 16
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from push_ab import make_env, push_rollout, stance_of                  # noqa: E402
import stance as _stance                                              # noqa: E402

DEGS = (0.0, 0.25, 0.5, 1.0, 1.72, 3.0, 5.0)


def run(milestone, deg, seeds, horizon, dv=0.0):
    env = make_env(milestone, _stance.BAL, horizon + 1.0, "reflex", key_ctrl=_stance.SHIPPED)
    env.cfg.reset_joint_noise = float(np.radians(deg))
    rows = [push_rollout(env, s, dv, "reflex", None, t_push=1.0, horizon_s=horizon)
            for s in seeds]
    return rows, stance_of(env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--horizon", type=float, default=12.0)
    ap.add_argument("--milestones", nargs="+", default=["m3", "m6"])
    ap.add_argument("--dv", type=float, default=0.0, help="backward push magnitude (0 = none)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    out = []

    for m in args.milestones:
        print("\n=== %s === (%d paired seeds, %.0f s horizon, push %.2f m/s)"
              % (m, len(seeds), args.horizon, args.dv))
        print("  homing +-deg   survivors   median t_alive   worst")
        for deg in DEGS:
            rows, (bz, tc) = run(m, deg, seeds, args.horizon, dv=-abs(args.dv))
            sv = sum(r["survived"] for r in rows)
            med = float(np.median([r["t_alive"] for r in rows]))
            worst = float(min(r["t_alive"] for r in rows))
            print("     %5.2f       %2d/%-2d        %6.2f s        %5.2f s"
                  % (deg, sv, len(seeds), med, worst))
            out.append(dict(milestone=m, deg=deg, survived=sv, n=len(seeds),
                            median=med, worst=worst, base_z=bz, toe_com=tc,
                            t_alive=[r["t_alive"] for r in rows]))
            sys.stdout.flush()

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print("\nwrote " + args.out)
    print("\nRead m6 first. Homing precision can only be the binding constraint on a rung that "
          "\nworks at zero jitter; if m6 is 0/N with the jitter off, better homing buys nothing.")


if __name__ == "__main__":
    main()
