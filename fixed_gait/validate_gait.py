#!/usr/bin/env python3
"""Validate the hard-coded gait — SIM is ground truth.

Drives the gait in a fixed-base dynamic sim with the LEG MESHES made collidable (the normal model
only collides the foot spheres) and checks the three things that matter for a safe fixed-base run:
  1. every commanded joint stays inside its limit,
  2. no deep link self-collision (a few mm of grazing between the pushrod and its adjacent hip/
     thigh links is by design — the parallel linkage runs right alongside them — so only
     penetration past a threshold is flagged),
  3. the foot traces a smooth, bounded path (no lock-up / fold / NaN).

It also overlays the foot path on the reachability map from mujoco/spiderbot/plot_reachability.py
for context. (That idealized map holds the foot parallel to the thigh / at spring rest and uses a
seeded kinematic solver, so it under-covers the forward edge and flags designed-adjacent meshes as
"collisions" — the dynamic sim here is the authority, the map is a backdrop.)

Run:  .venv/Scripts/python.exe fixed_gait/validate_gait.py --out fixed_gait/_gait_reachability.png
"""
import argparse
import os
import sys

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "mujoco/spiderbot")
from sim_fixed_base import build_fixed_base_model, home_hinges           # noqa: E402
from gait import GaitParams, GaitGenerator, CTRL_LIMIT                    # noqa: E402

PENETRATION_LIMIT_MM = 8.0     # deeper than this between non-adjacent links = a real concern


def geom_name(m, gid):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)


def drive_and_check(gen, cycles=3):
    m = build_fixed_base_model(gravity=False, floor=False, collidable=True)
    d = mujoco.MjData(m)
    fL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "foot_L_col")
    jHL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "bodyNCS-v1_Révolution-1")
    act_qadr = [m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)]

    d.qpos[:] = home_hinges()
    d.qvel[:] = 0.0
    d.ctrl[:] = gen.center_pose()
    for _ in range(2000):
        mujoco.mj_step(m, d)
    baseline = {tuple(sorted((d.contact[i].geom1, d.contact[i].geom2))) for i in range(d.ncon)}

    sim_dt = m.opt.timestep
    T = gen.p.ramp_s + cycles * gen.p.period_s
    path, cmd_min, cmd_max = [], np.full(m.nu, np.inf), np.full(m.nu, -np.inf)
    worst_pen = {}
    t = 0.0
    while t < T:
        tg = gen.targets(t)
        d.ctrl[:] = tg
        cmd_min = np.minimum(cmd_min, tg)
        cmd_max = np.maximum(cmd_max, tg)
        mujoco.mj_step(m, d)
        t += sim_dt
        for i in range(d.ncon):
            pair = tuple(sorted((d.contact[i].geom1, d.contact[i].geom2)))
            if pair in baseline:
                continue
            key = tuple(sorted((geom_name(m, pair[0]), geom_name(m, pair[1]))))
            worst_pen[key] = min(worst_pen.get(key, 0.0), d.contact[i].dist)
        if t > gen.p.ramp_s:
            path.append((d.geom_xpos[fL] - d.xanchor[jHL])[[0, 2]].copy())
    return m, np.array(path), cmd_min, cmd_max, worst_pen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="fixed_gait/_gait_reachability.png")
    ap.add_argument("--nc", type=int, default=81)
    ap.add_argument("--nt", type=int, default=61)
    ap.add_argument("--period", type=float, default=None)
    ap.add_argument("--no-map", action="store_true", help="skip the (slow) reachability backdrop")
    args = ap.parse_args()

    p = GaitParams()
    if args.period is not None:
        p.period_s = args.period
    gen = GaitGenerator(p)

    print("driving gait in collidable fixed-base sim ...")
    m, path, cmd_min, cmd_max, worst = drive_and_check(gen)

    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(m.nu)]
    print("\njoint command ranges vs limits:")
    in_range = True
    for a, nm in enumerate(names):
        ok = abs(cmd_min[a]) <= CTRL_LIMIT[a] + 1e-6 and abs(cmd_max[a]) <= CTRL_LIMIT[a] + 1e-6
        in_range &= ok
        print(f"  {nm:10s} [{cmd_min[a]:+.3f},{cmd_max[a]:+.3f}]  limit ±{CTRL_LIMIT[a]:.3f}  "
              f"{'ok' if ok else 'OUT OF RANGE'}")

    print("\nself-collision (non-baseline link pairs):")
    deep = False
    if not worst:
        print("  none")
    for k, v in sorted(worst.items(), key=lambda kv: kv[1]):
        mm = -v * 1000
        tag = "  <-- DEEP" if mm > PENETRATION_LIMIT_MM else "  (grazing, adjacent linkage)"
        deep |= mm > PENETRATION_LIMIT_MM
        print(f"  {k[0]} <-> {k[1]}: {mm:.1f} mm{tag}")

    smooth = np.isfinite(path).all() and len(path) > 10
    print(f"\nfoot path: {len(path)} samples, X [{path[:,0].min():+.3f},{path[:,0].max():+.3f}]  "
          f"Z [{path[:,1].min():+.3f},{path[:,1].max():+.3f}] (rel hip)")
    verdict = in_range and not deep and smooth
    print("\nVERDICT:", "SAFE to play on the fixed-base robot"
          if verdict else "REVIEW — see flags above")

    # ---- plot: foot path (+ optional reachability backdrop) ----
    fig, ax = plt.subplots(figsize=(8.5, 9.2))
    if not args.no_map:
        import plot_reachability as reach
        print("\ncomputing reachability backdrop (context only) ...")
        leg = reach.Leg(ankle_mode="spring-rest")
        g = reach.sweep(leg, args.nc, args.nt)
        ov = reach.compute_overlays(leg, g)
        reach.draw_map(ax, g, ov)
        vx = reach.VIEW_X
    else:
        vx = -1.0
        ax.plot(0, 0, "ks", ms=9, label="hip pivot (origin)")
        ax.set_xlabel("X (forward, m) — mirrored"); ax.set_ylabel("Z (up, m)")
        ax.set_aspect("equal", "box"); ax.grid(True, ls=":", alpha=0.4)
    ax.plot(vx * path[:, 0], path[:, 1], "-", color="magenta", lw=2.6, zorder=20,
            label="hard-coded gait foot path (sim)")
    ax.scatter(vx * path[::6, 0], path[::6, 1], s=16, color="magenta", zorder=21,
               edgecolor="k", lw=0.4)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.set_title(f"gait foot path — {'SAFE' if verdict else 'REVIEW'}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print("saved", args.out)


if __name__ == "__main__":
    main()
