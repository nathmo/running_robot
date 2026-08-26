#!/usr/bin/env python3
"""Generate the FK lookup table for the web UI (fixed_gait/webui/fk_lut.npz). DESKTOP ONLY.

Needs mujoco (the Pi doesn't have it): the closed 4-bar loop has no closed form, so every grid
cell is Newton-solved against the model via plot_reachability.Leg — exactly the machinery of the
reachability study (sweep(), plot_reachability.py:294-335), but storing the FULL linkage node
geometry per cell instead of only the foot tip.

    python mujoco/dash01/gen_fk_lut.py                 # -> fixed_gait/webui/fk_lut.npz
    python mujoco/dash01/gen_fk_lut.py --check         # + interpolation-accuracy self-check

Contents of fk_lut.npz:
    cam[nc], thigh[nt]        grid axes (rad, model joint space, LEFT leg; right leg mirrors)
    nodes[nc,nt,7,2] float32  XZ (m, rel. hip pivot) of: cam, thigh, push, knee, ank, ptip, ee
    valid[nc,nt] bool         assembles & no self-collision & ankle in range
    feas[nc,nt]  bool         assembles (the physical never-exceed band)
    ankle_mode                'spring-rest' (matches the real passive ankle) or 'parallel-thigh'

The cam axis defaults to the FULL CRANK window [-90, +270] deg, NOT the model's +-86 deg joint
range: that range is a CAD guess, while the real cam is a crank (the 4-bar assembles at every
cam angle; recorded hardware sweeps span ~245 deg landing on model cam ~[-64, +183] deg). The
webui maps normalized degrees into this grid per side via sign AND offset (the robot's captured
zero pose is not the MJCF qpos-0 pose) — see fixed_gait/webui/fklut.py.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import plot_reachability as reach                                     # noqa: E402

reach.MODEL = os.path.join(HERE, "dash01.xml")                     # CWD-independent

NODE_NAMES = ("cam", "thigh", "push", "knee", "ank", "ptip", "ee")
OUT_DEFAULT = os.path.join(REPO, "fixed_gait", "webui", "fk_lut.npz")


def generate(nc, nt, ankle_mode, cam_range=None):
    leg = reach.Leg(ankle_mode=ankle_mode, cam_range=cam_range)
    cam = np.linspace(*leg.cam_range, nc)
    thigh = np.linspace(*leg.thigh_range, nt)
    i0 = int(np.argmin(np.abs(cam)))
    j0 = int(np.argmin(np.abs(thigh)))

    nodes = np.full((nc, nt, len(NODE_NAMES), 2), np.nan, np.float32)
    feas = np.zeros((nc, nt), bool)
    valid = np.zeros((nc, nt), bool)
    sol = np.full((nc, nt, 2), np.nan, np.float32)        # solved passive (push, knee) per cell

    def store(i, j, res, sd):
        if res is None:
            return
        pts = reach.linkage_points(leg)
        for k, name in enumerate(NODE_NAMES):
            nodes[i, j, k] = pts[name]
        feas[i, j] = True
        valid[i, j] = not res["collide"] and not res["ank_oob"]
        sol[i, j] = sd

    # chained-seed sweep so Newton tracks the single physical assembly branch (sweep(), :310-333)
    def fill_column(i, center_seed):
        res, seed = leg.fk(cam[i], thigh[j0], center_seed)
        store(i, j0, res, seed)
        base = seed if res is not None else center_seed
        s = base
        for j in range(j0 + 1, nt):
            res, sd = leg.fk(cam[i], thigh[j], s)
            store(i, j, res, sd)
            if res is not None:
                s = sd
        s = base
        for j in range(j0 - 1, -1, -1):
            res, sd = leg.fk(cam[i], thigh[j], s)
            store(i, j, res, sd)
            if res is not None:
                s = sd
        return base

    t0 = time.time()
    seed0 = np.array([0.0, 0.0])
    s = fill_column(i0, seed0)
    for i in range(i0 + 1, nc):
        s = fill_column(i, s)
        print(f"\r  cam column {i - i0 + 1 + (i0 - 0)}/{nc}  ({time.time() - t0:.0f}s)",
              end="", flush=True)
    s = seed0
    for i in range(i0 - 1, -1, -1):
        s = fill_column(i, s)
        print(f"\r  cam column {nc - i}/{nc}  ({time.time() - t0:.0f}s)", end="", flush=True)
    print(f"\n  swept {nc * nt} cells in {time.time() - t0:.0f}s: "
          f"feasible {feas.sum()} ({100 * feas.mean():.1f}%), valid {valid.sum()}")
    return leg, cam, thigh, nodes, feas, valid, sol


def check(leg, cam, thigh, nodes, feas, sol, n=500, seed=1):
    """Bilinear interpolation of the LUT vs direct FK at random interior points.
    The direct FK is seeded from the cell's SOLVED passive angles so Newton stays on the
    physical assembly branch (a cold (0,0) seed lands on the wrong branch far from rest)."""
    rng = np.random.default_rng(seed)
    dc, dt = cam[1] - cam[0], thigh[1] - thigh[0]
    errs = []
    tries = 0
    while len(errs) < n and tries < n * 50:
        tries += 1
        c = rng.uniform(cam[0], cam[-1])
        t = rng.uniform(thigh[0], thigh[-1])
        i0 = min(int((c - cam[0]) / dc), len(cam) - 2)
        j0 = min(int((t - thigh[0]) / dt), len(thigh) - 2)
        if not feas[i0:i0 + 2, j0:j0 + 2].all():
            continue
        ee = nodes[i0:i0 + 2, j0:j0 + 2, 6, :]
        # cells straddling the fold (dead-center) are NOT interpolable — the runtime masks
        # them out the same way (fklut.py smooth mask), so only smooth cells are checked here
        spread = np.max(np.linalg.norm(ee.reshape(4, 2) - ee.reshape(4, 2).mean(0), axis=1))
        if spread > 0.03:
            continue
        res, _ = leg.fk(c, t, sol[i0, j0].astype(float).copy())
        if res is None:
            continue
        fi, fj = (c - cam[i0]) / dc, (t - thigh[j0]) / dt
        interp = ((1 - fi) * (1 - fj) * ee[0, 0] + (1 - fi) * fj * ee[0, 1]
                  + fi * (1 - fj) * ee[1, 0] + fi * fj * ee[1, 1])
        errs.append(np.linalg.norm(interp - res["tip"]))
    errs = np.array(errs)
    print(f"  interp check ({len(errs)} smooth-cell pts): mean {errs.mean() * 1000:.3f} mm, "
          f"max {errs.max() * 1000:.2f} mm")
    return errs.max() < 0.002                       # < 2 mm worst-case on interpolable cells


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc", type=int, default=501)
    ap.add_argument("--nt", type=int, default=201)
    ap.add_argument("--cam-lo", type=float, default=-90.0, metavar="DEG",
                    help="cam grid start, DEGREES (full-crank default -90)")
    ap.add_argument("--cam-hi", type=float, default=270.0, metavar="DEG",
                    help="cam grid end, DEGREES (full-crank default +270)")
    ap.add_argument("--ankle-mode", choices=["spring-rest", "parallel-thigh"],
                    default="spring-rest", help="spring-rest matches the real passive ankle")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--check", action="store_true", help="verify LUT interpolation vs direct FK")
    args = ap.parse_args()

    cam_range = np.radians([args.cam_lo, args.cam_hi])
    print(f"Generating FK LUT {args.nc}x{args.nt}, cam [{args.cam_lo:g},{args.cam_hi:g}] deg, "
          f"ankle={args.ankle_mode} ...")
    leg, cam, thigh, nodes, feas, valid, sol = generate(args.nc, args.nt, args.ankle_mode,
                                                        cam_range)
    np.savez_compressed(args.out, cam=cam, thigh=thigh, nodes=nodes, feas=feas, valid=valid,
                        sol=sol, ankle_mode=args.ankle_mode)
    print(f"  wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    if args.check:
        ok = check(leg, cam, thigh, nodes, feas, sol)
        print("  CHECK:", "PASS" if ok else "FAIL (interp error > 2 mm)")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
