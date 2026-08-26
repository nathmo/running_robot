"""Crouched `stand` keyframes solved THROUGH a real DashEnv — the only way that holds up.

WHY NOT make_balanced_keyframe --crouch. That tool settles against the raw model, where the ankle is
the shipped passive spring. Every walk_fwd rung runs ankle_mode="rigid", and "rigid" is not one
change but two: the lock_ankle_L/R equalities go ACTIVE *and* the ankle spring stiffness is zeroed
(the env's own note: a spring under a welded joint "is not physics, it is just ~14 N*m of stance
preload for the lock constraint to fight"). A crouch solved against the spring plant is therefore
not an equilibrium of the plant it is loaded into: env._resettle_keyframe re-settles it on reset and
the pose collapses. Measured on --crouch 0.05: the keyframe says base_z 0.9616, DashEnv resets to
0.8347 with the leg folded and the toe 431 mm ahead of the CoM, and all 8 seeds died at ~0.2 s --
an A/B table full of zeros that looked like a result.

Replicating that surgery in a standalone script was tried and got it wrong twice (first the weld,
then the spring). So this does not replicate it at all: it builds the env, then searches with the
env's OWN `_settle` -- the exact function whose fixed point `_resettle_keyframe` will take on reset.
Whatever else the plant does to itself at construction is included for free, and the result is
verified in a fresh DashEnv before anyone trusts it.

    python -m model.make_crouch_rigid --crouch 0.025 0.05 --verify
"""
import argparse
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from make_balanced_keyframe import write_variant                       # noqa: E402
from config import get_config                                          # noqa: E402
from env import DashEnv                                                # noqa: E402

BASE = os.path.join(HERE, "dash01.xml")


def build(milestone, model_rel="model/dash01.xml"):
    cfg = get_config("walk_fwd_" + milestone)
    cfg.model_path = model_rel
    cfg.push_interval_s = 0.0
    cfg.trip_prob = 0.0
    env = DashEnv(cfg)
    env.set_dr_scale(0.0)
    return env


class Stance:
    """Candidate stances evaluated with the ENV's settle, not a re-implementation of it."""

    def __init__(self, env):
        self.env = env
        self.foot = list(env.foot_gids)

    def settle(self, cam, thigh, t_s=2.0):
        env = self.env
        ctrl = np.array([0., cam, thigh, 0., -cam, -thigh])
        qpos, viol = env._settle(ctrl, t_s=t_s)
        if qpos is None:
            return np.nan, np.nan, None, True
        d = env.data
        d.qpos[:] = qpos
        d.qvel[:] = 0
        mujoco.mj_forward(env.model, d)
        toe = float(np.mean([d.geom_xpos[g][0] for g in self.foot]))
        return toe - float(d.subtree_com[0][0]), float(qpos[2]), qpos.copy(), bool(viol)

    def balance_at(self, cam, lo=-0.6, hi=1.0, iters=30, tol=3e-4):
        """Thigh that puts the toe under the CoM at this cam. BISECTION, not Newton: the settle is
        only locally smooth and Newton walks into the four-bar's fold branch and calls it solved."""
        a, _, _, _ = self.settle(cam, lo)
        b, _, _, _ = self.settle(cam, hi)
        if not (np.isfinite(a) and np.isfinite(b)) or (a > 0) == (b > 0):
            return None
        off = mid = None
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            off, z, q, viol = self.settle(cam, mid)
            if not np.isfinite(off):
                return None
            if (off > 0) == (a > 0):
                lo = mid
            else:
                hi = mid
            if abs(off) < tol:
                break
        if off is None or abs(off) > 2e-3 or viol:
            return None
        return float(mid), float(off), float(z), q


def solve(st, z_target, cams):
    best = None
    for cam in cams:
        r = st.balance_at(float(cam))
        if r is None:
            continue
        thigh, off, z, q = r
        if best is None or abs(z - z_target) < abs(best[2] - z_target):
            best = (float(cam), thigh, z, off, q)
    return best


def verify(path, milestone):
    env = build(milestone, os.path.join("model", os.path.basename(path)))
    env.reset(seed=0)
    toe = float(np.mean([env.data.geom_xpos[g][0] for g in env.foot_gids]))
    return dict(base_z=float(env.data.qpos[2]),
                toe_com=toe - float(env.data.subtree_com[0][0]),
                delta=getattr(env, "stance_search_delta", None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crouch", type=float, nargs="+", default=[0.025, 0.05])
    ap.add_argument("--milestone", default="m3")
    ap.add_argument("--cams", type=float, nargs=3, default=[-0.80, 0.10, 37],
                    metavar=("LO", "HI", "N"))
    ap.add_argument("--tag", default="r")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    cams = np.linspace(args.cams[0], args.cams[1], int(args.cams[2]))
    st = Stance(build(args.milestone))
    print("solving through DashEnv._settle on the %s plant (ankle_mode=%s)"
          % (args.milestone, st.env.ankle_mode), flush=True)

    base = solve(st, float(st.env.height_target), cams)
    print("balanced at the shipped height: cam %+.4f thigh %+.4f -> toe-CoM %+.2f mm, base_z %.4f"
          % (base[0], base[1], base[3] * 1000, base[2]), flush=True)

    for c in args.crouch:
        zt = base[2] - c
        r = solve(st, zt, cams)
        if r is None:
            print("  crouch %.3f m: NO balanced stance found" % c, flush=True)
            continue
        cam, thigh, z, off, q = r
        out = os.path.join(HERE, "dash01_%s%02d.xml" % (args.tag, int(round(c * 1000))))
        print("  crouch %.3f m -> cam %+.4f thigh %+.4f | toe-CoM %+.2f mm | base_z %.4f "
              "(asked %.4f)" % (c, cam, thigh, off * 1000, z, zt), flush=True)
        write_variant(BASE, out, q, np.array([0., cam, thigh, 0., -cam, -thigh]))
        if args.verify:
            v = verify(out, args.milestone)
            ok = abs(v["base_z"] - z) < 0.01 and abs(v["toe_com"]) < 0.02
            print("     DashEnv reset: base_z %.4f  toe-CoM %+.1f mm  search_delta %s   %s"
                  % (v["base_z"], v["toe_com"] * 1000, v["delta"],
                     "OK" if ok else "*** THE PLANT MOVED IT — do not use ***"), flush=True)


if __name__ == "__main__":
    main()
