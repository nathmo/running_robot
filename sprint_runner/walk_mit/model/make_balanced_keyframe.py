"""Re-solve the `stand` keyframe so the robot starts BALANCED and with usable down-reach.

Two defects of the shipped keyframe, both measured (see the scripted-controller autopsy):

  1. THE FEET ARE NOT UNDER THE COM. build_model.INIT_CTRL is commented "feet under the CoM"; the
     settled stance actually puts the toe contacts 9.1 cm AHEAD of the CoM. On a point-toe biped
     with no CoP authority that is a 5.7 deg backward lean at t=0 of every episode, and the
     free-topple time from there (0.93 s analytic) is exactly what both the scripted controller
     (0.99 s) and the trained m3 policy (1.15 s) survive. The rung is lost before the first action.

  2. THE LEG IS AT 96% OF REACH, so there is no usable DOWN-reach. The sagittal Jacobian is nearly
     horizontal there: commanding 6.5 cm of "extension" through the measured foot IK moves the toe
     8.5 cm FORWARD and 0.8 cm down. A capture step therefore cannot reach the ground it is aimed
     at -- measured as 0.0 N on the swung foot through an entire fall with every joint on target.

Both are properties of one 6-vector, `key_ctrl`, so both are fixed by re-solving it. This tool
searches (cam, thigh) -- mirrored L/R, hip-roll left at 0 -- for a stance that settles under
gravity with toe_x == com_x and a chosen ride height, using build_model's own settling procedure
so the result is a drop-in replacement for the shipped keyframe. It writes a VARIANT file and
never touches dash01.xml: the two plants have to be A/B-able, and every existing checkpoint's
plant must stay bit-identical.

    python -m model.make_balanced_keyframe --crouch 0.05 --out model/dash01_bal.xml
    python -m model.make_balanced_keyframe --scan          # just report the trade-off, write nothing

env.py picks this up through cfg.model_path; `ankle_resettle` re-settles qpos against the arm's
ankle law as usual and keeps this ctrl, because the deviation-0 candidate is tried first and a
rigid ankle can hold it.
"""
import argparse
import os
import re
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from build_model import compute_standing_keyframe          # noqa: E402

BASE = os.path.join(HERE, "dash01.xml")


def f3(a):
    return " ".join(f"{v:.6g}" for v in np.asarray(a).ravel())


class Stance:
    def __init__(self, path=BASE):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.foot = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_col")
                     for s in "LR"]
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bodyNCS-v1")
        self.act_qadr = np.array([self.model.jnt_qposadr[self.model.actuator_trnid[a, 0]]
                                  for a in range(self.model.nu)])

    @staticmethod
    def ctrl(cam, thigh):
        """Mirrored sagittal stance; hip-roll stays at the design zero."""
        return np.array([0.0, cam, thigh, 0.0, -cam, -thigh])

    def settle(self, cam, thigh):
        """(footprint_x - com_x, base_z, qpos) of the gravity-settled loaded stance.

        `geom_xpos` of the foot collision geom is the CENTRE OF THE FOOTPRINT for every foot shape
        in this project, not just the point toe: a level plate contacts the floor symmetrically
        about its centre, and a lateral-axis cylinder (the blade) contacts along the line directly
        under its axis. So "toe under the CoM" generalises to "footprint centre under the CoM" with
        no change to the residual, which is why the flat and blade plants can reuse this solve.
        It stops being true if the contact face is TILTED — the flatten pass in make_foot_variants
        is what keeps it honest, and it reports the residual tilt."""
        q = compute_standing_keyframe(self.model, self.ctrl(cam, thigh))
        d = self.data
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(self.model, d)
        toe = float(np.mean([d.geom_xpos[g][0] for g in self.foot]))
        com = float(d.subtree_com[0][0])
        return toe - com, float(q[2]), q

    # ---------- down-reach diagnostics ----------
    def jacobian(self, qpos, h=0.02):
        """Sagittal toe Jacobian d(toe_x, toe_z)/d(cam, thigh) at a pose, base FIXED (pure
        kinematics of the closed loop -- the joints are stepped to their targets with gravity off
        so the 4-bar stays consistent)."""
        m, d = self.model, self.data
        g = m.opt.gravity.copy()
        m.opt.gravity[:] = 0
        # baseline ctrl captured ONCE: reading it back out of d.ctrl inside the probe makes each
        # perturbation compound onto the previous one and reports a rank-deficient Jacobian
        ctrl0 = np.asarray(qpos[self.act_qadr], float).copy()

        def toe_at(dcam, dth):
            d.qpos[:] = qpos
            d.qvel[:] = 0
            c = ctrl0.copy()
            c[1] += dcam; c[2] += dth
            c[4] -= dcam; c[5] -= dth
            d.ctrl[:] = c
            base = d.qpos[:6].copy()
            for _ in range(600):
                mujoco.mj_step(m, d)
                d.qpos[:6] = base
                d.qvel[:6] = 0
            mujoco.mj_forward(m, d)
            p = d.geom_xpos[self.foot[0]]
            b = d.xpos[self.base_id]
            R = d.xmat[self.base_id].reshape(3, 3)
            t = R.T @ (p - b)
            return np.array([t[0], t[2]])

        cols = []
        for i in range(2):
            dp = [0.0, 0.0]; dp[i] = +h
            dm = [0.0, 0.0]; dm[i] = -h
            cols.append((toe_at(*dp) - toe_at(*dm)) / (2 * h))
        m.opt.gravity[:] = g
        return np.column_stack(cols)          # 2x2, rows (x,z), cols (cam,thigh)

    def downreach(self, qpos, lateral=0.05, span=0.55, n=19):
        """USABLE down-reach: how far below the stance toe can this leg put its foot without
        dragging it more than `lateral` metres fore-aft?

        Deliberately NOT a Jacobian. The limit here is a reach BOUNDARY, not a rank defect: at 96%
        extension the leg shortens freely (commanding the toe 5 cm up delivers 5.1 cm, measured)
        and lengthens not at all, so a symmetric finite difference straddles the boundary and
        reports garbage -- an apparently singular Jacobian and a 1372 m/m "drag" that no real
        motion exhibits. A direct sweep of the reachable set answers the question a capture step
        actually asks: is there ground I can reach down to?

        Returns (usable down-reach m, usable up-reach m, fore-aft span available at stance level).
        """
        m, d = self.model, self.data
        g = m.opt.gravity.copy()
        m.opt.gravity[:] = 0
        ctrl0 = np.asarray(qpos[self.act_qadr], float).copy()
        grid = np.linspace(-span, span, n)
        pts = []
        for dc in grid:
            for dt in grid:
                d.qpos[:] = qpos
                d.qvel[:] = 0
                c = ctrl0.copy()
                c[1] += dc; c[2] += dt
                c[4] -= dc; c[5] -= dt
                d.ctrl[:] = c
                base = d.qpos[:6].copy()
                for _ in range(300):
                    mujoco.mj_step(m, d)
                    d.qpos[:6] = base
                    d.qvel[:6] = 0
                mujoco.mj_forward(m, d)
                b = d.xpos[self.base_id]
                R = d.xmat[self.base_id].reshape(3, 3)
                t = R.T @ (d.geom_xpos[self.foot[0]] - b)
                pts.append((t[0], t[2]))
        m.opt.gravity[:] = g
        pts = np.array(pts)
        d.qpos[:] = qpos
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        b = d.xpos[self.base_id]
        R = d.xmat[self.base_id].reshape(3, 3)
        t0 = R.T @ (d.geom_xpos[self.foot[0]] - b)
        near = np.abs(pts[:, 0] - t0[0]) <= lateral
        if not near.any():
            return 0.0, 0.0, 0.0
        down = float(t0[2] - pts[near, 1].min())
        up = float(pts[near, 1].max() - t0[2])
        # fore-aft span reachable while staying within 1 cm of the stance toe HEIGHT
        lvl = np.abs(pts[:, 1] - t0[2]) <= 0.01
        fa = float(pts[lvl, 0].max() - pts[lvl, 0].min()) if lvl.any() else 0.0
        return down, up, fa


def solve(st, z_target, cam0=0.0, th0=0.12, iters=12, tol=2e-3, verbose=True):
    """Newton on (cam, thigh) -> (toe_x - com_x, base_z) toward (0, z_target)."""
    x = np.array([cam0, th0])
    for it in range(iters):
        off, z, q = st.settle(*x)
        r = np.array([off, z - z_target])
        if verbose:
            print(f"    it{it}: cam {x[0]:+.4f} thigh {x[1]:+.4f} -> toe-CoM {off:+.4f} "
                  f"z {z:.4f}  |r| {np.linalg.norm(r):.4f}")
        if np.linalg.norm(r) < tol:
            return x, off, z, q
        h = 0.02
        J = np.zeros((2, 2))
        for i in range(2):
            xp = x.copy(); xp[i] += h
            o2, z2, _ = st.settle(*xp)
            J[:, i] = [(o2 - off) / h, (z2 - z) / h]
        try:
            step = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -0.08, 0.08)          # the settle is only locally smooth
        x = x + step
        x[0] = float(np.clip(x[0], -1.2, 1.2))   # joint ranges are cam +-1.5, thigh +-1.047
        x[1] = float(np.clip(x[1], -0.9, 0.9))
    off, z, q = st.settle(*x)
    return x, off, z, q


def write_variant(src, out, qpos, ctrl):
    txt = open(src, encoding="utf-8").read()
    new = f'<key name="stand" qpos="{f3(qpos)}" ctrl="{f3(ctrl)}" />'
    txt2, n = re.subn(r'<key name="stand"[^/]*/>', new, txt)
    if n != 1:
        raise RuntimeError(f"expected exactly one stand key in {src}, replaced {n}")
    open(out, "w", encoding="utf-8").write(txt2)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crouch", type=float, default=0.05,
                    help="m of ride height to give up vs the shipped stance (0 = balance only)")
    ap.add_argument("--src", default=BASE,
                    help="plant to re-balance (default dash01.xml; the foot variants pass their own)")
    ap.add_argument("--out", default=os.path.join(HERE, "dash01_bal.xml"))
    ap.add_argument("--scan", action="store_true", help="report the crouch trade-off, write nothing")
    args = ap.parse_args()

    st = Stance(args.src)
    # start from the SOURCE's own stance, not the hard-coded (0, 0.12): a foot variant that has
    # already been re-settled carries its own key_ctrl, and seeding the Newton off it is both the
    # honest baseline to report and a warm start.
    cam_src, th_src = float(st.model.key_ctrl[0][1]), float(st.model.key_ctrl[0][2])
    off0, z0, q0 = st.settle(cam_src, th_src)
    dn0, up0, fa0 = st.downreach(q0)
    print(f"SOURCE stance ({args.src}): cam {cam_src:+.4f} thigh {th_src:+.4f}  "
          f"toe-CoM {off0:+.4f} m   base_z {z0:.4f} m")
    print(f"   lean {np.degrees(np.arctan2(off0, 0.916)):+.2f} deg   "
          f"DOWN-reach {dn0*100:.1f} cm   up-reach {up0*100:.1f} cm   fore-aft span {fa0*100:.1f} cm")
    print()

    targets = ([z0 - c for c in (0.0, 0.03, 0.05, 0.08, 0.10, 0.12)] if args.scan
               else [z0 - c for c in np.arange(0.0, args.crouch + 1e-9, 0.025)])
    best = None
    cam0, th0 = cam_src, th_src
    for zt in targets:
        print(f"  solving for toe-CoM = 0, base_z = {zt:.4f} (crouch {z0 - zt:.3f} m)")
        # CONTINUATION: warm-start each target from the previous solution. The settle is only
        # locally smooth (the 4-bar has a fold branch), and a cold Newton at deep crouch walks
        # straight into it -- measured as a solve that "converged" 8 cm off balance.
        x, off, z, q = solve(st, zt, cam0=cam0, th0=th0, verbose=not args.scan)
        cam0, th0 = float(x[0]), float(x[1])
        dn, up, fa = st.downreach(q)
        print(f"    -> cam {x[0]:+.4f} thigh {x[1]:+.4f}  toe-CoM {off:+.4f}  z {z:.4f}  "
              f"lean {np.degrees(np.arctan2(off, 0.916)):+.2f} deg  "
              f"DOWN {dn*100:.1f} cm  up {up*100:.1f} cm  fore-aft {fa*100:.1f} cm")
        best = (x, off, z, q)          # the LAST target is the requested crouch
    if args.scan:
        return
    x, off, z, q = best
    if abs(off) > 0.01:
        print(f"WARNING: residual toe-CoM offset {off:+.4f} m — the solve did not converge")
    write_variant(args.src, args.out, q, Stance.ctrl(*x))


if __name__ == "__main__":
    main()
