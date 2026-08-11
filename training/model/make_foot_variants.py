"""Generate the FOOT-SHAPE plant variants: a flat plate and a blade (rolling line contact).

THE QUESTION. Every controller this project has produced walks on a 25 mm point toe. A point toe
has exactly zero centre-of-pressure authority: it cannot carry an ankle moment about the contact,
so the only way to arrest a lean is to move the foot, and moving the foot needs a step the 4-bar
cannot reach in time (see [[m3-ankle-stiffness-foot-ahead]] and the scripted-controller autopsy).
That is the shape of the m3 wall. A foot with a FOOTPRINT changes the premise, so it is worth
measuring rather than arguing about.

    flat   30 x 100 x 10 mm plate  (30 mm fore-aft X, 100 mm lateral Y, 10 mm thick)
           Lateral CoP authority +-50 mm, sagittal +-15 mm.
    blade  cylinder, radius 25 mm, 100 mm long, AXIS ALONG WORLD Y
           The sagittal profile is bit-identical to the shipped 25 mm ball -- it still rolls
           fore-aft carrying no pitch moment -- but the point contact becomes a 100 mm LINE across
           the robot. That is the running-blade geometry: curved in the sagittal plane, wide
           across it.
    sphere the shipped point toe, re-emitted through this same pipeline so the balanced-stance
           A/B has a control built by the identical procedure (dash01_bal.xml was built by the
           older file-based solve and stays untouched as the historical reference).

So the set is a decomposition, not three goes at one idea: sphere->blade isolates ROLL authority,
blade->flat isolates the PITCH increment on top.

WHY THE ENV IS THE ORACLE, not compute_standing_keyframe. Every preset that will ever load these
plants runs ankle_mode="rigid" + ankle_resettle, so DashEnv WELDS the ankle and re-settles the
stance at construction -- and that stance is not the spring-ankle one the file keyframe records.
Measured on the blade: levelled against the file keyframe it reads 0.000 deg of tilt, but in the
env the left cylinder sits 8 deg rolled and the right foot hangs 4.8 mm off the floor, i.e. the
robot stands on one edge of one blade. Invisible on a sphere (which is why nobody caught it), fatal
for a foot whose whole point is its footprint. So every settle below is a real DashEnv.

WHAT THIS HAS TO SOLVE. The foot body is pitched ~51 deg nose-down in stance and its pitch is a
function of the leg pose, so a plate bolted on at a fixed local angle is level at ONE stance -- and
the stance is exactly what the balance solve is allowed to move. Contact-face orientation and
stance are coupled, so this is a fixed point:

    repeat:   level the geoms at the current stance  ->  Newton the stance back onto its target

    python -m model.make_foot_variants                     # all three plants, both stances
    python -m model.make_foot_variants --kinds blade
    python -m model.make_foot_variants --no-balance        # skip the *_bal stance solve

Writes, from model/dash01.xml:
    dash01_flat.xml   dash01_flat_bal.xml
    dash01_blade.xml  dash01_blade_bal.xml
                      dash01_sphere_bal.xml
The *_bal plants are the ones the classical controller wants: same foot, stance re-solved so the
FOOTPRINT CENTRE sits under the CoM instead of ~85 mm ahead of it. All three land on the SAME
absolute ride height, so a foot A/B is not secretly a ride-height A/B. dash01.xml is never touched
-- every existing checkpoint's plant has to stay bit-identical.
"""
import argparse
import os
import re
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                      # training/
for p in (HERE, PKG):
    if p not in sys.path:
        sys.path.insert(0, p)
from build_model import geom_bottom_z             # noqa: E402
from make_balanced_keyframe import f3             # noqa: E402

BASE = os.path.join(HERE, "dash01.xml")

# The shipped point toe: a sphere of this radius centred at the toe tip vertex.
SPHERE_R = 0.025
# The milestone whose preset supplies the ankle law / resettle used as the settle oracle. Any rung
# works -- they differ only in base_lock, which the settle holds anyway.
ORACLE_PRESET = "walk_fwd_m3"

# --- the feet ------------------------------------------------------------------------------------
# `half` is MuJoCo's size vector. `drop` is how far the geom CENTRE moves down in world z relative
# to the sphere centre it replaces, chosen so the new contact surface sits where the old ball's
# bottom was -- leg length, ride height and stance all start from the same place and the only thing
# that changed is the shape of the contact.
#   flat:  the centre must sit half a plate thickness above the floor -> -25 + 5 = -20 mm
#   blade: a 25 mm cylinder's bottom is 25 mm below its axis, same as the ball -> 0
FEET = {
    "sphere": dict(type="sphere", half=(SPHERE_R,), drop=0.0, level=False,
                   world_rot=np.eye(3)),
    "flat": dict(type="box",
                 half=(0.015, 0.050, 0.005),      # 30 mm fore-aft, 100 mm lateral, 10 mm thick
                 drop=-(SPHERE_R - 0.005), level=True,
                 # contact face level: the geom's axes ARE the world axes, so the 100 mm runs
                 # across the robot and the 30 mm fore-aft.
                 world_rot=np.eye(3)),
    "blade": dict(type="cylinder",
                  half=(0.025, 0.050),            # radius 25 mm, half-length 50 mm => 100 mm long
                  drop=0.0, level=True,
                  # MuJoCo's cylinder axis is its LOCAL z; rotate -90 deg about world x so the axis
                  # lies along world y (across the robot) and the round profile faces fore-aft.
                  world_rot=np.array([[1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0],
                                      [0.0, -1.0, 0.0]])),
}

FOOT_GEOM_RE = {s: re.compile(rf'<geom\s+name="foot_{s}_col"[^>]*?/>') for s in "LR"}
# joint ranges the stance search may use: cam +-1.5, thigh +-1.047 (see build_model.J)
CAM_LIM, THIGH_LIM = 1.2, 0.9


def _read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def _write(p, txt):
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(txt)


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def stance_ctrl(cam, thigh):
    """Mirrored sagittal stance; hip-roll stays at the design zero (as make_balanced_keyframe)."""
    return np.array([0.0, cam, thigh, 0.0, -cam, -thigh])


def write_plant(src_txt, out, spec, pose, qpos, ctrl):
    """Splice the two foot geoms and the stand keyframe into the source XML."""
    txt = src_txt
    for s in "LR":
        size = " ".join(f"{v:.6g}" for v in spec["half"])
        new = (f'<geom name="foot_{s}_col" class="collision" type="{spec["type"]}" '
               f'size="{size}" pos="{f3(pose[s]["pos"])}" quat="{f3(pose[s]["quat"])}" />')
        txt, n = FOOT_GEOM_RE[s].subn(new, txt, count=1)
        if n != 1:
            raise SystemExit(f"foot_{s}_col geom not found in the source plant")
    key = f'<key name="stand" qpos="{f3(qpos)}" ctrl="{f3(ctrl)}" />'
    txt, n = re.subn(r'<key name="stand"[^/]*/>', key, txt, count=1)
    if n != 1:
        raise SystemExit(f"expected exactly one stand key, replaced {n}")
    _write(out, txt)
    return txt


def env_settle(plant):
    """Settle the plant the way the EXPERIMENT will: a real DashEnv on the ladder preset, so the
    ankle is welded and `_resettle_keyframe` has run. Returns the measurements the fixed point and
    the report need."""
    from config import get_config                 # imported late: pulls in torch-free deps only
    from env import DashEnv
    cfg = get_config(ORACLE_PRESET)
    cfg.model_path = plant
    e = DashEnv(cfg)
    m, d = e.model, e.data
    d.qpos[:] = e.default_qpos
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    out = dict(qpos=e.default_qpos.copy(), base_z=float(e.default_qpos[2]),
               com_x=float(d.subtree_com[0][0]),
               foot_x=float(np.mean(d.geom_xpos[e.foot_gids_arr, 0])),
               sole=e._sole_offsets().copy(), toe_h=e._toe_heights().copy(),
               stance_delta=tuple(getattr(e, "stance_search_delta", (0.0, 0.0))),
               ctrl=e.nominal_ctrl.copy())
    for i, s in enumerate("LR"):
        g = e.foot_gids[i]
        out[s] = dict(body_R=d.xmat[int(m.geom_bodyid[g])].reshape(3, 3).copy(),
                      geom_R=d.geom_xmat[g].reshape(3, 3).copy(),
                      bottom=geom_bottom_z(m, d, g))
    out["off"] = out["foot_x"] - out["com_x"]
    return out


def _tilt_deg(R_actual, R_target):
    c = (np.trace(R_actual @ R_target.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _level(spec, base_geom_pos, st):
    """Geom placement that makes the contact face meet the target world orientation at stance `st`,
    with the sole hung off the toe tip exactly where the shipped ball's bottom was."""
    pose = {}
    for s in "LR":
        Rb = st[s]["body_R"]
        R_local = Rb.T @ spec["world_rot"] if spec["level"] else np.eye(3)
        pos = base_geom_pos[s] + Rb.T @ np.array([0.0, 0.0, spec["drop"]])
        pose[s] = dict(pos=pos, quat=_quat(R_local))
    return pose


def _newton_stance(write_and_settle, x0, qpos0, z_target, iters=8, tol=2e-3, verbose=True):
    """Newton on (cam, thigh) -> (footprint_x - com_x, base_z - z_target), with the ENV as f.

    `write_and_settle(x, qpos) -> (st, qpos)`. The step is clipped: the settle is only locally
    smooth (the 4-bar has a fold branch), and an unclipped Newton walks straight into it.
    """
    def probe(xx, q):
        """A candidate stance can simply not stand -- DashEnv raises when its own stance search
        cannot find a posture that settles without the foot going through the floor. That is a
        legitimate answer from f(), not a crash, so a probe that hits it returns None and the
        caller backs off instead of taking the whole build down."""
        try:
            return write_and_settle(xx, q)
        except RuntimeError as exc:
            if verbose:
                print(f"      probe cam {xx[0]:+.4f} thigh {xx[1]:+.4f} does not stand ({exc})")
            return None

    x, qpos = np.asarray(x0, float).copy(), qpos0.copy()
    st = None
    for it in range(iters):
        got = probe(x, qpos)
        if got is None:
            break                       # keep the last posture that did stand
        st, qpos = got
        r = np.array([st["off"], st["base_z"] - z_target])
        if verbose:
            print(f"      it{it}: cam {x[0]:+.4f} thigh {x[1]:+.4f} -> "
                  f"footprint-CoM {r[0]:+.4f}  z {st['base_z']:.4f}  |r| {np.linalg.norm(r):.4f}")
        if np.linalg.norm(r) < tol:
            break
        J = np.zeros((2, 2))
        ok = True
        for i in range(2):
            col = None
            for h in (0.02, -0.02, 0.01, -0.01):     # try the other side / a shorter step
                xp = x.copy()
                xp[i] += h
                s2 = probe(xp, qpos)
                if s2 is not None:
                    col = [(s2[0]["off"] - r[0]) / h, (s2[0]["base_z"] - st["base_z"]) / h]
                    break
            if col is None:
                ok = False
                break
            J[:, i] = col
        if not ok:
            break
        try:
            step = np.clip(np.linalg.solve(J, -r), -0.08, 0.08)
        except np.linalg.LinAlgError:
            break
        x = x + step
        x[0] = float(np.clip(x[0], -CAM_LIM, CAM_LIM))
        x[1] = float(np.clip(x[1], -THIGH_LIM, THIGH_LIM))
    return x, st, qpos


def build(kind, out, balance=False, z_target=None, passes=4, tol_deg=0.25, tol_m=3e-3,
          verbose=True):
    """Fixed point over (contact-face orientation, stance). Each pass writes a complete, compilable
    plant, so an aborted run still leaves a valid file."""
    spec = FEET[kind]
    src_txt = _read(BASE)
    base_m = mujoco.MjModel.from_xml_path(BASE)
    base_geom_pos = {s: base_m.geom_pos[
        mujoco.mj_name2id(base_m, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_col")].copy() for s in "LR"}

    qpos = base_m.key_qpos[0].copy()
    ctrl = base_m.key_ctrl[0].copy()
    x = np.array([float(ctrl[1]), float(ctrl[2])])
    # pass 0 needs SOME plant to settle: the un-levelled foot on the shipped stance.
    pose = {s: dict(pos=base_geom_pos[s], quat=np.array([1.0, 0, 0, 0])) for s in "LR"}
    write_plant(src_txt, out, spec, pose, qpos, ctrl)
    st = env_settle(out)

    for p in range(1, passes + 1):
        pose = _level(spec, base_geom_pos, st)

        def write_and_settle(xx, q, _pose=pose):
            write_plant(src_txt, out, spec, _pose, q, stance_ctrl(*xx))
            s = env_settle(out)
            return s, s["qpos"]

        if balance:
            x, st, qpos = _newton_stance(write_and_settle, x, qpos, z_target, verbose=verbose)
        else:
            st, qpos = write_and_settle(x, qpos)
        ctrl = stance_ctrl(*x)
        write_plant(src_txt, out, spec, pose, qpos, ctrl)

        # a sphere has no contact-face orientation to get wrong; comparing its geom frame (which
        # just rides the foot body's 51 deg pitch) against the world would report ~57 deg forever
        # and the fixed point would never terminate.
        tilt = (max(_tilt_deg(st[s]["geom_R"], spec["world_rot"]) for s in "LR")
                if spec["level"] else 0.0)
        if verbose:
            soles = " ".join("%+.4f" % st[s]["bottom"] for s in "LR")
            print(f"  pass {p}: tilt {tilt:.3f} deg   footprint-CoM {st['off']:+.4f} m   "
                  f"base_z {st['base_z']:.4f} m   sole z {soles} m   "
                  f"toe air {np.round(st['toe_h'] * 1000, 1)} mm")
        if st["stance_delta"] != (0.0, 0.0):
            print(f"  WARNING: the env's stance search moved off the solved posture by "
                  f"{st['stance_delta']} — the plant does not stand where this tool put it")
        if tilt < tol_deg and (not balance or abs(st["off"]) < tol_m):
            break

    dims = " x ".join(f"{2000 * v:g}" for v in spec["half"])
    print(f"  wrote {os.path.basename(out)}: {spec['type']} {dims} mm | tilt {tilt:.2f} deg | "
          f"footprint-CoM {st['off']:+.4f} m | base_z {st['base_z']:.4f} m | "
          f"cam {x[0]:+.4f} thigh {x[1]:+.4f}")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="*", default=["sphere", "blade", "flat"], choices=list(FEET))
    ap.add_argument("--crouch", type=float, default=0.05,
                    help="ride height given up by the *_bal stance solve (matches dash01_bal.xml)")
    ap.add_argument("--no-balance", action="store_true", help="skip the *_bal plants")
    ap.add_argument("--passes", type=int, default=4)
    args = ap.parse_args()

    # ONE absolute ride-height target for every foot, taken from the shipped plant's own settled
    # stance. Solving each foot to "its own z0 minus 5 cm" instead would hand the flat plate a
    # 13 mm taller stance than the sphere and quietly turn the foot A/B into a ride-height A/B.
    z0 = env_settle(BASE)["base_z"]
    z_target = z0 - args.crouch
    print(f"shipped plant settles at base_z {z0:.4f} m in the env "
          f"({ORACLE_PRESET}, rigid ankle, resettled)")
    print(f"balanced-stance target: footprint-CoM = 0, base_z = {z_target:.4f} m "
          f"(crouch {args.crouch:.3f} m)\n")

    for kind in args.kinds:
        if kind != "sphere":            # the sphere control only exists in its balanced form
            print(f"=== {kind}: shipped stance ===")
            build(kind, os.path.join(HERE, f"dash01_{kind}.xml"), balance=False,
                  passes=args.passes)
        if not args.no_balance:
            print(f"=== {kind}: balanced stance (footprint centre under the CoM) ===")
            build(kind, os.path.join(HERE, f"dash01_{kind}_bal.xml"), balance=True,
                  z_target=z_target, passes=args.passes)
        print()


if __name__ == "__main__":
    main()
