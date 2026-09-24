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
FOOTPRINT CENTRE sits under the CoM instead of ~82 mm ahead of it. dash01.xml is never touched --
every existing checkpoint's plant has to stay bit-identical.

RIDE HEIGHT IS AN OUTPUT, NOT A TARGET, and that is a finding rather than a shortcut: on this leg
the balanced footprint and a chosen crouch are not independently achievable (see _newton_stance).
The heights the three balanced arms land on are printed at the end so the residual spread stays
visible as a confound.
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
# `tilt` names WHICH misorientations actually matter, and it is not cosmetic -- it is the fixed
# point's error signal, so a metric that measures something physically irrelevant makes the loop
# chase a phantom. Measured: scoring the blade on its FULL rotation reported 7.6 / 4.3 / 8.2 deg
# across three passes and never converged, while both blades sat flat on the floor the whole time
# -- all of it was SPIN ABOUT THE CYLINDER'S OWN AXIS, which changes nothing.
#   "none" sphere: no orientation to get wrong.
#   "axis" cylinder: only the direction of the axis (local z) counts; spin about it is free.
#   "full" box:     every axis counts -- roll and pitch set whether the sole is flat, yaw sets
#                   which way the 100 mm points.
    "sphere": dict(type="sphere", half=(SPHERE_R,), drop=0.0, tilt="none",
                   world_rot=np.eye(3)),
    "flat": dict(type="box",
                 half=(0.015, 0.050, 0.005),      # 30 mm fore-aft, 100 mm lateral, 10 mm thick
                 drop=-(SPHERE_R - 0.005), tilt="full",
                 # contact face level: the geom's axes ARE the world axes, so the 100 mm runs
                 # across the robot and the 30 mm fore-aft.
                 world_rot=np.eye(3)),
    "blade": dict(type="cylinder",
                  half=(0.025, 0.050),            # radius 25 mm, half-length 50 mm => 100 mm long
                  drop=0.0, tilt="axis",
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


def tilt_deg(spec, st):
    """How far the contact geometry is from where it was aimed, in degrees — see FEET["tilt"]."""
    mode = spec["tilt"]
    if mode == "none":
        return 0.0
    out = 0.0
    for s in "LR":
        R, T = st[s]["geom_R"], spec["world_rot"]
        if mode == "axis":                       # cylinder: only the axis direction counts
            c = float(np.dot(R[:, 2], T[:, 2]))
        else:                                    # box: the whole frame counts
            c = (np.trace(R @ T.T) - 1.0) / 2.0
        out = max(out, float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
    return out


def _level(spec, base_geom_pos, st):
    """Geom placement that makes the contact face meet the target world orientation at stance `st`,
    with the sole hung off the toe tip exactly where the shipped ball's bottom was."""
    pose = {}
    for s in "LR":
        Rb = st[s]["body_R"]
        R_local = np.eye(3) if spec["tilt"] == "none" else Rb.T @ spec["world_rot"]
        pos = base_geom_pos[s] + Rb.T @ np.array([0.0, 0.0, spec["drop"]])
        pose[s] = dict(pos=pos, quat=_quat(R_local))
    return pose


def _newton_stance(write_and_settle, x0, qpos0, iters=16, tol=2e-3, verbose=True):
    """Gauss-Newton on (cam, thigh) -> footprint_x - com_x, with the ENV as f.

    ONE objective, deliberately. The obvious formulation also pins base_z, and adding it made this
    solve strictly worse: LOCALLY the two are near-conflicting (every Jacobian taken near the
    shipped stance pointed along (cam, thigh) ~ (-1,+1) with d(footprint)/d(height) ~ -2.9 m/m, so
    14 mm of extra crouch cost 29 mm of balance and the line search correctly refused every such
    step), and the residual then plateaued with the height term dominating -- which let it trade
    away the fore-aft balance, the one thing that actually matters, to chase millimetres of height
    it was never going to get.

    Do NOT read that as "this leg cannot crouch": that was measured only near one stance, and the
    stronger claim ("no down-reach at 96% of extension") turned out to be an artifact of a sweep the
    floor was blocking -- the leg really has ~10 cm ([[scripted-walk-controller]]). The balanced
    stances this tool now produces sit at essentially the shipped ride height anyway. Height is an
    OUTPUT, reported with its across-feet spread so it stays visible as a confound.

    `write_and_settle(x, qpos) -> (st, qpos)`. Steps are clipped and line-searched: the settle is
    only locally smooth (the 4-bar has a fold branch), and an unclipped step walks straight in.
    """
    def probe(xx, q):
        """A candidate stance can simply not stand -- DashEnv raises when its own stance search
        cannot find a posture that settles without the foot going through the floor. That is a
        legitimate answer from f(), not a crash, so a probe that hits it returns None and the
        caller backs off instead of taking the whole build down."""
        try:
            return write_and_settle(xx, np.asarray(q, float).copy())
        except RuntimeError:
            if verbose:
                print(f"      probe cam {xx[0]:+.4f} thigh {xx[1]:+.4f} does not stand")
            return None

    def resid(s):
        return abs(float(s["off"]))

    # SAFEGUARDED, and it has to be. Two things bite here and both were measured:
    #   * the settle is PATH DEPENDENT -- DashEnv re-settles from the keyframe qpos it is handed,
    #     and handed a collapsed one it happily lands in the 4-bar's fold branch (base_z -0.048,
    #     feet 217 mm in the air). So every probe warm-starts from the BEST pose found so far,
    #     never from whatever the previous probe produced.
    #   * an undamped step overshoots into that same branch. So a step is only accepted if it
    #     actually reduces the residual, and is halved up to 4 times before being given up on.
    # Returning the best EVALUATED iterate matters just as much: returning the post-step x (which
    # was never evaluated) alongside the pre-step measurements is what silently mismatched the
    # written keyframe from its own settle.
    best_x = np.asarray(x0, float).copy()
    got = probe(best_x, qpos0)
    if got is None:
        return best_x, None, qpos0.copy()
    best_st, best_q = got
    best_r = resid(best_st)
    if verbose:
        print(f"      it0: cam {best_x[0]:+.4f} thigh {best_x[1]:+.4f} -> "
              f"footprint-CoM {best_st['off']:+.4f}  z {best_st['base_z']:.4f}")

    for it in range(1, iters + 1):
        if best_r < tol:
            break
        r = float(best_st["off"])
        g = np.zeros(2)
        ok = True
        for i in range(2):
            col = None
            # h=0.03, not 0.02: the settle is only good to ~1 mm, so a shorter step buries the
            # gradient in settle noise.
            for h in (0.03, -0.03, 0.015, -0.015):   # try the other side / a shorter step
                xp = best_x.copy()
                xp[i] += h
                s2 = probe(xp, best_q)
                if s2 is not None:
                    col = (s2[0]["off"] - r) / h
                    break
            if col is None:
                ok = False
                break
            g[i] = col
        if not ok or float(g @ g) < 1e-9:
            break
        # MINIMUM-NORM Gauss-Newton: one residual, two knobs, so the step is the shortest one that
        # zeroes the linearised residual. Moving both joints as little as possible is also what
        # keeps the stance close to the shipped posture, which is the point -- this is meant to be
        # the same robot standing differently, not a different crouch.
        full = np.clip(-r * g / float(g @ g), -0.12, 0.12)
        improved = False
        for shrink in (1.0, 0.5, 0.25, 0.125):
            xt = best_x + shrink * full
            xt[0] = float(np.clip(xt[0], -CAM_LIM, CAM_LIM))
            xt[1] = float(np.clip(xt[1], -THIGH_LIM, THIGH_LIM))
            cand = probe(xt, best_q)
            if cand is None:
                continue
            rn = resid(cand[0])
            if verbose:
                print(f"      it{it}: cam {xt[0]:+.4f} thigh {xt[1]:+.4f} (x{shrink:g}) -> "
                      f"footprint-CoM {cand[0]['off']:+.4f}  z {cand[0]['base_z']:.4f}"
                      f"{'' if rn < best_r else '  rejected'}")
            if rn < best_r:
                best_x, (best_st, best_q), best_r, improved = xt, cand, rn, True
                break
        if not improved:
            break                        # no step of any length helps: this is the fixed point
    return best_x, best_st, best_q


def build(kind, out, balance=False, passes=4, tol_deg=0.25, tol_m=3e-3,
          verbose=True, ref=None):
    """Fixed point over (contact-face orientation, stance). Each pass writes a complete, compilable
    plant, so an aborted run still leaves a valid file."""
    spec = FEET[kind]
    src_txt = _read(BASE)
    base_m = mujoco.MjModel.from_xml_path(BASE)
    base_geom_pos = {s: base_m.geom_pos[
        mujoco.mj_name2id(base_m, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_col")].copy() for s in "LR"}

    ctrl = base_m.key_ctrl[0].copy()
    x = np.array([float(ctrl[1]), float(ctrl[2])])
    if balance:
        # SEED THE BALANCED SOLVE FROM dash01_bal.xml's STANCE, not from the shipped one.
        #
        # "footprint centre under the CoM" is ONE equation in TWO unknowns, so its solutions form a
        # curve, and WHICH point on that curve you land on matters far more than the residual does.
        # Both of these balance the footprint:
        #     cam -0.037 thigh +0.084   (thigh DOWN from the shipped +0.12, base_z 0.997)
        #     cam -0.089 thigh +0.188   (thigh UP,                          base_z 1.012)
        # and only the second is worth anything: it is the keyframe measured to take m3 from 0.91 s
        # 0/6 to 24.7 s with 3/5 surviving the full 60 s ([[scripted-walk-controller]]). A
        # minimum-norm step finds the nearest solution to wherever it starts, which is a property
        # of the seed and not of the robot -- started at the shipped stance it walks down the wrong
        # branch. So start on the branch that is known to work and let each foot re-converge near
        # it; the feet perturb the settle by millimetres, not by a branch.
        bal = os.path.join(HERE, "dash01_bal.xml")
        if os.path.exists(bal):
            k = mujoco.MjModel.from_xml_path(bal).key_ctrl[0]
            x = np.array([float(k[1]), float(k[2])])
            if verbose:
                print(f"  seeding the stance solve from dash01_bal.xml: "
                      f"cam {x[0]:+.4f} thigh {x[1]:+.4f}")
    # SEED FROM THE SHIPPED ROBOT'S OWN SETTLED STANCE, and level pass 1 against it. Seeding from a
    # settle of the un-levelled variant instead (a cylinder with its axis pointing 51 deg up out of
    # the foot, which is not a foot anyone is proposing) is path-dependent enough to matter: it put
    # the blade's footprint 114 mm ahead of the CoM against the sphere's 82 mm, i.e. a 3 cm stance
    # difference that has nothing to do with the foot. Same robot, same pose, different foot.
    ref = ref if ref is not None else env_settle(BASE)
    st = ref
    qpos = ref["qpos"].copy()

    def score(tilt, s):
        """One number to rank passes by, in units of "how many tolerances out". Ranking matters:
        a later pass is not automatically a better one -- re-levelling perturbs the contact and a
        pass can land the robot on its side (measured: tilt 112 deg, one blade 802 mm in the air).
        Whatever pass was best is what gets written."""
        return tilt / tol_deg + (abs(s["off"]) / tol_m if balance else 0.0)

    def settle_at(_pose, xx, q):
        """Settle twice and return (the measurement OF THE FILE AS WRITTEN, the qpos IN it).

        Two separate things go wrong here and only doing both fixes it.

        First, DashEnv's `_resettle_keyframe` runs a 2 s settle from whatever `key_qpos` it is
        handed, and 2 s is not always enough: measured on the balanced sphere, one settle from the
        solved pose reported base_z 0.9865, and settling again from THAT reported 0.9972 and then
        stayed there to 1e-5. So a single settle can be a transient. Hence settle 1.

        Second — and this is the one that actually cost the plants — the returned qpos has to be
        the one LEFT IN THE FILE, not the one the last settle produced. Returning `s.qpos` while
        the file still held the previous qpos meant every reported number described a plant that
        was never written: the build claimed the plate was 0.80 deg off level and reloading it read
        6.58 deg, with one foot 3.8 mm off the floor. So settle 2 measures the file exactly as it
        now stands, and that measurement is what is reported and searched on."""
        c = stance_ctrl(*xx)
        write_plant(src_txt, out, spec, _pose, q, c)
        s = env_settle(out)                       # absorb the transient
        q = s["qpos"]
        write_plant(src_txt, out, spec, _pose, q, c)
        return env_settle(out), q                 # measure THIS file; q is what is in it

    def relevel(_pose, xx, q, _st, n=5):
        """Level the contact face at a FROZEN stance, repeatedly, until it stops moving.

        Levelling on its own is a contraction: the shipped-stance builds run 4.7 -> 0.63 -> 0.16
        deg without help. Interleaving it one-for-one with the stance solve is what broke it -- the
        Newton moves the posture, the posture rotates the foot, and the plate ends up 6.6 deg off
        level, i.e. standing on an edge, which is not a flat foot at all. So the two loops are
        separated: solve the stance, then hold it and let the levelling converge on its own."""
        best_local = (tilt_deg(spec, _st), _pose, _st, q)
        for _ in range(n):
            _pose = _level(spec, base_geom_pos, _st)
            _st, q = settle_at(_pose, xx, q)
            t = tilt_deg(spec, _st)
            if t < best_local[0]:
                best_local = (t, _pose, _st, q)
            if t < 0.5 * tol_deg:
                break
        return best_local[1], best_local[2], best_local[3]

    best = None
    for p in range(1, passes + 1):
        # ORDER MATTERS: level the foot FIRST, then correct the stance. Levelling is not neutral
        # for the footprint -- re-hanging the geom off the toe tip at a new orientation moved the
        # plate's contact patch 28 mm forward, so running it AFTER the stance solve simply undid
        # the solve (tilt converged to 0.017 deg with the footprint stuck at +28 mm for three
        # passes). The Newton's own stance change is small once seeded on the right branch, so it
        # barely disturbs the levelling, and the two converge together.
        pose = _level(spec, base_geom_pos, st)
        st, qpos = settle_at(pose, x, qpos)
        pose, st, qpos = relevel(pose, x, qpos, st)
        if balance:
            x, st, qpos = _newton_stance(lambda xx, q: settle_at(pose, xx, q), x, qpos,
                                         verbose=verbose)
            if st is None:
                raise SystemExit(f"{kind}: the stance solve has no standing posture to start from "
                                 f"(cam {x[0]:+.4f} thigh {x[1]:+.4f}) — nothing was written")
        ctrl = stance_ctrl(*x)

        tilt = tilt_deg(spec, st)
        sc = score(tilt, st)
        if verbose:
            soles = " ".join("%+.4f" % st[s]["bottom"] for s in "LR")
            print(f"  pass {p}: tilt {tilt:.3f} deg   footprint-CoM {st['off']:+.4f} m   "
                  f"base_z {st['base_z']:.4f} m   sole z {soles} m   "
                  f"toe air {np.round(st['toe_h'] * 1000, 1)} mm   score {sc:.2f}")
        if st["stance_delta"] != (0.0, 0.0):
            print(f"  WARNING: the env's stance search moved off the solved posture by "
                  f"{st['stance_delta']} — the plant does not stand where this tool put it")
        if best is None or sc < best[0]:
            best = (sc, {s: dict(pose[s]) for s in "LR"}, qpos.copy(), ctrl.copy(), st, tilt,
                    x.copy())
        if tilt < tol_deg and (not balance or abs(st["off"]) < tol_m):
            break
        if sc > 3.0 * best[0] + 5.0:
            # running away, not converging (the blade did exactly this: a good pass 2, then
            # 12 -> 53 -> 191 as re-levelling walked the settle into the 4-bar's fold branch).
            # The best pass is already banked; grinding out the rest just burns minutes.
            print(f"  pass {p} is {sc / best[0]:.0f}x worse than the best — stopping, "
                  f"keeping pass with score {best[0]:.2f}")
            break

    _, pose, qpos, ctrl, st, tilt, x = best
    write_plant(src_txt, out, spec, pose, qpos, ctrl)
    dims = " x ".join(f"{2000 * v:g}" for v in spec["half"])
    print(f"  wrote {os.path.basename(out)}: {spec['type']} {dims} mm | tilt {tilt:.2f} deg | "
          f"footprint-CoM {st['off']:+.4f} m | base_z {st['base_z']:.4f} m | "
          f"cam {x[0]:+.4f} thigh {x[1]:+.4f}")
    return st


PLANTS = ["dash01.xml", "dash01_blade.xml", "dash01_flat.xml",
          "dash01_sphere_bal.xml", "dash01_blade_bal.xml", "dash01_flat_bal.xml"]


def verify():
    """Report every foot plant side by side, as the env sees it. This IS the plant record for the
    study: contact tilt, where the footprint sits relative to the CoM, ride height, and whether
    both feet are actually on the floor at t=0."""
    kind_of = {"dash01.xml": "sphere", "dash01_blade.xml": "blade", "dash01_flat.xml": "flat",
               "dash01_sphere_bal.xml": "sphere", "dash01_blade_bal.xml": "blade",
               "dash01_flat_bal.xml": "flat"}
    print(f"{'plant':26s} {'foot':7s} {'tilt':>6s} {'foot-CoM':>10s} {'base_z':>8s} "
          f"{'sole L/R (mm)':>16s} {'cam':>8s} {'thigh':>8s}")
    for name in PLANTS:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            print(f"{name:26s} MISSING")
            continue
        st = env_settle(path)
        spec = FEET[kind_of[name]]
        print(f"{name:26s} {kind_of[name]:7s} {tilt_deg(spec, st):5.2f}d {st['off']:+10.4f} "
              f"{st['base_z']:8.4f} {st['toe_h'][0] * 1000:7.2f}/{st['toe_h'][1] * 1000:-6.2f} "
              f"{st['ctrl'][1]:+8.4f} {st['ctrl'][2]:+8.4f}"
              + ("   STANCE SEARCH MOVED" if st["stance_delta"] != (0.0, 0.0) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="report every existing foot plant as the env sees it; build nothing")
    ap.add_argument("--kinds", nargs="*", default=["sphere", "blade", "flat"], choices=list(FEET))
    ap.add_argument("--no-balance", action="store_true", help="skip the *_bal plants")
    ap.add_argument("--only-balance", action="store_true", help="skip the shipped-stance plants")
    ap.add_argument("--passes", type=int, default=6)
    args = ap.parse_args()

    if args.verify:
        verify()
        return

    # The shipped robot's own settled stance is the common reference every foot is built against:
    # same pose, same leg configuration, only the contact swapped.
    ref = env_settle(BASE)
    print(f"reference: the shipped plant settles at base_z {ref['base_z']:.4f} m with its footprint "
          f"{ref['off'] * 1000:+.0f} mm ahead of the CoM\n"
          f"({ORACLE_PRESET}, rigid ankle, resettled — the stance every episode actually starts "
          f"from)\n")

    zs = {}
    for kind in args.kinds:
        if kind != "sphere" and not args.only_balance:   # sphere exists only in its balanced form
            print(f"=== {kind}: shipped stance ===")
            build(kind, os.path.join(HERE, f"dash01_{kind}.xml"), balance=False,
                  passes=args.passes, ref=ref)
        if not args.no_balance:
            print(f"=== {kind}: balanced stance (footprint centre under the CoM) ===")
            zs[kind] = build(kind, os.path.join(HERE, f"dash01_{kind}_bal.xml"), balance=True,
                             passes=args.passes, ref=ref)["base_z"]
        print()

    if len(zs) > 1:
        # Ride height is an OUTPUT of the balanced solve, not a target (see _newton_stance). Say
        # out loud how far apart the arms landed, because that spread is the one confound left in
        # the classical-controller A/B and nobody should have to rediscover it from the XML.
        lo, hi = min(zs.values()), max(zs.values())
        print("balanced-stance ride heights: "
              + "  ".join(f"{k} {v:.4f}" for k, v in zs.items())
              + f"   (spread {1000 * (hi - lo):.0f} mm — a residual confound in any foot A/B "
                f"run on the *_bal plants)")


if __name__ == "__main__":
    main()
