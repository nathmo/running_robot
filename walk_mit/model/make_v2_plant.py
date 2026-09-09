"""Build the DASH-01 Walker v2 plant (dash01_v2.xml) from dash01.xml -- artifact §07, "Leg".

Two changes to the loop closure, both measured-driven:
  (1) the <connect> equalities loop_L/R go from a SOFT constraint (solref 0.005, solimp
      0.95-0.99: the base sank ~15 cm under load in the scripted-walk study, 30x the robot) to the
      ankle-lock values (solref 0.002, solimp 0.999-0.9999), so the constraint itself no longer
      yields;
  (2) an EXPLICIT series spring at the pushrod tip: a slide joint along each pushrod with a
      stiffness calibrated so the stance leg sinks 5 mm under the robot's own weight (~148 N,
      ~30 kN/m foot-referred) -- calibrate_loop_spring.py finds the number; this file bakes it.
      The randomizer scales that stiffness +-50 % per episode (dr_loop_k).
Optionally the joint armature per motor family (the drive fit, fit_drive.py) is baked too, though
the env also writes cfg.drive_armature at load, which is the path the presets use.

The keyframe is rebuilt by joint NAME (the two new slide joints sit at their free length, 0).

    python model/make_v2_plant.py --rod-k 250000 --rod-b 60 [--armature 0.046 0.0216]
    python model/make_v2_plant.py --check          # load + one-leg sink of the written file
"""
import argparse
import os
import sys

import copy

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "dash01.xml")
OUT = os.path.join(HERE, "dash01_v2.xml")

PUSHRODS = {"L": "PushrodLeftNCS-v1", "R": "PushrodRightNCS-v1"}
TIP_SITES = {"L": "pushrod_tip_L", "R": "pushrod_tip_R"}
SHINS = {"L": "LegLeftNCS-v1", "R": "LegRightNCS-v1"}
FEET = {"L": "FootLeftNCS-v1", "R": "FootRightNCS-v1"}
SPRING_JOINTS = {"rod": ("pushrod_slide_L", "pushrod_slide_R"),
                 "shin": ("leg_spring_L", "leg_spring_R")}
LOOPS = ("loop_L", "loop_R")
RIGID_SOLREF = [0.002, 1.0]
RIGID_SOLIMP = [0.999, 0.9999, 0.0001, 0.5, 2.0]
# actuated joints by family, resolved through the ASCII actuator names (the joint names carry a
# double-encoded accent in the export); the env writes cfg.drive_armature into the same DOFs
HIP_ACTS = ("hip_roll_L", "hip_roll_R")
CT_ACTS = ("cam_L", "thigh_L", "cam_R", "thigh_R")


def _joint_qpos_by_name(model, data_qpos):
    out = {}
    for j in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        a = int(model.jnt_qposadr[j])
        w = 7 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else (
            4 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1)
        out[n] = np.array(data_qpos[a:a + w])
    return out


def build_spec(rod_k, rod_b, armature=None, src=SRC, slide_range=0.03, spring="shin"):
    """MjSpec of the v2 plant (not yet compiled). armature = (hip, cam/thigh) or None.

    spring "shin": the series spring is a prismatic joint along the vertical strut of each
                   leg (the Foot body, ankle -> toe): the main load path, the only place a 5 mm
                   sink at 1 BW can come from (the GPU port's reading, adopted for parity)
    spring "rod" : the artifact's literal pushrod-tip spring (barely loaded at stance, see
                   calibrate_loop_spring.py)
    rod_k None   : no spring joint at all (the rigid reference)"""
    spec = mujoco.MjSpec.from_file(src)
    base = mujoco.MjSpec.from_file(src).compile()
    key = None
    for k in spec.keys:
        if k.name == "stand":
            key = k
    if key is None:
        raise RuntimeError("dash01.xml has no 'stand' keyframe")
    key_qpos = np.array(key.qpos)
    key_ctrl = np.array(key.ctrl)
    old = _joint_qpos_by_name(base, key_qpos)
    spec.delete(key)
    # (1) rigid loop closure
    for eq in spec.equalities:
        if eq.name in LOOPS:
            eq.solref = RIGID_SOLREF
            eq.solimp = RIGID_SOLIMP
    # (2) the series spring: a slide joint along each pushrod, from the cam pin toward the tip
    # (rod_k None = the RIGID-ROD reference: same plant, no slide joints)
    bodies = {} if rod_k is None else (PUSHRODS if spring == "rod" else FEET)
    for side, bname in bodies.items():
        body = spec.body(bname)
        if spring == "rod":
            tip = next(s for s in body.sites if s.name == TIP_SITES[side])
            axis = np.asarray(tip.pos, dtype=float)
            jname = f"pushrod_slide_{side}"
        else:
            # the vertical strut: the Foot body from the (locked) ankle down to the toe sphere
            toe = next(g for g in body.geoms if g.name == f"foot_{side}_col")
            axis = np.asarray(toe.pos, dtype=float)           # ankle -> toe, in the foot frame
            jname = f"leg_spring_{side}"
        axis = axis / np.linalg.norm(axis)
        j = body.add_joint(name=jname, type=mujoco.mjtJoint.mjJNT_SLIDE)
        j.axis = axis.tolist()
        j.stiffness = [float(rod_k), 0.0, 0.0]      # per-DOF vector in this MjSpec binding
        j.damping = [float(rod_b), 0.0, 0.0]
        j.springref = 0.0
        j.range = [-float(slide_range), float(slide_range)]
        j.armature = 0.0
    if armature is not None:
        a_hip, a_ct = (float(x) for x in armature)
        for acts, val in ((HIP_ACTS, a_hip), (CT_ACTS, a_ct)):
            for an in acts:
                spec.joint(spec.actuator(an).target).armature = val
    # rebuild the keyframe by joint name (new joints at 0)
    m = spec.compile()
    qpos = np.zeros(m.nq)
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n in old:
            a = int(m.jnt_qposadr[j])
            qpos[a:a + old[n].size] = old[n]
    k = spec.add_key()
    k.name = "stand"
    k.qpos = qpos.tolist()
    k.ctrl = key_ctrl.tolist()
    return spec


def build_model(rod_k, rod_b, armature=None, src=SRC, spring="shin"):
    return build_spec(rod_k, rod_b, armature, src, spring=spring).compile()


def write(rod_k, rod_b, armature=None, src=SRC, out=OUT, spring="shin"):
    spec = build_spec(rod_k, rod_b, armature, src, spring=spring)
    spec.compile()
    xml = spec.to_xml()
    where = "pushrod-tip" if spring == "rod" else "shin-axis"
    header = (f"<!-- DASH-01 Walker v2 plant, generated by model/make_v2_plant.py from dash01.xml:\n"
              f"     rigid loop closure (solref {RIGID_SOLREF[0]}) + {where} series spring "
              f"k={rod_k:.0f} N/m, b={rod_b:.1f} N s/m (calibrate_loop_spring.py)"
              + (f", armature hip/ct = {armature[0]}/{armature[1]}" if armature else "")
              + "\n     Do not edit by hand; re-run the generator. -->\n")
    with open(out, "w", encoding="utf-8") as f:
        f.write(header + xml)
    m = mujoco.MjModel.from_xml_path(out)
    return out, m


def one_leg_sink(model, t_s=2.0, rigid_ref=True, side="L"):
    """Base drop (m) of the stance leg under the robot's own weight, relative to a rigid rod.

    Rig: x/y/roll/pitch/yaw locked, z free, ankles locked (the rigid carbon tube), the OTHER foot
    made non-colliding so ONE leg carries all 148 N, motors holding the stance ctrl. The rod
    stiffness is the only difference between the two settles, so everything else that yields
    (contact, PD compliance under gravity torque) cancels."""
    def settle(m):
        d = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        mujoco.mj_resetDataKeyframe(m, d, kid)
        for n in ("lock_x", "lock_y", "lock_roll", "lock_pitch", "lock_yaw",
                  "lock_ankle_L", "lock_ankle_R"):
            e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, n)
            if e >= 0:
                d.eq_active[e] = 1
        other = "R" if side == "L" else "L"
        for g in (f"foot_{other}_col", f"heel_{other}_col"):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            m.geom_contype[gid] = 0
            m.geom_conaffinity[gid] = 0
        d.ctrl[:] = m.key_ctrl[kid]
        for _ in range(int(t_s / m.opt.timestep)):
            mujoco.mj_step(m, d)
        return float(d.qpos[2])

    z = settle(copy.deepcopy(model))
    if not rigid_ref:
        return z
    # rigid reference: the same plant with NO spring joints (a 1000x stiffer spring is not a
    # reference, it is an unstable integrator at 1 kHz)
    z_ref = settle(build_model(None, 0.0))
    return z_ref - z


def spring_mass(model, spring="shin"):
    """Mass the spring DOF moves (for the damping ratio): the pushrod, or shin + foot."""
    if spring == "rod":
        return 0.071
    return float(model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FootLeftNCS-v1")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rod-k", type=float, default=None, help="pushrod spring stiffness (N/m)")
    ap.add_argument("--rod-b", type=float, default=None, help="pushrod damping (N s/m); default "
                    "= 0.7 critical on the rod mass")
    ap.add_argument("--armature", type=float, nargs=2, default=None,
                    metavar=("HIP", "CAM_THIGH"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--spring", default="shin", choices=("shin", "rod"))
    ap.add_argument("--check", action="store_true", help="load the written file and report")
    args = ap.parse_args()
    if args.check and args.rod_k is None:
        m = mujoco.MjModel.from_xml_path(args.out)
        print(f"{args.out}: nq {m.nq} nv {m.nv} nu {m.nu}")
        for n in SPRING_JOINTS["rod"] + SPRING_JOINTS["shin"]:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j >= 0:
                print(f"  {n}: k {m.jnt_stiffness[j]:.0f} N/m  b {m.dof_damping[m.jnt_dofadr[j]]:.1f}")
        print(f"  one-leg sink (L) {1e3 * one_leg_sink(m):.2f} mm   (R) "
              f"{1e3 * one_leg_sink(m, side='R'):.2f} mm")
        return
    if args.rod_k is None:
        raise SystemExit("--rod-k is required (run calibrate_loop_spring.py to find it)")
    mass = spring_mass(mujoco.MjModel.from_xml_path(SRC), args.spring)
    rod_b = args.rod_b if args.rod_b is not None else 2.0 * 0.7 * np.sqrt(args.rod_k * mass)
    out, m = write(args.rod_k, rod_b, args.armature, out=args.out, spring=args.spring)
    print(f"wrote {out}: nq {m.nq} nv {m.nv} nu {m.nu}, rod k {args.rod_k:.0f} N/m b {rod_b:.1f}")
    if args.check:
        print(f"  one-leg sink (L) {1e3 * one_leg_sink(m):.2f} mm   (R) "
              f"{1e3 * one_leg_sink(m, side='R'):.2f} mm")


if __name__ == "__main__":
    main()
