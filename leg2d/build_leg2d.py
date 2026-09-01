"""Build leg2d/model/leg2d.xml: a planar (sagittal-plane) single-leg boom rig derived from the
canonical training/model/dash01.xml plant -- real measured mass (15.14 kg total), real actuator
specs (AKE90-8 cam/thigh, 144.5 N*m delivered peak, 8:1), real passive ankle spring.

THE RIG: base_x and base_z are free. base_y / base_roll / base_pitch / base_yaw are railed via
equality locks (the ones already in dash01.xml -- flip `active` to "true"). The torso can
translate in the sagittal plane but never rotate or drift sideways -- this sidesteps the whole-body
pitch-balance problem (the thing that actually walls every RL run at m3+, see
memory/cmd-curriculum-deadlock.md, memory/m3-pitch-fix.md), so a gait can be commanded open-loop
and just checked for feasibility, without needing a trained/tuned balance controller.

base_z is deliberately NOT railed. An earlier version of this rig also pinned z (rigid equality,
"the base can only go forward") to dodge a single-leg buckling problem that showed up once z was
loaded with the full 15.14 kg -- but a rigid equality lock on base_z holds the TORSO up directly,
bypassing the leg entirely, which means the stance foot no longer needs (or gets) any real normal
force: measured mean contact force was 0.56 N against a 148 N robot weight. That rig was answering
"how fast can this leg move the body horizontally with no weight on the foot", not "how fast can
this leg walk". Here z is free so the leg does the real work of holding the body up, and the
earlier buckling problem is fixed at its actual cause instead (see gait.py's `z_off`).

RIGHT LEG: deleted (single-leg rig) but its mass is preserved as a fixed point mass welded to the
torso -- real single support carries the WHOLE robot's weight, including the idle swing leg;
dropping that mass would make every gait look easier than it really is.

ACTUATION: cam_L / thigh_L are direct-torque <motor> actuators (not dash01's constant-forcerange
<position> actuators) -- see motor.py's module docstring for why.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "training" / "model" / "dash01.xml"
OUT_DIR = Path(__file__).resolve().parent / "model"
OUT = OUT_DIR / "leg2d.xml"
MESHDIR = "../../training/model"                 # relative to OUT; mesh files are "meshes/*.stl" under it

# crouch used for the settle (same LEFT-side values as dash01's INIT_CTRL)
CAM_TARGET, THIGH_TARGET = 0.0, 0.12
SETTLE_KP, SETTLE_KV = 200.0, 5.0                 # matches dash01's cam/thigh position-actuator gains
PEAK_NM = 144.5                                   # measured delivered peak, see spiderbot-hardware.md


def find_body(root, name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    return None


def subtree_mass(elem):
    return sum(float(i.get("mass")) for i in elem.iter("inertial"))


def transform():
    tree = ET.parse(SRC)
    root = tree.getroot()
    root.set("model", "leg2d")
    root.find("compiler").set("meshdir", MESHDIR)

    wb = root.find("worldbody")
    torso = wb.find("body")                       # bodyNCS-v1

    # ---- 1. delete the right leg, preserve its mass as a fixed point mass on the torso ----
    hip_right = find_body(root, "HipRightNCS-v1")
    motor_hr_r = find_body(root, "motor_hip_roll_R")
    dead_mass = subtree_mass(hip_right) + subtree_mass(motor_hr_r)
    torso.remove(hip_right)
    torso.remove(motor_hr_r)
    torso.append(ET.fromstring(
        f'<body name="deadweight_R"><inertial pos="0 0 0" mass="{dead_mass:.6f}" '
        f'diaginertia="0.001 0.001 0.001"/></body>'))

    # ---- 2. equality: rail y/roll/pitch/yaw (boom), free x/z, drop the dangling right-side eq ----
    eq = root.find("equality")
    for c in list(eq):
        name = c.get("name")
        if name in ("loop_R", "lock_ankle_R"):
            eq.remove(c)
        elif name in ("lock_y", "lock_roll", "lock_pitch", "lock_yaw"):
            c.set("active", "true")
        elif name in ("lock_x", "lock_z"):
            c.set("active", "false")

    # ---- 3. actuators: drop the right-side three; cam_L/thigh_L -> direct-torque motors ----
    act = root.find("actuator")
    for a in list(act):
        if a.get("name") in ("hip_roll_R", "cam_R", "thigh_R"):
            act.remove(a)
    for name in ("cam_L", "thigh_L"):
        a = next(e for e in act if e.get("name") == name)
        joint = a.get("joint")                     # reuse the exact string, avoids re-typing accents
        act.remove(a)
        act.append(ET.fromstring(
            f'<motor name="{name}" joint="{joint}" gear="1" '
            f'ctrlrange="-{PEAK_NM} {PEAK_NM}" forcerange="-{PEAK_NM} {PEAK_NM}"/>'))

    # ---- 4. sensors: drop right-side ----
    sen = root.find("sensor")
    for s in list(sen):
        nm = s.get("name") or ""
        if nm.startswith("hip_roll_R") or nm.startswith("cam_R") or nm.startswith("thigh_R"):
            sen.remove(s)

    # ---- 5. clear the stale keyframe (wrong nq now); rebuilt after settling below ----
    kf = root.find("keyframe")
    for k in list(kf):
        kf.remove(k)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="unicode", xml_declaration=False)
    return tree, kf, dead_mass


def _pd_step(model, data, j_cam, j_thigh, a_cam, a_thigh, a_hr):
    q_cam = data.qpos[model.jnt_qposadr[j_cam]]
    dq_cam = data.qvel[model.jnt_dofadr[j_cam]]
    q_thigh = data.qpos[model.jnt_qposadr[j_thigh]]
    dq_thigh = data.qvel[model.jnt_dofadr[j_thigh]]
    tau_cam = float(np.clip(SETTLE_KP * (CAM_TARGET - q_cam) - SETTLE_KV * dq_cam, -PEAK_NM, PEAK_NM))
    tau_thigh = float(np.clip(SETTLE_KP * (THIGH_TARGET - q_thigh) - SETTLE_KV * dq_thigh, -PEAK_NM, PEAK_NM))
    data.ctrl[a_cam], data.ctrl[a_thigh], data.ctrl[a_hr] = tau_cam, tau_thigh, 0.0
    return tau_cam, tau_thigh


def settle_keyframe(tree, kf, dead_mass):
    """Two-phase settle, mirroring training/model/build_model.py's compute_standing_keyframe:
    (1) gravity OFF, base frozen high in the air, so the closed 4-bar loop (a stiff <connect>
    equality) relaxes into a consistent pose from the crouch target BEFORE any contact or gravity
    force fights it -- skipping this step starts the sim with the loop badly violated and the
    stiff constraint detonates the pose in the first few steps.
    (2) drop the torso so the foot just touches the floor, then re-settle WITH gravity, base_x/y/
    attitude pinned (y/roll/pitch/yaw are also equality-locked in the compiled model, but pinning
    qpos directly here is robust regardless of the eq_active state), base_z free."""
    model = mujoco.MjModel.from_xml_path(str(OUT))
    data = mujoco.MjData(model)
    a_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cam_L")
    a_thigh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thigh_L")
    a_hr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_roll_L")
    # joint ids from the actuator's transmission target, NOT a re-typed "Révolution" name string --
    # the CAD export's joint names are mis-encoded bytes that don't round-trip through a literal in
    # this file (mj_name2id on the retyped name silently returns -1, and numpy's negative-index
    # wraparound then reads/writes the WRONG joint with no error -- this bit us once already).
    j_cam = int(model.actuator_trnid[a_cam, 0])
    j_thigh = int(model.actuator_trnid[a_thigh, 0])
    g_foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot_L_col")
    g_heel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "heel_L_col")

    # ---- phase 1: gravity off, base frozen at z=1.5, close the 4-bar loop from the crouch ----
    g = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    data.qpos[0:3] = [0, 0, 1.5]
    data.qpos[3:6] = 0
    for _ in range(2000):
        _pd_step(model, data, j_cam, j_thigh, a_cam, a_thigh, a_hr)
        mujoco.mj_step(model, data)
        data.qpos[0:6] = [0, 0, 1.5, 0, 0, 0]
        data.qvel[0:6] = 0
    mujoco.mj_forward(model, data)
    model.opt.gravity[:] = g

    # ---- phase 2: drop so the foot just touches the floor, then settle loaded ----
    zmin = min(float(data.geom_xpos[gg][2]) - float(model.geom_size[gg][0]) for gg in (g_foot, g_heel))
    data.qpos[0:3] = [0, 0, 1.5 - zmin + 0.002]
    data.qpos[3:6] = 0
    data.qvel[:] = 0
    # SINGLE-leg support carries the whole 15.1 kg (double what dash01's double-support crouch
    # target was tuned around), and the drive's position loop is P-only (see spiderbot-hardware.md
    # -- a loaded joint droops short of its target by ~gravity_torque/kp) -- so this settles several
    # seconds slower, and lower, than dash01's own two-leg keyframe. That droop is real plant
    # behaviour, not a settle bug (confirmed: raising the settle-only damping 6x barely changed the
    # convergence rate, i.e. it is a soft asymptotic approach, not an underdamped oscillation). The
    # gait's z_off parameter (gait.py) exists to compensate for exactly this droop during ACTIVE
    # stepping, once the optimizer is driving cam/thigh through a moving trajectory rather than
    # holding still.
    tau_cam = tau_thigh = 0.0
    for _ in range(15000):
        tau_cam, tau_thigh = _pd_step(model, data, j_cam, j_thigh, a_cam, a_thigh, a_hr)
        mujoco.mj_step(model, data)
        data.qpos[0:2] = 0
        data.qpos[3:6] = 0
        data.qvel[0:2] = 0
        data.qvel[3:6] = 0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    def f3(v):
        return " ".join(f"{x:.6g}" for x in v)

    ctrl = np.zeros(model.nu)
    ctrl[a_cam], ctrl[a_thigh] = tau_cam, tau_thigh
    key = ET.fromstring(f'<key name="stand" qpos="{f3(data.qpos)}" ctrl="{f3(ctrl)}"/>')
    kf.append(key)
    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="unicode", xml_declaration=False)

    final = mujoco.MjModel.from_xml_path(str(OUT))
    fd = mujoco.MjData(final)
    mujoco.mj_resetDataKeyframe(final, fd, 0)
    mujoco.mj_forward(final, fd)
    total_mass = float(final.body_subtreemass[
        mujoco.mj_name2id(final, mujoco.mjtObj.mjOBJ_BODY, "bodyNCS-v1")])
    print(f"wrote {OUT}: nq={final.nq} nv={final.nv} nu={final.nu} neq={final.neq}  "
          f"stand height z={float(fd.qpos[2]):.4f} m  total mass={total_mass:.3f} kg  "
          f"dead-weight (removed right leg) = {dead_mass:.3f} kg")


if __name__ == "__main__":
    tree, kf, dead_mass = transform()
    settle_keyframe(tree, kf, dead_mass)
