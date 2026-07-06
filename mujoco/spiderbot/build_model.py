"""Generate a simulation-ready MuJoCo model from the CAD OPEN_B export.

Copies the OPEN_B body tree verbatim (bodies/inertials/meshes are placeholders to be replaced
with measured values later) and layers on everything needed to actually simulate and train:
  - a free-floating base, world <option>, ground plane, lighting
  - PD position actuators on the 6 motors  (AKE90-8 on cam+thigh, AK60-39 on hip-roll)
  - the closed parallel loop (pushrod tip -> shin) via <equality><connect>
  - a passive preloaded ankle spring
  - an IMU + per-motor sensors (NO foot-contact sensor)
  - simple box foot collision (visual meshes are non-colliding)
  - a standing keyframe

All tunable numbers live in MOTORS / JOINTS below so the eventual real values are a one-edit swap.
Re-run after CAD regen:  .venv/Scripts/python.exe mujoco/spiderbot/build_model.py
"""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from geometry import loop_anchors, foot_tips, foot_heels

# slight thigh-forward crouch used as the standing init pose (feet under the CoM)
INIT_CTRL = np.array([0.0, 0.0, 0.12, 0.0, 0.0, -0.12])
# A small sphere at each foot TIP is the walking contact (a point/ball foot): the robot must
# actively balance / step on its toe points. A second sphere at the far (ankle) end of the foot is
# a HEEL STOP — it clears the floor in the toe-down stance (~28 cm up) so normal contact stays
# toe-only, but it catches the floor if the leg folds and the long foot flattens, so the foot can
# physically never clip through the ground. (Without it the ghost foot mesh traversed the floor,
# and the old vertex-scan termination that guarded against that fired on every leg-fold.)
FOOT_SPHERE_R = 0.025
HEEL_SPHERE_R = 0.03


def compute_standing_keyframe(model, init_ctrl):
    """Settle the init pose in the air (gravity off, base frozen) so the closed loop is
    consistent, drop the torso so the toes touch, then RE-SETTLE UNDER GRAVITY (base xy +
    attitude held, z free) so the recorded keyframe is the true LOADED stance — PD sag, ankle
    spring compression and contact deflection included. (The old gravity-off keyframe sat
    ~3 cm above the real loaded height, baking a constant error into the height reward.)"""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    g = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    data.qpos[:3] = [0, 0, 1.5]
    data.qpos[3:7] = [1, 0, 0, 0]
    base_q = data.qpos[:7].copy()
    data.ctrl[:] = init_ctrl
    for _ in range(2000):
        mujoco.mj_step(model, data)
        data.qpos[:7] = base_q
        data.qvel[:6] = 0
    mujoco.mj_forward(model, data)
    model.opt.gravity[:] = g
    foot_g = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{s}_col") for s in "LR"]
    zmin = min(data.geom_xpos[gg][2] - model.geom_rbound[gg] for gg in foot_g)  # sphere bottom
    data.qpos[0:3] = [0, 0, 1.5 - zmin + 0.002]   # drop torso so feet just touch z=0
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    for _ in range(1500):                          # 1.5 s loaded settle, xy + attitude pinned
        mujoco.mj_step(model, data)
        data.qpos[0:2] = 0
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qvel[0:2] = 0
        data.qvel[3:6] = 0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return data.qpos.copy()

OPEN_B = "robotCADdescription/MJCF_OPEN_MUJOCO_B/SpiderBot/SpiderBot.xml"
OUT = "mujoco/spiderbot/spiderbot.xml"
MESHDIR = "../../robotCADdescription/MJCF_OPEN_MUJOCO_B/SpiderBot"  # relative to OUT

# --- Motor specs (output/joint side, after gearbox). Edit here when real values are known. ---
# mass/inertia: the CAD part inertials OMIT the actuators (~7.1 kg of motors on a ~5.7 kg CAD
# model), so each motor is welded onto its stator body as a point mass at the joint anchor.
# inertia is a rough solid-cylinder estimate; replace with measured values with the rest.
AKE90 = dict(kp=200, kv=5.0, forcerange=170, armature=0.0216,   # cam + thigh (propulsion)
             mass=1.40, inertia=0.002)
AK60 = dict(kp=120, kv=4.0, forcerange=72, armature=0.046,      # hip-roll (lateral)
            mass=0.75, inertia=0.001)

# --- Per-joint setup. range=None => unlimited. act => actuator (motor spec). ---
#  name : (role, range, damping, armature, motor)
J = {
    "bodyNCS-v1_Révolution-1":   ("act", (-0.785, 0.785), 0.1, AK60["armature"],  AK60),  # hip-roll L
    "bodyNCS-v1_Révolution-2":   ("act", (-0.785, 0.785), 0.1, AK60["armature"],  AK60),  # hip-roll R
    "HipLeftNCS-v1_Révolution-3":  ("act", (-1.5, 1.5),   0.1, AKE90["armature"], AKE90), # cam L
    "HipRightNCS-v1_Révolution-4": ("act", (-1.5, 1.5),   0.1, AKE90["armature"], AKE90), # cam R
    "HipLeftNCS-v1_Révolution-5":  ("act", (-1.047, 1.047), 0.1, AKE90["armature"], AKE90), # thigh L
    "HipRightNCS-v1_Révolution-6": ("act", (-1.047, 1.047), 0.1, AKE90["armature"], AKE90), # thigh R
    "ThighLeftNCS-v1_Révolution-7":  ("passive", None, 0.2, 0.001, None),  # knee L (loop-driven)
    "ThighRightNCS-v1_Révolution-8": ("passive", None, 0.2, 0.001, None),  # knee R
    "LegLeftNCS-v1_Révolution-9":   ("ankle", (-1.047, 1.047), 0.3, 0.001, None),  # ankle L (spring)
    "LegRightNCS-v1_Révolution-10": ("ankle", (-1.047, 1.047), 0.3, 0.001, None),  # ankle R
    "CamLeftNCS-v1_Révolution-11":  ("passive", None, 0.2, 0.001, None),  # pushrod pivot L (loop)
    "CamRightNCS-v1_Révolution-12": ("passive", None, 0.2, 0.001, None),  # pushrod pivot R
}

# Ankle passive spring: real spec is 0.5 Nm/deg = 28.65 Nm/rad with a 2.27 Nm preload.
# springref = the foot's neutral (flat) pitch, tuned per side via tune_ankle.py (axes are mirrored).
# NOTE: MuJoCo's linear spring can't represent the 2.27 Nm preload "breakaway"; that's a TODO
# (one-sided joint limit + spring). Stiffness is the real value; preload is currently approximated.
ANKLE_STIFFNESS = 28.65
# Toe-DOWN neutral so the long foot rests on its toe ball with the heel/rest-of-foot clear of the
# floor (~8.5 cm), tuned via tune_foot_posture.py. Required because only the toe sphere may touch.
ANKLE_SPRINGREF = {"LegLeftNCS-v1_Révolution-9": -0.7, "LegRightNCS-v1_Révolution-10": 0.7}

# Actuator order (also the ctrl / observation order).
ACTUATORS = [
    ("hip_roll_L", "bodyNCS-v1_Révolution-1"),
    ("cam_L",      "HipLeftNCS-v1_Révolution-3"),
    ("thigh_L",    "HipLeftNCS-v1_Révolution-5"),
    ("hip_roll_R", "bodyNCS-v1_Révolution-2"),
    ("cam_R",      "HipRightNCS-v1_Révolution-4"),
    ("thigh_R",    "HipRightNCS-v1_Révolution-6"),
]


def f3(v):
    return " ".join(f"{x:.6g}" for x in v)


def add_motor_masses(root):
    """Weld each motor's mass onto its STATOR body (the joint's parent) at the joint anchor.
    The anchor comes from the compiled model's world-frame xanchor mapped into the parent frame
    (not hand-composed from body_pos/body_quat), so it stays correct even if a CAD regen inserts
    intermediate frames. Bodies with an explicit <inertial> ignore geom masses, hence a child
    body with its own inertial."""
    model = mujoco.MjModel.from_xml_path(OUT)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)                     # default qpos: frames + xanchor populated
    bodies = {b.get("name"): b for b in root.iter("body")}
    for aname, jname in ACTUATORS:
        spec = J[jname][4]
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        pb = model.body_parentid[model.jnt_bodyid[j]]
        Rp = data.xmat[pb].reshape(3, 3)
        anchor = Rp.T @ (data.xanchor[j] - data.xpos[pb])
        parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, pb)
        bodies[parent].append(ET.fromstring(
            f'<body name="motor_{aname}" pos="{f3(anchor)}">'
            f'<inertial pos="0 0 0" mass="{spec["mass"]}" '
            f'diaginertia="{spec["inertia"]} {spec["inertia"]} {spec["inertia"]}"/>'
            f'</body>'))


def build():
    anchors, tips, heels = loop_anchors(), foot_tips(), foot_heels()
    tree = ET.parse(OPEN_B)
    root = tree.getroot()
    root.set("model", "spiderbot")

    # compiler
    comp = root.find("compiler")
    if comp is None:
        comp = ET.SubElement(root, "compiler")
    comp.set("angle", "radian")
    comp.set("meshdir", MESHDIR)
    comp.set("autolimits", "true")

    # option + statistic (insert near the top). impratio=10: with the default 1, a loaded foot
    # creeps several mm/s BELOW the friction cone (measured 2.6-7.9 mm/s at 0.3-0.9 mu*N) — free
    # translation that real stick friction does not have; 10 cuts it ~10x.
    opt = ET.fromstring(
        '<option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="10" '
        'solver="Newton" iterations="100" ls_iterations="50"/>')
    root.insert(list(root).index(comp) + 1, opt)

    # defaults: visual meshes non-colliding; a collision class; site styling.
    # condim=6: at condim=3 the torsional/rolling friction entries are silently IGNORED, making
    # the toe spheres frictionless ball casters (free pivot + free rolling = the skating/spin
    # exploits). Coefficients are realistic for a rubber pad on indoor floor, NOT anti-skate
    # knobs: torsional ~ (2/3)*mu*patch_radius ~ 0.008 m, rolling ~ Crr*r ~ 0.001 m.
    default = ET.fromstring(
        '<default>'
        '<default class="spiderbot">'
        '<geom contype="0" conaffinity="0" group="2"/>'
        '<site group="4" size="0.012" rgba="0.95 0.45 0.1 1"/>'
        '<default class="collision">'
        '<geom contype="1" conaffinity="1" group="3" condim="6" '
        'friction="1 0.008 0.001" solref="0.01 1" solimp="0.95 0.99 0.001" '
        'rgba="0.2 0.8 0.2 0.4"/>'
        '</default></default></default>')
    root.insert(list(root).index(opt) + 1, default)

    # ground texture/material
    asset = root.find("asset")
    asset.append(ET.fromstring(
        '<texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" '
        'rgb2="0.1 0.15 0.2" width="300" height="300"/>'))
    asset.append(ET.fromstring(
        '<material name="grid" texture="grid" texrepeat="6 6" reflectance="0.1"/>'))

    wb = root.find("worldbody")
    # replace CAD's tiny visual plane with a proper collidable floor + light
    for g in wb.findall("geom"):
        if g.get("type") == "plane":
            wb.remove(g)
    for lt in wb.findall("light"):
        wb.remove(lt)
    wb.insert(0, ET.fromstring('<geom name="floor" type="plane" size="0 0 0.05" '
                               'material="grid" contype="1" conaffinity="1" condim="6" '
                               'friction="1 0.008 0.001" solref="0.01 1" '
                               'solimp="0.95 0.99 0.001"/>'))
    wb.insert(0, ET.fromstring('<light name="top" pos="0 0 3" dir="0 0 -1" directional="true"/>'))

    # base body: childclass + freejoint + imu site
    base = wb.find("body")
    base.set("childclass", "spiderbot")
    base.insert(0, ET.fromstring('<site name="imu" pos="0 0 0" size="0.015" rgba="0.1 0.5 0.95 1"/>'))
    base.insert(0, ET.fromstring('<freejoint name="root"/>'))

    # configure every joint
    for jnt in root.iter("joint"):
        name = jnt.get("name")
        role, rng, damping, armature, _ = J[name]
        jnt.set("damping", str(damping))
        jnt.set("armature", str(armature))
        if rng is None:
            jnt.set("limited", "false")
        else:
            jnt.set("limited", "true")
            jnt.set("range", f"{rng[0]:.6g} {rng[1]:.6g}")
        if role == "ankle":
            jnt.set("stiffness", str(ANKLE_STIFFNESS))
            jnt.set("springref", f"{ANKLE_SPRINGREF[name]:.6g}")

    # find bodies by name to attach loop sites + foot collision
    bodies = {b.get("name"): b for b in root.iter("body")}
    for s in "LR":
        push = "PushrodLeftNCS-v1" if s == "L" else "PushrodRightNCS-v1"
        leg = "LegLeftNCS-v1" if s == "L" else "LegRightNCS-v1"
        foot = "FootLeftNCS-v1" if s == "L" else "FootRightNCS-v1"
        bodies[push].append(ET.fromstring(
            f'<site name="pushrod_tip_{s}" pos="{f3(anchors[s]["pushrod_site"])}"/>'))
        bodies[leg].append(ET.fromstring(
            f'<site name="leg_anchor_{s}" pos="{f3(anchors[s]["leg_site"])}"/>'))
        bodies[foot].append(ET.fromstring(
            f'<geom name="foot_{s}_col" class="collision" type="sphere" '
            f'size="{FOOT_SPHERE_R}" pos="{f3(tips[s])}"/>'))
        bodies[foot].append(ET.fromstring(
            f'<geom name="heel_{s}_col" class="collision" type="sphere" '
            f'size="{HEEL_SPHERE_R}" pos="{f3(heels[s])}"/>'))

    # equality: close both loops
    eq = ET.SubElement(root, "equality")
    for s in "LR":
        eq.append(ET.fromstring(
            f'<connect name="loop_{s}" site1="pushrod_tip_{s}" site2="leg_anchor_{s}" '
            f'solref="0.005 1" solimp="0.95 0.99 0.001"/>'))

    # actuators (PD position)
    act = ET.SubElement(root, "actuator")
    for aname, jname in ACTUATORS:
        m = J[jname][4]
        rng = J[jname][1]
        act.append(ET.fromstring(
            f'<position name="{aname}" joint="{jname}" kp="{m["kp"]}" kv="{m["kv"]}" '
            f'forcerange="-{m["forcerange"]} {m["forcerange"]}" '
            f'ctrlrange="{rng[0]:.6g} {rng[1]:.6g}"/>'))

    # sensors: IMU + per-motor feedback (no contact sensor)
    sen = ET.SubElement(root, "sensor")
    sen.append(ET.fromstring('<accelerometer name="imu_acc" site="imu"/>'))
    sen.append(ET.fromstring('<gyro name="imu_gyro" site="imu"/>'))
    sen.append(ET.fromstring('<framequat name="base_quat" objtype="site" objname="imu"/>'))
    for aname, jname in ACTUATORS:
        sen.append(ET.fromstring(f'<jointpos name="{aname}_pos" joint="{jname}"/>'))
        sen.append(ET.fromstring(f'<jointvel name="{aname}_vel" joint="{jname}"/>'))
        sen.append(ET.fromstring(f'<actuatorfrc name="{aname}_frc" actuator="{aname}"/>'))

    # placeholder keyframe (overwritten below, once the model can be simulated)
    kf = ET.SubElement(root, "keyframe")
    key = ET.fromstring(
        f'<key name="stand" qpos="{f3([0,0,1.0,1,0,0,0]+[0.0]*12)}" ctrl="{f3([0]*6)}"/>')
    kf.append(key)
    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="unicode", xml_declaration=False)

    # weld the motor masses on (needs the compiled model for the anchor frames), then rewrite
    add_motor_masses(root)
    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="unicode", xml_declaration=False)

    # compute a real standing keyframe by settling the crouched init pose, then rewrite
    model = mujoco.MjModel.from_xml_path(OUT)
    qpos = compute_standing_keyframe(model, INIT_CTRL)
    key.set("qpos", f3(qpos))
    key.set("ctrl", f3(INIT_CTRL))
    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="unicode", xml_declaration=False)
    print(f"wrote {OUT}")

    # the keyframe edit only changes key_qpos, not masses/sizes — reuse the compiled model for
    # the summary and just verify the final file parses
    total_mass = float(model.body_subtreemass[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bodyNCS-v1")])
    mujoco.MjModel.from_xml_path(OUT)
    print(f"compiled OK: nq={model.nq} nv={model.nv} nu={model.nu} neq={model.neq} "
          f"nsensor={model.nsensor} nkey={model.nkey}; stance height={qpos[2]:.3f} m; "
          f"total mass={total_mass:.2f} kg")


if __name__ == "__main__":
    build()
