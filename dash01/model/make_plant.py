"""Build the DASH-01 simulation plant from the CAD export.

    python -m model.make_plant              # writes dash01_free.xml + dash01_planar.xml
    python -m model.make_plant --report     # build + print every validation number, write nothing

The input is ``Dash-01CAD/dash01.xml``, a raw CAD export: meshes, body frames, joint axes and
CAD-density inertias, with each closed kinematic loop broken by DUPLICATING a body.  It does not
compile (repeated names) and it is not a robot model (no base DOF, no actuators, no sensors, no
collision geometry, no ground).  This script turns it into one, and every step is checked rather
than asserted.

WHAT THE HARDWARE IS.  A 6-DOF biped.  Each leg carries three actuated joints -- hip roll, cam and
thigh -- and a passive planar four-bar (cam -> pushrod -> leg) that drives the knee.  The cam is a
crank: it does NOT set knee angle directly, it sets it through the rod, so the leg's effective
knee angle is a nonlinear function of both cam and thigh.  The leg segment is a single rigid part
that ends in a flat sole; there is no ankle joint.

THE LOOP.  The exporter writes the four-bar by emitting the shared body twice, once down each
branch, which is why the file has repeated names.  Both copies land at the same world pose in the
exported configuration (checked below to sub-micron), so the export is a consistent assembled
pose and the cut is free: we keep the leg under the thigh, keep the pushrod under the cam, delete
both duplicates, and close the loop with a ``connect`` equality at the rod's outboard hinge.  A
spatial ``connect`` is three constraints where a planar revolute needs two, so one is redundant;
it is satisfied identically by the mechanism's symmetry and MuJoCo's solver is regularised, which
is why this is the standard way to write a planar four-bar in MJCF.

JOINT ZERO IS A HARDWARE FACT, NOT AN EXPORT ARTIFACT.  The CAD was saved with the mechanism in
some arbitrary pose, and the exporter bakes that pose into the body frames -- so the export's
qpos=0 is not the robot's homed zero.  Every frame that differs from the previous plant differs by
a PURE rotation about that body's own joint axis (verified below, off-axis residual 0.000000 deg),
which is exactly what a pose difference looks like and is the proof that no geometry changed.
``ZERO_SHIFT`` rotates the actuated bodies back onto the robot's homing convention, so the joint
zeros the motor controllers and ``robot/deploy`` already use stay valid.  Passive joints keep the
export's zero: nothing outside this file refers to them.

MASSES.  CAD densities are wrong -- measured segments differ from CAD by up to 5x (the cam is
66 g, CAD says 13 g).  ``MEASURED_KG`` holds the weighed assembly masses; each CAD link keeps its
CAD centre of mass and gets its inertia tensor scaled by the mass ratio.  That is exact for a
density error and approximate for anything else, and it is the honest fallback until inertia
tensors are measured.  Motors are separate point-mass bodies welded to the link they sit on, so a
weighed assembly mass is split between the link and its motors.
"""
import argparse
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
CAD_DIR = PKG.parent / "Dash-01CAD"
CAD = CAD_DIR / "dash01.xml"

# ---------------------------------------------------------------------------------------------
# Naming.  The CAD export names bodies "bodyNCS-v1" and joints "HipLeftNCS-v1_Revolution-3" with a
# non-ASCII 'e' whose encoding differs between the file, ElementTree and MuJoCo -- a real source of
# bugs in the previous plant.  Everything is renamed once, here, to names the rest of the package
# uses literally.  Body and joint namespaces are separate in MuJoCo, so a body and a joint may
# share a name (body "cam_L" carries joint "cam_L"); that is deliberate and reads well.
# ---------------------------------------------------------------------------------------------
BODY = {
    "bodyNCS-v1": "torso",
    "HipLeftNCS-v1": "hip_L", "HipRightNCS-v1": "hip_R",
    "CamLeftNCS-v1": "cam_L", "CamRightNCS-v1": "cam_R",
    "PushrodLeftNCS-v1": "rod_L", "PushrodRightNCS-v1": "rod_R",
    "ThighLeftNCS-v1": "thigh_L", "ThighRightNCS-v1": "thigh_R",
    "FootFlatLeftNCS-v1": "leg_L", "FootFlatRightNCS-v1": "leg_R",
}
# joint name -> (clean name, side); keyed by the CAD suffix so the accented stem never matters
JOINT_SUFFIX = {
    "1": ("hip_roll_L", "L"), "2": ("hip_roll_R", "R"),
    "3": ("cam_L", "L"), "4": ("cam_R", "R"),
    "5": ("thigh_L", "L"), "6": ("thigh_R", "R"),
    "11": ("rod_L", "L"), "12": ("rod_R", "R"),
    "16": ("knee_L", "L"), "17": ("knee_R", "R"),
}
ACTUATED = ["hip_roll_L", "cam_L", "thigh_L", "hip_roll_R", "cam_R", "thigh_R"]
PASSIVE = ["rod_L", "rod_R", "knee_L", "knee_R"]

# ---------------------------------------------------------------------------------------------
# Joint-zero convention.  Rotation (rad, about the joint's own axis, right-handed) applied to the
# LEFT body's frame to carry the CAD's exported pose onto the robot's homed zero; the right side
# gets the negated value.  Magnitudes are the mean of the two sides measured against the previous
# plant and applied ANTI-SYMMETRICALLY on purpose: the previous plant's own export was slightly
# asymmetric (its right hip carried a 1.21 deg roll offset and its right-leg joint axes were not
# unit vectors) while this export is clean, so forcing symmetry here removes a defect rather than
# introducing one.  check_zero_convention() measures what is left.
#
# hip_roll stays at the new export's symmetric zero, which moves the RIGHT hip roll 1.21 deg from
# the previous plant.  That offset was an artefact of the older CAD export, not a hardware fact:
# it appeared on one side only, and the robot's two hip castings are mirror images.
#
# The passive joints (rod, knee) keep the export's zero.  Nothing outside the model refers to
# them -- they carry no encoder and no command -- so their zero is free.
# ---------------------------------------------------------------------------------------------
ZERO_SHIFT = {"cam": 1.8770715, "thigh": -0.1597815, "hip_roll": 0.0}
# Deviations from the previous plant that are intended, in degrees, checked but not failed.
ZERO_TOLERATED = {"hip_R": 1.25, "cam_L": 0.05, "cam_R": 0.05, "thigh_L": 0.05, "thigh_R": 0.05}

# Weighed assembly masses in kg. An assembly = one CAD link + the motor point masses welded to it.
MOTOR_KG = {"motor_hip_roll_L": 0.75, "motor_hip_roll_R": 0.75,
            "motor_cam_L": 1.4, "motor_thigh_L": 1.4, "motor_cam_R": 1.4, "motor_thigh_R": 1.4}
MEASURED_KG = {
    "torso": ("torso", ["motor_hip_roll_L", "motor_hip_roll_R"], 5.764),
    "hip_L": ("hip_L", ["motor_cam_L", "motor_thigh_L"], 3.271),
    "hip_R": ("hip_R", ["motor_cam_R", "motor_thigh_R"], 3.271),
    "cam_L": ("cam_L", [], 0.066), "cam_R": ("cam_R", [], 0.066),
    "rod_L": ("rod_L", [], 0.071), "rod_R": ("rod_R", [], 0.071),
    "thigh_L": ("thigh_L", [], 0.483), "thigh_R": ("thigh_R", [], 0.483),
    # The merged leg/foot is a NEW part and has not been weighed.  LEG_KG below is a CAD mass
    # corrected by the density error the previous leg showed; --leg-kg overrides it.  Distal mass
    # is the dominant cost term for a runner, so this is the number to replace first.
    "leg_L": ("leg_L", [], None), "leg_R": ("leg_R", [], None),
}
# CAD said 405 g.  The parts it replaces (shin 222 g CAD / 324 g weighed, foot 259 g CAD / 222 g
# weighed) came in 13.5% heavier than CAD in total, so 405 g * 1.135.  PROVISIONAL -- weigh it.
LEG_KG = 0.460
LEG_KG_IS_MEASURED = False

# Motor mounting points, in the frame of the body they are welded to (torso for the hip-roll
# motors, the hip casting for the cam and thigh motors) -- taken from each joint's own anchor.
MOTOR_MOUNT = {
    "motor_hip_roll_L": ("torso", "hip_roll_L"), "motor_hip_roll_R": ("torso", "hip_roll_R"),
    "motor_cam_L": ("hip_L", "cam_L"), "motor_thigh_L": ("hip_L", "thigh_L"),
    "motor_cam_R": ("hip_R", "cam_R"), "motor_thigh_R": ("hip_R", "thigh_R"),
}

# Per-joint dynamics.  Armature is reflected rotor inertia, fitted to the measured closed-loop
# corner (6.3 Hz at kp 200 / kd 5); damping is joint friction.  Ranges are hardware travel.
JOINT_DYN = {
    "hip_roll": dict(damping=0.1, armature=0.046, range=(-0.785, 0.785)),
    "cam": dict(damping=0.1, armature=0.0216, range=(-1.5, 1.5)),
    "thigh": dict(damping=0.1, armature=0.0216, range=(-1.047, 1.047)),
    "rod": dict(damping=0.2, armature=0.001, range=None),
    "knee": dict(damping=0.2, armature=0.001, range=None),
}
# Delivered peak torque at the joint, N*m (AKE90-8 on hip roll, AK60-39 on cam and thigh).
TORQUE_PEAK = {"hip_roll": 61.2, "cam": 144.5, "thigh": 144.5}

# The rod's outboard hinge, in the rod's own frame.  The rod body is bit-identical to the previous
# plant, so its hinge holes are unchanged; the matching point on the new leg is DERIVED from the
# CAD's closed pose by _loop_anchors() rather than guessed.
ROD_TIP = {"L": np.array([0.000743214, 0.007, -0.397969]),
           "R": np.array([-0.000743214, -0.007, -0.397969])}

# Loop-closure solver settings: stiff enough that the constraint itself does not yield (the
# previous plant's soft connect let the base sink ~15 cm under load; the robot yields 5 mm).
LOOP_SOLREF = "0.002 1"
LOOP_SOLIMP = "0.999 0.9999 0.0001"

# Leg-axis series spring.  The robot's 5 mm sink under one body weight is structural compliance
# along the leg axis, not loop yield (a vertical stance load barely loads the four-bar -- the leg
# is near full reach and the load path is axial).  It is modelled explicitly as a prismatic joint
# at the sole along the leg axis.  MEASURED ON THE PREVIOUS LEG; the merged leg is a different
# structure and this wants re-measuring.
LEG_SPRING_K = 148.5 / 0.005          # N/m, foot-referred: 5 mm at one body weight on one leg
LEG_SPRING_B = 600.0                  # N s/m, zeta ~0.45 against the robot's mass in stance
LEG_SPRING_RANGE = 0.03               # m, hard stop either way

# Contact.  Two spheres per foot, at the two ends of the flat sole, positioned by
# _discover_sole() from the mesh rather than by hand.
SOLE_SPHERE_R = 0.012
FRICTION = "1 0.008 0.001"
CONTACT_SOLREF = "0.01 1"
CONTACT_SOLIMP = "0.95 0.99 0.001"

# Joint targets the gait is centred on (the homed-zero convention, see ZERO_SHIFT).
NOMINAL_CTRL = np.array([0.0, 0.0, 0.12, 0.0, 0.0, -0.12])

BASE_JOINTS = [("base_x", "slide", "1 0 0"), ("base_y", "slide", "0 1 0"),
               ("base_z", "slide", "0 0 1"), ("base_roll", "hinge", "1 0 0"),
               ("base_pitch", "hinge", "0 1 0"), ("base_yaw", "hinge", "0 0 1")]
PLANAR_REMOVE = ("base_y", "base_roll", "base_yaw")


# =============================================================================================
# small helpers
# =============================================================================================
def _fa(s):
    return np.array([float(x) for x in s.split()])


def _fs(a):
    return " ".join(f"{float(x):.10g}" for x in np.asarray(a).ravel())


def _quat(euler):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.asarray(euler, float), "xyz")
    return q


def _mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _axis_angle(axis, ang):
    q = np.zeros(4)
    a = np.asarray(axis, float)
    mujoco.mju_axisAngle2Quat(q, a / np.linalg.norm(a), float(ang))
    return q


def _mulq(a, b):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.asarray(a, float), np.asarray(b, float))
    return q


def _joint_key(name):
    """'thigh_L' -> 'thigh'."""
    return name.rsplit("_", 1)[0]


# =============================================================================================
# stage 1 -- read the CAD, prove the loops close, cut them
# =============================================================================================
def _cad_with_duplicates():
    """The raw export with repeated names suffixed, so it compiles and can be inspected."""
    root = ET.parse(CAD).getroot()
    root.find("compiler").set("meshdir", str(CAD_DIR.resolve()))
    seen = set()
    for b in root.iter("body"):
        for el in [b] + b.findall("joint") + b.findall("geom"):
            n = el.get("name")
            if n in seen:
                el.set("name", n + "__DUP")
            else:
                seen.add(n)
    return root


def check_loops_close(verbose=True):
    """Both copies of each duplicated body must land at the same world pose in the exported
    configuration.  If they do not, the export is not a consistent assembly and nothing below it
    is meaningful.  Returns the compiled duplicate-tree model+data, used to derive the anchors."""
    root = _cad_with_duplicates()
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), {})
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    worst = 0.0
    for name in [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]:
        if not name or not name.endswith("__DUP"):
            continue
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name[:-5])
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        dp = float(np.linalg.norm(d.xpos[k] - d.xpos[i]))
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (d.xmat[i].reshape(3, 3).T @ d.xmat[k].reshape(3, 3)).ravel())
        dang = float(np.degrees(2 * np.arccos(np.clip(abs(q[0]), -1, 1))))
        worst = max(worst, dp)
        if verbose:
            print(f"  [loop] {name[:-5]:22s} both branches agree to "
                  f"{dp*1e6:7.3f} um / {dang*3600:7.3f} arcsec")
        if dp > 1e-6 or dang > 1e-3:
            raise SystemExit(f"CAD loop does not close for {name[:-5]}: "
                             f"{dp*1e3:.4f} mm, {dang:.5f} deg")
    return m, d, worst


def _loop_anchors(m, d):
    """The rod's outboard hinge expressed in the LEG's frame, read off the CAD's closed pose."""
    out = {}
    for side in "LR":
        rod = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"Pushrod{'Left' if side=='L' else 'Right'}NCS-v1")
        leg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"FootFlat{'Left' if side=='L' else 'Right'}NCS-v1")
        w = d.xpos[rod] + d.xmat[rod].reshape(3, 3) @ ROD_TIP[side]
        out[side] = d.xmat[leg].reshape(3, 3).T @ (w - d.xpos[leg])
    return out


def _cut_and_rename():
    """The CAD tree with duplicates deleted and everything renamed.  Keeps the leg under the
    thigh and the rod under the cam on BOTH sides (the exporter cuts the two sides differently)."""
    root = ET.parse(CAD).getroot()
    parent_of = {c: p for p in root.iter() for c in p}

    # delete the duplicate occurrence of each repeated body, whichever branch it sits in
    keep_parent = {"FootFlatLeftNCS-v1": "ThighLeftNCS-v1",
                   "FootFlatRightNCS-v1": "ThighRightNCS-v1",
                   "PushrodLeftNCS-v1": "CamLeftNCS-v1",
                   "PushrodRightNCS-v1": "CamRightNCS-v1"}
    for b in list(root.iter("body")):
        n = b.get("name")
        if n in keep_parent and parent_of[b].get("name") != keep_parent[n]:
            parent_of[b].remove(b)

    names = [b.get("name") for b in root.iter("body")]
    if len(names) != len(set(names)):
        raise SystemExit(f"cut left duplicates: {sorted(names)}")

    for b in root.iter("body"):
        b.set("name", BODY[b.get("name")])
        for g in b.findall("geom"):
            g.set("name", b.get("name") + "_visual")
        for j in b.findall("joint"):
            suffix = j.get("name").rsplit("-", 1)[-1]
            if suffix not in JOINT_SUFFIX:
                raise SystemExit(f"unmapped joint suffix {suffix!r} ({j.get('name')!r})")
            j.set("name", JOINT_SUFFIX[suffix][0])
    return root


# =============================================================================================
# stage 2 -- joint-zero convention
# =============================================================================================
def _apply_zero_shift(root):
    """Rotate each actuated body's frame back onto the robot's homing convention.

    A body's own joint axis is fixed in that body's frame, so post-multiplying the body's
    orientation by a rotation about it moves ONLY that joint's zero: geometry, inertia, the
    child subtree's relative placement and every site position are untouched."""
    shifted = {}
    for b in root.iter("body"):
        js = b.findall("joint")
        if not js:
            continue
        jname = js[0].get("name")
        key = _joint_key(jname)
        theta = ZERO_SHIFT.get(key)
        if not theta:
            continue
        sign = -1.0 if jname.endswith("_R") else 1.0
        axis = _fa(js[0].get("axis"))
        axis = axis / np.linalg.norm(axis)
        q_body = _fa(b.get("quat")) if b.get("quat") else _quat(_fa(b.get("euler", "0 0 0")))
        # R_new = R_cad * Rot(axis, -theta*sign): carries the exported pose to the homed zero
        q_new = _mulq(q_body, _axis_angle(axis, -theta * sign))
        b.set("quat", _fs(q_new))
        b.attrib.pop("euler", None)
        shifted[jname] = theta * sign
    missing = set(ACTUATED) - set(shifted) - {k for k in ACTUATED if not ZERO_SHIFT[_joint_key(k)]}
    if missing:
        raise SystemExit(f"zero shift not applied to {sorted(missing)}")
    return shifted


def check_zero_convention(root, prev_plant, verbose=True):
    """Compare the re-posed frames against the previous plant, which defines the convention the
    motor controllers and robot/deploy already use.  Shared bodies must now agree in orientation."""
    if not Path(prev_plant).exists():
        if verbose:
            print(f"  [zero] previous plant {prev_plant} not present -- skipping cross-check")
        return None
    old = {}

    def rec(el):
        for b in el.findall("body"):
            q = _fa(b.get("quat")) if b.get("quat") else _quat(_fa(b.get("euler", "0 0 0")))
            old[b.get("name")] = (_fa(b.get("pos", "0 0 0")), _mat(q))
            rec(b)
    rec(ET.parse(prev_plant).getroot().find("worldbody"))

    inv = {v: k for k, v in BODY.items()}
    worst = 0.0
    for b in root.iter("body"):
        n = b.get("name")
        src = inv.get(n)
        # the leg is a new part, and the rod's zero is free (passive joint) -- neither compares
        if src is None or src not in old or n in ("leg_L", "leg_R", "rod_L", "rod_R"):
            continue
        q = _fa(b.get("quat")) if b.get("quat") else _quat(_fa(b.get("euler", "0 0 0")))
        dR = old[src][1].T @ _mat(q)
        qd = np.zeros(4)
        mujoco.mju_mat2Quat(qd, dR.ravel())
        ang = float(np.degrees(2 * np.arccos(np.clip(abs(qd[0]), -1, 1))))
        dpos = float(np.linalg.norm(_fa(b.get("pos", "0 0 0")) - old[src][0]))
        budget = ZERO_TOLERATED.get(n, 0.01)
        worst = max(worst, ang)
        if verbose:
            tag = "intended" if n in ZERO_TOLERATED and ang > 0.01 else ""
            print(f"  [zero] {n:10s} vs previous plant: {ang:8.4f} deg "
                  f"(budget {budget:.2f}), {dpos*1e3:7.4f} mm {tag}")
        if dpos > 1e-6:
            raise SystemExit(f"{n}: frame ORIGIN moved {dpos*1e3:.4f} mm -- geometry changed")
        if ang > budget:
            raise SystemExit(f"{n}: joint zero off by {ang:.4f} deg (budget {budget:.2f}) "
                             f"-- check ZERO_SHIFT")
    return worst


# =============================================================================================
# stage 3 -- masses, motors, dynamics, actuators, sensors, loop
# =============================================================================================
def _apply_masses(root, leg_kg):
    targets = {}
    for group, (link, motors, total) in MEASURED_KG.items():
        if link in ("leg_L", "leg_R"):
            targets[link] = leg_kg
            continue
        want = total - sum(MOTOR_KG[mm] for mm in motors)
        if want <= 0:
            raise SystemExit(f"{group}: weighed assembly {total} kg is lighter than its motors")
        targets[link] = want
    report = []
    for b in root.iter("body"):
        n = b.get("name")
        if n not in targets:
            continue
        inert = b.find("inertial")
        have = float(inert.get("mass"))
        want = targets.pop(n)
        s = want / have
        inert.set("mass", f"{want:.9g}")
        inert.set("fullinertia", _fs(_fa(inert.get("fullinertia")) * s))
        report.append((n, have, want))
    if targets:
        raise SystemExit(f"no body for {sorted(targets)}")
    return report


def _add_motor_masses(root, m_cad, d_cad):
    """Motors as welded point masses at their own joint anchors."""
    by_name = {b.get("name"): b for b in root.iter("body")}
    inv = {v: k for k, v in BODY.items()}
    for motor, (host, jname) in MOTOR_MOUNT.items():
        cad_host = inv[host]
        hb = mujoco.mj_name2id(m_cad, mujoco.mjtObj.mjOBJ_BODY, cad_host)
        jsuffix = [k for k, v in JOINT_SUFFIX.items() if v[0] == jname][0]
        jid = next(j for j in range(m_cad.njnt)
                   if mujoco.mj_id2name(m_cad, mujoco.mjtObj.mjOBJ_JOINT, j).rsplit("-", 1)[-1]
                   == jsuffix)
        local = d_cad.xmat[hb].reshape(3, 3).T @ (d_cad.xanchor[jid] - d_cad.xpos[hb])
        b = ET.SubElement(by_name[host], "body")
        b.set("name", motor)
        b.set("pos", _fs(local))
        i = ET.SubElement(b, "inertial")
        i.set("pos", "0 0 0")
        i.set("mass", f"{MOTOR_KG[motor]:.6g}")
        i.set("diaginertia", "0.002 0.002 0.002" if MOTOR_KG[motor] > 1.0 else "0.001 0.001 0.001")


def _apply_joint_dynamics(root):
    for b in root.iter("body"):
        for j in b.findall("joint"):
            dyn = JOINT_DYN[_joint_key(j.get("name"))]
            j.set("damping", f"{dyn['damping']:g}")
            j.set("armature", f"{dyn['armature']:g}")
            if dyn["range"] is None:
                j.set("limited", "false")
            else:
                j.set("limited", "true")
                j.set("range", f"{dyn['range'][0]:g} {dyn['range'][1]:g}")
            # the exporter writes axes like (8.4e-16, 1, -1.4e-15); snap to exact unit axes
            a = _fa(j.get("axis"))
            a = a / np.linalg.norm(a)
            a[np.abs(a) < 1e-9] = 0.0
            j.set("axis", _fs(a))
            j.set("pos", _fs(np.where(np.abs(_fa(j.get("pos", "0 0 0"))) < 1e-9, 0.0,
                                      _fa(j.get("pos", "0 0 0")))))


def _add_leg_spring(root, side, k):
    """A prismatic joint at the leg's distal end along the leg axis: the structural compliance
    that produces the measured 5 mm stance sink.  Placed in the leg body, so it sits in series
    between the four-bar and the ground."""
    leg = next(b for b in root.iter("body") if b.get("name") == f"leg_{side}")
    j = ET.Element("joint")
    j.set("name", f"leg_spring_{side}")
    j.set("type", "slide")
    j.set("axis", "-1 0 0")          # the leg runs along -x in its own frame; sole at -x
    j.set("pos", "0 0 0")
    j.set("limited", "true")
    j.set("range", f"{-LEG_SPRING_RANGE:g} {LEG_SPRING_RANGE:g}")
    j.set("stiffness", f"{k:.6g}")
    j.set("damping", f"{LEG_SPRING_B:g}")
    j.set("springref", "0")
    j.set("armature", "0.02")
    leg.insert(0, j)


def _add_sites_and_loop(root, anchors):
    eq = ET.SubElement(root, "equality")
    for side in "LR":
        rod = next(b for b in root.iter("body") if b.get("name") == f"rod_{side}")
        s = ET.SubElement(rod, "site")
        s.set("name", f"rod_tip_{side}")
        s.set("pos", _fs(ROD_TIP[side]))
        leg = next(b for b in root.iter("body") if b.get("name") == f"leg_{side}")
        s = ET.SubElement(leg, "site")
        s.set("name", f"leg_anchor_{side}")
        s.set("pos", _fs(anchors[side]))
        c = ET.SubElement(eq, "connect")
        c.set("name", f"loop_{side}")
        c.set("site1", f"rod_tip_{side}")
        c.set("site2", f"leg_anchor_{side}")
        c.set("solref", LOOP_SOLREF)
        c.set("solimp", LOOP_SOLIMP)


def _add_actuators_and_sensors(root):
    act = ET.SubElement(root, "actuator")
    for name in ACTUATED:
        peak = TORQUE_PEAK[_joint_key(name)]
        a = ET.SubElement(act, "motor")
        a.set("name", name)
        a.set("joint", name)
        a.set("gear", "1")
        a.set("forcerange", f"{-peak:g} {peak:g}")
        a.set("ctrlrange", f"{-peak:g} {peak:g}")
    sen = ET.SubElement(root, "sensor")
    for tag, attr, val in (("accelerometer", "site", "imu"), ("gyro", "site", "imu")):
        s = ET.SubElement(sen, tag)
        s.set("name", f"imu_{'acc' if tag == 'accelerometer' else 'gyro'}")
        s.set(attr, val)
    s = ET.SubElement(sen, "framequat")
    s.set("name", "base_quat")
    s.set("objtype", "site")
    s.set("objname", "imu")
    for name in ACTUATED:
        for tag, suf in (("jointpos", "pos"), ("jointvel", "vel")):
            s = ET.SubElement(sen, tag)
            s.set("name", f"{name}_{suf}")
            s.set("joint", name)
        s = ET.SubElement(sen, "actuatorfrc")
        s.set("name", f"{name}_frc")
        s.set("actuator", name)


def _scaffold(root):
    """Compiler options, visual/collision defaults, the ground plane and the base DOFs."""
    root.set("model", "dash01")
    comp = root.find("compiler")
    comp.set("angle", "radian")
    comp.set("autolimits", "true")
    opt = ET.Element("option")
    for k, v in (("timestep", "0.001"), ("integrator", "implicitfast"), ("cone", "elliptic"),
                 ("impratio", "10"), ("solver", "Newton"), ("iterations", "100"),
                 ("ls_iterations", "50")):
        opt.set(k, v)
    root.insert(1, opt)

    dflt = ET.Element("default")
    d0 = ET.SubElement(dflt, "default")
    d0.set("class", "dash01")
    g = ET.SubElement(d0, "geom")
    g.set("contype", "0")
    g.set("conaffinity", "0")
    g.set("group", "2")
    st = ET.SubElement(d0, "site")
    st.set("group", "4")
    st.set("size", "0.012")
    st.set("rgba", "0.95 0.45 0.1 1")
    d1 = ET.SubElement(d0, "default")
    d1.set("class", "collision")
    g = ET.SubElement(d1, "geom")
    for k, v in (("contype", "1"), ("conaffinity", "1"), ("group", "3"), ("condim", "6"),
                 ("friction", FRICTION), ("solref", CONTACT_SOLREF), ("solimp", CONTACT_SOLIMP),
                 ("rgba", "0.2 0.8 0.2 0.4")):
        g.set(k, v)
    root.insert(2, dflt)

    asset = root.find("asset")
    t = ET.SubElement(asset, "texture")
    for k, v in (("name", "grid"), ("type", "2d"), ("builtin", "checker"),
                 ("rgb1", "0.2 0.3 0.4"), ("rgb2", "0.1 0.15 0.2"),
                 ("width", "300"), ("height", "300")):
        t.set(k, v)
    mt = ET.SubElement(asset, "material")
    for k, v in (("name", "grid"), ("texture", "grid"), ("texrepeat", "6 6"),
                 ("reflectance", "0.1")):
        mt.set(k, v)

    wb = root.find("worldbody")
    for el in list(wb):
        if el.tag in ("light", "geom"):
            wb.remove(el)                      # the export's own light + 1 m plane
    lt = ET.Element("light")
    for k, v in (("name", "top"), ("pos", "0 0 3"), ("dir", "0 0 -1"), ("directional", "true")):
        lt.set(k, v)
    wb.insert(0, lt)
    fl = ET.Element("geom")
    for k, v in (("name", "floor"), ("type", "plane"), ("size", "0 0 0.05"),
                 ("material", "grid"), ("contype", "1"), ("conaffinity", "1"), ("condim", "6"),
                 ("friction", FRICTION), ("solref", CONTACT_SOLREF), ("solimp", CONTACT_SOLIMP)):
        fl.set(k, v)
    wb.insert(1, fl)

    torso = next(b for b in wb.findall("body") if b.get("name") == "torso")
    torso.set("childclass", "dash01")
    for i, (name, typ, axis) in enumerate(BASE_JOINTS):
        j = ET.Element("joint")
        for k, v in (("name", name), ("type", typ), ("axis", axis),
                     ("limited", "false"), ("damping", "0"), ("armature", "0")):
            j.set(k, v)
        torso.insert(i, j)
    imu = ET.Element("site")
    for k, v in (("name", "imu"), ("pos", "0 0 0"), ("size", "0.015"),
                 ("rgba", "0.1 0.5 0.95 1")):
        imu.set(k, v)
    torso.insert(len(BASE_JOINTS), imu)

    cust = ET.SubElement(root, "custom")
    n = ET.SubElement(cust, "numeric")
    n.set("name", "nominal_ctrl")
    n.set("data", _fs(NOMINAL_CTRL))


# =============================================================================================
# stage 4 -- the sole, discovered from the mesh
# =============================================================================================
def discover_sole(root, verbose=True):
    """Where is the flat sole, in the leg's own frame?

    Compiled without any collision geometry and held at the nominal joint targets, the lowest
    mesh vertices of the leg ARE the sole.  Fit a plane to them, then place one sphere at each end
    of the contact line.  Doing it this way means the contact model cannot silently disagree with
    the CAD -- if the part changes, the spheres move with it."""
    import trimesh
    probe = copy.deepcopy(root)
    m = mujoco.MjModel.from_xml_string(ET.tostring(probe, encoding="unicode"), {})
    d = mujoco.MjData(m)
    for i, name in enumerate(ACTUATED):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qpos[m.jnt_qposadr[j]] = NOMINAL_CTRL[i]
    mujoco.mj_forward(m, d)

    out = {}
    for side in "LR":
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"leg_{side}")
        mesh = f"FootFlat{'Left' if side == 'L' else 'Right'}NCS-v1"
        tm = trimesh.load(CAD_DIR / "meshes" / f"{mesh}.stl")
        v = np.asarray(tm.vertices) * 0.001
        R, p = d.xmat[bid].reshape(3, 3), d.xpos[bid]
        w = v @ R.T + p                                        # mesh vertices in world
        zmin = w[:, 2].min()
        sole = v[w[:, 2] < zmin + 0.004]                       # the 4 mm closest to the ground
        # principal in-plane direction of that patch, in the leg frame
        c = sole.mean(0)
        u, s, vt = np.linalg.svd(sole - c)
        long_axis = vt[0]
        t = (sole - c) @ long_axis
        a, b = c + long_axis * t.min(), c + long_axis * t.max()
        normal = vt[2]                                         # plane normal, leg frame
        if (R @ normal)[2] > 0:
            normal = -normal                                   # point it at the ground
        out[side] = dict(toe=a, heel=b, normal=normal, length=float(t.max() - t.min()),
                         width=float(np.ptp((sole - c) @ vt[1])),
                         flatness=float(np.abs((sole - c) @ vt[2]).max()), n=len(sole),
                         world_z=float(zmin))
        if verbose:
            o = out[side]
            print(f"  [sole] {side}: {o['n']:4d} verts, patch {o['length']*1000:6.1f} x "
                  f"{o['width']*1000:5.1f} mm, flat to {o['flatness']*1000:.3f} mm, "
                  f"normal (leg frame) {np.round(o['normal'], 4)}")
    # the toe is the end further from the leg's attachment (the origin)
    for side, o in out.items():
        if np.linalg.norm(o["toe"]) < np.linalg.norm(o["heel"]):
            o["toe"], o["heel"] = o["heel"], o["toe"]
    return out


def _add_sole_geoms(root, sole):
    for side in "LR":
        leg = next(b for b in root.iter("body") if b.get("name") == f"leg_{side}")
        o = sole[side]
        for tag, pt in (("foot", o["toe"]), ("heel", o["heel"])):
            # sink the sphere centre so the sphere is TANGENT to the sole plane
            c = np.asarray(pt) - o["normal"] * SOLE_SPHERE_R
            g = ET.SubElement(leg, "geom")
            g.set("name", f"{tag}_{side}_col")
            g.set("class", "collision")
            g.set("type", "sphere")
            g.set("size", f"{SOLE_SPHERE_R:g}")
            g.set("pos", _fs(c))


# =============================================================================================
# stage 5 -- settle the keyframe
# =============================================================================================
def _held(m, planar):
    names = (["base_x", "base_pitch"] if planar else
             ["base_x", "base_y", "base_roll", "base_pitch", "base_yaw"])
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    return [int(m.jnt_qposadr[j]) for j in ids], [int(m.jnt_dofadr[j]) for j in ids]


def _pd(m, d, target, kp=(120, 200, 200, 120, 200, 200), kd=(4, 5, 5, 4, 5, 5)):
    tau = np.zeros(m.nu)
    for a in range(m.nu):
        jid = m.actuator_trnid[a, 0]
        tau[a] = (kp[a] * (target[a] - d.qpos[m.jnt_qposadr[jid]])
                  - kd[a] * d.qvel[m.jnt_dofadr[jid]])
    lim = m.actuator_forcerange[:, 1]
    return np.clip(tau, -lim, lim)


def settle(m, z0=1.05, t_s=3.0, planar=True):
    """Gravity-settle onto the floor with the base held except z, motors holding NOMINAL_CTRL.

    Held, not free: on a free base the solver sees six loose DOFs inside every step and the legs
    walk sideways over a 3 s settle.  The settled LEG posture is what we want; the base pose is
    imposed."""
    d = mujoco.MjData(m)
    zadr = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_z")])
    d.qpos[zadr] = z0
    for i, name in enumerate(ACTUATED):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qpos[m.jnt_qposadr[j]] = NOMINAL_CTRL[i]
    held, heldv = _held(m, planar)
    hold = d.qpos[held].copy()
    for _ in range(int(t_s / m.opt.timestep)):
        d.ctrl[:] = _pd(m, d, NOMINAL_CTRL)
        mujoco.mj_step(m, d)
        d.qpos[held] = hold
        d.qvel[heldv] = 0.0
    if not np.all(np.isfinite(d.qpos)):
        raise RuntimeError("settle diverged")
    mujoco.mj_forward(m, d)
    return d.qpos.copy(), dict(z=float(d.qpos[zadr]),
                               stand_torque=[float(x) for x in _pd(m, d, NOMINAL_CTRL)],
                               max_qvel=float(np.abs(d.qvel).max()),
                               loop_err=float(np.abs(d.efc_pos[:0]).max()) if False else 0.0)


# =============================================================================================
# build
# =============================================================================================
def build(variant="free", leg_kg=LEG_KG, leg_spring_k=LEG_SPRING_K, verbose=True):
    if verbose:
        print(f"[build] {variant}")
    m_cad, d_cad, _ = check_loops_close(verbose)
    anchors = _loop_anchors(m_cad, d_cad)
    if verbose:
        for s, a in anchors.items():
            print(f"  [loop] anchor {s} in leg frame {np.round(a, 6)}")

    root = _cut_and_rename()
    root.find("compiler").set("meshdir", str(CAD_DIR.resolve()))
    _apply_zero_shift(root)
    check_zero_convention(root, PKG.parent / "walk_v4" / "model" / "dash01_base.xml", verbose)
    mass_report = _apply_masses(root, leg_kg)
    _apply_joint_dynamics(root)
    _scaffold(root)
    _add_motor_masses(root, m_cad, d_cad)
    _add_sites_and_loop(root, anchors)
    _add_actuators_and_sensors(root)
    sole = discover_sole(root, verbose)
    _add_sole_geoms(root, sole)
    for side in "LR":
        _add_leg_spring(root, side, leg_spring_k)

    if variant == "planar":
        torso = next(b for b in root.iter("body") if b.get("name") == "torso")
        for j in list(torso.findall("joint")):
            if j.get("name") in PLANAR_REMOVE:
                torso.remove(j)

    # settle on the PLANAR tree always -- see settle() -- then map the leg posture onto this one
    settle_root = root
    if variant != "planar":
        settle_root = copy.deepcopy(root)
        torso = next(b for b in settle_root.iter("body") if b.get("name") == "torso")
        for j in list(torso.findall("joint")):
            if j.get("name") in PLANAR_REMOVE:
                torso.remove(j)
    m_s = mujoco.MjModel.from_xml_string(ET.tostring(settle_root, encoding="unicode"), {})
    q_s, info = settle(m_s)

    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), {})
    qpos = np.zeros(m.nq)
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        k = mujoco.mj_name2id(m_s, mujoco.mjtObj.mjOBJ_JOINT, name)
        if k >= 0:
            qpos[m.jnt_qposadr[j]] = q_s[m_s.jnt_qposadr[k]]
    kf = ET.SubElement(root, "keyframe")
    key = ET.SubElement(kf, "key")
    key.set("name", "stand")
    key.set("qpos", _fs(qpos))
    key.set("ctrl", _fs(np.zeros(m.nu)))

    root.find("compiler").set("meshdir", "../../Dash-01CAD")
    xml = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")
    return xml, dict(info, mass=mass_report, sole=sole, anchors=anchors, model=m)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leg-kg", type=float, default=LEG_KG,
                    help=f"mass of the merged leg/foot segment (default {LEG_KG}, PROVISIONAL)")
    ap.add_argument("--leg-spring-k", type=float, default=LEG_SPRING_K)
    ap.add_argument("--report", action="store_true", help="build and validate, write nothing")
    args = ap.parse_args()
    for variant in ("free", "planar"):
        xml, info = build(variant, args.leg_kg, args.leg_spring_k)
        m = info["model"]
        print(f"  [plant] nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} "
              f"mass={m.body_subtreemass[1]:.4f} kg")
        print(f"  [plant] settled stand height {info['z']:.4f} m, "
              f"stand torque {np.round(info['stand_torque'], 2).tolist()}")
        if not args.report:
            out = HERE / f"dash01_{variant}.xml"
            out.write_text(xml, encoding="utf-8")
            print(f"  [plant] wrote {out.name}")
        print()


if __name__ == "__main__":
    main()
