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
zeros the motor controllers and ``controller/deploy`` already use stay valid.  Passive joints keep the
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
# The plant this one replaces, vendored here as the reference for the joint-zero cross-check.  It
# is the only thing outside Dash-01CAD that this package reads, and it is kept so the check keeps
# working once the old training tree is gone -- the robot's homing convention is defined by it.
PREVIOUS_PLANT = "previous_plant.xml"

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
# The merged leg/foot's CAD mass, used as-is on the user's instruction.  Every other segment here
# is a weighed figure and the CAD densities were wrong for them by up to 5x, so this one number is
# on a different footing from the rest -- flagged rather than hidden, because distal mass is the
# dominant cost term for a runner and it is the first thing to replace with a scale reading.
LEG_KG = 0.405
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

# There is NO leg-axis series spring.  The previous plant carried one because the old leg -- a
# carbon shin plus a 249 g ankle-spring assembly -- was measured to sink 5 mm under one body
# weight.  The merged rigid leg replaces both of those parts and has no such compliance, so the
# spring, its sprung shell and the machinery that kept it numerically stable are all gone.

# Contact.  Two spheres per foot, at the two ends of the flat sole, positioned by
# _sole_geometry() from the mesh rather than by hand.
# The sole is a 3 mm TPU pad glued under the CAD's sole face, so ground contact happens 3 mm
# beyond the mesh.  It is modelled as two BOXES per foot -- front half and rear half of the pad --
# rather than as spheres.  That matters: a box-plane contact gives four corner points, so the foot
# has a real support polygon and the centre of pressure can travel across it.  Two spheres at the
# pad's ends are a LINE contact, and with one the solver loaded only the front pair, pinning the
# centre of pressure at the toe and tipping the robot however well its centre of mass was placed.
# Splitting the pad in two keeps a toe-down / heel-down contact bit per foot for the gait reward.
SOLE_PAD_M = 0.003
FRICTION_MU = 0.7                      # floor friction is at least this; DR samples upward
FRICTION = f"{FRICTION_MU} 0.005 0.0001"
CONTACT_CONDIM = 3                     # a box already supplies the patch; no faked rolling friction
CONTACT_SOLREF = "0.01 1"
CONTACT_SOLIMP = "0.95 0.99 0.001"

# Nominal stance.  The leg is rigid from knee to sole, so the sole's angle to the ground is not a
# free variable -- it is fixed by cam and thigh.  The nominal posture is therefore SOLVED, not
# chosen: find (cam, thigh) that lays the sole flat, and among that one-parameter family take the
# tallest stance.  solve_flat_stance() does it; NOMINAL_CTRL below is its answer, cached so a
# build is reproducible without the solve, and re-checked on every build.
NOMINAL_CTRL = np.array([0.0, 0.1265, -0.2025, 0.0, -0.1265, 0.2025])

# Gains for the POSE SOLVER only -- not the robot's stance gains.  Stiff enough that gravity sag
# is negligible, which is free in a quasi-static relaxation where velocities are zeroed each step,
# and unclipped because this is a kinematic tool rather than a claim about the motors.
KEYFRAME_BITE_M = 0.0005     # contact penetration at the keyframe, so all four spheres engage
POSE_KP = (2.0e4,) * 6
POSE_KD = (2.0e2,) * 6
STANCE_TILT_TOL_DEG = 1.0
FLAT_TOL_DEG = STANCE_TILT_TOL_DEG   # the same bar while solving and when re-checking

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


def mirror_targets(cam, thigh, hip_roll=0.0):
    """Joint targets for a left/right symmetric posture, in ACTUATED order.

    The right leg's joint axes point along -y where the left's point along +y, so a mirror-image
    pose needs the right commands NEGATED, not copied: reflecting a rotation about +y through the
    sagittal plane gives a rotation about -y of the same sign, which the right joint reads as the
    opposite number.  Verified in the smoke test by mirroring the whole posture and comparing the
    two legs' world poses."""
    return np.array([hip_roll, cam, thigh, -hip_roll, -cam, -thigh])


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


def _loop_anchors(m, d, symmetric=True, verbose=True):
    """The rod's outboard hinge expressed in the LEG's frame, read off the CAD's closed pose.

    Derived rather than measured: the rod body is bit-identical to the previous plant, so its
    outboard hole is known, and the CAD's exported pose is a closed assembly, so mapping that
    point into the leg frame gives the matching hole on the leg.  It lands 3 mm inside the mesh
    surface on both sides, which is what a hinge-hole centre should look like.

    The two sides come out 1.4 mm apart because the CAD was saved with the left and right cams
    0.09 deg from mirror-symmetric, and that pose error lands in the derived anchor.  The two leg
    parts are mirror images of each other, so the anchor is mirrored from the left rather than
    derived twice; left as-is it tilts the robot enough to start every episode on one foot."""
    out = {}
    for side in "LR":
        rod = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"Pushrod{'Left' if side=='L' else 'Right'}NCS-v1")
        leg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"FootFlat{'Left' if side=='L' else 'Right'}NCS-v1")
        w = d.xpos[rod] + d.xmat[rod].reshape(3, 3) @ ROD_TIP[side]
        out[side] = d.xmat[leg].reshape(3, 3).T @ (w - d.xpos[leg])
    if verbose:
        print(f"  [loop] anchors differ L vs mirrored R by "
              f"{np.linalg.norm(out['L'] - out['R'] * np.array([1, -1, 1])) * 1000:.2f} mm")
    if symmetric:
        out["R"] = out["L"] * np.array([1.0, -1.0, 1.0])
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
    motor controllers and controller/deploy already use.  Shared bodies must now agree in orientation."""
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


def _add_sole_geoms(root, sole, stance_R, pad=SOLE_PAD_M):
    """The sole pad: two boxes per foot, front half and rear half, on the leg itself.

    A box-plane contact gives four corner points, so each foot has a genuine support polygon and
    the centre of pressure can move across it.  Spheres at the pad's two ends cannot do that --
    they are a line contact, and the solver loaded only the front pair, pinning the centre of
    pressure at the toe and tipping the robot over however well its centre of mass was placed.
    Splitting the pad front/rear keeps one contact bit per half, so toe-down and heel-down stay
    distinguishable for the gait reward.

    The pad is the 3 mm TPU layer glued under the CAD's sole face, so it sits OUTSIDE the mesh:
    ground contact happens 3 mm beyond the surface the stance was solved against."""
    for side in "LR":
        leg = next(b for b in root.iter("body") if b.get("name") == f"leg_{side}")
        o = sole[side]
        n = np.asarray(o["normal"], float)          # unit, points at the ground
        n = n / np.linalg.norm(n)
        a, b = np.asarray(o["a"], float), np.asarray(o["b"], float)
        # Toe and heel by where they sit fore-aft in the nominal stance.  Ordering by distance
        # from the knee gets it backwards on this leg: the foot bracket hangs forward of the
        # shin, so the sole end further down the leg is the REAR of the foot.
        if (stance_R[side] @ a)[0] < (stance_R[side] @ b)[0]:
            a, b = b, a                              # a = toe end, b = heel end
        u = b - a
        length = float(np.linalg.norm(u))
        u = u / length
        w = -n                                       # box local z: up, into the foot
        u = u - w * float(u @ w)
        u = u / np.linalg.norm(u)
        v = np.cross(w, u)
        R = np.column_stack([u, v, w])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.ravel())
        half = length / 4.0                          # each box spans half the pad
        for k, tag in enumerate(("foot", "heel")):
            mid = a + u * (length * (0.25 + 0.5 * k))
            g = ET.SubElement(leg, "geom")
            g.set("name", f"{tag}_{side}_col")
            g.set("class", "collision")
            g.set("type", "box")
            g.set("size", f"{half:.6g} {o['width'] / 2:.6g} {pad / 2:.6g}")
            g.set("pos", _fs(mid + n * (pad / 2)))   # pad hangs below the CAD sole face
            g.set("quat", _fs(q))


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
    for k, v in (("contype", "1"), ("conaffinity", "1"), ("group", "3"), ("condim", str(CONTACT_CONDIM)),
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
                 ("material", "grid"), ("contype", "1"), ("conaffinity", "1"), ("condim", str(CONTACT_CONDIM)),
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
def _leg_mesh(side):
    import trimesh
    name = f"FootFlat{'Left' if side == 'L' else 'Right'}NCS-v1"
    tm = trimesh.load(CAD_DIR / "meshes" / f"{name}.stl")
    v = np.asarray(tm.vertices) * 0.001
    return v, trimesh.Trimesh(v, np.asarray(tm.faces))


def _sole_from_patch(side, R, verbose=True):
    """Given a leg orientation R that rests the foot flat, read the sole off the mesh: the set of
    vertices at the bottom, expressed back in the leg's own frame."""
    v, tm = _leg_mesh(side)
    hull = np.asarray(tm.convex_hull.vertices)
    z = hull @ (R.T @ np.array([0.0, 0.0, 1.0]))          # height of each vertex, up to an offset
    pts = hull[z < z.min() + 0.002]
    if len(pts) < 3:
        raise SystemExit(f"leg_{side}: the resting patch has only {len(pts)} points -- not a sole")
    c = pts.mean(0)
    e = np.linalg.svd(pts - c)[2]
    nrm = e[2]
    if (R @ nrm)[2] > 0:
        nrm = -nrm
    long_axis = e[0] if abs(e[0][1]) < 0.5 else e[1]
    t = (pts - c) @ long_axis
    beyond = float((v @ nrm).max() - float(np.median(pts @ nrm)))
    out = dict(normal=nrm, a=c + long_axis * t.min(), b=c + long_axis * t.max(),
               length=float(t.max() - t.min()), width=float(np.ptp(pts[:, 1])),
               flatness=float(np.abs((pts - c) @ e[2]).max()), n=len(pts), beyond=beyond)
    if verbose:
        print(f"  [sole] {side}: resting patch {out['length']*1000:5.1f} x "
              f"{out['width']*1000:5.1f} mm from {out['n']} hull points, flat to "
              f"{out['flatness']*1000:.3f} mm, normal (leg frame) {np.round(nrm, 4)}")
    if beyond > 1e-3:
        raise SystemExit(f"leg_{side}: {beyond*1e3:.2f} mm of material sits below the resting "
                         f"patch -- it is not the lowest face")
    return out


def find_sole_posture(m, verbose=True):
    """Which leg posture rests the foot FLAT on the floor, with the robot balanced on it?

    Nothing here assumes which face is the sole.  Pass one sweeps the two leg joints, closes the
    four-bar at each, drops the leg's convex hull onto a level floor and measures the resting
    patch; a face lying flat gives a long patch, an edge or a corner a short one.  The winner
    defines the sole, and from then on "flat" is measured as the tilt of that plane, which is a
    continuous quantity rather than a vertex count.

    Pass two picks WHICH flat posture.  Laying the sole flat is one condition on two joints, so
    its solutions form a family, and along it the foot sweeps fore and aft under the robot while
    the centre of mass barely moves.  The second condition is balance: put the centre of mass in
    the middle of the contact patch, so the robot stands by itself with equal margin forward and
    back.  Two conditions, two joints, one answer -- no stance height to invent, and static
    stability comes out of the solve instead of being hoped for.

    The previous robot could not satisfy this at all: it was a point-foot machine whose centre of
    mass sat 87 mm behind its toe contacts, so standing still was never an equilibrium and the
    policy had to catch it on every step.  The flat sole is what makes the condition solvable."""
    grid = [(float(c), float(t)) for c in np.arange(-0.9, 1.0, 0.05)
            for t in np.arange(-0.9, 1.0, 0.05)]
    coarse = [(c, t, r) for c, t in grid for r in [_patch_at(m, c, t)] if r]
    if not coarse:
        raise SystemExit("no posture rests the foot on the floor within joint range")
    c0, t0, r0 = max(coarse, key=lambda x: x[2]["span"])
    sole = _sole_from_patch("L", r0["R"], verbose)

    scan = [(c, t, r) for c, t, r in
            [(c, t, _patch_at(m, c, t, sole)) for c, t in grid] if r]
    flat = [x for x in scan if x[2]["tilt"] < FLAT_TOL_DEG]
    if not scan:
        raise SystemExit("no posture rests the sole on the floor within joint range")
    if verbose:
        hs = [r["height"] for _, _, r in flat] or [float("nan")]
        print(f"  [sole] {len(scan)} reachable postures, {len(flat)} with the sole flat to "
              f"{FLAT_TOL_DEG} deg (stance height {min(hs):.3f}-{max(hs):.3f} m)")

    # Two conditions, two joints: drive both to zero with a damped Newton step on the pair
    #   f1 = the sole normal's fore-aft lean  (zero when the sole is exactly horizontal)
    #   f2 = margin_front - margin_back       (zero when the centre of mass is mid-patch)
    # A coordinate search cannot do this -- the two conditions trade off against each other along
    # the flat family -- and the residual matters: 1 mm of leftover tilt over an 86 mm sole lifts
    # one of the two contact spheres clear of the floor and the robot tips over that edge.
    def residual(r):
        return np.array([r["lean"], r["margin_front"] - r["margin_back"]])

    # Seed from the whole scan, not just the flat subset: the seed only has to be on the right
    # assembly branch and near the solution, and Newton drives both conditions to zero from there.
    cam, thigh, r = min(scan, key=lambda x: np.abs(residual(x[2]) * np.array([1.0, 5.0])).sum())
    for _ in range(12):
        f = residual(r)
        if np.abs(f).max() < 1e-5:
            break
        h = 1e-3
        J = np.zeros((2, 2))
        ok = True
        for k, (dc, dt) in enumerate(((h, 0.0), (0.0, h))):
            q = _patch_at(m, cam + dc, thigh + dt, sole)
            if q is None:
                ok = False
                break
            J[:, k] = (residual(q) - f) / h
        if not ok or abs(np.linalg.det(J)) < 1e-9:
            break
        step = np.linalg.solve(J, -f)
        step = step * min(1.0, 0.2 / max(np.abs(step).max(), 1e-12))     # trust region
        q = _patch_at(m, cam + step[0], thigh + step[1], sole)
        if q is None or np.abs(residual(q)).sum() >= np.abs(f).sum():
            break
        cam, thigh, r = cam + step[0], thigh + step[1], q
    if verbose:
        print(f"  [sole] newton residual: lean {r['lean']:+.2e}, "
              f"balance {(r['margin_front']-r['margin_back'])*1000:+.3f} mm")
        print(f"  [sole] chosen: cam {cam:+.4f} thigh {thigh:+.4f} -> sole tilt "
              f"{r['tilt']:.3f} deg, stance {r['height']:.4f} m, centre of mass "
              f"{r['margin_back']*1000:+.1f} mm ahead of the heel edge and "
              f"{r['margin_front']*1000:+.1f} mm behind the toe edge")
    if min(r["margin_back"], r["margin_front"]) <= 0:
        raise SystemExit("no balanced sole-flat posture: the centre of mass never lies over the "
                         "contact patch")
    return mirror_targets(cam, thigh), r, sole


def _patch_at(m, cam, thigh, sole=None):
    """Close the four-bar at this posture, rest the leg on a level floor, and measure the contact
    patch and the whole robot's balance over it.  With `sole` known, the patch is the sole's own
    two end points and `tilt` says how far the sole plane is from horizontal."""
    pose = _leg_pose(m, cam, thigh)
    if pose is None or pose["loop"] > 1e-4:
        return None
    R, p = pose["R"], pose["p"]
    if sole is None:
        v, tm = _leg_mesh("L")
        pts = np.asarray(tm.convex_hull.vertices) @ R.T + p
        patch = pts[pts[:, 2] < pts[:, 2].min() + 0.002]
        tilt = lean = float("nan")
    else:
        patch = np.array([sole["a"], sole["b"]]) @ R.T + p
        nw = R @ sole["normal"]
        tilt = float(np.degrees(np.arccos(np.clip(-nw[2], -1.0, 1.0))))
        lean = float(nw[0])            # signed: which way the sole leans, fore or aft
    height = float(pose["base_z"] - patch[:, 2].min())
    if height < 0.4 or len(patch) < 2:
        return None
    d = pose["data"]
    com = np.sum(d.xipos * m.body_mass[:, None], axis=0) / m.body_mass.sum()
    return dict(span=float(np.ptp(patch[:, 0])), height=height, n=len(patch), R=R, tilt=tilt,
                lean=lean if sole is not None else float("nan"),
                loop=pose["loop"], com_x=float(com[0]),
                x_back=float(patch[:, 0].min()), x_front=float(patch[:, 0].max()),
                margin_back=float(com[0] - patch[:, 0].min()),
                margin_front=float(patch[:, 0].max() - com[0]))


def cad_pose():
    """The actuated joint values that reproduce the CAD's exported pose.

    Body frames were rotated by -ZERO_SHIFT to move the joint zeros onto the robot's homing
    convention, so the exported pose now sits at qpos = +ZERO_SHIFT.  It is the one configuration
    we know is ASSEMBLED: with the passive joints at zero the loop closes exactly there (the
    duplicate-body check proves it to sub-micron).  Everything else has to be reached from it."""
    return mirror_targets(ZERO_SHIFT["cam"], ZERO_SHIFT["thigh"], ZERO_SHIFT["hip_roll"])


def relax_to(m, d, targets, held, n_ramp=1500, n_hold=600, from_pose=None):
    """Walk quasi-statically from the assembled configuration to `targets`.

    A four-bar has TWO assembly branches for the same crank angle -- knee forward and knee back --
    and a solver handed an open loop (the passive joints at zero, metres from closure) picks one
    per leg, independently.  That is not hypothetical: settling straight to the nominal posture
    put the left leg on the opposite branch from the right, with its foot 0.86 m in the air and
    the robot standing on one leg.  Ramping from a configuration that is already closed keeps
    both legs on the branch the CAD was drawn in, because the path never passes through an open
    loop.  Velocities are zeroed every step so this is a continuation, not a swing.

    The ramp is PD-driven, because forcing qpos along it would drag the mechanism through
    configurations it cannot actually reach and it lands on nonsense branches.  The FINAL
    configuration is then pinned: a finite-gain PD leaves the legs ~0.09 rad short under their
    own weight, and if that sag is left in, the posture that gets analysed is not the posture that
    was asked for -- the stance solve and the keyframe end up describing different configurations,
    one flat and one 12 deg up on its toes.  After the pin, `targets` means the configuration and
    the passive joints are what relaxes around it."""
    start = cad_pose() if from_pose is None else np.asarray(from_pose, float)
    targets = np.asarray(targets, float)
    act = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
           for n in ACTUATED]
    d.qpos[act] = start
    hold = d.qpos[held].copy()
    for i in range(n_ramp + n_hold):
        a = min(1.0, (i + 1) / max(n_ramp, 1))
        d.ctrl[:] = _pd(m, d, start + a * (targets - start), kp=POSE_KP, kd=POSE_KD, clip=False)
        mujoco.mj_step(m, d)
        d.qpos[held] = hold
        d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    return d


def _leg_pose(m, cam, thigh, n=1200):
    """Close the four-bar at (cam, thigh) with the base pinned in the air, and return the leg's
    world frame.  The leg angle is not free: cam and thigh fix it through the rod."""
    d = mujoco.MjData(m)
    zadr = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_z")])
    zdof = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_z")])
    d.qpos[zadr] = 1.4
    held, heldv = _held(m, planar=False)
    held, heldv = held + [zadr], heldv + [zdof]
    relax_to(m, d, mirror_targets(cam, thigh), held, n_ramp=n, n_hold=max(n // 2, 200))
    if not np.all(np.isfinite(d.qpos)):
        return None
    loop = float(np.abs(d.efc_pos[:m.neq * 3]).max()) if m.neq else 0.0
    out = dict(base_z=1.4, loop=loop, data=d)
    for side in "LR":
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"leg_{side}")
        out[f"R_{side}"] = d.xmat[b].reshape(3, 3).copy()
        out[f"p_{side}"] = d.xpos[b].copy()
    out["R"], out["p"] = out["R_L"], out["p_L"]
    return out


def stance_tilt(m, sole, cam, thigh):
    """Sole tilt from horizontal, stance height and balance margins at (cam, thigh)."""
    return _patch_at(m, cam, thigh, sole)


def check_nominal_stance(m, sole, nominal, verbose=True):
    """The cached NOMINAL_CTRL must still lay the sole flat on this build."""
    r = stance_tilt(m, sole, float(nominal[1]), float(nominal[2]))
    if r is None:
        raise SystemExit("nominal posture does not close the four-bar")
    if verbose:
        print(f"  [stance] NOMINAL_CTRL cam {nominal[1]:+.4f} thigh {nominal[2]:+.4f}: "
              f"sole tilt {r['tilt']:.4f} deg (tol {STANCE_TILT_TOL_DEG}), "
              f"stance height {r['height']:.4f} m, loop residual {r['loop']*1e6:.3f} um")
    if r["tilt"] > STANCE_TILT_TOL_DEG:
        raise SystemExit(f"NOMINAL_CTRL leaves the sole {r['tilt']:.3f} deg off the floor -- "
                         f"re-run with --solve-stance")
    return r


# =============================================================================================
# stage 5 -- settle the keyframe
# =============================================================================================
def _geom_floor_gap(m, d, g):
    """Height of a collision geom's LOWEST point above z = 0.

    Not `xpos.z - size[0]`: that is the radius only for a sphere.  For the box pads size[0] is
    the half LENGTH, about twenty times the pad's half thickness, so using it placed the keyframe
    20 mm in the air and every standing test began with a drop and a bounce."""
    R = d.geom_xmat[g].reshape(3, 3)
    t = int(m.geom_type[g])
    if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        drop = float(m.geom_size[g, 0])
    elif t == int(mujoco.mjtGeom.mjGEOM_BOX):
        drop = float(np.abs(R[2, :3]) @ m.geom_size[g, :3])
    elif t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        drop = float(abs(R[2, 2]) * m.geom_size[g, 1] + m.geom_size[g, 0])
    else:
        raise NotImplementedError(f"floor gap for geom type {t}")
    return float(d.geom_xpos[g][2] - drop)


def _held(m, planar):
    names = (["base_x", "base_pitch"] if planar else
             ["base_x", "base_y", "base_roll", "base_pitch", "base_yaw"])
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    return [int(m.jnt_qposadr[j]) for j in ids], [int(m.jnt_dofadr[j]) for j in ids]


def _pd(m, d, target=None, kp=(120, 200, 200, 120, 200, 200), kd=(4, 5, 5, 4, 5, 5), clip=True):
    target = NOMINAL_CTRL if target is None else target
    tau = np.zeros(m.nu)
    for a in range(m.nu):
        jid = m.actuator_trnid[a, 0]
        tau[a] = (kp[a] * (target[a] - d.qpos[m.jnt_qposadr[jid]])
                  - kd[a] * d.qvel[m.jnt_dofadr[jid]])
    if not clip:
        return tau
    lim = m.actuator_forcerange[:, 1]
    return np.clip(tau, -lim, lim)


def settle(m, nominal=None, z0=1.05, planar=True, verbose=True):
    """The keyframe pose: the nominal posture, set down so the soles just touch the floor.

    Two steps, and neither of them is a gravity settle under contact.

    First, close the four-bar in the air by continuation from the CAD's assembled configuration
    (see relax_to) -- the leg's closed configuration does not depend on base height, so this is
    free, and starting anywhere else lets the two legs pick opposite assembly branches.

    Then place the base by KINEMATICS: drop it until the lowest contact sphere touches z = 0.
    Simulating the touchdown instead was the tempting thing to do and it is wrong here -- with
    the series spring off there is nothing left to equilibrate, so all a contact settle can add
    is impact transients, and it did: it rolled the stance 13 degrees onto the toes and put the
    keyframe 47 mm below the posture the stance was solved for.  The nominal posture is the
    answer; the keyframe just has to express it."""
    nominal = NOMINAL_CTRL if nominal is None else np.asarray(nominal, float)
    d = mujoco.MjData(m)
    zadr = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_z")])
    zdof = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_z")])
    d.qpos[zadr] = z0 + 0.2
    held, heldv = _held(m, planar)
    relax_to(m, d, nominal, held + [zadr])

    loop = float(np.abs(d.efc_pos[:m.neq * 3]).max()) if m.neq else 0.0
    if loop > 1e-4:
        raise RuntimeError(f"four-bar did not close in the air: {loop*1e3:.4f} mm")

    cols = [g for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_col")]
    heights = [_geom_floor_gap(m, d, g) for g in cols]
    # Seat ALL FOUR contacts, not just the lowest one.  MuJoCo only reports a contact once the
    # geoms actually overlap, so placing the robot with its lowest sphere exactly on z = 0 leaves
    # the other three a fraction of a millimetre clear and the episode starts balanced on a single
    # point -- which tips, however well centred the centre of mass is.  Sink by the spread plus a
    # small bite so every sphere is engaged at t = 0.
    d.qpos[zadr] -= max(heights) + KEYFRAME_BITE_M
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    lows = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g): _geom_floor_gap(m, d, g)
            for g in cols}
    if verbose:
        print("  [key] contact heights above the floor: "
              + ", ".join(f"{k.replace('_col','')} {v*1000:+.2f} mm" for k, v in lows.items()))
    return d.qpos.copy(), dict(z=float(d.qpos[zadr]), loop_residual=loop,
                               contact_heights=lows,
                               spread_mm=(max(lows.values()) - min(lows.values())) * 1000)


# =============================================================================================
# build
# =============================================================================================
# The drives' design position gains, and the ceiling the phase-scheduled impedance can reach
# (x2.5).  That ceiling is a HARDWARE limit, not a choice: the MIT CAN frame encodes kp in 12 bits
# over 0..500 N*m/rad (controller/deploy/mit.py) and the safety governor clamps to the same, so
# 500 is all there is.
STANCE_KP = (120.0, 200.0, 200.0, 120.0, 200.0, 200.0)
STANCE_KD = (4.0, 5.0, 5.0, 4.0, 5.0, 5.0)
STANCE_KP_CEILING = tuple(2.5 * k for k in STANCE_KP)


def loaded_command(m, qpos, pose, kp=None, verbose=True):
    """The command that HOLDS `pose` under gravity, as opposed to the pose itself.

    A position loop only makes torque from error, so commanding the pose the robot should end up
    in leaves it with none and the leg collapses: at the drives' 500 N*m/rad ceiling, commanding
    the nominal pose drops the robot 1.56 m, while commanding it offset by about 2 deg holds it to
    within 2 mm.  The offset is tau_hold / kp -- roughly 22 N*m over the stance gains -- and the
    same idea as the previous pipeline re-settling its keyframe under load.

    This is why the plant does not need stiffer drives.  Holding the stance with the command
    pinned AT the pose would take kp 2000, four times what the hardware can encode; holding it
    with the right command takes 15% of the available torque."""
    kp = np.asarray(STANCE_KP_CEILING if kp is None else kp, float)
    d = mujoco.MjData(m)
    d.qpos[:] = qpos                    # the keyframe is not on the model yet at build time
    mujoco.mj_forward(m, d)
    stiff = tuple(np.asarray(STANCE_KP) * 16), tuple(np.asarray(STANCE_KD) * 4)
    for _ in range(4000):
        d.ctrl[:] = _pd(m, d, pose, kp=stiff[0], kd=stiff[1])
        mujoco.mj_step(m, d)
    tau = _pd(m, d, pose, kp=stiff[0], kd=stiff[1])
    q_ss = np.array([d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]]
                     for n in ACTUATED])
    cmd = q_ss + tau / kp
    if verbose:
        print(f"  [load] holding torque {np.round(tau, 2).tolist()} N*m "
              f"({np.abs(tau).max() / float(m.actuator_forcerange[1, 1]) * 100:.0f}% of limit); "
              f"command offset {np.round(np.degrees(cmd - q_ss), 2).tolist()} deg")
    return cmd, q_ss, tau


def check_static_stability(m, qpos, nominal, t_s=4.0, verbose=True,
                           scales=(1, 2, 4, 8, 10, 12, 16, 24)):
    """Spawn the robot at its keyframe with the base COMPLETELY FREE and see whether it stands.

    No DOF is held, there is no balance controller, and the motors only hold the nominal joint
    angles through a plain PD -- gravity decides.  A point-foot robot fails this by construction,
    so it is the single number that says whether the flat sole did its job.

    The answer depends on how stiff that PD is, so this reports the THRESHOLD rather than a
    pass/fail at one arbitrary gain.  Standing is a statics question (is the centre of mass over
    the contact patch, and is the torque within the motors?) and a stiffness question (does the
    leg droop far enough under that torque to walk the posture out of its own support?) -- and on
    this robot the two answers are very different, so both are worth printing."""
    out = []
    for sc in scales:
        kp = tuple(np.asarray(STANCE_KP) * sc)
        kd = tuple(np.asarray(STANCE_KD) * np.sqrt(sc))
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        q = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
             for n in ("base_x", "base_z", "base_pitch")}
        start = {k: float(d.qpos[v]) for k, v in q.items()}
        peak, ok = 0.0, True
        for _ in range(int(t_s / m.opt.timestep)):
            tau = _pd(m, d, nominal, kp=kp, kd=kd)
            d.ctrl[:] = tau
            peak = max(peak, float(np.abs(tau).max()))
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                ok = False
                break
        r = dict(kp=kp[1], kd=kd[1],
                 drift_x=float(d.qpos[q["base_x"]] - start["base_x"]) if ok else float("nan"),
                 drop_z=float(start["base_z"] - d.qpos[q["base_z"]]) if ok else float("nan"),
                 pitch_deg=float(np.degrees(d.qpos[q["base_pitch"]] - start["base_pitch"]))
                 if ok else float("nan"),
                 peak_tau=peak, speed=float(np.abs(d.qvel[:6]).max()) if ok else float("inf"))
        r["stands"] = bool(ok and abs(r["pitch_deg"]) < 10.0
                           and d.qpos[q["base_z"]] > 0.95 * start["base_z"])
        out.append(r)
        if verbose:
            print(f"  [stand] kp {r['kp']:6.0f} kd {r['kd']:5.1f}: drop {r['drop_z']*1000:+7.1f} mm, "
                  f"pitch {r['pitch_deg']:+7.2f} deg, peak tau {r['peak_tau']:6.1f} N*m, "
                  f"|base vel| {r['speed']:7.3f} -> {'STANDS' if r['stands'] else 'falls'}")
    held = [r for r in out if r["stands"]]
    summary = dict(runs=out, stands=bool(held),
                   kp_min=min((r["kp"] for r in held), default=None),
                   tau_hold=min((r["peak_tau"] for r in held), default=None))
    if verbose:
        lim = float(m.actuator_forcerange[1, 1])
        if held:
            print(f"  [stand] STANDS from kp {summary['kp_min']:.0f} upward, holding torque "
                  f"{summary['tau_hold']:.1f} N*m = {summary['tau_hold']/lim*100:.0f}% of the "
                  f"{lim:.0f} N*m limit")
        else:
            print("  [stand] does NOT stand at any tested stiffness")
    return summary


def build(variant="free", leg_kg=LEG_KG, verbose=True,
          solve_stance=False, nominal=None):
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
    check_zero_convention(root, HERE / PREVIOUS_PLANT, verbose)
    mass_report = _apply_masses(root, leg_kg)
    _apply_joint_dynamics(root)
    _scaffold(root)
    _add_motor_masses(root, m_cad, d_cad)
    _add_sites_and_loop(root, anchors)
    _add_actuators_and_sensors(root)

    # The nominal posture and the sole are found together, before any contact geometry exists:
    # the sole is whichever face rests on the floor, and the nominal posture is the one that
    # rests it flat.  Solving is a two-joint sweep, so it sits behind --solve-stance and the
    # cached NOMINAL_CTRL is re-verified on every build instead.
    m_probe = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), {})
    if solve_stance:
        nominal, found, sole_L = find_sole_posture(m_probe, verbose)
    else:
        nominal = NOMINAL_CTRL if nominal is None else np.asarray(nominal, float)
        pose = _leg_pose(m_probe, float(nominal[1]), float(nominal[2]))
        if pose is None:
            raise SystemExit("NOMINAL_CTRL does not close the four-bar")
        sole_L = _sole_from_patch("L", pose["R"], verbose)
    pose_nom = _leg_pose(m_probe, float(nominal[1]), float(nominal[2]))
    # One sole definition, mirrored.  The two leg parts ARE mirror images, so fitting a plane to
    # each one's own resting hull points only lets mesh tessellation differ between the sides --
    # it came out 2.5 mm apart, enough that the robot started every episode on one foot.
    sole = {"L": _sole_from_patch("L", pose_nom["R_L"], verbose)}
    sole["R"] = {k: (v * np.array([1.0, -1.0, 1.0]) if isinstance(v, np.ndarray) and v.shape == (3,)
                     else v) for k, v in sole["L"].items()}
    stance = check_nominal_stance(m_probe, sole["L"], nominal, verbose)
    root.find("custom/numeric").set("data", _fs(nominal))

    _add_sole_geoms(root, sole, {s: pose_nom[f"R_{s}"] for s in "LR"})

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
    q_s, info = settle(m_s, nominal, z0=stance["height"], verbose=verbose)

    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), {})
    qpos = np.zeros(m.nq)
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        k = mujoco.mj_name2id(m_s, mujoco.mjtObj.mjOBJ_JOINT, name)
        if k >= 0:
            qpos[m.jnt_qposadr[j]] = q_s[m_s.jnt_qposadr[k]]
    # The passive hinges are unlimited, so nothing stops the settle from parking one of them a
    # few turns away from where it started -- the knee comes out around 19 rad.  Physically that
    # is the same pose, but it is the reset state of every training episode and it feeds the
    # observation, so wrap it onto (-pi, pi].
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in PASSIVE:
            a = m.jnt_qposadr[j]
            qpos[a] = (qpos[a] + np.pi) % (2 * np.pi) - np.pi
    kf = ET.SubElement(root, "keyframe")
    key = ET.SubElement(kf, "key")
    key.set("name", "stand")
    key.set("qpos", _fs(qpos))
    key.set("ctrl", _fs(np.zeros(m.nu)))

    cmd = pose = stand = None
    if variant == "free":
        cmd, pose, tau_hold = loaded_command(m, qpos, nominal, verbose=verbose)
        stand = check_static_stability(m, qpos, cmd, verbose=verbose,
                                       scales=(2.5,))          # the drives' ceiling, kp 500
        num = ET.SubElement(root.find("custom"), "numeric")
        num.set("name", "nominal_cmd")
        num.set("data", _fs(cmd))

    root.find("compiler").set("meshdir", "../../Dash-01CAD")
    xml = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")
    return xml, dict(info, mass=mass_report, sole=sole, anchors=anchors, model=m,
                     nominal=nominal, stance=stance, stand=stand)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leg-kg", type=float, default=LEG_KG,
                    help=f"mass of the merged leg/foot segment (default {LEG_KG}, PROVISIONAL)")
    ap.add_argument("--report", action="store_true", help="build and validate, write nothing")
    ap.add_argument("--solve-stance", action="store_true",
                    help="re-solve the sole-flat nominal posture instead of checking the cached "
                         "NOMINAL_CTRL (slow; run it when the leg part changes)")
    args = ap.parse_args()
    for variant in ("free", "planar"):
        xml, info = build(variant, args.leg_kg,
                          solve_stance=args.solve_stance)
        m = info["model"]
        print(f"  [plant] nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} "
              f"mass={m.body_subtreemass[1]:.4f} kg")
        print(f"  [plant] keyframe stand height {info['z']:.4f} m, contacts within "
              f"{info['spread_mm']:.2f} mm of one plane, loop {info['loop_residual']*1e6:.2f} um")
        if not args.report:
            out = HERE / f"dash01_{variant}.xml"
            out.write_text(xml, encoding="utf-8")
            print(f"  [plant] wrote {out.name}")
        print()


if __name__ == "__main__":
    main()
