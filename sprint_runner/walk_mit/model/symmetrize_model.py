"""Make dash01.xml's right leg an exact mirror of the left, and re-derive the standing keyframe.

*** NOT APPLIED. Run this only deliberately -- it was tried on 2026-08-10 and REVERTED. ***

It does what it says (worst mirror error 1.211 deg -> 2.1e-05, and it fixes a genuinely stale pair
of ankle locks that sat 0.20 deg apart on a mirror pair). But it was written to fix a crooked
standing stance, and it does not:

    env re-settled keyframe, hip_roll sum (0 = symmetric)
        before   L -7.322  R -0.812   sum -8.134
        after    L +0.268  R +7.427   sum +7.695

Same magnitude, opposite side -- a bifurcation. The model asymmetry only chose WHICH WAY the robot
leans. The real cause is that _resettle_keyframe settles into a ~7 deg lean from a symmetric start,
in BOTH ankle modes, while the raw XML keyframe is symmetric to 0.006 deg. That is the thing to fix.

It also costs a regression nobody has explained: the m3 pitch reflex goes from arresting a +1.2 kick
by 57% (0.183 vs 0.429) to 19% (0.258 vs 0.318), which FAILS smoke_test -- and smoke_test gates
every cluster job, so applying this aborts the queue. The -1.2 direction is unaffected (58%). Joint
axes were checked and are already mirrored to 2.8e-07, so they are not the cause.

Kept because the analysis is worth more than the diff, and because if the re-settle bifurcation is
ever fixed, this is the other half of the job.

build_model.py now does this at generation time (mirror_right_leg), but it regenerates from the CAD
export at robotCADdescription/, which is not in this repo -- so the shipped dash01.xml cannot be
rebuilt here. This applies the same transform to the model in place and then recomputes the
keyframe, which is the part that actually matters: compute_standing_keyframe SETTLES a symmetric
initial pose, so the model's asymmetry is what the standing stance inherits.

Measured on the model before this ran: the right hip body frame carried a 1.211 deg rotation the
left did not, the right foot 0.133 deg, the left foot sat 1.6 mm off the mirror line, and several
inertial CoMs differed by ~0.5 mm -- 17 of 30 compared quantities. Through the near-singular 4-bar
that became 6.5 deg of hip-roll difference between the legs in the standing keyframe, which every
episode resets to.

    python training/model/symmetrize_model.py [--dry-run]

Writes dash01.xml in place (a .bak copy is kept). Joint AXES are not touched -- see mirror_right_leg.
"""
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np
import mujoco

from build_model import compute_standing_keyframe, INIT_CTRL, f3

MIRROR_POS = (1.0, -1.0, 1.0)             # reflect across the sagittal plane y = 0
MIRROR_ROT = (-1.0, 1.0, -1.0)            # euler xyz / rotation vector
MIRROR_QUAT = (1.0, -1.0, 1.0, -1.0)      # (w, x, y, z)
MIRROR_INERTIA = (1.0, 1.0, 1.0, -1.0, 1.0, -1.0)   # ixx iyy izz ixy ixz iyz


def _mirror_from(src, dst, sign, attr):
    """dst.attr := sign * src.attr, only if BOTH already carry the attribute."""
    if src is None or dst is None:
        return
    v = src.get(attr)
    if v is None or dst.get(attr) is None:
        return
    vals = [float(x) for x in v.split()]
    if len(vals) != len(sign):
        return
    dst.set(attr, f3([a * b for a, b in zip(vals, sign)]))


def mirror_right_leg(root):
    """Force the RIGHT leg geometry to be an exact mirror of the LEFT.

    Left is authoritative because its frames are the clean, all-zero ones. JOINT AXES ARE
    NOT TOUCHED: they carry the sim<->real sign convention the webui model_map mirrors by
    hand, and they were measured to be already mirrored to 2.8e-07 anyway. Joint POSITIONS
    are geometry and are mirrored."""
    bodies = {b.get('name'): b for b in root.iter('body') if b.get('name')}
    n = 0
    for name, L in list(bodies.items()):
        if 'Left' in name:
            rname = name.replace('Left', 'Right')
        elif name.endswith('_L'):
            rname = name[:-2] + '_R'
        else:
            continue
        R = bodies.get(rname)
        if R is None:
            continue
        for attr, sign in (('pos', MIRROR_POS), ('euler', MIRROR_ROT), ('quat', MIRROR_QUAT)):
            _mirror_from(L, R, sign, attr)
        iL, iR = L.find('inertial'), R.find('inertial')
        if iL is not None and iR is not None:
            _mirror_from(iL, iR, MIRROR_POS, 'pos')
            _mirror_from(iL, iR, MIRROR_QUAT, 'quat')
            _mirror_from(iL, iR, MIRROR_INERTIA, 'fullinertia')
            if iL.get('mass') is not None and iR.get('mass') is not None:
                iR.set('mass', iL.get('mass'))
        for tag in ('geom', 'site'):
            for a, b in zip(L.findall(tag), R.findall(tag)):
                for attr, sign in (('pos', MIRROR_POS), ('euler', MIRROR_ROT),
                                   ('quat', MIRROR_QUAT)):
                    _mirror_from(a, b, sign, attr)
        for a, b in zip(L.findall('joint'), R.findall('joint')):
            _mirror_from(a, b, MIRROR_POS, 'pos')   # position only -- NOT axis, NOT range
        n += 1
    print(f'[symmetrize] mirrored {n} left/right body pairs (geometry only)')

XML = HERE / "dash01.xml"
DRY = "--dry-run" in sys.argv


def joint_angles(model, qpos, label):
    """The six actuated joint angles, L block against R block."""
    out = {}
    for i in range(model.nu):
        j = model.actuator_trnid[i, 0]
        out[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)] = qpos[model.jnt_qposadr[j]]
    names = list(out)
    print(f"  {label}")
    for k in range(3):
        l, r = out[names[k]], out[names[k + 3]]
        print(f"    {names[k]:<12} L {np.degrees(l):+8.3f}   R {np.degrees(r):+8.3f}   "
              f"sum {np.degrees(l + r):+7.3f}   diff {np.degrees(l - r):+7.3f}")
    return out


print(f"reading {XML}")
model_before = mujoco.MjModel.from_xml_path(str(XML))
kid = mujoco.mj_name2id(model_before, mujoco.mjtObj.mjOBJ_KEY, "stand")
print("\nBEFORE:")
joint_angles(model_before, model_before.key_qpos[kid], "standing keyframe")
print(f"    base z {model_before.key_qpos[kid][2]:.5f} m")

tree = ET.parse(XML)
root = tree.getroot()
mirror_right_leg(root)

tmp = XML.with_suffix(".sym.tmp.xml")
ET.indent(tree, space="  ")
tree.write(tmp, encoding="unicode", xml_declaration=False)

# re-derive the standing keyframe on the now-symmetric tree
model = mujoco.MjModel.from_xml_path(str(tmp))
qpos = compute_standing_keyframe(model, INIT_CTRL)
key = root.find("keyframe").find("key")
key.set("qpos", f3(qpos))
key.set("ctrl", f3(INIT_CTRL))

# The rigid-ankle locks are pinned at the SETTLED stance angle (make_ankle_variants), so a new
# keyframe makes the old polycoef stale: the weld then fights the stance instead of holding it, and
# the ankle drifts off the target. Measured before this was added: lock_ankle_L -0.201849 against
# lock_ankle_R +0.205416 -- 0.20 deg apart on a pair that must be exact mirrors -- and the rigid arm
# drifted 0.0053 rad off its own lock. Re-pin both from the freshly settled keyframe.
eq = root.find("equality")
for side, jname in (("L", "LegLeftNCS-v1"), ("R", "LegRightNCS-v1")):
    lock = next((e for e in eq if e.get("name") == f"lock_ankle_{side}"), None)
    if lock is None:
        continue
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, lock.get("joint1"))
    q = float(qpos[model.jnt_qposadr[jid]])
    old = lock.get("polycoef").split()[0]
    lock.set("polycoef", f"{q:.9g} 0 0 0 0")
    print(f"  lock_ankle_{side}: {old} -> {q:.9g} rad")
ET.indent(tree, space="  ")
tree.write(tmp, encoding="unicode", xml_declaration=False)

model_after = mujoco.MjModel.from_xml_path(str(tmp))
kid2 = mujoco.mj_name2id(model_after, mujoco.mjtObj.mjOBJ_KEY, "stand")
print("\nAFTER:")
joint_angles(model_after, model_after.key_qpos[kid2], "standing keyframe (re-settled)")
print(f"    base z {model_after.key_qpos[kid2][2]:.5f} m")
print(f"    total mass {model_after.body_mass.sum():.4f} kg "
      f"(was {model_before.body_mass.sum():.4f})")

if DRY:
    tmp.unlink()
    print("\n--dry-run: nothing written")
else:
    shutil.copy2(XML, XML.with_suffix(".xml.bak"))
    shutil.move(str(tmp), str(XML))
    print(f"\nwrote {XML}  (previous kept as dash01.xml.bak)")
