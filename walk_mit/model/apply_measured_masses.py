"""Patch the CAD-placeholder segment masses with the USER'S MEASURED weights (2026-08-03).

Until now every mass in dash01.xml was a CAD placeholder (see [[spiderbot-hardware]]: "Mass/inertia
are partial placeholders") plus 6 datasheet motor point-masses welded on by build_model. The user
weighed the real segments, and the difference is NOT cosmetic: total 12.83 -> 15.14 kg (+18%), and
the distal mass distribution changes a lot (the shin is 2.6x heavier than modelled). Since distal
leg mass is the dominant cost term for a runner (see [[hardware-speed-ceiling]]: ~9x torso mass),
the ankle-spring study has to run on the corrected plant or it answers a question about a robot
that does not exist.

Patches dash01.xml IN PLACE (2026-08-04 decision). The first version of this script wrote a separate
dash01_measured.xml so the m1..m7 / slow_gait / sym_gait lineages could keep their plant and the
ankle study could A/B the mass correction. That framing was wrong: there is one robot and it weighs
15.14 kg, so the mass is a plant FACT, not an experimental variable, and a second plant file only
creates ways to train on a robot that does not exist. The CAD masses are recoverable from git
history if a comparison is ever wanted.

    python -m model.apply_measured_masses            # patches model/dash01.xml

Idempotent: re-running on an already-patched file computes a scale of 1.0 and changes nothing, which
is what makes it safe to call from build_model at the end of a CAD regen (a regen rewrites dash01.xml
from CAD and would otherwise silently restore the placeholder masses).

MEASURED INPUT (grams, from the user):
    base 5764 (incl. BOTH abduction motors + battery) | hip 3271 (incl. its two motors)
    cam 66 | thigh 483 | pushrod 71 | ankle spring 249 | shin 324 | foot 222

Two modelling decisions, both flagged in the printout:
  1. The spring is not a body in the CAD tree, so its 249 g is folded into the SHIN (the user's
     instruction: they are parallel and very close together).
  2. We have MASSES ONLY -- no measured CoM offsets and no inertia tensors. Each body keeps its CAD
     CoM and gets its inertia tensor scaled by the mass ratio (i.e. same shape, uniform density
     change). That is exact for a pure density error and approximate for anything else; it is the
     honest fallback until inertias are measured. The base (+2.4 kg, mostly BATTERY) is the weakest
     case -- a battery is a compact block, not a scaled-up shell.
"""
import os
import re
import sys

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "dash01.xml")
DST = SRC                                          # in place: one plant file, see the module docstring

# Measured GROUP masses in kg. A group = one CAD link body + the motor point-masses welded onto it
# by build_model.add_motor_masses (the user weighed assemblies, the model splits motors out).
#   group name -> (link body, [welded motor bodies], measured total kg)
GROUPS = {
    "base":    ("bodyNCS-v1",      ["motor_hip_roll_L", "motor_hip_roll_R"], 5.764),
    "hip_L":   ("HipLeftNCS-v1",   ["motor_cam_L", "motor_thigh_L"],         3.271),
    "hip_R":   ("HipRightNCS-v1",  ["motor_cam_R", "motor_thigh_R"],         3.271),
    "cam_L":   ("CamLeftNCS-v1",      [], 0.066),
    "cam_R":   ("CamRightNCS-v1",     [], 0.066),
    "push_L":  ("PushrodLeftNCS-v1",  [], 0.071),
    "push_R":  ("PushrodRightNCS-v1", [], 0.071),
    "thigh_L": ("ThighLeftNCS-v1",    [], 0.483),
    "thigh_R": ("ThighRightNCS-v1",   [], 0.483),
    # shin carries the ankle spring (249 g): parallel to the shin and very close to it (user).
    "shin_L":  ("LegLeftNCS-v1",      [], 0.324 + 0.249),
    "shin_R":  ("LegRightNCS-v1",     [], 0.324 + 0.249),
    "foot_L":  ("FootLeftNCS-v1",     [], 0.222),
    "foot_R":  ("FootRightNCS-v1",    [], 0.222),
}

_BODY = re.compile(r'<body\s+name="([^"]+)"')
_MASS = re.compile(r'(<inertial\b[^>]*?\bmass=")([^"]+)(")')
_FULLI = re.compile(r'(\bfullinertia=")([^"]+)(")')
_DIAGI = re.compile(r'(\bdiaginertia=")([^"]+)(")')


def _link_targets(src):
    """measured group total - welded motor masses = the target mass for the CAD link body."""
    model = mujoco.MjModel.from_xml_path(src)

    def bmass(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise SystemExit(f"body {name!r} not in {src}")
        return float(model.body_mass[i])

    targets, report = {}, []
    for g, (link, motors, total) in GROUPS.items():
        motor_kg = sum(bmass(m) for m in motors)
        want = total - motor_kg
        have = bmass(link)
        targets[link] = want
        report.append((g, link, have, want, motor_kg, total))
    return targets, report


def patch(src=SRC, dst=DST):
    targets, report = _link_targets(src)

    print(f"{'group':8s} {'link body':22s} {'CAD g':>9s} {'->':^4s} {'meas g':>9s} "
          f"{'motors g':>9s} {'group g':>9s}")
    bad = []
    for g, link, have, want, motor_kg, total in report:
        print(f"{g:8s} {link:22s} {have*1000:9.1f} {'->':^4s} {want*1000:9.1f} "
              f"{motor_kg*1000:9.1f} {total*1000:9.1f}")
        if want <= 0:
            bad.append((g, want))
    if bad:
        raise SystemExit(
            "NEGATIVE link mass for " + ", ".join(g for g, _ in bad) + ".\n"
            "The measured group total is less than the datasheet motor masses inside it -- either\n"
            "the group total excludes those motors or the motor masses in build_model.MOTORS are\n"
            "wrong. Resolve before training on this model.")

    with open(src, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out, cur, n = [], None, 0
    for line in lines:
        m = _BODY.search(line)
        if m:
            cur = m.group(1)
        if cur in targets and "<inertial" in line:
            want = targets[cur]
            have = float(_MASS.search(line).group(2))
            s = want / have                      # uniform-density scale: I ~ m for a fixed shape
            line = _MASS.sub(lambda mm: f"{mm.group(1)}{want:.9g}{mm.group(3)}", line, count=1)
            for rx in (_FULLI, _DIAGI):
                line = rx.sub(
                    lambda mm: mm.group(1)
                    + " ".join(f"{float(v)*s:.9g}" for v in mm.group(2).split())
                    + mm.group(3), line, count=1)
            del targets[cur]                     # a body's FIRST inertial only
            n += 1
        out.append(line)

    if targets:
        raise SystemExit(f"no <inertial> found for: {sorted(targets)}")

    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.writelines(out)
    print(f"\npatched {n} inertials -> {dst}")
    return dst


def restand(path):
    """The keyframe is a LOADED stance settled under gravity, so +18% mass invalidates it: the PD
    joints sag further and the ankle springs compress more. Re-settle and rewrite <key>."""
    sys.path.insert(0, HERE)
    from build_model import compute_standing_keyframe

    model = mujoco.MjModel.from_xml_path(path)
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    ctrl = [float(v) for v in re.search(r'<key\b[^>]*\bctrl="([^"]+)"', txt).group(1).split()]
    old = [float(v) for v in re.search(r'<key\b[^>]*\bqpos="([^"]+)"', txt).group(1).split()]

    qpos = compute_standing_keyframe(model, ctrl)
    txt = re.sub(r'(<key\b[^>]*\bqpos=")([^"]+)(")',
                 lambda m: m.group(1) + " ".join(f"{v:.6g}" for v in qpos) + m.group(3),
                 txt, count=1)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(txt)

    # ankle qpos: L is index 11, R index 17 in the composite-base qpos layout (see env.py).
    print(f"\nloaded stance height {old[2]:.5f} -> {qpos[2]:.5f} m  "
          f"(sag {(old[2]-qpos[2])*1000:+.1f} mm)")
    print(f"ankle L {old[11]:+.4f} -> {qpos[11]:+.4f} rad, "
          f"R {old[17]:+.4f} -> {qpos[17]:+.4f} rad  (springref -+0.7)")
    m2 = mujoco.MjModel.from_xml_path(path)
    print(f"total mass {model.body_subtreemass[1]:.3f} kg -> "
          f"{m2.body_subtreemass[1]:.3f} kg")
    return qpos


if __name__ == "__main__":
    restand(patch())
