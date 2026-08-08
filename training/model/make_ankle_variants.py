"""Generate the PLANT variants for the ankle-spring study from dash01.xml.

The study asks three things about the passive foot spring: is it useful at all, must it be actuated,
and is there an optimal stiffness. Two of those need a different PLANT, not just a different number:

  dash01.xml           (patched in place) -- gains `lock_ankle_L/R` equality constraints, INACTIVE
                       by default. Activating them welds the ankle => the RIGID null arm. Done with
                       equalities rather than by deleting the joint on purpose: the qpos/qvel
                       layout, the obs width and every hard-coded ankle index stay identical, so
                       rigid/free/passive arms are the same network shape and can even share a warm
                       start. Inactive equalities change no dynamics, which is why they can live in
                       the one shared plant rather than in a study-only copy.
  dash01_active.xml    -- adds a real ankle ACTUATOR per side (position servo, motor point mass
                       welded at the ankle joint) => the ACTIVE arms. This one does change the
                       action/obs width (nu 6 -> 8), so active arms are their own lineage and never
                       warm-start from a passive checkpoint.

The `free` (k=0, floppy ankle) and the passive stiffness sweep need no new plant -- they are runtime
settings of ankle_mode / ankle_stiffness in env.py.

    python -m model.make_ankle_variants

ANKLE MOTOR SPEC (the mass-realistic requirement). Defaults to the AK60-39 the robot already uses
on the hip-roll axis: 0.75 kg, 72 N*m peak, 39:1, no-load 10.3 rad/s, reflected armature 0.046.
It is welded AT THE ANKLE JOINT, which is the pessimistic-but-honest placement -- distal mass is
the expensive kind (see [[hardware-speed-ceiling]]: ~9x torso mass). A remote drive (motor at the
knee or hip, belt/cable to the ankle) would move most of that 0.75 kg proximal and is a genuinely
different, kinder design; if the active arm loses only on mass, that is the follow-up to run.
"""
import os
import re
import sys

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "dash01.xml")
ACTIVE = os.path.join(HERE, "dash01_active.xml")

ANKLE_JOINTS = {"L": "LegLeftNCS-v1_Révolution-9", "R": "LegRightNCS-v1_Révolution-10"}
ANKLE_BODY = {"L": "FootLeftNCS-v1", "R": "FootRightNCS-v1"}
# ankle qpos addresses in the composite-scalar-base layout (see env.py); used for the lock targets
ANKLE_QADR = {"L": 11, "R": 17}

# --- ankle motor spec (output/joint side, after gearbox) ---------------------------------------
# MASSLESS, with an AKE90-8's PERFORMANCE envelope (170 N*m peak / 55 continuous, no-load 22 rad/s).
#
# Deliberate: we do not yet know whether an ankle motor is worth anything, nor what performance it
# would need, and those are the two things this study is for. Charging the active arm a specific
# motor's mass and rotor inertia would answer a narrower question ("is THIS motor worth it") and
# would let a loss be blamed on the hardware choice rather than on the idea. So the active arm is
# an UPPER BOUND: real torque and real speed limits, no mass, no reflected inertia. If it still
# loses, an actuated ankle is dead on the merits and no motor selection rescues it. If it wins, the
# telemetry (peak torque, peak speed, peak power, time above continuous torque) is the SPEC — and
# only then is it worth re-running with that motor's actual mass at the ankle.
ANKLE_MOTOR = dict(mass=0.0,            # massless: see above. >0 welds a point mass at the ankle.
                   inertia=0.0,
                   armature=None,       # None = leave the joint's own; no rotor inertia charged
                   forcerange=170.0,    # N*m peak, AKE90-8
                   kp=200.0, kv=5.0,    # same servo gains as the cam/thigh AKE90s
                   ctrlrange=(-1.047, 1.047))


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write(path, txt):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(txt)


def _joint_name_in(txt, side):
    """The on-disk file stores the accented joint names in a mangled encoding; match whatever is
    actually there rather than assuming, so a CAD regen with different mojibake still works."""
    tail = "9" if side == "L" else "10"
    m = re.search(rf'<joint\s+name="((?:LegLeft|LegRight)[^"]*-{tail})"', txt)
    if not m:
        raise SystemExit(f"ankle joint for side {side} not found in the model")
    return m.group(1)


def add_ankle_locks(path=BASE):
    """Add lock_ankle_L/R equality joint constraints (inactive) -> the RIGID arm.

    Pinned at the settled LOADED stance angle, so activating the lock does not teleport the foot.

    solref 0.002 (NOT the 0.005 the base-DOF locks use). At 0.005 the weld holds on MuJoCo 3.7 but
    GIVES WAY on 3.3.7 -- measured on the cluster, the ankle drifts 0.858 rad over a 60-step
    rollout, i.e. the "rigid" arm is silently a compliant one there, and the smoke gate fails and
    aborts every queued job. At 0.002 the drift is 0.005 rad on 3.3.7. Local dev is on 3.7.0 and the
    cluster venv on 3.3.7, so anything solver-marginal has to be tuned against the OLDER one."""
    txt = _read(path)
    if 'name="lock_ankle_L"' in txt:
        print("ankle locks already present")
        return path

    model = mujoco.MjModel.from_xml_path(path)
    qpos = model.key_qpos[0]
    eqs = []
    for side in ("L", "R"):
        j = _joint_name_in(txt, side)
        q = float(qpos[ANKLE_QADR[side]])
        eqs.append(f'    <joint name="lock_ankle_{side}" joint1="{j}" '
                   f'polycoef="{q:.9g} 0 0 0 0" active="false" '
                   f'solref="0.002 1" solimp="0.999 0.9999 0.0001" />\n')
        print(f"lock_ankle_{side}: pinning {j} at {q:+.4f} rad")

    txt = txt.replace("  </equality>", "".join(eqs) + "  </equality>", 1)
    _write(path, txt)
    print(f"added ankle locks -> {path}")
    return path


def make_active(src=BASE, dst=ACTIVE, motor=ANKLE_MOTOR):
    """Add a position actuator + a welded motor point mass on each ankle joint."""
    txt = _read(src)
    lo, hi = motor["ctrlrange"]

    # 1. rotor inertia, only if the spec charges it (armature=None keeps the joint's own value, so
    #    the idealized motor adds no reflected inertia either -- at the ankle that matters a lot:
    #    an AKE90's 0.0216 kg*m^2 is ~4x the foot's own 0.006 swing inertia)
    if motor["armature"] is not None:
        for side in ("L", "R"):
            j = _joint_name_in(txt, side)
            txt = re.sub(rf'(<joint\s+name="{re.escape(j)}"[^>]*?)\barmature="[^"]*"',
                         lambda m: f'{m.group(1)}armature="{motor["armature"]:g}"', txt, count=1)

    # 2. motor mass welded at the ankle, inside the FOOT body (the ankle joint's own body) at the
    #    joint anchor -- the foot body's frame origin IS the ankle, so pos="0 0 0". Skipped for the
    #    massless spec, which is the default (see ANKLE_MOTOR).
    if motor["mass"] > 0:
        for side in ("L", "R"):
            body = ANKLE_BODY[side]
            weld = (f'<body name="motor_ankle_{side}" pos="0 0 0">'
                    f'<inertial pos="0 0 0" mass="{motor["mass"]:g}" '
                    f'diaginertia="{motor["inertia"]:g} {motor["inertia"]:g} '
                    f'{motor["inertia"]:g}"/></body>')
            m = re.search(rf'(<body\s+name="{re.escape(body)}"[^>]*>\n)([ \t]*)', txt)
            if not m:
                raise SystemExit(f"body {body} not found")
            indent = m.group(2)
            txt = txt[:m.end(1)] + indent + weld + "\n" + txt[m.end(1):]

    # 3. the actuators themselves, appended so the ankle dims land AFTER the 6 gait actuators
    #    (action/ctrl order = actuator order; env.py appends the ankle channel at the tail).
    acts = "".join(
        f'    <position name="ankle_{side}" joint="{_joint_name_in(txt, side)}" '
        f'kp="{motor["kp"]:g}" kv="{motor["kv"]:g}" '
        f'forcerange="{-motor["forcerange"]:g} {motor["forcerange"]:g}" '
        f'ctrlrange="{lo:g} {hi:g}" />\n'
        for side in ("L", "R"))
    txt = txt.replace("  </actuator>", acts + "  </actuator>", 1)

    # 4. keyframe ctrl grows 6 -> 8; the ankle servo holds the settled stance angle so the initial
    #    pose is unchanged and the study's arms all start from the same posture.
    model = mujoco.MjModel.from_xml_path(src)
    q = model.key_qpos[0]
    ank = " ".join(f"{float(q[ANKLE_QADR[s]]):.6g}" for s in ("L", "R"))
    txt = re.sub(r'(<key\b[^>]*\bctrl=")([^"]+)(")',
                 lambda m: f"{m.group(1)}{m.group(2)} {ank}{m.group(3)}", txt, count=1)

    _write(dst, txt)

    # 5. re-settle: the ankle is now a servo fighting the spring, and it carries +0.75 kg per side
    sys.path.insert(0, HERE)
    from build_model import compute_standing_keyframe
    m2 = mujoco.MjModel.from_xml_path(dst)
    txt = _read(dst)
    ctrl = [float(v) for v in re.search(r'<key\b[^>]*\bctrl="([^"]+)"', txt).group(1).split()]
    qk = compute_standing_keyframe(m2, ctrl)
    txt = re.sub(r'(<key\b[^>]*\bqpos=")([^"]+)(")',
                 lambda m: m.group(1) + " ".join(f"{v:.6g}" for v in qk) + m.group(3), txt, count=1)
    _write(dst, txt)

    m3 = mujoco.MjModel.from_xml_path(dst)
    print(f"\n{dst}\n  nu {model.nu} -> {m3.nu}, "
          f"mass {model.body_subtreemass[1]:.3f} -> {m3.body_subtreemass[1]:.3f} kg "
          + (f"(+{2*motor['mass']:.2f} kg at the ankles)" if motor["mass"] > 0
             else "(MASSLESS motor — upper bound, see ANKLE_MOTOR)"))
    print(f"  torque {motor['forcerange']:g} N*m peak"
          + ("" if motor["armature"] is not None else ", no reflected rotor inertia"))
    print(f"  stance height {float(model.key_qpos[0][2]):.5f} -> {float(qk[2]):.5f} m")
    print(f"  actuators: " + ", ".join(
        mujoco.mj_id2name(m3, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(m3.nu)))
    return dst


if __name__ == "__main__":
    add_ankle_locks()
    make_active()
