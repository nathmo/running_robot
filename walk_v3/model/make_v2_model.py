"""Build the v2 plant(s) from walk_mit's dash01.xml (copied here as dash01_base.xml).

    python walk_v3/model/make_v2_model.py [--arm-cam-thigh A --arm-hip A]

Writes dash01_v2_free.xml (all six base DOFs) and dash01_v2_planar.xml (x, z, pitch only).
What changes vs the base plant, and why (artifact §07 / §12):

  1. RIGID ANKLE, FOLDED. The sprint lineage welded the ankle with a joint equality at the
     keyframe angle. Here the joint is removed and its angle is folded into the foot body's pose,
     so the kinematic tree is one DOF shorter per leg and there is no equality for the solver to
     hold. The spring's 249 g assembly is gone, the 40 g carbon tube is not: each shin loses
     0.209 kg (inertia scaled by the mass ratio, as apply_measured_masses.py does).
  2. LOOP CLOSURE STIFFENED + LEG-AXIS SERIES SPRING. The base plant's soft <connect>
     (solref 0.005 s, solimp 0.95..0.99) let the base sink ~15 cm under load; the robot yields
     5 mm. The loop now carries the ankle-lock values (solref 0.002, solimp 0.999 0.9999) so the
     constraint itself no longer yields. The measured compliance is then an EXPLICIT spring --
     but not at the pushrod tip: measured while building this, a vertical stance load barely
     loads the loop at all (the leg is at 99.5% reach, the load path is axial), so a rod spring
     cannot reproduce a 5 mm sink whatever its stiffness. The spring is a prismatic joint along
     the shin axis at the foot, k = 148.5 N / 5 mm = 29.7 kN/m foot-referred (5 mm at one body
     weight on one leg by construction, verified by calibrate_plant.py), damping 600 N s/m
     (zeta ~0.45 against the robot's mass in stance). dr_link_spring scales k ±50% per env
     (jnt_stiffness is a batched MJX field).
  3. <motor> ACTUATORS. The PD (kp(phi), kd(phi) from the latched spec), the torque-speed clamp,
     the delay and the thermal node all live in walk_v3/drive.py, evaluated every 1 kHz substep
     in JAX, so the MJCF actuator is a pure torque input with the delivered peak as forcerange.
  4. ARMATURE FITTED to the measured closed-loop corner (6.3 Hz at kp 200 / kd 5) by
     calibrate_plant.py; the numbers land here as joint armature.
  5. NO base-lock equalities: the planar plant simply has no y / roll / yaw joints.
  6. The keyframe is RE-SETTLED under the new ankle and masses (z free, the other base DOFs held,
     motors holding the nominal targets), exactly as the loaded-stance numbers were measured.
     `<numeric name="nominal_ctrl">` carries the six joint targets the gait is centred on; the
     keyframe ctrl is torque and stays zero.
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
BASE = HERE / "dash01_base.xml"
# visual meshes live in walk_mit/model/meshes (tracked in git); the written XMLs reference them
# relatively, in-process compiles use the absolute path (from_xml_string has no file location)
MESH_REL = "../../walk_mit/model"
MESH_ABS = str((HERE.parent.parent / "walk_mit" / "model").resolve())


def _base_model():
    tree = ET.parse(BASE)
    tree.getroot().find("compiler").set("meshdir", MESH_ABS)
    return mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"), {})

FOOT_BODIES = {"L": "FootLeftNCS-v1", "R": "FootRightNCS-v1"}
SHIN_BODIES = {"L": "LegLeftNCS-v1", "R": "LegRightNCS-v1"}
SHIN_REMOVAL_KG = 0.249 - 0.040        # spring assembly out, carbon tube in
LEG_SPRING_K = 148.5 / 0.005            # N/m foot-referred: 5 mm at one body weight (§07)
LEG_SPRING_B = 600.0                    # N s/m
LEG_SPRING_RANGE = 0.03                 # m, hard stop either way
LOOP_SOLREF = "0.002 1"                 # the ankle-lock values: the constraint no longer yields
LOOP_SOLIMP = "0.999 0.9999 0.0001"
ACT_ORDER = ["hip_roll_L", "cam_L", "thigh_L", "hip_roll_R", "cam_R", "thigh_R"]
NOMINAL_CTRL = np.array([0.0, 0.0, 0.12, 0.0, 0.0, -0.12])
PLANAR_REMOVE = ("base_y", "base_roll", "base_yaw")


def _fa(s):
    return np.array([float(x) for x in s.split()])


def _fs(a):
    return " ".join(f"{float(x):.10g}" for x in np.asarray(a).ravel())


def _euler_to_quat(euler):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.asarray(euler, float), "xyz")
    return q


def _axis_angle_quat(axis, ang):
    q = np.zeros(4)
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    mujoco.mju_axisAngle2Quat(q, a, float(ang))
    return q


def _mul(q1, q2):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q1, q2)
    return q


def _rot(q, v):
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, float), q)
    return out


def fold_ankle(root, side, angle):
    """Remove the ankle joint of `side` (the only joint in the foot body), folding `angle` into
    the foot body's pose. Joints are found by BODY, never by name: the exported joint names carry
    an accented character whose encoding differs between the file, ElementTree and MuJoCo."""
    for body in root.iter("body"):
        if body.get("name") != FOOT_BODIES[side]:
            continue
        joints = body.findall("joint")
        if len(joints) != 1:
            raise ValueError(f"{FOOT_BODIES[side]} has {len(joints)} joints, expected 1 (ankle)")
        joint = joints[0]
        jpos = _fa(joint.get("pos", "0 0 0"))
        jaxis = _fa(joint.get("axis"))
        body_pos = _fa(body.get("pos", "0 0 0"))
        if body.get("quat") is not None:
            q_body = _fa(body.get("quat"))
        else:
            q_body = _euler_to_quat(_fa(body.get("euler", "0 0 0")))
        q_j = _axis_angle_quat(jaxis, angle)
        # x_parent = body_pos + R_body (jpos + R_q (x - jpos))  ->  R' = R_body R_q,
        # p' = body_pos + R_body (jpos - R_q jpos)
        q_new = _mul(q_body, q_j)
        p_new = body_pos + _rot(q_body, jpos - _rot(q_j, jpos))
        body.set("pos", _fs(p_new))
        body.set("quat", _fs(q_new))
        body.attrib.pop("euler", None)
        body.remove(joint)
        # the leg-axis series spring: a slide joint in the foot body along the SHIN axis (the
        # foot sits at -x of the shin frame), expressed in the foot's own frame
        q_conj = np.array([q_new[0], -q_new[1], -q_new[2], -q_new[3]])
        axis_foot = _rot(q_conj, np.array([1.0, 0.0, 0.0]))
        spring = ET.Element("joint")
        spring.set("name", f"leg_spring_{side}")
        spring.set("type", "slide")
        spring.set("axis", _fs(axis_foot))
        spring.set("pos", "0 0 0")
        spring.set("limited", "true")
        spring.set("range", f"{-LEG_SPRING_RANGE:.4g} {LEG_SPRING_RANGE:.4g}")
        spring.set("stiffness", f"{LEG_SPRING_K:.6g}")
        spring.set("damping", f"{LEG_SPRING_B:.6g}")
        spring.set("springref", "0")
        spring.set("armature", "0.02")
        body.insert(0, spring)
        return
    raise KeyError(f"foot body {FOOT_BODIES[side]} not found")


def build(arm_cam_thigh=0.0216, arm_hip=0.046, planar=False, out=None, write=True,
          leg_spring_k=LEG_SPRING_K):
    tree = ET.parse(BASE)
    root = tree.getroot()
    root.set("model", "dash01_v2_planar" if planar else "dash01_v2_free")
    root.find("compiler").set("meshdir", MESH_ABS)          # relative path restored at write time
    # visual meshes come from walk_mit/model/meshes (tracked in git); no second copy in walk_v2
    root.find("compiler").set("meshdir", "../../walk_mit/model")

    # --- keyframe of the base plant: the ankle angles to fold, the qpos to seed the settle ---
    key = root.find("keyframe/key")
    base_qpos = _fa(key.get("qpos"))
    m0 = _base_model()
    ankle_angle = {}
    for s, bn in FOOT_BODIES.items():
        bid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, bn)
        jids = [j for j in range(m0.njnt) if m0.jnt_bodyid[j] == bid]
        assert len(jids) == 1, jids
        ankle_angle[s] = float(base_qpos[m0.jnt_qposadr[jids[0]]])

    # 1. rigid ankle folded, shin mass
    for s in "LR":
        fold_ankle(root, s, ankle_angle[s])
    for body in root.iter("body"):
        if body.get("name") in SHIN_BODIES.values():
            inert = body.find("inertial")
            m_old = float(inert.get("mass"))
            m_new = m_old - SHIN_REMOVAL_KG
            inert.set("mass", f"{m_new:.6g}")
            fi = _fa(inert.get("fullinertia")) * (m_new / m_old)
            inert.set("fullinertia", _fs(fi))

    # 2. loop closure: stiff constraint + direct-stiffness series spring; drop every lock
    eq = root.find("equality")
    for e in list(eq):
        name = e.get("name", "")
        if name.startswith("lock_"):
            eq.remove(e)
        elif name in ("loop_L", "loop_R"):
            e.set("solref", LOOP_SOLREF)
            e.set("solimp", LOOP_SOLIMP)

    # 3. <motor> actuators (torque input); delivered peaks as force + ctrl range
    act = root.find("actuator")
    for a in list(act):
        fr = a.get("forcerange")
        new = ET.SubElement(act, "motor")
        new.set("name", a.get("name"))
        new.set("joint", a.get("joint"))
        new.set("gear", "1")
        new.set("forcerange", fr)
        new.set("ctrlrange", fr)
        act.remove(a)

    # 4. armature per family, by the body the joint moves (hip bodies: hip roll; cam/thigh bodies)
    for body in root.iter("body"):
        bn = body.get("name", "")
        for j in body.findall("joint"):
            if bn.startswith("Hip"):
                j.set("armature", f"{arm_hip:.6g}")
            elif bn.startswith("Cam") or bn.startswith("Thigh"):
                j.set("armature", f"{arm_cam_thigh:.6g}")
            elif j.get("name", "").startswith("leg_spring_"):
                j.set("stiffness", f"{leg_spring_k:.6g}")

    # 5. planar: remove base y / roll / yaw joints
    if planar:
        base = next(b for b in root.iter("body") if b.get("name") == "bodyNCS-v1")
        for j in list(base.findall("joint")):
            if j.get("name") in PLANAR_REMOVE:
                base.remove(j)

    # 6. nominal joint targets as a <custom><numeric>; keyframe ctrl (torque) = 0, and the
    #    keyframe qpos is re-settled below on the compiled model
    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    num = ET.SubElement(custom, "numeric")
    num.set("name", "nominal_ctrl")
    num.set("data", _fs(NOMINAL_CTRL))

    # Settle ON THE PLANAR TREE, always: with no y/roll/yaw joints the base is a true weld
    # (except z), the way the loaded-stance numbers were measured. Overwriting qpos after every
    # step on the free tree is not a hold -- the solver sees a free base within each step and
    # the left leg walked 12 cm inward over a 3 s settle (measured while building this). The
    # settled LEG posture is then mapped onto whichever tree is being written, by joint name.
    keyframe_el = root.find("keyframe")
    root.remove(keyframe_el)
    xml_target = ET.tostring(root, encoding="unicode")
    m = mujoco.MjModel.from_xml_string(xml_target, {})
    if planar:
        m_settle, xml_settle = m, xml_target
    else:
        import copy
        root_p = copy.deepcopy(root)
        base_p = next(b for b in root_p.iter("body") if b.get("name") == "bodyNCS-v1")
        for j in list(base_p.findall("joint")):
            if j.get("name") in PLANAR_REMOVE:
                base_p.remove(j)
        xml_settle = ET.tostring(root_p, encoding="unicode")
        m_settle = mujoco.MjModel.from_xml_string(xml_settle, {})
    q_seed = np.zeros(m_settle.nq)
    for j in range(m_settle.njnt):
        name = mujoco.mj_id2name(m_settle, mujoco.mjtObj.mjOBJ_JOINT, j)
        j0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j0 >= 0:                                   # the leg springs start at rest length
            q_seed[m_settle.jnt_qposadr[j]] = base_qpos[m0.jnt_qposadr[j0]]
    q_settled, info = settle(m_settle, q_seed, planar=True)
    qpos_settled = np.zeros(m.nq)
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        js = mujoco.mj_name2id(m_settle, mujoco.mjtObj.mjOBJ_JOINT, name)
        if js >= 0:
            qpos_settled[m.jnt_qposadr[j]] = q_settled[m_settle.jnt_qposadr[js]]
    key.set("qpos", _fs(qpos_settled))
    key.set("ctrl", _fs(np.zeros(m.nu)))
    key.set("name", "stand")
    root.append(keyframe_el)

    root.find("compiler").set("meshdir", MESH_REL)
    xml = ET.tostring(root, encoding="unicode")
    xml = '<?xml version="1.0" encoding="utf-8"?>\n' + xml
    if write:
        out = Path(out) if out else HERE / ("dash01_v2_planar.xml" if planar else "dash01_v2_free.xml")
        out.write_text(xml, encoding="utf-8")
        return out, info
    return xml, info


def _held_dofs(m, planar):
    """(qpos addresses, dof addresses) of the base DOFs held during a settle: all but z."""
    names = ["base_x", "base_pitch"] if planar else ["base_x", "base_y", "base_roll",
                                                     "base_pitch", "base_yaw"]
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    return ([int(m.jnt_qposadr[j]) for j in jids], [int(m.jnt_dofadr[j]) for j in jids])


def pd_torque(m, d, target, kp=(120, 200, 200, 120, 200, 200), kd=(4, 5, 5, 4, 5, 5)):
    tau = np.zeros(m.nu)
    for a in range(m.nu):
        jid = m.actuator_trnid[a, 0]
        q = d.qpos[m.jnt_qposadr[jid]]
        v = d.qvel[m.jnt_dofadr[jid]]
        tau[a] = kp[a] * (target[a] - q) - kd[a] * v
    lim = m.actuator_forcerange[:, 1]
    return np.clip(tau, -lim, lim)


def settle(m, qpos0, planar, t_s=3.0):
    """Gravity-settle from qpos0 with the base held except z, motors holding NOMINAL_CTRL."""
    d = mujoco.MjData(m)
    d.qpos[:] = qpos0
    held, held_v = _held_dofs(m, planar)
    base_q = d.qpos[held].copy()
    z0 = float(d.qpos[_z_adr(m)])
    n = int(t_s / m.opt.timestep)
    for _ in range(n):
        d.ctrl[:] = pd_torque(m, d, NOMINAL_CTRL)
        mujoco.mj_step(m, d)
        d.qpos[held] = base_q
        d.qvel[held_v] = 0.0
    if not np.all(np.isfinite(d.qpos)):
        raise RuntimeError("settle diverged")
    mujoco.mj_forward(m, d)
    tau = pd_torque(m, d, NOMINAL_CTRL)
    info = dict(z_settled=float(d.qpos[_z_adr(m)]), z_before=z0,
                stand_torque=[float(x) for x in tau],
                max_qvel=float(np.abs(d.qvel).max()))
    return d.qpos.copy(), info


def _z_adr(m):
    return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_z")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-cam-thigh", type=float, default=None)
    ap.add_argument("--arm-hip", type=float, default=None)
    ap.add_argument("--fit", default=str(HERE / "plant_fit.json"),
                    help="plant_fit.json from calibrate_plant.py (CLI values override it)")
    args = ap.parse_args()
    kw = {}
    fit = Path(args.fit)
    if fit.exists():
        f = json.loads(fit.read_text())
        kw = dict(arm_cam_thigh=f["arm_cam_thigh"], arm_hip=f["arm_hip"],
                  leg_spring_k=f["leg_spring_k"])
        print(f"[model] using {fit.name}: {kw}")
    for k in ("arm_cam_thigh", "arm_hip"):
        v = getattr(args, k)
        if v is not None:
            kw[k] = v
    for planar in (False, True):
        out, info = build(planar=planar, **kw)
        print(f"[model] wrote {out.name}: z {info['z_before']:.4f} -> {info['z_settled']:.4f} m "
              f"settled, stand torque {np.round(info['stand_torque'], 2).tolist()}")


if __name__ == "__main__":
    main()
