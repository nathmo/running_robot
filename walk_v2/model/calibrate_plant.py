"""Fit the two v2 plant numbers the artifact says must be MEASURED in sim, then rebuild the XMLs.

    python walk_v2/model/calibrate_plant.py            # writes plant_fit.json + rebuilds both XMLs

1. LEG SERIES SPRING (§07): one-parameter fit. The prismatic spring acts along the shin axis,
   which is tilted at stance, so the VERTICAL sink at the foot is the spring's compression
   times a projection factor (measured 0.79: 29.7 kN/m axial reads 3.94 mm, 23.4 kN/m reads 5.0). This measures the vertical foot deflection under one body weight
   on ONE leg (base welded in the air, six motor joints pinned by joint equalities so the whole
   leg's compliance is read), rescales k so the vertical sink is exactly 5.0 mm, re-measures,
   reports the ±50% DR corners (expect ~10 and ~3.3 mm) and refuses to proceed outside 5 ± 0.5.

2. DRIVE ARMATURE (§07): closed-loop position Bode of the thigh in the air (base welded, legs
   unloaded), stepped sines 1-30 Hz, the same protocol as ak_bode_sweep.py on the robot.
   Target corners: 6.3 Hz at kp 200 / kd 5 and 19 Hz at kp 500 / kd 5. The armature is one
   number per motor family, so the two-point fit is a compromise the report states honestly:
   the achieved corners and the peaking are written to plant_fit.json next to the targets. The
   hip family is fitted to kp/kd (120/4 -> 4.8 Hz) with the same sweep on hip_roll.

Both measurements use classic MuJoCo (CPU, float64); MJX runs the same XML.
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_v2_model as mk  # noqa: E402

BW_N = 148.5                     # 15.14 kg
TARGET_DEFL_M = 0.005
TARGET_CORNERS = {"cam_thigh": [(200.0, 5.0, 6.3), (500.0, 5.0, 19.0)],
                  "hip": [(120.0, 4.0, 120.0 / 4.0 / (2 * np.pi))]}


# ------------------------------------------------------------------ welded-in-air variant
def welded_xml(arm_ct, arm_hip, lift=0.15, pin=(), leg_spring_k=mk.LEG_SPRING_K):
    """The planar tree with ALL base joints removed and the base hung `lift` m above stance.
    `pin` = actuator names whose joints are welded at the keyframe angle by a joint equality."""
    xml, _ = mk.build(arm_cam_thigh=arm_ct, arm_hip=arm_hip, planar=True, write=False,
                      leg_spring_k=leg_spring_k)
    root = ET.fromstring(xml)
    root.find("compiler").set("meshdir", mk.MESH_ABS)
    xml = ET.tostring(root, encoding="unicode")
    base = next(b for b in root.iter("body") if b.get("name") == "bodyNCS-v1")
    key = root.find("keyframe/key")
    q = mk._fa(key.get("qpos"))
    m_tmp = mujoco.MjModel.from_xml_string(xml, {})
    z = float(q[m_tmp.jnt_qposadr[mujoco.mj_name2id(m_tmp, mujoco.mjtObj.mjOBJ_JOINT, "base_z")]])
    for j in list(base.findall("joint")):
        base.remove(j)
    base.set("pos", mk._fs([0.0, 0.0, z + lift]))
    # keyframe: drop the 3 base entries (x, z, pitch are the first three qpos of the planar tree)
    key.set("qpos", mk._fs(q[3:]))
    if pin:
        eq = root.find("equality")
        act = {a.get("name"): a.get("joint") for a in root.find("actuator")}
        for name in pin:
            jn = act[name]
            jid = mujoco.mj_name2id(m_tmp, mujoco.mjtObj.mjOBJ_JOINT, jn)
            q0 = float(q[m_tmp.jnt_qposadr[jid]])
            e = ET.SubElement(eq, "joint")
            e.set("name", f"pin_{name}")
            e.set("joint1", jn)
            e.set("polycoef", f"{q0:.10g} 0 0 0 0")
            e.set("solref", "0.002 1")
            e.set("solimp", "0.9999 0.99999 0.0001")
    return ET.tostring(root, encoding="unicode")


def _act_joint(m, name):
    a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    jid = m.actuator_trnid[a, 0]
    return a, int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])


def _pd(m, d, target, kp, kd):
    tau = np.zeros(m.nu)
    for a in range(m.nu):
        jid = m.actuator_trnid[a, 0]
        q = d.qpos[m.jnt_qposadr[jid]]
        v = d.qvel[m.jnt_dofadr[jid]]
        tau[a] = kp[a] * (target[a] - q) - kd[a] * v
    return tau


# ------------------------------------------------------------------ 1. link spring
def foot_deflection(xml, force_n=BW_N, t_s=1.5):
    """Vertical deflection (m, positive = up) of the LEFT toe under +force_n at the foot body,
    every motor joint pinned by a joint equality (welded_xml(pin=...)). Returns (defl, ok)."""
    m = mujoco.MjModel.from_xml_string(xml, {})
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "foot_L_col")
    foot_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "FootLeftNCS-v1")
    n = int(t_s / m.opt.timestep)
    z = []
    for i in range(n):
        d.xfrc_applied[foot_b, 2] = force_n if i >= n // 3 else 0.0
        mujoco.mj_step(m, d)
        if i == n // 3 - 1:
            z0 = float(d.geom_xpos[g, 2])
        z.append(float(d.geom_xpos[g, 2]))
    if not np.all(np.isfinite(d.qpos)):
        return np.nan, False
    tail = np.array(z[-200:])
    return float(tail.mean() - z0), bool(tail.std() < 2e-5)


PINS = ("hip_roll_L", "cam_L", "thigh_L", "hip_roll_R", "cam_R", "thigh_R")


def verify_leg_spring(arm_ct, arm_hip):
    k = mk.LEG_SPRING_K
    def defl(kk):
        # the rig applies the GROUND REACTION (+148.5 N, up) at the foot, so the foot rises:
        # a positive reading is the sink the same load produces in stance
        d, ok = foot_deflection(welded_xml(arm_ct, arm_hip, pin=PINS, leg_spring_k=kk))
        return d, ok
    for _ in range(3):                    # linear: one rescale converges, two confirm
        d_nom, ok = defl(k)
        k = k * d_nom / TARGET_DEFL_M
    d_nom, ok = defl(k)
    d_soft, _ = defl(0.5 * k)
    d_stiff, _ = defl(1.5 * k)
    out = dict(leg_spring_k=float(k), leg_spring_b=float(mk.LEG_SPRING_B),
               leg_spring_k_axial_design=float(mk.LEG_SPRING_K),
               leg_defl_nominal_m=float(d_nom), leg_defl_x0p5_m=float(d_soft),
               leg_defl_x1p5_m=float(d_stiff),
               foot_stiffness_n_per_m=float(BW_N / max(d_nom, 1e-9)), settled=bool(ok))
    if not ok or abs(d_nom - TARGET_DEFL_M) > 0.0005:
        raise SystemExit(f"leg spring fit FAILED: {1e3*d_nom:.2f} mm at 1 BW "
                         f"(target 5 +- 0.5 mm), settled={ok}")
    return out


# ------------------------------------------------------------------ 2. armature
def bode_corner(xml, act_name, kp_j, kd_j, freqs=None, amp=0.05):
    """-3 dB corner (Hz) and max peaking (dB) of the closed position loop on `act_name`; the
    other five joints are pinned by equalities in `xml` (welded_xml(pin=others))."""
    freqs = freqs if freqs is not None else np.geomspace(1.0, 30.0, 14)
    m = mujoco.MjModel.from_xml_string(xml, {})
    d = mujoco.MjData(m)
    a_id, qadr, dadr = _act_joint(m, act_name)
    kp = np.zeros(m.nu)
    kd = np.zeros(m.nu)
    kp[a_id], kd[a_id] = kp_j, kd_j
    gains = []
    for f in freqs:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        target0 = np.array([d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]] for a in range(m.nu)])
        n_settle = int(1.0 / m.opt.timestep)
        n_meas = int(max(4.0 / f, 1.0) / m.opt.timestep)
        t, q = [], []
        for i in range(n_settle + n_meas):
            tt = i * m.opt.timestep
            tgt = target0.copy()
            tgt[a_id] += amp * np.sin(2 * np.pi * f * tt)
            tau = _pd(m, d, tgt, kp, kd)
            lim = m.actuator_forcerange[:, 1]
            d.ctrl[:] = np.clip(tau, -lim, lim)
            mujoco.mj_step(m, d)
            if i >= n_settle:
                t.append(tt)
                q.append(d.qpos[qadr] - target0[a_id])
        t, q = np.array(t), np.array(q)
        A = np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones_like(t)], 1)
        c, *_ = np.linalg.lstsq(A, q, rcond=None)
        gains.append(np.hypot(c[0], c[1]) / amp)
    gains = np.array(gains)
    db = 20 * np.log10(np.maximum(gains, 1e-9))
    peaking = float(db.max())
    below = np.flatnonzero(db < -3.0103)
    if below.size == 0:
        return float(freqs[-1]), peaking, gains
    i = int(below[0])
    if i == 0:
        return float(freqs[0]), peaking, gains
    # log-interpolate the -3 dB crossing
    f0, f1, d0, d1 = np.log(freqs[i - 1]), np.log(freqs[i]), db[i - 1], db[i]
    fc = np.exp(f0 + (-3.0103 - d0) * (f1 - f0) / (d1 - d0))
    return float(fc), peaking, gains


def mk_drive_kp(m):
    return [120.0 if "hip" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) else 200.0
            for a in range(m.nu)]


def mk_drive_kd(m):
    return [4.0 if "hip" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) else 5.0
            for a in range(m.nu)]


def fit_armature(leg_spring_k):
    grid = np.geomspace(0.002, 0.6, 16)
    out = {}
    others = lambda name: tuple(p for p in PINS if p != name)
    # cam/thigh family on the thigh sweep, two operating points. Objective: log-corner error
    # plus peaking above 1 dB (the robot shows none; trading corner for ringing is not a fit).
    rows = []
    for arm in grid:
        xml = welded_xml(arm, 0.046, pin=others("thigh_L"), leg_spring_k=leg_spring_k)
        cs = []
        for kp_j, kd_j, tgt in TARGET_CORNERS["cam_thigh"]:
            fc, pk, _ = bode_corner(xml, "thigh_L", kp_j, kd_j)
            cs.append((fc, pk, tgt))
        err = sum(np.log(c[0] / c[2]) ** 2 + 0.1 * max(0.0, c[1] - 1.0) ** 2 for c in cs)
        rows.append((float(arm), err, cs))
        print(f"  armature {arm:.4f}: thigh corner {cs[0][0]:5.2f} Hz (pk {cs[0][1]:+.2f} dB) @200/5, "
              f"{cs[1][0]:5.2f} Hz (pk {cs[1][1]:+.2f} dB) @500/5   err {err:.3f}", flush=True)
    best = min(rows, key=lambda r: r[1])
    out["arm_cam_thigh"] = best[0]
    out["thigh_corner_hz_200_5"] = best[2][0][0]
    out["thigh_peaking_db_200_5"] = best[2][0][1]
    out["thigh_corner_hz_500_5"] = best[2][1][0]
    out["thigh_peaking_db_500_5"] = best[2][1][1]
    out["thigh_targets_hz"] = [6.3, 19.0]
    out["cam_thigh_grid"] = [(r[0], r[2][0][0], r[2][1][0]) for r in rows]
    # hip family on hip_roll at 120/4
    rows = []
    for arm in grid:
        xml = welded_xml(out["arm_cam_thigh"], arm, pin=others("hip_roll_L"),
                         leg_spring_k=leg_spring_k)
        kp_j, kd_j, tgt = TARGET_CORNERS["hip"][0]
        fc, pk, _ = bode_corner(xml, "hip_roll_L", kp_j, kd_j)
        rows.append((float(arm), np.log(fc / tgt) ** 2 + 0.1 * max(0.0, pk - 1.0) ** 2, fc, pk))
        print(f"  hip armature {arm:.4f}: corner {fc:5.2f} Hz (pk {pk:+.2f} dB) @120/4", flush=True)
    best = min(rows, key=lambda r: r[1])
    out["arm_hip"] = best[0]
    out["hip_corner_hz_120_4"] = best[2]
    out["hip_peaking_db_120_4"] = best[3]
    out["hip_target_hz"] = TARGET_CORNERS["hip"][0][2]
    return out


def main():
    fit_path = HERE / "plant_fit.json"
    print("[calibrate] 1/2 leg series spring (verify 5 mm at 1 BW on one leg)", flush=True)
    arm_ct, arm_hip = 0.0216, 0.046
    spring = verify_leg_spring(arm_ct, arm_hip)
    print(f"  k = {spring['leg_spring_k']:.0f} N/m: {1e3*spring['leg_defl_nominal_m']:.2f} mm nominal / "
          f"{1e3*spring['leg_defl_x0p5_m']:.2f} mm at x0.5 / {1e3*spring['leg_defl_x1p5_m']:.2f} mm at x1.5 "
          f"-> {spring['foot_stiffness_n_per_m']/1e3:.1f} kN/m", flush=True)
    print("[calibrate] 2/2 armature (thigh corner 6.3 Hz @200/5, 19 Hz @500/5; hip kp/kd)", flush=True)
    arm = fit_armature(spring["leg_spring_k"])
    fit = dict(spring, **arm)
    fit_path.write_text(json.dumps(fit, indent=1))
    print(f"[calibrate] wrote {fit_path.name}; rebuilding the plants", flush=True)
    for planar in (False, True):
        out, info = mk.build(arm_cam_thigh=fit["arm_cam_thigh"], arm_hip=fit["arm_hip"],
                             planar=planar, leg_spring_k=fit["leg_spring_k"])
        print(f"  {out.name}: settled z {info['z_settled']:.4f} m")


if __name__ == "__main__":
    main()
