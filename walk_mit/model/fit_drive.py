"""Fit the sim drive to the measured MIT-mode position-loop Bode (artifact §07, "Drive").

Measured 2026-09-01 (ak_bode_sweep.py, thighs, legs attached, unloaded, robot homed):
    kp 200 / kd 5 : -3 dB at 6.1 Hz (L) / 6.5 Hz (R)
    kp 500 / kd 5 : ~19 Hz, +0.02 dB max peaking, -138 deg at 30 Hz
    corner scales LINEARLY in kp (damping-dominated, corner ~ kp/kd), 11-13 ms phase-slope delay.
Sim today: <position kp=200 kv=5> on the model inertia -> critically damped 2nd order at ~13 Hz.

This tool runs the SAME protocol in sim (base bolted in the air, stepped sines 1-30 Hz on the thigh
at both gain pairs), for a sweep of joint armature on the cam/thigh family, and reports the corner
and the peaking at each point against the two measured corners. The recommendation is the armature
minimising the two-point log-corner error with a peaking penalty; hip-roll gets the same SCALE on
its own modelled armature. The env writes the result at load through cfg.drive_armature.

    python model/fit_drive.py [--model dash01.xml] [--write drive_fit.json]
"""
import argparse
import json
import os

import copy

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURED = {(200.0, 5.0): 6.3, (500.0, 5.0): 19.0}      # Hz, -3 dB
# joints resolved through the (ASCII) actuator names: the joint names carry a double-encoded
# accent in the export and do not round-trip through mj_name2id
CT_ACTS = ("cam_L", "thigh_L", "cam_R", "thigh_R")
HIP_ACTS = ("hip_roll_L", "hip_roll_R")


def act_dofs(model, names):
    out = []
    for n in names:
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if a < 0:
            raise KeyError(n)
        out.append(int(model.jnt_dofadr[model.actuator_trnid[a, 0]]))
    return out


def _rig(model, clearance=0.25):
    """Base locked in the air, ankles locked, keyframe posture, motors holding the stance."""
    d = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, d, kid)
    for n in ("lock_x", "lock_y", "lock_z", "lock_roll", "lock_pitch", "lock_yaw",
              "lock_ankle_L", "lock_ankle_R"):
        e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, n)
        if e >= 0:
            d.eq_active[e] = 1
    ez = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "lock_z")
    model.eq_data[ez, 0] = float(model.key_qpos[kid][2]) + clearance
    d.qpos[2] = model.eq_data[ez, 0]
    d.ctrl[:] = model.key_ctrl[kid]
    return d


def bode(model, act="thigh_L", kp=200.0, kd=5.0, freqs=None, amp=0.05, settle_s=1.0, meas_s=3.0):
    """(freqs, gain_dB, phase_deg) of joint response / commanded target at each frequency."""
    freqs = np.geomspace(1.0, 30.0, 14) if freqs is None else np.asarray(freqs, float)
    m = copy.deepcopy(model)
    a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
    m.actuator_gainprm[a, 0] = kp
    m.actuator_biasprm[a, 1] = -kp
    m.actuator_biasprm[a, 2] = -kd
    jid = m.actuator_trnid[a, 0]
    qadr = int(m.jnt_qposadr[jid])
    dt = float(m.opt.timestep)
    d = _rig(m)
    for _ in range(int(2.0 / dt)):
        mujoco.mj_step(m, d)
    q0 = float(d.qpos[qadr])
    c0 = float(d.ctrl[a])
    gains, phases = [], []
    for f in freqs:
        n_set, n_meas = int(settle_s / dt), int(meas_s / dt)
        t = 0.0
        ys, ts = [], []
        for i in range(n_set + n_meas):
            d.ctrl[a] = c0 + amp * np.sin(2 * np.pi * f * t)
            mujoco.mj_step(m, d)
            t += dt
            if i >= n_set:
                ys.append(float(d.qpos[qadr]) - q0)
                ts.append(t - dt)     # the sample follows the command issued at t - dt
        ys, ts = np.asarray(ys), np.asarray(ts)
        A = np.stack([np.sin(2 * np.pi * f * ts), np.cos(2 * np.pi * f * ts), np.ones_like(ts)], 1)
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        mag = np.hypot(coef[0], coef[1]) / amp
        ph = np.degrees(np.arctan2(coef[1], coef[0]))
        gains.append(20 * np.log10(max(mag, 1e-9)))
        phases.append(ph)
    return freqs, np.asarray(gains), np.asarray(phases)


def corner_hz(freqs, gain_db):
    """-3 dB crossing by log-log interpolation (None if never crossed)."""
    g = gain_db - gain_db[0]
    for i in range(1, len(freqs)):
        if g[i] <= -3.0 <= g[i - 1] or (g[i] <= -3.0 and i == 1):
            x0, x1 = np.log(freqs[i - 1]), np.log(freqs[i])
            y0, y1 = g[i - 1], g[i]
            if y1 == y0:
                return float(freqs[i])
            return float(np.exp(x0 + (x1 - x0) * (-3.0 - y0) / (y1 - y0)))
    return None


def sweep(model_path, scales, write=None):
    base = mujoco.MjModel.from_xml_path(model_path)
    ct_dofs = act_dofs(base, CT_ACTS)
    hip_dofs = act_dofs(base, HIP_ACTS)
    a_ct0 = float(base.dof_armature[ct_dofs[0]])
    a_hip0 = float(base.dof_armature[hip_dofs[0]])
    rows = []
    print(f"model armature: cam/thigh {a_ct0} kg m^2, hip_roll {a_hip0} kg m^2")
    print(f"{'scale':>6} {'arm_ct':>8} | {'kp200 corner':>13} {'peak dB':>8} | {'kp500 corner':>13} {'peak dB':>8} | {'err':>6}")
    for sc in scales:
        m = copy.deepcopy(base)
        for dd in ct_dofs:
            m.dof_armature[dd] = a_ct0 * sc
        for dd in hip_dofs:
            m.dof_armature[dd] = a_hip0 * sc
        res = {}
        err = 0.0
        for (kp, kd), target in MEASURED.items():
            fr, g, ph = bode(m, kp=kp, kd=kd)
            c = corner_hz(fr, g)
            peak = float(np.max(g - g[0]))
            res[(kp, kd)] = (c, peak, ph)
            if c is None:
                err += 4.0
            else:
                err += np.log(c / target) ** 2 + 0.5 * max(0.0, peak - 0.5) ** 2
        c2, p2, _ = res[(200.0, 5.0)]
        c5, p5, ph5 = res[(500.0, 5.0)]
        rows.append(dict(scale=sc, armature_ct=a_ct0 * sc, armature_hip=a_hip0 * sc,
                         corner_200=c2, peak_200=p2, corner_500=c5, peak_500=p5,
                         phase_500_30hz=float(ph5[-1]), err=float(err)))
        print(f"{sc:6.2f} {a_ct0 * sc:8.4f} | {('%.1f Hz' % c2) if c2 else '  none':>13} {p2:8.2f} | "
              f"{('%.1f Hz' % c5) if c5 else '  none':>13} {p5:8.2f} | {err:6.3f}")
    best = min(rows, key=lambda r: r["err"])
    print(f"\nmeasured: kp200 -> {MEASURED[(200.0, 5.0)]} Hz, kp500 -> {MEASURED[(500.0, 5.0)]} Hz (no peaking)")
    print(f"best two-point fit: armature scale {best['scale']:.2f} -> cam/thigh {best['armature_ct']:.4f}, "
          f"hip_roll {best['armature_hip']:.4f}  (corners {best['corner_200']:.1f} / {best['corner_500']:.1f} Hz, "
          f"peaking {best['peak_200']:.2f} / {best['peak_500']:.2f} dB)")
    print("  -> cfg.drive_armature = (%.4f, %.4f)" % (best["armature_hip"], best["armature_ct"]))
    if write:
        with open(write, "w") as f:
            json.dump(dict(measured={f"{k[0]:.0f}/{k[1]:.0f}": v for k, v in MEASURED.items()},
                           rows=rows, best=best,
                           drive_armature=[best["armature_hip"], best["armature_ct"]]), f, indent=1)
        print(f"wrote {write}")
    return rows, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "dash01.xml"))
    ap.add_argument("--scales", default="0,0.25,0.5,1,2,4,8,16")
    ap.add_argument("--write", default=None)
    args = ap.parse_args()
    sweep(args.model, [float(x) for x in args.scales.split(",")], args.write)


if __name__ == "__main__":
    main()
