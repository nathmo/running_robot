"""Shared simulation core for leg2d/optimize.py and leg2d/render_gait.py.

The rig: base_x AND base_z are free (only y/roll/pitch/yaw are railed) -- see build_leg2d.py for
why z has to stay free (a rigid z-lock bypasses the leg and removes real stance traction). Because
z is real, the single-support leg CAN buckle under a badly-shaped gait, so this still checks for
collapse/fly-off, same as the very first version of this rig.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

PKG = Path(__file__).resolve().parent
REPO = PKG.parents[0]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
if str(REPO / "training") not in sys.path:
    sys.path.insert(0, str(REPO / "training"))

import gait
import motor
import cpg_gait  # training/cpg_gait.py -- the measured foot-IK LUT

MODEL_PATH = PKG / "model" / "leg2d.xml"
LUT_PATH = REPO / "training" / "model" / "cpg_foot_lut.npz"

CONTROL_HZ = 200.0          # matches the rest of the project (memory/pi-loop-timing.md)
N_CYCLES = 6                # hop cycles per evaluation
N_TRANSIENT = 2             # cycles discarded before measuring (settle from the keyframe)
MAX_SIM_S = 6.0             # wall/sim-time cap per evaluation regardless of cadence
KP, KV = 200.0, 5.0         # outer PD gains, matching dash01's cam/thigh position-actuator gains
COLLAPSE_FRAC = 0.4         # base_z below this fraction of the keyframe height = collapsed
FLYOFF_FRAC = 2.5           # base_z above this multiple of the keyframe height = diverged


def build_sim():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    a_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cam_L")
    a_thigh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thigh_L")
    a_hr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_roll_L")
    j_cam = int(model.actuator_trnid[a_cam, 0])
    j_thigh = int(model.actuator_trnid[a_thigh, 0])
    j_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_x")
    j_z = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_z")
    ids = dict(a_cam=a_cam, a_thigh=a_thigh, a_hr=a_hr, j_cam=j_cam, j_thigh=j_thigh,
               qadr_cam=model.jnt_qposadr[j_cam], dadr_cam=model.jnt_dofadr[j_cam],
               qadr_thigh=model.jnt_qposadr[j_thigh], dadr_thigh=model.jnt_dofadr[j_thigh],
               qadr_x=model.jnt_qposadr[j_x], qadr_z=model.jnt_qposadr[j_z])
    return model, data, ids


def nominal_pose(model, data, ids):
    """cam_L/thigh_L keyframe angles (the reference the gait's foot-IK offsets are added to) and
    the keyframe standing height (the collapse/fly-off reference)."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return (float(data.qpos[ids["qadr_cam"]]), float(data.qpos[ids["qadr_thigh"]]),
            float(data.qpos[ids["qadr_z"]]))


def load_lut():
    lut = cpg_gait.load_lut(str(LUT_PATH))
    ik0 = np.asarray(cpg_gait.foot_ik(0.0, 0.0, lut), float)
    return lut, ik0


def evaluate(model, data, ids, lut, ik0, nominal_cam, nominal_thigh, stand_z,
             f_hz, duty, stride, clearance, z_off, record=False):
    """Run N_CYCLES of the prescribed gait from the standing keyframe; return achieved speed,
    per-motor RMS torque, saturation fraction (how often the raw PD command asked for more torque
    than the real motor can deliver at that joint velocity), and collapse/fly-off flags.
    `record=True` also returns full per-step traces for the limiting-factor breakdown."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    sub_steps = max(1, int(round(1.0 / CONTROL_HZ / model.opt.timestep)))
    control_dt = sub_steps * model.opt.timestep
    t_end = min(N_CYCLES / f_hz, MAX_SIM_S)
    t_transient = min(N_TRANSIENT / f_hz, 0.5 * t_end)

    therm_cam, therm_thigh = motor.ThermalTracker(), motor.ThermalTracker()
    sat_steps = total_steps = 0
    x_meas = []
    z_hist = []
    trace = dict(t=[], tau_cam=[], tau_thigh=[], w_cam=[], w_thigh=[]) if record else None
    t = 0.0
    finite = True
    while t < t_end:
        dx, dz = gait.foot_xz(t, f_hz, duty, stride, clearance, z_off)
        djk = cpg_gait.foot_ik(dx, dz, lut) - ik0
        q_cam = data.qpos[ids["qadr_cam"]]
        dq_cam = data.qvel[ids["dadr_cam"]]
        q_thigh = data.qpos[ids["qadr_thigh"]]
        dq_thigh = data.qvel[ids["dadr_thigh"]]
        tau_cam_cmd = KP * ((nominal_cam + djk[0]) - q_cam) - KV * dq_cam
        tau_thigh_cmd = KP * ((nominal_thigh + djk[1]) - q_thigh) - KV * dq_thigh
        tau_cam = float(motor.clamp_torque(tau_cam_cmd, dq_cam))
        tau_thigh = float(motor.clamp_torque(tau_thigh_cmd, dq_thigh))
        data.ctrl[ids["a_cam"]] = tau_cam
        data.ctrl[ids["a_thigh"]] = tau_thigh
        data.ctrl[ids["a_hr"]] = 0.0
        for _ in range(sub_steps):
            mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            finite = False
            break
        z_hist.append(float(data.qpos[ids["qadr_z"]]))

        if t >= t_transient:
            therm_cam.add(tau_cam)
            therm_thigh.add(tau_thigh)
            sat_steps += int(abs(tau_cam) < abs(tau_cam_cmd) - 1e-6 or
                             abs(tau_thigh) < abs(tau_thigh_cmd) - 1e-6)
            total_steps += 1
            x_meas.append((t, float(data.qpos[ids["qadr_x"]])))
            if record:
                trace["t"].append(t)
                trace["tau_cam"].append(tau_cam)
                trace["tau_thigh"].append(tau_thigh)
                trace["w_cam"].append(float(dq_cam))
                trace["w_thigh"].append(float(dq_thigh))
        t += control_dt

    z_hist = np.array(z_hist) if z_hist else np.array([stand_z])
    collapsed = bool(z_hist.min() < COLLAPSE_FRAC * stand_z)
    flew_off = bool(z_hist.max() > FLYOFF_FRAC * stand_z)
    ok = finite and not collapsed and not flew_off

    if len(x_meas) >= 2 and ok:
        ts = np.array([p[0] for p in x_meas])
        xs = np.array([p[1] for p in x_meas])
        v_measured = float(np.polyfit(ts, xs, 1)[0])
    else:
        v_measured = 0.0

    out = dict(
        f_hz=f_hz, duty=duty, stride=stride, clearance=clearance, z_off=z_off,
        v_target=gait.nominal_speed(f_hz, stride), v_measured=v_measured,
        rms_cam=therm_cam.rms, rms_thigh=therm_thigh.rms,
        thermal_ok=bool(therm_cam.feasible() and therm_thigh.feasible()),
        sat_frac=sat_steps / max(total_steps, 1),
        finite=finite, collapsed=collapsed, flew_off=flew_off, ok=ok,
        z_min=float(z_hist.min()), z_max=float(z_hist.max()),
    )
    if record:
        out["trace"] = {k: np.array(v) for k, v in trace.items()}
    return out
