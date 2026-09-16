"""Turn a captured MEASURE run into model-frame trajectories the estimator can use. Needs scipy
(zero-phase filtering + derivatives) and mujoco (loop closure for the unmeasured passive joints).

The web UI logs NORMALIZED motor degrees for the 6 ACTUATED joints; the model needs radians for ALL
18 DOF. So per sample we:
  1. map each actuated joint  norm_deg -> model rad  via  model = sign*norm + offset  (cam/thigh use
     the run's saved FK map; abduction assumes the captured zero == model zero, override if not),
  2. hold the base fixed (the real robot is bolted at the base: qpos/qvel/qacc[base] = 0),
  3. solve the two closed 4-bar loops for the passive knee + pushrod joints (Newton on the connect
     site residual), and set the passive ankle to its spring neutral (a documented approximation —
     the light foot's dynamic ankle deflection is second order for the driving-joint torques),
  4. zero-phase low-pass filter and differentiate to get qvel/qacc for every DOF.

Everything downstream (Kt calibration, mujoco_id) consumes these arrays; MuJoCo's mj_inverse then
gives the predicted joint torque INCLUDING the loop + ankle-spring forces, so the closed chain never
has to be hand-derived.
"""
import numpy as np
from scipy.signal import butter, filtfilt

import mujoco

# webui motor (side.role)  ->  model actuated joint name (build_model.py ACTUATORS)
MOTOR_TO_JOINT = {
    "right.abd": "bodyNCS-v1_Révolution-2", "left.abd": "bodyNCS-v1_Révolution-1",
    "right.cam": "HipRightNCS-v1_Révolution-4", "left.cam": "HipLeftNCS-v1_Révolution-3",
    "right.thigh": "HipRightNCS-v1_Révolution-6", "left.thigh": "HipLeftNCS-v1_Révolution-5",
}
ACT_JOINTS = ["bodyNCS-v1_Révolution-1", "HipLeftNCS-v1_Révolution-3", "HipLeftNCS-v1_Révolution-5",
              "bodyNCS-v1_Révolution-2", "HipRightNCS-v1_Révolution-4", "HipRightNCS-v1_Révolution-6"]
BASE_JOINTS = ["base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw"]
# passive loop joints solved by loop closure, and the site pair whose weld defines each leg's loop
LOOP = {
    "L": dict(knee="ThighLeftNCS-v1_Révolution-7", push="CamLeftNCS-v1_Révolution-11",
              ankle="LegLeftNCS-v1_Révolution-9", s1="pushrod_tip_L", s2="leg_anchor_L"),
    "R": dict(knee="ThighRightNCS-v1_Révolution-8", push="CamRightNCS-v1_Révolution-12",
              ankle="LegRightNCS-v1_Révolution-10", s1="pushrod_tip_R", s2="leg_anchor_R"),
}


def _jid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _qadr(model, name):
    return model.jnt_qposadr[_jid(model, name)]


def _dadr(model, name):
    return model.jnt_dofadr[_jid(model, name)]


def _sid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


# The model mirrors the right leg's sagittal axes (Y vs -Y), but the webui's FK map is fitted per
# leg in that leg's OWN frame, so it reports the same cam/thigh signs for both sides. The mirror is a
# property of the model, not of the drive, so it belongs here — in the norm->model conversion — and
# not in fklut. Without it the right leg's reconstructed pose walks out of the 4-bar assembly band
# and loop closure sticks at ~0.78 m (measured on three separate runs; 1e-8 with the mirror).
# Abduction does not need it: right.abd's identified Kt lands within 6% of left.abd's as-is.
MODEL_SAGITTAL_MIRROR = {"left": 1.0, "right": -1.0}


def joint_map_from_meta(meta):
    """Per motor: (sign, offset_deg) for  model_deg = sign*norm_deg + offset_deg. cam/thigh come from
    the run's saved FK map (model_map), mirrored into the model frame for the right leg; abduction
    defaults to identity (captured zero == model zero)."""
    mm = meta.get("model_map") or {}
    out = {}
    for motor in MOTOR_TO_JOINT:
        side, role = motor.split(".")
        m = mm.get(side, {})
        if role in ("cam", "thigh"):
            out[motor] = (float(m.get(role, 1)) * MODEL_SAGITTAL_MIRROR[side],
                          float(m.get(f"{role}_off_deg", 0.0)))
        else:                                        # abduction: no FK entry — assume identity
            out[motor] = (1.0, 0.0)
    return out


def _close_loops(model, data, act_adr, act_val, base_adr, iters=16, tol=1e-8):
    """Kinematic loop closure by damped site-Jacobian Newton over each leg's passive knee+pushrod
    joints, holding base=0 and actuated=act_val. data.qpos supplies the passive warm start. Returns
    the largest residual after solving. Pure kinematics (mj_kinematics), no dynamics -> no ringing."""
    for a in base_adr:
        data.qpos[a] = 0.0
    for a, v in zip(act_adr, act_val):
        data.qpos[a] = v
    legs = []
    for s, L in LOOP.items():
        legs.append((_sid(model, L["s1"]), _sid(model, L["s2"]),
                     [_dadr(model, L["knee"]), _dadr(model, L["push"])],
                     [_qadr(model, L["knee"]), _qadr(model, L["push"])]))
    jac1 = np.zeros((3, model.nv)); jac2 = np.zeros((3, model.nv))
    worst = 0.0
    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)                        # cdof for mj_jacSite (else J is zero)
        worst = 0.0
        for s1, s2, cols, qadr in legs:
            r = data.site_xpos[s1] - data.site_xpos[s2]
            worst = max(worst, np.linalg.norm(r))
            mujoco.mj_jacSite(model, data, jac1, None, s1)
            mujoco.mj_jacSite(model, data, jac2, None, s2)
            J = (jac1 - jac2)[:, cols]                       # 3x2
            dq = np.linalg.solve(J.T @ J + 1e-9 * np.eye(2), -J.T @ r)   # damped LS
            for a, d in zip(qadr, dq):
                data.qpos[a] += float(d)
        if worst < tol:
            break
    return worst


def reconstruct_qpos(model, q_act_rad, cont_steps=40):
    """q_act_rad: [N,6] actuated joint angles (ACT_JOINTS order) -> [N, nq] full model qpos with the
    base fixed at 0, the passive knee/pushrod solved by loop closure, and the ankle at spring
    neutral. The FIRST sample is reached by CONTINUATION from the assembled default pose (qpos=0,
    where the loop is exactly closed) so the passive joints track the correct assembly branch; every
    later sample warm-starts from the previous solution. Returns (qpos[N,nq], max_residual)."""
    N = len(q_act_rad)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)                        # default qpos: loop exactly closed
    act_adr = [_qadr(model, j) for j in ACT_JOINTS]
    base_adr = [_qadr(model, j) for j in BASE_JOINTS]
    q_start = np.array([data.qpos[a] for a in act_adr])
    for f in np.linspace(0.0, 1.0, cont_steps)[1:]:          # walk actuated default -> q[0]
        _close_loops(model, data, act_adr, q_start + f * (q_act_rad[0] - q_start), base_adr)
    qpos = np.zeros((N, model.nq))
    worst = 0.0
    for i in range(N):
        worst = max(worst, _close_loops(model, data, act_adr, q_act_rad[i], base_adr))
        qpos[i] = data.qpos.copy()                           # carries over as next warm start
    return qpos, worst


def loop_dof(model):
    """DOF indices of the passive loop joints [knee_L, push_L, knee_R, push_R]."""
    return np.array([_dadr(model, LOOP[s][k]) for s in ("L", "R") for k in ("knee", "push")])


def loop_sites(model):
    return [(_sid(model, LOOP[s]["s1"]), _sid(model, LOOP[s]["s2"])) for s in ("L", "R")]


def loop_jacobian(model, data, sites, d_dof, act_dof, damp=1e-6):
    """G = dq_passive/dq_actuated (4x6) from the connect constraint, damped near the 4-bar
    singularity so it stays bounded. Assumes data.qpos is set and kinematics fresh-ish (recomputes).
    Used to express loop-coupled gravity/torque in the actuated (reduced) coordinates.

    `damp` was 1e-3, which is far more than this constraint needs away from the singularity —
    cond(Jd'Jd) is ~157 over a normal measurement sweep, and that damping shrank G enough to inflate
    the identified gravity torque by ~20%. Results converge for damp <= 1e-5; 1e-6 keeps a margin
    against the singular poses while leaving the well-conditioned ones untouched."""
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    Jc = np.zeros((6, model.nv)); j1 = np.zeros((3, model.nv)); j2 = np.zeros((3, model.nv))
    for li, (s1, s2) in enumerate(sites):
        mujoco.mj_jacSite(model, data, j1, None, s1)
        mujoco.mj_jacSite(model, data, j2, None, s2)
        Jc[3 * li:3 * li + 3] = j1 - j2
    Jd = Jc[:, d_dof]; Ja = Jc[:, act_dof]
    return -np.linalg.solve(Jd.T @ Jd + damp * np.eye(len(d_dof)), Jd.T @ Ja)   # damped LS, 4x6


def _lowpass(x, fs, cutoff):
    cutoff = min(cutoff, 0.45 * fs)
    b, a = butter(4, cutoff / (0.5 * fs))
    pad = 3 * max(len(a), len(b))
    if len(x) <= pad:
        return x
    return filtfilt(b, a, x, axis=0)


def build(model, run, meta, cutoff_hz=12.0, decimate=1):
    """Full model-frame dataset from a MEASURE run.

    Returns a dict with (all trimmed of filter edge transients, optionally decimated):
      t, q_act, qd_act, qdd_act   [M,6]   actuated joint pos/vel/acc (model rad)
      cur                          [M,6]   measured current (A), same column order as q_act
      qpos, qvel, qacc             [M,nq/nv]  full model state for mj_inverse
      act_dof                      [6]     dof indices of the actuated joints
      moving                       [M,6]   |qd_act| above a small threshold (for friction sign)
    Column order is ACT_JOINTS = [hipL, camL, thighL, hipR, camR, thighR]; the run's motor columns
    (right.* , left.*) are remapped to that order here."""
    t = np.asarray(run["t"], float)
    fs = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 200.0
    jm = joint_map_from_meta(meta)
    motors = list(meta.get("motor_names") or
                  ["right.abd", "right.cam", "right.thigh", "left.abd", "left.cam", "left.thigh"])

    # remap the 6 measured motor columns into ACT_JOINTS order, converting deg->model rad.
    # Torque and angle must share a frame: pos_norm is already in the CALIBRATED frame (the webui
    # applied calibration.sign when normalizing) and `sign` then takes it to the model frame, but
    # `cur` is raw motor-frame. So current needs BOTH signs to become a model-frame torque. Getting
    # this wrong flips the sign of every identified Kt.
    cal_motors = ((meta.get("calibration") or {}).get("motors") or {})
    q_act = np.zeros((len(t), 6))
    cur = np.zeros((len(t), 6))
    for k, jname in enumerate(ACT_JOINTS):
        motor = next(m for m, j in MOTOR_TO_JOINT.items() if j == jname)
        col = motors.index(motor)
        sign, off = jm[motor]
        cal_sign = float((cal_motors.get(motor) or {}).get("sign", 1.0))
        q_act[:, k] = np.radians(sign * np.asarray(run["pos_norm"])[:, col] + off)
        cur[:, k] = cal_sign * sign * np.asarray(run["cur"])[:, col]

    q_f = _lowpass(q_act, fs, cutoff_hz)
    qd = np.gradient(q_f, 1.0 / fs, axis=0)
    qdd = np.gradient(_lowpass(qd, fs, cutoff_hz), 1.0 / fs, axis=0)

    qpos, loop_resid = reconstruct_qpos(model, q_f)
    qvel = np.gradient(_lowpass(qpos, fs, cutoff_hz), 1.0 / fs, axis=0)
    # convert qpos derivative to qvel/qacc (hinge/slide DOF: dof index == derivative of qpos scalar)
    # for this model every joint is 1-DOF hinge/slide so nq==nv and columns correspond 1:1
    qacc = np.gradient(_lowpass(qvel, fs, cutoff_hz), 1.0 / fs, axis=0)

    # trim filter edges, then optionally decimate for the optimizer
    e = min(int(0.05 * fs) + 5, len(t) // 4)
    sl = slice(e, len(t) - e, max(1, decimate))
    act_dof = np.array([_dadr(model, j) for j in ACT_JOINTS])
    out = dict(t=t[sl], q_act=q_f[sl], qd_act=qd[sl], qdd_act=qdd[sl], cur=cur[sl],
               qpos=qpos[sl], qvel=qvel[sl], qacc=qacc[sl], act_dof=act_dof, fs=fs,
               loop_resid=float(loop_resid),
               leg=meta.get("leg"), hold_other=bool(meta.get("hold_other", True)),
               moving=(np.abs(qd[sl]) > np.radians(2.0)))
    return out
