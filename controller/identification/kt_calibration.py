"""Torque-constant (Kt) + gravity model + Coulomb friction from a QUASI-STATIC run.

At low speed the measured joint torque is dominated by gravity, so with the limb MASSES known
(the user weighs each limb) we can compute the gravity torque in real Nm and read Kt off the ratio
tau_grav = Kt * current. The gravity torque is computed from the body-CoM Jacobians (mj_jacBodyCom,
pure forward kinematics) — the ONE MuJoCo primitive that stays numerically sane through the
near-singular 4-bar (mj_inverse does NOT; see the plan / project memory). Approaching each pose from
both directions cancels Coulomb friction in the average and reveals it in the half-difference.

Notes / honesty:
  * abduction + thigh sit on the open serial chain, so their open-tree gravity torque is accurate.
  * cam is the loop crank: its open-tree gravity misses the load transmitted through the pushrod, so
    its Kt is inherited from the thigh (same AKE90-8 motor) rather than fit from gravity.
"""
import numpy as np

import mujoco

from . import dataset as ds

# with the reduced-coordinate gravity (loop projection) all 6 actuated joints are fittable
GRAVITY_FITTABLE = {f"{s}.{r}" for s in ("right", "left") for r in ("abd", "cam", "thigh")}


def set_masses(model, masses):
    """Override model.body_mass from a {body_name: kg} dict. Returns the saved originals to restore."""
    saved = {}
    for name, kg in (masses or {}).items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            saved[bid] = float(model.body_mass[bid])
            model.body_mass[bid] = float(kg)
    return saved


def gravity_torque(model, data, qpos, act_dof, loop=None):
    """Gravity generalized force at the actuated joints for one full-state pose (Nm), computed from
    body-CoM Jacobians (pure forward kinematics — stable through the loop singularity, unlike
    mj_inverse). If `loop`=(sites, d_dof) is given, the loop-coupled passive-joint gravity is
    projected onto the actuated joints via the damped loop Jacobian (REDUCED coordinates) — required
    for cam/thigh, which drive the leg through the 4-bar; abduction is loop-independent either way."""
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    g = model.opt.gravity
    tau = np.zeros(model.nv)
    J = np.zeros((3, model.nv))
    for b in range(1, model.nbody):
        m = model.body_mass[b]
        if m <= 0:
            continue
        mujoco.mj_jacBodyCom(model, data, J, None, b)
        tau += -m * (J.T @ g)
    if loop is None:
        return tau[act_dof]
    sites, d_dof = loop
    G = ds.loop_jacobian(model, data, sites, d_dof, act_dof)
    return tau[act_dof] + G.T @ tau[d_dof]


def calibrate(model, dset, kt_prior=None, masses=None, low_speed_dps=8.0):
    """Kt (Nm/A) + Coulomb friction (Nm) per actuated joint from a quasi-static run.

    dset  : dataset.build() output (must include qpos + q_act/qd_act + cur).
    masses: {body_name: kg} weighed masses (override the model's CAD masses for the gravity calc).
    Returns {kt:{motor:..}, friction:{motor:{coulomb:..}}, grav_rms_nm:{motor:..}}.
    """
    act_dof = dset["act_dof"]
    loop = (ds.loop_sites(model), ds.loop_dof(model))
    saved = set_masses(model, masses)
    data = mujoco.MjData(model)
    try:
        tau_g = np.array([gravity_torque(model, data, dset["qpos"][i], act_dof, loop=loop)
                          for i in range(len(dset["qpos"]))])
    finally:
        for bid, m in saved.items():
            model.body_mass[bid] = m

    low = np.abs(dset["qd_act"]) < np.radians(low_speed_dps)      # quasi-static mask per joint
    cur = np.asarray(dset["cur"])
    qd = np.asarray(dset["qd_act"])
    jm = {j: m for m, j in ds.MOTOR_TO_JOINT.items()}
    kt_prior = kt_prior or {}
    kt, coulomb, grav_rms = {}, {}, {}

    for k, jname in enumerate(ds.ACT_JOINTS):
        motor = jm[jname]
        mask = low[:, k]
        grav_rms[motor] = float(np.sqrt(np.mean(tau_g[:, k] ** 2)))
        if motor in GRAVITY_FITTABLE and mask.sum() > 20:
            # Kt = slope of tau_grav vs current at low speed (through the origin, robust median ratio
            # over samples with enough current to avoid divide-by-noise)
            g, c = tau_g[mask, k], cur[mask, k]
            good = np.abs(c) > 0.05
            if good.sum() > 10:
                kt[motor] = float(np.median(g[good] / c[good]))
            # Coulomb: half the |current| gap between the two sweep directions, in Nm
            pos = np.abs(qd[:, k]) > np.radians(low_speed_dps)
            if pos.sum() > 20 and motor in kt:
                up = cur[(qd[:, k] > 0) & pos, k]
                dn = cur[(qd[:, k] < 0) & pos, k]
                if len(up) > 5 and len(dn) > 5:
                    coulomb[motor] = float(abs(np.median(up) - np.median(dn)) / 2.0 * abs(kt[motor]))
    for motor in [jm[j] for j in ds.ACT_JOINTS]:      # fall back to any prior for un-fit joints
        if motor not in kt and motor in kt_prior:
            kt[motor] = float(kt_prior[motor])
    return {"kt": kt, "friction": {m: {"coulomb": v} for m, v in coulomb.items()},
            "grav_rms_nm": grav_rms}
