"""Dynamic identification by ONE per-joint linear regression on measured data.

MuJoCo inverse-dynamics through this robot's near-singular 4-bar loop is numerically unusable (see
the plan / project memory), so the dynamic parameters are identified WITHOUT it. Every joint obeys

    Kt * current  =  I_eff * qddot  +  b_v * qdot  +  b_c * sign(qdot)  +  tau_grav(q)

which, dividing by Kt, is LINEAR in the measured regressors:

    current = (I_eff/Kt)*qddot + (b_v/Kt)*qdot + (b_c/Kt)*sign(qdot) + (1/Kt)*tau_grav(q)

tau_grav(q) is the reduced-coordinate gravity torque (kt_calibration.gravity_torque — stable through
the loop, weighed masses). A single least-squares fit of `current` onto [qddot, qdot, sign(qdot),
tau_grav] recovers everything at once, cleanly separated:
  * the tau_grav column (varies with pose across the slow sweep)  -> Kt = 1/coeff
  * the qddot column (large during the fast sweep)                -> I_eff = coeff*Kt
  * qdot / sign(qdot)                                             -> viscous / Coulomb friction
So the quasi-static AND dynamic runs are simply concatenated. The rotor armature is then separated
from I_eff via the model's tree mass-matrix link term (data.qM — forward/stable). If Kt is already
known it can be passed in and only the mechanical params are fit.
"""
import numpy as np

import mujoco

from . import dataset as ds
from . import kt_calibration as ktc


def _tree_link_inertia(model, qpos_med, act_dof):
    """Link-only joint-space inertia diagonal at a representative pose (tree mass matrix minus rotor
    armature). data.qM is the composite-rigid-body inertia — forward/stable, unaffected by the loop
    singularity that breaks inverse dynamics."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_med
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    return np.array([M[d, d] - model.dof_armature[d] for d in act_dof])


def identify(model, datasets, masses=None, kt=None):
    """Identify Kt + {I_eff, armature, viscous, coulomb} per actuated joint from measured runs.

    datasets : list of dataset.build() outputs (concatenate quasi-static + dynamic runs).
    masses   : {body_name: kg} weighed masses for the (reduced-coordinate) gravity model.
    kt       : optional {motor: Nm/A} — if given, Kt is fixed and only mechanical params are fit.
    Returns per-motor kt, effective_inertia, armature, friction{viscous,coulomb}, residual RMS.
    """
    act_dof = datasets[0]["act_dof"]
    jm = {j: m for m, j in ds.MOTOR_TO_JOINT.items()}
    loop = (ds.loop_sites(model), ds.loop_dof(model))

    saved = ktc.set_masses(model, masses)
    data = mujoco.MjData(model)
    QDD, QD, TG, CUR = [], [], [], []
    try:
        for d in datasets:
            TG.append(np.array([ktc.gravity_torque(model, data, d["qpos"][i], act_dof, loop=loop)
                                for i in range(len(d["qpos"]))]))
            QDD.append(d["qdd_act"]); QD.append(d["qd_act"]); CUR.append(np.asarray(d["cur"]))
        qpos_med = np.median(np.concatenate([d["qpos"] for d in datasets]), axis=0)
        link_Mjj = _tree_link_inertia(model, qpos_med, act_dof)
    finally:
        for bid, m in saved.items():
            model.body_mass[bid] = m
    QDD = np.concatenate(QDD); QD = np.concatenate(QD)
    TG = np.concatenate(TG); CUR = np.concatenate(CUR)

    out = {"kt": {}, "effective_inertia": {}, "armature": {}, "friction": {},
           "residual_rms_nm": {}}
    for k, jname in enumerate(ds.ACT_JOINTS):
        motor = jm[jname]
        qdd, qd, tg, cur = QDD[:, k], QD[:, k], TG[:, k], CUR[:, k]
        s = np.sign(qd)
        if kt and motor in kt:
            kt_j = float(kt[motor])                                   # Kt fixed -> fit mechanicals
            Phi = np.column_stack([qdd, qd, s])
            theta, *_ = np.linalg.lstsq(Phi, cur * kt_j - tg, rcond=None)
            i_eff, b_v, b_c = theta
        else:
            # fit current on [qdd, qd, sign(qd), tau_grav] -> [I_eff/Kt, b_v/Kt, b_c/Kt, 1/Kt]
            Phi = np.column_stack([qdd, qd, s, tg])
            c, *_ = np.linalg.lstsq(Phi, cur, rcond=None)
            if abs(c[3]) < 1e-9:
                continue
            kt_j = 1.0 / c[3]
            i_eff, b_v, b_c = c[0] * kt_j, c[1] * kt_j, c[2] * kt_j
        i_eff = max(float(i_eff), 0.0)
        pred_nm = (i_eff * qdd + b_v * qd + b_c * s + tg)            # predicted torque
        out["kt"][motor] = float(kt_j)
        out["effective_inertia"][motor] = i_eff
        out["armature"][motor] = max(i_eff - float(link_Mjj[k]), 0.0)
        out["friction"][motor] = {"viscous": max(float(b_v), 0.0), "coulomb": abs(float(b_c))}
        out["residual_rms_nm"][motor] = float(np.sqrt(np.mean((pred_nm - cur * kt_j) ** 2)))
    out["residual_rms_nm_overall"] = float(np.sqrt(np.mean(
        [v ** 2 for v in out["residual_rms_nm"].values()]))) if out["residual_rms_nm"] else None
    return out
