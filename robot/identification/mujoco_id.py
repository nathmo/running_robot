"""Dynamic identification by ONE per-joint linear regression on measured data.

MuJoCo inverse-dynamics through this robot's near-singular 4-bar loop is numerically unusable (see
the plan / project memory), so the dynamic parameters are identified WITHOUT it. Every joint obeys

    Kt * current  =  I_eff * qddot  +  b_v * qdot  +  b_c * sign(qdot)  +  tau_grav(q)

which, dividing by Kt, is LINEAR in the measured regressors:

    current = (I_eff/Kt)*qddot + (b_v/Kt)*qdot + (b_c/Kt)*sign(qdot) + (1/Kt)*tau_grav(q) + i0

tau_grav(q) is the reduced-coordinate gravity torque (kt_calibration.gravity_torque — stable through
the loop, weighed masses); i0 is the drive's DC current bias. A single least-squares fit of `current`
onto [qddot, qdot, sign(qdot), tau_grav, 1] recovers everything at once, cleanly separated:
  * the tau_grav column (varies with pose across the slow sweep)  -> Kt = 1/coeff
  * the qddot column (large during the fast sweep)                -> I_eff = coeff*Kt
  * qdot / sign(qdot)                                             -> viscous / Coulomb friction
  * the constant column                                           -> i0, which MUST have its own
    column: over a short sweep tau_grav is itself ~85% DC, so an unmodelled bias lands in Kt and can
    flip its sign outright.
So the quasi-static AND dynamic runs are simply concatenated. The rotor armature is then separated
from I_eff via the model's tree mass-matrix link term (data.qM — forward/stable). If Kt is already
known it can be passed in and only the mechanical params are fit.

CAVEAT (structural, not a bug): Kt and the gravity model are perfectly confounded — only their
product is observable, so "the model under-predicts gravity by 1/x" and "effective Kt is x times
datasheet" fit identically. Breaking it needs an independent torque reference (a known mass on a
known lever arm). Do not read the recovered Kt as a motor property until that is done.
"""
import numpy as np

import mujoco

from . import dataset as ds
from . import kt_calibration as ktc

MIN_SAMPLES = 200            # fewer excited samples than this and the joint is not fittable at all
MIN_GRAV_SWING_NM = 0.30     # tau_grav must actually vary, or 1/Kt rides on the current DC offset


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

    # A run with hold_other excites ONE leg; the other leg's three joints just sit there. Pooling
    # those held blocks into the regression adds thousands of rows carrying no gravity/inertia
    # signal, only the drive's DC current offset — enough to drag the fit through zero and hand back
    # a NEGATIVE Kt. Fit each joint on the runs that actually moved it.
    EXCITED = np.concatenate([
        np.tile([(not d.get("hold_other", True)) or (jm[j].split(".")[0] == d.get("leg"))
                 for j in ds.ACT_JOINTS], (len(d["t"]), 1))
        for d in datasets])

    out = {"kt": {}, "effective_inertia": {}, "armature": {}, "friction": {},
           "residual_rms_nm": {}, "current_bias_a": {}, "skipped": {}}
    for k, jname in enumerate(ds.ACT_JOINTS):
        motor = jm[jname]
        sel = EXCITED[:, k]
        if sel.sum() < MIN_SAMPLES:
            out["skipped"][motor] = f"only {int(sel.sum())} excited samples"
            continue
        qdd, qd, tg, cur = QDD[sel, k], QD[sel, k], TG[sel, k], CUR[sel, k]
        if np.ptp(tg) < MIN_GRAV_SWING_NM and not (kt and motor in kt):
            out["skipped"][motor] = (f"gravity torque only swings {np.ptp(tg):.2f} Nm over the run "
                                     f"— Kt is not identifiable, excite a wider angle range")
            continue
        # Every fit carries an INTERCEPT for the drive's DC current bias. Without it that bias has
        # nowhere to go but the tau_grav column, because over a short sweep tau_grav is itself mostly
        # DC (~85% here) — which is enough to flip the sign of the recovered Kt.
        s = np.sign(qd)
        ones = np.ones_like(qd)
        if kt and motor in kt:
            kt_j = float(kt[motor])                                   # Kt fixed -> fit mechanicals
            Phi = np.column_stack([qdd, qd, s, ones])
            theta, *_ = np.linalg.lstsq(Phi, cur * kt_j - tg, rcond=None)
            i_eff, b_v, b_c, bias_nm = theta
            i0 = float(bias_nm) / kt_j
        else:
            # fit current on [qdd, qd, sign(qd), tau_grav, 1] -> [I/Kt, b_v/Kt, b_c/Kt, 1/Kt, i0]
            Phi = np.column_stack([qdd, qd, s, tg, ones])
            c, *_ = np.linalg.lstsq(Phi, cur, rcond=None)
            if abs(c[3]) < 1e-9:
                continue
            kt_j = 1.0 / c[3]
            i_eff, b_v, b_c, i0 = c[0] * kt_j, c[1] * kt_j, c[2] * kt_j, float(c[4])
        i_eff = max(float(i_eff), 0.0)
        pred_nm = (i_eff * qdd + b_v * qd + b_c * s + tg)            # predicted torque
        out["kt"][motor] = float(kt_j)
        out["current_bias_a"][motor] = i0
        out["effective_inertia"][motor] = i_eff
        out["armature"][motor] = max(i_eff - float(link_Mjj[k]), 0.0)
        out["friction"][motor] = {"viscous": max(float(b_v), 0.0), "coulomb": abs(float(b_c))}
        out["residual_rms_nm"][motor] = float(np.sqrt(np.mean((pred_nm - (cur - i0) * kt_j) ** 2)))
    out["residual_rms_nm_overall"] = float(np.sqrt(np.mean(
        [v ** 2 for v in out["residual_rms_nm"].values()]))) if out["residual_rms_nm"] else None
    return out
