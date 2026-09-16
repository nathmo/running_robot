"""Cross-validation + a torque-prediction report for the identified parameters.

The honest success metric for a base-fixed, current-as-torque identification is NOT per-parameter
accuracy but how well the model predicts torque on a HELD-OUT trajectory. We fit on part of the data
and report the predicted-vs-measured torque residual on the rest, per joint.
"""
import numpy as np

from . import dataset as ds
from . import kt_calibration as ktc
from . import mujoco_id


def _predict(model, dset, params, loop):
    """Per-joint predicted torque (Nm) for a dataset given identified params."""
    act_dof = dset["act_dof"]
    jm = {j: m for m, j in ds.MOTOR_TO_JOINT.items()}
    data = __import__("mujoco").MjData(model)
    tg = np.array([ktc.gravity_torque(model, data, dset["qpos"][i], act_dof, loop=loop)
                   for i in range(len(dset["qpos"]))])
    pred = np.zeros_like(tg)
    for k, jname in enumerate(ds.ACT_JOINTS):
        m = jm[jname]
        ie = params["effective_inertia"].get(m, 0.0)
        fr = params["friction"].get(m, {"viscous": 0.0, "coulomb": 0.0})
        pred[:, k] = (ie * dset["qdd_act"][:, k] + fr["viscous"] * dset["qd_act"][:, k]
                      + fr["coulomb"] * np.sign(dset["qd_act"][:, k]) + tg[:, k])
    return pred, tg


def cross_validate(model, datasets, masses=None, kt=None, holdout=0.3):
    """Fit on the first (1-holdout) of each run, predict the held-out tail, report residual RMS and
    the normalized RMS (residual / measured-torque RMS) per joint."""
    SCALARS = ("act_dof", "fs", "loop_resid", "leg", "hold_other")   # per-run, never sliced
    train, test = [], []
    for d in datasets:
        n = len(d["t"]); cut = int(n * (1 - holdout))
        train.append({k: (v[:cut] if hasattr(v, "__len__") and getattr(v, "ndim", 1) and
                           k not in SCALARS else v) for k, v in d.items()})
        test.append({k: (v[cut:] if hasattr(v, "__len__") and getattr(v, "ndim", 1) and
                          k not in SCALARS else v) for k, v in d.items()})
    params = mujoco_id.identify(model, train, masses=masses, kt=kt)
    loop = (ds.loop_sites(model), ds.loop_dof(model))
    saved = ktc.set_masses(model, masses)
    jm = {j: m for m, j in ds.MOTOR_TO_JOINT.items()}
    try:
        rms, nrms = {}, {}
        for d in test:
            pred, _ = _predict(model, d, params, loop)
            for k, jname in enumerate(ds.ACT_JOINTS):
                m = jm[jname]
                # score a joint only on runs that actually moved it, and only if it was identified
                if m not in params["kt"] or (d.get("hold_other", True)
                                             and m.split(".")[0] != d.get("leg")):
                    continue
                meas = params["kt"][m] * d["cur"][:, k]
                rms.setdefault(m, []).append(pred[:, k] - meas)
                nrms.setdefault(m, []).append(meas)
    finally:
        for bid, mm in saved.items():
            model.body_mass[bid] = mm
    report = {}
    for m in rms:
        r = np.concatenate(rms[m]); tot = np.concatenate(nrms[m])
        report[m] = {"residual_rms_nm": float(np.sqrt(np.mean(r ** 2))),
                     "normalized_rms": float(np.sqrt(np.mean(r ** 2)) /
                                             max(np.sqrt(np.mean(tot ** 2)), 1e-6))}
    overall = float(np.mean([v["normalized_rms"] for v in report.values()])) if report else None
    return {"per_joint": report, "overall_normalized_rms": overall}
