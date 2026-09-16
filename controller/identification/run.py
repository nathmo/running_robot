"""Estimator entry point:  python -m identification.run --model ... --out ... --measures a.npz b.npz

Loads captured MEASURE runs, identifies Kt + reflected inertia + friction per joint (data-driven,
no unstable inverse dynamics), builds each link's inertia from the weighed masses (inertia scales
with mass for a fixed shape/density), cross-validates the torque prediction, and writes
identified_params.json. Run on a machine with mujoco + scipy (the training/dev box), then upload the
JSON to the Pi web UI.
"""
import argparse
import json
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (robot/)

from identification import dataset as ds, mujoco_id, validate, paramio, frames, model_inertials

ARRAYS = ("t", "cmd_norm", "pos_norm", "pos_raw", "spd", "cur")
# link body -> its actuator motor (for the rotor-armature write-back), same-motor grouping
BODY_MOTOR = None


def _load_measure(path):
    z = np.load(path, allow_pickle=False)
    run = {k: z[k] for k in ARRAYS}
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    return run, meta


def _load_config(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="path to dash01.xml")
    ap.add_argument("--out", required=True, help="output identified_params.json path")
    ap.add_argument("--config", default=None, help="dynamics_config.json (weighed masses, Kt prior)")
    ap.add_argument("--measure-dir", default=".", help="dir holding the measurement npz files")
    ap.add_argument("--measures", nargs="+", required=True, help="measurement npz file names")
    ap.add_argument("--cutoff-hz", type=float, default=12.0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    cfg = _load_config(args.config)
    masses = {k: float(v) for k, v in (cfg.get("masses") or {}).items()}
    kt_prior = {k: float(v) for k, v in (cfg.get("kt") or {}).items()} or None

    datasets, sources = [], []
    for name in args.measures:
        path = os.path.join(args.measure_dir, name)
        run, meta = _load_measure(path)
        d = ds.build(model, run, meta, cutoff_hz=args.cutoff_hz)
        if d["loop_resid"] > 1e-3:
            print(f"  WARNING: {name} loop closure residual {d['loop_resid']:.1e} m "
                  f"(FK map / cam offset may push the pose outside the model's assembly band)")
        datasets.append(d)
        sources.append(name)
        print(f"  loaded {name}: {len(d['t'])} samples @ {d['fs']:.0f} Hz, leg={meta.get('leg')}")
    if not datasets:
        print("no usable measurements"); sys.exit(1)

    print("identifying ...")
    ident = mujoco_id.identify(model, datasets, masses=masses, kt=kt_prior)
    cv = validate.cross_validate(model, datasets, masses=masses, kt=kt_prior)

    # --- assemble identified_params.json ---
    cad = model_inertials.read_bodies(args.model)
    params = paramio.default()
    params["sources"] = sources
    params["kt"] = ident["kt"]
    params["friction"] = ident["friction"]
    params["rotor_armature"] = ident["armature"]
    params["validation"] = {"fit_residual_rms_nm": ident["residual_rms_nm"],
                            "fit_residual_rms_nm_overall": ident["residual_rms_nm_overall"],
                            "heldout_normalized_rms": cv["overall_normalized_rms"],
                            "heldout_per_joint": cv["per_joint"]}
    # link inertias: scale the CAD tensor by the weighed-mass ratio (inertia ~ mass for fixed shape)
    for name, c in cad.items():
        mass_w = masses.get(name)
        body = {"mass": c["mass"], "com": c["com"], "inertia": dict(c["inertia"])}
        if mass_w and c["mass"] > 1e-9:
            r = mass_w / c["mass"]
            body["mass"] = mass_w
            body["inertia"] = {k: v * r for k, v in c["inertia"].items()}
            body["cad_delta"] = {"mass_ratio": r}
            body["method"] = "cad_tensor_scaled_by_weighed_mass"
        else:
            body["method"] = "cad_only (no weighed mass)"
        params["bodies"][name] = body
    params["notes"] = ("Kt + reflected rotor inertia (armature) + viscous/Coulomb friction are "
                       "identified from measured torque (per-joint regression, gravity via the "
                       "stable reduced-coordinate Jacobian). Link inertia tensors are the CAD "
                       "tensors rescaled by the weighed mass; full per-link tensor identification "
                       "is not separable on a base-fixed 3-DOF leg with a closed loop (see plan).")

    paramio.save(params, args.out)
    print(f"\nwrote {args.out}")
    for motor, why in (ident.get("skipped") or {}).items():
        print(f"  SKIPPED {motor}: {why}")
    print("Kt (Nm/A):", {m: round(v, 3) for m, v in ident["kt"].items()})
    print("rotor armature (kg m^2):", {m: round(v, 4) for m, v in ident["armature"].items()})
    print("fit residual RMS (Nm):", round(ident["residual_rms_nm_overall"] or 0, 3),
          "| held-out normalized RMS:", round(cv["overall_normalized_rms"] or 0, 3))


if __name__ == "__main__":
    main()
