"""Sensitivity of the rail-sweep speed bound (envelope + 7 ms MIT delay) to four design
parameters, each scaled x0.7 / x1.0 / x1.3 with everything else nominal:

  leg inertia    -- mass + rotational inertia of every leg body (distal mass lever)
  rotor inertia  -- cam/thigh reflected armature (motor + 8:1 gearbox inertia)
  torque limit   -- deliverable peak (envelope ceiling AND actuator forcerange)
  bus voltage    -- no-load speed w0 scales with V (back-EMF ceiling); stall torque unchanged
                    (current-limited, not voltage-limited at stall)

Each point re-runs the full pass-2 search (ascending-k scan) at the two gain settings that
bracket the nominal optimum, keeping the best -- so a parameter change is allowed to move the
controller's operating point, same as in rail_bound.py. Reference path/reference shape stay
nominal (the k search absorbs uniform time rescaling).

Writes results/rail_sensitivity.json + results/rail_sensitivity.png.
Run:  .venv/Scripts/python.exe leg2d/rail_sensitivity.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import rail_bound as rb  # noqa: E402

OUT_JSON = rb.RESULTS / "rail_sensitivity.json"
OUT_PNG = rb.RESULTS / "rail_sensitivity.png"

SCALES = [0.7, 1.0, 1.3]
GAINS = [(1000.0, 12.0), (4000.0, 35.0)]
PARAMS = [
    ("leg inertia", "leg_mass_scale", "#2a78d6"),
    ("rotor inertia", "armature_scale", "#eb6834"),
    ("torque limit", "peak_scale", "#1baf7a"),
    ("bus voltage (no-load speed)", "w0_scale", "#eda100"),
]


def best_v(p1, base_h, rig_kwargs):
    best = None
    for kp, kv in GAINS:
        # fine k-grid (~3% steps after refine): the default 24-point grid quantizes v in ~13%
        # steps, which buried every <=10% sensitivity under grid noise on the first attempt
        r = rb.pass2_tracked(p1, base_h, delay_ms=7, kp=kp, kv=kv, k_max=20.0,
                             rig_kwargs=rig_kwargs, n_k=48, refines=(0.97, 0.94))
        if r and (best is None or r["v_mean"] > best["v_mean"]):
            best = r
    return best


def main():
    if "--replot" in sys.argv and OUT_JSON.exists():
        plot(json.loads(OUT_JSON.read_text()))
        return
    rec, arc_p, p1, base_h = rb.prepare()
    base = best_v(p1, base_h, None)
    v0 = base["v_mean"]
    print(f"[sens] baseline v = {v0:.2f} m/s")

    out = {"baseline_v": v0, "scales": SCALES, "params": {}}
    for label, key, _c in PARAMS:
        row = {}
        for s in SCALES:
            if s == 1.0:
                row[s] = v0
                continue
            r = best_v(p1, base_h, {key: s})
            row[s] = r["v_mean"] if r else 0.0
            print(f"[sens] {label} x{s}: v = {row[s]:.2f} m/s ({row[s] / v0 * 100:.0f}%)")
        # directional elasticities d(ln v)/d(ln p) from the secants
        e_dn = (np.log(row[0.7] / v0) / np.log(0.7)) if row[0.7] > 0 else float("nan")
        e_up = (np.log(row[1.3] / v0) / np.log(1.3)) if row[1.3] > 0 else float("nan")
        out["params"][label] = dict(key=key, v={str(k): v for k, v in row.items()},
                                    elasticity_down=e_dn, elasticity_up=e_up)
        print(f"[sens] {label}: elasticity down {e_dn:+.2f} / up {e_up:+.2f}")

    OUT_JSON.write_text(json.dumps(out, indent=2))
    plot(out)
    print(f"wrote {OUT_JSON}")


def plot(out):
    v0 = out["baseline_v"]
    scales = out["scales"]
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=130)
    for label, key, c in PARAMS:
        row = out["params"][label]["v"]
        vv = np.array([row[str(s)] for s in scales]) / v0 * 100.0
        ax.plot(scales, vv, "-o", color=c, lw=2.0, ms=7, mec="white", mew=1.2)
        dy = {"leg inertia": -3.0, "rotor inertia": 3.0}.get(label, 0.0)
        ax.annotate(label, (scales[-1], vv[-1] + dy), xytext=(8, 0),
                    textcoords="offset points", color=c, fontsize=10.5, va="center")
    ax.axhline(100, color="#8a8f98", lw=1.0, ls="--")
    ax.axvline(1.0, color="#8a8f98", lw=0.8, ls=":")
    ax.set_xlabel("parameter scale (x nominal)")
    ax.set_ylabel(f"speed bound  [% of baseline {v0:.2f} m/s]")
    ax.set_title("Sensitivity of the rail-sweep bound (envelope + 7 ms MIT delay)", fontsize=12)
    ax.grid(True, lw=0.5, alpha=0.35)
    ax.set_xlim(0.65, 1.62)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
