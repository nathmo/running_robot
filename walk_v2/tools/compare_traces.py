"""Compare two v2 traces (walk_v2/tools/trace.py format) from two implementations.

    python walk_v2/tools/compare_traces.py trace_mjx.json trace_cpu.json

Reports:
  1. FUNCTIONAL agreement on shared states: for every tick of trace B, recompute the gait
     targets / kp / kd from B's own (spec, phase, state) with walk_v2/gait.py in numpy and diff
     against what B recorded — and the same for A. Both must match their own recomputation
     (proves each implementation's control law IS the v2 law), then A's and B's laws are diffed
     on B's states.
  2. TRAJECTORY divergence: max |qpos_A - qpos_B| per tick, the first tick above 1e-3 / 1e-2 rad,
     base height and reward-term correlation over the overlap.
"""
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np

import gait


Q_LO = np.array([-0.785, -1.5, -1.047, -0.785, -1.5, -1.047])
Q_HI = -Q_LO
VEL_LIMIT = np.array([10.30, 22.01, 22.01, 10.30, 22.01, 22.01])


def recompute(tr, row, prev_row, prev_target, prate):
    """The v2 control law on the recorded state: gait.assemble, joint clip, no-load slew limit.
    `prate` is the EMA-filtered pitch rate the fixed reflex reads (alpha = pitch_reflex_rate_lp,
    state carried by the caller, 0 at reset)."""
    gpd = {k: (tuple(v) if isinstance(v, list) else v) for k, v in tr["gait_params"].items()}
    alpha = float(gpd.pop("pitch_reflex_rate_lp", 0.9))
    gp = gait.GaitParams(**gpd)
    if tr.get("rig"):
        gp = gp._replace(pitch_clip=0.0)
    spec = np.array(row["spec"])
    grav = np.array(prev_row["grav"]) if prev_row else np.array([0, 0, -1.0])
    gyro = np.array(prev_row["gyro"]) if prev_row else np.zeros(3)
    phi = prev_row["phase"] if prev_row else 0.0
    t = row["t"] - tr["control_dt"]
    res = 0.05 * np.sin(2 * np.pi * 3.0 * t + np.arange(6))
    prate = alpha * prate + (1.0 - alpha) * gyro[1] if alpha > 0.0 else gyro[1]
    target, kp, kd, _ = gait.assemble(spec, res, phi, grav[1], gyro[0], grav[0], prate,
                                      np.array(tr["nominal_ctrl"]), gp, xp=np)
    target = np.clip(target, Q_LO, Q_HI)
    if np.max(np.abs(row["kp"])) < 20.0:          # the CPU arm records the gain MULTIPLIERS
        kp = kp / np.array(gp.drive_kp)
        kd = kd / np.array(gp.drive_kd)
    dt = tr["control_dt"]
    v = np.clip((target - prev_target) / dt, -VEL_LIMIT, VEL_LIMIT)
    return prev_target + v * dt, kp, kd, prate


def self_consistency(tr, name):
    errs = []
    prev = None
    prev_target = np.array(tr["nominal_ctrl"])
    prate = 0.0
    for row in tr["rows"]:
        if row.get("done"):
            break                     # the done row holds the auto-reset state, not this tick's command
        target, kp, kd, prate = recompute(tr, row, prev, prev_target, prate)
        errs.append((np.abs(np.array(row["kp"]) - kp).max(), np.abs(np.array(row["kd"]) - kd).max(),
                     np.abs(np.array(row["target"]) - target).max()))
        prev = row
        prev_target = np.array(row["target"])
    e = np.array(errs)
    print(f"  {name}: recomputed-vs-recorded (max over ticks)  kp {e[:, 0].max():.2e}  kd {e[:, 1].max():.2e}  "
          f"target {e[:, 2].max():.2e}")


def main():
    a = json.loads(Path(sys.argv[1]).read_text())
    b = json.loads(Path(sys.argv[2]).read_text())
    print(f"A: {a['impl']}  {a['ticks']} ticks   B: {b['impl']}  {b['ticks']} ticks")
    print("1. functional agreement (the control law)")
    self_consistency(a, "A")
    self_consistency(b, "B")
    n = min(a["ticks"], b["ticks"])
    print("2. trajectory divergence")
    qa = np.array([r["qpos"] for r in a["rows"][:n]])
    qb = np.array([r["qpos"] for r in b["rows"][:n]])
    d = np.abs(qa - qb).max(1)
    for thr in (1e-4, 1e-3, 1e-2, 1e-1):
        i = np.flatnonzero(d > thr)
        print(f"   first tick with |dq| > {thr:g}: {int(i[0]) if i.size else 'never'} of {n}")
    za = np.array([r["base_z"] for r in a["rows"][:n]])
    zb = np.array([r["base_z"] for r in b["rows"][:n]])
    print(f"   base z max diff {np.abs(za - zb).max():.4f} m; ended A {'fall' if a['rows'][-1]['done'] else 'cap'} "
          f"at {a['ticks']} / B {'fall' if b['rows'][-1]['done'] else 'cap'} at {b['ticks']}")
    ra = np.array([r["reward"] for r in a["rows"][:n]])
    rb = np.array([r["reward"] for r in b["rows"][:n]])
    k = min(n, 30)
    print(f"   reward: first-{k}-tick max diff {np.abs(ra[:k] - rb[:k]).max():.4f}, corr over overlap "
          f"{np.corrcoef(ra, rb)[0, 1] if n > 2 else float('nan'):.3f}")
    terms = sorted(set(a["rows"][0]["terms"]) & set(b["rows"][0]["terms"]))
    print(f"   shared reward terms: {len(terms)}; A-only {sorted(set(a['rows'][0]['terms']) - set(terms))}; "
          f"B-only {sorted(set(b['rows'][0]['terms']) - set(terms))}")
    for t in terms:
        ta = np.array([r["terms"][t] for r in a["rows"][:k]])
        tb = np.array([r["terms"][t] for r in b["rows"][:k]])
        dd = np.abs(ta - tb).max()
        if dd > 1e-3:
            print(f"     {t:16s} first-{k} max diff {dd:.4f}")


if __name__ == "__main__":
    main()
