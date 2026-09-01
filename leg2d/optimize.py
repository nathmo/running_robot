"""Find the maximum-speed gait on the leg2d x-only rig by real (derivative-free) optimization over
gait parameters, then break down what actually limited it.

Global search (differential evolution) over (cadence, duty, stride, clearance) to maximize measured
forward speed, polished with a local Nelder-Mead step, then one detailed diagnostic rollout at the
optimum to report which constraint bound: torque saturation, the motor's power corner, or the
thermal (continuous-torque) rating.

Usage
  .venv/Scripts/python.exe leg2d/optimize.py                 # full run
  .venv/Scripts/python.exe leg2d/optimize.py --quick          # small popsize/iters, fast smoke test
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

import motor
import sim

PKG = Path(__file__).resolve().parent

# (f_hz, duty, stride, clearance, z_off). z_off's lower bound is well past dash01's own
# double-support -0.13 m (see GAIN_SPACE in training/scripted_walk.py) since single support here
# carries ~2x the load -> more droop to compensate for (see gait.py's z_off docstring).
BOUNDS = [(0.3, 10.0), (0.15, 0.95), (0.02, 0.60), (0.0, 0.10), (-0.25, 0.02)]


def make_objective(model, data, ids, lut, ik0, nominal_cam, nominal_thigh, stand_z, log=None):
    def objective(x):
        f_hz, duty, stride, clearance, z_off = x
        r = sim.evaluate(model, data, ids, lut, ik0, nominal_cam, nominal_thigh, stand_z,
                         f_hz, duty, stride, clearance, z_off)
        if log is not None:
            log.append(r)
        # differential_evolution MINIMIZES -> maximize v_measured by minimizing its negative.
        # sim.evaluate already zeroes v_measured for non-finite/collapsed/flew-off rollouts, so
        # there is nothing extra to penalize here -- a bad gait is just worth 0, same as standing
        # still, and the optimizer moves away from it on its own.
        return -r["v_measured"]
    return objective


def run(quick=False, seed=0):
    model, data, ids = sim.build_sim()
    lut, ik0 = sim.load_lut()
    nominal_cam, nominal_thigh, stand_z = sim.nominal_pose(model, data, ids)

    log = []
    objective = make_objective(model, data, ids, lut, ik0, nominal_cam, nominal_thigh, stand_z, log)

    popsize = 6 if quick else 15
    maxiter = 8 if quick else 40
    t0 = time.time()
    print(f"[optimize] differential_evolution: popsize={popsize} maxiter={maxiter} "
          f"(~{popsize * len(BOUNDS) * maxiter} evaluations)")

    def cb(intermediate_result):
        best_v = max((r["v_measured"] for r in log), default=0.0)
        gen = len(log) // (popsize * len(BOUNDS))
        print(f"  gen {gen:3d}  evals={len(log):5d}  best so far v={best_v:.3f} m/s  "
              f"({time.time()-t0:.0f}s)")

    result = differential_evolution(
        objective, BOUNDS, popsize=popsize, maxiter=maxiter, seed=seed,
        polish=False, tol=1e-3, mutation=(0.4, 1.2), recombination=0.7, workers=1,
        updating="deferred", callback=cb,
    )
    print(f"\n[optimize] DE done: {len(log)} evaluations, {time.time()-t0:.0f}s  "
          f"best v={-result.fun:.4f} m/s at f={result.x[0]:.3f} duty={result.x[1]:.3f} "
          f"stride={result.x[2]:.3f} clearance={result.x[3]:.3f} z_off={result.x[4]:.3f}")

    # local polish
    local = minimize(objective, result.x, method="Nelder-Mead",
                     bounds=BOUNDS, options=dict(xatol=1e-3, fatol=1e-4, maxiter=300))
    x_best = local.x if -local.fun > -result.fun else result.x
    v_best = max(-local.fun, -result.fun)
    print(f"[optimize] Nelder-Mead polish: v={-local.fun:.4f} m/s  "
          f"-> final best v={v_best:.4f} m/s at "
          f"f={x_best[0]:.3f}Hz duty={x_best[1]:.3f} stride={x_best[2]:.3f} "
          f"clearance={x_best[3]:.3f} z_off={x_best[4]:.3f}")

    # ---- limiting-factor breakdown at the optimum ----
    f_hz, duty, stride, clearance, z_off = x_best
    detail = sim.evaluate(model, data, ids, lut, ik0, nominal_cam, nominal_thigh, stand_z,
                          f_hz, duty, stride, clearance, z_off, record=True)
    tr = detail["trace"]
    report = limiting_factor(tr, detail)
    print("\n" + "=" * 78)
    print("LIMITING FACTOR at the optimum")
    print("=" * 78)
    for line in report["lines"]:
        print(" ", line)
    print("=" * 78)

    out = dict(x_best=list(x_best), v_best=v_best, detail={k: v for k, v in detail.items() if k != "trace"},
              report=report, n_evaluations=len(log))
    out_path = PKG / "results" / "optimize.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return out


def limiting_factor(tr, detail):
    """Classify what bound the optimal gait: torque saturation (the clamp actually cut the
    command), proximity to the motor's peak-power corner, or the thermal continuous rating."""
    tau = np.concatenate([np.abs(tr["tau_cam"]), np.abs(tr["tau_thigh"])])
    w = np.concatenate([np.abs(tr["w_cam"]), np.abs(tr["w_thigh"])])
    power = tau * w
    peak_p = motor.peak_power_w()

    near_peak_torque = float(np.mean(tau > 0.95 * motor.PEAK_NM))
    near_no_load_speed = float(np.mean(w > 0.85 * motor.NO_LOAD_RAD_S))
    near_peak_power = float(np.mean(power > 0.85 * peak_p))
    sat_frac = detail["sat_frac"]
    rms_cam, rms_thigh = detail["rms_cam"], detail["rms_thigh"]
    thermal_margin = min(motor.CONT_NM - rms_cam, motor.CONT_NM - rms_thigh)

    lines = [
        f"gait: f={detail['f_hz']:.2f}Hz duty={detail['duty']:.2f} stride={detail['stride']:.3f}m "
        f"clearance={detail['clearance']:.3f}m z_off={detail['z_off']:.3f}m",
        f"achieved speed: {detail['v_measured']:.3f} m/s (commanded target {detail['v_target']:.3f} m/s)",
        f"stance height range: {detail['z_min']:.3f}-{detail['z_max']:.3f} m  "
        f"({'COLLAPSED' if detail['collapsed'] else 'flew off' if detail['flew_off'] else 'stable'})",
        f"torque-speed clamp engaged (saturated) on {sat_frac*100:.1f}% of measured control steps",
        f"time near peak torque ({0.95*motor.PEAK_NM:.0f}+ N*m): {near_peak_torque*100:.1f}%",
        f"time near no-load speed ({0.85*motor.NO_LOAD_RAD_S:.1f}+ rad/s): {near_no_load_speed*100:.1f}%",
        f"time near peak power ({0.85*peak_p:.0f}+ W of {peak_p:.0f} W available): {near_peak_power*100:.1f}%",
        f"RMS torque: cam {rms_cam:.1f} N*m, thigh {rms_thigh:.1f} N*m "
        f"(continuous rating {motor.CONT_NM:.0f} N*m, margin {thermal_margin:+.1f} N*m)",
        "thermally SUSTAINABLE" if detail["thermal_ok"] else
        "BURST ONLY -- exceeds the continuous torque rating, would need to slow down to sustain",
    ]
    verdict = ("torque-limited" if near_peak_torque > 0.3 else
              "power-limited" if near_peak_power > 0.3 else
              "thermally-limited" if not detail["thermal_ok"] else
              "NOT motor-limited -- something else (gait kinematics / contact) caps the speed")
    lines.append(f"verdict: {verdict}")
    return dict(lines=lines, verdict=verdict, sat_frac=sat_frac, near_peak_torque=near_peak_torque,
               near_peak_power=near_peak_power, thermal_margin=thermal_margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(quick=args.quick, seed=args.seed)


if __name__ == "__main__":
    main()
