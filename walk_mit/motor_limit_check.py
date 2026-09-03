"""Validate the actuator back-EMF speed/torque-speed model against the datasheets, then
measure how badly the CURRENT policy violates it (the fix run's baseline).

Model (walk_mit/env._apply_motor_torque_speed, per substep):
    tau(w) = min( tau_peak,  Kt_j * max(V_bus - Kt_j*|w_joint|, 0) / R_total )
plus the command-side hard cap: motor_vel_limit slew-limits the COMMANDED joint velocity to the
48 V no-load speed, so the policy cannot even request a speed the motor cannot reach.

Datasheet ground truth (cubemars.com, pulled 2026-09-03):
  AK60-39 V3.0 KV80 (hip-roll): Kt 0.12 Nm/A, R_ll 0.600 ohm, peak 17 A / 72 Nm, no-load 98 rpm
  AKE90-8  KV35 (cam+thigh):    Kt 0.272 Nm/A, R_ll 0.164 ohm, peak 72 A / 170 Nm, no-load 210 rpm
Pack: 48 V, internal resistance 0.065 ohm (user-measured), in series with each motor circuit.

Validity checks:
  1. back-EMF closure: no-load speed predicted by V_bus/Kt_joint must equal the datasheet
     no-load speed (Kt == Ke in SI units — if these disagree the whole model is wrong).
  2. peak-torque reachability: the current needed for the XML's forcerange peak must not
     exceed the datasheet peak current, and the corner speed must be positive.
  3. empirical baseline: greedy rollouts of the current best policy, max and 99th-pct joint
     speed vs the no-load bound, and the fraction of steps the voltage branch would clamp.

Usage:  python walk_mit/motor_limit_check.py --run walk_mit/runs/sprint_m6_mit_s0 [--episodes 3]
"""
import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np

V_BUS = 48.0
R_PACK = 0.065
# model actuator order: hip_roll_L, cam_L, thigh_L, hip_roll_R, cam_R, thigh_R
NAMES = ["hip_roll_L", "cam_L", "thigh_L", "hip_roll_R", "cam_R", "thigh_R"]
KT_J = np.array([4.655, 2.176, 2.176, 4.655, 2.176, 2.176])
GEAR = np.array([39.0, 8.0, 8.0, 39.0, 8.0, 8.0])
R_LL = np.array([0.600, 0.164, 0.164, 0.600, 0.164, 0.164])
R_TOT = R_LL + R_PACK
DATASHEET_NOLOAD_RPM_MOTOR = np.array([98 * 39, 210 * 8, 210 * 8, 98 * 39, 210 * 8, 210 * 8])
DATASHEET_PEAK_A = np.array([17.0, 72.0, 72.0, 17.0, 72.0, 72.0])


def static_checks():
    print("== 1. back-EMF closure (no-load speed from V/Kt vs datasheet)")
    w_noload_j = V_BUS / KT_J                                   # rad/s at the joint, I=0
    w_ds_j = DATASHEET_NOLOAD_RPM_MOTOR * 2 * np.pi / 60 / GEAR
    for i, n in enumerate(NAMES[:3]):
        err = 100 * (w_noload_j[i] - w_ds_j[i]) / w_ds_j[i]
        print(f"  {n:10s}  model {w_noload_j[i]:6.2f} rad/s   datasheet {w_ds_j[i]:6.2f}   "
              f"err {err:+.2f}%")
    ok = np.allclose(w_noload_j, w_ds_j, rtol=0.01)
    print(f"  -> {'PASS' if ok else 'FAIL'} (Kt==Ke closure within 1%)")

    print("== 2. envelope shape with R_total = R_ll + pack 0.065 ohm")
    import config as cfgmod
    c = cfgmod.get_config("sprint_m6_mit")
    from env import DashEnv
    e = DashEnv(c)
    tau_peak = e.model.actuator_forcerange[:6, 1].copy()
    i_peak_needed = tau_peak / KT_J                              # q-axis A for XML peak torque
    w_corner = (V_BUS - i_peak_needed * R_TOT) / KT_J            # joint rad/s where derate starts
    for i, n in enumerate(NAMES):
        print(f"  {n:10s}  tau_peak(XML) {tau_peak[i]:6.1f} Nm  needs {i_peak_needed[i]:5.1f} A "
              f"(datasheet peak {DATASHEET_PEAK_A[i]:.0f} A)  corner {w_corner[i]:5.2f} rad/s  "
              f"no-load {V_BUS/KT_J[i]:5.2f}")
    ok2 = bool(np.all(i_peak_needed <= DATASHEET_PEAK_A * 1.05) and np.all(w_corner > 0))
    print(f"  -> {'PASS' if ok2 else 'FAIL'} (XML peak reachable at datasheet current, corner > 0)")
    return ok and ok2


def rollout_speeds(run, episodes, with_limits):
    """Greedy rollouts; return per-joint |qvel| samples. with_limits mutates the loaded config."""
    from evaluate import build
    import json
    run = Path(run)
    # build() reads resolved_config.json; patch AFTER construction via a fresh env if limiting
    model, venv, raw = build(run, None, None)
    if with_limits:
        # rebuild the raw env with the limits on, keeping the same policy/vecnorm
        import config as cfgmod
        d = json.loads((run / "resolved_config.json").read_text())
        c = cfgmod.config_from_dict(d.get("config", d))
        c.push_interval_s = 0.0
        c.motor_vel_limit = tuple(V_BUS / KT_J)          # no-load hard cap, command side
        c.motor_r_ohm = tuple(R_TOT)                     # back-EMF torque-speed clamp
        from env import DashEnv
        import evaluate as ev
        new_raw = DashEnv(c)
        # transplant curriculum restores the same way build() did
        cur = run / "curriculum.json"
        if cur.exists():
            dd = json.loads(cur.read_text())
            if "stance_ratio" in dd:
                new_raw.set_stance_ratio(dd["stance_ratio"])
            if "eff_scale" in dd:
                new_raw.set_efficiency_scale(dd["eff_scale"])
            if "sprint_dist_m" in dd:
                new_raw.set_sprint_dist(dd["sprint_dist_m"])
            if "torque_scale" in dd:
                new_raw.set_torque_limit(dd["torque_scale"])
            if "drive_bw_log10" in dd:
                new_raw.set_drive_bandwidth_log10(dd["drive_bw_log10"])
        venv.venv.envs[0] = new_raw
        raw = new_raw
    speeds, dists, steps_ep = [], [], []
    for _ in range(episodes):
        obs = venv.reset()
        done = False
        x0 = float(raw.data.qpos[0])
        n = 0
        while not done and n < 12000:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, dones, info = venv.step(act)
            done = bool(dones[0])
            speeds.append(np.abs(raw.data.qvel[raw.act_dadr]).copy())
            n += 1
        dists.append(float(raw.data.qpos[0]) - x0)
        steps_ep.append(n)
    return np.array(speeds), dists, steps_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="walk_mit/runs/sprint_m6_mit_s0")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    ok = static_checks()

    print(f"== 3. empirical baseline: {args.run}, {args.episodes} greedy episodes, NO limits")
    w_free = V_BUS / KT_J
    s, d, n = rollout_speeds(args.run, args.episodes, with_limits=False)
    for i, nm in enumerate(NAMES):
        frac_over = float(np.mean(s[:, i] > w_free[i]))
        frac_derate = float(np.mean(s[:, i] > (V_BUS - (KT_J[i] / R_TOT[i]) * 0 - 0) / KT_J[i]))
        p99 = float(np.percentile(s[:, i], 99))
        print(f"  {nm:10s}  max {s[:, i].max():6.2f}  p99 {p99:6.2f}  no-load {w_free[i]:5.2f}  "
              f"over-no-load {100*frac_over:5.1f}% of steps")
    print(f"  dists {['%.1f' % x for x in d]} m   steps {n}")

    print(f"== 4. same policy WITH limits (vel cap + torque-speed clamp), unretrained")
    s2, d2, n2 = rollout_speeds(args.run, args.episodes, with_limits=True)
    for i, nm in enumerate(NAMES):
        frac_over = float(np.mean(s2[:, i] > w_free[i] * 1.02))
        print(f"  {nm:10s}  max {s2[:, i].max():6.2f}  p99 {float(np.percentile(s2[:, i], 99)):6.2f}  "
              f"over-no-load {100*frac_over:5.1f}%")
    print(f"  dists {['%.1f' % x for x in d2]} m   steps {n2}")
    print("PASS" if ok else "STATIC CHECKS FAILED")


if __name__ == "__main__":
    main()
