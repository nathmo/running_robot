"""Diagnose the LATERAL (roll) balance channel of a trained policy.

Written to answer why the CPG ladder walls at m5 (the roll unlock) while the Fourier ladder clears
it, when both generators drive hip_roll through the *identical* code: a learned linear reflex
u = kp*roll + kd*roll_rate + bias on mirrored hip_roll targets, plus the per-step residual.

Since the channel is shared, the question is not "which formula" but whether the policy is
AUTHORITY-limited on it (pushing the reflex gains and the hip_roll residual against their limits) or
simply not using it (leaving authority on the table, i.e. a learning/expressiveness problem).

    python training/diagnose_roll.py --run training/runs/m5_CPG_60M --episodes 4

Reports, over the rollout: base roll magnitude; the reflex output u and each of its three terms; the
fraction of the available reflex/residual authority actually used; and how often the raw action dims
sit against +-1 (saturated). Works for both action modes — it decodes with the run's own generator.
"""
import argparse
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np

import fourier_gait
import cpg_gait
from evaluate import build

HIP_ROLL_L = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    model, venv, raw = build(run, None, args.checkpoint)
    c = raw.cfg
    cpg = (c.action_mode == "cpg")

    rec = {k: [] for k in ("roll", "roll_rate", "u", "p", "d", "b", "res_hr", "gk", "gd", "gb")}

    def hook():
        a = raw._prev_action                       # the policy's raw [-1,1] output this step
        grav, angv = raw._gravity_body(), raw._ang_vel_body()
        roll, roll_rate = float(grav[1]), float(angv[0])
        if cpg:
            _, _, _, reflex, _, res = cpg_gait.decode(a, raw.n_steer, c.cpg_residual)
        else:
            _, _, _, reflex, _, res = fourier_gait.decode(a, c.n_harmonics, raw.n_steer)
        p = c.reflex_kp_scale * reflex[0] * roll
        d = c.reflex_kd_scale * reflex[1] * roll_rate
        b = c.reflex_bias_scale * reflex[2]
        for k, v in (("roll", roll), ("roll_rate", roll_rate), ("u", p + d + b),
                     ("p", p), ("d", d), ("b", b),
                     ("res_hr", float(res[HIP_ROLL_L])),
                     ("gk", float(reflex[0])), ("gd", float(reflex[1])), ("gb", float(reflex[2]))):
            rec[k].append(v)
    raw.on_control_step = hook

    for _ in range(args.episodes):
        obs = venv.reset()
        done = [False]
        while not done[0]:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = venv.step(a)

    R = {k: np.asarray(v) for k, v in rec.items()}
    n = len(R["roll"])
    print(f"\n=== lateral / roll channel: {run.name}  ({n} control steps, mode={c.action_mode}) ===")
    print(f"base roll (grav_y):     mean|.| {np.abs(R['roll']).mean():.4f}   p95 {np.percentile(np.abs(R['roll']),95):.4f}   max {np.abs(R['roll']).max():.4f}")
    print(f"roll rate (gyro x):     mean|.| {np.abs(R['roll_rate']).mean():.3f}    p95 {np.percentile(np.abs(R['roll_rate']),95):.3f}    max {np.abs(R['roll_rate']).max():.3f} rad/s")

    print("\nreflex output u = kp*roll + kd*roll_rate + bias   (rad of mirrored hip_roll)")
    print(f"  u            mean|.| {np.abs(R['u']).mean():.4f}  p95 {np.percentile(np.abs(R['u']),95):.4f}  max {np.abs(R['u']).max():.4f}")
    for k, lab in (("p", "kp*roll   "), ("d", "kd*rollrate"), ("b", "bias      ")):
        print(f"    {lab} mean|.| {np.abs(R[k]).mean():.4f}  share of |u| {np.abs(R[k]).sum()/max(np.abs(R['u']).sum(),1e-9)*100:5.1f}%")

    # AUTHORITY: the gains are policy outputs in [-1,1] scaled by cfg.reflex_*_scale. Compare what
    # the policy asked for against the most it COULD have asked for at the roll it was seeing.
    u_max = (c.reflex_kp_scale * np.abs(R["roll"]) + c.reflex_kd_scale * np.abs(R["roll_rate"])
             + c.reflex_bias_scale)
    used = np.abs(R["u"]) / np.maximum(u_max, 1e-9)
    print(f"\nauthority used: |u| / max possible |u|   mean {used.mean()*100:5.1f}%   p95 {np.percentile(used,95)*100:5.1f}%")
    for k, lab in (("gk", "kp gain"), ("gd", "kd gain"), ("gb", "bias   ")):
        sat = np.mean(np.abs(R[k]) > 0.95) * 100
        print(f"    {lab} raw action: mean {R[k].mean():+.3f}  mean|.| {np.abs(R[k]).mean():.3f}  |a|>0.95 {sat:5.1f}% of steps")

    rs = c.residual_scale
    print(f"\nhip_roll residual (scale {rs} rad):")
    print(f"    raw mean|.| {np.abs(R['res_hr']).mean():.3f}  |a|>0.95 {np.mean(np.abs(R['res_hr'])>0.95)*100:5.1f}% of steps"
          f"   -> {np.abs(R['res_hr']).mean()*rs:.4f} rad mean deviation")

    tot = np.abs(R["u"]).mean() + np.abs(R["res_hr"]).mean() * rs
    print(f"\ntotal lateral command  mean|.| {tot:.4f} rad "
          f"(reflex {np.abs(R['u']).mean()/max(tot,1e-9)*100:.0f}%, residual {np.abs(R['res_hr']).mean()*rs/max(tot,1e-9)*100:.0f}%)")
    print("\nread: gains pinned at |a|>0.95 and high authority-used => AUTHORITY-limited (needs a")
    print("      bigger reflex/residual scale). Low authority-used with unsaturated gains => the")
    print("      policy is NOT using the channel it already has, i.e. a learning problem.")


if __name__ == "__main__":
    main()
