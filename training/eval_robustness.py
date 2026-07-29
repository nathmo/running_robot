"""Score a policy across a grid of plant perturbations — the sim2real readiness metric.

  python training/eval_robustness.py --run training/runs/teleop --episodes 4
  python training/eval_robustness.py --run training/runs/teleop --axis ankle_k --episodes 8
  python training/eval_robustness.py --run training/runs/teleop --json out.json

`ep_len` on the nominal plant tells you nothing about transfer: the nominal plant is the one thing
the real robot is guaranteed not to be. What matters is the SHAPE of the performance surface over
the parameters you do not know — where it falls off a cliff, and whether the as-built value sits
near an edge. Each axis is swept with everything else held at nominal, so a failure is attributable.

Reported per operating point:
  survival   fraction of episodes that reached the time limit without falling
  track MAE  mean |v - v_cmd| over moving commands (command objective only)
  stand      mean |v| while the stand command is held (command objective only)

The ankle-spring axis is the one to read first. The whole m3 balance result rests on k=350, that
spring is a physical part that will not be built to spec, and the m7 work showed its DAMPING sets
whether the leg rings at 6 Hz. If survival collapses at 0.8x or 1.2x k, the demo depends on a
component tolerance nobody is holding.
"""
import argparse
import json
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np

from evaluate import build

# axis -> (label, multipliers or absolute values)
AXES = {
    "mass":      ("total mass x", [0.80, 0.90, 1.00, 1.10, 1.20, 1.35]),
    "inertia":   ("link inertia x", [0.6, 0.8, 1.0, 1.25, 1.6]),
    "friction":  ("floor friction", [0.30, 0.50, 0.70, 1.00, 1.30]),
    "ankle_k":   ("ankle spring k x", [0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]),
    "ankle_c":   ("ankle damping x", [0.3, 0.6, 1.0, 1.5, 2.5]),
    "kp":        ("servo kp x", [0.7, 0.85, 1.0, 1.2, 1.4]),
    "torque":    ("torque limit x", [0.6, 0.75, 0.9, 1.0]),
    "delay":     ("action delay (steps)", [2, 3, 4, 5, 6, 8]),
    "slope":     ("ground slope (deg)", [0.0, 2.0, 4.0, 6.0, 8.0]),
    "com":       ("CoM offset (m)", [0.0, 0.01, 0.02, 0.04]),
}


def apply_point(raw, axis, val):
    """Set ONE parameter to an exact value on top of the nominal plant. Domain randomization is
    disabled first, so the measured point is the point and not the point plus a random draw."""
    dr = raw._dr
    dr.restore(raw.model)
    m = raw.model
    if axis == "mass":
        m.body_mass[:] = dr.n_mass * val
        m.body_inertia[:] = dr.n_inertia * val
    elif axis == "inertia":
        m.body_inertia[:] = dr.n_inertia * val
    elif axis == "friction":
        m.geom_friction[:, 0] = val
    elif axis == "ankle_k":
        for j in dr._ankle_j:
            qadr = int(dr._jnt_qposadr[j])
            k_old = float(dr.n_jnt_stiffness[j])
            k_new = k_old * val
            ref_old = float(dr.n_qpos_spring[qadr])
            q_stand = float(dr.stand_qpos[qadr])
            m.qpos_spring[qadr] = q_stand - (k_old / k_new) * (q_stand - ref_old)
            m.jnt_stiffness[j] = k_new
    elif axis == "ankle_c":
        for j in dr._ankle_j:
            d = int(dr._jnt_dofadr[j])
            m.dof_damping[d] = dr.n_dof_damping[d] * val
    elif axis == "kp":
        m.actuator_gainprm[:, 0] = dr.n_gainprm[:, 0] * val
        m.actuator_biasprm[:, 1] = dr.n_biasprm[:, 1] * val
    elif axis == "torque":
        m.actuator_forcerange[:] = dr.n_forcerange * val
    elif axis == "delay":
        # with DR disabled the randomizer echoes cfg.action_delay_steps straight back, so setting
        # the config value IS setting the delay for the next reset
        raw.cfg.action_delay_steps = int(val)
    elif axis == "slope":
        g = float(np.linalg.norm(dr.n_gravity))
        tilt = np.deg2rad(val)
        m.opt.gravity[:] = g * np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    elif axis == "com":
        m.body_ipos[:] = dr.n_ipos + np.array([val, 0.0, 0.0])


def run_point(model_, venv, raw, episodes, max_steps, rng):
    """Roll out `episodes` from this operating point. Commands are drawn from a FIXED sequence so
    every point is graded on the same task — otherwise a lucky draw of easy commands reads as
    robustness."""
    surv, errs, stands = 0, [], []
    for ep in range(episodes):
        obs = venv.reset()
        if raw.command_mode:
            # the same command schedule at every point: stand, forward, turn, back
            sched = [(0.0, 0.0), (0.75, 0.0), (0.5, 0.6), (0.0, 0.0), (-0.5, 0.0), (0.9, 0.0)]
            seg = max(1, max_steps // len(sched))
        fell = False
        for n in range(max_steps):
            if raw.command_mode:
                f, w = sched[min(n // seg, len(sched) - 1)]
                v_fwd, v_back, yaw = raw._cmd_box()
                v = f * (v_fwd if f >= 0 else v_back)
                raw.set_command(v, w * yaw)
            a, _ = model_.predict(obs, deterministic=True)
            obs, r, d, info = venv.step(a)
            if raw.command_mode:
                vx = float(raw._vel_body()[0])
                if abs(raw._v_cmd) > 1e-6:
                    errs.append(abs(vx - raw._v_cmd))
                else:
                    stands.append(abs(vx))
            if d[0]:
                # SB3 VecEnv auto-resets; a real fall is a termination, not the time limit
                fell = not bool(info[0].get("TimeLimit.truncated", False))
                break
        surv += 0 if fell else 1
    return (surv / episodes,
            float(np.mean(errs)) if errs else float("nan"),
            float(np.mean(stands)) if stands else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--axis", default=None, choices=sorted(AXES),
                    help="sweep one axis (default: all)")
    ap.add_argument("--json", default=None, help="write the full grid here")
    args = ap.parse_args()

    run = Path(args.run)
    model_, venv, raw = build(run, args.preset, args.checkpoint)
    # the sweep sets exact operating points, so the per-episode random draw must be off
    raw._dr.enabled = False
    raw.cfg.push_interval_s = 0.0
    raw.cfg.trip_prob = 0.0
    max_steps = int(args.seconds / raw.control_dt)
    rng = np.random.default_rng(0)
    axes = [args.axis] if args.axis else list(AXES)
    nominal_delay = int(raw.cfg.action_delay_steps)

    print(f"[robust] {run.name}: {args.episodes} eps x {args.seconds:.0f}s per point, "
          f"command_mode={raw.command_mode}\n")
    out = {}
    for ax in axes:
        label, vals = AXES[ax]
        print(f"  {label}")
        out[ax] = []
        for v in vals:
            apply_point(raw, ax, v)
            s, e, st = run_point(model_, venv, raw, args.episodes, max_steps, rng)
            bar = "#" * int(round(s * 20))
            extra = ""
            if raw.command_mode and not np.isnan(e):
                extra = f"  track {e:.3f} m/s  stand {st:.3f} m/s"
            print(f"    {v:>7}  survival {s:5.0%} |{bar:<20}|{extra}")
            out[ax].append(dict(value=v, survival=s, track_mae=e, stand=st))
        print()
        raw._dr.restore(raw.model)
        raw.cfg.action_delay_steps = nominal_delay

    # one-line verdict per axis: the widest contiguous band around nominal with full survival
    print("[robust] safe band (contiguous 100%-survival range around nominal):")
    for ax in axes:
        rows = out[ax]
        ok = [r["value"] for r in rows if r["survival"] >= 1.0]
        print(f"  {AXES[ax][0]:<24} {('%s .. %s' % (min(ok), max(ok))) if ok else 'NONE'}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\n[robust] wrote {args.json}")


if __name__ == "__main__":
    main()
