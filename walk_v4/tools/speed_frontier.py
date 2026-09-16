"""What does the stick actually buy? A fine command ladder, closed loop, greedy.

`v_max` is the number that gives the operator's stick its meaning: `task[0] = v_cmd / v_max`, so a
full stick asks for `v_max` m/s and the whole contract ("50% of stick = 50% of top speed") is
measured against it. It was set to 3.6 from `tools/speed_lib.py`, which is an OPEN-LOOP CEM search
over gait specs -- it found specs that produce 0.5-3.5 m/s when played into the plant, which is a
statement about the action space, not about any policy's closed-loop capability.

Measured 2026-09-13 on the stage-2 keeper, the closed-loop picture is different: 100% upright at
every command up to 2.70 m/s, 0% at 3.60, and a systematic ~20% undershoot over the top half. If
the top of the stick asks for a speed the policy cannot hold, the top of the stick is a lie and the
tracking check fails there by construction.

This prints the closed-loop frontier on a grid fine enough to read the knee off it:

    python walk_v4/tools/speed_frontier.py --run walk_v4/runs/v3_q_s5 --step 0.3

and reports, per commanded speed, the achieved body-frame speed, the fraction still upright, and the
heading drift -- plus the largest command that clears a survival and a tracking bar, which is the
honest v_max.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np

from evaluate import load_run
from verify import ladder_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--step", type=float, default=0.3, help="ladder spacing, m/s")
    ap.add_argument("--n-per", type=int, default=8, help="envs per rung")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--upright-bar", type=float, default=0.90)
    ap.add_argument("--err-bar", type=float, default=0.15, help="fraction of the COMMAND")
    ap.add_argument("--dr", action="store_true", help="draw the plant from the DR ranges")
    ap.add_argument("--train-env", action="store_true",
                    help="leave the weather ON (obs noise, pushes, wind, trips, hot thermal start) "
                         "-- what a TRAINING rollout sees, not what the eval env shows")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample the action at the policy's own std instead of taking the mean")
    ap.add_argument("--var-floor", type=float, default=None,
                    help="override warmstart_var_floor for --warm-start (0 = no floor)")
    ap.add_argument("--count-cap", type=float, default=None,
                    help="override warmstart_obs_count_cap for --warm-start (0 = no cap)")
    ap.add_argument("--warm-start", action="store_true",
                    help="load through the SAME obs-stat surgery a warm start applies "
                         "(warmstart_var_floor, warmstart_obs_count_cap) -- i.e. measure the "
                         "policy the next stage actually inherits, not the one that was saved")
    args = ap.parse_args()

    run = Path(args.run)
    ckpt = Path(args.checkpoint) if args.checkpoint else run / "best.msgpack"
    # Apply the surgery HERE rather than through load_run's warm_start flag, so the floor and the
    # cap can be swept. They are the two things a warm start does to the obs statistics, and
    # measured 2026-09-13 they are not free: the default floor of 0.01 pins 59 of 412 dims and
    # takes the stage-2 keeper from 100% upright at 1.80 m/s to 0% at every command.
    cfg, _env, agent = load_run(run, ckpt, warm_start=False)
    if args.warm_start:
        import jax.numpy as jnp
        cap = cfg.warmstart_obs_count_cap if args.count_cap is None else args.count_cap
        floor = cfg.warmstart_var_floor if args.var_floor is None else args.var_floor
        if cap > 0:
            agent.stats = agent.stats.replace(count=jnp.minimum(agent.stats.count, cap))
        if floor > 0:
            agent.stats = agent.stats.replace(var=jnp.maximum(agent.stats.var, floor))
        v = np.asarray(agent.stats.var)
        print(f"[warm] obs-stat surgery: cap={cap:,.0f} floor={floor}  ->  "
              f"count={float(agent.stats.count):,.0f}, "
              f"{int((v <= floor + 1e-12).sum()) if floor > 0 else 0}/{v.size} dims on the floor")

    # the ladder is a FRACTION of v_max inside command_ladder, so build the fractions we want
    rungs = np.arange(args.step, float(cfg.v_max) + 1e-9, args.step) / float(cfg.v_max)
    rungs = np.concatenate([[0.0], rungs])
    c = replace(cfg, eval_ladder=tuple(float(r) for r in rungs))
    n_envs = int(len(rungs) * args.n_per)
    rows = ladder_eval(c, agent, n_envs, args.seconds, dr=args.dr, bringup=False, seed=11,
                       quiet=not args.train_env, greedy=not args.stochastic)

    print(f"\n[frontier] {run.name}  {ckpt.name}  v_max={cfg.v_max:.2f}  "
          f"plant={'DR' if args.dr else 'nominal'}  "
          f"{'TRAINING env (weather on)' if args.train_env else 'eval env (quiet)'}  "
          f"{'stochastic' if args.stochastic else 'greedy'}  "
          f"{args.n_per} envs/rung  {args.seconds:.0f}s")
    print(f"{'commanded':>10} {'achieved':>9} {'err':>7} {'err/cmd':>8} {'upright':>8} {'heading':>8}")
    ok = []
    for r in rows:
        frac = r["err"] / r["cmd"] if r["cmd"] > 1e-6 else float("nan")
        print(f"{r['cmd']:>10.2f} {r['speed']:>9.2f} {r['err']:>7.2f} "
              f"{'' if np.isnan(frac) else f'{frac * 100:>7.0f}%'}"
              f"{'      -' if np.isnan(frac) else ''} "
              f"{r['upright'] * 100:>7.0f}% {r['yaw_deg']:>7.1f}d")
        if r["upright"] >= args.upright_bar and (np.isnan(frac) or frac <= args.err_bar):
            ok.append(r["cmd"])

    print()
    if ok:
        print(f"largest command holding {args.upright_bar * 100:.0f}% upright AND "
              f"within {args.err_bar * 100:.0f}% of the command: {max(ok):.2f} m/s")
        print(f"  -> an honest v_max for this policy is {max(ok):.2f}, not {cfg.v_max:.2f}")
    else:
        print(f"NO command clears both bars ({args.upright_bar * 100:.0f}% upright, "
              f"{args.err_bar * 100:.0f}% error). The stick does not track anywhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
