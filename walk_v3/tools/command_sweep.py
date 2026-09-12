"""Does the stick mean what it says? Hold v_cmd at each of several values and measure achieved speed.

This is the deliverable measurement for the joystick policy: "50% stick = 50% top speed". Everything
else (reward budget, episode length, cmd err during training) is a proxy; this is the thing itself.

Three details decide whether the number means anything:

  1. **Hold the command.** The env redraws `v_cmd` on a timer and again on auto-reset, so a rollout that
     merely samples commands measures a moving target. Every tick here re-pins it.
  2. **Settle first.** The policy starts from a reset at whatever speed the previous command left it;
     the interesting quantity is the speed it converges to, not the transient. Reported speed is the
     mean over the final `--settle-frac` of the surviving window.
  3. **Separate tracking from falling.** A fallen robot has speed ~0, which at a 0 m/s command looks
     like perfect tracking. Survivors and speed are reported apart, and the error is computed over
     survivors only.

Speed is world-frame (d/dt of base x), because that is what a corridor test measures with a tape.

    python walk_v3/tools/command_sweep.py --run walk_v3/runs/<run> [--checkpoint ...] \
        [--commands 0,0.25,0.5,0.75,1.0] [--bringup nominal|full]

Exit code 0 always; read the table. `--json` writes the same numbers for a report.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from env import EnvParams
from evaluate import load_run
from ppo import initial_params


def sweep_one(env, agent, params, v_cmd, seed, ticks, settle_frac):
    """Hold v_cmd for a whole rollout; return (speed_mean, survived_frac, n_envs)."""
    state, obs = env.reset(jax.random.PRNGKey(seed), params)
    pin = lambda s: s.replace(v_cmd=jnp.full_like(s.v_cmd, v_cmd),
                              cmd_left=jnp.full_like(s.cmd_left, 1e4))
    state = pin(state)
    x0 = state.data.qpos[:, env.plant.base_q["x"]]

    def step(carry, _):
        state, obs = carry
        a = jnp.clip(agent._act_greedy(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        state2, obs2, _, done, info = env.step(state, a, params)
        state2 = pin(state2)                       # auto-reset would redraw the command
        x = state.data.qpos[:, env.plant.base_q["x"]]
        return (state2, obs2), (x, done, info["fallen"])

    (_, _), (xs, dones, fell) = jax.lax.scan(step, (state, obs), None, length=ticks)
    xs, dones, fell = np.asarray(xs), np.asarray(dones), np.asarray(fell)

    # alive[t] is True while the env has not yet ended; after the first `done` the trajectory is a
    # fresh episode and its x jumps, so every quantity must stop at the first termination.
    alive = np.cumprod(np.vstack([np.ones((1, dones.shape[1]), bool), ~dones[:-1]]), axis=0).astype(bool)
    n_live = alive.sum(0)
    survived = n_live >= ticks
    dt = float(env.control_dt)

    speeds = np.full(dones.shape[1], np.nan)
    for i in range(dones.shape[1]):
        k = int(n_live[i])
        if k < 20:                                  # too short to say anything about a settled speed
            continue
        lo = max(1, int(k * (1.0 - settle_frac)))
        seg = np.concatenate([xs[:k, i], [xs[k - 1, i]]])
        speeds[i] = (seg[k - 1] - seg[lo]) / (dt * max(k - 1 - lo, 1))
    return speeds, survived, np.asarray(fell).max(0), float(n_live.mean() * dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--commands", default="0,0.25,0.5,0.75,1.0", help="stick fractions of v_max")
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--settle-frac", type=float, default=0.5, help="average over this tail fraction")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--bringup", choices=("nominal", "full"), default="nominal",
                    help="nominal = settled keyframe start (measures TRACKING); full = drops and "
                         "tilted releases (measures bring-up, and contaminates the speed average)")
    ap.add_argument("--dr", action="store_true",
                    help="randomised PLANT only (disturbances stay off -- see the note in main)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg, env, agent = load_run(args.run, args.checkpoint, n_envs=args.n_envs, dr=False,
                               warm_start=False)
    if args.dr:
        # NOT load_run(dr=True): that path keeps the training config wholesale, so it adds pushes,
        # wind, trips, a hot thermal start and observation noise on top of the plant draw. Measured
        # 2026-09-12, it kills the fully DR-trained v2 runner in 0.21 s (against 6.00/6.00 s upright
        # on the nominal path) -- i.e. it measures the harness, not the policy. This rebuilds the
        # eval env with the plant draw ON and every disturbance still off, which is what "does
        # tracking survive a different plant?" actually asks.
        from dataclasses import replace
        from env import DashEnvV2
        c = replace(cfg, dr_enable=True, obs_noise_enable=False, push_interval_s=0.0,
                    wind_force_max=0.0, wind_gust_n=0.0, trip_prob=0.0, thermal_hot_start_max=0.0,
                    pitch_assist_kp=0.0, roll_assist_kp=0.0, yaw_assist_kp=0.0)
        env = DashEnvV2(c, n_envs=args.n_envs)
    if cfg.objective != "joystick":
        print(f"[sweep] WARNING: objective is '{cfg.objective}', not 'joystick' -- task[0] is not a "
              f"speed command in this checkpoint, so the sweep is meaningless.")
    base = EnvParams.final(cfg) if args.bringup == "full" else initial_params(cfg)
    params = base._replace(dr_scale=1.0 if args.dr else 0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0,
                           pitch_assist=0.0)
    ticks = int(args.seconds / env.control_dt)
    v_max = float(cfg.v_max)
    fracs = [float(x) for x in args.commands.split(",")]

    print(f"[sweep] {args.run} v_max {v_max:.2f} m/s | bring-up {args.bringup} | plant "
          f"{'DR' if args.dr else 'nominal'} | {args.n_envs} envs x {args.seconds:.0f} s "
          f"(settled over the last {args.settle_frac * 100:.0f}%)")
    print(f"\n{'stick':>7} {'commanded':>10} {'achieved':>10} {'err':>8} {'err %':>7} "
          f"{'upright':>8} {'alive s':>8}")
    rows = []
    for f in fracs:
        v = f * v_max
        speeds, survived, fell, live_s = sweep_one(env, agent, params, v, args.seed, ticks,
                                                   args.settle_frac)
        up = survived & ~fell
        ok = np.isfinite(speeds) & up
        ach = float(np.mean(speeds[ok])) if ok.any() else float("nan")
        err = abs(ach - v) if ok.any() else float("nan")
        pct = (err / v_max) * 100.0 if ok.any() else float("nan")
        print(f"{f * 100:>6.0f}% {v:>10.2f} {ach:>10.2f} {err:>8.2f} {pct:>6.1f}% "
              f"{up.sum():>4}/{len(up):<3} {live_s:>8.1f}")
        rows.append(dict(stick=f, commanded=v, achieved=ach, err=err, err_pct_of_range=pct,
                         upright=int(up.sum()), n=int(len(up)), alive_s=live_s))

    fin = [r for r in rows if np.isfinite(r["err"])]
    worst = max((r["err_pct_of_range"] for r in fin), default=float("nan"))
    print(f"\n[sweep] worst error across the range: {worst:.1f}% of v_max "
          f"({'PASS' if worst <= 15.0 else 'FAIL'} against the 15% acceptance)")
    if any(not np.isfinite(r["err"]) for r in rows):
        print("[sweep] NOTE: a row with no achieved speed means nothing stayed upright at that "
              "command -- that is a survival failure, not a tracking one.")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(run=args.run, v_max=v_max, bringup=args.bringup,
                                                   worst_pct=worst, rows=rows), indent=2))
        print(f"[sweep] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
