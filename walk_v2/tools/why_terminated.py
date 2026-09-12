"""Which termination fires, and what plant was drawn when it did?

"Everything dies in 0.3 s" is not a finding, it is a question. The env already reports every
termination reason separately (term_low, term_tip, term_floor, term_ws, term_nan) and the plant draw is
a pure function of the reset key, so both are measurable rather than guessable. This prints the reason
histogram and the drawn plant next to it.

Use it whenever an eval path kills policies unexpectedly -- in this project that has been the harness six
times out of six (see the DR trap: load_run(dr=True) re-enables pushes, wind, trips and a hot thermal
start, which killed the fully DR-trained v2 runner in 0.21 s).

    python walk_v2/tools/why_terminated.py --run walk_v2/runs/<run> --checkpoint ... --mode dr
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from env import DashEnvV2, EnvParams
from evaluate import load_run
from ppo import initial_params

REASONS = ("term_low", "term_tip", "term_floor", "term_ws", "term_nan")


def build(cfg, mode, n):
    """Return (env, dr_scale). Each mode isolates ONE difference from the nominal eval plant."""
    base = dict(dr_enable=False, obs_noise_enable=False, push_interval_s=0.0, wind_force_max=0.0,
                wind_gust_n=0.0, trip_prob=0.0, thermal_hot_start_max=0.0,
                pitch_assist_kp=0.0, roll_assist_kp=0.0, yaw_assist_kp=0.0)
    if mode == "nominal":
        return DashEnvV2(replace(cfg, **base), n_envs=n), 0.0
    if mode == "dr":                     # plant draw on, every disturbance still off
        return DashEnvV2(replace(cfg, **dict(base, dr_enable=True)), n_envs=n), 1.0
    if mode == "dr_nodelay":             # same, but pin the actuator delay to nominal
        c = replace(cfg, **dict(base, dr_enable=True))
        c.drive_delay_range_ms = (cfg.drive_delay_ms, cfg.drive_delay_ms)
        return DashEnvV2(c, n_envs=n), 1.0
    if mode == "delay_only":             # ONLY the delay draw, nominal plant otherwise
        return DashEnvV2(replace(cfg, **dict(base, dr_enable=True)), n_envs=n), 0.0
    if mode == "raw":                    # what load_run(dr=True) does: training cfg wholesale
        return DashEnvV2(cfg, n_envs=n), 1.0
    raise SystemExit(f"unknown mode {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--modes", default="nominal,dr,dr_nodelay,delay_only,raw")
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--v-cmd", type=float, default=None)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    cfg, _, agent = load_run(args.run, args.checkpoint, n_envs=args.n_envs, dr=False)
    print(f"[why] {args.run} | delay nominal {cfg.drive_delay_ms} ms, DR range "
          f"{cfg.drive_delay_range_ms} ms | dr_enable in cfg: {cfg.dr_enable}")

    for mode in args.modes.split(","):
        env, dsc = build(cfg, mode, args.n_envs)
        params = initial_params(cfg)._replace(dr_scale=dsc, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0,
                                              pitch_assist=0.0)
        ticks = int(args.seconds / env.control_dt)
        state, obs = env.reset(jax.random.PRNGKey(args.seed), params)
        if args.v_cmd is not None:
            state = state.replace(v_cmd=jnp.full_like(state.v_cmd, args.v_cmd),
                                  cmd_left=jnp.full_like(state.cmd_left, 1e4))

        def step(carry, _):
            state, obs = carry
            a = jnp.clip(agent._act_greedy(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
            state2, obs2, _, done, info = env.step(state, a, params)
            return (state2, obs2), (done, tuple(info[r] for r in REASONS))

        (_, _), (dones, flags) = jax.lax.scan(step, (state, obs), None, length=ticks)
        dones = np.asarray(dones)
        alive = np.cumprod(np.vstack([np.ones((1, dones.shape[1]), bool), ~dones[:-1]]), axis=0)
        live_ticks = alive.sum(0)
        # attribute each env's death to the reasons true on its LAST live tick
        counts = {}
        for r, f in zip(REASONS, flags):
            f = np.asarray(f).astype(bool)
            hit = (f & alive.astype(bool)).any(0)
            counts[r] = int(hit.sum())
        print(f"\n[{mode}] mean alive {live_ticks.mean() * env.control_dt:5.2f} s of {args.seconds:.1f} s"
              f" | survived {(live_ticks >= ticks).sum()}/{args.n_envs}")
        print("        " + "  ".join(f"{r.replace('term_', ''):>6}:{counts[r]:>3}" for r in REASONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
