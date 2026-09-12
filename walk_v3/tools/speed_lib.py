"""Can this controller HOLD a commanded speed? Search specs per target, not policy weights.

The joystick question -- "go at 1.5 m/s" -- is the same question as "stop", which ten shaped
training configurations could not answer (walk_v3/README.md). Before spending another training
run on it, settle whether the action space contains a steady gait at each speed at all.

Method is `brake_search.py`'s, minus the ramp: hold the policy's per-tick residual (45-95% of the
joint motion -- zeroing it tests nothing), override the LATCHED spec with a CONSTANT modulation of
the cruise spec, run for a few seconds and measure the SETTLED speed over the last third. Four
parameters, so the CEM converges fast.

    python walk_v3/tools/speed_lib.py --run ... --checkpoint ... --targets 1.0,1.5,2.0,2.5,3.0

If a spec exists at every target, speed tracking is EXPRESSIBLE: any RL failure is exploration,
and the table itself is already a joystick controller by interpolation. If the low targets have no
solution, the plant cannot hold them with this gait and no reward will change that.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

import gait
from env import EnvParams
from evaluate import load_run

CHANNELS = ("freq_scale", "amp_scale", "o_cam", "pitch_bias")
BOUNDS = np.array([[0.3, 1.8], [0.1, 1.5], [-1.0, 1.0], [-1.0, 1.0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--targets", default="0.5,1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--elite", type=float, default=0.1)
    ap.add_argument("--cruise-s", type=float, default=6.0)
    ap.add_argument("--hold-s", type=float, default=6.0, help="how long the spec is held")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg, env, agent = load_run(args.run, args.checkpoint, n_envs=args.pop, dr=False, free_clock=True)
    # no finish line and no stop phase: a joystick run has neither, and leaving the line in would
    # hand the policy its pre-stop behaviour partway through the measurement
    params = EnvParams.final(cfg)._replace(dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0,
                                           pitch_assist=0.0, stoplight_prob=0.0, sprint_dist_m=1e4)
    dt = env.control_dt
    n_cruise, n_hold = int(args.cruise_s / dt), int(args.hold_s / dt)
    n_tail = max(1, n_hold // 3)
    act = agent._act_greedy
    key = jax.random.PRNGKey(args.seed)

    state, obs = env.reset(key, params)

    def cruise(carry, _):
        state, obs = carry
        a = jnp.clip(act(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        state2, obs2, _, _, _ = env.step(state, a, params)
        return (state2, obs2), None

    (state, obs), _ = jax.lax.scan(cruise, (state, obs), None, length=n_cruise)
    v0 = float(np.asarray(state.prev_vel_body)[:, 0].mean())
    print(f"[speed] free-running cruise {v0:.2f} m/s after {args.cruise_s:.0f} s", flush=True)
    cruise_spec = state.spec

    def hold(theta, state, obs):
        def step(carry, i):
            state, obs, alive, vsum, nsum = carry
            spec = cruise_spec
            spec = spec.at[:, gait.I_FREQ].set(jnp.clip(cruise_spec[:, gait.I_FREQ] * theta[:, 0], -1.0, 1.0))
            spec = spec.at[:, gait.I_S_CAM].multiply(theta[:, 1:2])
            spec = spec.at[:, gait.I_S_THIGH].multiply(theta[:, 1:2])
            spec = spec.at[:, gait.I_O.start].set(jnp.clip(theta[:, 2], -1.0, 1.0))
            spec = spec.at[:, gait.I_O.start + 1].set(jnp.clip(theta[:, 3], -1.0, 1.0))
            pol = jnp.clip(act(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
            a = jnp.concatenate([spec, pol[:, gait.SPEC_DIM:]], axis=1)
            state2, obs2, _, done, info = env.step(state, a, params)
            alive2 = alive & ~done
            vx = (info["sprint_d"] - state.sprint_d) / dt
            tail = (i >= n_hold - n_tail) & alive2          # SETTLED speed, not a momentary touch
            vsum = vsum + jnp.where(tail, vx, 0.0)
            nsum = nsum + tail.astype(jnp.float32)
            return (state2, obs2, alive2, vsum, nsum), None

        n = theta.shape[0]
        init = (state, obs, jnp.ones(n, bool), jnp.zeros(n), jnp.zeros(n))
        (_, _, alive, vsum, nsum), _ = jax.lax.scan(step, init, jnp.arange(n_hold))
        return alive, vsum / jnp.maximum(nsum, 1.0)

    hold_jit = jax.jit(hold)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    out = {"cruise": v0, "targets": {}}

    for tgt in [float(x) for x in args.targets.split(",")]:
        mu, sd = (lo + hi) / 2, (hi - lo) / 4
        n_cand = max(4, args.pop // args.reps)
        n_elite = max(4, int(n_cand * args.elite))
        best = dict(err=np.inf, v=np.nan, theta=None, upright=False)
        for it in range(args.iters):
            key, k = jax.random.split(key)
            th = np.clip(np.asarray(jax.random.normal(k, (n_cand, 4))) * sd + mu, lo, hi)
            th_env = np.repeat(th, args.reps, axis=0)
            pad = args.pop - th_env.shape[0]
            if pad > 0:
                th_env = np.concatenate([th_env, np.repeat(th[-1:], pad, axis=0)], axis=0)
            alive, vset = hold_jit(jnp.asarray(th_env, jnp.float32), state, obs)
            k2 = n_cand * args.reps
            alive = np.asarray(alive)[:k2].reshape(n_cand, args.reps)
            vset = np.asarray(vset)[:k2].reshape(n_cand, args.reps)
            # a fall is disqualifying, but graded so the search can climb from a dead start
            err = np.where(alive, np.abs(vset - tgt), 100.0).mean(axis=1)
            order = np.argsort(err)
            if err[order[0]] < best["err"]:
                best = dict(err=float(err[order[0]]), v=float(vset[order[0]].mean()),
                            theta=th[order[0]].tolist(), upright=bool(alive[order[0]].all()))
            el = order[:n_elite]
            mu, sd = th[el].mean(0), th[el].std(0) + 1e-3
        ok = best["upright"] and best["err"] < 0.25
        print(f"[speed] target {tgt:.1f} m/s -> settled {best['v']:+.2f} m/s "
              f"(err {best['err']:.2f}, upright={best['upright']}) {'OK' if ok else 'NOT REACHED'}",
              flush=True)
        out["targets"][f"{tgt:.1f}"] = dict(best, ok=bool(ok))

    reached = [t for t, d in out["targets"].items() if d["ok"]]
    print(f"\n[speed] reachable targets: {', '.join(reached) if reached else 'NONE'}")
    print("[speed] a spec per speed = a joystick controller by interpolation; gaps = speeds this "
          "gait cannot hold, which no reward term will fix")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
