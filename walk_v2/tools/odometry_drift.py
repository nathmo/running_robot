"""How far off is dead-reckoned distance from the velocity estimator over one dash?

The brake schedule is OPEN-LOOP and fired at a distance (~88 m). Under --hold-run the policy
itself needs no position at all -- its task input stays saturated -- so the trigger is the only
thing that needs to know where the robot is, and DASH-01 has no absolute position sensor. The
only candidate is the supervised velocity estimator (actor obs -> 128 -> 64 -> 3, trained on the
privileged body velocity), integrated tick by tick.

That matters because the brake is sensitive to where it starts: the same schedule on the same
episodes stopped 93/512 from 96.0 m and 55/512 from 94.9 m. This measures the error the trigger
would actually carry.

    python walk_v2/tools/odometry_drift.py --run walk_v2/runs/<run> --checkpoint <ckpt.msgpack>

Caveat stated up front: the estimator predicts BODY velocity, and distance down the track is a
WORLD-frame quantity. On a straight dash they nearly coincide, so this is a lower bound on the
real error -- any yaw drift adds to it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from env import EnvParams
from evaluate import load_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--trigger-m", type=float, default=88.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--free-clock", action="store_true", default=True)
    args = ap.parse_args()

    cfg, env, agent = load_run(args.run, args.checkpoint, n_envs=args.episodes, dr=False,
                              free_clock=args.free_clock)
    params = EnvParams.final(cfg)._replace(dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0,
                                           pitch_assist=0.0, stoplight_prob=0.0)
    dt = env.control_dt
    n = int(args.seconds / dt)
    act = agent._act_greedy
    net = agent.net
    a_dim = env.actor_dim
    scale = float(cfg.obs_scales["base_vel"])

    state, obs = env.reset(jax.random.PRNGKey(args.seed), params)

    def step(carry, _):
        state, obs, d_est = carry
        nobs = agent.stats.normalize(obs)
        # the estimator predicts the NORMALISED privileged velocity, so undo the normalisation
        # with the same statistics before integrating anything
        est_n = net.apply(agent.params, nobs, method=net.estimate)
        est = agent.stats.denormalize(
            jnp.concatenate([jnp.zeros_like(nobs[:, :a_dim]), est_n,
                             jnp.zeros_like(nobs[:, a_dim + 3:])], axis=1))[:, a_dim:a_dim + 3]
        v_est = est[:, 0] / scale
        a = jnp.clip(act(agent.params, nobs), -1.0, 1.0)
        state2, obs2, _, done, info = env.step(state, a, params)
        alive = ~done
        d_est = d_est + jnp.where(alive, v_est * dt, 0.0)
        return (state2, obs2, d_est), (info["sprint_d"], d_est, alive)

    (_, _, _), (d_true, d_est, alive) = jax.lax.scan(
        step, (state, obs, jnp.zeros(args.episodes)), None, length=n)
    d_true, d_est, alive = np.asarray(d_true), np.asarray(d_est), np.asarray(alive)

    # the tick each env first passes the trigger distance, by TRUTH and by dead reckoning
    print(f"[odom] {args.episodes} episodes, free_clock={args.free_clock}, trigger {args.trigger_m:.0f} m")
    errs, early = [], []
    for e in range(args.episodes):
        live = np.where(alive[:, e])[0]
        if live.size == 0:
            continue
        t_hi = live[-1]
        hit_true = np.where(d_true[:t_hi + 1, e] >= args.trigger_m)[0]
        if hit_true.size == 0:
            continue
        i_true = hit_true[0]
        hit_est = np.where(d_est[:t_hi + 1, e] >= args.trigger_m)[0]
        # where the robot ACTUALLY is when dead reckoning says it has reached the trigger
        if hit_est.size:
            errs.append(d_true[hit_est[0], e] - args.trigger_m)
            early.append((hit_est[0] - i_true) * dt)
        else:
            errs.append(np.nan)      # never believed it got there
    errs = np.asarray(errs, float)
    ok = np.isfinite(errs)
    if ok.sum() == 0:
        print("[odom] dead reckoning never reached the trigger in any episode")
        return
    e = errs[ok]
    print(f"[odom] when dead reckoning says '{args.trigger_m:.0f} m', the robot is really at "
          f"{args.trigger_m + np.median(e):.1f} m (median)")
    print(f"[odom] TRIGGER ERROR: median {np.median(e):+.2f} m, 10-90% {np.percentile(e, 10):+.2f} .. "
          f"{np.percentile(e, 90):+.2f} m, worst {e[np.argmax(np.abs(e))]:+.2f} m over {ok.sum()} episodes")
    print(f"[odom] for scale: a 1.1 m placement error moved the held-out stop count 55 -> 93")


if __name__ == "__main__":
    main()
