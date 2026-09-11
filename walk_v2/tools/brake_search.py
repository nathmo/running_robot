"""Is a STOP expressible by the v2 controller at all? Search per-cycle gait specs, not policy weights.

Nine reward-shaped training configurations failed to teach this robot to brake: it never slows, and it
falls while ACCELERATING (see the table in walk_v2/README.md). Two explanations remain, and they call for
opposite fixes:

  (a) an RL exploration problem -- a braking spec sequence exists, PPO never found it;
  (b) an ARCHITECTURE problem -- the latched Fourier action space does not contain a stop, in which case
      no reward term will ever produce one and the fix is a bigger residual or a brake primitive.

This decides between them. Take a trained runner, let it reach steady state, then STOP USING THE POLICY
and drive the spec directly with a smooth schedule: from the cruise spec, modulate frequency, the cam and
thigh amplitudes, and the fore-aft offset o_cam over a few seconds. The search is a cross-entropy method
run entirely inside the batched env -- one candidate schedule per env, so a whole population is a single
vmapped rollout. If any schedule brings the robot to a near standstill upright, the action space contains
a stop and (a) holds. If thousands cannot, (b) holds.

    python walk_v2/tools/brake_search.py --run walk_v2/runs/<run> --checkpoint <ckpt.msgpack>

Knobs: --pop, --iters, --elite, --cruise-s, --brake-s.
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

# the schedule: 4 channels x 3 knots, piecewise linear in time over the braking window
CHANNELS = ("freq_scale", "amp_scale", "o_cam", "pitch_bias")
KNOTS = 3
DIM = len(CHANNELS) * KNOTS
# per-channel (low, high) at every knot; knot 0 starts at the cruise value by construction
BOUNDS = np.array([[0.4, 1.6],      # frequency multiplier on the cruise frequency
                   [0.1, 1.4],      # cam/thigh amplitude multiplier (shorter strides = less push)
                   [-1.0, 1.0],     # o_cam in spec units (fore-aft foot placement: the capture step)
                   [-1.0, 1.0]])    # thigh offset (lean)


def schedule(theta, frac):
    """theta (pop, DIM), frac scalar in [0,1] -> (pop, 4) channel values, piecewise linear over knots."""
    th = theta.reshape(theta.shape[0], len(CHANNELS), KNOTS)
    x = frac * (KNOTS - 1)
    i0 = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, KNOTS - 2)
    w = x - i0
    a = jnp.take_along_axis(th, i0[None, None, None] * jnp.ones((th.shape[0], len(CHANNELS), 1), jnp.int32), axis=2)[:, :, 0]
    b = jnp.take_along_axis(th, (i0 + 1)[None, None, None] * jnp.ones((th.shape[0], len(CHANNELS), 1), jnp.int32), axis=2)[:, :, 0]
    return a + w * (b - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--elite", type=float, default=0.1)
    ap.add_argument("--cruise-s", type=float, default=6.0, help="policy-driven run-up before braking")
    ap.add_argument("--brake-s", type=float, default=8.0, help="length of the open-loop braking window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg, env, agent = load_run(args.run, args.checkpoint, n_envs=args.pop, dr=False)
    params = EnvParams.final(cfg)._replace(dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0,
                                           pitch_assist=0.0, stoplight_prob=0.0)
    dt = env.control_dt
    n_cruise, n_brake = int(args.cruise_s / dt), int(args.brake_s / dt)
    key = jax.random.PRNGKey(args.seed)

    # ---- run-up: every env is the same policy, so they only differ once the schedule takes over
    state, obs = env.reset(key, params)
    act = agent._act_greedy

    def cruise_step(carry, _):
        state, obs = carry
        a = jnp.clip(act(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        state2, obs2, _, _, info = env.step(state, a, params)
        return (state2, obs2), info["freq_hz"]

    (state, obs), fz = jax.lax.scan(cruise_step, (state, obs), None, length=n_cruise)
    v0 = float(np.asarray(state.prev_vel_body)[:, 0].mean())
    print(f"[brake] run-up {args.cruise_s:.0f} s: cruise speed {v0:.2f} m/s, clock {float(np.asarray(fz[-1]).mean()):.2f} Hz")
    if v0 < 0.8:
        print("[brake] the policy is not running; nothing to brake from")
        return
    cruise_spec = state.spec                                  # (pop, 44), identical across envs

    # ---- braking window: the POLICY IS OFF, the spec is driven by the schedule
    def brake_rollout(theta, state, obs):
        def step(carry, i):
            state, obs, alive, vmin = carry
            ch = schedule(theta, i / max(n_brake - 1, 1))
            spec = cruise_spec
            spec = spec.at[:, gait.I_FREQ].set(jnp.clip(cruise_spec[:, gait.I_FREQ] * ch[:, 0], -1.0, 1.0))
            spec = spec.at[:, gait.I_S_CAM].multiply(ch[:, 1:2])
            spec = spec.at[:, gait.I_S_THIGH].multiply(ch[:, 1:2])
            spec = spec.at[:, gait.I_O.start].set(jnp.clip(ch[:, 2], -1.0, 1.0))
            spec = spec.at[:, gait.I_O.start + 1].set(jnp.clip(ch[:, 3], -1.0, 1.0))
            a = jnp.concatenate([spec, jnp.zeros((theta.shape[0], gait.ACTION_DIM - gait.SPEC_DIM))], axis=1)
            state2, obs2, _, done, info = env.step(state, a, params)
            alive2 = alive & ~info["fallen"]
            vx = info["sprint_d"] - state.sprint_d
            vmin = jnp.where(alive2, jnp.minimum(vmin, jnp.abs(vx) / dt), vmin)
            return (state2, obs2, alive2, vmin), None

        n = theta.shape[0]
        init = (state, obs, jnp.ones(n, bool), jnp.full(n, jnp.inf))
        (_, _, alive, vmin), _ = jax.lax.scan(step, init, jnp.arange(n_brake))
        return alive, vmin

    brake_jit = jax.jit(brake_rollout)

    # ---- cross-entropy method over the schedule
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    mu = np.repeat(((lo + hi) / 2)[:, None], KNOTS, axis=1).reshape(-1)
    sd = np.repeat(((hi - lo) / 4)[:, None], KNOTS, axis=1).reshape(-1)
    n_elite = max(4, int(args.pop * args.elite))
    best = dict(score=np.inf, v_min=np.inf, upright=False, theta=None)
    for it in range(args.iters):
        key, k = jax.random.split(key)
        theta = np.asarray(jax.random.normal(k, (args.pop, DIM))) * sd + mu
        lo_t = np.repeat(lo[:, None], KNOTS, axis=1).reshape(-1)
        hi_t = np.repeat(hi[:, None], KNOTS, axis=1).reshape(-1)
        theta = np.clip(theta, lo_t, hi_t)
        alive, vmin = brake_jit(jnp.asarray(theta, jnp.float32), state, obs)
        alive, vmin = np.asarray(alive), np.asarray(vmin)
        # a fall is disqualifying: score = slowest speed reached while staying upright
        score = np.where(alive, vmin, 1e3)
        order = np.argsort(score)
        el = order[:n_elite]
        if score[order[0]] < best["score"]:
            best = dict(score=float(score[order[0]]), v_min=float(vmin[order[0]]),
                        upright=bool(alive[order[0]]), theta=theta[order[0]].tolist())
        mu, sd = theta[el].mean(0), theta[el].std(0) + 1e-3
        n_up = int(alive.sum())
        print(f"[brake] iter {it}: upright {n_up}/{args.pop}, best |v| while upright "
              f"{score[order[0]] if score[order[0]] < 1e2 else float('nan'):.3f} m/s "
              f"(elite mean {np.mean(score[el][score[el] < 1e2]) if np.any(score[el] < 1e2) else float('nan'):.3f})", flush=True)

    print(f"\n[brake] cruise {v0:.2f} m/s -> best reachable |v| {best['v_min']:.3f} m/s while upright "
          f"(upright={best['upright']})")
    verdict = ("A STOP IS EXPRESSIBLE: the action space contains it, so the failure is RL exploration -- "
               "warm-start from this schedule." if best["v_min"] < 0.3 and best["upright"] else
               "NO STOP FOUND: the latched Fourier action space very likely does not contain one. More "
               "reward terms will not help; widen the residual or add a brake primitive.")
    print("[brake] " + verdict)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(cruise_speed=v0, best=best, verdict=verdict,
                                                   pop=args.pop, iters=args.iters), indent=1))


if __name__ == "__main__":
    main()
