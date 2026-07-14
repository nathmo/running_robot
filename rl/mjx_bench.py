"""Phase 5: measure steady-state GPU env-steps/sec for the MJX training loop, excluding first-
call JIT compilation. One-off benchmarking script (not part of the training pipeline).

Run:
    .venv/bin/python -m rl.mjx_bench --preset m2_walk --n-envs 2048
"""
import argparse
import time

import jax
import jax.numpy as jnp
import optax

from .config import get_config
from .mjx_env import Dash01MjxEnv
from .mjx_train import make_networks, make_train_step
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.acme import running_statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="m2_walk")
    ap.add_argument("--n-envs", type=int, default=2048)
    ap.add_argument("--num-minibatches", type=int, default=32)
    ap.add_argument("--iters", type=int, default=5, help="timed iterations after the warmup one")
    args = ap.parse_args()

    cfg = get_config(args.preset)
    env = Dash01MjxEnv(cfg)
    nets = make_networks(env, cfg)

    key = jax.random.PRNGKey(0)
    key, k_policy, k_value, k_env = jax.random.split(key, 4)
    params = ppo_losses.PPONetworkParams(
        policy=nets.policy_network.init(k_policy), value=nets.value_network.init(k_value))
    normalizer_params = running_statistics.init_state(jnp.zeros(env.obs_size))
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
    opt_state = optimizer.init(params)

    train_step = make_train_step(env, nets, cfg, optimizer, args.n_envs, cfg.n_steps,
                                 args.num_minibatches)
    env_keys = jax.random.split(k_env, args.n_envs)
    env_state = jax.jit(jax.vmap(env.reset))(env_keys, jnp.zeros(args.n_envs))

    steps_per_iter = args.n_envs * cfg.n_steps
    print(f"[bench] preset={args.preset} n_envs={args.n_envs} unroll_length={cfg.n_steps} "
          f"steps_per_iter={steps_per_iter}")

    # warmup: triggers JIT compilation, excluded from timing
    t0 = time.perf_counter()
    key, k = jax.random.split(key)
    env_state, params, opt_state, normalizer_params, key, _, _ = train_step(
        env_state, params, opt_state, normalizer_params, k, jnp.asarray(cfg.ent_coef))
    jax.block_until_ready(params)
    print(f"[bench] warmup (incl. compile) took {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    for _ in range(args.iters):
        key, k = jax.random.split(key)
        env_state, params, opt_state, normalizer_params, key, _, _ = train_step(
            env_state, params, opt_state, normalizer_params, k, jnp.asarray(cfg.ent_coef))
    jax.block_until_ready(params)
    elapsed = time.perf_counter() - t0

    total_steps = steps_per_iter * args.iters
    print(f"[bench] {args.iters} steady-state iterations, {total_steps} env-steps, "
          f"{elapsed:.2f}s -> {total_steps/elapsed:,.0f} env-steps/sec")


if __name__ == "__main__":
    main()
