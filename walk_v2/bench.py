"""Throughput benchmark: env steps/s of the MJX plant vs batch size, and the whole PPO iteration.

    python walk_v2/bench.py                       # sweep n_envs 256..8192 on the local device
    python walk_v2/bench.py --n-envs 4096 --ppo   # + one full PPO iteration at that size
    python walk_v2/bench.py --json results/bench_lyra_rtx6000.json

Reports, per batch size: env-only steps/s (random actions, jitted), and with --ppo the
rollout + update time of one iteration. The reference numbers to beat are the CPU stack's:
walk_mit at 64 SubprocVecEnv workers on a 72-core JED node ~ 1,500-2,500 env steps/s at
200 Hz (i.e. ~2x fewer sim-seconds per env step than here at 100 Hz).
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import jax
import jax.numpy as jnp

from config import get_config
from env import DashEnvV2, EnvParams


def bench_env(cfg, n_envs, n_steps=64, n_warm=2):
    env = DashEnvV2(cfg, n_envs=n_envs)
    params = EnvParams.final(cfg)
    key = jax.random.PRNGKey(0)
    t = time.time()
    state, obs = env.reset(key, params)
    obs.block_until_ready()
    t_reset = time.time() - t

    from functools import partial

    @partial(jax.jit, static_argnums=(3,))
    def run(state, obs, key, n):
        def body(carry, _):
            state, obs, key = carry
            key, k = jax.random.split(key)
            a = jax.random.uniform(k, (n_envs, env.action_dim), minval=-1.0, maxval=1.0)
            state, obs, r, d, info = env.step(state, a, params)
            return (state, obs, key), r.mean()
        (state, obs, key), rs = jax.lax.scan(body, (state, obs, key), None, length=n)
        return state, obs, key, rs

    # warm up with the SAME static length: a different n recompiles, and that compile (~40 s on a
    # V100) used to land inside the timed call as a fake ~600 ms/step floor
    t = time.time()
    state, obs, key, _ = run(state, obs, key, n_steps)
    obs.block_until_ready()
    t_compile = time.time() - t
    for _ in range(n_warm - 1):
        state, obs, key, _ = run(state, obs, key, n_steps)
    obs.block_until_ready()
    t = time.time()
    state, obs, key, rs = run(state, obs, key, n_steps)
    rs.block_until_ready()
    dt = time.time() - t
    sps = n_envs * n_steps / dt
    return dict(n_envs=n_envs, env_sps=sps, substeps_per_s=sps * cfg.control_decimation,
                sim_seconds_per_wall_s=sps * env.control_dt, reset_s=t_reset, compile_s=t_compile,
                step_ms=1e3 * dt / n_steps)


def bench_ppo(cfg, n_envs, n_iters=3):
    from ppo import PPO
    cfg.n_envs = n_envs
    env = DashEnvV2(cfg, n_envs=n_envs)
    agent = PPO(cfg, env, PKG_DIR / "runs" / "_bench", cfg.total_steps, eval_env=None)
    agent.key, k = jax.random.split(agent.key)
    agent.env_state, agent.obs = env.reset(k, agent.env_params)
    t = time.time()
    row = agent.iterate()
    t_first = time.time() - t
    ts = []
    for _ in range(n_iters):
        t = time.time()
        row = agent.iterate()
        ts.append(time.time() - t)
    return dict(n_envs=n_envs, n_steps=cfg.n_steps, batch=agent.batch, iter_first_s=t_first,
                iter_s=float(np.mean(ts)), ppo_sps=agent.batch / float(np.mean(ts)),
                rollout_s=row["time/rollout_s"], minibatches=agent.n_minibatches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v2_s2_free")
    ap.add_argument("--n-envs", type=int, nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--ppo", action="store_true")
    ap.add_argument("--iterations", type=int, default=0, help="MJX solver cap override (0 = preset/XML)")
    ap.add_argument("--ls-iterations", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cfg = get_config(args.preset)
    if args.iterations or args.ls_iterations:
        cfg = dataclasses.replace(cfg, mjx_iterations=args.iterations or cfg.mjx_iterations,
                                  mjx_ls_iterations=args.ls_iterations or cfg.mjx_ls_iterations)
    dev = jax.devices()[0]
    sizes = args.n_envs or ([8, 32] if dev.platform == "cpu" else [256, 1024, 2048, 4096, 8192])
    print(f"[bench] {dev.platform} {getattr(dev, 'device_kind', dev)}  preset {args.preset}  "
          f"solver cap iterations={cfg.mjx_iterations or 'xml'} ls={cfg.mjx_ls_iterations or 'xml'}")
    out = dict(device=str(dev), platform=dev.platform, preset=args.preset, env=[], ppo=[],
               mjx_iterations=cfg.mjx_iterations, mjx_ls_iterations=cfg.mjx_ls_iterations)
    for n in sizes:
        r = bench_env(cfg, n, n_steps=args.steps)
        out["env"].append(r)
        print(f"  n_envs {n:6d}: {r['env_sps']:>10,.0f} env steps/s = {r['sim_seconds_per_wall_s']:>8,.1f} sim-s/s "
              f"(step {r['step_ms']:.2f} ms, reset {r['reset_s']:.1f}s, compile {r['compile_s']:.1f}s)", flush=True)
        if args.ppo:
            p = bench_ppo(cfg, n)
            out["ppo"].append(p)
            print(f"           PPO iter {p['iter_s']:.2f}s (rollout {p['rollout_s']:.2f}s) -> {p['ppo_sps']:,.0f} steps/s "
                  f"[batch {p['batch']:,}, first iter {p['iter_first_s']:.0f}s incl. compile]", flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"[bench] wrote {args.json}")


if __name__ == "__main__":
    main()
