"""Bisect the env step's cost: time a scan of env.step at a fixed batch with pieces stubbed out.

    python walk_v2/tools/profile_env.py --n-envs 8 --modes full no_reset no_physics no_reset_no_physics

modes: full          the real step
       no_reset      auto-reset returns a constant (no draw_plant / mjx.forward / obs rebuild per step)
       no_physics    mjx.step replaced by identity (everything but the 10 substeps of physics)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from mujoco import mjx  # noqa: E402

import env as env_mod  # noqa: E402
from config import get_config  # noqa: E402
from env import DashEnvV2, EnvParams  # noqa: E402


def run(mode: str, cfg, n: int, n_steps: int, reps: int):
    real_step = mjx.step
    if "no_physics" in mode:
        env_mod.mjx.step = lambda m, d: d
    try:
        e = DashEnvV2(cfg, n)
        params = EnvParams.final(cfg)
        key = jax.random.PRNGKey(0)
        t = time.time()
        state, obs = e.reset(key, params)
        jax.block_until_ready(obs)
        t_reset = time.time() - t
        if "no_reset" in mode:
            s1 = jax.tree_util.tree_map(lambda x: x[0], state)
            o1 = obs[0]
            e._reset_one = lambda k, p, ov: (s1, o1)
            e._step_v = jax.jit(jax.vmap(e._step_one, in_axes=(0, 0, None)))

        @jax.jit
        def roll(state, obs, key):
            def body(c, _):
                st, ob, k = c
                k, ka = jax.random.split(k)
                a = jax.random.uniform(ka, (n, e.action_dim), minval=-1.0, maxval=1.0)
                st, ob, r, d, info = e.step(st, a, params)
                return (st, ob, k), (r, d)
            (state, obs, key), (rs, ds) = jax.lax.scan(body, (state, obs, key), None, length=n_steps)
            return state, obs, key, rs, ds

        t = time.time()
        state, obs, key, rs, ds = roll(state, obs, key)
        jax.block_until_ready(rs)
        t_compile = time.time() - t
        t = time.time()
        for _ in range(reps):
            state, obs, key, rs, ds = roll(state, obs, key)
        jax.block_until_ready(rs)
        ms = (time.time() - t) / (reps * n_steps) * 1e3
        out = dict(mode=mode, n_envs=n, step_ms=ms, env_sps=n / ms * 1e3, reset_s=t_reset, compile_s=t_compile,
                   done_frac=float(ds.mean()))
        print(f"  {mode:22s} n={n:5d}: {ms:9.1f} ms/step  {out['env_sps']:10,.0f} env steps/s "
              f"(done {out['done_frac']:.3f}, reset {t_reset:.0f}s, compile {t_compile:.0f}s)", flush=True)
        return out
    finally:
        env_mod.mjx.step = real_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v2_s2_free")
    ap.add_argument("--n-envs", type=int, nargs="+", default=[8])
    ap.add_argument("--n-steps", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--modes", nargs="+", default=["full", "no_reset", "no_physics", "no_reset_no_physics"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cfg = get_config(args.preset)
    print(f"[profile_env] {jax.devices()[0].platform} {jax.devices()[0].device_kind} preset {args.preset}", flush=True)
    rows = [run(m, cfg, n, args.n_steps, args.reps) for n in args.n_envs for m in args.modes]
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(device=str(jax.devices()[0]), preset=args.preset, rows=rows), f, indent=1)
        print(f"[profile_env] wrote {args.json}")


if __name__ == "__main__":
    main()
