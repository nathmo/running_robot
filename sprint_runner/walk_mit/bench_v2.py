"""Throughput benchmark of the v2 (latched) CPU stack -- the number to hold against the GPU port.

Measures, on this machine:
  1. raw env steps/s of DashEnv(v2_s1) single-process (the MuJoCo + latch + reward cost)
  2. vectorised rollout steps/s with N workers (SubprocVecEnv), the training-loop figure
  3. PPO update time per rollout (SymPPO + masked policy, the real batch sizes)
and prints env-steps per second of wall clock for the whole learn() cycle, which is the unit the
cluster runs are budgeted in (300 M steps at 100 Hz).

    python walk_mit/bench_v2.py --n-envs 8 --n-steps 288            # laptop
    python walk_mit/bench_v2.py --n-envs 64 --n-steps 288 --subproc   # one JED node (72 cores)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v2_s1")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=288)
    ap.add_argument("--subproc", action="store_true")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--single-steps", type=int, default=2000)
    args = ap.parse_args()
    from config import get_config
    from env import DashEnv
    cfg = get_config(args.preset)
    cfg.n_steps = args.n_steps
    # 1. single env
    env = DashEnv(cfg)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    n = 0
    for _ in range(args.single_steps):
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, env.action_dim).astype(np.float32))
        n += 1
        if term or trunc:
            env.reset()
    single = n / (time.perf_counter() - t0)
    print(f"[bench] single env: {single:.0f} steps/s ({1e3 / single:.2f} ms/step at 100 Hz control, "
          f"{cfg.control_decimation} substeps)")
    # 2 + 3. the training loop
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    import train as tr
    Algo = tr.algo_class(cfg)
    vec_cls = SubprocVecEnv if args.subproc and args.n_envs > 1 else DummyVecEnv
    base = vec_cls([(lambda: Monitor(DashEnv(cfg))) for _ in range(args.n_envs)])
    venv = VecNormalize(base, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=cfg.gamma)
    policy, kw = tr.v2_policy_kwargs(cfg, base, venv.observation_space.shape[0])
    pk = dict(net_arch=list(cfg.policy_hidden), **kw)
    extra = tr.v2_sym_kwargs(cfg, base) if Algo.__name__ == "SymPPO" else {}
    model = Algo(policy, venv, n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
                 gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, learning_rate=cfg.learning_rate,
                 clip_range=cfg.clip_range, ent_coef=cfg.ent_coef, target_kl=cfg.target_kl,
                 policy_kwargs=pk, seed=0, verbose=0, **extra)
    print(f"[bench] {Algo.__name__} + {policy.__name__}, {args.n_envs} envs ({vec_cls.__name__}), "
          f"rollout {args.n_envs * cfg.n_steps} steps, batch {cfg.batch_size} x {cfg.n_epochs} epochs, "
          f"device {model.device}, torch threads {torch.get_num_threads()}")
    # warm-up
    model.learn(args.n_envs * cfg.n_steps)
    roll_t, upd_t = [], []
    orig_train = model.train

    def timed_train():
        t = time.perf_counter()
        orig_train()
        upd_t.append(time.perf_counter() - t)
    model.train = timed_train
    t0 = time.perf_counter()
    model.learn(args.rollouts * args.n_envs * cfg.n_steps, reset_num_timesteps=False)
    total = time.perf_counter() - t0
    steps = args.rollouts * args.n_envs * cfg.n_steps
    upd = float(np.mean(upd_t)) if upd_t else float("nan")
    print(f"[bench] learn(): {steps / total:.0f} env-steps/s wall  (rollout {(total - sum(upd_t)) / args.rollouts:.2f} s, "
          f"update {upd:.2f} s per {args.n_envs * cfg.n_steps}-step rollout)")
    print(f"[bench] 300 M steps at this rate = {300e6 / (steps / total) / 3600:.1f} h")
    venv.close()


if __name__ == "__main__":
    main()
