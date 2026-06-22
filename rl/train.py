"""Train a SpiderBot locomotion policy with PPO (Stable-Baselines3).

Examples:
  # short pipeline check (a few minutes on CPU)
  .venv/Scripts/python.exe -m rl.train --preset m1_stand --steps 200000 --n-envs 4
  # longer run, true parallel envs
  .venv/Scripts/python.exe -m rl.train --preset m1_stand --steps 20000000 --n-envs 8 --subproc

Outputs go to rl/runs/<name>/ : checkpoints, final model, VecNormalize stats, TensorBoard logs.
Watch:  .venv/Scripts/python.exe -m tensorboard.main --logdir rl/runs
"""
import argparse
import platform
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from .config import get_config
from .env import SpiderBotEnv


def make_env(preset):
    def _init():
        return Monitor(SpiderBotEnv(get_config(preset)))
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="m1_stand")
    ap.add_argument("--name", default=None, help="run folder name (default: preset)")
    ap.add_argument("--steps", type=int, default=None, help="override total timesteps")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--subproc", action="store_true", help="use SubprocVecEnv (true parallelism)")
    ap.add_argument("--resume", default=None, help="path to a .zip checkpoint to continue from")
    ap.add_argument("--no-progress", action="store_true", help="disable the rich progress bar (for logs)")
    args = ap.parse_args()

    cfg = get_config(args.preset)
    n_envs = args.n_envs or cfg.n_envs
    total = args.steps or cfg.total_steps
    name = args.name or args.preset
    run = Path("rl/runs") / name
    run.mkdir(parents=True, exist_ok=True)

    vec_cls = SubprocVecEnv if (args.subproc and n_envs > 1) else DummyVecEnv
    venv = vec_cls([make_env(args.preset) for _ in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg.gamma)

    if args.resume:
        model = PPO.load(args.resume, env=venv, tensorboard_log=str(run))
    else:
        model = PPO(
            "MlpPolicy", venv,
            n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
            gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, learning_rate=cfg.learning_rate,
            clip_range=cfg.clip_range, ent_coef=cfg.ent_coef,
            policy_kwargs=dict(net_arch=list(cfg.policy_hidden)),
            seed=cfg.seed, verbose=1, tensorboard_log=str(run),
        )

    ckpt = CheckpointCallback(
        save_freq=max(200_000 // n_envs, 1), save_path=str(run),
        name_prefix="ppo", save_vecnormalize=True)

    print(f"[train] preset={args.preset} n_envs={n_envs} ({vec_cls.__name__}) "
          f"total_steps={total} -> {run}")
    model.learn(total_timesteps=total, callback=ckpt, progress_bar=not args.no_progress)
    model.save(run / "final_model")
    venv.save(str(run / "vecnormalize.pkl"))
    print(f"[train] done -> {run/'final_model.zip'}")


if __name__ == "__main__":
    main()
