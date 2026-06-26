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
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.logger import configure

from .config import get_config
from .env import SpiderBotEnv


def make_env(preset):
    def _init():
        return Monitor(SpiderBotEnv(get_config(preset)))
    return _init


class RewardTermCallback(BaseCallback):
    """Average each env's per-step reward-term dict over a rollout and log them as reward_terms/*,
    so the TensorBoard/CSV/plots show WHICH terms the policy is actually earning (or gaming)."""
    def __init__(self):
        super().__init__()
        self._sums, self._count = {}, 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            terms = info.get("reward_terms")
            if terms:
                for k, v in terms.items():
                    self._sums[k] = self._sums.get(k, 0.0) + float(v)
                self._count += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._count:
            for k, s in self._sums.items():
                self.logger.record(f"reward_terms/{k}", s / self._count)
        self._sums, self._count = {}, 0


class PlotCallback(BaseCallback):
    """Periodically (and at the end) render training_plots.png from progress.csv. Never fatal."""
    def __init__(self, run_dir, every_steps=500_000):
        super().__init__()
        self.run_dir, self.every, self._last = str(run_dir), every_steps, 0

    def _plot(self):
        try:
            from .plot_training import plot_run
            plot_run(self.run_dir)
        except Exception as e:               # plotting must never crash a training run
            if self.verbose:
                print(f"[plot] skipped: {e}")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            self._plot()
        return True

    def _on_training_end(self) -> None:
        self._plot()


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

    # log to CSV (for plots) + TensorBoard + stdout, all in the run dir
    model.set_logger(configure(str(run), ["stdout", "csv", "tensorboard"]))

    callbacks = CallbackList([
        CheckpointCallback(save_freq=max(200_000 // n_envs, 1), save_path=str(run),
                           name_prefix="ppo", save_vecnormalize=True),
        RewardTermCallback(),
        PlotCallback(run, every_steps=500_000),
    ])

    print(f"[train] preset={args.preset} n_envs={n_envs} ({vec_cls.__name__}) "
          f"total_steps={total} -> {run}")
    model.learn(total_timesteps=total, callback=callbacks, progress_bar=not args.no_progress)
    model.save(run / "final_model")
    venv.save(str(run / "vecnormalize.pkl"))
    plots = None
    try:
        from .plot_training import plot_run
        plots = plot_run(run)
    except Exception as e:                # matplotlib missing etc. — models are already saved
        plots = f"(skipped: {e})"
    print(f"[train] done -> {run/'final_model.zip'}  plots -> {plots}")


if __name__ == "__main__":
    main()
