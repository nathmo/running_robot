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
import json
import platform
from pathlib import Path

import torch
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


class CurriculumCallback(BaseCallback):
    """Linearly ramp the sampled forward-command fraction from `start` to `target` over `warmup`
    timesteps, so the policy first learns to move a little before full speed is demanded. Pushes the
    value into every worker env via env_method (works for DummyVecEnv and SubprocVecEnv alike).
    The current value is persisted to <run>/curriculum.json so a --resume can restore it."""
    def __init__(self, target, warmup, start=0.0, run_dir=None):
        super().__init__()
        self.target, self.warmup, self.start = float(target), int(warmup), float(start)
        self.run_dir = run_dir
        self._last = None

    def _apply(self):
        frac = 1.0 if self.warmup <= 0 else min(1.0, self.num_timesteps / self.warmup)
        val = self.start + frac * (self.target - self.start)
        self.logger.record("curriculum/cmd_vx_frac", val)
        if val == self._last:          # post-warmup the value is constant: skip the worker
            return                     # broadcast and the (identical) file rewrite
        self._last = val
        self.training_env.env_method("set_cmd_vx_frac", val)
        if self.run_dir:
            try:
                (Path(self.run_dir) / "curriculum.json").write_text(
                    json.dumps({"cmd_vx_frac": val}))
            except OSError:
                pass

    def _on_rollout_start(self) -> None:   # once per rollout is plenty; avoids per-step IPC
        self._apply()

    def _on_step(self) -> bool:
        return True


class EntropyCallback(BaseCallback):
    """Two jobs, both per rollout:
    1. Clamp log_std at cfg.max_log_std — the entropy bonus of a CLIPPED Gaussian keeps paying as
       std grows past the action range while behavior stops changing (free reward); the previous
       run ratcheted std to 2.1 this way, training on saturated bang-bang actions.
    2. Competence-keyed entropy anneal: HOLD ent_coef until stepping has emerged (rollout-mean
       reward_terms/air_time > gate for `patience` consecutive rollouts), then anneal linearly to
       cfg.ent_final over cfg.ent_anneal_steps. A clock-driven anneal kills exploration on
       schedule even if stepping hasn't been found; a fixed low value collapsed std onto the
       skating optimum in earlier runs."""
    def __init__(self, cfg, patience=5):
        super().__init__()
        self.cfg = cfg
        self.patience = patience
        self._streak = 0
        self._anneal_from = None       # num_timesteps when the gate opened
        self._anneal_base = None       # model.ent_coef at that moment (respects resume/--ent-coef)
        self._air_sum, self._air_n = 0.0, 0
        # a preset whose commands never reach the gait gate (e.g. m1_stand) has air_time
        # structurally zero — gate on episode-length competence instead so the anneal still fires
        self._stand_only = cfg.cmd_vx_frac * cfg.vx_max < cfg.gait_cmd_gate

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            terms = info.get("reward_terms")
            if terms and "air_time" in terms:
                self._air_sum += float(terms["air_time"])
                self._air_n += 1
        return True

    def _competent(self, air):
        if self._stand_only:
            ep = self.model.ep_info_buffer
            if not ep:
                return False
            max_steps = self.cfg.episode_s / 0.02       # 50 Hz control (sim 1 kHz, decimation 20)
            return float(sum(e["l"] for e in ep)) / len(ep) > 0.9 * max_steps
        return air > self.cfg.ent_gate_air_time

    def _on_rollout_end(self) -> None:
        with torch.no_grad():
            self.model.policy.log_std.clamp_(max=self.cfg.max_log_std)
        air = self._air_sum / max(self._air_n, 1)
        self._air_sum, self._air_n = 0.0, 0
        if self._anneal_from is None:
            self._streak = self._streak + 1 if self._competent(air) else 0
            if self._streak >= self.patience:
                self._anneal_from = self.num_timesteps
                self._anneal_base = float(self.model.ent_coef)   # NOT cfg: keep any override
                print(f"[train] competence gate open (air_time={air:.3f}) at "
                      f"{self.num_timesteps} steps -> annealing ent_coef "
                      f"{self._anneal_base} -> {self.cfg.ent_final}")
        if self._anneal_from is not None and self._anneal_base > self.cfg.ent_final:
            frac = min(1.0, (self.num_timesteps - self._anneal_from)
                       / max(self.cfg.ent_anneal_steps, 1))
            self.model.ent_coef = (self._anneal_base
                                   + frac * (self.cfg.ent_final - self._anneal_base))
        self.logger.record("curriculum/ent_coef", float(self.model.ent_coef))


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
    ap.add_argument("--vecnormalize", default=None,
                    help="VecNormalize .pkl to reload on resume (default: auto-match the checkpoint)")
    ap.add_argument("--ent-coef", type=float, default=None,
                    help="override ent_coef (e.g. anneal down when resuming a policy that already walks)")
    ap.add_argument("--no-progress", action="store_true", help="disable the rich progress bar (for logs)")
    args = ap.parse_args()

    cfg = get_config(args.preset)
    n_envs = args.n_envs or cfg.n_envs
    total = args.steps or cfg.total_steps
    name = args.name or args.preset
    run = Path("rl/runs") / name
    run.mkdir(parents=True, exist_ok=True)
    # record the preset so evaluate/gait_probe rebuild the SAME env config later, whatever the
    # run folder is called (name-based inference is only the legacy fallback)
    (run / "preset.json").write_text(json.dumps({"preset": args.preset}))

    vec_cls = SubprocVecEnv if (args.subproc and n_envs > 1) else DummyVecEnv
    base_venv = vec_cls([make_env(args.preset) for _ in range(n_envs)])

    # NOTE: reward normalization is OFF. The reward is hand-balanced in raw units (and
    # suicide-proofed against the raw fall penalty); the old norm_reward=True divided it all by a
    # running return-std that reached ~110, shrinking the fall penalty to ~1.8 and every shaping
    # term to noise. Observation normalization stays on.
    if args.resume:
        # ONLY resume checkpoints trained on the CURRENT model + reward (v3, 2026-07-06+).
        # Pre-v3 checkpoints are incompatible three ways: the plant changed (+7.1 kg motor mass,
        # condim 6, new keyframe), their value nets learned VecNormalize-SCALED rewards (~1/110th
        # of today's raw scale), and PPO.load keeps the checkpoint's own hyperparameters (gamma,
        # lr schedule, target_kl) — the config's new values do NOT apply on resume.
        print("[train] WARNING: --resume assumes a v3-era checkpoint (current model + raw-reward "
              "training). Resuming a pre-v3 run here will silently destroy it — retrain instead.")
        # reload the matching VecNormalize stats so obs normalization CONTINUES (a fresh wrapper
        # would reset them to mean 0 / var 1 and cause a big transient dip on resume).
        from .evaluate import pick_vecnormalize
        ckpt = args.resume[:-4] if args.resume.endswith(".zip") else args.resume
        vn = Path(args.vecnormalize) if args.vecnormalize else \
            pick_vecnormalize(Path(ckpt).parent, ckpt)   # stats live next to the checkpoint
        if vn.exists():
            venv = VecNormalize.load(str(vn), base_venv)
            venv.training = True
            venv.norm_reward = False
            print(f"[train] resumed VecNormalize stats <- {vn}")
        else:
            venv = VecNormalize(base_venv, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=cfg.gamma)
            print(f"[train] WARNING: no VecNormalize stats at {vn}; starting normalization fresh")
        model = PPO.load(args.resume, env=venv, tensorboard_log=str(run))
        if args.ent_coef is not None:
            model.ent_coef = args.ent_coef
            print(f"[train] ent_coef override -> {args.ent_coef}")
        # restore the curriculum point the checkpoint was trained at (resume restarts the step
        # counter, so re-ramping would silently retrain on easier commands than the policy knows)
        cur = Path(ckpt).parent / "curriculum.json"
        if cur.exists():
            val = json.loads(cur.read_text()).get("cmd_vx_frac", cfg.cmd_vx_frac)
            base_venv.env_method("set_cmd_vx_frac", val)
            print(f"[train] resumed curriculum cmd_vx_frac={val:.3f} <- {cur}")
    else:
        venv = VecNormalize(base_venv, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=cfg.gamma)
        lr = lambda p: cfg.lr_final + p * (cfg.learning_rate - cfg.lr_final)  # p: 1 -> 0 over the run
        model = PPO(
            "MlpPolicy", venv,
            n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
            gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, learning_rate=lr,
            clip_range=cfg.clip_range, ent_coef=cfg.ent_coef, target_kl=cfg.target_kl,
            policy_kwargs=dict(net_arch=list(cfg.policy_hidden)),
            seed=cfg.seed, verbose=1, tensorboard_log=str(run),
        )

    # log to CSV (for plots) + TensorBoard + stdout, all in the run dir
    model.set_logger(configure(str(run), ["stdout", "csv", "tensorboard"]))

    cb_list = [
        CheckpointCallback(save_freq=max(200_000 // n_envs, 1), save_path=str(run),
                           name_prefix="ppo", save_vecnormalize=True),
        RewardTermCallback(),
        EntropyCallback(cfg),
        PlotCallback(run, every_steps=500_000),
    ]
    # ramp the forward command from cmd_vx_frac_start up to cmd_vx_frac (skip if no ramp requested).
    # On resume the step counter restarts at 0, so skip the ramp and hold the value restored from
    # curriculum.json above (re-ramping from 0 would throw away the walking already learned).
    if not args.resume and cfg.curriculum_steps > 0 and cfg.cmd_vx_frac > cfg.cmd_vx_frac_start:
        cb_list.append(CurriculumCallback(cfg.cmd_vx_frac, cfg.curriculum_steps,
                                          cfg.cmd_vx_frac_start, run_dir=run))
        print(f"[train] curriculum: cmd_vx_frac {cfg.cmd_vx_frac_start} -> {cfg.cmd_vx_frac} "
              f"over {cfg.curriculum_steps} steps")
    callbacks = CallbackList(cb_list)

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
