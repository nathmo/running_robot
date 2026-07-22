"""Train a DASH-01 sprint policy with PPO (Stable-Baselines3).

Run from anywhere (paths resolve relative to this folder):
  # short pipeline check (a few minutes on CPU)
  python training/train.py --preset m1_sprint --steps 200000 --n-envs 4
  # the weekend run (per-step Fourier + residuals, m2 = X,Z free)
  python training/train.py --preset m2_sprint --steps 240000000 --n-envs 20 --subproc
  # milestone chaining: warm-start the next stage from the previous run
  python training/train.py --preset m3_sprint --warm-start training/runs/m2_sprint/final_model.zip
  # cluster requeue-safe: same command works for the first start AND every restart
  python training/train.py --preset m2_sprint --steps 240000000 --n-envs 36 --subproc --resume auto

Outputs go to training/runs/<name>/ : checkpoints, final model, VecNormalize stats, TensorBoard
logs, progress.csv, training_plots.png.  Watch:  python -m tensorboard.main --logdir training/runs
"""
import argparse
import json
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import safe_mean

from config import Config, config_from_dict, config_to_dict, get_config, PRESETS
from env import DashEnv


def make_env(cfg):
    def _init():
        return Monitor(DashEnv(cfg))
    return _init


def _persist_curriculum(run_dir, key, val):
    """Merge one key into <run>/curriculum.json (three ramps share the file)."""
    p = Path(run_dir) / "curriculum.json"
    try:
        d = json.loads(p.read_text()) if p.exists() else {}
        d[key] = val
        p.write_text(json.dumps(d))
    except OSError:
        pass


class RewardTermCallback(BaseCallback):
    """Average each env's per-step reward-term dict over a rollout and log them as reward_terms/*,
    so the plots show WHICH terms the policy is actually earning (or gaming)."""
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


class RampCallback(BaseCallback):
    """Generic linear curriculum ramp start -> target over `warmup` timesteps, pushed into every
    worker env via env_method (DummyVecEnv and SubprocVecEnv alike) and persisted to
    curriculum.json. num_timesteps CONTINUES across --resume (reset_num_timesteps=False), so a
    requeued cluster job picks the ramp up exactly where it left off with no extra state."""
    def __init__(self, key, method, start, target, warmup, run_dir):
        super().__init__()
        self.key, self.method = key, method
        self.start, self.target, self.warmup = float(start), float(target), int(warmup)
        self.run_dir = run_dir
        self._last = None

    def _on_rollout_start(self) -> None:
        frac = 1.0 if self.warmup <= 0 else min(1.0, self.num_timesteps / self.warmup)
        val = self.start + frac * (self.target - self.start)
        self.logger.record(f"curriculum/{self.key}", val)
        if val == self._last:        # post-warmup the value is constant: skip the broadcast
            return
        self._last = val
        self.training_env.env_method(self.method, val)
        _persist_curriculum(self.run_dir, self.key, val)

    def _on_step(self) -> bool:
        return True


class GatedRampCallback(BaseCallback):
    """Competence-gated linear ramp: HOLD `start` until rollout ep_len_mean exceeds `gate_len` for
    `patience` consecutive rollouts, THEN ramp start -> target over `warmup` timesteps from the
    open step. Combines RampCallback's broadcast + curriculum.json persistence with EntropyCallback's
    gate/persist pattern. Fixes the clock-driven-curriculum failure: an m3 policy that can't yet
    balance the freed pitch DOF is never hardened toward flight-phase running until it can actually
    survive. A gate that never opens holds `start` (the easy regime) forever — the correct failure
    mode. num_timesteps-based, so gate + ramp both continue across --resume."""
    def __init__(self, key, method, start, target, warmup, run_dir, gate_len, patience=5):
        super().__init__()
        self.key, self.method = key, method
        self.start, self.target, self.warmup = float(start), float(target), int(warmup)
        self.run_dir, self.gate_len, self.patience = run_dir, float(gate_len), int(patience)
        self._streak = 0
        self._open_from = None
        self._last = None

    def _on_training_start(self) -> None:
        # requeue persistence: restore the gate-open step so a resumed job continues the ramp
        p = Path(self.run_dir) / "curriculum.json"
        if p.exists():
            d = json.loads(p.read_text())
            if f"{self.key}_gate_from" in d:
                self._open_from = int(d[f"{self.key}_gate_from"])
                print(f"[train] restored curriculum gate '{self.key}' (open from step "
                      f"{self._open_from})")

    def _on_rollout_start(self) -> None:
        if self._open_from is None:
            ep_len = (safe_mean([ep["l"] for ep in self.model.ep_info_buffer])
                      if len(self.model.ep_info_buffer) > 0 else 0.0)
            self._streak = self._streak + 1 if ep_len > self.gate_len else 0
            if self._streak >= self.patience:
                self._open_from = self.num_timesteps
                _persist_curriculum(self.run_dir, f"{self.key}_gate_from", self._open_from)
                print(f"[train] curriculum gate '{self.key}' opened at {self.num_timesteps} "
                      f"steps (ep_len_mean={ep_len:.0f} > {self.gate_len:.0f})")
        if self._open_from is None:
            val = self.start
        else:
            frac = 1.0 if self.warmup <= 0 else \
                min(1.0, (self.num_timesteps - self._open_from) / self.warmup)
            val = self.start + frac * (self.target - self.start)
        self.logger.record(f"curriculum/{self.key}", val)
        if val != self._last:
            self._last = val
            self.training_env.env_method(self.method, val)
            _persist_curriculum(self.run_dir, self.key, val)

    def _on_step(self) -> bool:
        return True


class EntropyCallback(BaseCallback):
    """Two jobs, both per rollout:
    1. Clamp log_std at cfg.max_log_std — the entropy bonus of a CLIPPED Gaussian keeps paying as
       std grows past the action range while behavior stops changing (free reward).
    2. Competence-keyed entropy anneal: HOLD ent_coef until stepping has emerged (rollout-mean
       reward_terms/air_time > gate for `patience` consecutive rollouts), then anneal linearly to
       cfg.ent_final over cfg.ent_anneal_steps. A clock-driven anneal kills exploration on
       schedule even if stepping hasn't been found."""
    def __init__(self, cfg, run_dir=None, patience=5):
        super().__init__()
        self.cfg = cfg
        self.run_dir = run_dir
        self.patience = patience
        self._streak = 0
        self._anneal_from = None
        self._anneal_base = None
        self._air_sum, self._air_n = 0.0, 0

    def _on_training_start(self) -> None:
        # requeue persistence: restore the gate state so a resumed cluster job continues the
        # anneal (both values are num_timesteps-based and the counter continues across --resume)
        if self.run_dir:
            p = Path(self.run_dir) / "curriculum.json"
            if p.exists():
                d = json.loads(p.read_text())
                if "ent_anneal_from" in d:
                    self._anneal_from = int(d["ent_anneal_from"])
                    self._anneal_base = float(d["ent_anneal_base"])
                    print(f"[train] restored entropy anneal state (from step "
                          f"{self._anneal_from}, base {self._anneal_base})")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            terms = info.get("reward_terms")
            if terms and "air_time" in terms:
                self._air_sum += float(terms["air_time"])
                self._air_n += 1
        return True

    def _on_rollout_start(self) -> None:
        # clamp here too: the clamp at rollout-END runs before that iteration's gradient update,
        # which can push log_std back over the cap — without this the whole next rollout samples
        # with an over-cap std.
        with torch.no_grad():
            self.model.policy.log_std.clamp_(max=self.cfg.max_log_std)

    def _on_rollout_end(self) -> None:
        with torch.no_grad():
            self.model.policy.log_std.clamp_(max=self.cfg.max_log_std)
        air = self._air_sum / max(self._air_n, 1)
        self._air_sum, self._air_n = 0.0, 0
        if self._anneal_from is None:
            self._streak = self._streak + 1 if air > self.cfg.ent_gate_air_time else 0
            gate_open = self._streak >= self.patience
            # hard fallback: if the competence gate never opens (air_time stuck below the gate
            # keeps std pinned at max_log_std forever — the m3_speed_v2 deadlock), start the anneal
            # anyway once ent_anneal_deadline_steps is reached, so exploration can finally collapse.
            deadline = self.cfg.ent_anneal_deadline_steps
            deadline_hit = deadline > 0 and self.num_timesteps >= deadline
            if gate_open or deadline_hit:
                self._anneal_from = self.num_timesteps
                self._anneal_base = float(self.model.ent_coef)
                if self.run_dir:
                    _persist_curriculum(self.run_dir, "ent_anneal_from", self._anneal_from)
                    _persist_curriculum(self.run_dir, "ent_anneal_base", self._anneal_base)
                reason = "competence gate" if gate_open else "hard deadline"
                print(f"[train] entropy anneal opened via {reason} (air_time={air:.3f}) at "
                      f"{self.num_timesteps} steps -> ent_coef "
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
            from plot_training import plot_run
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


def latest_checkpoint(run: Path):
    """Newest ppo_<steps>_steps.zip in the run dir (falling back to final_model.zip, so a
    COMPLETED run relaunched with a higher --steps extends instead of restarting), or None."""
    ckpts = sorted(run.glob("ppo_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    if ckpts:
        return ckpts[-1]
    fm = run / "final_model.zip"
    return fm if fm.exists() else None


def matching_vecnormalize(ckpt: Path):
    """The VecNormalize stats saved alongside a checkpoint / final model."""
    if ckpt.name == "final_model.zip":
        return ckpt.parent / "vecnormalize.pkl"
    return ckpt.parent / f"ppo_vecnormalize_{ckpt.stem[4:]}.pkl"


def rejuvenate_obs_rms(venv, count_cap, var_floor):
    """Warm-start fix: after VecNormalize.load, keep the prior mean/var but (a) cap the running
    count so newly-freed obs dims re-adapt within ~count_cap fresh samples instead of glacially
    (the source run carries count ~ its total_steps, weighting the stale prior far too heavily),
    and (b) floor the variance so a dim that was rail-locked in the source stage (var ~ 0 ->
    normalized = raw / sqrt(var) ~ raw*1e4, clipped to +-10 = binarized) is readable from the
    first batch. Both are one-directional (min for count, max for var); mean is untouched.
    Operates on obs_rms (and ret_rms for hygiene, though norm_reward is off)."""
    rms = venv.obs_rms
    if count_cap > 0:
        rms.count = min(float(rms.count), float(count_cap))
    if var_floor > 0:
        np.maximum(rms.var, float(var_floor), out=rms.var)
    ret = getattr(venv, "ret_rms", None)
    if ret is not None and count_cap > 0:
        ret.count = min(float(ret.count), float(count_cap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="m2_sprint", choices=sorted(PRESETS))
    ap.add_argument("--config", default=None,
                    help="resolved_config.json from a previous run (overrides --preset)")
    ap.add_argument("--name", default=None, help="run folder name (default: the preset)")
    ap.add_argument("--steps", type=int, default=None,
                    help="TOTAL timesteps for the run (also the target when resuming)")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--subproc", action="store_true", help="use SubprocVecEnv (true parallelism)")
    ap.add_argument("--resume", default=None,
                    help="continue THIS run: a checkpoint .zip, or 'auto' = newest checkpoint in "
                         "the run dir (starts fresh if none — safe as a requeued cluster command)")
    ap.add_argument("--warm-start", default=None,
                    help="initialize from ANOTHER run's checkpoint .zip (milestone chaining); "
                         "fresh step counter + ramps, weights and obs-normalization carried over")
    ap.add_argument("--description", default=None, help="free-text note stored in the run dir")
    ap.add_argument("--no-progress", action="store_true", help="disable the progress bar (logs)")
    args = ap.parse_args()

    if args.config:
        d = json.loads(Path(args.config).read_text())
        if "config" not in d:
            raise SystemExit(f"[train] --config {args.config}: no top-level 'config' key "
                             "(expected a resolved_config.json written by train.py)")
        cfg = config_from_dict(d["config"])
        # the sibling values recorded at train time are the defaults; CLI still wins
        n_envs = args.n_envs or d.get("n_envs") or cfg.n_envs
        total = args.steps or d.get("total_steps") or cfg.total_steps
        default_name = Path(args.config).parent.name
    else:
        cfg = get_config(args.preset)
        n_envs = args.n_envs or cfg.n_envs
        total = args.steps or cfg.total_steps
        default_name = args.preset
    name = args.name or default_name
    run = PKG_DIR / "runs" / name
    run.mkdir(parents=True, exist_ok=True)

    # resolve resume/warm-start BEFORE touching the run dir (fail fast on a bad path)
    resume_ckpt = None
    if args.resume:
        if args.resume == "auto":
            resume_ckpt = latest_checkpoint(run)
            if resume_ckpt is None:
                print(f"[train] --resume auto: no checkpoint in {run}, starting fresh")
        else:
            resume_ckpt = Path(args.resume)
            if not resume_ckpt.exists():
                raise SystemExit(f"[train] --resume checkpoint not found: {resume_ckpt}")
    warm_ckpt = None
    if args.warm_start and resume_ckpt is None:
        warm_ckpt = Path(args.warm_start)
        if not warm_ckpt.exists():
            raise SystemExit(f"[train] --warm-start checkpoint not found: {warm_ckpt}")

    # stale-checkpoint guard: resuming into a run dir whose recorded config differs from the
    # requested one silently trains a MIXED config (PPO.load keeps the checkpoint's own
    # hyperparameters while the envs use the new cfg) — refuse instead. Use --name for a new run.
    rc_path = run / "resolved_config.json"
    if resume_ckpt is not None and rc_path.exists():
        old_cfg = config_from_dict(json.loads(rc_path.read_text()).get("config", {}))
        if old_cfg != cfg:
            from dataclasses import fields as _fields
            diff = [f.name for f in _fields(cfg)
                    if getattr(cfg, f.name) != getattr(old_cfg, f.name)]
            raise SystemExit(f"[train] refusing to resume: run '{name}' was trained with a "
                             f"different config (differs in: {diff}). Use a fresh --name, or "
                             f"pass --config {rc_path} to reuse the recorded config.")
    rc_path.write_text(json.dumps(
        {"config": config_to_dict(cfg), "n_envs": n_envs, "total_steps": total,
         "preset": args.preset}, indent=1))
    if args.description:
        (run / "description.txt").write_text(args.description)

    vec_cls = SubprocVecEnv if (args.subproc and n_envs > 1) else DummyVecEnv
    base_venv = vec_cls([make_env(cfg) for _ in range(n_envs)])

    # NOTE: reward normalization is OFF — the reward is hand-balanced in raw units and
    # suicide-proofed against the raw fall penalty. Observation normalization stays ON.
    def fresh_vecnorm():
        return VecNormalize(base_venv, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, gamma=cfg.gamma)

    reset_counter = True
    if resume_ckpt is not None:
        vn = matching_vecnormalize(resume_ckpt)
        if vn.exists():
            venv = VecNormalize.load(str(vn), base_venv)
            venv.training = True
            venv.norm_reward = False
            print(f"[train] resumed VecNormalize stats <- {vn.name}")
        else:
            venv = fresh_vecnorm()
            print(f"[train] WARNING: no VecNormalize stats at {vn}; starting normalization fresh")
        model = PPO.load(str(resume_ckpt), env=venv)
        reset_counter = False        # num_timesteps continues -> ramps + lr schedule continue
        print(f"[train] resumed {resume_ckpt.name} at {model.num_timesteps} steps")
        # rotate the CSV log: SB3's configure() reopens progress.csv in WRITE mode, wiping the
        # previous segments' history (and the plots with it) — keep each segment; plot_training
        # merges progress*.csv.
        if (run / "progress.csv").exists():
            seg = run / f"progress.{model.num_timesteps}.csv"
            if not seg.exists():
                (run / "progress.csv").rename(seg)
    else:
        if warm_ckpt is not None:
            vn = matching_vecnormalize(warm_ckpt)
            if vn.exists():
                venv = VecNormalize.load(str(vn), base_venv)
                venv.training = True
                venv.norm_reward = False
                print(f"[train] warm-start VecNormalize stats <- {vn}")
                # a milestone hop frees a base DOF -> a formerly-constant obs dim starts varying,
                # but the loaded stats carry ~zero variance + a huge count on it (glacial adapt,
                # signal clipped at +-10). Rejuvenate so it's readable and adapts fast.
                if cfg.warmstart_obs_count_cap > 0 or cfg.warmstart_var_floor > 0:
                    rejuvenate_obs_rms(venv, cfg.warmstart_obs_count_cap, cfg.warmstart_var_floor)
                    print(f"[train] rejuvenated obs_rms: count<= {cfg.warmstart_obs_count_cap:g}, "
                          f"var>= {cfg.warmstart_var_floor:g}")
            else:
                venv = fresh_vecnorm()
            # PPO.load keeps the checkpoint's OWN hyperparameters (gamma, schedules); milestones
            # share them by design. Fresh step counter: ramps and lr restart for the new stage.
            model = PPO.load(str(warm_ckpt), env=venv)
            model.num_timesteps = 0
            # the source stage's ent_coef is typically fully annealed (0.002) — a new milestone
            # needs its exploration back; the EntropyCallback re-anneals once competent again.
            model.ent_coef = cfg.ent_coef
            print(f"[train] warm-started weights <- {warm_ckpt} (ent_coef reset to {cfg.ent_coef})")
        else:
            venv = fresh_vecnorm()
            lr = lambda p: cfg.lr_final + p * (cfg.learning_rate - cfg.lr_final)  # p: 1 -> 0
            model = PPO(
                "MlpPolicy", venv,
                n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
                gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, learning_rate=lr,
                clip_range=cfg.clip_range, ent_coef=cfg.ent_coef, target_kl=cfg.target_kl,
                policy_kwargs=dict(net_arch=list(cfg.policy_hidden)),
                seed=cfg.seed, verbose=1,
            )

    model.set_logger(configure(str(run), ["stdout", "csv", "tensorboard"]))

    cb_list = [
        CheckpointCallback(save_freq=max(1_000_000 // n_envs, 1), save_path=str(run),
                           name_prefix="ppo", save_vecnormalize=True),
        RewardTermCallback(),
        EntropyCallback(cfg, run_dir=run),
        PlotCallback(run, every_steps=500_000),
    ]
    if cfg.objective == "sprint" and cfg.sprint_curriculum_steps > 0 \
            and cfg.sprint_dist_m > cfg.sprint_dist_start_m:
        cb_list.append(RampCallback("sprint_dist_m", "set_sprint_dist",
                                    cfg.sprint_dist_start_m, cfg.sprint_dist_m,
                                    cfg.sprint_curriculum_steps, run))
    gate = cfg.curriculum_gate_ep_len
    if cfg.gait_curriculum_steps > 0 and cfg.w_phase_contact > 0:
        if gate > 0:
            cb_list.append(GatedRampCallback("stance_ratio", "set_stance_ratio",
                                             cfg.stance_ratio_start, cfg.stance_ratio_final,
                                             cfg.gait_curriculum_steps, run, gate))
        else:
            cb_list.append(RampCallback("stance_ratio", "set_stance_ratio",
                                        cfg.stance_ratio_start, cfg.stance_ratio_final,
                                        cfg.gait_curriculum_steps, run))
    if cfg.efficiency_ramp_steps > 0:
        if gate > 0:
            cb_list.append(GatedRampCallback("eff_scale", "set_efficiency_scale",
                                             0.0, 1.0, cfg.efficiency_ramp_steps, run, gate))
        else:
            cb_list.append(RampCallback("eff_scale", "set_efficiency_scale",
                                        0.0, 1.0, cfg.efficiency_ramp_steps, run))
    callbacks = CallbackList(cb_list)

    # SB3 semantics: with reset_num_timesteps=False, learn() ADDS its total_timesteps argument
    # to the restored counter — pass the REMAINING budget so --steps stays the absolute target
    # for the whole (possibly chained/requeued) run instead of growing by `total` every restart.
    budget = total if reset_counter else max(0, total - model.num_timesteps)
    print(f"[train] preset={args.preset} n_envs={n_envs} ({vec_cls.__name__}) "
          f"target={total} remaining={budget} device={model.device} -> {run}")
    if budget > 0:
        model.learn(total_timesteps=budget, callback=callbacks,
                    reset_num_timesteps=reset_counter, progress_bar=not args.no_progress)
    else:
        print("[train] target already reached — nothing to train, refreshing final artifacts")
    model.save(run / "final_model")
    venv.save(str(run / "vecnormalize.pkl"))
    try:
        from plot_training import plot_run
        plots = plot_run(run)
    except Exception as e:
        plots = f"(skipped: {e})"
    print(f"[train] done -> {run / 'final_model.zip'}  plots -> {plots}")


if __name__ == "__main__":
    main()
