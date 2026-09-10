"""Train a DASH-01 Walker v2 policy: PPO in JAX over N MJX environments on one GPU.

    # pipeline check (CPU, minutes)
    python walk_v2/train.py --preset v2_smoke --steps 4000 --n-envs 8
    # the run (one RTX PRO 6000 / B200; ~4096 envs)
    python walk_v2/train.py --preset v2_s1_planar --steps 300000000 --name v2_s1_planar_s0
    # S1 -> S2 warm start (identical obs/action widths, §11)
    python walk_v2/train.py --preset v2_s2_free --warm-start walk_v2/runs/v2_s1_planar_s0/final.msgpack
    # cluster requeue-safe: the same command is correct for the first start and every restart
    python walk_v2/train.py --preset v2_s2_free --name v2_s2_free_s0 --resume auto

Outputs -> walk_v2/runs/<name>/: ckpt_<steps>.msgpack (+ .json with the curriculum state),
final.msgpack, resolved_config.json, progress.csv, eval.csv, training_plots.png, tb/ (if
tensorboardX is installed).
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import jax

from config import PRESETS, get_config, config_from_dict, config_to_dict
from env import DashEnvV2
from ppo import PPO


# config fields a resumed run may change: they schedule evaluation/checkpoints, not the learning
RESUME_FREE_FIELDS = {"eval_every_rollouts", "checkpoint_every_steps"}


def latest_checkpoint(run: Path):
    ck = sorted(run.glob("ckpt_*.msgpack"), key=lambda p: int(p.stem.split("_")[1]))
    if ck:
        return ck[-1]
    f = run / "final.msgpack"
    return f if f.exists() else None


class CsvLog:
    def __init__(self, path):
        self.path = Path(path)
        self.keys = None
        self.f = None

    def write(self, row):
        if self.keys is None:
            self.keys = list(row.keys())
            new = not self.path.exists() or self.path.stat().st_size == 0
            self.f = open(self.path, "a", newline="")
            self.w = csv.DictWriter(self.f, fieldnames=self.keys, extrasaction="ignore")
            if new:
                self.w.writeheader()
        self.w.writerow({k: row.get(k, "") for k in self.keys})
        self.f.flush()


def make_eval_env(cfg, n=16, keep_assist=False):
    """The greedy-eval plant: nominal, no noise, no disturbances, nominal delay, no assist."""
    from dataclasses import replace
    c = replace(cfg, dr_enable=False, obs_noise_enable=False, push_interval_s=0.0,
                wind_force_max=0.0, wind_gust_n=0.0, trip_prob=0.0, thermal_hot_start_max=0.0,
                pitch_assist_kp=cfg.pitch_assist_kp if keep_assist else 0.0,
                roll_assist_kp=cfg.roll_assist_kp if keep_assist else 0.0)
    return DashEnvV2(c, n_envs=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v2_s1_planar", choices=sorted(PRESETS))
    ap.add_argument("--config", default=None, help="resolved_config.json of a previous run")
    ap.add_argument("--name", default=None)
    ap.add_argument("--steps", type=int, default=None, help="TOTAL env steps (absolute target)")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--devices", type=int, default=1, help="data-parallel GPUs (n_envs split across them)")
    ap.add_argument("--n-steps", type=int, default=None, help="rollout length per env")
    ap.add_argument("--resume", default=None, help="'auto' or a ckpt_*.msgpack of THIS run")
    ap.add_argument("--warm-start", default=None, help="another run's .msgpack (weights + obs stats)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval-envs", type=int, default=16)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--description", default=None)
    args = ap.parse_args()

    if args.config:
        d = json.loads(Path(args.config).read_text())
        cfg = config_from_dict(d["config"])
        default_name = Path(args.config).parent.name
    else:
        cfg = get_config(args.preset)
        default_name = args.preset
    if args.n_envs:
        cfg.n_envs = int(args.n_envs)
    if args.n_steps:
        cfg.n_steps = int(args.n_steps)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    total = int(args.steps or cfg.total_steps)
    name = args.name or default_name
    run = PKG_DIR / "runs" / name
    run.mkdir(parents=True, exist_ok=True)

    resume_ckpt = None
    if args.resume:
        resume_ckpt = latest_checkpoint(run) if args.resume == "auto" else Path(args.resume)
        if resume_ckpt is None:
            print(f"[train] --resume auto: no checkpoint in {run}, starting fresh")
        elif not resume_ckpt.exists():
            raise SystemExit(f"[train] --resume checkpoint not found: {resume_ckpt}")
    warm = Path(args.warm_start) if (args.warm_start and resume_ckpt is None) else None
    if warm is not None and not warm.exists():
        raise SystemExit(f"[train] --warm-start not found: {warm}")

    rc = run / "resolved_config.json"
    if resume_ckpt is not None and rc.exists():
        old = config_from_dict(json.loads(rc.read_text())["config"])
        if old != cfg:
            from dataclasses import fields
            diff = [f.name for f in fields(cfg) if getattr(cfg, f.name) != getattr(old, f.name)]
            free = [f for f in diff if f in RESUME_FREE_FIELDS]     # bookkeeping only, not the learning
            diff = [f for f in diff if f not in RESUME_FREE_FIELDS]
            if diff:
                raise SystemExit(f"[train] refusing to resume '{name}' with a different config "
                                 f"(differs in {diff}); use a fresh --name or --config {rc}")
            if free:
                print(f"[train] resuming with new bookkeeping settings {free}: "
                      + ", ".join(f"{f} {getattr(old, f)} -> {getattr(cfg, f)}" for f in free))
    rc.write_text(json.dumps({"config": config_to_dict(cfg), "n_envs": cfg.n_envs,
                              "total_steps": total, "preset": args.preset}, indent=1))
    if args.description:
        (run / "description.txt").write_text(args.description)

    print(f"[train] jax {jax.__version__} devices {jax.devices()}")
    if cfg.n_envs % args.devices:
        raise SystemExit(f"[train] n_envs {cfg.n_envs} not divisible by --devices {args.devices}")
    env = DashEnvV2(cfg, n_envs=cfg.n_envs // args.devices)
    print(f"[train] env: {cfg.model_path} spec_source={cfg.spec_source} n_envs={env.n_envs}x{args.devices} "
          f"actor_obs={env.actor_dim} obs={env.obs_dim} action={env.action_dim} "
          f"control {1 / env.control_dt:.0f} Hz")
    eval_env = None if args.no_eval else make_eval_env(cfg, args.eval_envs)
    agent = PPO(cfg, env, run, total, seed=cfg.seed, eval_env=eval_env, n_devices=args.devices)
    if resume_ckpt is not None:
        agent.load(resume_ckpt)
    elif warm is not None:
        agent.load(warm, warm_start=True)

    # env reset (a resumed run restarts its episodes; the plant draws are fresh)
    agent.key, k = jax.random.split(agent.key)
    t = time.time()
    agent.reset_envs(k)
    agent.obs.block_until_ready()
    print(f"[train] reset {agent.n_envs} envs on {agent.n_dev} device(s) in {time.time() - t:.1f}s")

    best_score = None
    _bj = run / "best_eval.json"
    if _bj.exists():
        _b = json.loads(_bj.read_text())
        best_score = (int(_b.get("finishes", 0)), float(_b.get("dist_mean", 0.0)))
    log = CsvLog(run / "progress.csv")
    evlog = CsvLog(run / "eval.csv")
    tb = None
    try:
        from tensorboardX import SummaryWriter
        tb = SummaryWriter(str(run / "tb"))
    except Exception:
        pass
    last_ckpt = agent.step
    t_start = time.time()
    print(f"[train] target {total:,} steps, rollout {agent.batch:,} ({env.n_envs}x{cfg.n_steps}), "
          f"{agent.n_minibatches} minibatches of {agent.mb_size}, starting at {agent.step:,}")
    while agent.step < total:
        row = agent.iterate()
        row["time/elapsed_s"] = time.time() - t_start
        log.write(row)
        if tb is not None:
            for k_, v in row.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    tb.add_scalar(k_, v, agent.step)
        if agent.rollout_n == 1 or agent.rollout_n % 10 == 0:
            print(f"[{agent.step:>12,}] ep_len {row['rollout/ep_len_mean']:7.0f}  ret {row['rollout/ep_ret_mean']:8.1f}  "
                  f"fin {row['rollout/finishes']:3d}/{row['rollout/episodes']:<4d} rew {row['rollout/reward_mean']:+.3f}  "
                  f"kl {row['train/approx_kl']:.4f} std {row['train/std_mean']:.2f} "
                  f"f {row['diag/freq_hz_median']:.2f}Hz sat {row['diag/res_sat_sampled']:.2f}  "
                  f"{row['time/sps']:,.0f} sps", flush=True)
        if eval_env is not None and cfg.eval_every_rollouts > 0 and agent.rollout_n % cfg.eval_every_rollouts == 0:
            t = time.time()
            ev = agent.evaluate()
            ev_row = {"step": agent.step, "finishes": ev["finishes"], "falls": ev["falls"], "n": ev["n"],
                      "t_line_mean": ev["t_line_mean"], "dist_mean": ev["dist_mean"],
                      "speed_mean": ev["speed_mean"], "eval_s": time.time() - t}
            evlog.write(ev_row)
            if tb is not None:
                for k_, v in ev_row.items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        tb.add_scalar(f"eval/{k_}", v, agent.step)
            print(f"[eval @ {agent.step:,}] greedy: {ev['finishes']}/{ev['n']} finish, {ev['falls']} falls, "
                  f"t_line {ev['t_line_mean']:.2f} s, dist {ev['dist_mean']:.1f} m, "
                  f"speed {ev['speed_mean']:.2f} m/s ({time.time() - t:.0f}s)", flush=True)
            # keep the best greedy checkpoint (finishes first, then distance): a later collapse (the
            # end-of-fade cliff, v2c_s1_planar_s1 at 47 M) must not lose a usable policy
            _score = (int(ev.get("finishes", 0)), float(ev.get("dist_mean", float("nan"))))
            if _score[1] == _score[1] and (best_score is None or _score > best_score):
                best_score = _score
                agent.save(run / "best.msgpack")
                (run / "best_eval.json").write_text(json.dumps({"step": agent.step, "finishes": _score[0],
                                                           "dist_mean": _score[1], **{k: float(v) for k, v in ev.items()
                                                                                       if isinstance(v, (int, float))}}, indent=1))
                print(f"[train] best checkpoint -> best.msgpack (step {agent.step:,}, finishes {_score[0]}, dist {_score[1]:.1f} m)")

        if agent.step - last_ckpt >= cfg.checkpoint_every_steps:
            agent.save(run / f"ckpt_{agent.step}.msgpack")
            last_ckpt = agent.step
    agent.save(run / "final.msgpack")
    try:
        from plot_training import plot_run
        plot_run(run)
    except Exception as e:                       # plotting must never fail a run
        print(f"[train] plot skipped: {e!r}")
    print(f"[train] done -> {run / 'final.msgpack'}")


if __name__ == "__main__":
    main()
