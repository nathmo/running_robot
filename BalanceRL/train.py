"""Train the balance policy.

    python BalanceRL/train.py --preset balance --name bal_s0 --seed 0 --devices 2
    python BalanceRL/train.py --preset smoke --name smoke            # CPU, a few iterations

Writes to BalanceRL/runs/<name>/: progress.jsonl (one line per iteration), ckpt_<step>.{msgpack,json}
every cfg.ckpt_every iterations, best.{msgpack,json} by the greedy push-ladder score, final.*.
--resume auto picks up the newest checkpoint in the run directory (the SLURM chain relies on it).
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                                                    # noqa: E402

from config import get_config, config_to_dict                          # noqa: E402


def newest_ckpt(run):
    c = sorted(run.glob("ckpt_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if (run / "final.json").exists():
        return run / "final"
    return c[-1].with_suffix("") if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="balance")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--eval-envs", type=int, default=240)
    ap.add_argument("--resume", default=None, help="'auto' or a checkpoint path (no suffix)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (python literal)")
    a = ap.parse_args()

    over = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        import ast
        over[k] = ast.literal_eval(v)
    if a.n_envs:
        over["n_envs"] = a.n_envs
    if a.n_steps:
        over["n_steps"] = a.n_steps
    if a.steps:
        over["total_steps"] = a.steps
    cfg = get_config(a.preset, **over)

    import jax
    from ppo import Trainer
    run = HERE / "runs" / a.name
    tr = Trainer(cfg, run, seed=a.seed, n_devices=a.devices, n_eval=a.eval_envs)
    if a.resume:
        ck = newest_ckpt(run) if a.resume == "auto" else Path(a.resume)
        if ck is not None:
            side = tr.load(ck)
            print(f"[train] resumed {ck} at step {tr.step:,}, curriculum {tr.cur}")
            if ck.name == "final":
                print("[train] run already finished")
                return
    (run / "config.json").write_text(json.dumps(config_to_dict(cfg), indent=1))
    print(f"[train] {a.name}: {jax.devices()} x{a.devices}, batch {tr.batch:,} "
          f"({cfg.n_envs} envs x {cfg.n_steps}), {tr.n_mb} minibatches, actor {tr.env.actor_dim} "
          f"obs {tr.env.obs_dim}, action {tr.env.action_dim}, budget {cfg.total_steps:,}")
    prog = open(run / "progress.jsonl", "a")
    t_start = time.time()
    while tr.step < cfg.total_steps:
        log, moved = tr.iterate()
        n = tr.rollout_n
        if moved:
            print(f"[curriculum] @ {tr.step:,}: {moved.strip()}")
        if n % cfg.eval_every == 0 and tr.eval_env is not None:
            ev = tr.evaluate()
            log["eval/score"] = ev["score"]
            log["eval/quiet_falls"] = ev["quiet_falls"]
            for k, v in ev["survival"].items():
                log[f"eval/surv_{k}"] = v
            if ev["score"] > tr.cur["best_score"]:
                tr.cur["best_score"] = ev["score"]
                tr.save("best", extra=dict(eval=ev))
            print(f"[eval] @ {tr.step:,}: score {ev['score']:.3f} {ev['survival']} quiet {ev['quiet_falls']}"
                  f"  (best {tr.cur['best_score']:.3f})")
        log["curriculum/bins"] = tr.survival_bins()
        prog.write(json.dumps(log) + "\n")
        prog.flush()
        if n % 10 == 0 or n <= 3:
            print(f"[{tr.step/1e6:7.2f}M] sps {log['time/sps']:,.0f} ep_len {log['rollout/ep_len_mean']:.0f} "
                  f"falls {log['rollout/falls']} quiet {log['rollout/quiet_falls']} "
                  f"hard_surv {log['rollout/hard_survival']:.2f} | plant {tr.cur['plant_scale']:.2f} "
                  f"push {tr.cur['push_level']:.3f} | kl {log['train/kl']:.4f} lr {tr.lr:.1e} "
                  f"std {log['train/std_mean']:.3f} vf {log['train/vf']:.3f} kp {log['diag/kp_mean']:.0f} "
                  f"kd {log['diag/kd_mean']:.2f}", flush=True)
        if n % cfg.ckpt_every == 0:
            tr.save(f"ckpt_{tr.step}")
    tr.save("final")
    print(f"[train] done: {tr.step:,} steps in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
