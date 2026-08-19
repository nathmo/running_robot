"""Compare the foot-shape RL ladders against ladder3 at MATCHED conditions.

WHY NOT JUST READ progress.csv. Because the curricula are ADAPTIVE and they did not land in the
same place, so the final training ep_len of two arms is not the same measurement:

    arm         m5 ep_len   dr_scale   eff_scale
    ladder3      404 / 386   0.16/0.24  0.019/0.028
    blade        361 / 379   0.52/0.46  0.060/0.054
    flat         403 / 394   0.33/0.75  0.039/0.087

Equal episode length while carrying 2-3x the domain randomisation is not equal performance, and the
control is the arm that had to back off furthest to get there. Reading those columns as an A/B
would understate both feet. (This is the same trap the ladder3 write-up caught in the other
direction: a training-time peak that was a measurement at an EASIER curriculum point, not a better
policy.)

WHY NOT JUST RUN evaluate.py. Because it deliberately restores each run's OWN curriculum state
(cmd_scale, drive bandwidth, sprint distance, eff_scale) so the printed reward terms match what that
policy was actually optimizing. That is the right thing for reading one run and the wrong thing for
comparing three: each arm would be graded on its own task. Here every arm is put on the SAME task.

WHAT IS HELD FIXED, and why each one:
  dr_scale = 0        the plant is the question; randomisation is training noise
  cmd_scale           forced to the MINIMUM across the arms being compared, so no arm is handed
                      commands from outside its training distribution (evaluating a policy trained
                      to 0.0 at 0.4 is the documented way to make a good policy look broken)
  drive bandwidth     restored from each run's curriculum.json -- this is PLANT, not task, and all
                      these runs trained at a fixed 3 Hz with no drive curriculum, so it is the
                      same number everywhere; restoring it is a guard, not a knob
  eff_scale/stance    left alone: they scale REWARD TERMS only and cannot move survival
  seeds               the same list for every arm -- paired, because outcomes here are bimodal and
                      an unpaired sample once invented a 27x effect that does not exist

    python training/foot_rl_ab.py --rungs m3 m5 --episodes 8
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from stable_baselines3 import PPO                                    # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize   # noqa: E402

from config import config_from_dict                                  # noqa: E402
from env import DashEnv                                              # noqa: E402

ARMS = ["ladder3", "foot_blade", "foot_flat"]
LABEL = {"ladder3": "sphere (control)", "foot_blade": "blade", "foot_flat": "flat plate"}


def run_dir(root, arm, rung, seed):
    return Path(root) / f"{arm}_{rung}_s{seed}"


def curriculum(run):
    p = run / "curriculum.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load(run, cmd_scale):
    cfg = config_from_dict(json.loads((run / "resolved_config.json").read_text())["config"])
    cfg.push_interval_s = 0.0          # shoves are a training disturbance, not part of the measure
    cfg.trip_prob = 0.0
    raw = DashEnv(cfg)
    raw.set_dr_scale(0.0)
    cur = curriculum(run)
    if "drive_bw_log10" in cur:        # PLANT: same for every arm here, restored as a guard
        raw.set_drive_bandwidth_log10(cur["drive_bw_log10"])
    raw.set_cmd_scale(cmd_scale)       # TASK: forced common across arms
    venv = DummyVecEnv([lambda: raw])
    ckpt = run / "final_model.zip"
    if not ckpt.exists():
        cks = sorted(run.glob("ppo_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
        if not cks:
            raise FileNotFoundError(f"no checkpoint in {run}")
        ckpt = cks[-1]
    vn = run / "vecnormalize.pkl"
    if not vn.exists():
        vns = sorted(run.glob("ppo_vecnormalize_*_steps.pkl"),
                     key=lambda p: int(p.stem.split("_")[2]))
        vn = vns[-1] if vns else None
    if vn is not None and vn.exists():
        venv = VecNormalize.load(str(vn), venv)
        venv.training = False
        venv.norm_reward = False
    model = PPO.load(str(ckpt)[:-4], custom_objects={"learning_rate": 0.0,
                                                     "lr_schedule": lambda _: 0.0,
                                                     "clip_range": lambda _: 0.2})
    return model, venv, raw, ckpt.name


def episode(model, venv, raw, seed):
    venv.seed(seed)
    obs = venv.reset()
    x0 = float(raw.data.qpos[0])
    n, ret = 0, 0.0
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = venv.step(a)
        ret += float(r[0])
        n += 1
        if done[0]:
            break
    return n, ret, float(raw.data.qpos[0]) - x0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PKG_DIR / "runs"))
    ap.add_argument("--rungs", nargs="*", default=["m2", "m3", "m4", "m5"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ep_seeds = list(range(args.episodes))
    out = {}
    for rung in args.rungs:
        present = [(a, s) for a in ARMS for s in args.seeds
                   if (run_dir(args.root, a, rung, s) / "resolved_config.json").exists()]
        if not present:
            continue
        # one common command scale for the whole rung: the smallest any arm was trained to
        cmds = [curriculum(run_dir(args.root, a, rung, s)).get("cmd_scale", 0.0)
                for a, s in present]
        cmd = min(cmds)
        print(f"\n=== {rung}   cmd_scale forced to {cmd:.2f} "
              f"(arms trained to {sorted(set(round(c, 2) for c in cmds))}), dr=0, greedy, "
              f"{args.episodes} paired episodes")
        print(f"{'arm':20s}{'sd':>3s}{'survived':>10s}{'ep_len med':>12s}{'mean':>9s}"
              f"{'dist med':>10s}{'return':>9s}   checkpoint")
        for arm, s in present:
            run = run_dir(args.root, arm, rung, s)
            model, venv, raw, ck = load(run, cmd)
            res = [episode(model, venv, raw, sd) for sd in ep_seeds]
            n = [r[0] for r in res]
            surv = sum(1 for r in res if r[0] >= raw.max_steps) if hasattr(raw, "max_steps") else 0
            out[f"{rung}/{arm}_s{s}"] = dict(ep_len=n, ret=[r[1] for r in res],
                                             dist=[r[2] for r in res], cmd_scale=cmd)
            print(f"{LABEL[arm]:20s}{s:>3d}{surv:>7d}/{len(n):<2d}"
                  f"{statistics.median(n):12.1f}{np.mean(n):9.1f}"
                  f"{statistics.median(r[2] for r in res):10.2f}"
                  f"{np.mean([r[1] for r in res]):9.1f}   {ck}")
            venv.close()
        rows = {k: v for k, v in out.items() if k.startswith(f"{rung}/")}
        allv = [x for v in rows.values() for x in v["ep_len"]]
        print(f"  spread across ALL episodes at this rung: {min(allv)} .. {max(allv)} "
              f"— a per-arm difference smaller than this is not a result")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
