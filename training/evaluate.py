"""Evaluate / visualize a trained sprint policy.

  # dash metrics over several episodes (headless)
  python training/evaluate.py --run training/runs/m2_sprint --episodes 5
  # record an mp4 of episode 1 (headless offscreen render; on a cluster: MUJOCO_GL=egl or osmesa)
  python training/evaluate.py --run training/runs/m2_sprint --video dash.mp4
  # live viewer (local only)
  python training/evaluate.py --run training/runs/m2_sprint --viewer
"""
import argparse
import json
import sys
import time
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import config_from_dict, get_config
from env import DashEnv


def pick_model(run: Path, checkpoint):
    if checkpoint:
        return checkpoint
    if (run / "final_model.zip").exists():
        return str(run / "final_model")
    ckpts = sorted(run.glob("ppo_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no model in {run}")
    return str(ckpts[-1])[:-4]


def pick_vecnormalize(run: Path, model_path: str):
    """Match VecNormalize stats to the chosen model (final, or a specific checkpoint step)."""
    if (run / "vecnormalize.pkl").exists() and model_path.endswith("final_model"):
        return run / "vecnormalize.pkl"
    name = Path(model_path).name          # e.g. ppo_1999980_steps
    if name.startswith("ppo_") and name.endswith("_steps"):
        cand = run / f"ppo_vecnormalize_{name[4:]}.pkl"
        if cand.exists():
            return cand
    vns = sorted(run.glob("ppo_vecnormalize_*_steps.pkl"), key=lambda p: int(p.stem.split("_")[2]))
    if vns:
        return vns[-1]
    return run / "vecnormalize.pkl"


def load_run_config(run: Path, preset=None):
    """Rebuild the env config for a run: an explicit --preset wins, else the resolved config
    recorded at train time (the ground truth), else the default preset."""
    if preset:
        return get_config(preset)
    rc = run / "resolved_config.json"
    if rc.exists():
        d = json.loads(rc.read_text())
        return config_from_dict(d.get("config", d))
    raise SystemExit(f"no resolved_config.json in {run}; pass --preset")


def build(run: Path, preset, checkpoint):
    cfg = load_run_config(run, preset)
    cfg.push_interval_s = 0.0   # shoves are a TRAINING disturbance; measure the policy, not
    #                             push recovery
    raw = DashEnv(cfg)
    # a mid-training checkpoint was trained at its curriculum point, not the final one — restore
    # stance_ratio / eff_scale from the run's curriculum.json so the printed reward terms match
    # what the policy was actually optimizing (the sprint line stays at the full target distance)
    cur = run / "curriculum.json"
    if cur.exists():
        d = json.loads(cur.read_text())
        if "stance_ratio" in d:
            raw.set_stance_ratio(d["stance_ratio"])
        if "eff_scale" in d:
            raw.set_efficiency_scale(d["eff_scale"])
        # a mid-training checkpoint was trained with the sprint line at its CURRICULUM distance,
        # not the final 100 m — restore it too, else the dist-to-go task obs is out of distribution
        # (evaluated at 100 m, an m2 policy trained at ~47 m reverses and collapses; at its true
        # distance it runs forward and balances ~20 s). Applies from the next reset.
        if "sprint_dist_m" in d:
            raw.set_sprint_dist(d["sprint_dist_m"])
        if "torque_scale" in d:                 # the torque-budget curriculum's tightened limit
            raw.set_torque_limit(d["torque_scale"])
        # same class of bug as sprint_dist_m above, third instance: a command policy evaluated at
        # cmd_scale 1.0 when it was only ever trained to 0.4 is being handed commands from outside
        # its training distribution, and will look far worse than it is.
        if "cmd_scale" in d:
            raw.set_cmd_scale(d["cmd_scale"])
    venv = DummyVecEnv([lambda: raw])
    model_path = pick_model(run, checkpoint)
    vn = pick_vecnormalize(run, model_path)
    if vn.exists():
        venv = VecNormalize.load(str(vn), venv)
        venv.training = False
        venv.norm_reward = False
        print(f"[eval] using {Path(model_path).name} + {vn.name}")
    model = PPO.load(model_path)
    return model, venv, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--preset", default=None, help="default: the run's resolved_config.json")
    ap.add_argument("--checkpoint", default=None, help="model path w/o .zip (default: final/latest)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--video", default=None, help="path to write an mp4 of episode 1")
    ap.add_argument("--seconds", type=float, default=None,
                    help="max video length (default: the whole episode — a full dash)")
    ap.add_argument("--viewer", action="store_true", help="live passive viewer (local only)")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions like training does (check the deterministic gap)")
    args = ap.parse_args()
    run = Path(args.run)
    model, venv, raw = build(run, args.preset, args.checkpoint)

    if args.viewer:
        from mujoco import viewer as mjviewer
        obs = venv.reset()
        with mjviewer.launch_passive(raw.model, raw.data) as v:
            t_next = [time.perf_counter()]

            def on_ctrl():
                v.sync()
                t_next[0] += raw.control_dt
                lag = t_next[0] - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
                else:
                    t_next[0] = time.perf_counter()
            raw.on_control_step = on_ctrl
            while v.is_running():
                a, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = venv.step(a)
                if done[0]:
                    obs = venv.reset()
        return

    renderer, writer = None, None
    if args.video:
        try:
            import imageio.v2 as imageio
        except ImportError as e:
            raise SystemExit(f"--video needs imageio (+imageio-ffmpeg): {e}")
        renderer = mujoco.Renderer(raw.model, 480, 640)
        # STREAM frames to the encoder: a whole-episode 60 s dash is 3000 frames — don't buffer.
        writer = imageio.get_writer(args.video, fps=int(round(1 / raw.control_dt)))
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(raw.model, cam)
    cam.distance, cam.elevation = 2.5, -15

    lengths, returns, speeds = [], [], []
    n_ctrl = [0]
    capture = [False]
    n_frames = [0]
    video_s = args.seconds if args.seconds is not None else raw.cfg.episode_s
    max_frames = int(video_s / raw.control_dt)

    def on_ctrl():
        n_ctrl[0] += 1
        speeds.append(float(raw._vel_body()[0]))
        # record across episode boundaries until the --seconds budget is filled (an early-falling
        # policy would otherwise yield a near-empty video of just its first 1-2 s episode)
        if capture[0] and writer is not None and n_frames[0] < max_frames:
            cam.lookat[:] = raw.data.qpos[:3]        # follow-cam: tracks the base the whole run
            renderer.update_scene(raw.data, cam)
            writer.append_data(renderer.render())
            n_frames[0] += 1
    raw.on_control_step = on_ctrl

    sprints = []
    for ep in range(args.episodes):
        obs = venv.reset()
        capture[0] = True
        ep_start = n_ctrl[0]
        done, ep_ret, sprint = [False], 0.0, None
        while not done[0]:
            a, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, r, done, info = venv.step(a)
            ep_ret += float(r[0])   # raw env reward incl. terminal bonus/penalty
            sprint = info[0].get("sprint", sprint)
        lengths.append(n_ctrl[0] - ep_start)
        returns.append(ep_ret)
        if sprint:
            sprints.append(sprint)
    print(f"episodes={args.episodes}  ep_len {np.mean(lengths):.0f}+/-{np.std(lengths):.0f} "
          f"control steps (max {raw.max_steps})  mean_return {np.mean(returns):.1f}  "
          f"mean vx {np.mean(speeds):.2f} m/s  peak vx {np.max(speeds):.2f} m/s")
    for i, s in enumerate(sprints):
        line = f"line {s['t_line']:6.2f} s" if s["t_line"] is not None else "line    DNF"
        stop = f"stopped at {s['t']:.2f} s" if s["finished"] else "never stopped"
        print(f"  dash ep{i}: {s['d']:6.1f} m of {s['dist_target']:.0f} m   {line}   {stop}   "
              f"avg {s['d'] / max(s['t'], 1e-9):.2f} m/s")

    if writer is not None:
        writer.close()
        print(f"wrote {args.video} ({n_frames[0]} frames, {n_frames[0] * raw.control_dt:.1f} s)")


if __name__ == "__main__":
    main()
