"""Evaluate / visualize a trained SpiderBot policy.

  # metrics over several episodes (headless)
  .venv/Scripts/python.exe -m rl.evaluate --run rl/runs/m1_stand --episodes 5
  # record an mp4 (headless offscreen render)
  .venv/Scripts/python.exe -m rl.evaluate --run rl/runs/m1_stand --video rl/runs/m1_stand/rollout.mp4
  # live viewer (run locally, not headless)
  .venv/Scripts/python.exe -m rl.evaluate --run rl/runs/m1_stand --viewer
  # drive a fixed joystick command (normalized -1..1)
  .venv/Scripts/python.exe -m rl.evaluate --run rl/runs/m2_walk --vx 0.7 --yaw 0.0 --viewer
"""
import argparse
from pathlib import Path
import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .config import get_config
from .env import SpiderBotEnv


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
    name = Path(model_path).name  # e.g. ppo_1999980_steps
    if name.startswith("ppo_") and name.endswith("_steps"):
        cand = run / f"ppo_vecnormalize_{name[4:]}.pkl"
        if cand.exists():
            return cand
    vns = sorted(run.glob("ppo_vecnormalize_*_steps.pkl"), key=lambda p: int(p.stem.split("_")[2]))
    if vns:
        return vns[-1]
    return run / "vecnormalize.pkl"


def build(run: Path, preset, checkpoint):
    raw = SpiderBotEnv(get_config(preset))
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


def set_command(raw, vx, yaw):
    """Pin a fixed joystick command (disables in-episode resampling)."""
    raw._resample_every = 10 ** 9
    raw._command[:] = [vx, yaw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--preset", default="m1_stand")
    ap.add_argument("--checkpoint", default=None, help="model path w/o .zip (default: final / latest)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--vx", type=float, default=None, help="fixed forward command in [-1,1]")
    ap.add_argument("--yaw", type=float, default=None, help="fixed yaw command in [-1,1]")
    ap.add_argument("--video", default=None, help="path to write an mp4 of episode 1")
    ap.add_argument("--seconds", type=float, default=8.0, help="max video length")
    ap.add_argument("--viewer", action="store_true", help="live passive viewer (local only)")
    args = ap.parse_args()
    run = Path(args.run)

    model, venv, raw = build(run, args.preset, args.checkpoint)
    fixed = args.vx is not None or args.yaw is not None

    if args.viewer:
        from mujoco import viewer as mjviewer
        obs = venv.reset()
        if fixed:
            set_command(raw, args.vx or 0.0, args.yaw or 0.0)
        with mjviewer.launch_passive(raw.model, raw.data) as v:
            while v.is_running():
                a, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = venv.step(a)
                if fixed:
                    set_command(raw, args.vx or 0.0, args.yaw or 0.0)
                v.sync()
                if done[0]:
                    obs = venv.reset()
                    if fixed:
                        set_command(raw, args.vx or 0.0, args.yaw or 0.0)
        return

    renderer = None
    frames = []
    if args.video:
        renderer = mujoco.Renderer(raw.model, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(raw.model, cam)
    cam.distance, cam.elevation = 2.5, -15

    lengths, returns, vx_err = [], [], []
    for ep in range(args.episodes):
        obs = venv.reset()
        if fixed:
            set_command(raw, args.vx or 0.0, args.yaw or 0.0)
        done, ep_len, ep_ret = [False], 0, 0.0
        while not done[0]:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = venv.step(a)
            if fixed:
                set_command(raw, args.vx or 0.0, args.yaw or 0.0)
            ep_len += 1
            ep_ret += float(info[0].get("reward_terms") and sum(info[0]["reward_terms"].values()) or r[0])
            v_body = raw._base_rot().T @ raw.data.qvel[0:3]
            vx_err.append(abs(raw._command[0] * raw.cfg.vx_max - v_body[0]))
            if ep == 0 and renderer is not None and len(frames) < args.seconds / raw.control_dt:
                cam.lookat[:] = raw.data.qpos[:3]
                renderer.update_scene(raw.data, cam)
                frames.append(renderer.render())
        lengths.append(ep_len)
        returns.append(ep_ret)
    print(f"episodes={args.episodes}  ep_len {np.mean(lengths):.0f}+/-{np.std(lengths):.0f} "
          f"(max {raw.max_steps})  mean_return {np.mean(returns):.1f}  "
          f"mean |vx-cmd| {np.mean(vx_err):.3f} m/s")

    if args.video and frames:
        try:
            import imageio.v2 as imageio
            imageio.mimsave(args.video, frames, fps=int(1 / raw.control_dt))
            print(f"wrote {args.video} ({len(frames)} frames)")
        except Exception as e:
            from PIL import Image
            Image.fromarray(frames[len(frames) // 2]).save(args.video.replace(".mp4", ".png"))
            print(f"(imageio unavailable: {e}); saved a single PNG instead")


if __name__ == "__main__":
    main()
