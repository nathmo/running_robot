"""Evaluate / visualize a trained DASH-01 policy.

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
import time
from pathlib import Path
import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .config import get_config
from .env import Dash01Env


def infer_preset(run: Path):
    """The preset recorded at train time (preset.json), else the longest preset name contained
    in the run folder name (rl/runs/m2_walk_cont -> m2_walk). Evaluating an m2 policy under the
    old m1_stand default silently showed a standing robot with near-zero 'tracking error'."""
    import json
    from .config import PRESETS
    rec = run / "preset.json"
    if rec.exists():
        p = json.loads(rec.read_text()).get("preset")
        if p in PRESETS:
            return p
    hits = [p for p in PRESETS if p != "default" and p in run.name]
    if not hits:
        raise SystemExit(f"cannot infer a preset from run name '{run.name}'; pass --preset")
    return max(hits, key=len)


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


def load_run_config(run: Path, preset=None):
    """Rebuild the env config for a run. Precedence: an explicit --preset; else the full
    resolved config recorded at train time (resolved_config.json — required for
    experiment runs whose overrides changed env-shaping fields); else preset.json /
    name inference (legacy runs)."""
    if preset:
        return get_config(preset)
    rc = run / "resolved_config.json"
    if rc.exists():
        import json
        from .config import config_from_dict
        d = json.loads(rc.read_text())
        return config_from_dict(d.get("config", d))
    return get_config(infer_preset(run))


def build(run: Path, preset, checkpoint):
    cfg = load_run_config(run, preset)
    cfg.push_interval_s = 0.0   # random shoves are a TRAINING disturbance; evaluation, videos and
    #                             the gait-probe gates must measure the policy, not push recovery
    raw = Dash01Env(cfg)
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
    ap.add_argument("--preset", default=None, help="default: inferred from the run name")
    ap.add_argument("--checkpoint", default=None, help="model path w/o .zip (default: final / latest)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--vx", type=float, default=None, help="fixed forward command in [-1,1]")
    ap.add_argument("--yaw", type=float, default=None, help="fixed yaw command in [-1,1]")
    ap.add_argument("--video", default=None, help="path to write an mp4 of episode 1")
    ap.add_argument("--seconds", type=float, default=None,
                    help="max video length (default: the whole episode — a full 100 m dash)")
    ap.add_argument("--viewer", action="store_true", help="live passive viewer (local only)")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions like training does (a policy with a large std behaves "
                         "differently deterministic vs sampled — check the gap before hardware)")
    args = ap.parse_args()
    run = Path(args.run)
    # None -> build() prefers the run's resolved_config.json, then preset inference
    model, venv, raw = build(run, args.preset, args.checkpoint)
    fixed = args.vx is not None or args.yaw is not None

    if args.viewer:
        from mujoco import viewer as mjviewer
        obs = venv.reset()
        if fixed:
            set_command(raw, args.vx or 0.0, args.yaw or 0.0)
        with mjviewer.launch_passive(raw.model, raw.data) as v:
            # sync + pace once per 50 Hz CONTROL step via the env hook — in fourier mode one
            # env.step() replays a whole gait cycle, so syncing per step() jumped a cycle a frame
            t_next = [time.perf_counter()]

            def on_ctrl():
                v.sync()
                t_next[0] += raw.control_dt
                lag = t_next[0] - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
                else:
                    t_next[0] = time.perf_counter()   # slow frame: don't bank a fast-forward debt
            raw.on_control_step = on_ctrl
            while v.is_running():
                a, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = venv.step(a)
                if fixed:
                    set_command(raw, args.vx or 0.0, args.yaw or 0.0)
                if done[0]:
                    obs = venv.reset()
                    if fixed:
                        set_command(raw, args.vx or 0.0, args.yaw or 0.0)
        return

    renderer, writer = None, None
    if args.video:
        try:
            import imageio.v2 as imageio
        except ImportError as e:
            raise SystemExit(f"--video needs imageio (+imageio-ffmpeg): {e}")
        renderer = mujoco.Renderer(raw.model, 480, 640)
        # STREAM frames to the encoder: a whole-episode video (60 s dash = 3000 frames) is ~2.7 GB
        # as a raw in-memory frame list — do not buffer it.
        writer = imageio.get_writer(args.video, fps=int(round(1 / raw.control_dt)))
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(raw.model, cam)
    cam.distance, cam.elevation = 2.5, -15

    # frames + vx error are sampled per 50 Hz CONTROL step via the env hook. In fourier mode one
    # env.step() replays a whole gait cycle, so the old per-step() capture produced one frame per
    # CYCLE played back at 50 fps — the 'twitching robot' video — and cycle-boundary-only metrics.
    lengths, returns, vx_err = [], [], []
    n_ctrl = [0]              # control steps so far (== ep_len bookkeeping across both modes)
    capture = [False]
    n_frames = [0]
    video_s = args.seconds if args.seconds is not None else raw.cfg.episode_s
    max_frames = int(video_s / raw.control_dt)

    def on_ctrl():
        n_ctrl[0] += 1
        v_body = raw._base_rot().T @ raw.data.qvel[0:3]
        vx_err.append(abs(raw._command[0] * raw.cfg.vx_max - v_body[0]))
        if capture[0] and writer is not None and n_frames[0] < max_frames:
            cam.lookat[:] = raw.data.qpos[:3]           # follow-cam: tracks the base the whole run
            renderer.update_scene(raw.data, cam)
            writer.append_data(renderer.render())
            n_frames[0] += 1
    raw.on_control_step = on_ctrl

    sprints = []
    for ep in range(args.episodes):
        obs = venv.reset()
        if fixed:
            set_command(raw, args.vx or 0.0, args.yaw or 0.0)
        capture[0] = ep == 0
        ep_start = n_ctrl[0]
        done, ep_ret, sprint = [False], 0.0, None
        while not done[0]:
            a, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, r, done, info = venv.step(a)
            if fixed:
                set_command(raw, args.vx or 0.0, args.yaw or 0.0)
            ep_ret += float(r[0])   # raw env reward incl. fall penalty (norm_reward off at eval)
            sprint = info[0].get("sprint", sprint)
        lengths.append(n_ctrl[0] - ep_start)   # control steps, comparable across pd/fourier
        returns.append(ep_ret)
        if sprint:
            sprints.append(sprint)
    print(f"episodes={args.episodes}  ep_len {np.mean(lengths):.0f}+/-{np.std(lengths):.0f} "
          f"control steps (max {raw.max_steps})  mean_return {np.mean(returns):.1f}  "
          f"mean |vx-cmd| {np.mean(vx_err):.3f} m/s")
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
