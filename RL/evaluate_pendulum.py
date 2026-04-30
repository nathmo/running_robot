"""Evaluate a pendulum policy in simulation.

Supports either a Stable-Baselines3 checkpoint or an exported ONNX model.
The environment can be visualized with MuJoCo's viewer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco.viewer
import numpy as np
import onnxruntime as ort
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config as cfg
from environment import InvertedPendulumEnv


def _load_stats(stats_path: Path):
    base_env = DummyVecEnv([lambda: InvertedPendulumEnv(cfg.get_config("default"))])
    loaded = VecNormalize.load(str(stats_path), base_env)
    return loaded.obs_rms.mean, loaded.obs_rms.var, loaded.clip_obs


class OnnxPolicy:
    def __init__(self, model_path: Path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, obs):
        if obs.ndim == 1:
            obs = obs[None, :]
        action = self.session.run([self.output_name], {self.input_name: obs.astype(np.float32)})[0]
        return action, None


class Sb3Policy:
    def __init__(self, checkpoint: Path, stats_path: Path | None):
        self.model = PPO.load(str(checkpoint), device="cpu")
        self.obs_mean = None
        self.obs_var = None
        self.clip_obs = None
        if stats_path is not None and stats_path.exists():
            self.obs_mean, self.obs_var, self.clip_obs = _load_stats(stats_path)

    def predict(self, obs):
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        return self.model.predict(obs, deterministic=True)


def evaluate(policy, episodes: int, render: bool, curriculum_progress: float):
    config = cfg.get_config("default")
    env = InvertedPendulumEnv(config)
    env.set_curriculum_progress(curriculum_progress)

    episode_rewards = []
    episode_lengths = []
    success_rates = []
    stable_times = []
    first_stable_steps = []
    mean_abs_actions = []

    mj_model = env.model
    mj_data = env.data

    def rollout_episode(viewer=None):
        obs, _ = env.reset()
        total_reward = 0.0
        total_length = 0
        abs_action_sum = 0.0

        while True:
            action, _ = policy.predict(obs)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.size == 0:
                action = np.array([0.0], dtype=np.float32)
            action = np.clip(action[0], -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
            total_reward += float(reward)
            total_length += 1
            abs_action_sum += abs(float(action))

            if viewer is not None:
                viewer.sync()
                time_step = mj_model.opt.timestep * env.config["ROBOT"]["action_repeat"]
                import time
                time.sleep(time_step)

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(total_length)
        stable_times.append(float(info.get("stable_time", 0.0)))
        success_rates.append(1.0 if info.get("success_achieved", False) else 0.0)
        first_stable_steps.append(info.get("first_stable_step", -1) if info.get("first_stable_step", None) is not None else -1)
        mean_abs_actions.append(abs_action_sum / max(total_length, 1))

        print(
            f"reward={total_reward:8.3f}  len={total_length:4d}  "
            f"success={bool(info.get('success_achieved', False))}  "
            f"stable={float(info.get('stable_time', 0.0)):5.2f}s  "
            f"reason={info.get('termination_reason', 'done')}"
        )

    if render:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            for _ in range(episodes):
                rollout_episode(viewer)
    else:
        for _ in range(episodes):
            rollout_episode()

    summary = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": float(np.mean(success_rates)),
        "mean_stable_time": float(np.mean(stable_times)),
        "mean_first_stable_step": float(np.mean([x for x in first_stable_steps if x >= 0])) if any(x >= 0 for x in first_stable_steps) else -1.0,
        "mean_abs_action": float(np.mean(mean_abs_actions)),
    }

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pendulum policy in simulation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx", type=str, help="Path to exported ONNX policy")
    group.add_argument("--checkpoint", type=str, help="Path to SB3 .zip checkpoint")
    parser.add_argument("--stats", type=str, default=None, help="Optional VecNormalize stats .pkl for SB3 evaluation")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="Render with MuJoCo viewer")
    parser.add_argument("--curriculum-progress", type=float, default=1.0, help="Curriculum progress to use during evaluation")
    args = parser.parse_args()

    if args.onnx:
        policy = OnnxPolicy(Path(args.onnx))
    else:
        policy = Sb3Policy(Path(args.checkpoint), Path(args.stats) if args.stats else None)

    evaluate(policy, episodes=args.episodes, render=args.render, curriculum_progress=args.curriculum_progress)
