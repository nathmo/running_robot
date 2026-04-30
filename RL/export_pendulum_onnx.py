"""Export a trained pendulum PPO checkpoint to ONNX.

The exported graph includes the VecNormalize observation scaling so the runtime
can consume raw hardware observations:
[position_turns, velocity_turns_per_s, torque_nm]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config as cfg
from environment import InvertedPendulumEnv


class NormalizedDeterministicPolicy(torch.nn.Module):
    def __init__(self, policy, obs_mean, obs_var, clip_obs):
        super().__init__()
        self.policy = policy
        self.register_buffer("obs_mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_var", torch.as_tensor(obs_var, dtype=torch.float32))
        self.clip_obs = float(clip_obs)
        self.eps = 1e-8

    def forward(self, obs):
        obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.eps)
        obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
        features = self.policy.extract_features(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        action = self.policy.action_net(latent_pi)
        return torch.clamp(action, -1.0, 1.0)


def _load_vecnormalize_stats(stats_path: Path):
    base_env = DummyVecEnv([lambda: InvertedPendulumEnv(cfg.get_config("default"))])
    loaded = VecNormalize.load(str(stats_path), base_env)
    return loaded.obs_rms.mean, loaded.obs_rms.var, loaded.clip_obs


def export_checkpoint(checkpoint_path: Path, output_path: Path, stats_path: Path | None = None):
    model = PPO.load(str(checkpoint_path), device="cpu")

    if stats_path is None:
        candidate = checkpoint_path.with_name(checkpoint_path.name.replace("model_epoch_", "vecnormalize_epoch_").replace(".zip", ".pkl"))
        stats_path = candidate if candidate.exists() else None

    if stats_path is not None and stats_path.exists():
        obs_mean, obs_var, clip_obs = _load_vecnormalize_stats(stats_path)
        stats_meta = {
            "stats_path": str(stats_path),
            "obs_mean": np.asarray(obs_mean).tolist(),
            "obs_var": np.asarray(obs_var).tolist(),
            "clip_obs": float(clip_obs),
        }
    else:
        obs_mean = np.zeros(model.observation_space.shape[0], dtype=np.float32)
        obs_var = np.ones(model.observation_space.shape[0], dtype=np.float32)
        clip_obs = 10.0
        stats_meta = {
            "stats_path": None,
            "obs_mean": obs_mean.tolist(),
            "obs_var": obs_var.tolist(),
            "clip_obs": clip_obs,
        }

    wrapper = NormalizedDeterministicPolicy(model.policy, obs_mean, obs_var, clip_obs)
    wrapper.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, model.observation_space.shape[0]), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(stats_meta, f, indent=2)

    print(f"Exported ONNX policy: {output_path}")
    print(f"Wrote export metadata: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a pendulum checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to SB3 .zip checkpoint")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--stats", default=None, help="Optional VecNormalize .pkl path")
    args = parser.parse_args()

    export_checkpoint(Path(args.checkpoint), Path(args.output), Path(args.stats) if args.stats else None)
