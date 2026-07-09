"""Convert an rl/mjx_train.py checkpoint (raw Flax/Brax params) into an SB3-loadable checkpoint
(final_model.zip + vecnormalize.pkl) so rl/evaluate.py, rl/gait_probe.py, rl/joystick.py -- and
the eventual ONNX/Raspberry-Pi export -- work UNCHANGED on an MJX-trained policy.

This is a pure weight copy, not a retrain: rl/mjx_train.py's network is deliberately configured
(net_arch=[256,256], tanh, distribution_type='normal', state_dependent_std=False,
noise_std_type='log') to be structurally IDENTICAL to SB3's default continuous-action MlpPolicy,
so every layer maps 1:1. Both sides of every mapping below were confirmed against the actually-
installed packages, not assumed from memory:
  - brax: mujoco/spiderbot/../rl/mjx_train.py's saved params (introspected directly -- see the
    policy/value key dump this docstring's author ran before writing this file).
  - SB3: stable_baselines3/common/torch_layers.py's MlpExtractor.__init__ (Sequential layer
    order) and common/policies.py's ActorCriticPolicy._build + DiagGaussianDistribution.
    proba_distribution_net (action_net / log_std construction).

Mapping (flax Dense stores `kernel` shaped (in,out); torch nn.Linear stores `weight` shaped
(out,in) -- every kernel copy below is transposed):
  policy_params['params']['MLP_0']['hidden_0']  -> mlp_extractor.policy_net[0]  (Linear obs->256)
  policy_params['params']['MLP_0']['hidden_1']  -> mlp_extractor.policy_net[2]  (Linear 256->256)
  policy_params['params']['Dense_0']            -> action_net                  (Linear 256->action)
  policy_params['params']['std_logparam']['log_value'] -> log_std              (already log-space
                                                                                  on both sides)
  value_params['params']['hidden_0']            -> mlp_extractor.value_net[0]  (Linear obs->256)
  value_params['params']['hidden_1']            -> mlp_extractor.value_net[2]  (Linear 256->256)
  value_params['params']['hidden_2']            -> value_net                   (Linear 256->1)
  normalizer_params.mean/std/count              -> VecNormalize.obs_rms.mean/var(=std**2)/count

Run:
    .venv/bin/python -m rl.mjx_export --checkpoint rl/runs/m2_walk_mjx/final.pkl
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .config import get_config
from .env import SpiderBotEnv


def _copy_linear(torch_linear, kernel, bias):
    assert tuple(torch_linear.weight.shape) == (kernel.shape[1], kernel.shape[0]), \
        f"shape mismatch: torch {tuple(torch_linear.weight.shape)} vs flax kernel {kernel.shape}"
    torch_linear.weight.data = torch.from_numpy(np.asarray(kernel).T.copy())
    torch_linear.bias.data = torch.from_numpy(np.asarray(bias).copy())


def _uint64_to_int(count):
    if hasattr(count, "to_numpy"):
        return int(count.to_numpy())
    return int(count)


def build_sb3_model(cfg, checkpoint):
    policy = checkpoint["policy_params"]["params"]
    value = checkpoint["value_params"]["params"]
    normalizer = checkpoint["normalizer_params"]

    raw = SpiderBotEnv(cfg)
    venv = DummyVecEnv([lambda: raw])
    model = PPO(
        "MlpPolicy", venv,
        policy_kwargs=dict(net_arch=list(cfg.policy_hidden)),  # SB3 default activation: Tanh,
        #                                                        matching mjx_train.py's linen.tanh
        seed=cfg.seed, verbose=0,
    )
    p = model.policy

    _copy_linear(p.mlp_extractor.policy_net[0], policy["MLP_0"]["hidden_0"]["kernel"],
                policy["MLP_0"]["hidden_0"]["bias"])
    _copy_linear(p.mlp_extractor.policy_net[2], policy["MLP_0"]["hidden_1"]["kernel"],
                policy["MLP_0"]["hidden_1"]["bias"])
    _copy_linear(p.action_net, policy["Dense_0"]["kernel"], policy["Dense_0"]["bias"])
    log_std = np.asarray(policy["std_logparam"]["log_value"])
    assert p.log_std.shape == log_std.shape, f"log_std shape {p.log_std.shape} vs {log_std.shape}"
    p.log_std.data = torch.from_numpy(log_std.copy())

    _copy_linear(p.mlp_extractor.value_net[0], value["hidden_0"]["kernel"], value["hidden_0"]["bias"])
    _copy_linear(p.mlp_extractor.value_net[2], value["hidden_1"]["kernel"], value["hidden_1"]["bias"])
    _copy_linear(p.value_net, value["hidden_2"]["kernel"], value["hidden_2"]["bias"])

    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=cfg.gamma)
    venv.obs_rms.mean = np.asarray(normalizer.mean, dtype=np.float64).copy()
    venv.obs_rms.var = (np.asarray(normalizer.std, dtype=np.float64) ** 2).copy()
    venv.obs_rms.count = _uint64_to_int(normalizer.count)
    venv.training = False

    return model, venv, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="an rl/mjx_train.py .pkl checkpoint")
    ap.add_argument("--run", default=None,
                    help="output rl/runs/<run> dir (default: checkpoint's own parent dir)")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)
    preset = checkpoint["preset"]
    cfg = get_config(preset)

    run = Path(args.run) if args.run else ckpt_path.parent
    run.mkdir(parents=True, exist_ok=True)

    model, venv, raw = build_sb3_model(cfg, checkpoint)

    # sanity: the just-built model must actually run through the CPU env without NaN/crash
    # before we trust this checkpoint enough to write it out.
    obs = venv.reset()
    for _ in range(5):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = venv.step(a)
        assert np.all(np.isfinite(obs)) and np.isfinite(r[0]), "exported policy produced NaN"

    model.save(run / "final_model")
    venv.save(str(run / "vecnormalize.pkl"))
    (run / "preset.json").write_text(f'{{"preset": "{preset}"}}')
    print(f"[export] {ckpt_path} -> {run/'final_model.zip'} + {run/'vecnormalize.pkl'}  "
          f"(preset={preset})")
    print(f"[export] verify with: python -m rl.gait_probe --run {run}")


if __name__ == "__main__":
    main()
