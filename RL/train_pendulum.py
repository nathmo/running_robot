import argparse
import os
import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from RL.pendulum_env import PendulumEnv


def make_env(seed=0, start_range=0.1):
    def _init():
        env = PendulumEnv(start_angle_range=start_range)
        env.seed(seed)
        return env
    return _init


def train(total_timesteps=200000, n_envs=4, start_range=0.1, export_path="policy.onnx"):
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, start_range) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, start_range)])

    policy_kwargs = dict(activation_fn=None)

    model = PPO('MlpPolicy', env, verbose=1, n_steps=256, batch_size=64)

    ck_callback = CheckpointCallback(save_freq=5000, save_path='./models/', name_prefix='pendulum')
    model.learn(total_timesteps=total_timesteps, callback=ck_callback)

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/pendulum_final')

    # Export to ONNX: create a dummy observation and export policy forward
    try:
        import torch
        obs_shape = env.observation_space.shape
        dummy = torch.zeros((1, obs_shape[0]), dtype=torch.float32)
        policy = model.policy
        policy.to('cpu')
        policy.eval()

        # Wrap in a small wrapper since sb3 policy expects specific inputs
        class PolicyWrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy
            def forward(self, x):
                # x: (N, obs_dim)
                # sb3 policy expects tensor and returns actions as a tensor
                with torch.no_grad():
                    # forward returns a dict-like; use _predict
                    actions, *_ = self.policy.forward(x)
                    return actions

        wrapper = PolicyWrapper(policy)
        torch.onnx.export(wrapper, dummy, export_path, opset_version=11,
                          input_names=['obs'], output_names=['action'], do_constant_folding=True)
        print('Exported ONNX policy to', export_path)
    except Exception as e:
        print('ONNX export failed:', e)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=200000)
    p.add_argument('--n-envs', type=int, default=4)
    p.add_argument('--start-range', type=float, default=0.1)
    p.add_argument('--export', type=str, default='models/pendulum.onnx')
    args = p.parse_args()

    train(total_timesteps=args.timesteps, n_envs=args.n_envs, start_range=args.start_range, export_path=args.export)
