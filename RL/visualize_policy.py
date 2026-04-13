"""
Visualize a trained policy in 3D using MuJoCo's viewer
"""

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
import mujoco
import mujoco.viewer
import time
import sys
import os

# Add RL directory to path for imports
script_dir = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(script_dir))

import config as cfg
from environment import LeggedRobotEnv
from utils import get_models_dir


def add_friction_visualization(mj_model, mj_data):
    """
    Add colored ground tiles to visualize friction zones.
    Creates a grid of boxes with colors representing friction values.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
    """
    # Find the ground geom
    try:
        ground_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

        # Color the ground with a gradient showing friction
        # Get base friction from config
        friction_low = 0.3
        friction_high = 1.5

        # Create slight transparency to see through
        # Red = low friction, Green = medium, Blue = high friction
        friction = mujoco.mj_geomFriction(mj_model, ground_id)
        normalized_friction = np.clip(
            (friction[0] - friction_low) / (friction_high - friction_low), 0, 1
        )

        # Color based on friction: low (red) -> medium (yellow) -> high (green)
        if normalized_friction < 0.5:
            r = 1.0
            g = normalized_friction * 2
            b = 0.2
        else:
            r = (1 - normalized_friction) * 2
            g = 1.0
            b = 0.2

        # Apply color to ground geom
        mj_model.geom(ground_id).rgba[:3] = [r, g, b]
        mj_model.geom(ground_id).rgba[3] = 0.8  # alpha

        print(f"[INFO] Ground friction visualization enabled")
        print(f"       Friction level: {friction[0]:.2f} (Red=low, Green=high)")

    except Exception as e:
        print(f"[WARNING] Could not visualize friction: {e}")


def visualize_policy(
    variant="default",
    epoch=None,
    episodes=3,
    max_steps=1000,
    speed=1.0,
):
    """
    Load a trained policy and visualize it in the MuJoCo viewer

    Args:
        variant: Variant name (e.g., 'default' matches 'default_TIMESTAMP')
        epoch: Epoch number to load. If None, loads latest
        episodes: Number of episodes to run
        max_steps: Max steps per episode
        speed: Playback speed multiplier (1.0 = real-time)
    """

    # Find model directory (models in running_robot/models)
    models_dir = get_models_dir(__file__)

    if not models_dir.exists():
        print(f"[ERROR] Models directory not found: {models_dir}")
        print(f"[INFO] Make sure you ran: python RL/train.py")
        return

    # Try to match variant - support both "default" and "default_20260413_153921"
    variant_dirs = []

    # First try exact match
    if (models_dir / variant).exists():
        variant_dirs = [models_dir / variant]
    else:
        # Try prefix match with timestamp wildcard
        variant_dirs = sorted([d for d in models_dir.glob(f"{variant}_*")])

    if not variant_dirs:
        print(f"[ERROR] No model found for variant '{variant}'")
        print(f"\nSearching in: {models_dir}")
        print(f"Available variants:")
        for d in sorted(models_dir.glob("*")):
            if d.is_dir() and "_" in d.name:
                print(f"  - {d.name}")
        return

    variant_dir = variant_dirs[-1]  # Latest variant
    checkpoints_dir = variant_dir / "checkpoints"

    # Find epoch checkpoint
    if epoch is None:
        # Find latest epoch
        checkpoint_files = sorted(checkpoints_dir.glob("model_epoch_*.zip"))
        if not checkpoint_files:
            print(f"[ERROR] No checkpoints found in {checkpoints_dir}")
            return
        checkpoint_path = checkpoint_files[-1]
        epoch = int(checkpoint_path.stem.split("_")[-1])
    else:
        checkpoint_path = checkpoints_dir / f"model_epoch_{epoch:06d}.zip"

    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print(f"\nAvailable checkpoints:")
        for cp in sorted(checkpoints_dir.glob("model_epoch_*.zip")):
            print(f"  - {cp.name}")
        return

    print(f"\n{'='*60}")
    print(f"POLICY VISUALIZATION")
    print(f"{'='*60}")
    print(f"Model: {variant} (epoch {epoch})")
    print(f"Path: {checkpoint_path}")

    # Load config and create single environment
    config = cfg.get_config("default")

    # Update config paths to be absolute
    if not Path(config["ROBOT"]["urdf_path"]).is_absolute():
        config["ROBOT"]["urdf_path"] = str(script_dir / config["ROBOT"]["urdf_path"])

    env = LeggedRobotEnv(config)

    # Load trained policy
    print(f"\nLoading policy...")
    model = PPO.load(str(checkpoint_path))
    print(f"Policy loaded: {type(model.policy).__name__}")

    # Get MuJoCo model and data
    mj_model = env.model
    mj_data = env.data

    print(f"\n{'='*60}")
    print(f"Starting visualization...")
    print(f"Episodes: {episodes}, Max steps: {max_steps}")
    print(f"Playback speed: {speed}x")
    print(f"Close the viewer window to exit")
    print(f"{'='*60}\n")

    episode_rewards = []
    episode_lengths = []
    episode_speeds = []

    # Launch viewer
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Set viewer options
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        for ep in range(episodes):
            print(f"\n[Episode {ep + 1}/{episodes}]")

            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            episode_reward = 0.0
            episode_length = 0
            episode_speed = 0.0
            last_x = env.data.body(env.base_body_id).xpos[0]

            for step in range(max_steps):
                # Predict action from policy
                action, _ = model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                episode_length += 1
                episode_speed += abs(
                    env.data.body(env.base_body_id).xpos[0] - last_x
                )
                last_x = env.data.body(env.base_body_id).xpos[0]

                # Synchronize viewer with simulation
                viewer.sync()

                # Control playback speed
                time.sleep(mj_model.opt.timestep / speed)

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_speeds.append(episode_speed)

            print(f"  Reward: {episode_reward:8.3f}")
            print(f"  Length: {episode_length:4d} steps")
            print(f"  Distance: {episode_speed:6.2f} m")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(
        f"Mean Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}"
    )
    print(f"Mean Length: {np.mean(episode_lengths):.1f} steps")
    print(f"Mean Distance: {np.mean(episode_speeds):.2f} m")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained RL policy in 3D"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="default",
        help="Variant name (e.g., 'default' matches 'default_TIMESTAMP')",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to visualize (default: latest)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed (1.0 = real-time, 0.5 = half-speed, 2.0 = 2x)",
    )

    args = parser.parse_args()

    visualize_policy(
        variant=args.variant,
        epoch=args.epoch,
        episodes=args.episodes,
        max_steps=args.max_steps,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
