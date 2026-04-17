"""
Main training script for legged robot RL
"""

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from tqdm import tqdm
import time

import config as cfg
from environment import LeggedRobotEnv, create_env
from utils import (
    CheckpointManager,
    MetricsLogger,
    create_experiment_folder,
    save_config,
    get_project_root,
)


def train(
    variant_name="default",
    config_preset="default",
    resume_from_epoch=None,
    custom_config=None,
):
    """
    Train RL policy for legged robot

    Args:
        variant_name: Name for this experiment
        config_preset: Config preset ("default", "easy_terrain", "hard_terrain", etc.)
        resume_from_epoch: Resume from specific epoch (None = start fresh)
        custom_config: Override specific config values
    """

    # Load configuration
    config = cfg.get_config(config_preset)

    if custom_config:
        # Deep merge custom config
        def deep_merge(base, override):
            for key, value in override.items():
                if isinstance(value, dict) and key in base:
                    deep_merge(base[key], value)
                else:
                    base[key] = value

        deep_merge(config, custom_config)

    print(f"\n{'='*60}")
    print(f"Training variant: {variant_name}")
    print(f"Config preset: {config_preset}")
    print(f"{'='*60}\n")

    # Create experiment folders using project root
    project_root = get_project_root(__file__)
    models_dir, logs_dir, variant_full_name = create_experiment_folder(
        project_root,
        variant_name,
        include_timestamp=config["LOGGING"]["append_timestamp"],
    )

    # Save config
    save_config(config, models_dir / variant_full_name / "config.json")

    # Setup checkpoint and metrics management
    checkpoint_manager = CheckpointManager(
        models_dir, variant_full_name, keep_last_n=config["RL"]["keep_last_n_checkpoints"]
    )
    metrics_logger = MetricsLogger(logs_dir, variant_full_name)

    # Create environment
    print("Creating environment...")
    env = create_env(
        config,
        num_envs=config["RL"]["n_envs"],
        render_mode="human" if config["ENVIRONMENT"]["render"] else None,
    )

    # Normalize observations/actions
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Dedicated evaluation env: single-env, path not randomized, no reward normalization.
    # Obs-normalization stats are synced from `env` before each eval call so the policy
    # sees the same observation distribution it was trained on.
    eval_env = DummyVecEnv([
        lambda: LeggedRobotEnv(config, randomize_path=False)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            training=False)

    # Create or load model
    print("Creating model...")
    model_kwargs = {
        "learning_rate": config["RL"]["learning_rate"],
        "n_steps": config["RL"]["n_steps"],
        "batch_size": config["RL"]["batch_size"],
        "gamma": config["RL"]["gamma"],
        "gae_lambda": config["RL"]["gae_lambda"],
        "clip_range": config["RL"]["clip_range"],
        "ent_coef": config["RL"]["ent_coef"],
        "vf_coef": config["RL"]["vf_coef"],
        "tensorboard_log": logs_dir / variant_full_name,
    }

    if resume_from_epoch is not None:
        print(f"Resuming from epoch {resume_from_epoch}...")
        try:
            model = checkpoint_manager.load_checkpoint(PPO, resume_from_epoch)
            start_epoch = resume_from_epoch
        except FileNotFoundError as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch...")
            model = PPO("MlpPolicy", env, verbose=1, **model_kwargs)
            start_epoch = 0
    else:
        model = PPO("MlpPolicy", env, verbose=1, **model_kwargs)
        start_epoch = 0

    # Save metadata
    metadata = {
        "config_preset": config_preset,
        "start_epoch": start_epoch,
        "n_envs": config["RL"]["n_envs"],
        "algorithm": config["RL"]["algorithm"],
    }
    checkpoint_manager.save_metadata(metadata)

    print("\nStarting training loop...\n")

    # Training loop
    checkpoint_interval = config["RL"]["checkpoint_interval"]
    log_interval = config["RL"]["log_interval"]
    n_epochs = config["RL"]["n_epochs"]

    epoch_pbar = tqdm(range(start_epoch, n_epochs), desc="Training Epochs", unit="epoch")
    for epoch in epoch_pbar:
        epoch_start_time = time.time()

        # Update progress bar description
        epoch_pbar.set_description(f"Epoch {epoch + 1}/{n_epochs}")

        # Train for one epoch
        model.learn(
            total_timesteps=config["RL"]["n_steps"] * config["RL"]["n_envs"],
            log_interval=log_interval,
            progress_bar=True,
        )

        epoch_time = time.time() - epoch_start_time

        # Periodically save and evaluate
        if (epoch + 1) % checkpoint_interval == 0 or epoch == n_epochs - 1:
            # Evaluate on test environment (no randomization)
            epoch_pbar.write(f"\n[Epoch {epoch+1}/{n_epochs}] Evaluating...")
            eval_start_time = time.time()

            # Copy running obs mean/std from train env to eval env so the policy
            # gets consistent inputs. Reward stats are not used (norm_reward=False).
            sync_envs_normalization(env, eval_env)

            eval_metrics = evaluate_policy(
                model, eval_env, config, num_episodes=5
            )

            eval_time = time.time() - eval_start_time

            # Log metrics
            log_metrics = {
                "epoch": epoch + 1,
                **eval_metrics,
            }
            metrics_logger.log_epoch(epoch + 1, log_metrics)

            # Save checkpoint
            checkpoint_manager.save_checkpoint(model, epoch + 1, log_metrics)

            # Print results nicely
            epoch_pbar.write(
                f"  [OK] Reward: {eval_metrics['eval_mean_reward']:7.3f} ± {eval_metrics['eval_std_reward']:.3f} | "
                f"Len: {eval_metrics['eval_mean_length']:5.0f} | "
                f"Speed: {eval_metrics['eval_mean_speed']:6.3f} | "
                f"Time: {epoch_time:.1f}s"
            )

        # Periodically plot convergence
        if (epoch + 1) % (checkpoint_interval * 2) == 0:
            metrics_logger.plot_convergence()

        # Update progress bar with metrics if available
        if (epoch + 1) % checkpoint_interval == 0:
            epoch_pbar.set_postfix({
                "reward": f"{eval_metrics['eval_mean_reward']:.3f}",
                "time": f"{epoch_time:.0f}s"
            })

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    # Final plots
    metrics_logger.plot_convergence()

    # Save normalized env stats for evaluation (if available)
    if hasattr(env, "save_running_average_std"):
        env.save_running_average_std(
            models_dir / variant_full_name / "running_avg.pkl"
        )

    env.close()
    eval_env.close()

    return checkpoint_manager, metrics_logger


def evaluate_policy(model, env, config, num_episodes=5):
    """Evaluate policy on a (preferably single-env, non-randomized) VecEnv.

    Metrics:
        eval_mean_reward:   mean total reward per episode
        eval_mean_length:   mean step count per episode
        eval_mean_speed:    mean forward speed along the path (m/s),
                            averaged over steps of each episode from info["forward_speed_mps"].
    """
    episode_rewards = []
    episode_lengths = []
    episode_speeds = []

    for _ in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        episode_reward = 0.0
        episode_length = 0
        speed_samples = []

        for _ in range(config["RL"]["max_episode_steps"]):
            action, _ = model.predict(obs, deterministic=True)
            result = env.step(action)
            if len(result) == 5:  # Gymnasium single-env
                obs, reward, terminated, truncated, info = result
                done = terminated
            else:  # SB3 VecEnv: (obs, reward, done, info) — done merges terminated|truncated
                obs, reward, done, info = result
                truncated = False

            episode_reward += float(np.mean(reward))
            episode_length += 1

            # Extract per-step forward speed from info. VecEnv infos is a list of dicts.
            if isinstance(info, (list, tuple)):
                for sub in info:
                    if "forward_speed_mps" in sub:
                        speed_samples.append(float(sub["forward_speed_mps"]))
            elif isinstance(info, dict) and "forward_speed_mps" in info:
                speed_samples.append(float(info["forward_speed_mps"]))

            if np.any(done) or np.any(truncated):
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_speeds.append(float(np.mean(speed_samples)) if speed_samples else 0.0)

    return {
        "eval_mean_reward": float(np.mean(episode_rewards)),
        "eval_std_reward": float(np.std(episode_rewards)),
        "eval_mean_length": float(np.mean(episode_lengths)),
        "eval_mean_speed": float(np.mean(episode_speeds)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train RL policy for legged robot"
    )
    parser.add_argument(
        "--variant", type=str, default="default", help="Experiment variant name"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=["default", "easy_terrain", "hard_terrain", "fast_training", "long_training"],
        help="Configuration preset",
    )
    parser.add_argument(
        "--resume", type=int, default=None, help="Resume from epoch N"
    )
    parser.add_argument(
        "--terrain-type",
        type=str,
        choices=["flat", "perlin", "stairs"],
        help="Override terrain type",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        help="Override number of parallel environments",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        help="Override total number of epochs",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        help="Override rollout size per env (n_steps in PPO)",
    )

    args = parser.parse_args()

    # Build custom config from args
    custom_config = {}
    if args.terrain_type:
        custom_config["TERRAIN"] = {"type": args.terrain_type}
    if args.n_envs:
        custom_config["RL"] = {"n_envs": args.n_envs}
    if args.n_epochs:
        custom_config["RL"] = {**custom_config.get("RL", {}), "n_epochs": args.n_epochs}
    if args.n_steps:
        custom_config["RL"] = {**custom_config.get("RL", {}), "n_steps": args.n_steps}

    train(
        variant_name=args.variant,
        config_preset=args.preset,
        resume_from_epoch=args.resume,
        custom_config=custom_config if custom_config else None,
    )


if __name__ == "__main__":
    main()
