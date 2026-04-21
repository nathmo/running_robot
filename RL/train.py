"""
Main training script for inverted pendulum RL
"""

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import LinearSchedule
from tqdm import tqdm
import time

import config as cfg
from environment import InvertedPendulumEnv, create_env
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
    Train RL policy for inverted pendulum

    Args:
        variant_name: Name for this experiment
        config_preset: Config preset ("default", "quick", "long_training", etc.)
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

    # Create experiment folders using project root.
    # When resuming, reuse the existing (already-timestamped) variant folder
    # instead of forking a new one — otherwise the checkpoint can't be found.
    project_root = get_project_root(__file__)
    models_dir, logs_dir, variant_full_name = create_experiment_folder(
        project_root,
        variant_name,
        include_timestamp=config["LOGGING"]["append_timestamp"],
        reuse=resume_from_epoch is not None,
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

    # Normalize observations/actions (disable reward normalization — too unstable for pendulum)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Dedicated evaluation env: single-env, no reward normalization.
    # Obs-normalization stats are synced from `env` before each eval call so the policy
    # sees the same observation distribution it was trained on.
    eval_env = DummyVecEnv([
        lambda: InvertedPendulumEnv(config)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            training=False)

    # Create or load model
    print("Creating model...")
    # Linear LR decay from config lr → 0 over the full training horizon.
    # `LinearSchedule(initial, final, end_fraction)` is the non-deprecated
    # SB3 API; PPO calls it each update with `progress_remaining` in [1, 0].
    # Decays stabilize late-stage updates — constant lr tends to destabilize
    # after a few thousand epochs and blow up the reward variance.
    lr_schedule = LinearSchedule(
        config["RL"]["learning_rate"], 0.0, 1.0
    )
    model_kwargs = {
        "learning_rate": lr_schedule,
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
        # Restore VecNormalize running stats first so the loaded policy sees
        # the same observation distribution it was trained against. Without
        # this, fresh obs_rms would corrupt the value function on resume.
        if not checkpoint_manager.load_vecnormalize_into(env, resume_from_epoch):
            print(
                "  [warn] No VecNormalize stats found for this epoch — "
                "continuing with fresh obs-normalization stats."
            )
        model = checkpoint_manager.load_checkpoint(
            PPO, resume_from_epoch, env=env
        )
        # Re-install the LR schedule after load; PPO.load otherwise uses whatever
        # schedule was baked into the checkpoint (which may already be exhausted).
        model.learning_rate = lr_schedule
        model._setup_lr_schedule()
        start_epoch = resume_from_epoch
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

            # Save checkpoint + normalization stats so resume is lossless
            checkpoint_manager.save_checkpoint(model, epoch + 1, log_metrics)
            checkpoint_manager.save_vecnormalize(env, epoch + 1)

            # Print results nicely
            epoch_pbar.write(
                f"  [OK] Reward: {eval_metrics['eval_mean_reward']:7.3f} ± {eval_metrics['eval_std_reward']:.3f} | "
                f"Len: {eval_metrics['eval_mean_length']:5.0f} | "
                f"Upright: {eval_metrics['eval_mean_upright_time']:6.3f}s | "
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
    """Evaluate policy on a (preferably single-env) VecEnv.

    Metrics:
        eval_mean_reward:   mean total reward per episode
        eval_mean_length:   mean step count per episode
        eval_mean_upright_time: mean time spent upright (within ±10° of 90°)
    """
    episode_rewards = []
    episode_lengths = []
    episode_upright_times = []

    for _ in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        episode_reward = 0.0
        episode_length = 0
        max_upright_time = 0.0

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

            # Extract upright time from info
            if isinstance(info, (list, tuple)):
                for sub in info:
                    if "upright_time" in sub:
                        max_upright_time = max(max_upright_time, float(sub["upright_time"]))
            elif isinstance(info, dict) and "upright_time" in info:
                max_upright_time = max(max_upright_time, float(info["upright_time"]))

            if np.any(done) or np.any(truncated):
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_upright_times.append(max_upright_time)

    return {
        "eval_mean_reward": float(np.mean(episode_rewards)),
        "eval_std_reward": float(np.std(episode_rewards)),
        "eval_mean_length": float(np.mean(episode_lengths)),
        "eval_mean_upright_time": float(np.mean(episode_upright_times)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train RL policy for inverted pendulum"
    )
    parser.add_argument(
        "--variant", type=str, default="default", help="Experiment variant name"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=["default", "quick", "long_training"],
        help="Configuration preset",
    )
    parser.add_argument(
        "--resume", type=int, default=None, help="Resume from epoch N"
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
