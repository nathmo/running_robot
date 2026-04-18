"""
Utility functions for logging, plotting, and checkpointing
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pickle


def get_project_root(script_file=None):
    """
    Get the project root directory (running_robot/).

    Args:
        script_file: __file__ from the calling script (use __file__ when calling)

    Returns:
        Path to running_robot/ directory
    """
    if script_file is None:
        # Default: assume called from RL/ subdirectory
        script_file = __file__

    script_path = Path(os.path.abspath(script_file))
    rl_dir = script_path.parent if script_path.parent.name == "RL" else script_path.parent.parent
    project_root = rl_dir.parent if rl_dir.name == "RL" else rl_dir

    return project_root


def get_models_dir(script_file=None):
    """Get models directory path (running_robot/models)."""
    project_root = get_project_root(script_file)
    return project_root / "models"


def get_logs_dir(script_file=None):
    """Get logs directory path (running_robot/logs)."""
    project_root = get_project_root(script_file)
    return project_root / "logs"


class CheckpointManager:
    """Manage model checkpoints and training state"""

    def __init__(self, save_dir, variant_name, keep_last_n=5):
        """
        Args:
            save_dir: Directory to save checkpoints
            variant_name: Name of this variant/experiment
            keep_last_n: Number of last checkpoints to keep
        """
        self.save_dir = Path(save_dir)
        self.variant_name = variant_name
        self.keep_last_n = keep_last_n

        # Create variant directory
        self.variant_dir = self.save_dir / variant_name
        self.variant_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file
        self.metadata_file = self.variant_dir / "metadata.json"
        self.checkpoints_dir = self.variant_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, model, epoch, metrics=None):
        """
        Save model checkpoint

        Args:
            model: Trained model (Stable-Baselines3)
            epoch: Current epoch number
            metrics: Dict of metrics to save
        """
        checkpoint_file = self.checkpoints_dir / f"model_epoch_{epoch:06d}.zip"
        model.save(str(checkpoint_file))

        # Save metrics
        if metrics is not None:
            metrics_file = self.checkpoints_dir / f"metrics_epoch_{epoch:06d}.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        print(f"Checkpoint saved: {checkpoint_file}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N"""
        checkpoints = sorted(self.checkpoints_dir.glob("model_epoch_*.zip"))

        if len(checkpoints) > self.keep_last_n:
            for old_checkpoint in checkpoints[: -self.keep_last_n]:
                epoch_str = old_checkpoint.stem.split("_")[-1]
                old_checkpoint.unlink()
                # Also remove the sibling metrics JSON and VecNormalize pkl
                for sibling in (
                    old_checkpoint.parent / f"metrics_epoch_{epoch_str}.json",
                    old_checkpoint.parent / f"vecnormalize_epoch_{epoch_str}.pkl",
                ):
                    if sibling.exists():
                        sibling.unlink()

    def load_latest_checkpoint(self, model_class):
        """Load latest checkpoint"""
        checkpoints = sorted(self.checkpoints_dir.glob("model_epoch_*.zip"))

        if not checkpoints:
            return None, 0

        latest = checkpoints[-1]
        model = model_class.load(str(latest))

        # Extract epoch from filename
        epoch = int(latest.stem.split("_")[-1])

        return model, epoch

    def load_checkpoint(self, model_class, epoch, env=None):
        """Load specific epoch checkpoint.

        If `env` is passed, it is attached to the loaded model so `learn()`
        can continue without re-creating a rollout buffer against a detached env.
        """
        checkpoint_file = self.checkpoints_dir / f"model_epoch_{epoch:06d}.zip"

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        if env is not None:
            return model_class.load(str(checkpoint_file), env=env)
        return model_class.load(str(checkpoint_file))

    def vecnormalize_path(self, epoch):
        """Path to VecNormalize stats file for a given epoch."""
        return self.checkpoints_dir / f"vecnormalize_epoch_{epoch:06d}.pkl"

    def save_vecnormalize(self, vec_env, epoch):
        """Persist VecNormalize running stats alongside the checkpoint."""
        path = self.vecnormalize_path(epoch)
        vec_env.save(str(path))
        print(f"VecNormalize stats saved: {path}")

    def load_vecnormalize_into(self, vec_env, epoch):
        """Load VecNormalize stats (in-place) from a previous checkpoint.

        Returns True if stats were loaded, False if no pkl was found.
        """
        from stable_baselines3.common.vec_env import VecNormalize

        path = self.vecnormalize_path(epoch)
        if not path.exists():
            return False

        # VecNormalize.load rewraps; we instead copy running stats onto the
        # already-constructed env so the caller keeps its env reference.
        loaded = VecNormalize.load(str(path), vec_env.venv)
        vec_env.obs_rms = loaded.obs_rms
        vec_env.ret_rms = loaded.ret_rms
        vec_env.clip_obs = loaded.clip_obs
        vec_env.clip_reward = loaded.clip_reward
        vec_env.gamma = loaded.gamma
        vec_env.epsilon = loaded.epsilon
        vec_env.returns = loaded.returns
        print(f"VecNormalize stats loaded: {path}")
        return True

    def save_metadata(self, metadata):
        """Save experiment metadata"""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def load_metadata(self):
        """Load experiment metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def list_checkpoints(self):
        """List available checkpoints"""
        checkpoints = sorted(self.checkpoints_dir.glob("model_epoch_*.zip"))
        epochs = [int(cp.stem.split("_")[-1]) for cp in checkpoints]
        return epochs


class MetricsLogger:
    """Log and manage training metrics"""

    def __init__(self, log_dir, variant_name):
        """
        Args:
            log_dir: Directory for logs
            variant_name: Name of variant
        """
        self.log_dir = Path(log_dir)
        self.variant_name = variant_name

        # Create log directory
        self.variant_log_dir = self.log_dir / variant_name
        self.variant_log_dir.mkdir(parents=True, exist_ok=True)

        # Log file
        self.log_file = self.variant_log_dir / "training_log.txt"
        self.metrics_file = self.variant_log_dir / "metrics.json"

        # Initialize empty metrics
        self.metrics = {}
        self.epochs = []

    def log_epoch(self, epoch, metrics):
        """
        Log metrics for an epoch

        Args:
            epoch: Epoch number
            metrics: Dict of metric names -> values
        """
        self.epochs.append(epoch)

        # Store metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

        # Append to text file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Epoch {epoch}: "
        log_entry += " | ".join(
            [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
        )

        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")

        # Save metrics JSON
        self._save_metrics_json()

        print(log_entry)

    def _save_metrics_json(self):
        """Save metrics to JSON file"""
        data = {"epochs": self.epochs}
        for key, values in self.metrics.items():
            data[key] = values

        with open(self.metrics_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_metrics(self, key):
        """Get metric history"""
        return self.metrics.get(key, [])

    def plot_convergence(self, save_path=None):
        """Plot training convergence curves"""
        if not self.metrics:
            print("No metrics to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Plot each metric
        plot_idx = 0
        for key in sorted(self.metrics.keys()):
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]
            values = self.metrics[key]
            ax.plot(self.epochs, values, "b-", linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key)
            ax.set_title(f"{key} over training")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if save_path is None:
            save_path = self.variant_log_dir / "convergence.png"

        plt.savefig(save_path, dpi=150)
        print(f"Convergence plot saved: {save_path}")
        plt.close()


def create_experiment_folder(base_dir, variant_name, include_timestamp=True, reuse=False):
    """
    Create experiment folder structure

    Args:
        base_dir: Project root
        variant_name: Variant name. When reuse=True, treated as the already-
            timestamped folder name (e.g. "gpu_mjx_20260417_172700").
        include_timestamp: Append a fresh timestamp to variant_name
        reuse: If True, skip timestamp and require the variant folder to
            already exist (used when resuming a previous run).

    Returns:
        (models_dir, logs_dir, variant_full_name)
    """
    models_dir = Path(base_dir) / "models"
    logs_dir = Path(base_dir) / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if reuse:
        variant_full_name = variant_name
        if not (models_dir / variant_full_name).exists():
            raise FileNotFoundError(
                f"Cannot resume: {models_dir / variant_full_name} does not exist. "
                f"Pass --variant <existing_folder_name> when using --resume."
            )
    elif include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        variant_full_name = f"{variant_name}_{timestamp}"
    else:
        variant_full_name = variant_name

    return models_dir, logs_dir, variant_full_name


def save_config(config, save_path):
    """Save configuration to file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(
            config,
            f,
            indent=2,
            default=lambda o: str(o) if not isinstance(o, (int, float, str, list, dict)) else o,
        )


def load_config(load_path):
    """Load configuration from file"""
    with open(load_path, "r") as f:
        return json.load(f)


# Example usage
if __name__ == "__main__":
    # Test checkpoint manager
    cm = CheckpointManager("RL/models", "test_variant")
    print(f"Variant directory: {cm.variant_dir}")

    # Test metrics logger
    ml = MetricsLogger("RL/logs", "test_variant")

    # Simulate logging
    for epoch in range(10):
        metrics = {
            "mean_reward": np.random.normal(10, 2),
            "mean_speed": np.random.normal(2, 0.5),
            "energy_cost": np.random.normal(5, 1),
        }
        ml.log_epoch(epoch, metrics)

    # Save and plot
    ml.plot_convergence()
