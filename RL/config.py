"""
Configuration for inverted pendulum RL training.

The task is to stabilize the pendulum upright while minimizing control effort,
using the same observation tuple that will be available on hardware:
position (revolutions), velocity (revolutions/s), and torque (Nm).
"""

import os
import platform

# ==============================================================================
# ROBOT CONFIGURATION
# ==============================================================================
ROBOT = {
    "urdf_path": "assets/inverted_pendulum.xml",

    # Control parameters
    "control_mode": "torque",  # Direct torque control
    "action_repeat": 20,  # 20 x 1ms sim steps = 50 Hz control loop
    "sim_dt": 0.001,  # Simulation timestep (seconds)
    "control_dt": 0.02,  # 50 Hz control timestep

    # Motor spec: 1 Nm capable motor
    # Action space is [-1, 1] which maps to [-1, 1] Nm (linear)
    "torque_limit_nm": 1.0,
}

# ==============================================================================
# START CONFIGURATION (Pendulum-specific)
# ==============================================================================
START = {
    "default_start_angle": 90.0,  # Upright (degrees)
    # Randomized starts reduce overfitting. Curriculum starts near upright and
    # gradually expands to full-swing randomization.
    "randomize_start_angle": True,
    "curriculum_enabled": True,
    "curriculum_episodes": 400,
    "curriculum_initial_span_deg": 10.0,
    "curriculum_final_span_deg": 180.0,
}

# ==============================================================================
# RL TRAINING CONFIGURATION
# ==============================================================================
RL = {
    # Algorithm
    "algorithm": "PPO",

    # Training parameters
    "n_steps": 1024,  # Steps per epoch (per environment)
    "n_epochs": 1000,  # Total epochs to train
    "batch_size": 64,
    # 3e-1 caused policy collapse; PPO is much stabler here around 3e-4.
    "learning_rate": 3e-4,
    "gamma": 0.995,  # Higher discount — rewards staying upright longer
    "gae_lambda": 0.95,  # GAE lambda
    "clip_range": 0.2,  # PPO default range, less policy thrash
    "ent_coef": 0.20,  # Much stronger exploration pressure early in training
    "vf_coef": 0.5,  # Value function coefficient
    "target_kl": 0.03,  # Keep PPO updates bounded while entropy is high
    "max_grad_norm": 0.7,  # Mildly tighter gradients for stability under larger action std

    # Policy exploration settings (continuous Gaussian policy)
    "policy": {
        # Larger initial std so sampled training actions can explore wider torque values.
        # (exp(0.5) ~= 1.65 before squashing/clipping to [-1, 1]).
        "log_std_init": 0.5,
        # Orthogonal init can produce overly confident early means in this task.
        "ortho_init": False,
    },

    # Environment
    "n_envs": 4,  # Windows: use 1, Linux: use all cores
    "max_episode_steps": 1000,  # 20 seconds at 50 Hz
    "max_upright_episode_steps": 1000,  # Hold success until the end of the episode
    "success_hold_seconds": 5.0,
    "success_deadline_seconds": 5.0,
    "success_angle_threshold_turns": 0.1,
    "success_velocity_threshold_turns_s": 0.1,

    # Observation/Action
    "observation_space": {
        "include": ["position", "velocity", "torque"],
        "normalize": False,
    },
    "action_space": "continuous",  # Single motor torque command in Nm

    # Checkpointing
    "checkpoint_interval": 5,  # Save checkpoint every N epochs
    "keep_last_n_checkpoints": 50,

    # Logging
    "log_interval": 5,  # Print stats every N epochs
}

# ==============================================================================
# REWARD CONFIGURATION
# ==============================================================================
REWARD = {
    # Dense shaping reward: higher is better.
    "angle_weight": 10.0,
    "velocity_weight": 0.5,
    "effort_weight": 0.02,

    # Small alive bonus to encourage surviving the full 20 s episode.
    "alive_bonus": 0.02,

    # Bonus for being inside the success region; helps the policy settle.
    "stable_bonus": 0.25,

    # Final bonus paid only if the episode ends successfully.
    "success_bonus": 200.0,
}

# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================
ENVIRONMENT = {
    "render": False,
    "render_mode": "human",
    "gravity": [0, 0, -9.81],
}

# ==============================================================================
# LOGGING & CHECKPOINTING
# ==============================================================================
LOGGING = {
    "log_dir": "RL/logs",
    "models_dir": "RL/models",
    "append_timestamp": True,  # Folder names include timestamp

    # Metrics to track
    "metrics": [
        "episode_reward",
        "episode_length",
        "forward_speed",
        "energy_consumed",
        "success_rate",
        "track_deviation",
    ],
}

# ==============================================================================
# VISUALIZATION CONFIGURATION
# ==============================================================================
VISUALIZATION = {
    "render_every_n_episodes": 10,
    "video_length": 200,  # Steps per video
    "save_videos": True,
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_config(config_name: str = "default"):
    """Load a named configuration preset"""
    presets = {
        "default": {},
        "quick": {
            "RL": {**RL, "n_steps": 1024, "n_epochs": 100},
        },
        "long_training": {
            "RL": {**RL, "n_steps": 4096, "n_epochs": 2000},
        },
    }

    preset = presets.get(config_name, {})

    config = {
        "ROBOT": {**ROBOT, **preset.get("ROBOT", {})},
        "START": {**START, **preset.get("START", {})},
        "RL": {**RL, **preset.get("RL", {})},
        "REWARD": {**REWARD, **preset.get("REWARD", {})},
        "ENVIRONMENT": {**ENVIRONMENT, **preset.get("ENVIRONMENT", {})},
        "LOGGING": {**LOGGING, **preset.get("LOGGING", {})},
    }

    # On Linux servers, default to using all available CPU cores unless the
    # preset explicitly overrides it.
    if platform.system() != "Windows" and config["RL"].get("n_envs", None) == RL["n_envs"]:
        config["RL"]["n_envs"] = max(1, os.cpu_count() or 1)

    return config
