"""
Configuration for inverted pendulum RL training
Task: Balance a 200g mass on a 0.5m rod with a 1 Nm motor
"""

# ==============================================================================
# ROBOT CONFIGURATION
# ==============================================================================
ROBOT = {
    "urdf_path": "assets/inverted_pendulum.xml",

    # Control parameters
    "control_mode": "torque",  # Direct torque control
    "action_repeat": 2,  # Repeat action for N simulation steps (faster control)
    "sim_dt": 0.001,  # Simulation timestep (seconds)
    "control_dt": 0.002,  # Control timestep after action_repeat

    # Motor spec: 1 Nm capable motor
    # Action space is [-1, 1] which maps to [-1, 1] Nm (linear)
    "torque_limit_nm": 1.0,
}

# ==============================================================================
# START CONFIGURATION (Pendulum-specific)
# ==============================================================================
START = {
    "default_start_angle": 270.0,  # Start hanging down (degrees)
    "randomize_start_angle": False,
    "start_angle_range": [260.0, 280.0],  # ±10° from hanging down
}

# ==============================================================================
# RL TRAINING CONFIGURATION
# ==============================================================================
RL = {
    # Algorithm
    "algorithm": "PPO",

    # Training parameters
    "n_steps": 2048,  # Steps per epoch (per environment)
    "n_epochs": 1000,  # Total epochs to train (pendulum converges faster than biped)
    "batch_size": 64,
    "learning_rate": 3e-2,  # Lower from 1e-3 for stability
    "gamma": 0.995,  # Higher discount — rewards staying upright longer
    "gae_lambda": 0.95,  # GAE lambda
    "clip_range": 0.5,  # PPO clip range
    "ent_coef": 0.01,  # Entropy coefficient
    "vf_coef": 0.5,  # Value function coefficient

    # Environment
    "n_envs": 4,  # Windows: use 1, Linux: can use 4+ for speed
    "max_episode_steps": 500,  # Max steps per episode (5 seconds at 0.01s control_dt)

    # Observation/Action
    "observation_space": {
        "include": ["angle", "angular_velocity"],
        "normalize": True,
    },
    "action_space": "continuous",  # Single motor

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
    # Penalty for deviation from upright (90°). Normalized to [0, 1] where
    # 0° error = 0 penalty, 180° error = 1 penalty
    "angle_weight": 2.0,  # Increased from 1.0 — more penalty for falling

    # Penalty for angular velocity. Encourages gentle, controlled motion.
    "velocity_weight": 0.2,  # Increased from 0.1 — penalize jerky motion more

    # Penalty for control effort. Penalizes large motor commands.
    "effort_weight": 0.005,  # Decreased from 0.01 — less penalizing control

    # Bonus reward for being upright (within ±10° of 90°)
    "upright_bonus": 0.5,  # Decreased from 10 — more gradual reward signal
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
    "save_videos": False,
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

    return config
