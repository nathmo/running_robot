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
    # Randomized starts reduce overfitting to a single low-torque local mode.
    "randomize_start_angle": True,
    "start_angle_range": [240.0, 300.0],
}

# ==============================================================================
# RL TRAINING CONFIGURATION
# ==============================================================================
RL = {
    # Algorithm
    "algorithm": "PPO",

    # Training parameters
    "n_steps": 1024,  # Steps per epoch (per environment)
    "n_epochs": 1000,  # Total epochs to train (pendulum converges faster than biped)
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
    "n_envs": 4,  # Windows: use 1, Linux: can use 4+ for speed
    "max_episode_steps": 5000,  # 10x longer horizon: more time to recover and settle
    "max_upright_episode_steps": 50000,  # Allow long balancing runs once upright

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
    # Proximity-based reward: 0 at 270° (start), increases quadratically to upright (90°)
    # At 270°: reward = 0
    # At 180°: reward = (90/180)^2 * scale = 0.25 * scale
    # At 90°:  reward = (180/180)^2 * scale = 1.0 * scale
    "proximity_scale": 1.0,  # Base reward scale for distance to upright

    # Baseline subtraction to eliminate sideways local optimum.
    # With scale=1.0 and baseline=0.30:
    #   270° -> -0.30, 180°/0° -> -0.05, 90° -> +0.70
    "proximity_baseline": 0.30,

    # Bonus reward for being within ±10° of upright (STRONG incentive to reach & stay upright)
    "upright_bonus": 100.0,  # Much stronger — goal is to reach & balance upright

    # Penalty for control effort (motor torque magnitude)
    "effort_weight": 0.0,  # Disabled — let agent use motor freely
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

    return config
