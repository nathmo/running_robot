"""
Configuration for RL training framework
Organized by section for easy modification
"""

# ==============================================================================
# TERRAIN CONFIGURATION
# ==============================================================================
TERRAIN = {
    "type": "perlin",  # "flat", "perlin", or "stairs"
    "seed": 42,  # Fixed seed for reproducibility

    # Perlin noise parameters
    "perlin_scale": 1.0,  # Scale of the noise (0.5 = smaller features, 2.0 = larger)
    "perlin_octaves": 4,  # Number of octaves for detail
    "perlin_persistence": 0.5,  # How much each octave contributes
    "perlin_lacunarity": 2.0,  # Frequency multiplier for octaves

    # Height parameters
    "height_scale": 0.2,  # Max height variation (meters)
    "height_offset": 0.0,  # Base height

    # Grid size (resolution)
    "grid_size": 256,  # NxN grid for heightfield
    "grid_spacing": 0.1,  # Physical spacing between grid points (m)

    # Friction randomization during training
    "friction_range": [0.6, 1.2],  # Random friction per episode
    "friction_base": 0.8,
}

# ==============================================================================
# PATH/TRACK CONFIGURATION
# ==============================================================================
PATHS = {
    "track_types": ["circle", "sine_wave", "spiral"],
    "radii": [2.0, 3.0, 5.0, 10.0],  # Different radii for variety
    "sine_amplitudes": [0.5, 1.0, 1.5],  # For sine wave track
    "spawn_distribution": "uniform",  # "uniform" or "clustered"

    # Start position randomization
    "start_position_noise": 0.3,  # std dev for x,y perturbation (m)
    "heading_randomization": True,  # Random initial orientation
}

# ==============================================================================
# ROBOT CONFIGURATION
# ==============================================================================
ROBOT = {
    "urdf_path": "assets/simple_biped.xml",  # Can be overridden per run
    "base_mass": 7.0,  # Optional: override URDF mass for quick experiments
    "use_base_mass_override": False,

    # Control parameters
    "control_mode": "pd",  # "pd" or "torque"
    "action_repeat": 10,  # Repeat action for N simulation steps
    "sim_dt": 0.001,  # Simulation timestep (seconds)
    "control_dt": 0.01,  # Control timestep after action_repeat

    # PD gains (if control_mode == "pd")
    "motor_kp": 55.0,
    "motor_kd": 0.8,
    "torque_limit": 33.5,
    "velocity_limit": 21.0,
}

# ==============================================================================
# RL TRAINING CONFIGURATION
# ==============================================================================
RL = {
    # Algorithm
    "algorithm": "PPO",  # "PPO", "TD3", "SAC"

    # Training parameters
    "n_steps": 2048,  # Steps per epoch (per environment)
    "n_epochs": 500,  # Total epochs to train
    "batch_size": 64,
    "learning_rate": 3e-4,
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # GAE lambda
    "clip_range": 0.2,  # PPO clip range
    "ent_coef": 0.0,  # Entropy coefficient
    "vf_coef": 0.5,  # Value function coefficient

    # Environment
    "n_envs": 1,  # Windows: use 1, Linux: can use 4+ for speed
    "max_episode_steps": 500,  # Max steps per episode

    # Observation/Action
    "observation_space": {
        "include": ["joint_pos", "joint_vel", "base_orient", "base_vel", "prev_action"],
        "normalize": True,
    },
    "action_space": "continuous",  # All motors

    # Checkpointing
    "checkpoint_interval": 10,  # Save checkpoint every N epochs
    "keep_last_n_checkpoints": 20,

    # Logging
    "log_interval": 1,  # Print stats every N steps
}

# ==============================================================================
# REWARD CONFIGURATION
# ==============================================================================
REWARD = {
    "forward_speed_weight": 1.0,
    "forward_speed_scale": 1.0,  # Normalize speed (~2 m/s reference)

    # Stability and smoothness
    "stability_weight": 0.1,  # Penalize base rotation
    "action_smoothness_weight": 0.01,  # Penalize abrupt changes

    # Energy efficiency
    "energy_weight": 0.01,  # Penalize motor power

    # Track deviation
    "track_deviation_weight": 0.05,

    # Ground contact
    "foot_contact_weight": 0.0,  # Encourage contact (if needed)

    # Alive bonus per step
    "alive_bonus": 0.01,
}

# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================
ENVIRONMENT = {
    "render": False,
    "render_mode": "human",  # "human" or "rgb_array"
    "camera_distance": 3.0,
    "camera_lookat": [0, 0, 0],

    # Domain randomization
    "randomize_friction": True,
    "randomize_mass": True,  # Can enable for robustness
    "mass_range": [0.9, 1.1],  # Multiplier on URDF mass

    # Physics
    "gravity": [0, 0, -9.81],
    "wind": [0, 0, 0],  # Can be randomized
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
        "easy_terrain": {
            "TERRAIN": {**TERRAIN, "height_scale": 0.05, "perlin_scale": 2.0},
        },
        "hard_terrain": {
            "TERRAIN": {**TERRAIN, "height_scale": 0.4, "perlin_scale": 0.5},
        },
        "fast_training": {
            "RL": {**RL, "n_steps": 8192, "n_epochs": 100},
        },
        "long_training": {
            "RL": {**RL, "n_steps": 64000, "n_epochs": 2000},
        },
    }

    preset = presets.get(config_name, {})

    config = {
        "TERRAIN": {**TERRAIN, **preset.get("TERRAIN", {})},
        "ROBOT": {**ROBOT, **preset.get("ROBOT", {})},
        "RL": {**RL, **preset.get("RL", {})},
        "REWARD": {**REWARD, **preset.get("REWARD", {})},
        "ENVIRONMENT": {**ENVIRONMENT, **preset.get("ENVIRONMENT", {})},
        "LOGGING": {**LOGGING, **preset.get("LOGGING", {})},
        "PATHS": {**PATHS, **preset.get("PATHS", {})},
    }

    return config
