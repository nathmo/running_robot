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
    # When True, the environment uses a StraightPath (infinite radius) instead
    # of sampling from `radii`. Simpler task: learn to run in a straight line.
    "straight_line": True,

    "track_types": ["circle", "sine_wave", "spiral"],
    "radii": [2.0, 3.0, 5.0, 10.0],  # Ignored when straight_line=True
    "sine_amplitudes": [0.5, 1.0, 1.5],
    "spawn_distribution": "uniform",

    # Start position randomization
    "start_position_noise": 0.3,
    "heading_randomization": True,
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

    # Physical motor spec (hard-enforced by the sim, not just reward-shaped):
    #   Torque: ±100 Nm, enforced by actuator gear=100 + ctrlrange [-1,1] in the XML.
    #   Speed:  ±100 rpm = ±10.47 rad/s, enforced in env.step as motor saturation
    #           (torque is zeroed when the joint is at limit and being pushed further).
    "torque_limit_nm": 100.0,
    "joint_velocity_limit": 10.47,

    # Spawn height of the base link at reset. Chosen so the feet just clear the
    # ground for the simple_biped model (foot bottom = base_z - 0.61 with zero
    # joint angles). Change this when switching to a robot with different leg
    # geometry.
    "spawn_height": 0.62,
}

# ==============================================================================
# RL TRAINING CONFIGURATION
# ==============================================================================
RL = {
    # Algorithm
    "algorithm": "PPO",  # "PPO", "TD3", "SAC"

    # Training parameters
    "n_steps": 2048,  # Steps per epoch (per environment)
    "n_epochs": 2500,  # Total epochs to train
    "batch_size": 64,
    "learning_rate": 3e-4,  # Used as the starting LR; train.py decays linearly to 0
                            # over `n_epochs` to stabilize late-stage PPO updates.
    "gamma": 0.995,  # Discount factor. 0.99 only values ~1s into the future at
                     # control_dt=0.01, which encourages burst-and-fall gaits.
                     # 0.995 ≈ 2s horizon, which rewards sustained locomotion.
    "gae_lambda": 0.95,  # GAE lambda
    "clip_range": 0.2,  # PPO clip range
    "ent_coef": 0.05,  # Entropy coefficient. Bumped from 0.01 after a policy-
                       # collapse incident (deterministic fall-backward at epoch 200);
                       # more entropy pressure keeps exploration alive early on.
    "vf_coef": 0.5,  # Value function coefficient

    # Environment
    "n_envs": 8,  # Windows: use 1, Linux: can use 4+ for speed
    "max_episode_steps": 2500,  # Max steps per episode

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
    "log_interval": 10,  # Print stats every N steps
}

# ==============================================================================
# REWARD CONFIGURATION
# ==============================================================================
REWARD = {
    # Forward progress along the path. Reward is min(v_along_path, target_speed)*weight,
    # so going faster than target_speed stops earning extra — prevents reward runaway
    # and gives the policy a clear "good enough" signal.
    "forward_speed_weight": 1.0,
    "forward_target_speed": 2.0,  # m/s — realistic biped running speed. Was 20,
                                  # which made the reward effectively uncapped and
                                  # amplified early-training gradient noise.

    # Upright penalty: 1 - (body_z . world_z). 0 when perfectly upright, up to 2 upside-down.
    # Replaces the old ||angvel|| penalty, which punished the swing motion we actually want.
    "upright_weight": 0.5,

    # Discourage abrupt changes between consecutive actions. 0.05 was too
    # aggressive early in training (policy collapsed to "don't move / fall
    # fast"). 0.02 is still 2× the original 0.01 — enough smoothness pressure
    # for sim2real prep without suffocating exploration.
    "action_smoothness_weight": 0.02,

    # Distance from the path (m). For StraightPath this is |y|.
    "track_deviation_weight": 0.2,

    # Survival incentive. 10/step was way too high — a 500-step episode was
    # worth +5000 alive bonus alone, swamping every other signal and making
    # "survive at all costs" the attractor. With no ankle, the robot can't
    # passively survive, so this created a paralysis trap.
    "alive_bonus": 0.5,

    # One-shot penalty applied at the step that terminates an episode via a fall.
    # Disabled (0) while 100% of episodes end in a fall: early termination already
    # truncates future reward, so an extra terminal penalty just floods the gradient
    # with "every action leads to -20" and can't distinguish better/worse falls.
    # Re-enable (try 2-5) once the policy sometimes survives past the settling window.
    "fall_penalty": 0.0,

    # Feet-air-time reward (ANYmal / Rudin 2022). At each foot-landing event,
    # add (airtime_at_landing - threshold) * weight. Encourages a stepping gait
    # at a target cadence; negative contribution if the step was shorter than
    # threshold. Gated off below min_speed so it can't be farmed by marching in place.
    "feet_air_time_weight": 0.1,
    "feet_air_time_threshold": 0.5,  # seconds per foot per swing phase
    "feet_air_time_min_speed": 0.0,   # m/s along-path gate. Was 0.5, but that
                                      # created a chicken-and-egg: the robot
                                      # needs to step to reach 0.5 m/s, but the
                                      # stepping reward is gated off until then.
                                      # Re-raise after walking emerges, to prevent
                                      # march-in-place farming.
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
