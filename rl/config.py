"""All tunable parameters for SpiderBot RL, in one place.

Grouped as a single dataclass so training scripts can override fields and presets are explicit.
Milestones raise a few knobs (e.g. command ranges, domain randomization); see the presets below.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ----- model & timing -----
    model_path: str = "mujoco/spiderbot/spiderbot.xml"
    control_decimation: int = 20        # sim steps per control step. sim is 1 kHz -> 50 Hz control
    keyframe: str = "stand"

    # ----- joystick command -----
    # command is 2 numbers in [-1, 1]: [forward, yaw]. These map to physical units:
    vx_max: float = 1.5                 # m/s  at forward = +1   (config constant; raise later for speed)
    yaw_max: float = 2.0                # rad/s at yaw     = +1
    # how much of that range we actually SAMPLE during training (curriculum: M1=0 -> stand only):
    cmd_vx_frac: float = 0.0
    cmd_yaw_frac: float = 0.0
    p_stand: float = 0.2                # fraction of episodes/commands forced to exactly (0,0)
    cmd_resample_s: float = 4.0         # resample the command every few seconds within an episode

    # ----- observation -----
    history_len: int = 5                # number of past control steps stacked into the observation
    obs_scales: dict = field(default_factory=lambda: dict(
        motor_pos=1.0, motor_vel=0.1, motor_torque=0.01, gravity=1.0, ang_vel=0.25))

    # ----- action (PD position targets) -----
    action_scale: float = 0.5           # action in [-1,1] -> +/- this many rad around the standing pose
    action_filter: float = 0.2          # EMA smoothing of targets (0 = off, 1 = frozen); helps sim2real

    # ----- reward weights -----
    w_track_vx: float = 1.0
    w_track_yaw: float = 0.5
    w_upright: float = 5.0
    w_height: float = 10.0
    w_vz: float = 1.0
    w_action_rate: float = 0.01
    w_torque: float = 1.0e-4
    w_alive: float = 0.5
    # feet air-time (anti-vibration stepping): reward each foot at touchdown by (air_time - min);
    # steps shorter than the minimum get NEGATIVE reward, so tiny chattering is punished while
    # deliberate strides are rewarded. Uses sim contact (reward-only, never in the observation).
    w_air_time: float = 2.0
    foot_air_time_min: float = 0.3      # seconds; the minimum useful step (swing) duration
    fall_penalty: float = 200.0
    track_sigma_vx: float = 0.25        # width of the exp tracking kernel (m/s)
    track_sigma_yaw: float = 0.25       # (rad/s)
    height_target: float = 0.843        # standing torso height (from the model keyframe)

    # ----- episode / termination -----
    episode_s: float = 20.0
    term_height: float = 0.45           # torso below this -> fall
    term_gravity_z: float = -0.5        # body-frame gravity z above this (less negative) -> tipped > 60 deg
    floor_penetration_tol: float = 0.02 # only the toe spheres may touch; if any foot/shin point sinks
    #                                     deeper than this below the floor -> forbidden -> terminate
    reset_joint_noise: float = 0.03     # rad of random noise added to the standing pose on reset

    # ----- domain randomization (enabled at M4; ranges are multiplicative unless noted) -----
    dr_enabled: bool = False
    dr_mass: float = 0.15               # +/- fraction on body masses
    dr_friction: tuple = (0.6, 1.2)     # foot-ground friction range
    dr_motor_strength: float = 0.15     # +/- fraction on applied torque
    dr_pd_gain: float = 0.15            # +/- fraction on kp/kv
    dr_latency_steps: int = 2           # max control-step delay applied to actions
    dr_imu_gyro_noise: float = 0.1      # rad/s std
    dr_motor_pos_noise: float = 0.01    # rad std
    dr_push_interval_s: float = 0.0     # 0 = no random pushes

    # ----- PPO / training -----
    n_envs: int = 8
    total_steps: int = 20_000_000
    n_steps: int = 2048                 # rollout length per env
    batch_size: int = 4096
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3.0e-4
    clip_range: float = 0.2
    ent_coef: float = 0.0
    seed: int = 0
    policy_hidden: List[int] = field(default_factory=lambda: [256, 256])

    @property
    def control_dt(self) -> float:
        return 0.0  # filled in by the env from the model's sim timestep * decimation


def m1_stand() -> Config:
    """Milestone 1: learn to stand/balance at command (0,0)."""
    return Config(cmd_vx_frac=0.0, cmd_yaw_frac=0.0, p_stand=1.0)


def m2_walk() -> Config:
    """Milestone 2: track forward velocity (curriculum raises cmd_vx_frac toward 1.0)."""
    return Config(cmd_vx_frac=0.3, cmd_yaw_frac=0.0, p_stand=0.2)


def m3_turn() -> Config:
    """Milestone 3: full joystick (forward + yaw)."""
    return Config(cmd_vx_frac=1.0, cmd_yaw_frac=1.0, p_stand=0.15)


PRESETS = {"m1_stand": m1_stand, "m2_walk": m2_walk, "m3_turn": m3_turn, "default": Config}


def get_config(name: str = "default") -> Config:
    return PRESETS[name]()
