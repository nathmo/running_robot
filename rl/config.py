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
    # move commands are sampled with |vx| in [cmd_vx_min_frac, cmd_vx_frac]: the near-zero-speed
    # regime is NOT trained early — at 0.02-0.1 m/s sliding is the cheapest way to satisfy the
    # tracking terms, and a policy that consolidates there (the measured skating basin) never
    # climbs out. Stand (exactly 0) is trained via p_stand instead.
    cmd_vx_min_frac: float = 0.0
    cmd_forward_only: bool = False      # M2: forward walking only (backward returns at M3)
    p_stand: float = 0.2                # fraction of episodes/commands forced to exactly (0,0)
    cmd_resample_s: float = 4.0         # resample the command every few seconds within an episode
    # forward-command curriculum: linearly ramp cmd_vx_frac from *_start up to cmd_vx_frac over
    # curriculum_steps timesteps, so the policy learns to move a little before full speed is demanded.
    cmd_vx_frac_start: float = 0.0
    curriculum_steps: int = 8_000_000   # 0 disables the ramp (cmd_vx_frac used from the start)

    # ----- observation -----
    history_len: int = 5                # number of past control steps stacked into the observation
    obs_scales: dict = field(default_factory=lambda: dict(
        motor_pos=1.0, motor_vel=0.1, motor_torque=0.01, gravity=1.0, ang_vel=0.25))

    # ----- action (PD position targets) -----
    action_scale: float = 0.5           # action in [-1,1] -> +/- this many rad around the standing pose
    action_filter: float = 0.2          # EMA smoothing of targets (0 = off, 1 = frozen); helps sim2real
    action_delay_steps: int = 1         # fixed actuation delay in control steps: the Pi + moteus/CAN
    #                                     round trip is ~one 50 Hz step on hardware — plant truth, not DR

    # ----- reward weights -----
    # Every penalty term is floored at -penalty_term_cap so no reachable state makes per-step
    # reward so negative that diving into the floor becomes value-optimal (reward normalization
    # is OFF — these raw scales are exactly what PPO sees).
    penalty_term_cap: float = 2.0
    w_track_vx: float = 1.0
    w_track_yaw: float = 0.5
    # forward progress along the commanded heading, SQUARED: fraction of the commanded speed
    # actually achieved. Squared so cruising at 30% of the command is not a comfortable plateau
    # (it pays 9%), unlike the previous linear form the skating policy parked on.
    w_progress: float = 1.0
    # quadratic (bounded, see penalty_term_cap) penalties that keep a restoring gradient far from
    # target, where the exp-kernel tracking rewards saturate to ~0 gradient.
    w_yaw_rate: float = 2.0             # penalize (yaw_rate - cmd_yaw)^2 -> spinning is genuinely costly
    w_heading_pen: float = 1.0          # penalize accumulated heading error^2 (companion to track_heading)
    w_pos_l1: float = 0.5               # linear |pos err| companion to the saturating track_pos kernel
    w_upright: float = 5.0
    w_height: float = 2.5               # soft: walking NEEDS a height bob; 10 froze the torso and
    #                                     made two-feet-planted translation the only cheap motion
    w_vz: float = 0.5
    w_action_rate: float = 0.1          # smoothness: raised 10x now that std is controlled (the old
    #                                     0.01 was noise next to the tracking income)
    w_torque: float = 1.0e-4            # on torque ABOVE the standing baseline (see env), so stance
    #                                     load-sharing isn't cheaper than single-support by design
    w_alive: float = 0.5
    # ----- gait shaping (all sim-side, reward-only; nothing enters the observation) -----
    # foot slip — THE anti-skate term: horizontal toe speed while grounded, quadratic above a
    # deadband, capped. Sized so skating at command speed costs ~1.5-2/step (comparable to the
    # tracking income it buys), while an honest planted stance (<5 cm/s) is free.
    w_foot_slip: float = 8.0
    slip_deadband: float = 0.05         # m/s of tolerated in-contact toe motion
    # per-foot stance-time cap: when a move command is active, a foot grounded for longer than
    # stance_cap_s starts paying per step. Forces EVERY foot to cycle: kills both double-support
    # skating and the one-foot 'flamingo perch'. Honest walking stance (~0.5-0.6 s) is free.
    w_stance_time: float = 0.5
    stance_cap_s: float = 0.7           # allowed continuous ground time at speed
    stance_cap_slow_s: float = 1.0      # allowance below stance_slow_speed (slow walks stance long)
    stance_slow_speed: float = 0.4      # m/s boundary between the two allowances
    # swing clearance — the gradient bridge that makes the FIRST lift-offs pay immediately
    # (the touchdown credit alone arrives 15+ steps late and GAE-attenuated). Paid per step only
    # while a swing is FRESH (air time <= swing_fresh_s: a parked foot stops earning), only above
    # a 2 cm dead zone (ghost-dragging at 1 mm earns nothing), and scaled by progress (marching
    # in place earns little).
    w_clearance: float = 0.4
    clearance_dead_m: float = 0.02
    clearance_scale_m: float = 0.03     # full credit at dead + scale = 5 cm
    swing_fresh_s: float = 0.45
    gait_cmd_gate: float = 0.25         # m/s of |cmd_vx| above which stance-cap + clearance engage
    # feet air-time at touchdown: ONE-SIDED and capped — clip(air_time - min, 0, cap). Chatter
    # earns exactly 0 (slip/stance-cap punish it instead); the old negative side taught the policy
    # that lifting a foot at all was a mistake, which is how the feet ended up glued to the floor.
    w_air_time: float = 2.0
    foot_air_time_min: float = 0.25     # seconds; minimum swing that earns touchdown credit
    air_credit_cap_s: float = 0.45      # credit saturates: long holds can't be farmed
    # a foot counts as grounded if it has sim contact OR its sphere bottom is within grounded_h of
    # the floor (accumulated across all sim substeps + debounced 1 control step), so micro-hops and
    # 1-2 mm ghost-contact dragging can't dodge the contact-gated terms.
    grounded_h: float = 0.005
    fall_penalty: float = 100.0         # raw (no reward normalization); with gamma=0.995 suicide
    #                                     only beats living below -0.5/step, unreachable with the caps
    track_sigma_vx: float = 0.25        # width of the exp tracking kernel (m/s)
    track_sigma_yaw: float = 0.5        # (rad/s); widened so the kernel keeps gradient out to the spin
    # (the height target is NOT configured here: the env derives it from the model's stand
    # keyframe at load time, so it can never go stale when the model is rebuilt)
    # anti-crossing: keep the feet apart laterally so the legs don't cross / scissor.
    w_no_cross: float = 50.0            # weight on the one-sided stance-width penalty
    stance_min_sep: float = 0.25        # m; below this body-frame lateral foot separation -> penalty
    #                                     (nominal stance is ~0.40 m; sep < 0 means feet have crossed)
    w_hip_roll: float = 3.0             # keep the hip-roll (lateral) joints near their neutral pose
    # integrated command-POSE tracking: anchor where the robot should be (xy + heading), advancing
    # by the command, so standing still really stays put and forward really goes straight. This
    # accumulates error, so the policy can't fake it with twitches (unlike instantaneous velocity).
    w_track_pos: float = 2.0            # reward for the base being at the integrated target xy
    track_sigma_pos: float = 0.4        # m; width of the position tracking kernel
    w_track_heading: float = 1.5        # reward for the base heading matching the integrated target
    track_sigma_heading: float = 0.6    # rad; widened so the kernel keeps gradient past a large drift
    w_lat_vel: float = 1.0              # penalize body-frame lateral velocity (go straight, no wander)
    w_angvel_xy: float = 0.05           # penalize roll/pitch angular velocity (steadier, less jerky)

    # ----- episode / termination -----
    episode_s: float = 20.0
    term_height: float = 0.45           # torso below this -> fall
    term_gravity_z: float = -0.5        # body-frame gravity z above this (less negative) -> tipped > 60 deg
    # (no floor-penetration tolerance knob any more: the toe + heel collision spheres physically
    # stop the foot from clipping through the ground, so only a deep-penetration solver-blowup
    # check remains in env._floor_violation — nothing to tune.)
    reset_joint_noise: float = 0.03     # rad of random noise added to the standing pose on reset
    # gentle random base pushes: destabilize the both-feet-planted equilibrium (a push during
    # double support forces a step) and train the step-recovery a real floor demands.
    push_interval_s: float = 0.0        # 0 = off; else mean seconds between pushes
    push_dv: float = 0.4                # m/s of horizontal delta-v per push (random direction)

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
    n_steps: int = 1024                 # rollout length per env (shorter + more envs decorrelates)
    batch_size: int = 4096
    n_epochs: int = 4
    gamma: float = 0.995                # 4 s value horizon at 50 Hz: prices falling + the pose
    #                                     integral correctly (0.99 = 2 s was too short for both)
    gae_lambda: float = 0.95
    learning_rate: float = 3.0e-4       # linearly annealed to lr_final over the run (train.py)
    lr_final: float = 1.0e-4
    clip_range: float = 0.2
    target_kl: float = 0.03             # stop over-updating: prior runs sat at clip_fraction ~0.28
    # entropy: hold ent_coef until stepping has emerged (reward_terms/air_time > gate), THEN anneal
    # to ent_final over ent_anneal_steps (train.py EntropyCallback). A fixed low value collapsed
    # std onto the skate; a fixed high value (0.015) grew std to 2.1 because entropy of a CLIPPED
    # Gaussian is free reward — which is also why log_std is clamped at max_log_std.
    ent_coef: float = 0.01
    ent_final: float = 0.002
    ent_anneal_steps: int = 4_000_000
    ent_gate_air_time: float = 0.02     # raw reward_terms/air_time that counts as 'stepping emerged'
    max_log_std: float = 0.0            # std <= 1.0: beyond the clipped action range is pure farming
    seed: int = 0
    policy_hidden: List[int] = field(default_factory=lambda: [256, 256])

    @property
    def control_dt(self) -> float:
        return 0.0  # filled in by the env from the model's sim timestep * decimation


def m1_stand() -> Config:
    """Milestone 1: learn to stand/balance at command (0,0)."""
    return Config(cmd_vx_frac=0.0, cmd_yaw_frac=0.0, p_stand=1.0)


def m2_walk() -> Config:
    """Milestone 2 (v3): stand + walk FORWARD. Move commands are >= 0.3 m/s from the first step
    (no protected low-speed regime where sliding is cheapest), ramped 0.3 -> 0.6 m/s over 4M
    steps; forward-only (backward doubles the task for no M2 value); gentle pushes on."""
    return Config(cmd_vx_frac=0.6, cmd_vx_frac_start=0.3, cmd_vx_min_frac=0.3,
                  curriculum_steps=4_000_000, cmd_forward_only=True,
                  cmd_yaw_frac=0.0, p_stand=0.10, vx_max=1.0, push_interval_s=5.0)


def m3_turn() -> Config:
    """Milestone 3: full joystick (forward + yaw)."""
    return Config(cmd_vx_frac=1.0, cmd_yaw_frac=1.0, p_stand=0.15)


PRESETS = {"m1_stand": m1_stand, "m2_walk": m2_walk, "m3_turn": m3_turn, "default": Config}


def get_config(name: str = "default") -> Config:
    return PRESETS[name]()
