"""All tunable parameters for DASH-01 RL, in one place.

Grouped as a single dataclass so training scripts can override fields and presets are explicit.
Milestones raise a few knobs (e.g. command ranges, domain randomization); see the presets below.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ----- model & timing -----
    model_path: str = "mujoco/dash01/dash01.xml"
    control_decimation: int = 20        # sim steps per control step. sim is 1 kHz -> 50 Hz control
    keyframe: str = "stand"

    # ----- base DOF locking (the M1..M6 curriculum) -----
    # The base has 6 explicit scalar joints [X, Y, Z, roll, pitch, yaw] (build_model.py). A milestone
    # locks a subset of them to a rigid rail; base_lock[i]=1 activates that DOF's <equality><joint>
    # lock at reset (data.eq_active). Default = all free = today's free-floating plant (M6).
    base_lock: tuple = (0, 0, 0, 0, 0, 0)   # 1 = locked
    # M1 rails Z at a ride-height that is RANDOMIZED per episode so the policy adapts to any height.
    # The legs are seated to that height from a ride-height->posture table (measure_ride_band.py).
    z_rail_randomize: bool = False
    # meters. Default is a sane sub-band around the natural stance (1.0235 m); the FULL measured
    # feasible band is ~[0.81, 1.04] but the low end needs deep fore/aft lunges (the linkage can't
    # crouch without swinging the toe), so we keep the default moderate. Widen for more adaptation.
    z_rail_range: tuple = (0.90, 1.03)
    ride_height_lut: str = "mujoco/dash01/ride_height_lut.npz"

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

    # ----- action representation -----
    # "pd": the policy outputs 6 per-step PD targets (default, unchanged). "fourier": the policy
    # instead outputs, ONCE PER GAIT CYCLE, a Fourier series for cam+thigh (a periodic propulsion gait)
    # + a learnable frequency + a learned abduction (hip_roll) balance reflex. See rl/fourier_gait.py.
    # In fourier mode env.step is a MACRO-step = one full gait cycle (~30/episode instead of 1000).
    # "fourier_step": the SAME 18-dim Fourier action, but re-emitted (and applied) at EVERY 50 Hz
    # control step — the policy can instantly override the gait spec mid-cycle (PD tracks the freshly
    # reconstructed setpoint); abrupt mid-cycle rewrites are priced by w_coef_rate (phase-gated:
    # free at the cycle boundary). One env.step == one control step, like "pd".
    action_mode: str = "pd"             # "pd" | "fourier" | "fourier_step"
    n_harmonics: int = 3                # Fourier harmonics per joint (coeffs/joint = 1 + 2N)
    gait_freq_hz: tuple = (0.5, 3.0)    # learnable cadence range (Hz); policy picks f in this band
    # max Fourier DEVIATION (rad) of cam/thigh from the nominal stance posture (which is always an
    # in-band, validated pose). Kept within the empirically-valid ctrl ranges here (cam ~[-0.6,0.6],
    # thigh ~[-0.2,0.8] from the ride-height LUT) so the coupled 4-bar stays assemblable.
    cam_amp: float = 0.30
    thigh_amp: float = 0.35
    reflex_kp_scale: float = 0.5        # abduction reflex: hip_roll += kp*roll (rad per rad), policy-scaled
    reflex_kd_scale: float = 0.1        #                 + kd*roll_rate (rad per rad/s)
    reflex_bias_scale: float = 0.2      #                 + bias (rad, per-cycle lateral offset)

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
    # ----- maximum-forward-speed objective (M1..M5: run as fast as possible on the rail) -----
    # When speed_mode is on, the command-TRACKING terms (track_vx/progress/track_pos/track_heading/
    # yaw/pos_pen) are switched off and replaced by a single monotone linear reward on forward speed,
    # clip(vx, 0, v_ceiling). "Faster is strictly better" up to the cap; bounded so the per-step
    # return stays below the fall-penalty/(1-gamma) suicide threshold. Straightness is enforced by the
    # base locks (Y/yaw locked in M1..M3) plus the surviving lat_vel/heading penalties, and the gait
    # gate is driven from v_ceiling so all the anti-skating terms stay active at speed.
    speed_mode: bool = False
    w_fwd_speed: float = 2.0
    v_ceiling: float = 2.5              # m/s cap on the speed reward (well above the current ~1.5 max)
    # ----- 100 m sprint objective (sprint_mode) -----
    # One episode = one dash: start standing at x0, run to a finish line sprint_dist_m ahead, then
    # STOP. Reward = dense speed income while running + a constant per-step clock cost + a stop
    # bonus. The clock is what makes TIME matter: per-step vx alone integrates to w*distance no
    # matter how fast it's covered, while sum(-w_time) = -w_time * T — maximizing reward minimizes
    # the dash time. The policy's 'stop' signal is the command channel flipping [1,0] -> [0,0] at
    # the line ([1,0] is exactly what speed_mode policies trained on, so sprint fine-tunes cleanly).
    sprint_mode: bool = False
    sprint_dist_m: float = 100.0        # the finish line (meters of base X from the reset pose)
    sprint_dist_start_m: float = 25.0   # curriculum: line starts here so slow early policies still
    sprint_curriculum_steps: int = 0    #   reach it and learn the stop; 0 = no ramp
    sprint_brake_m: float = 5.0         # free braking zone past the line (sprinters run THROUGH
    #                                     the line; fourier reacts at the next cycle boundary)
    w_time: float = 0.5                 # per-control-step clock cost until stopped; with w_fwd_speed
    #                                     income this sets the break-even pace (~0.25 m/s) below
    #                                     which moving is worse than useless
    w_stop_vel: float = 2.0             # stop-phase income for being stationary: w*exp(-(vx/sigma)^2)
    stop_sigma: float = 0.3             # m/s width of the 'stationary' kernel
    w_overrun: float = 1.0              # per-meter penalty past line+brake zone (capped, see _pen)
    stop_speed_eps: float = 0.15        # |vx| below this counts as stopped
    stop_hold_s: float = 1.0            # must stay stopped this long -> success termination
    finish_bonus: float = 100.0         # terminal success bonus (mirror of fall_penalty; must beat
    #                                     farming the stop-phase income, so keep gamma modest)
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
    # fourier_step only: phase-gated coefficient-change penalty. The policy may rewrite the whole
    # gait spec at any 50 Hz step; billed = sum((applied_action - prev_applied_action)**2)
    # * sin(phase/2)**2, so changing the gait spec exactly at a cycle boundary (phase ~ 0 == 2pi)
    # is FREE and mid-cycle changes pay proportionally. Capped by penalty_term_cap.
    w_coef_rate: float = 0.5
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

    # ----- framework module injection (experiments/ schema; "" = built-in) -----
    # Set by framework/compile.py when an experiment overrides an axis with a local
    # python module ("./reward.py" etc.). Paths are relative to experiment_dir.
    reward_module: str = ""
    obs_module: str = ""
    curriculum_module: str = ""
    network_module: str = ""
    steering_module: str = ""
    experiment_dir: str = ""

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
    """Milestone 3 (legacy): full joystick (forward + yaw) on the free-floating plant."""
    return Config(cmd_vx_frac=1.0, cmd_yaw_frac=1.0, p_stand=0.15)


# ----- base-DOF curriculum M1..M6 (base_lock = [X, Y, Z, roll, pitch, yaw], 1 = locked) -----
# Each milestone frees one more base DOF and asks the robot to run forward as fast as possible.
# One shared model (mujoco/dash01/dash01.xml) serves them all; only the lock mask changes.
def _speed(**kw) -> Config:
    """Common max-forward-speed setup: rail the base, drive the command forward at the speed ceiling,
    forward-only, no protected low-speed regime. Anti-skating gait terms stay on (see reward)."""
    base = dict(speed_mode=True, cmd_forward_only=True, vx_max=2.5, cmd_vx_frac=1.0,
                cmd_vx_min_frac=1.0, cmd_yaw_frac=0.0, p_stand=0.0, push_interval_s=0.0)
    base.update(kw)
    return Config(**base)


def m1() -> Config:
    """M1: only X (fore/aft) free; Y,Z,roll,pitch,yaw railed. Z locked at a per-episode RANDOM
    ride-height so the policy adapts to any height. Max forward speed, alternating L/R gait."""
    return _speed(base_lock=(0, 1, 1, 1, 1, 1), z_rail_randomize=True, z_rail_range=(0.90, 1.03))


def m2() -> Config:
    """M2: X and Z free (robot maintains its own height); Y and all rotation railed. Straight, fast."""
    return _speed(base_lock=(0, 1, 0, 1, 1, 1))


def m3() -> Config:
    """M3: X, Z, pitch free; Y, roll, yaw railed."""
    return _speed(base_lock=(0, 1, 0, 1, 0, 1))


def m4() -> Config:
    """M4: X, Y, Z, pitch free; roll, yaw railed. (Y free -> lat_vel penalty keeps it straight.)"""
    return _speed(base_lock=(0, 0, 0, 1, 0, 1))


def m5() -> Config:
    """M5: X, Y, Z, roll, pitch free; only yaw railed."""
    return _speed(base_lock=(0, 0, 0, 0, 0, 1))


def m6() -> Config:
    """M6: everything free (the full free-floating plant). Max forward speed, fully unconstrained."""
    return _speed(base_lock=(0, 0, 0, 0, 0, 0))


# Fourier cyclic-gait variants: same rail + max-speed task, but the policy outputs a per-cycle Fourier
# gait (cam+thigh) + learnable cadence + a learned abduction balance reflex. gamma is lowered because
# a macro-step is now a whole gait cycle (~0.3-1 s), so the effective horizon is ~30 steps not 1000.
# a macro-step is one gait cycle (~0.3-1 s = ~15-65 control steps), so a "timestep" here is a whole
# cycle: use a shorter rollout, a smaller total budget (in cycles), and a lower gamma than the PD path.
_FOURIER_TRAIN = dict(action_mode="fourier", gamma=0.93,
                      n_steps=256, batch_size=512, total_steps=800_000)


def m1_fourier() -> Config:
    """M1 (rail, random ride-height) with the Fourier cyclic-gait policy."""
    return _speed(base_lock=(0, 1, 1, 1, 1, 1), z_rail_randomize=True, z_rail_range=(0.90, 1.03),
                  **_FOURIER_TRAIN)


def m2_fourier() -> Config:
    """M2 (X,Z free) with the Fourier cyclic-gait policy."""
    return _speed(base_lock=(0, 1, 0, 1, 1, 1), **_FOURIER_TRAIN)


def m3_fourier() -> Config:
    """M3 (X, Z, pitch free) with the Fourier cyclic-gait policy. Freeing pitch on top of m2_fourier;
    base_lock doesn't change the obs/action dims (220/18), so an m2 Fourier checkpoint loads as-is."""
    return _speed(base_lock=(0, 1, 0, 1, 0, 1), **_FOURIER_TRAIN)


# Per-STEP Fourier override variants ("fourier_step"): the same 18-dim gait-spec action, but the
# network re-emits it at every 50 Hz control step (instant override; mid-cycle rewrites priced by
# the phase-gated w_coef_rate penalty, free at the cycle boundary). One env.step == one control
# step again, so NONE of the _FOURIER_TRAIN macro-step overrides apply — the stock per-step PPO
# settings (gamma 0.995, n_steps 1024, batch 4096, 20M steps) are correct here.
def m1_fourier_step() -> Config:
    """M1 (rail, random ride-height) with the per-step Fourier override policy."""
    return _speed(base_lock=(0, 1, 1, 1, 1, 1), z_rail_randomize=True, z_rail_range=(0.90, 1.03),
                  action_mode="fourier_step")


def m2_fourier_step() -> Config:
    """M2 (X,Z free) with the per-step Fourier override policy."""
    return _speed(base_lock=(0, 1, 0, 1, 1, 1), action_mode="fourier_step")


def m3_fourier_step() -> Config:
    """M3 (X, Z, pitch free) with the per-step Fourier override policy."""
    return _speed(base_lock=(0, 1, 0, 1, 0, 1), action_mode="fourier_step")


# 100 m dash (see sprint_mode in Config): run to the line as fast as possible, then stop.
# w_alive must be 0 — a sprinter must not be paid per second of existence (the clock cost w_time
# replaces it); curriculum_steps=0 keeps the cmd_vx ramp (pointless under speed sampling) from
# clobbering curriculum.json, which the sprint distance ramp uses.
_SPRINT = dict(sprint_mode=True, episode_s=60.0, w_alive=0.0, curriculum_steps=0)


def m1_sprint() -> Config:
    """M1 rail 100 m dash, per-step PD policy. gamma lowered so farming the stop-phase income
    can never out-value the finish bonus (2.0/(1-0.99) = 200 gross vs bonus 100 + episode end —
    the hold detector terminates anyway; this just removes the incentive to jitter at the line)."""
    return _speed(base_lock=(0, 1, 1, 1, 1, 1), z_rail_randomize=True, z_rail_range=(0.90, 1.03),
                  gamma=0.99, sprint_curriculum_steps=6_000_000, **_SPRINT)


def m1_sprint_fourier() -> Config:
    """M1 rail 100 m dash with the Fourier cyclic-gait policy."""
    return _speed(base_lock=(0, 1, 1, 1, 1, 1), z_rail_randomize=True, z_rail_range=(0.90, 1.03),
                  sprint_curriculum_steps=400_000, **_FOURIER_TRAIN, **_SPRINT)


def m2_sprint_fourier() -> Config:
    """M2 (X and Z free) 100 m dash with the Fourier cyclic-gait policy: the robot carries its own
    ride height for the whole dash. Strictly harder than m1_sprint_fourier — z_rail_randomize is
    moot (nothing to rail), the height/vz reward terms come back on (the env neutralizes them only
    while Z is locked), and term_height is live, so a gait that doesn't support the body now FALLS
    instead of hanging from the rail. Warm-start it from an m1 sprint policy: base_lock doesn't
    change the obs/action dims (220/18), so the checkpoint loads as-is."""
    return _speed(base_lock=(0, 1, 0, 1, 1, 1), sprint_curriculum_steps=400_000,
                  **_FOURIER_TRAIN, **_SPRINT)


def m3_sprint_fourier() -> Config:
    """M3 (X, Z, pitch free) 100 m dash with the Fourier cyclic-gait policy: the m2_sprint_fourier
    task with the pitch DOF additionally freed, so the body must also stabilize fore/aft attitude
    while it sprints. base_lock doesn't change the obs/action dims (220/18), so an m2 sprint Fourier
    checkpoint loads as-is."""
    return _speed(base_lock=(0, 1, 0, 1, 0, 1), sprint_curriculum_steps=400_000,
                  **_FOURIER_TRAIN, **_SPRINT)


PRESETS = {
    # base-DOF curriculum
    "m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5, "m6": m6,
    # Fourier cyclic-gait variants
    "m1_fourier": m1_fourier, "m2_fourier": m2_fourier, "m3_fourier": m3_fourier,
    # per-step Fourier override variants
    "m1_fourier_step": m1_fourier_step, "m2_fourier_step": m2_fourier_step,
    "m3_fourier_step": m3_fourier_step,
    # 100 m dash
    "m1_sprint": m1_sprint, "m1_sprint_fourier": m1_sprint_fourier,
    "m2_sprint_fourier": m2_sprint_fourier, "m3_sprint_fourier": m3_sprint_fourier,
    # legacy free-floating presets (all-free plant)
    "m1_stand": m1_stand, "m2_walk": m2_walk, "m3_turn": m3_turn, "default": Config,
}


def get_config(name: str = "default") -> Config:
    return PRESETS[name]()


# ----- serialization (orchestrator / --config path) -----------------------------------
# JSON round-trip: lists come back where tuples went in, so coerce by field default type.
_TUPLE_FIELDS = None


def _tuple_fields():
    global _TUPLE_FIELDS
    if _TUPLE_FIELDS is None:
        from dataclasses import fields as _f, MISSING
        _TUPLE_FIELDS = {x.name for x in _f(Config)
                         if x.default is not MISSING and isinstance(x.default, tuple)}
    return _TUPLE_FIELDS


def config_to_dict(cfg: Config) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


def config_from_dict(d: dict) -> Config:
    """Rebuild a Config from a JSON-loaded dict. Unknown keys are warned about and
    dropped (forward compatibility with configs written by newer code)."""
    from dataclasses import fields as _f
    known = {x.name for x in _f(Config)}
    clean = {}
    for k, v in d.items():
        if k not in known:
            print(f"[config] WARNING: dropping unknown Config field '{k}'")
            continue
        clean[k] = tuple(v) if k in _tuple_fields() and isinstance(v, list) else v
    return Config(**clean)


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    """Return a copy of cfg with `overrides` applied. Unknown field names are a hard
    error (a typo'd sweep axis must fail at submit time, not train silently)."""
    from dataclasses import replace, fields as _f
    known = {x.name for x in _f(Config)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise KeyError(f"unknown Config field(s): {unknown}")
    fixed = {k: (tuple(v) if k in _tuple_fields() and isinstance(v, list) else v)
             for k, v in overrides.items()}
    return replace(cfg, **fixed)
