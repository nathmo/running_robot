"""BalanceRL -- a push-recovery balance controller for DASH-01. Every tunable, one dataclass.

The policy stands still and gets shoved. It writes the MIT force-control frame directly, per joint
and every 10 ms tick: a position target, Kp and Kd (no torque feed-forward: the drives' torque span
is unidentified, see controller/deploy/mit.py). Nothing else sits between the network and the
drive -- no gait generator, no clock, no latch, no reflex.

Shared with RLframework/ (copied, not imported, so the walker can move without breaking this):
the plant (model/dash01_free.xml), the drive law and delay (drive.py), the per-episode plant draw
(plant.py) and the measurement chain (noise, biases, IMU mount rotation, accelerometer leak,
dropout -- env._frame). Dimensioned for 100 Hz control on MJX, 1 kHz physics.
"""
from dataclasses import dataclass, field, asdict, fields
from typing import Any


@dataclass
class Config:
    # ----- plant & timing ------------------------------------------------------------------------
    model_path: str = "model/dash01_free.xml"
    keyframe: str = "stand"
    control_decimation: int = 10            # 1 kHz physics / 10 = 100 Hz control
    mjx_iterations: int = 16                # see RLframework/config.py: 16 x 8 replays the CPU plant
    mjx_ls_iterations: int = 8
    episode_s: float = 12.0                 # ~3 pushes per episode

    # ----- the operator hold (how this robot is really started) -----------------------------------
    # The base is supported for the first hold_s of the episode, then let go -- the bring-up the
    # runner's approach phase performs (crawl to the stance, hold, release). It is also what makes the
    # slow channels usable: the DC lean that cancels a CoM offset is inferred over ~1 s, and without a
    # hold the robot is already toppling by then (measured: ~20% of episodes fell unprompted).
    hold_s_range: tuple = (0.4, 1.0)
    hold_release_v: float = 0.05            # m/s of stray velocity the hand imparts on release
    # the basin is +-0.02 rad, so a +-0.02 rad reset put every episode on its edge; the robot homes
    # to +-0.3 deg flat / +-0.8 deg knee on the bench anyway (dash01-homing-tolerance)
    reset_joint_noise: float = 0.01         # rad, uniform, on every leg hinge at reset

    # ----- the action: the MIT frame, 3 x 6 ------------------------------------------------------
    # a = [q 6 | kp 6 | kd 6] in actuator order [hip_roll_L, cam_L, thigh_L, hip_roll_R, cam_R, thigh_R]
    # q_ref = clip(nominal_cmd + q_scale * a_q, q_lo, q_hi), then the no-load slew cap.
    # kp = kp0 * (kp_hi/kp0)^a   for a >= 0,   kp0 * (kp0/kp_lo)^a   for a < 0   (same for kd)
    # so a = 0 is EXACTLY the standing command the walker holds its stance with.
    # rad per action unit. 0.5 rad of thigh is ~40 cm of foot travel: room for a recovery step,
    # while exploration noise on it stays small (measured: std 0.1 x 0.8 rad fell every 1.6 s)
    q_scale: tuple = (0.25, 0.50, 0.50, 0.25, 0.50, 0.50)
    drive_kp: tuple = (120.0, 200.0, 200.0, 120.0, 200.0, 200.0)   # a_kp = 0
    drive_kd: tuple = (4.0, 5.0, 5.0, 4.0, 5.0, 5.0)               # a_kd = 0
    kp_lo: float = 20.0
    kp_hi: float = 500.0                    # the MIT wire maximum
    kd_lo: float = 0.2
    kd_hi: float = 5.0                      # the MIT wire maximum
    motor_vel_limit: tuple = (10.30, 22.01, 22.01, 10.30, 22.01, 22.01)
    motor_accel_limit: float = 0.0

    # ----- the stabilising prior: a pitch reflex the policy's action ADDS to --------------------
    # Measured on this plant (2026-09-23): the standing basin is only about +-0.02 rad wide in
    # commanded thigh, and a constant pose cannot hold a CoM offset -- but thigh = -(1.0 g_x +
    # 0.05 w_y) holds +3 cm for 6 s. So the policy is not asked to discover a stabiliser from
    # scratch (two campaigns failed to); it starts from one and learns the residual, exactly as the
    # walker learns a residual on a gait spec. Built from MEASURED gravity and gyro (the newest
    # observation frame), so the Pi runtime reproduces it in numpy.
    reflex_enable: bool = True
    # MOSTLY RATE DAMPING, deliberately. The stance is PASSIVELY stable (zero action stands 32/32 at
    # nominal), and the measured gravity carries a calibration error -- IMU mount 1 deg, bias 0.01 --
    # as large as the whole +-0.02 rad basin, so a stiff kp on it injects that error into the command:
    # measured 2026-09-23, kp 1.0 / kd 0.05 stood only 9/32 where doing nothing stood 32/32.
    reflex_kp_pitch: float = 0.3            # rad of thigh per unit of measured gravity x
    reflex_kd_pitch: float = 0.06           # rad of thigh per rad/s of measured gyro y
    reflex_kp_roll: float = 0.0             # rad of hip roll per unit of measured gravity y (untested)
    reflex_kd_roll: float = 0.0
    reflex_clip_rad: float = 0.25           # the prior's own authority per joint

    # ----- drive (identical to RLframework) ------------------------------------------------------
    drive_delay_ms: float = 12.0
    drive_delay_range_ms: tuple = (6.0, 18.0)
    motor_kt_joint: tuple = (4.655, 2.176, 2.176, 4.655, 2.176, 2.176)
    motor_r_ohm: tuple = (0.665, 0.229, 0.229, 0.665, 0.229, 0.229)
    motor_bus_volts: float = 48.0
    ctrl_jitter_ms: float = 2.0             # +- per tick, at full dr_scale
    ctrl_drop_prob: float = 0.02            # hold-last-action probability, at full dr_scale

    # ----- the slow channels (the once-block) ----------------------------------------------------
    # The DC lean that cancels a CoM offset is ~0.04 rad per 3 cm and has to be INFERRED inside the
    # episode: the calibration error is basin-sized, so the policy cannot read the right pose off one
    # frame. 190 ms of history cannot separate a systematic offset from noise, so it gets explicit
    # leaky integrals of what it measures. All from MEASURED signals: the Pi runtime reproduces them.
    ema_slow_s: float = 2.0                 # gravity xy, joint error, torque
    ema_mid_s: float = 0.5                  # gravity xy again: 2 s alone is too slow for the first
    #                                         second of an episode, which is where the robot fell
    ema_fast_s: float = 0.3                 # gyro xy
    # the block is [g_xy 2, g_xy(mid) 2, w_xy 2, q-q_nom 6, tau/peak 6] = env.ONCE_DIM

    # ----- the action filter --------------------------------------------------------------------
    # The task wants a nearly CONSTANT command plus slow corrections, so per-tick exploration noise is
    # almost pure disturbance. A one-pole filter on the target keeps the learned mean responsive while
    # attenuating the sampled noise the plant actually feels. walk_mit's v1 bundle had one too.
    action_filter_tau_s: float = 0.08

    # ----- observation ---------------------------------------------------------------------------
    history_len: int = 10
    history_stride: int = 2                 # 10 frames x 20 ms = 190 ms window
    obs_scales: dict = field(default_factory=lambda: dict(
        motor_pos=1.0, motor_vel=0.1, motor_torque=0.01, gravity=1.0, ang_vel=0.25, base_vel=1.0))
    clip_obs: float = 10.0
    obs_eps: float = 1e-8

    # ----- CoM shift: the whole-robot centre of mass, +-com_shift_m per axis ---------------------
    # Applied as an offset of the TORSO's inertial frame, scaled by M_total / m_torso so the
    # whole-robot CoM moves by the drawn amount (the legs are a third of the mass).
    com_shift_m: tuple = (0.03, 0.03, 0.03)

    # ----- plant DR (every range is scaled by dr_scale) ------------------------------------------
    dr_enable: bool = True
    dr_mass_global: float = 0.10
    dr_mass_body: float = 0.10
    dr_inertia: float = 0.20
    dr_com_offset: float = 0.005            # per-body CAD inertial jitter, m (the big one is above)
    dr_friction_range: tuple = (0.6, 1.2)
    dr_kp: float = 0.15
    dr_kv: float = 0.20
    dr_torque: float = 0.10
    dr_joint_damping: float = 0.30
    # A 3 deg slope is a ~4 cm equivalent CoM shift at this CoM height -- bigger than the +-3 cm the
    # task asks for, and it alone can exceed the 4.3 cm sole margin. 1 deg (1.4 cm) leaves the spec
    # the dominant term instead of a slope nobody asked for.
    dr_gravity_tilt_deg: float = 1.0        # floor slope
    dr_loop_site_m: float = 0.0015          # in-plane only (plant.py masks y: it kills the solver)
    dr_joint_zero_deg: float = 0.5          # homing error (measured +-0.3 flat, +-0.8 knee)
    dr_imu_rot_deg: float = 1.0
    dr_imu_dropout_prob: float = 0.001
    dr_imu_dropout_s: float = 0.25

    # ----- sensor noise: EXACTLY the walker's chain ----------------------------------------------
    obs_noise_enable: bool = True
    noise_encoder: float = 0.003            # rad
    noise_motor_vel: float = 0.15           # rad/s
    noise_torque: float = 1.5               # N*m
    noise_torque_gain: float = 0.08
    noise_grav: float = 0.0013
    noise_grav_bias: float = 0.01
    noise_gyro: float = 0.0012
    noise_gyro_bias: float = 0.00005
    noise_gyro_walk: float = 0.00002
    noise_accel_leak: float = 0.15

    # ----- pushes: the curriculum ----------------------------------------------------------------
    # A force pulse on the torso, any direction (azimuth uniform, elevation +-push_elev_deg), at a
    # random point on the torso (so it also torques it). Its size is the IMPULSE, expressed as the
    # whole-robot velocity change dv = J / M_total -- the level the curriculum moves.
    push_first_s: tuple = (1.0, 2.0)        # first push after the robot has settled
    push_interval_s: tuple = (2.5, 4.0)
    push_dur_s: tuple = (0.05, 0.15)
    push_elev_deg: float = 30.0
    push_point_box: tuple = ((-0.06, 0.06), (-0.08, 0.08), (-0.05, 0.12))   # torso frame, m
    push_easy_frac: float = 0.3             # share of pushes drawn from [0, 0.5 L] (no forgetting)
    push_survive_s: float = 2.0             # upright this long after the push starts = survived
    push_level_start: float = 0.15          # m/s
    push_level_max: float = 2.5
    push_level_up: float = 1.05             # x on promotion
    push_level_down: float = 0.95           # x on retreat
    # 0.70, not 0.80: measured at 0.58 and climbing slowly, and the level self-regulates anyway --
    # promote too early and survival falls under push_gate_down, which retreats it.
    push_gate_up: float = 0.70              # hard-push survival EMA to promote
    push_gate_down: float = 0.45            # ... to retreat
    push_gate_min_n: int = 400              # hard pushes observed at a level before it may move

    # ----- the other curriculum: plant width (CoM shift + DR) ------------------------------------
    # A CLOCK, not a gate, and it does not hold the pushes back. Measured the hard way (bal_s0/s1,
    # 2026-09-22): gating it on "<= 5% of episodes fall with no push" stalled at plant_scale 0.50-0.55
    # for 171 M steps, because a stander that has never been pushed hard sits at ~10% -- and with the
    # push level chained behind it, the pushes never grew past 0.15 m/s either. The CoM shift is a
    # STATIC property of the robot: the policy has to meet it early and train against it, not be
    # protected from it until it is already good.
    plant_scale_start: float = 0.2
    plant_ramp_steps: int = 20_000_000      # 0.2 -> 1.0, then full width for the rest of the run

    # ----- reward (per tick) ---------------------------------------------------------------------
    w_alive: float = 1.0
    w_upright: float = 1.0                  # exp(-tilt^2 / sig^2)
    sig_upright: float = 0.20               # rad
    w_height: float = 0.5
    sig_height: float = 0.04                # m
    # THE CENTRAL SHAPING TERM: the CoM over the centre of the feet that are down. This is what pays
    # for a capture step and for shuffling the stance under an offset CoM. It replaced a
    # "return to the nominal joint pose" term that paid the policy NOT to move its feet.
    w_com_support: float = 1.5
    sig_com_support: float = 0.03           # m
    w_calm_vel: float = 0.3                 # exp(-|v_xy|^2 / sig^2), only once a push is over
    sig_calm_vel: float = 0.30              # wide: re-centring the stance is not "not calm"
    w_calm_pose: float = 0.2                # tie-breaker only, and only while the CoM is centred
    sig_calm_pose: float = 0.50
    calm_after_s: float = 1.0               # a push is "over" this long after it ends
    w_angvel: float = 0.05                  # |w_xy|^2
    w_torque: float = 0.5                   # mean (tau / tau_peak)^2
    w_action_rate: float = 0.02             # |a - a_prev|^2
    w_slip: float = 0.5                     # foot |v_xy|^2, once it has been down slip_dwell_ticks
    slip_dwell_ticks: int = 10              # 100 ms: a touchdown is not a slip
    # mean kp / kp_hi. Small: at 0.1 the first campaign walked Kp down from 200 to 92-130, buying a
    # cheap standing pose at the cost of the authority a recovery needs.
    w_gain: float = 0.02
    w_joint_limit: float = 1.0              # target within 5% of a joint stop
    fall_penalty: float = 20.0
    term_height: float = 0.55               # torso (stand 0.838)
    term_gravity_z: float = -0.5            # tilt > 60 deg

    # ----- PPO -----------------------------------------------------------------------------------
    n_envs: int = 2048
    n_steps: int = 24
    total_steps: int = 200_000_000
    gamma: float = 0.995                    # 2 s horizon at 100 Hz: a recovery takes 1-2 s
    gae_lambda: float = 0.95
    n_epochs: int = 5
    batch_size: int = 16384                 # minibatch
    learning_rate: float = 1e-4             # small std = a sensitive KL; the adaptive rule raises it
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    # A KL target is a budget in units of sigma, and sigma here is 0.02: the same policy change is
    # ~40x the KL it would be at sigma 0.4. 0.02 with the warmup below is what keeps the adaptive rule
    # from flooring the step size on the first update (measured: KL 19.8 on update 1 without it).
    target_kl: float = 0.02                 # adaptive lr (rl_games): /1.5 above 2x, x1.5 below 0.5x
    lr_warmup_updates: int = 300            # Adam's first step moves every weight at once
    clip_range: float = 0.2
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    ent_coef: float = 0.0
    # the stand is a knife-edge (measured on this env, 20% plant, 6 s: std 0.03 on every dim x 0.8 rad
    # -> 11/32 upright, std 0.1 -> 0/32), so position noise starts small; gain noise is gentler
    # Exploration has to be able to FIND a recovery step, which is a large coordinated motion, not a
    # stance tweak. bal_s0/s1 at 0.05 (= 0.025 rad) never found one in 171 M steps. 0.12 is 0.06 rad
    # per tick, between the two doses the pre-training probe measured (0.024 rad -> 11/32 upright over
    # 6 s, 0.08 rad -> 0/32): early episodes are short and full of falls, which is the lesson.
    # MEASURED basin: +-0.02 rad of commanded thigh. Exploration must sit INSIDE it -- campaign 1 at
    # 0.025 rad was at its edge and campaign 2 at 0.06 rad was three times over it and measurably
    # worse (ep_len 517 vs 851). 0.02 action units = 0.01 rad on the thigh pair.
    init_std: float = 0.02                  # position dims
    init_std_gain: float = 0.15             # kp / kd dims
    min_std: float = 0.008                  # position dims
    max_std: float = 0.05
    min_std_gain: float = 0.02               # kp / kd dims
    max_std_gain: float = 0.30
    w_bound: float = 1.0                    # relu(|mu| - 1)^2
    w_sym: float = 0.5                      # ||pi(M s) - M pi(s)||^2 on the actor mean
    policy_hidden: tuple = (256, 256)       # two layers: the deploy PolicyNet is wired for exactly two
    est_hidden: tuple = (128, 64)
    est_lr: float = 1e-3
    est_epochs: int = 2
    est_batch: int = 16384
    eval_every: int = 50                    # rollouts
    ckpt_every: int = 100


PRESETS = {
    "balance": dict(),
    # CPU smoke: tiny batch, few steps
    "smoke": dict(n_envs=16, n_steps=8, total_steps=16 * 8 * 3, batch_size=64, est_batch=64,
                  eval_every=2, ckpt_every=2, mjx_iterations=8),
}


def get_config(name="balance", **over) -> Config:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; have {sorted(PRESETS)}")
    c = Config(**PRESETS[name])
    for k, v in over.items():
        if not hasattr(c, k):
            raise KeyError(k)
        setattr(c, k, v)
    return c


def config_to_dict(c: Config) -> dict:
    return asdict(c)


def config_from_dict(d: dict) -> Config:
    names = {f.name for f in fields(Config)}
    kw = {}
    for k, v in d.items():
        if k in names:
            dflt = getattr(Config(), k)
            kw[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v) \
                if isinstance(dflt, tuple) and isinstance(v, list) else v
    return Config(**kw)
