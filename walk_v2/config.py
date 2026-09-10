"""DASH-01 Walker v2 — all tunable parameters, one dataclass, explicit presets.

Ground truth: the "DASH-01 Walker v2" design artifact (rev 2026-09-09). Section numbers in the
comments below (§01..§13) refer to it. Everything here is dimensioned for 100 Hz control on the
MJX (GPU-batched) plant; the walk_mit/ stack this succeeds ran 200 Hz on classic MuJoCo.

Every step-denominated number (curriculum spans, deadlines, anneals) is HALF its walk_mit value:
300 M steps at 100 Hz is the same 34.7 sim-days as 600 M at 200 Hz (§07 "timing and discount").
"""
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Config:
    # ----- plant & timing ---------------------------------------------------------------------
    # model/make_v2_model.py builds both from walk_mit/model/dash01.xml: rigid ankle folded into
    # the shin (no joint, -0.209 kg of spring hardware per shin), loop closure stiffened to the
    # ankle-lock values with a DIRECT-stiffness series spring (§07, negative solref), <motor>
    # actuators (the PD runs in drive.py so kp(phi)/kd(phi) and the torque-speed clamp are ours),
    # armature fitted to the measured 6.3 Hz corner. "planar" removes base y/roll/yaw joints
    # (S1 = m3), "free" keeps all six (S2). Same obs/action widths on both — the frozen interface.
    model_path: str = "model/dash01_v2_free.xml"
    control_decimation: int = 10            # 1 kHz physics / 10 = 100 Hz control (§07)
    keyframe: str = "stand"
    # MJX solver cap (0 = keep the XML's classic-MuJoCo values). Under jax.vmap the Newton loop runs to
    # the WORST env's iteration count for the whole batch: with the XML's 100 x 50 line-search cap one
    # env in a garbage state (falling, penetrating) costs every env ~5000 sequential solver kernels per
    # substep. Healthy states converge (tolerance 1e-8) in a few iterations either way.
    # 16 x 8 replays the CPU golden fixture identically to the XML (100 x 50) and holds a stance to
    # 3e-3 rad over 3 s; 1-4 iterations break the plant (tools/replay_golden.py, profile_step.py).
    mjx_iterations: int = 16
    mjx_ls_iterations: int = 8
    episode_s: float = 60.0                 # ep cap 6000 ticks
    # ----- who emits the gait spec (§09) -------------------------------------------------------
    # "policy": the single-network variant (§02-§04): action 50 = 44 latched + 6 residual.
    # "library": the stabilizer variant (§09 Stage 3): action 6 residual (+3 latched catch-event
    #            dims), spec drawn per episode from a library box + Raibert prior at each commit.
    spec_source: str = "policy"
    library_path: str = "gait_lib/library.json"   # library variant: entries (theta, x*, v*)
    library_box_amp: float = 0.10           # ±fraction on the series amplitudes per episode
    library_box_f_hz: float = 0.3           # ±Hz on the latched frequency
    library_box_knob: float = 0.10          # ±on Δ/Δmax, s, o/o_max
    library_catch_dims: bool = True         # +3 latched dims [Δf/f, amp scale] ±20% (§09 Stage 3)
    library_catch_scale: float = 0.20
    # Raibert prior (Stage 2): o_cam <- o_theta - k_p (v - v_ref) - k_i I; o_hip <- o_theta,hip
    # - k_y v_y - k_r roll_lp. Gains are priors here; §09 says set k_p from the return map.
    raibert_kp: float = 0.05                # rad per m/s of forward-speed error
    raibert_ki: float = 0.02                # rad per m of integrated speed error
    raibert_imax: float = 2.0
    raibert_ky: float = 0.05                # rad per m/s of lateral velocity
    raibert_kr: float = 0.3                 # rad per rad of low-passed roll

    # ----- gait spec (§02, §06) ---------------------------------------------------------------
    n_harmonics: int = 3                    # N=3 series per family: 7 coefficients
    gait_freq_hz: tuple = (0.5, 5.0)        # latched frequency range (§02: 0.5 cannot run)
    cam_amp: float = 0.45                   # rad amplitude budget on S_cam
    thigh_amp: float = 0.45
    roll_amp: float = 0.20                  # S_hip amplitude (bounded by the slow AK60-39); CPU contract 0.20
    delta_max: float = 0.6                  # Δ_max rad. §13 open decision: 0.6 (CPG arm) excludes
    #                                         the bound, pi allows it. Priced either way by w_knob.
    # offset scales: the artifact gives no number; these are the CPU reference's (V2_CONTRACT.md)
    o_cam_max: float = 0.06                 # rad, fore-aft split
    o_thigh_max: float = 0.06               # rad, lift (with o_cam)
    o_hip_max: float = 0.15                 # rad, lean — the old steer_width_scale under its name
    # kp(phi)/kd(phi): series -> exp map. kp x2.5 / ÷3, kd soften-only ÷4 (measured MIT frame
    # ranges: Kp 0-500 with plant 200, Kd 0-5 with plant 5 AT the ceiling).
    imp_kp_up: float = 2.5
    imp_kp_dn: float = 3.0
    imp_kd_dn: float = 4.0
    imp_kd_up: float = 1.0
    # roll reflex (learned gains, latched; feedback at 100 Hz; enters hip_roll (+,+) — §06)
    reflex_kp_scale: float = 0.5
    reflex_kd_scale: float = 0.1
    reflex_bias_scale: float = 0.2
    # pitch reflex (fixed PD, symmetric (+,-) on the thighs; unchanged from walk_mit)
    # retuned by the CPU arm for the no-EMA drive (walk_mit had 2.0 / 0.2 / raw rate)
    pitch_kp: float = 1.0
    pitch_kd: float = 0.1
    pitch_clip: float = 0.25
    pitch_bias: float = 0.0
    pitch_reflex_rate_lp: float = 0.9       # EMA alpha on the reflex's pitch rate (0 = raw)
    residual_scale: float = 0.20            # rad per residual unit (§02: ±0.20, position only)
    action_scale: float = 0.5               # normalization of motor_cmd for the action_rate term
    # ----- the clock (§05) ---------------------------------------------------------------------
    resync_enable: bool = True
    resync_kappa: float = 0.5               # nominal κ
    resync_kappa_range: tuple = (0.3, 0.7)  # per-episode DR on κ
    resync_window_cycle: float = 0.15       # W: ±0.15 cycle around the expected touchdown phase
    resync_ema_cycles: float = 5.0          # N_ema on the per-foot touchdown-phase estimate
    resync_warmup_cycles: int = 3           # κ held at 0 for the first 3 cycles of an episode

    # ----- drive (§07) -------------------------------------------------------------------------
    drive_kp: tuple = (120.0, 200.0, 200.0, 120.0, 200.0, 200.0)   # base gains, actuator order
    drive_kd: tuple = (4.0, 5.0, 5.0, 4.0, 5.0, 5.0)
    drive_delay_ms: float = 12.0            # nominal actuation delay (measured 11-13 ms)
    drive_delay_range_ms: tuple = (6.0, 18.0)   # per-episode draw (in the privileged tail)
    motor_vel_limit: tuple = (10.30, 22.01, 22.01, 10.30, 22.01, 22.01)   # no-load, rad/s
    motor_accel_limit: float = 0.0          # rad/s^2 cap on the commanded target (0 = off)
    motor_kt_joint: tuple = (4.655, 2.176, 2.176, 4.655, 2.176, 2.176)   # N*m/A output side
    motor_r_ohm: tuple = (0.665, 0.229, 0.229, 0.665, 0.229, 0.229)      # datasheet + 0.065 pack
    motor_bus_volts: float = 48.0
    ctrl_jitter_ms_final: float = 4.0       # ±substeps per control step (sim2real timing)
    ctrl_drop_prob_final: float = 0.05      # hold-last-action probability
    jitter_curriculum_gate_ep_len: float = 800.0    # halved from 1600 @200 Hz
    jitter_curriculum_steps: int = 40_000_000

    # ----- thermal budget (§07) ----------------------------------------------------------------
    thermal_enable: bool = True
    thermal_tau_s: float = 45.0             # single-node winding constant (5 s to limit at peak)
    thermal_tau_cont: tuple = (23.0, 55.0, 55.0, 23.0, 55.0, 55.0)   # N*m continuous, act order
    thermal_penalty_frac: float = 0.85      # quadratic penalty above ΔT/ΔT_max = 0.85
    w_thermal: float = 20.0
    thermal_hot_start_max: float = 0.7      # initial ΔT ~ U[0, 0.7 ΔT_max]
    thermal_scale_range: tuple = (0.8, 1.2) # ΔT_max scale DR (unknown Kt roll-off)

    # ----- observation (§03) -------------------------------------------------------------------
    history_len: int = 10
    history_stride: int = 2                 # 10 frames x 20 ms = the 190 ms window at 100 Hz
    obs_scales: dict = field(default_factory=lambda: dict(
        motor_pos=1.0, motor_vel=0.1, motor_torque=0.01, gravity=1.0, ang_vel=0.25, base_vel=1.0))
    lp_yaw_tau_s: float = 0.7               # EMA on gyro z; the same signal the reward bills
    task_brake_m: float = 8.0               # task[1] = clip(d_to_go / 8, 0, 1)

    # ----- domain randomization (§07, §08) -----------------------------------------------------
    dr_enable: bool = True
    dr_curriculum_steps: int = 60_000_000   # gated ramp on the width of every range below
    dr_mass_global: float = 0.12
    dr_mass_body: float = 0.15
    dr_inertia: float = 0.25
    dr_com_offset: float = 0.03             # m (CAD inertials are placeholders — §07)
    dr_friction_range: tuple = (0.4, 1.3)
    dr_kp: float = 0.20
    dr_kv: float = 0.25
    dr_torque: float = 0.12
    dr_joint_damping: float = 0.30
    dr_gravity_tilt_deg: float = 5.0        # roll and pitch (§08: 3° gate, trained at 5°)
    dr_link_spring: float = 0.50            # ±50% on the 30 kN/m series spring
    dr_loop_site_m: float = 0.0015          # ±1.5 mm on the loop-closure sites (the yaw bias)
    dr_joint_zero_deg: float = 2.0          # homing error, obs AND command side
    dr_imu_rot_deg: float = 2.0             # IMU mount rotation (gravity + gyro together)
    dr_imu_dropout_prob: float = 0.001
    dr_imu_dropout_s: float = 0.25
    # ----- sensor noise: measured figures (IMU noise figure, 2026-09-01) ----------------------
    obs_noise_enable: bool = True
    noise_encoder: float = 0.003            # rad
    noise_motor_vel: float = 0.15           # rad/s
    noise_torque: float = 1.5               # N*m
    noise_torque_gain: float = 0.08
    noise_grav: float = 0.0013              # 190 ug/√Hz x √(50 Hz) in g
    noise_grav_bias: float = 0.01
    noise_gyro: float = 0.0012              # 0.0097 dps/√Hz x √(50 Hz) -> rad/s
    noise_gyro_bias: float = 0.00005        # bias floor 0.0024 dps
    noise_gyro_walk: float = 0.00002
    noise_accel_leak: float = 0.15          # attitude-filter accelerometer leak, U[0, this]
    # ----- disturbances (§08) ----------------------------------------------------------------
    push_interval_s: float = 4.0
    push_dv_range: tuple = (0.3, 0.6)       # |Δv| m/s, random direction in the free plane
    wind_force_max: float = 30.0            # N, constant per episode ~ U[-max, max], x and y
    wind_gust_n: float = 30.0               # N step
    wind_gust_s: float = 1.0
    wind_gust_interval_s: tuple = (5.0, 10.0)
    trip_prob: float = 0.0008               # per tick (the artifact's number; ~12 s at 100 Hz)
    trip_force_range: tuple = (30.0, 90.0)
    trip_duration_s: float = 0.05
    adversity_curriculum: bool = True       # pushes/trips/wind ride the dr_scale ramp

    # ----- objective: the 100 m dash ----------------------------------------------------------
    objective: str = "sprint"               # "sprint" | "speed" (endless, debug)
    v_ceiling: float = 3.0
    w_fwd_speed: float = 2.0
    sprint_world_speed: bool = False        # RUN 8 recipe: body-frame income + LP yaw + lane
    w_yaw_rate: float = 3.0                 # on the LOW-PASSED yaw rate (§10)
    w_lane: float = 2.0                     # -w (|y| - 0.25)^2 beyond 0.25 m (§10)
    lane_free_m: float = 0.25
    speed_upright_gate: bool = True
    speed_upright_c0: float = 0.5
    speed_upright_k: float = 1.0
    w_alive: float = 0.0
    sprint_dist_m: float = 100.0
    sprint_dist_start_m: float = 25.0
    sprint_curriculum_steps: int = 60_000_000
    sprint_brake_m: float = 5.0
    w_time: float = 0.5
    w_stop_vel: float = 0.4
    stop_sigma: float = 0.3
    w_overrun: float = 1.0
    stop_speed_eps: float = 0.15
    stop_hold_s: float = 1.0
    finish_bonus: float = 100.0
    fall_penalty: float = 100.0
    penalty_term_cap: float = 2.0
    step_reward_floor: float = 1.0

    # ----- reward: gait shaping (RUN 8 set, kept — §10) ---------------------------------------
    w_foot_slip: float = 8.0
    slip_deadband: float = 0.05
    w_stance_time: float = 0.5
    stance_cap_s: float = 0.7
    stance_cap_slow_s: float = 1.0
    stance_slow_speed: float = 0.4
    w_clearance: float = 0.4
    clearance_dead_m: float = 0.02
    clearance_scale_m: float = 0.03
    swing_fresh_s: float = 0.45
    gait_cmd_gate: float = 0.25
    w_air_time: float = 2.0
    foot_air_time_min: float = 0.25
    air_credit_cap_s: float = 0.45
    grounded_h: float = 0.005
    w_contact_switch: float = 0.20
    w_duty_sym: float = 8.0
    duty_floor: float = 0.30
    duty_sym_tau_s: float = 1.0
    w_swing_floor: float = 0.0
    swing_floor_frac: float = 0.15
    swing_floor_tau_s: float = 0.5
    w_phase_contact: float = 1.0            # Siekmann schedule on the latched clock
    stance_ratio_start: float = 0.65
    stance_ratio_final: float = 0.42
    gait_curriculum_steps: int = 120_000_000
    curriculum_gate_ep_len: float = 600.0   # the DR ramp's competence gate (contract: DR gated at 600)
    gait_curriculum_gate_ep_len: float = 0.0    # stance-ratio ramp: 0 = clock from step 0 (contract)
    efficiency_gate_ep_len: float = 0.0         # efficiency ramp: 0 = clock from step 0 (contract)
    curriculum_retreat_frac: float = 0.5
    workspace_kill: bool = True
    workspace_dx_max: float = 0.34
    workspace_dz_max: float = 0.14
    workspace_dz_min: float = -0.18
    workspace_grace_s: float = 0.10
    # ----- reward: efficiency + smoothness -----------------------------------------------------
    w_torque: float = 1.0e-4
    w_motor_vel: float = 1.0e-4
    w_energy: float = 2.0e-4
    efficiency_ramp_steps: int = 120_000_000
    efficiency_target: float = 1.0
    w_action_rate: float = 0.1
    w_spec_cycle: float = 0.5               # ||S_{k+1} - S_k||^2 billed ONCE at commit (§02)
    w_knob: float = 0.02                    # standing price on (Δ/Δmax, s, o/omax) per tick (contract)
    w_residual: float = 0.10                # was 0.02 (§10: every baseline lived at the bound)
    w_residual_rate: float = 0.02
    w_upright: float = 5.0
    w_height: float = 2.5
    height_floor_m: float = 0.81            # one-sided: quadratic BELOW this, 0 above (§10)
    w_vz: float = 0.5
    w_lat_vel: float = 1.0
    w_angvel_xy: float = 0.05
    w_angmom: float = 0.2
    w_no_cross: float = 50.0
    stance_min_sep: float = 0.25
    w_hip_roll: float = 3.0
    # library variant only: tracking of the reference (§09 Stage 3, "small")
    w_track_ref: float = 0.5
    # ----- termination -------------------------------------------------------------------------
    term_height: float = 0.45
    term_gravity_z: float = -0.5
    reset_joint_noise: float = 0.03
    # ----- pitch assist (RUN 8 training wheel; clock-faded, sim-only) --------------------------
    pitch_assist_kp: float = 100.0
    pitch_assist_kd: float = 10.0
    pitch_assist_ramp_steps: int = 30_000_000
    pitch_assist_gate_ep_len: float = 0.0   # 0 = clock fade from step 0 (v2); >0 = full help until
                                            # ep_len > gate for 5 rollouts, then a monotonic fade (v2b)
    w_assist_penalty: float = 0.0

    # ----- PPO (§04) ---------------------------------------------------------------------------
    n_envs: int = 1024
    n_steps: int = 18                       # per env per rollout: 1024 x 18 = 18 432 samples = the contract's
                                            # 64 x 288 (V2_CONTRACT §PPO): same rollout size, same minibatch,
                                            # hence the same gradient updates per sample as the CPU arm
    total_steps: int = 300_000_000
    batch_size: int = 4096
    n_epochs: int = 4
    gamma: float = 0.995                    # 2 s horizon at 100 Hz (was 0.9975 @200 Hz)
    gae_lambda: float = 0.95
    learning_rate: float = 3.0e-4
    lr_final: float = 1.0e-4
    clip_range: float = 0.2
    target_kl: float = 0.03
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    grad_guard: bool = False                # optax.apply_if_finite around the optimizer (new runs only)
    lr_kl_adaptive: bool = False            # rl_games adaptive lr on the mean KL (x1.5 / /1.5 around target_kl)
    lr_kl_min: float = 1.0e-5
    lr_kl_max: float = 1.0e-3
    ent_coef: float = 0.01
    ent_final: float = 0.0
    ent_anneal_steps: int = 40_000_000
    ent_gate_swing_frac: float = 0.13
    ent_gate_ep_len: float = 0.0            # v2b: the competence gate also needs ep_len > this
                                            # (a falling robot has both feet airborne)
    ent_anneal_deadline_steps: int = 12_500_000
    ent_schedule_autoscale: bool = True
    max_log_std: float = 0.0
    std_anneal_target: float = 0.25
    policy_hidden: List[int] = field(default_factory=lambda: [256, 256])
    est_hidden: List[int] = field(default_factory=lambda: [128, 64])
    est_epochs: int = 2
    est_batch: int = 4096
    est_lr: float = 1.0e-3
    # symmetry loss (§06): w_sym ||k(M_o s) + k(s)||^2 on the actor mean's five knobs
    w_sym: float = 0.5
    w_sym_res: float = 0.5                  # + ||r(M_o s) - M_r r(s)||^2 on the residual (contract)
    w_knob_loss: float = 0.0
    w_bound: float = 0.0                    # v2c: action-mean bounds loss w*mean(relu(|mu|-bound_soft)^2)
    bound_soft: float = 1.0
    mask_spec_logprob: bool = True          # §04: spec dims scored only on commit ticks
    seed: int = 0
    warmstart_reset_log_std: bool = True
    warmstart_obs_count_cap: float = 100_000.0
    warmstart_var_floor: float = 1.0e-2
    eval_every_rollouts: int = 400          # greedy 16-env dash eval (the determinism-gap rule): every
                                            # 7.4 M steps; a full 60 s greedy episode costs ~200 s on a V100
    checkpoint_every_steps: int = 5_000_000


# ----- presets ------------------------------------------------------------------------------
def _v2(**kw) -> Config:
    return Config(**kw)


# Readout-1 deltas (V2_CONTRACT.md 2026-09-10): the artifact-literal v2 runs on BOTH arms fell into a
# clock-rail exploit (CPU: 5.00 Hz on 100 % of commits, bang-bang spec; GPU: 0.5 Hz on 94 %) and
# regressed. v2b keeps the plant, obs, action layout and reward set and changes only these; the GPU
# port must mirror the first two rows, the gates are training-side.
_V2B = dict(
    gait_freq_hz=(1.5, 4.0),                 # neutral stays 2.75 Hz; no 5 Hz max-commit rail, no 0.5 Hz freeze
    residual_scale=0.10, w_residual=0.20,    # half the per-tick authority, twice the bill
    pitch_assist_gate_ep_len=600.0,          # fade the wheel once the policy runs on it, then 30 M monotonic
    ent_gate_ep_len=600.0, ent_anneal_deadline_steps=40_000_000,   # precision phase on competence, not falls
    curriculum_gate_ep_len=1200.0, jitter_curriculum_gate_ep_len=1200.0,   # harden a runner, not a stander
)

# Readout-2 deltas (V2_CONTRACT.md): v2b finished 3/3 greedy dashes at 23 M with the wheel at 0.8 and then
# collapsed; 68-79 % of the spec means and 53-60 % of the residual means lay OUTSIDE [-1, 1] (clip(mu) of
# drifted means = bang-bang spec, parked clock, saturated residual). v2c = v2b + the bounds loss, a
# retreating curriculum (0.7), a 0.7 std cap before the anneal, and the anti-crutch assist torque bill.
_V2C = dict(_V2B, w_bound=1.0, curriculum_retreat_frac=0.7, max_log_std=-0.3567,   # ln 0.7
            w_assist_penalty=0.005)

# GPU throughput sizing (2026-09-10, goal: a policy in < 3 h on one arm). The batched step is latency-bound
# (solver loops), so more envs per rollout cost little: 2048 x 9 keeps the contract's 18 432-sample rollout
# (same updates per sample) at ~1.8x the samples/s of 1024 x 18, with a 9-tick (90 ms) GAE horizon; the
# 8 x 8 solver cap adds ~29 % in the flailing regime and replays the golden fixture identically.
_FAST = dict(n_envs=2048, n_steps=9, mjx_iterations=8, mjx_ls_iterations=8)

# 4-GPU sizing (8192 envs x 9 = 73 728-sample rollouts): the recipe's step-based schedules are really
# iteration counts (the CPU arm's 40 M entropy deadline = 2170 policy updates at 18 432/rollout); at 4x
# the rollout they arrive after a quarter of the updates and froze an incompetent policy at 40 M
# (v2c_fast_dp4x_s0). Scale them x2 (a compromise between iterations and wall-clock), lr x sqrt(4)
# for the 4x batch, 450 M steps (~3 h at 42k steps/s), and guard the update against non-finite grads.
_DP4 = dict(_FAST, n_envs=8192, learning_rate=6.0e-4, lr_final=2.0e-4, total_steps=450_000_000,
            ent_anneal_deadline_steps=80_000_000, ent_anneal_steps=80_000_000,
            pitch_assist_ramp_steps=60_000_000, sprint_curriculum_steps=120_000_000,
            gait_curriculum_steps=240_000_000, efficiency_ramp_steps=240_000_000,
            dr_curriculum_steps=120_000_000, jitter_curriculum_steps=80_000_000, grad_guard=True)

PRESETS = {
    "default": Config,
    # S1 "planar" (= m3): x, z, pitch free; y, roll, yaw absent from the model. Iteration sandbox.
    "v2_s1_planar": lambda: _v2(model_path="model/dash01_v2_planar.xml"),
    # S2 "free": the real run. Warm from S1 (identical obs/action widths).
    "v2_s2_free": lambda: _v2(model_path="model/dash01_v2_free.xml"),
    # readout-1 fixes on top (see _V2B); the v2_* presets stay the artifact-literal reference
    "v2b_s1_planar": lambda: _v2(model_path="model/dash01_v2_planar.xml", **_V2B),
    "v2b_s2_free": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V2B),
    # plant experiment (2026-09-10): the CPU arm's literal 30 kN/m rod spring yields ~nothing under load, so
    # its leg is effectively rigid; this side has the measured 5 mm/BW compliance. 10x stiffer (0.5 mm/BW)
    # isolates whether the per-step learning lag vs the CPU arm is the plant.
    "v2b_s1_planar_stiff": lambda: _v2(model_path="model/dash01_v2_planar_stiff.xml", **_V2B),
    # the wheel question (2026-09-10): v2 and v2b collapse on BOTH arms exactly when the pitch-assist fade
    # runs out, and the greedy eval without the wheel falls within a second at every checkpoint. No
    # assist from step 0: does the latched design learn balance at all?
    "v2b_s1_planar_noassist": lambda: _v2(model_path="model/dash01_v2_planar.xml", **_V2B, pitch_assist_kp=0.0),
    # readout-2 fixes on top (see _V2C): the CPU arm finishes 3/3 greedy dashes at 30 M with these
    "v2c_s1_planar": lambda: _v2(model_path="model/dash01_v2_planar.xml", **_V2C),
    "v2c_s2_free": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V2C),
    "v2c_s1_planar_fast": lambda: _v2(model_path="model/dash01_v2_planar.xml", **_V2C, **_FAST),
    "v2c_s2_free_fast": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V2C, **_FAST),
    "v2c_s1_planar_dp4": lambda: _v2(model_path="model/dash01_v2_planar.xml", **{**_V2C, **_DP4}),
    # the end-of-fade cliff (v2c_s1_planar_s1 at 47 M: KL early-stop storm, means drifting out of the box):
    # KL-adaptive lr instead of the early stop throttling learning
    "v2c_s1_planar_fast_kl": lambda: _v2(model_path="model/dash01_v2_planar.xml", **_V2C, **_FAST,
                                         lr_kl_adaptive=True, grad_guard=True),
    "v2c_s1_planar_dp4_kl": lambda: _v2(model_path="model/dash01_v2_planar.xml", **{**_V2C, **_DP4},
                                        lr_kl_adaptive=True),
    "v2c_s2_free_dp4": lambda: _v2(model_path="model/dash01_v2_free.xml", **{**_V2C, **_DP4}),
    # Δ_max = pi: the bound becomes reachable (§13 open decision, priced not forbidden)
    "v2_s2_free_wide": lambda: _v2(model_path="model/dash01_v2_free.xml", delta_max=3.14159265),
    # honesty-off debug arms: nominal plant, no noise, no disturbances (fast signal on latch/reward)
    "v2_s1_planar_easy": lambda: _v2(model_path="model/dash01_v2_planar.xml", dr_enable=False,
                                     obs_noise_enable=False, push_interval_s=0.0,
                                     wind_force_max=0.0, wind_gust_n=0.0, trip_prob=0.0,
                                     ctrl_jitter_ms_final=0.0, ctrl_drop_prob_final=0.0,
                                     thermal_hot_start_max=0.0),
    "v2_s2_free_easy": lambda: _v2(model_path="model/dash01_v2_free.xml", dr_enable=False,
                                   obs_noise_enable=False, push_interval_s=0.0,
                                   wind_force_max=0.0, wind_gust_n=0.0, trip_prob=0.0,
                                   ctrl_jitter_ms_final=0.0, ctrl_drop_prob_final=0.0,
                                   thermal_hot_start_max=0.0),
    # library variant (§09 Stage 3): residual-only stabilizer around library entries
    "v2_lib_s2_free": lambda: _v2(model_path="model/dash01_v2_free.xml", spec_source="library"),
    "v2_lib_s1_planar": lambda: _v2(model_path="model/dash01_v2_planar.xml",
                                    spec_source="library"),
    # endless-speed debug objective
    "v2_speed_planar": lambda: _v2(model_path="model/dash01_v2_planar.xml", objective="speed",
                                   w_alive=0.5, episode_s=20.0, sprint_curriculum_steps=0),
    # smoke: tiny everything
    "v2_smoke": lambda: _v2(model_path="model/dash01_v2_free.xml", n_envs=8, n_steps=16,
                            batch_size=64, episode_s=2.0, total_steps=2000,
                            checkpoint_every_steps=1000, eval_every_rollouts=2),
}


def get_config(name: str = "default") -> Config:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
    return PRESETS[name]()


def _tuple_fields():
    from dataclasses import fields
    return {f.name for f in fields(Config) if isinstance(getattr(Config(), f.name), tuple)}


def config_to_dict(cfg: Config) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


def config_from_dict(d: dict) -> Config:
    tf = _tuple_fields()
    kw = {}
    for k, v in d.items():
        if k in tf and isinstance(v, list):
            v = tuple(v)
        kw[k] = v
    return Config(**kw)
