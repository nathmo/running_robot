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
    # FOOT CONTACT IS NOT AVAILABLE ON DASH-01. Off by default from v3: when False the whole
    # resync block in env._step_one is compiled out, so no contact quantity reaches the clock --
    # and therefore neither the actor's phase nor its commit flag. True reproduces v2 runs.
    resync_enable: bool = False
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
    objective: str = "sprint"               # "sprint" | "speed" (endless, debug) | "joystick"
    v_ceiling: float = 3.0
    # ----- the joystick (objective="joystick") -------------------------------------------------
    # task[0] = v_cmd / v_max, so the operator's stick is a FRACTION of top speed: 0.5 asks for half.
    # Nothing in this objective reads sprint_d, which is what removes the odometry the v2 task[1]
    # carried. The command is redrawn mid-episode (cmd_interval_s) so the policy is trained WITH a
    # moving command instead of having one retrofitted -- the v2 failure mode, where a command pinned
    # at 1 for a whole run and then flipped acted as a step disturbance, not as an input.
    v_max: float = 3.6                      # what a full stick asks for; measured top speed is ~3.5
    cmd_range: tuple = (0.0, 1.0)           # FINAL fraction-of-v_max band the curriculum widens to
    cmd_range_start: tuple = (0.8, 1.0)     # where it starts: what the warm-start runner already does
    cmd_curriculum_steps: int = 40_000_000
    cmd_gate_ep_len: float = 600.0
    cmd_interval_s: float = 4.0             # mean seconds between redraws (x U(0.6, 1.4))
    cmd_zero_frac: float = 0.25             # share of draws that are exactly 0 = step in place
    w_track: float = 3.0                    # income for tracking the command
    track_sigma: float = 0.6                # Laplace width, m/s (Gaussian is flat where we live)
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
    # ----- stop curriculum (2026-09-10 evening): every runner on both arms crossed the line at full
    # speed and fell (the post-line phase is never experienced; the finish bonus is unreachable).
    # Red light / green light: in a fraction of episodes the run flag drops at random times before
    # the line (obs task[0] -> 0 with task[1] still > 0), the speed income stops and the stop term
    # pays for tracking a target speed that ramps from the speed at the switch to 0 over stop_decel_s
    # (then the plain stop term), then the light turns green again. The final line is one more red.
    # All off by default (contract behaviour); the *_stoplight presets turn it on.
    # ----- gait-frequency floor curriculum (2026-09-10 night): the low rail (gait_freq_hz[0]) is a
    # free-plant attractor -- cold S2 seeds park the clock there and never run. Start the lower rail
    # at gait_freq_lo_start and ramp it down to gait_freq_hz[0] over gait_freq_floor_steps, so a slow
    # gait is simply not selectable while the running gait forms. 0 steps = off (every other preset).
    gait_freq_lo_start: float = 3.0
    gait_freq_floor_steps: int = 0
    stoplight_prob_final: float = 0.0       # fraction of episodes with red/green cycles
    stoplight_curriculum_steps: int = 0     # ramp 0 -> final, competence-gated like DR
    stoplight_gate_ep_len: float = 600.0
    stoplight_green_s: tuple = (3.0, 8.0)   # green phase duration, uniform
    stoplight_red_s: tuple = (2.0, 4.0)     # red phase duration, uniform
    stop_decel_s: float = 0.0               # > 0: target speed ramps to 0 over this after a red / the line
    # The binary run flag is a STEP DISTURBANCE: measured on v2c_s2_stopwarm_s6 at 59 M, the greedy
    # policy falls 0.7-0.8 s after every red light in 16/16 episodes and never slows (slow 0%). With
    # this on, task[0] carries the TARGET SPEED (normalised) instead: 1 while running, then the same
    # ramp the stop reward tracks, so the command is continuous and is a speed command a deployment
    # panel can drive directly. Off = the contract's binary flag.
    stop_cmd_continuous: bool = False
    # The Gaussian tracking reward VANISHES far from the target: asked to slow from 3.3 m/s toward
    # 1.5, the error is 1.8 m/s and exp(-(1.8/0.8)^2) = 0.006, so the policy sits in a flat region
    # with no gradient telling it which way to go -- and what it actually does is lower the cadence
    # (4.0 -> 2.4 Hz) while ACCELERATING to 3.3 m/s, i.e. overstriding into a fall. The Laplace form
    # exp(-|e|/sigma) gives 0.105 at the same error, 17x the signal, with gradient everywhere.
    stop_track_laplace: bool = False
    # AMBER. Five measured fixes (weight, continuous command, 8 s and even 20 s ramps, live gait
    # shaping, Laplace tracking) all failed the same way: asked to slow, the robot drops the cadence
    # a little and ACCELERATES to ~3.1 m/s, overstrides, falls in ~1.3 s, 0% of red spent slow. The
    # deficit is not braking, it is that THIS ROBOT ONLY KNOWS ONE SPEED -- every policy in the
    # project has been trained at the sprint and nowhere else, and lowering the gait clock on this
    # plant makes it FASTER (longer stance, bigger push) rather than slower. So a fraction of the
    # light phases now ramp down to a SLOW RUN instead of to a standstill: the ramp target becomes a
    # speed drawn from amber_speed_band, and the policy has to hold a gait there. Stopping is then
    # the bottom of a range it knows, not a regime it has never visited. 0 = off.
    # Braking needs a CAPTURE STEP: the stance foot planted AHEAD of the CoM so the ground reaction
    # pushes backwards. Instrumented, this robot does the opposite -- it drops the gait clock, which
    # on this plant lengthens the stance and the push, so it pitches forward and ACCELERATES into the
    # fall (3.0-3.4 m/s at the fall against a 2.3 m/s command). Cadence is the only lever it uses and
    # it is the wrong one. This rewards feet ahead of the CoM, but only while the robot is running
    # FASTER than commanded, so it never fights the running gait. 0 = off.
    # BRAKE PRIOR (2026-09-11, measured by tools/brake_search.py). A CEM search over gait specs found
    # that a stop IS expressible: from 3.3 m/s hundreds of schedules reach a standstill upright. The
    # braking direction it found is the OPPOSITE of what the policy learned: raise the cadence, hold
    # the stride, shift the feet FORWARD (the capture step) and lean back. The policy instead drops
    # the clock, overstrides and accelerates into a fall. So bias the latched spec that way while the
    # robot is above its commanded speed, exactly like the pitch reflex already in the control law:
    # the policy keeps full authority to modulate it, but braking is now reachable from where it
    # starts instead of being a needle PPO has to find in the dark. 0 = off.
    brake_prior: float = 0.0
    w_brake_foot: float = 0.0
    brake_foot_max_m: float = 0.25          # saturation of the foot-ahead offset
    amber_frac: float = 0.0
    amber_speed_band: tuple = (0.8, 1.6)
    decel_sigma: float = 0.6                # width (m/s) of the tracking reward while the target is > 0
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
    # S2 roll wheel (2026-09-10): the S1 runner falls sideways in 0.7-1.4 s on the free plant, greedy,
    # with or without the pitch wheel, so a warm-started S2 starts from nothing; a roll spring-damper on
    # the base, driven by the SAME fade scalar as the pitch wheel, carries the S1 competence over.
    # 0 = off (every existing preset); ignored on the planar plant (no base_roll joint)
    roll_assist_kp: float = 0.0
    roll_assist_kd: float = 0.0
    # yaw wheel: the S1 runner on the free plant with pitch AND roll held still falls in 1.4-1.8 s -- the
    # frames show the base yawing ~90 deg in the first second (the gait's yaw impulse, absorbed by the planar
    # tree, spins it); a heading spring-damper, same fade scalar
    yaw_assist_kp: float = 0.0
    yaw_assist_kd: float = 0.0
    # ----- bring-up hold (probe only, off in every preset) --------------------------------------
    # Deployment question: the operator holds the base on its stand, feet on the floor, starts the
    # policy, and lets go a few seconds later. With this on, EnvParams.hold_s clamps the six base
    # DOFs to (key x/y/yaw, hold_z, hold_roll, hold_pitch) at every 1 kHz substep -- an infinitely
    # stiff hand -- and suppresses fall termination while held; at t = hold_s the base is released
    # with zero velocity. A kinematic clamp, not a spring: the 1000 N m/rad wheel probe went NaN.
    hold_enable: bool = False
    # ----- bring-up randomisation (training, needs hold_enable) ---------------------------------
    # The operator does not hand this robot a settled keyframe. Two real cases, sampled per episode:
    #   drop  -- let go with the feet off the ground, from bringup_drop_m above TOUCHING height
    #            (touching is tilt-dependent: model/touch_height.npz, built by tools/make_touch_table.py)
    #   held  -- feet down but the body not square, held a moment, then released at a pitch
    # The measured envelope today is upright to 5 deg BACK with both feet flat; +5 deg forward gave
    # 16/16 floor violations and +-10 deg is outside it entirely. So the bands RAMP on competence --
    # opening them at step 0 would start most episodes already lost.
    bringup_enable: bool = False
    bringup_drop_frac: float = 0.25         # share of episodes dropped
    bringup_held_frac: float = 0.35         # share held-misaligned then released (rest = nominal)
    bringup_drop_m: tuple = (0.05, 0.10)    # FINAL drop band above touching height
    bringup_drop_m_start: tuple = (0.01, 0.02)
    bringup_pitch_deg: float = 20.0         # FINAL +- pitch at release
    bringup_pitch_deg_start: float = 5.0
    bringup_roll_deg: float = 8.0           # FINAL +- roll at release
    bringup_roll_deg_start: float = 2.0
    bringup_hold_s: tuple = (0.3, 2.5)      # how long the hand stays on
    bringup_grace_s: float = 0.35           # no fall termination while dropping / just released
    bringup_curriculum_steps: int = 40_000_000
    bringup_gate_ep_len: float = 600.0

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
    # ---- v3: the joystick, with no privileged input anywhere in the actor path -------------
    # task[0] = commanded speed / v_max (no odometry), the clock free-runs (no foot contact),
    # brake_prior off (it read true v_body), and the command moves throughout training. Zero
    # command means step in place: gait_cmd_gate is bypassed under this objective.
    "v3_joystick_s2": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V2C, **_FAST,
                                  objective="joystick", resync_enable=False, brake_prior=0.0,
                                  hold_enable=True, bringup_enable=True,
                                  w_alive=0.5, episode_s=30.0, sprint_curriculum_steps=0,
                                  total_steps=140_000_000),
    # S2 warm-start experiment: keep the S1 policy's std (contract re-inflates log sigma; the seeds start at ep_len 77)
    "v2c_s2_free_fast_keepstd": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V2C, **_FAST,
                                            warmstart_reset_log_std=False),
    # the stop curriculum (red light / green light + deceleration target), cold S2
    "v2c_s2_free_fast_stoplight": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V2C, **_FAST,
                                              stoplight_prob_final=0.5, stoplight_curriculum_steps=20_000_000,
                                              stoplight_gate_ep_len=600.0, stop_decel_s=1.5),
    # stronger stop incentive: at the contract w_stop_vel 0.4 a red phase pays at most 0.4/step against
    # ~5/step of running income, so the measured stop income stayed ~0.005 (the policy ignores the light
    # and simply loses the income). Pay braking on the same order as running, widen the tracking window
    # and give it longer to bleed off speed; fewer lit episodes so the running signal stays strong.
    "v2c_s2_free_fast_stoplight_hard": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V2C, **_FAST,
                                                   stoplight_prob_final=0.35, stoplight_curriculum_steps=20_000_000,
                                                   stoplight_gate_ep_len=600.0, stop_decel_s=2.0,
                                                   w_stop_vel=2.0, decel_sigma=0.8, stop_speed_eps=0.25,
                                                   stop_cmd_continuous=True),
    # GENTLE stop (2026-09-11 01:00). Measured: even with the continuous command and 5x the stop reward,
    # the policy survives only 1.2 s of red (up from 0.8) and still falls, 16/16, 0% of red spent slow.
    # The demand was the problem: stop_decel_s 2.0 asks for 1.45 m/s^2 from 2.9 m/s, and this morphology
    # has never learned a capture step (the foot lands ~8 cm BEHIND the CoM -- walk_mit m3 finding).
    # The actual requirement is "stop within ~20 m of the line" = 0.21 m/s^2 over ~14 s, SEVEN times
    # gentler. So: ramp over 8 s (0.36 m/s^2, ~11.6 m of stopping distance, inside the 20 m budget), red
    # phases long enough to finish the stop and hold it, and the overrun bill moved out to 20 m.
    "v2c_s2_free_fast_stopgentle": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V2C, **_FAST,
                                               stoplight_prob_final=0.35, stoplight_curriculum_steps=15_000_000,
                                               stoplight_gate_ep_len=600.0, stop_decel_s=8.0,
                                               stoplight_red_s=(10.0, 14.0), stoplight_green_s=(6.0, 12.0),
                                               w_stop_vel=2.0, decel_sigma=0.8, stop_speed_eps=0.25,
                                               stop_cmd_continuous=True, sprint_brake_m=20.0,
                                               stop_track_laplace=True,
                                               amber_frac=0.6, w_brake_foot=1.5, brake_prior=1.0),
    # SLOW RUNNER. Eight measured interventions have failed to teach a stop FROM ~2.8 m/s. The
    # requirement is "run 100 m and stop within ~20 m", not "run at 3 m/s": at 1.8 m/s the 20 m budget
    # needs only 0.08 m/s^2, a quarter of what it needs at 2.9. Income saturates at v_ceiling, so above it
    # extra speed earns nothing and still costs energy -- the policy should settle near the cap. A slower
    # runner that can stop is worth more than a fast one that cannot.
    "v2c_s2_free_fast_slowstop": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V2C, **_FAST,
                                             v_ceiling=1.8, stoplight_prob_final=0.35,
                                             stoplight_curriculum_steps=15_000_000, stoplight_gate_ep_len=600.0,
                                             stop_decel_s=8.0, stoplight_red_s=(10.0, 14.0),
                                             stoplight_green_s=(6.0, 12.0), w_stop_vel=2.0, decel_sigma=0.8,
                                             stop_speed_eps=0.25, stop_cmd_continuous=True,
                                             sprint_brake_m=20.0, stop_track_laplace=True,
                                             amber_frac=0.6, amber_speed_band=(0.5, 1.1), w_brake_foot=1.5),
    # cold S2, the whole recipe: frequency floor (no slow-gait rail while the gait forms) + the stop
    # curriculum. This is the "no S1 stage" configuration.
    "v2c_s2_free_fast_floorstop": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V2C, **_FAST,
                                              gait_freq_lo_start=3.0, gait_freq_floor_steps=60_000_000,
                                              stoplight_prob_final=0.5, stoplight_curriculum_steps=20_000_000,
                                              stoplight_gate_ep_len=600.0, stop_decel_s=1.5),
    # S2 with the roll wheel (pitch + roll held at the start, both faded once ep_len > 600 for 5 rollouts)
    "v2c_s2_free_fast_rollassist": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V2C, **_FAST,
                                               roll_assist_kp=100.0, roll_assist_kd=10.0),
    # all three base wheels (pitch, roll, yaw) held at the start and faded together
    "v2c_s2_free_fast_basewheels": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V2C, **_FAST,
                                               roll_assist_kp=100.0, roll_assist_kd=10.0,
                                               yaw_assist_kp=100.0, yaw_assist_kd=10.0),
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
