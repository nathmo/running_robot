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
    model_path: str = "model/dash01_free.xml"
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
    # v4: BOTH REFLEXES ARE GONE from the control law, with the three latched dims that carried the
    # roll gains (gait.py). The residual is the only per-tick authority, per joint.
    residual_scale: float = 0.20            # rad per residual unit (§02: ±0.20, position only).
    #                                         v4: may be a 6-tuple in actuator order (gait.residual_scale6)
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
        motor_pos=1.0, motor_vel=0.1, motor_torque=0.01, gravity=1.0, ang_vel=0.25, base_vel=1.0,
        heading=1.0))                       # v3: integrated-gyro heading, radians, clipped +-pi/2
    lp_yaw_tau_s: float = 0.7               # EMA on gyro z; the same signal the reward bills
    # ----- heading (v3): run STRAIGHT -----------------------------------------------------
    # v2 penalised the low-passed yaw RATE and gave the actor the same rate. A rate controller
    # has no set point, so heading was free to random-walk (walk_mit measured 171 m of path for
    # 15-38 m of net progress). v3 adds the INTEGRATED heading on both sides: the actor reads its
    # own dead-reckoned estimate and the reward bills the true angle. w_yaw_rate stays, at a
    # third of its old weight -- it damps gait wobble, which is what it was measured to bill
    # (~95% wobble, ~5% drift); heading is what bills drift.
    w_heading: float = 6.0                  # -w * yaw^2, true base yaw, radians
    heading_cap_rad: float = 1.5708         # the penalty saturates here, as the obs channel does
    # v4: > 0 bills the robot's OWN integrated-gyro heading (what the actor reads) through a signed
    # moving average of this time constant, instead of the true yaw every tick. 0 = the v3 term.
    heading_avg_s: float = 0.0
    # v4: integrate the Euler heading rate (gyro y and z through the measured roll and pitch) rather
    # than body gyro z alone, which drifted 10 deg in 15 s on a 2-deg-leaning runner (env.heading_rate).
    # Changes the obs channel AND what heading_avg_s bills; the bundle carries it to the Pi.
    heading_euler: bool = False
    task_brake_m: float = 8.0               # task[1] = clip(d_to_go / 8, 0, 1)

    # ----- domain randomization (§07, §08) -----------------------------------------------------
    dr_enable: bool = True
    dr_curriculum_steps: int = 60_000_000   # gated ramp on the width of every range below
    # THE DR FLOOR -- where the ramp STARTS, not 0.
    #
    # Measured 2026-09-17 on dash_s0/s1/s2: dr_scale sat at exactly 0.000 for the first 119 M
    # steps (it is last but one in the queue), so the policy converged on a plant that is not
    # merely easy but DETERMINISTIC -- the same masses, the same gains, the same friction, every
    # episode. All three seeds climbed to ep_len 2376 and then collapsed to 45 within 10 M steps
    # of dr_scale first becoming nonzero, at a dose of 0.088: mass +-1.1%, kp +-1.8%, gravity
    # tilt +-0.44 deg, wind +-2.6 N against a 141 N robot. Nothing at that scale should topple a
    # policy surviving 2376 ticks; it did because the policy had spent its whole life on a single
    # point in plant space and had no margin to spend.
    #
    # A floor keeps that from ever being true: the plant varies from the first rollout, so
    # robustness is built with the gait instead of asked of a finished one. The ramp above still
    # runs, from here to 1.0.
    dr_scale_start: float = 0.0
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
    v_min: float = 0.0                      # forward only for now; negative = walk backwards
    cmd_range: tuple = (0.0, 1.0)           # FINAL fraction-of-v_max band the curriculum widens to
    cmd_range_start: tuple = (0.8, 1.0)     # where it starts: what the warm-start runner already does
    cmd_curriculum_steps: int = 40_000_000
    cmd_gate_ep_len: float = 600.0
    cmd_interval_s: float = 4.0             # mean seconds between redraws (x U(0.6, 1.4))
    cmd_zero_frac: float = 0.25             # share of draws that are exactly 0 = step in place
    # RUN/STOP (walk_v4): the command is 0 or v_max, nothing in between, and under RUN the income is
    # linear in forward speed (as fast as you can, capped at v_ceiling) instead of the tracking kernel
    cmd_binary: bool = False
    run_income_linear: bool = False
    w_track: float = 3.0                    # income for tracking the command
    track_sigma: float = 0.6                # Laplace width, m/s (Gaussian is flat where we live)
    track_sigma_start: float = 0.6          # cold start: begin wide so the first m/s pays, then
    track_sigma_steps: int = 0              # tighten to track_sigma over this many steps (0 = off)
    # the gait-quality penalties (residual, phase_contact, foot_slip, smoothness, posture trim)
    # ramp from shape_scale_start to 1. Measured: at full weight from step 0 they make LIVING
    # net-negative for a policy with no gait yet, so dying beats trying. 0 steps = off (v2).
    shape_curriculum_steps: int = 0
    shape_scale_start: float = 0.15
    shape_curriculum_gated: bool = False    # True = retreating gate instead of a clock; see ppo.py
    w_fwd_speed: float = 2.0
    # v4: MONOTONE in achieved speed, on top of the tracking kernel -- w * clip(vx, 0, v_cmd)/v_ceiling
    # (env.joystick_income). The kernel alone is flat where the policy lives, so the last 0.1 m/s pays
    # nothing while slip and contact still charge for it, and speed plateaus (measured 1.9 m/s at every
    # command from 2.4 up). Capped at the command, so under-stick behaviour is unchanged.
    w_speed_income: float = 0.0
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
    # A contact deeper than this is the solver having lost the foot inside the floor, not the
    # sole compressing: the 3 mm TPU pad legitimately sinks a millimetre or two under load.
    floor_viol_m: float = 0.010
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
    workspace_dx_max: float = 0.28
    workspace_dz_max: float = 0.12
    workspace_dz_min: float = -0.15
    workspace_grace_s: float = 0.10
    # ----- reward: efficiency + smoothness -----------------------------------------------------
    w_torque: float = 1.0e-4
    w_motor_vel: float = 1.0e-4
    w_energy: float = 2.0e-4
    efficiency_ramp_steps: int = 120_000_000
    efficiency_target: float = 1.0
    w_action_rate: float = 0.1
    w_spec_cycle: float = 0.5               # ||S_{k+1} - S_k||^2 billed ONCE at commit (§02)
    spec_cycle_rate_invariant: bool = False  # v3: divide that bill by cadence, so a slower
                                            # clock does not buy a cheaper spec (see env.py)
    w_knob: float = 0.02                    # standing price on (Δ/Δmax, s, o/omax) per tick (contract)
    w_residual: float = 0.10                # was 0.02 (§10: every baseline lived at the bound)
    w_residual_rate: float = 0.02
    # A grounded foot with only one of its two sole pads down is rolling over an edge.  The robot
    # has no ankle, so that is the leg arriving at the wrong angle rather than a foot articulating.
    w_foot_flat: float = 0.5
    w_upright: float = 5.0
    w_height: float = 2.5
    height_floor_m: float = 0.67            # one-sided: quadratic BELOW this, 0 above
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
    # Heights scale with the robot.  The flat-foot DASH-01 stands at 0.838 m against the previous
    # plant's 1.011 m, so every threshold measured as a fraction of stance moved with it (x0.829).
    term_height: float = 0.37
    term_gravity_z: float = -0.5
    reset_joint_noise: float = 0.03
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
    bringup_pitch_deg_start: float = 2.0    # 5 deg is the MEASURED failure edge, not a mild start:
                                            # +5 deg forward gave 16/16 floor violations
    bringup_roll_deg: float = 8.0           # FINAL +- roll at release
    bringup_roll_deg_start: float = 2.0
    bringup_hold_s: tuple = (0.3, 2.5)      # how long the hand stays on
    bringup_grace_s: float = 0.35           # no fall termination while dropping / just released
    # HOW WIDE THE RAMP GOES, not just how fast. 1.0 is a 5-10 cm free drop at +-20 deg of pitch,
    # and measured 2026-09-13 that target does not merely fail, it DESTROYS the policy: three seeds
    # sitting at 0.15-0.29 m/s error and 98-100% upright at 14.7 M were at instant death by 29.5 M,
    # the gate then correctly retreated bringup_scale all the way from 0.30 back to 0.000 -- and at
    # 40 M, with bring-up fully off again, ep_len was still ~100 and the return still -100. The
    # retreat does not undo the damage, so the curriculum has to be gentle enough never to do it.
    #
    # 0.40 is ~+-9 deg of pitch and a 2-4 cm drop, comfortably outside the MEASURED hardware
    # bring-up envelope (upright to 5 deg back, feet flat) that the contract is scored on. Training
    # a stunt envelope the robot will never see, at the cost of the policy, is a bad trade.
    bringup_target: float = 1.0
    bringup_curriculum_steps: int = 40_000_000
    bringup_gate_ep_len: float = 600.0
    # THE ENVELOPE THE CONTRACT IS SCORED ON, as a fraction of the trained one. Training opens to
    # +-20 deg of pitch and a 5-10 cm drop, which no operator produces: measured on the real robot
    # the bring-up a person can actually do is upright to 5 deg BACK with both feet flat after a
    # >=1 s hold, and +5 deg FORWARD was 16/16 floor violations. 0.25 puts the scored band at
    # ~+-6.5 deg and 2-3 cm, a little wider than the measured envelope so it is not scored on its
    # own edge. verify.py reports the full width too, as margin.
    bringup_operator_scale: float = 0.25

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
    # minibatch updates over which the step size ramps from 2% of learning_rate to all of it. 0 =
    # off, which is right for a cold run. It matters for a WARM one: Adam's first step is ~lr on
    # every parameter at once, and measured that is approx_kl 0.93 against a 0.03 target on the
    # first update of a stage-3 run -- the converged policy the warm start just loaded, wrecked
    # before the early stop gets a chance to look.
    lr_warmup_updates: int = 0
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
    # ----- evaluation and the keeper (v3) ------------------------------------------------------
    # v2 kept "the checkpoint with the most survivors, then the best tracking", scored on whatever
    # commands the env drew that episode and on world-x speed. Two checkpoints were therefore never
    # asked the same question, and the run that produced the shipped 55 M policy saved a 37% -error
    # checkpoint as `best`. v3 scores a fixed command ladder from a settled start, measures BODY
    # forward speed over the settled tail only, and folds falling into one scalar so the keeper
    # cannot trade all of one for a little of the other.
    # ----- curriculum gating (v3) --------------------------------------------------------------
    # "absolute" is what v2 did: advance a curriculum only while the exploring policy's episodes
    # exceed a fixed tick count. Set above what the task reaches, it never advances and says nothing
    # -- three runs in this lineage finished with dr_scale = 0.000 that way.
    # "relative" asks for a fraction of what THIS policy has actually reached, so the ramp moves
    # whenever the policy is near its own best and retreats when it degrades.
    curriculum_gate_mode: str = "absolute"
    curriculum_gate_frac: float = 0.6       # relative: gate at 60% of the recent best episode
    curriculum_gate_floor: float = 150.0    # ... but never below 1.5 s of survival
    curriculum_gate_ref_decay: float = 0.999    # per update; the reference forgets an old peak
    # HOW LONG ONE GROUP MAY HOLD THE QUEUE. 0 = forever, which is what stage 2 did: every ramp is
    # competence-gated and retreats, so a group can hover below 0.99 indefinitely and starve the
    # rest -- all five 200 M seeds finished with dr_scale at 0.000 for exactly this reason. With a
    # cap the group keeps whatever progress it has and the next one starts anyway.
    curriculum_group_max_steps: int = 0
    # SEQUENTIAL curricula: a name may advance only once every name before it has reached 1.0.
    # Empty = the v2/early-v3 behaviour, everything advancing off one gate at once. The order
    # below is 'be able to do the job, then do it well, then do it from a bad start, then do
    # it on a different robot, then do it with a worse controller' -- with the crutch removed
    # first, while the task is still at its easiest.
    curriculum_order: tuple = ()
    eval_ladder: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)   # stick positions, fraction of v_max
    eval_warm_ticks: int = 200              # 2 s of bring-up transient excluded from the tracking mean
    eval_seconds: float = 12.0              # in-training tracking block; verify.py uses longer
    eval_bringup_envs: int = 32             # a second eval block, started the dirty way
    keeper_fall_weight: float = 2.0         # m/s of tracking error that one unit of fall RATE is worth
    keeper_heading_weight: float = 1.0      # ... and per radian of mean heading error
    eval_every_rollouts: int = 400          # greedy 16-env dash eval (the determinism-gap rule): every
                                            # 7.4 M steps; a full 60 s greedy episode costs ~200 s on a V100
    checkpoint_every_steps: int = 5_000_000


# ----- presets ------------------------------------------------------------------------------
# =============================================================================================
# Presets
# =============================================================================================
# One recipe, trained from random weights on the flat-foot robot.  There is no stage that depends
# on a policy someone else trained and no checkpoint hand-off: `train.py --preset dash` is the
# whole thing.
#
# The lineage this inherits is four measured corrections, kept as separate dicts so each one still
# says why it exists:
#
#   _TUNED    the readout-1/2 corrections.  The cadence band that removed the 5 Hz max-commit and
#             0.5 Hz freeze rails; half the per-tick residual authority at twice the bill; the
#             bounds loss and the 0.7 std cap, after 68-79% of spec means and 53-60% of residual
#             means were measured OUTSIDE [-1, 1] (a clipped mean is a bang-bang spec); and gates
#             that harden a runner rather than a stander.
#   _FAST     GPU sizing.  The batched step is latency-bound, so 2048 envs x 9 ticks keeps the
#             18 432-sample rollout at ~1.8x the samples/s of 1024 x 18.
#   _JOYSTICK the from-scratch joystick recipe: relative curriculum gates (fixed ones deadlocked --
#             dr_scale ended at 0.000 in every run of the generation before), heading on both the
#             observation and the bill, a command band a cold policy can climb, and one difficulty
#             at a time.
#   _DASH     what the flat foot changes.
# =============================================================================================


def _cfg(**kw) -> Config:
    return Config(**kw)


_TUNED = dict(
    gait_freq_hz=(1.5, 4.0),                 # neutral 2.75 Hz; no 5 Hz commit rail, no 0.5 Hz freeze
    residual_scale=0.10, w_residual=0.20,    # half the per-tick authority, twice the bill
    ent_gate_ep_len=600.0, ent_anneal_deadline_steps=40_000_000,
    curriculum_gate_ep_len=1200.0, jitter_curriculum_gate_ep_len=1200.0,
    w_bound=1.0, curriculum_retreat_frac=0.7, max_log_std=-0.3567,      # ln 0.7
)

_FAST = dict(n_envs=2048, n_steps=9, mjx_iterations=8, mjx_ls_iterations=8)

_JOYSTICK = dict(
    _TUNED,
    objective="joystick", resync_enable=False, brake_prior=0.0,
    w_heading=6.0, w_yaw_rate=1.0,
    dr_enable=True,
    curriculum_gate_mode="relative", curriculum_gate_frac=0.6, curriculum_gate_floor=150.0,
    curriculum_retreat_frac=0.5,
    cmd_range_start=(0.15, 0.45), track_sigma_start=1.5, track_sigma_steps=40_000_000,
    shape_scale_start=0.15,
    w_alive=1.5, episode_s=30.0, sprint_curriculum_steps=0,
    cmd_curriculum_steps=25_000_000,
    shape_curriculum_steps=25_000_000,
    dr_curriculum_steps=40_000_000,
    jitter_curriculum_steps=25_000_000,
)

# --------------------------------------------------------------------------------------------
# What the flat foot changes
# --------------------------------------------------------------------------------------------
# NO BASE SPRING and no bring-up.  The previous robot was a point-foot machine whose centre of
# mass sat 87 mm behind its toe contacts: standing still was never an equilibrium, so the
# curriculum had to hold the torso up with a 6-DOF spring while the policy learned to balance.
# This robot stands on its own feet -- released with the motors merely holding the nominal
# command it settles 2.5 mm and stays -- so the scaffolding is gone and every episode starts from
# a standing robot on the free plant.
#
# w_lane 0: lateral POSITION is not observable to the actor (there is no odometry, and
# double-integrated IMU noise only grows), so billing it bills an impossible task.  Heading and
# lateral velocity are observable and stay billed.
#
# The stick asks for 4 m/s and the monotone speed income keeps paying all the way there, so the
# policy plateaus at its own ceiling rather than at the edge of a reward kernel.  v_ceiling
# follows v_max: it caps both the commanded-speed weighting and the monotone term's denominator.
#
# Per-joint residual authority: the cam is a crank, so the same residual angle moves the foot much
# further through it than through the thigh.  0.10 on the cams, 0.20 elsewhere.
_DASH = dict(
    _JOYSTICK,
    residual_scale=(0.20, 0.10, 0.20, 0.20, 0.10, 0.20),
    w_lane=0.0,
    heading_avg_s=1.0, heading_euler=True,
    v_max=4.0, v_ceiling=4.0, w_speed_income=3.0,
    bringup_enable=False, hold_enable=False,
    curriculum_order=(("cmd_lo", "cmd_hi", "cmd_zero_p"),
                      ("shape_scale", "eff_scale", "stance_ratio"), "dr_scale",
                      ("ctrl_jitter_ms", "ctrl_drop_prob")),
    # NO SURVIVAL BONUS. It existed because the previous robot could not stand: its CoM sat 87 mm
    # behind its toe contacts, so staying upright was itself the hard part and had to be paid for.
    # The flat foot removed that -- released with the motors merely holding the nominal command
    # this robot settles 1.8 mm and stays -- so w_alive had become a flat subsidy for doing
    # nothing, and the first campaign collected it: 450 M steps produced 0.31 m/s at ANY command,
    # 64/64 upright, command_sweep FAIL at 92% of v_max.
    #
    # Measured 2026-09-17. The task income already favours walking 5.8x (8.500/tick at a tracked
    # 2 m/s against 0.651 standing), so the reward never preferred standing -- but a flat 1.5/tick
    # is 121% of a stander's whole income and 21% of a walker's, which compressed that edge to
    # 3.1x and made the zero-command harbour worth 4.5/tick for nothing. At 0 the stander's LIVING
    # goes to -0.231/tick, a LOSING strategy, while a walker sits near +2.7.
    #
    # It does not reintroduce the die-is-better bug: step_reward_floor clamps a tick at -0.500 and
    # -0.231 sits above it, so living still beats dying (the budget's "w_alive: live>die 0.000").
    # Cold start stays positive too (+0.230) because shape_scale starts at 0.15.
    # w_alive STAYS at the _JOYSTICK value of 1.5, and that number is load-bearing.
    #
    # Zeroing it looked right on paper -- the flat foot makes standing free, and the task income
    # already favours walking 5.8x (8.500/tick tracked at 2 m/s against 0.651 standing), so a flat
    # 1.5 only compresses that edge to 3.1x and pays a stander 121% of its whole income. Measured
    # 2026-09-17, it is still wrong: a COLD policy earns 0.051/tick of income, so with no alive
    # term LIVING clamps to the -0.500 step floor and living-vs-dying is an exact tie. The bonus
    # is not a subsidy for a finished policy, it is the ONLY learning signal before any tracking
    # income is reachable. tools/reward_budget.py --cold reports the threshold directly:
    # "w_alive for LIVING>0  1.3710", and 1.5 sits just above it.
    #
    # Run at 0: alive_frac 0.19/0.30/0.00 (was 0.98/0.83/0.75) and EVERY curriculum group handed
    # over at progress 0.00, because the gate needs ep_len 150 and the robot could not stay up --
    # 200 M steps trained entirely at the starting values.
    #
    # The standing basin is real but it is NOT priced here: walking already pays 5.8x. Attack it
    # with the income shape or the command distribution, not by removing the balance signal.
    # 200 M, not 450 M: a shorter walltime schedules far sooner, and the first campaign had
    # settled into its final behaviour long before 200 M anyway.
    total_steps=200_000_000,
    # scaled with the budget -- at the 450 M value of 80 M one group could hold 40% of this run.
    curriculum_group_max_steps=40_000_000,
    dr_scale_start=0.15,
    # NO CALIBRATION DR. Homing error was randomised because the previous robot had no mechanical
    # zero reference: it stood on point feet, so nothing about a pose on the floor pinned the leg
    # angles, and the measured drift was large -- deploy_map needed a re-fit of cam -9.0 deg and
    # thigh +4.0 deg. The flat sole replaces that. Stood on a flat floor the sole IS the reference,
    # so the zero is measured rather than guessed and there is no residual distribution to train
    # over.
    #
    # Worth removing on its own merits: this is a per-episode SYSTEMATIC bias on all six joints,
    # present on both sides of the loop (the encoder reads in the offset frame and the command is
    # issued in it), which is far harder to reject than white noise -- on the previous robot's
    # leave-one-in it caused 81% of falls at dr_scale 0.25 while every other component stayed
    # under 6%. Randomising over an uncertainty the hardware no longer has only spends margin.
    dr_joint_zero_deg=0.0,
)


PRESETS = {
    # The recipe.  One run, random weights, free plant, 450 M steps.
    "dash": lambda: _cfg(model_path="model/dash01_free.xml", **_DASH, **_FAST),

    # The same objective on the planar plant (x, z, pitch only).  Not a stage of the recipe -- the
    # flat foot removed the need for one -- but the cheapest way to ask whether a failure is about
    # balance or about the gait.
    "dash_planar": lambda: _cfg(model_path="model/dash01_planar.xml", **_DASH, **_FAST),

    # Tiny and fast: what smoke_test.py and the CI path build.  Never train on it.
    "smoke": lambda: _cfg(**dict(_DASH, model_path="model/dash01_free.xml",
                                 n_envs=8, n_steps=4, total_steps=4096,
                                 mjx_iterations=4, mjx_ls_iterations=4)),
}


def get_config(name: str = "dash") -> Config:
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
