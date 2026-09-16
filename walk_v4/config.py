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
    # ----- the 6-DOF base spring (walk_v4): ONE run instead of a planar stage 1 plus a warm start ----
    # A critically damped spring-damper on every base DOF (x y z roll pitch yaw), applied at 1 kHz,
    # billed on the force it pushes, masked per DOF per episode (A2CF), and softened on the DEFLECTION
    # it sees until the robot stops leaning on it. Stiff on y / roll / yaw it IS the planar plant;
    # released it is the free one. env.spring_law, ppo._spring_gate.
    base_spring_enable: bool = False
    # full-stiffness natural freq per DOF (0 = none). x and z free, as in the planar stage 1. Pitch 2 Hz
    # = k 120 N m/rad on the measured 0.76 kg m^2: at least the old pitch wheel's kp 100 (1 Hz was 30).
    base_spring_hz: tuple = (0.0, 2.0, 0.0, 2.0, 2.0, 2.0)
    base_spring_ref: tuple = (0.10, 0.05, 0.05, 0.10, 0.10, 0.10) # one unit of deflection (m, rad): bill + gate
    base_spring_leash: tuple = (0.10, 0.05, 1e9, 1e9, 1e9, 1e9)   # how far an anchor may lag the body (m, rad)
    base_spring_mask_p: float = 0.2              # per episode, per DOF: chance that DOF gets no spring
    w_base_spring: float = 0.1                   # bill: w * sum_i (F_i / (k0_i ref_i))^2 per tick
    # the spring is released on EPISODE LENGTH at the current stiffness (ppo._spring_gate), on the
    # same relative gate and hysteresis as every other curriculum. Deflection keeps one job:
    base_spring_retreat: float = 1.2             # SAGGING into the spring -> stiffness goes back up
    base_spring_decay_steps: int = 40_000_000    # full -> base_spring_min if every rollout advances
    base_spring_min: float = 0.01                # below this fraction of full stiffness the spring is off
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
    spec_cycle_rate_invariant: bool = False  # v3: divide that bill by cadence, so a slower
                                            # clock does not buy a cheaper spec (see env.py)
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
def _v2(**kw) -> Config:
    return Config(**kw)


# Readout-1 deltas (V2_CONTRACT.md 2026-09-10): the artifact-literal v2 runs on BOTH arms fell into a
# clock-rail exploit (CPU: 5.00 Hz on 100 % of commits, bang-bang spec; GPU: 0.5 Hz on 94 %) and
# regressed. v2b keeps the plant, obs, action layout and reward set and changes only these; the GPU
# port must mirror the first two rows, the gates are training-side.
_V2B = dict(
    gait_freq_hz=(1.5, 4.0),                 # neutral stays 2.75 Hz; no 5 Hz max-commit rail, no 0.5 Hz freeze
    residual_scale=0.10, w_residual=0.20,    # half the per-tick authority, twice the bill
    # fade the wheel once the policy runs on it, then 30 M monotonic
    ent_gate_ep_len=600.0, ent_anneal_deadline_steps=40_000_000,   # precision phase on competence, not falls
    curriculum_gate_ep_len=1200.0, jitter_curriculum_gate_ep_len=1200.0,   # harden a runner, not a stander
)

# Readout-2 deltas (V2_CONTRACT.md): v2b finished 3/3 greedy dashes at 23 M with the wheel at 0.8 and then
# collapsed; 68-79 % of the spec means and 53-60 % of the residual means lay OUTSIDE [-1, 1] (clip(mu) of
# drifted means = bang-bang spec, parked clock, saturated residual). v2c = v2b + the bounds loss, a
# retreating curriculum (0.7), a 0.7 std cap before the anneal, and the anti-crutch assist torque bill.
_V2C = dict(_V2B, w_bound=1.0, curriculum_retreat_frac=0.7, max_log_std=-0.3567)   # ln 0.7

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
            sprint_curriculum_steps=120_000_000,
            gait_curriculum_steps=240_000_000, efficiency_ramp_steps=240_000_000,
            dr_curriculum_steps=120_000_000, jitter_curriculum_steps=80_000_000, grad_guard=True)

# ============================================================================================
# v3: the from-scratch joystick recipe
# ============================================================================================
# Everything below trains from RANDOM WEIGHTS. No warm start, no checkpoint from another run, no
# stage that depends on a policy someone else trained -- `python train.py --preset v3 --name ...`
# is the whole recipe. What it changes relative to v2, and why each one is here:
#
#   1. RELATIVE curriculum gates. v2 gated every curriculum on the exploring policy exceeding a
#      fixed 1200 ticks; the task lives at 300-900, so DR, control jitter and the command widening
#      never started in ANY v2 run (dr_scale 0.000 at 55 M, 135 M and 215 M, in three runs, one of
#      which the project's own notes called robust-trained). Relative gates ask for 60% of what
#      this policy has actually reached, so they cannot deadlock.
#   2. HEADING on both sides. The actor reads its own integrated-gyro heading; the reward bills the
#      true angle. v2 had only a yaw RATE, on both sides, which is a controller with no set point:
#      measured 171 m of path for 15-38 m of net progress.
#   3. A COMMAND BAND a cold policy can climb, and a tracking tolerance that starts wide. At the
#      shipped sigma a cold robot earns 0.5% of the tracking income at full stick -- flat, and four
#      cold seeds parked on the clock rail rather than search it.
#   4. A KEEPER that scores the thing we want (`train.keeper_score`): a fixed command ladder from a
#      settled start, plus survival from a dirty one, priced against each other in one scalar.
_V3 = dict(
    _V2C,
    # --- no privileged input anywhere in the actor path
    objective="joystick", resync_enable=False, brake_prior=0.0,
    # --- run straight
    w_heading=6.0, w_yaw_rate=1.0,
    # --- be droppable
    hold_enable=True, bringup_enable=True,
    # --- and survive a plant that is not the nominal one
    dr_enable=True,
    curriculum_gate_mode="relative", curriculum_gate_frac=0.6, curriculum_gate_floor=150.0,
    curriculum_retreat_frac=0.5,
    # --- cold-start shaping
    cmd_range_start=(0.15, 0.45), track_sigma_start=1.5, track_sigma_steps=40_000_000,
    shape_scale_start=0.15,
    w_alive=1.5, episode_s=30.0, sprint_curriculum_steps=0,
    # --- ONE DIFFICULTY AT A TIME. See ppo._queued: every run before this one followed the same
    # arc, climbing to a peak and then declining from the point where the curricula started biting
    # together -- six of them advancing off one gate. The order is: be able to do the job (widen the
    # command band), stand on your own (fade the assist), do it well (the gait-quality penalties),
    # do it from a bad start (bring-up), do it on a different robot (DR), do it with a worse
    # controller (jitter and dropped ticks).
    # eff_scale and stance_ratio ride in the "do it well" group rather than being left out: a name
    # omitted from this tuple is NOT disabled, it advances unqueued, which is how DR ended up ramping
    # during a stage 1 that was supposed to have none. Grouping them with shape_scale costs no extra
    # budget -- a group takes as long as its longest member.
    curriculum_order=(("cmd_lo", "cmd_hi", "cmd_zero_p"), ("shape_scale", "eff_scale", "stance_ratio"),
                      "bringup_scale", "dr_scale", ("ctrl_jitter_ms", "ctrl_drop_prob")),
    # --- budget. SEQUENTIAL ramps do not overlap, so the run needs the SUM of them, not the max.
    # Sized so the whole queue completes inside the budget with room to consolidate afterwards:
    # 25 + 30 + 25 + 30 + 40 + 25 = 175 M of 200 M, and stage 1's shorter queue inside its 80 M.
    cmd_curriculum_steps=25_000_000,

    shape_curriculum_steps=25_000_000,
    bringup_curriculum_steps=30_000_000,
    dr_curriculum_steps=40_000_000,
    jitter_curriculum_steps=25_000_000,
    total_steps=200_000_000,
)

_V3_PROBE = dict(_V3, total_steps=40_000_000, track_sigma_steps=20_000_000,
                 shape_curriculum_steps=30_000_000,
                 dr_curriculum_steps=30_000_000, bringup_curriculum_steps=30_000_000,
                 cmd_curriculum_steps=30_000_000, eval_every_rollouts=150)

# v4 (see the v4 block at the end of PRESETS): the reflexes off, and the per-tick authority they
# carried moved into the residual of the two joints that lose one. Actuator order:
# hip_roll_L, cam_L, thigh_L, hip_roll_R, cam_R, thigh_R.
_V4_RESIDUAL = dict(residual_scale=(0.20, 0.10, 0.20, 0.20, 0.10, 0.20))


# RUN/STOP: full stick means run as fast as the income pays (v_max = v_ceiling, so the eval's
# "error" at the top rung is metres per second short of the cap), rest means step in place. Two
# rungs on the ladder, because there are only two commands.
_RUNSTOP = dict(cmd_binary=True, run_income_linear=True, eval_ladder=(0.0, 1.0))


# ONE RUN, NO CHECKPOINT: the 6-DOF base spring replaces BOTH the planar stage 1 and the stage-2
# training wheels. It starts stiff on y / roll / yaw (the planar plant), is released on deflection
# after the command band opens -- the slot the pitch wheel held -- and every old wheel is off.
# w_lane 0: lateral POSITION is not observable to the actor (no odometry; double-integrated IMU
# noise only grows), so billing it bills an impossible task, and measured it taxed travel until the
# in-place stander out-earned the runner. Heading (observable) and lateral velocity stay billed.
# Bring-up off: it has broken every stage 3 in this lineage and the deliverable does not need it.
_V4_ONE = dict(
    base_spring_enable=True,


    w_lane=0.0,
    # heading on what the robot can see, averaged: a stride's yaw wobble and the start transient
    # average out, a held offset is billed in full after ~2-3 s. Weaving is still billed by
    # yaw_rate and lat_vel.
    heading_avg_s=1.0,
    heading_euler=True,
    # the stick asks for 4 m/s and the monotone income keeps paying all the way there, so the policy
    # plateaus at its own ceiling instead of at the edge of a reward kernel. v_ceiling follows v_max
    # (it is the cap on both the commanded-speed weighting and the monotone term's denominator).
    v_max=4.0, v_ceiling=4.0, w_speed_income=3.0,
    bringup_enable=False, hold_enable=False,
    curriculum_order=(("cmd_lo", "cmd_hi", "cmd_zero_p"), "base_spring",
                      ("shape_scale", "eff_scale", "stance_ratio"), "dr_scale",
                      ("ctrl_jitter_ms", "ctrl_drop_prob")),
    curriculum_group_max_steps=80_000_000,
    # the compute of the three stages it replaces (100 + 200 + 150 M)
    total_steps=450_000_000,
)


def _v4(parent: str, **kw) -> "Config":
    """A v3 preset with v4's deltas applied on top -- built FROM the parent's Config so the two
    arms of the comparison cannot drift apart by a copy-and-edit."""
    from dataclasses import replace
    base = PRESETS[parent]()
    if kw.get("cmd_binary"):
        kw.setdefault("v_max", float(base.v_ceiling))
    return replace(base, **kw)


PRESETS = {
    "default": Config,
    # ---- v3: THE RECIPE. Two stages, the first from random weights ----------------------------
    # Measured 2026-09-12 (a 7-arm probe fleet, cold, 40 M each) the cold JOYSTICK does not work on
    # this plant, and the reason is in the income, not the optimiser. The tracking income is
    # (w_track + w_fwd*v_cmd) * exp(-|v - v_cmd| / sigma): it is FLAT far from the command, so a
    # robot that cannot walk earns almost the same whatever it does, and `alive` (75% of all income
    # at 10 M, measured with tools/reward_budget.py) pays identically for standing still. Every
    # free-plant arm parked on the 1.5 Hz clock floor and went backwards; the planar arm left the
    # floor and then railed the 4.0 Hz CEILING instead. Neither rail is a gait.
    #
    # A SPEED income does not have that shape: w_fwd_speed * vx is linear, so the first centimetre
    # per second pays, and it is the one objective this project has ever trained cold successfully
    # (walk_mit's sprint_m3_mit_s0: 600 M cold, 16/16 hundred-metre dashes at 3.07 m/s). So learn to
    # run first, then learn the stick.
    #
    # Stage 1 uses objective="speed", NOT "sprint": "speed" pins task = [1, 1] and has no distance
    # ramp, so no odometry ever enters the actor -- the no-privilege property holds from the first
    # step of the first stage rather than being restored later.
    #
    #   python walk_v4/train.py --preset v3_stage1 --name v3_stage1_s0 --seed 0 ...
    #   python walk_v4/train.py --preset v3_stage2 --name v3_stage2_s0 --seed 0     #          --warm-start walk_v4/runs/v3_stage1_s0/final.msgpack ...
    #   python walk_v4/train.py --preset v3       --name v3_s0       --seed 0     #          --warm-start walk_v4/runs/v3_stage2_s0/final.msgpack ...
    #
    # All three share one observation and action layout, so each warm start is exact, and stage 1
    # starts from random weights -- nothing outside this folder is required.
    #
    # Stage 1 -- the same objective, on the easy plant. Planar: x, z and pitch are free, y, roll
    # and yaw are absent from the model, so the policy learns a commanded-speed gait without also
    # having to stay upright sideways. Measured 2026-09-13 at 11-17 M steps, cold, everything else
    # equal: joystick-on-planar reached ep_len 473 / return 400, the SPEED objective on the same
    # plant reached 411 / 134, and the joystick on the FREE plant went backwards (143 / -46). So
    # the plant is what a cold start cannot take, not the command -- and keeping one objective
    # across both stages means the task channel never changes meaning under a warm start, which is
    # its own class of bug in this lineage.
    "v3_stage1": lambda: _v2(model_path="model/dash01_v2_planar.xml",
                             **dict(_V3, total_steps=100_000_000,
                                    # stage 1 keeps the FULL queue and simply runs out of budget
                                    # partway through it: 80 M covers the command band, the assist
                                    # fade and the shaping, and bring-up / DR / jitter never start.
                                    # A name left out of the order is NOT frozen -- it advances
                                    # unqueued -- so "off" has to be expressed as "queued behind
                                    # something this budget will not reach".
                                    ),
                             **_FAST),
    # Stage 1b -- the rung that introduces ROLL, and only roll. `dash01_v2_noyaw.xml` is the free
    # plant minus heading: x, y, z, roll, pitch. Roll is the degree of freedom that kills a
    # planar-trained policy on the free plant -- the workspace box is measured in the BASE frame and
    # roll spends most of its dz budget geometrically before the legs move -- while yaw is the
    # benign one, costing heading drift rather than terminations. Introducing them together needs a
    # training wheel, and every wheel in this lineage has to be taken away again. Introducing them
    # one at a time may not.
    #
    # Heading is a no-op here (no yaw DOF to bill), so the objective is one term simpler too.
    "v3_stage1b": lambda: _v2(model_path="model/dash01_v2_noyaw.xml",
                              **dict(_V3, total_steps=100_000_000,
                                     ),
                              **_FAST),
    # Stage 2 -- THE DELIVERABLE, AND THE OPEN PROBLEM AS OF 2026-09-13.
    #
    # Stage 1 is solved and reproducible. This stage is not yet: warm-starting a planar policy onto
    # the free plant needs a base assist to survive at all, and NO way of removing that assist has
    # worked. Measured, in order:
    #   * no assist            -- 0% upright through 44 M; 44 of 64 deaths to the WORKSPACE check
    #   * roomier workspace    -- peaks at ep_len ~250 by 6 M, back to 145 by 14.7 M
    #   * assist, 30 M fade    -- 4/4 seeds collapse at assist 0; dr_scale retreats 0.25 -> 0.000
    #   * assist, 100 M fade   -- 3/3 degrade from ep_len 600-1150 to 102-168 by assist 0.30
    #   * assist, 150 M fade   -- same shape, slower
    #   * per-episode assist   -- best at matched assist (ep_len 736 vs 276-542 at 0.73) but the
    #                             greedy eval is still 0%: the mean is carried by the assisted
    #                             share, and unassisted episodes die too fast to contribute samples
    # The rung under test is `v3_stage1b`, which introduces roll WITHOUT yaw so that no assist is
    # needed. Read walk_v4/README.md "Known limits" before spending GPU hours here.
    #
    # Joystick, free plant, heading, bring-up and DR, warm from stage 1.
    #
    # ROLL AND YAW GET TRAINING WHEELS, on the same competence-gated fade as the pitch assist.
    # Measured 2026-09-13: a stage-1 policy warm-started straight onto the free plant died in 0.38 s
    # and the killer was not falling -- `why_terminated` attributed 44 of 64 deaths to the WORKSPACE
    # check and 20 to tipping, with zero to term_low or the floor. The workspace box is +-0.14 m of
    # dz in the BASE frame, and roll spends that budget geometrically before the legs move at all: a
    # foot 0.15 m off the centreline sits h(1-cos phi) + y sin(phi) lower in a rolled base frame,
    # which is ~0.11 m at 20 deg. So a policy that has never had a roll DOF rolls, trips a
    # termination it cannot attribute to anything it did, and learns nothing from it.
    #
    # The wheel is the same mechanism the pitch assist has always used (`params.pitch_assist` scales
    # all three), so it fades once the policy runs on it and is gone from the shipped controller.
    # The fade is SLOW -- 100 M of the 200 M budget, not the inherited 30 M. Measured 2026-09-13:
    # with a 30 M fade, three of four seeds collapsed at the moment the wheels reached zero (one
    # went straight back to the 1.5 Hz clock floor) and the DR curriculum retreated from 0.25 to
    # 0.000 with them, undoing the randomisation as well. The fourth survived at 60% upright and
    # 0.62 m/s of error, so the recipe works -- it just has to hand the robot back to itself slowly
    # enough that it notices. This is the end-of-fade cliff the v2 and v2b runs died on twice.
    "v3": lambda: _v2(model_path="model/dash01_v2_free.xml",
                      **dict(_V3),         # the best-measured handover so far
                      **_FAST),
    # Stage 3 -- THE REST OF THE QUEUE. Measured 2026-09-13 on the five 200 M stage-2 seeds: the
    # joystick works (0.31-0.55 m/s of error, 60-80% upright) and the queue got exactly three
    # groups deep. Every seed finished with `bringup_scale` at 0.00-0.31 and `dr_scale` at
    # **0.000** -- the two things the deliverable is actually specified on. Sequential curricula
    # cost what they cost, and 200 M does not buy six of them.
    #
    # So stage 3 buys the last three, from the stage-2 keeper, and the first three are PINNED AT
    # FINAL rather than left out of the order: a name omitted from `curriculum_order` advances
    # unqueued from its START value, which would re-open the command band and re-fade an assist
    # that is already gone. Pinning is expressed as "no ramp at all", because `initial_params`
    # reads a zero ramp as "begin at the target".
    #
    # The training wheels do not come back. roll/yaw/pitch assist kp are ZERO here, not faded --
    # the parent already stands unassisted (every eval in stage 2 ran at pitch_assist 0), so there
    # is nothing to hand back and no second end-of-fade cliff to fall off.
    #
    # Order is bring-up BEFORE DR, and that way round on purpose: bring-up is what shortens
    # episodes, and both gates are relative to `_ep_len_ref`. Letting the reference settle onto the
    # bring-up task first means DR ramps against a stable yardstick; the other order would have DR
    # climb against a runner's episode length and then retreat when bring-up cut it in half, which
    # is the failure that left dr_scale at 0.000 three times in this lineage.
    "v3_stage3": lambda: _v2(model_path="model/dash01_v2_free.xml",
                             **dict(_V3,
                                    # --- pinned at final (the parent finished these)
                                    cmd_curriculum_steps=0,        # cmd_lo/hi -> cmd_range, zero_p -> 0.25
                                    shape_curriculum_steps=0,      # shape_scale -> 1.0
                                    efficiency_ramp_steps=0,       # eff_scale -> efficiency_target
                                    gait_curriculum_steps=0,       # stance_ratio -> final
                                    track_sigma_steps=0,           # track_sigma -> 0.6

                                    # --- and the wheels are off for good




                                    # --- DO NOT REFILL THE ACTION NOISE.
                                    # Measured 2026-09-13, 13 M steps into the first attempt at
                                    # this stage: four seeds warm-started from a policy running at
                                    # ep_len 1413 and immediately fell to 120-241, one straight
                                    # onto the 1.5 Hz clock floor -- on an env EASIER than the one
                                    # the checkpoint came from (bringup_scale starts at 0). The
                                    # task did not change; the exploration did. The parent had
                                    # annealed to std_mean 0.17 and `warmstart_reset_log_std`
                                    # refills log_std to max_log_std, i.e. std 0.70. Four times the
                                    # action noise on a policy already at its stability edge reads
                                    # as a collapse and costs tens of millions of steps to undo --
                                    # which is also, in hindsight, where stage 2 spent the 59 M it
                                    # took to get off 0% upright.
                                    #
                                    # Resetting exploration is right when the TASK changes and the
                                    # old policy's habits are wrong. Here the plant, the objective
                                    # and the command band are identical and the two new curricula
                                    # ramp from zero; the exploration this stage needs is state
                                    # diversity (drops, held releases, randomised plants), which
                                    # bring-up and DR supply directly. So keep the parent's log_std
                                    # and CAP it where the parent left off, or the entropy bonus
                                    # simply walks it back up to 0.70 over the first few updates.
                                    # AND DO NOT FLOOR THE OBSERVATION VARIANCE.
                                    # `warmstart_var_floor` raises every obs channel's variance to
                                    # 0.01 before normalising, which shrinks the normalised
                                    # magnitude of every channel that genuinely varies less than
                                    # that -- 59 of 412 dims here. Measured 2026-09-13 by applying
                                    # the surgery to the stage-2 keeper and re-running the ladder:
                                    #
                                    #            floor 0.01   no floor
                                    #   1.80 m/s   0% upright  100% upright
                                    #   heading      23.6 deg     2.0 deg (at rest)
                                    #
                                    # i.e. the thing that is supposed to protect a warm start is
                                    # what was destroying it, and had been doing so at every stage
                                    # handover in this lineage. The count cap is innocent: capped
                                    # at 1e5, 1e7 or not at all, the ladder is identical.
                                    #
                                    # The floor is not WRONG in general -- it exists because v2's
                                    # task[0] had variance 6.5e-5, so a command of 0.89 normalised
                                    # to -13.6 sigma, and because a channel that is identically
                                    # zero on one plant (lateral velocity on planar) and non-zero
                                    # on the next divides by ~0 in a stage-1 -> stage-2 transfer.
                                    # Neither applies here: the joystick command sweeps its whole
                                    # range, and stage 3 inherits the SAME plant it will train on.
                                    warmstart_var_floor=0.0,
                                    warmstart_reset_log_std=False,
                                    max_log_std=-1.3863,          # ln(0.25) = the parent's clamp
                                    std_anneal_target=0.17,   # not below where the parent ran
                                    ent_coef=0.003,
                                    # --- what this stage is for
                                    curriculum_order=("bringup_scale", "dr_scale",
                                                      ("ctrl_jitter_ms", "ctrl_drop_prob")),
                                    # HOW MUCH OF TRAINING IS THE FIRST HALF-SECOND.
                                    # At full bringup_scale the stock shares put 25% of episodes on a free
                                    # drop and 35% on a held release: 60% of every rollout starts dirty, and
                                    # almost all of those end in the 100-point fall penalty. Measured
                                    # 2026-09-13, three seeds went from 50-62% upright at 29 M to 0-6% at
                                    # 44 M as bring-up opened -- and that eval starts from the SETTLED
                                    # keyframe, so what was lost is the cruise, not the bring-up.
                                    #
                                    # Bring-up is a transient the robot does once per run; cruising is what
                                    # it does for the other 99% of the time, and spending 60% of the
                                    # episodes on the transient prices it accordingly. 10% drop / 25% held
                                    # keeps a clear majority of episodes on the task, and the split leans to
                                    # HELD because that is the only bring-up the hardware actually has -- a
                                    # person holds the robot and lets go. Nobody drops it 10 cm onto its feet.
                                    bringup_drop_frac=0.10,
                                    bringup_held_frac=0.25,
                                    lr_warmup_updates=300,   # ~8 rollouts at 36 updates each
                                    # HOW LONG THE HAND STAYS ON -- a TRAINING cost, not a realism dial.
                                    # While held, env.py pins every base DOF (x, y, z, roll, pitch, yaw) and
                                    # zeroes every base velocity at each 1 kHz substep. So on a held episode the
                                    # robot is commanded, say, 2.4 m/s, achieves exactly 0, earns
                                    # exp(-2.4/0.6) = 1.8% of the tracking income -- and NOTHING it does can
                                    # change any of that. At the stock 0.3-2.5 s that is up to 250 control ticks
                                    # of uncontrollable, low-reward transitions entering the PPO batch as
                                    # ordinary ones, on states that look exactly like normal stance.
                                    #
                                    # That is why a bring-up share of 8% at +-6 deg of tilt could destroy a
                                    # policy sitting at 81% upright: it was never the tilt, it was the dead
                                    # ticks. The hold only has to last long enough for the feet to settle into
                                    # contact before the release -- a few ticks, not a few hundred.
                                    bringup_hold_s=(0.05, 0.25),
                                    bringup_target=0.40,
                                    bringup_curriculum_steps=40_000_000,
                                    dr_curriculum_steps=40_000_000,
                                    jitter_curriculum_steps=25_000_000,
                                    # 40 + 40 + 25 = 105 M of ramp in a 150 M budget. The cap is
                                    # what guarantees DR gets its turn: bring-up may hold the queue
                                    # for 50 M and then hands over at whatever width it reached.
                                    curriculum_group_max_steps=50_000_000,
                                    total_steps=150_000_000),
                             **_FAST),
    # Stage 3, with an HONEST TOP OF THE STICK.
    #
    # v_max is what gives the operator's stick its meaning -- task[0] = v_cmd / v_max -- and 3.6
    # came from `tools/speed_lib.py`, a CEM search over open-loop gait specs. That search answers
    # "what can the action space express", not "what can a policy hold". Closed loop it cannot hold
    # 3.6. Measured 2026-09-13 on the stage-2 keeper, greedy, 8 envs per rung, 12 s
    # (tools/speed_frontier.py):
    #
    #     commanded  1.80  2.10  2.40  2.70  3.00  3.30  3.60
    #     achieved   1.64  1.82  1.98  2.16  2.50  2.64  2.79
    #     upright    100%  100%  100%  100%   25%    0%    0%
    #     heading     8.0   6.4   6.7   5.8  10.5  13.7  17.5  deg
    #
    # Two things follow. The obvious one: the top third of the shipped slider is a speed the robot
    # falls over at, and an operator pushing the stick all the way forward is entitled to expect
    # otherwise. The less obvious one, and the reason this is a TRAINING change and not just an
    # export flag: `cmd_hi` reaches 1.0, so about a fifth of every episode in stage 2 was spent
    # asking for 3.0-3.6 m/s, i.e. asking the policy to fall. That is a fifth of the sample budget
    # spent training the failure and paying the 100-point fall penalty for it.
    #
    # 2.4 rather than 2.7: 2.7 is upright but it is the last rung before the cliff, and the whole
    # point is that full stick should be comfortable. What it does NOT try to fix is the ~0.15 m/s
    # undershoot that runs through the whole band -- that one is the policy's honest risk-adjusted
    # optimum (income (3 + 2*v_cmd) * exp(-err/0.6) against a 100-point fall penalty), and
    # sharpening the tracking term to close it would buy speed with survival, which is the wrong
    # trade for a machine an operator is holding.
    #
    # THE WARM-START TRAP APPLIES HERE. task[0] changes meaning: 1.0 meant 3.6 m/s to the parent and
    # means 2.4 here, so the policy starts by over-delivering against every command until the
    # tracking income re-teaches the scale. That is a real perturbation and the reason this ships as
    # a SEPARATE ARM next to the v_max 3.6 stage-3 runs rather than as a change to them.
    "v3_stage3_v24": lambda: _v2(model_path="model/dash01_v2_free.xml",
                                 **dict(_V3,
                                        v_max=2.4,
                                        cmd_curriculum_steps=0,
                                        shape_curriculum_steps=0,
                                        efficiency_ramp_steps=0,
                                        gait_curriculum_steps=0,
                                        track_sigma_steps=0,





                                        # AND DO NOT FLOOR THE OBSERVATION VARIANCE.
                                        # `warmstart_var_floor` raises every obs channel's variance to
                                        # 0.01 before normalising, which shrinks the normalised
                                        # magnitude of every channel that genuinely varies less than
                                        # that -- 59 of 412 dims here. Measured 2026-09-13 by applying
                                        # the surgery to the stage-2 keeper and re-running the ladder:
                                        #
                                        #            floor 0.01   no floor
                                        #   1.80 m/s   0% upright  100% upright
                                        #   heading      23.6 deg     2.0 deg (at rest)
                                        #
                                        # i.e. the thing that is supposed to protect a warm start is
                                        # what was destroying it, and had been doing so at every stage
                                        # handover in this lineage. The count cap is innocent: capped
                                        # at 1e5, 1e7 or not at all, the ladder is identical.
                                        #
                                        # The floor is not WRONG in general -- it exists because v2's
                                        # task[0] had variance 6.5e-5, so a command of 0.89 normalised
                                        # to -13.6 sigma, and because a channel that is identically
                                        # zero on one plant (lateral velocity on planar) and non-zero
                                        # on the next divides by ~0 in a stage-1 -> stage-2 transfer.
                                        # Neither applies here: the joystick command sweeps its whole
                                        # range, and stage 3 inherits the SAME plant it will train on.
                                        warmstart_var_floor=0.0,
                                        warmstart_reset_log_std=False,
                                        max_log_std=-1.3863,
                                        std_anneal_target=0.17,   # not below where the parent ran
                                        ent_coef=0.003,
                                        curriculum_order=("bringup_scale", "dr_scale",
                                                          ("ctrl_jitter_ms", "ctrl_drop_prob")),
                                        # HOW MUCH OF TRAINING IS THE FIRST HALF-SECOND.
                                        # At full bringup_scale the stock shares put 25% of episodes on a free
                                        # drop and 35% on a held release: 60% of every rollout starts dirty, and
                                        # almost all of those end in the 100-point fall penalty. Measured
                                        # 2026-09-13, three seeds went from 50-62% upright at 29 M to 0-6% at
                                        # 44 M as bring-up opened -- and that eval starts from the SETTLED
                                        # keyframe, so what was lost is the cruise, not the bring-up.
                                        #
                                        # Bring-up is a transient the robot does once per run; cruising is what
                                        # it does for the other 99% of the time, and spending 60% of the
                                        # episodes on the transient prices it accordingly. 10% drop / 25% held
                                        # keeps a clear majority of episodes on the task, and the split leans to
                                        # HELD because that is the only bring-up the hardware actually has -- a
                                        # person holds the robot and lets go. Nobody drops it 10 cm onto its feet.
                                        bringup_drop_frac=0.10,
                                        bringup_held_frac=0.25,
                                        lr_warmup_updates=300,   # ~8 rollouts at 36 updates each
                                        # HOW LONG THE HAND STAYS ON -- a TRAINING cost, not a realism dial.
                                        # While held, env.py pins every base DOF (x, y, z, roll, pitch, yaw) and
                                        # zeroes every base velocity at each 1 kHz substep. So on a held episode the
                                        # robot is commanded, say, 2.4 m/s, achieves exactly 0, earns
                                        # exp(-2.4/0.6) = 1.8% of the tracking income -- and NOTHING it does can
                                        # change any of that. At the stock 0.3-2.5 s that is up to 250 control ticks
                                        # of uncontrollable, low-reward transitions entering the PPO batch as
                                        # ordinary ones, on states that look exactly like normal stance.
                                        #
                                        # That is why a bring-up share of 8% at +-6 deg of tilt could destroy a
                                        # policy sitting at 81% upright: it was never the tilt, it was the dead
                                        # ticks. The hold only has to last long enough for the feet to settle into
                                        # contact before the release -- a few ticks, not a few hundred.
                                        bringup_hold_s=(0.05, 0.25),
                                        bringup_target=0.40,
                                        bringup_curriculum_steps=40_000_000,
                                        dr_curriculum_steps=40_000_000,
                                        jitter_curriculum_steps=25_000_000,
                                        curriculum_group_max_steps=50_000_000,
                                        total_steps=150_000_000),
                                 **_FAST),
    # The same, with a MINIMAL bring-up: 0.15 => +-4.7 deg of pitch and a 1.6-3.2 cm drop on 5% of
    # episodes. That is the measured hardware envelope (upright to 5 deg back, feet flat) and no
    # more. If even this collapses, bring-up on this plant is a research problem rather than a
    # tuning one, and the answer is to ship without it and let the operator square the robot up.
    "v3_stage3_dr_bu": lambda: _v2(model_path="model/dash01_v2_free.xml",
                                   **dict(_V3,
                                          v_max=2.4,
                                          cmd_curriculum_steps=0, shape_curriculum_steps=0,
                                          efficiency_ramp_steps=0, gait_curriculum_steps=0,
                                          track_sigma_steps=0, 




                                          warmstart_var_floor=0.0,
                                          warmstart_reset_log_std=False,
                                          max_log_std=-1.3863, std_anneal_target=0.17,
                                          ent_coef=0.003, lr_warmup_updates=300,
                                          bringup_target=0.15,
                                          bringup_drop_frac=0.10, bringup_held_frac=0.25,
                                          bringup_hold_s=(0.05, 0.25),
                                          # DR FIRST this time. Bring-up is what has broken every
                                          # run, so it goes last, where the budget it can spoil is
                                          # the smallest.
                                          curriculum_order=("dr_scale", "bringup_scale",
                                                            ("ctrl_jitter_ms", "ctrl_drop_prob")),
                                          dr_curriculum_steps=50_000_000,
                                          bringup_curriculum_steps=30_000_000,
                                          jitter_curriculum_steps=25_000_000,
                                          curriculum_group_max_steps=60_000_000,
                                          total_steps=150_000_000),
                                   **_FAST),
    # Stage 3, DR ONLY -- bring-up left out entirely.
    #
    # Measured 2026-09-14, after five separate structural fixes (the warm-start variance floor, the
    # log_std refill, Adam's first step, the held-start dead ticks, the queue deadline): every seed
    # still reaches 0.27-0.43 m/s of error at 61-92% upright by 15 M and then falls off a cliff
    # between 30 M and 59 M, from which it never returns. The cliff is sharp -- reward_mean +0.42 ->
    # -3.81 across four rollouts, 1381 of 1381 episodes falling at ep_len 103 -- and it arrives when
    # `bringup_scale` is around 0.34, i.e. +-8 deg of tilt on 12% of episodes.
    #
    # It is NOT the objective being upside down. `tools/reward_budget.py` on the parent under this
    # exact preset: income 4.36, cost 2.05, LIVING **+1.157/tick** at curriculum start and +1.091 at
    # final. Living pays. (That check is the project's own gate and running it earlier would have
    # saved a day: see the cold-start note.)
    #
    # So bring-up is the thing that breaks it, and the deliverable does not actually need the
    # trained envelope. DASH-01's measured bring-up is upright to 5 deg BACK with both feet flat
    # after a >=1 s hold -- an operator holding the robot and letting go, not a drop. This preset
    # buys what stage 2 demonstrably lacks (`dr_scale` 0.000 in all five seeds, and 0% upright at
    # every command on a randomised plant) and leaves the start state alone.
    #
    # bringup_enable=False is what freezes it: a name omitted from `curriculum_order` advances
    # UNQUEUED, so "off" has to be expressed as the feature being off, not as the ramp being absent.
    "v3_stage3_dr": lambda: _v2(model_path="model/dash01_v2_free.xml",
                                **dict(_V3,
                                       v_max=2.4,
                                       cmd_curriculum_steps=0,
                                       shape_curriculum_steps=0,
                                       efficiency_ramp_steps=0,
                                       gait_curriculum_steps=0,
                                       track_sigma_steps=0,





                                       warmstart_var_floor=0.0,
                                       warmstart_reset_log_std=False,
                                       max_log_std=-1.3863,
                                       # do NOT anneal the action noise below where the parent ran.
                                       # It finished at std_mean 0.17; annealing to 0.12 leaves a
                                       # nearly deterministic policy with no way back out of a bad
                                       # basin, which is what "collapses and never recovers" looks
                                       # like from the inside.
                                       std_anneal_target=0.17,
                                       ent_coef=0.003,
                                       lr_warmup_updates=300,
                                       bringup_enable=False,
                                       hold_enable=False,
                                       curriculum_order=("dr_scale",
                                                         ("ctrl_jitter_ms", "ctrl_drop_prob")),
                                       dr_curriculum_steps=50_000_000,
                                       jitter_curriculum_steps=25_000_000,
                                       curriculum_group_max_steps=60_000_000,
                                       total_steps=150_000_000),
                                **_FAST),
    # THE HANDOVER, done as a distribution instead of a dial: assist_per_episode makes
    # pitch_assist the share of episodes that get help, so unassisted episodes are in the training
    # distribution from the first rollout and the fade reweights rather than removes.
    "v3_epiassist": lambda: _v2(model_path="model/dash01_v2_free.xml",
                                **dict(_V3),
                                **_FAST),
    # insurance against the cliff: the same recipe, handing the robot back over 150 M of 200 M
    # instead of 100. The fade length is the parameter every collapse in this lineage has turned on.
    "v3_slowfade": lambda: _v2(model_path="model/dash01_v2_free.xml",
                               **dict(_V3),
                               **_FAST),
    # the same, without the wheels: the control that says whether they are what mattered
    "v3_nowheels": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V3, **_FAST),
    # ... and the other way of paying for roll: leave the robot to hold itself up, and stop the
    # WORKSPACE BOX from calling body attitude a reach violation. The box is what killed the
    # unassisted transfer (44 of 64 deaths), and it was calibrated on a plant that cannot roll: a
    # planted foot 0.15 m off centre moves ~0.11 m in the base frame at 20 deg of roll, against a
    # 0.14 m budget. Widening it by roughly that much, and tripling the 0.10 s grace, asks whether
    # the box alone was the problem -- which would be a better answer than a training wheel, because
    # a wheel has to be taken away again and this lineage falls over every time it is.
    # RESULT 2026-09-13: a roomier box is NOT enough on its own. Two seeds peaked at ep_len ~250
    # around 6 M and were back to 145-155 with negative returns by 14.7 M, while the wheeled runs
    # were at 600-1150. So the box was a contributing killer -- 44 of 64 deaths -- but the roll and
    # yaw assists are doing more than dodging it: they hold the two new degrees of freedom still
    # long enough for the policy to learn a gait in them at all. Kept as the record of the
    # experiment, not as a recipe.
    "v3_roomybox": lambda: _v2(model_path="model/dash01_v2_free.xml",
                               **dict(_V3, workspace_dz_max=0.22, workspace_dz_min=-0.26,
                                      workspace_dx_max=0.40, workspace_grace_s=0.30),
                               **_FAST),
    # the same recipe on the planar model (x, z, pitch free): an iteration sandbox, and the control
    # that says whether a cold-start failure is the objective or the plant
    "v3_planar": lambda: _v2(model_path="model/dash01_v2_planar.xml", **_V3, **_FAST),
    # ---- cold-start probes (40 M each): which of these does a from-scratch run actually need? ---
    # Arm A is the recipe. Each other arm changes ONE thing, so a difference is attributable.
    "v3_probe_a_lowcmd": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V3_PROBE, **_FAST),
    # B: the v2 command band (start at 80-100% of top speed). Does asking a cold policy to sprint
    # actually cost anything, or was the band never the problem?
    "v3_probe_b_highcmd": lambda: _v2(model_path="model/dash01_v2_free.xml",
                                      **dict(_V3_PROBE, cmd_range_start=(0.8, 1.0)), **_FAST),
    # C: no tolerance curriculum -- the shipped 0.6 m/s Laplace width from step 0
    "v3_probe_c_tightsig": lambda: _v2(model_path="model/dash01_v2_free.xml",
                                       **dict(_V3_PROBE, track_sigma_start=0.6, track_sigma_steps=0),
                                       **_FAST),
    # D: no training wheel at all. The pitch assist props the base up and then fades, and every
    # collapse in this lineage happened exactly when it ran out -- is it load-bearing or is it the
    # cliff?
    "v3_probe_d_noassist": lambda: _v2(model_path="model/dash01_v2_free.xml",
                                       **dict(_V3_PROBE), **_FAST),
    # E: the planar plant. If A fails and E succeeds, the free plant's roll and yaw are the blocker
    # and the recipe needs the S1 -> S2 ladder rather than a better objective.
    "v3_probe_e_planar": lambda: _v2(model_path="model/dash01_v2_planar.xml", **_V3_PROBE, **_FAST),
    # G: the spec-cycle bill made cadence-invariant. It is charged ONCE per commit, so halving the
    # clock halves what the policy pays per second for rewriting its spec -- a standing discount
    # collected by exactly the 1.5 Hz rail every free-plant arm sat on.
    "v3_probe_g_rateinv": lambda: _v2(model_path="model/dash01_v2_free.xml",
                                      **dict(_V3_PROBE, spec_cycle_rate_invariant=True), **_FAST),
    # H: curricula that wait for real competence. The relative gate opened everything at ep_len 157
    # (1.6 s of survival), and probe_a went backwards right afterwards -- the deadlock is fixed but
    # the bar is now too low. 400 ticks still cannot deadlock the way an absolute 1200 did, because
    # it is a floor under a relative gate, not the gate itself.
    "v3_probe_h_lategate": lambda: _v2(model_path="model/dash01_v2_free.xml",
                                       **dict(_V3_PROBE, curriculum_gate_floor=400.0), **_FAST),
    # F: no dirty starts. Bring-up randomisation makes a share of episodes begin already falling;
    # on a policy that cannot walk yet that may be all cost and no lesson.
    "v3_probe_f_nobringup": lambda: _v2(model_path="model/dash01_v2_free.xml",
                                        **dict(_V3_PROBE, bringup_enable=False, hold_enable=False),
                                        **_FAST),

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
    "v2b_s1_planar_noassist": lambda: _v2(model_path="model/dash01_v2_planar.xml", **_V2B),
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
                                  # w_alive 1.5, not 0.5: measured 2026-09-11 (tools/reward_budget.py,
                                  # warm-start stats, curriculum start) the warm start runs 600/600 at a
                                  # 3.2 m/s command but still earns income 1.42 against 1.66 of cost, so
                                  # living was NEGATIVE at every command and the optimiser prefers to
                                  # fall. `alive` is a constant per-tick term, so +1.0 lifts every command
                                  # to positive living (+0.20 to +0.42/tick) while staying small against
                                  # the 9.0 peak of the tracking income -- it buys survival a floor, not a
                                  # standing attractor: at a 3.2 command, standing pays 0.5/tick against
                                  # running's 3.8.
                                  w_alive=1.5, episode_s=30.0, sprint_curriculum_steps=0,
                                  total_steps=140_000_000),
    # v3 + the DR the whole lineage has never actually had. Measured 2026-09-12 from the checkpoint
    # sidecars: dr_scale ended at 0.000 in v3_joy_s7 AND s8 AND v2c_s2_brakeprior_s41 (peak 0.041 of
    # 215 M), because _V2B raised the competence gate to ep_len > 1200 while these policies live at
    # 300-900 ticks -- so `_gated` subtracts progress more often than it adds and the ramp sits on the
    # floor. ctrl_jitter/ctrl_drop are zero for the same reason. Every policy so far is nominal-plant
    # only, which is the largest sim-to-real risk in the stack.
    #
    # Back to the contract's 600 (and retreat 0.5, not _V2C's 0.7) so the ramp can actually move, warm
    # started from the 55 M joystick that already tracks. Budget 70 M, not 140: both v3 seeds peaked
    # near 55 M and collapsed after ~60 M, so the extra steps bought a worse policy.
    "v3_joystick_dr": lambda: _v2(model_path="model/dash01_v2_free.xml",
                                  **dict(_V2C, curriculum_gate_ep_len=600.0,
                                         jitter_curriculum_gate_ep_len=600.0,
                                         curriculum_retreat_frac=0.5),
                                  **_FAST,
                                  objective="joystick", resync_enable=False, brake_prior=0.0,
                                  hold_enable=True, bringup_enable=True,
                                  w_alive=1.5, episode_s=30.0, sprint_curriculum_steps=0,
                                  dr_curriculum_steps=40_000_000,
                                  total_steps=70_000_000),
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
    "v2c_s2_free_fast_rollassist": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V2C, **_FAST),
    # all three base wheels (pitch, roll, yaw) held at the start and faded together
    "v2c_s2_free_fast_basewheels": lambda: _v2(model_path="model/dash01_v2_free.xml", **_V2C, **_FAST),
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
    # ============================================================================================
    # v4: THE v3 RECIPE WITHOUT THE REFLEXES (2026-09-15)
    # ============================================================================================
    # Measured on the v2 S2 runner (walk_v2/tools/reflex_ablation.py, 16 greedy x 20 s, nominal):
    #   as trained 16/16 at 2.98 m/s; pitch reflex off 0/16 in 0.42 s; roll reflex off 0/16 in
    #   0.57 s; roll FEEDBACK off with its bias kept 15/16 upright but -3.3 m (it stands, it does
    #   not run); roll BIAS off 0/16. And zero action (= a fresh policy's mean, roll gains 0) with
    #   the pitch reflex on 1.07 s vs off 1.00 s -- the fixed reflex buys a cold policy nothing.
    # So on a trained policy both reflexes are load-bearing, but neither has a property the network
    # lacks: the residual runs at the same 100 Hz on the same IMU through the same delay. What they
    # have is AUTHORITY -- the roll feedback writes 21 deg p-p on the hips and the pitch reflex
    # reaches 0.25 rad, against a +-0.10 rad residual. Two consequences for a reflex-free retrain:
    #   1. every reflex gain is ZERO (dims 36:39 stay in the layout, inert, so the exporter, the
    #      mirror and the Pi runtime are untouched); the sim reflexes read GROUND-TRUTH attitude
    #      while the Pi feeds them the measured IMU, so this also removes an untrained sim2real path;
    #   2. the hip and thigh residual is widened to +-0.20 rad and the cams stay at +-0.10. The
    #      periodic part of the old hip-roll swing fits the latched S_hip series (roll_amp 0.20);
    #      the corrective part is what the wider residual is for. The cam stays narrow because the
    #      per-tick channel is where walk_mit's runner hid a whole gait (residual 45-95 % of motion,
    #      saturated on 60-95 % of steps) -- v2b halved it for that reason, and this reopens only
    #      the two joints that lost a reflex. w_residual bills action units, so the bill is unchanged.
    # Stages are v3's, unchanged: planar, then free, then DR. The *_reflex control twins are gone
    # with the reflexes themselves (2026-09-16): the ablation question was answered (reflex-free
    # trains at least as well) and the control law no longer has the code to switch back on.
    "v4_stage1": lambda: _v4("v3_stage1", **_V4_RESIDUAL),
    "v4": lambda: _v4("v3", **_V4_RESIDUAL),
    "v4_stage3_dr": lambda: _v4("v3_stage3_dr", **_V4_RESIDUAL),
    # RUN/STOP, the same two arms. Why: on the joystick objective 2 of 3 reflex-free stage-2 seeds
    # and 1 of 3 controls settled into walking in place at every command, 100 % upright, error =
    # the command. The tracking kernel makes that basin pay; see env._reward (run_income_linear).
    # Alive is NOT replaced by a time-proportional fall penalty: with truncation terminal (ppo.py)
    # alive b and a terminal penalty b(1 - gamma^left)/(1 - gamma) differ by an action-independent
    # constant, i.e. they are the same objective. Its SIZE is the lever; measure it with
    # tools/reward_budget.py --cold before choosing.
    # STAGE 2 WITHOUT THE LANE TERM. Measured 2026-09-15 (reward_budget.py --profile lane, stage-2
    # keepers, full stick, 30 s): lane = -2(|y| - 0.25)^2 reaches its -2/tick cap for the runner
    # by ~8-15 s and for the in-place stander by ~20 s, on a y the actor cannot observe -- it taxes
    # distance x heading error. With it the stander out-earns the runner; without it the order
    # flips. Heading (true yaw, observable through the integrated gyro) and lat_vel stay. The
    # paired control is v4_s{0,1,2}: same stage-1 parent, same seed, this one weight different.
    "v4_nolane": lambda: _v4("v3", **_V4_RESIDUAL, w_lane=0.0),
    # ONE RUN from random weights on the free plant -- see _V4_ONE. No stage-1 checkpoint needed.
    "v4_one": lambda: _v4("v3", **_V4_RESIDUAL, **_V4_ONE),
    "v4_one_runstop": lambda: _v4("v3", **_V4_RESIDUAL, **_V4_ONE, **_RUNSTOP),
    "v4_runstop_stage1": lambda: _v4("v3_stage1", **_V4_RESIDUAL, **_RUNSTOP),
    "v4_runstop": lambda: _v4("v3", **_V4_RESIDUAL, **_RUNSTOP),
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
