"""All tunable parameters for DASH-01 sprint training, in one place.

One dataclass, explicit presets. The design is the post-review "state of the art" stack
(2026-07-17): per-step Fourier gait spec + residuals (CPG-RL / PMTG hybrid), privileged
base-velocity + distance-to-go observations, dense speed income + per-step clock cost,
phase-gated contact scheduling (Siekmann) with a stance-ratio curriculum that morphs the
gait from walking (stance 65%) toward running (stance 42% -> double-swing flight window),
and Cassie-100m-style efficiency terms (torque / motor-velocity / mechanical power).

Milestone curriculum: base_lock rails base DOFs ([X, Y, Z, roll, pitch, yaw], 1 = locked).
m1 = only X free (Z railed at a randomized ride height), m2 = +Z, m3 = +pitch,
m4 = +Y, m5 = +roll, m6 = fully free. Warm-start each stage from the previous one:
base_lock does not change obs/action dims, so checkpoints load as-is.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ----- model & timing -----
    model_path: str = "model/dash01.xml"   # resolved relative to this package if not found from CWD
    control_decimation: int = 20           # sim steps per control step. sim is 1 kHz -> 50 Hz control
    keyframe: str = "stand"

    # ----- base DOF locking (the m1..m6 curriculum) -----
    base_lock: tuple = (0, 0, 0, 0, 0, 0)  # [X, Y, Z, roll, pitch, yaw], 1 = locked (rail)
    z_rail_randomize: bool = False         # m1: rail Z at a per-episode RANDOM ride height
    z_rail_range: tuple = (0.90, 1.03)     # meters; sub-band of the measured feasible ride band
    ride_height_lut: str = "model/ride_height_lut.npz"

    # ----- objective -----
    # "sprint": one episode = one dash — stand at x0, run sprint_dist_m, stop past the line.
    #   Reward = dense speed income + a constant per-step clock cost (what actually prices TIME:
    #   per-step vx alone integrates to w*distance no matter the pace) + stop-phase income +
    #   finish bonus. The stop signal the policy observes is the task obs flipping run->stop.
    # "speed": endless max-forward-speed on the same plant (debug / gait-shaping runs).
    objective: str = "sprint"
    v_ceiling: float = 3.0              # m/s cap on the speed income (kinematic ceiling ~ stride*cadence)
    # income per m/s, linear and SYMMETRIC: clip(vx, -v_ceiling, +v_ceiling). Backward motion must
    # pay negative income — with a one-sided clip(vx, 0, ...) a policy can shuttle back and forth
    # in front of the line forever (forward legs earn, backward legs only pay the clock) and that
    # strictly out-values ever crossing (verified exploit, review 2026-07-17).
    w_fwd_speed: float = 2.0
    # anti-lunge: multiply POSITIVE fwd_speed income by an uprightness factor so a toppling robot
    # cannot bank speed reward while falling (the m3_speed_v2 exploit). Inert on m1/m2 (pitch+roll
    # railed -> grav_z~=-1 -> factor 1). Pitch-free presets (m3..m6) set the flag via _extras.
    speed_upright_gate: bool = False
    speed_upright_c0: float = 0.5       # -grav_z at/below which income is fully killed (=-term_gravity_z)
    speed_upright_k: float = 1.0        # exponent on the uprightness factor (>1 = sharper gate)
    w_alive: float = 0.0                # MUST stay 0 in sprint (paying per second rewards dawdling);
    #                                     the speed presets set 0.5 (no clock exists there)
    sprint_dist_m: float = 100.0        # the finish line (meters of base X from the reset pose)
    sprint_dist_start_m: float = 25.0   # curriculum: line starts near so slow early policies still
    sprint_curriculum_steps: int = 30_000_000   # reach it and learn the stop; 0 = no ramp
    sprint_brake_m: float = 5.0         # free braking zone past the line (sprinters run THROUGH it)
    w_time: float = 0.5                 # per-control-step clock cost until stopped; with w_fwd_speed
    #                                     this sets the break-even pace (0.25 m/s) below which
    #                                     moving is worse than useless
    # stop-phase 'be stationary' kernel: w*exp(-(vx/sigma)^2). MUST stay BELOW w_time so the stop
    # phase is net-negative per step (max 0.4 - 0.5 = -0.1): the hold-timer reset is under policy
    # control, so any net-positive stop income is farmable forever by twitching just before the
    # 1 s hold completes (verified exploit: at the old w=2.0, hovering at the line paid ~149
    # discounted vs ~120 for finishing). Net-negative + finish_bonus makes finishing dominant
    # while the kernel still provides the braking gradient.
    w_stop_vel: float = 0.4
    stop_sigma: float = 0.3             # m/s width of the 'stationary' kernel
    w_overrun: float = 1.0              # per-meter penalty past line+brake zone (capped, see _pen)
    stop_speed_eps: float = 0.15        # |vx| below this counts as stopped
    stop_hold_s: float = 1.0            # must stay stopped this long -> success termination
    finish_bonus: float = 100.0         # terminal success bonus (mirror of fall_penalty)

    # ----- joystick command objective ("command") ---------------------------------------------
    # Speed caps come from the hardware feasibility oracle (memory/hardware-speed-ceiling.md), not
    # from what the sim will tolerate: ~3.7 m/s at a credible cadence is BURST capability, and the
    # continuous-torque (thermal) ceiling is 1.8 m/s. Since this is a demo the robot must be able
    # to hold indefinitely without cooking a motor, the commandable maximum IS the thermal number.
    # Everything the joystick can ask for is therefore thermally sustainable by construction.
    cmd_v_fwd_max: float = 1.8          # m/s forward at full stick (= the continuous-torque ceiling)
    cmd_v_back_max: float = 0.6         # m/s backward (backing up is a manoeuvre, not a gait)
    cmd_yaw_max: float = 1.0            # rad/s yaw at full stick (~57 deg/s; 3 m radius at 1.8 m/s)
    # curriculum START box: wide enough to be a real task, narrow enough that every command in it
    # is achievable from the warm-start policy on day one. set_cmd_scale interpolates start->max.
    cmd_v_fwd_start: float = 0.6
    cmd_v_back_start: float = 0.2
    cmd_yaw_start: float = 0.3
    # FIXED observation normalizers. These must NEVER track the curriculum — see env.set_cmd_scale.
    cmd_v_norm: float = 2.0             # obs = v_cmd / this (so full stick reads ~0.9, not 1.0)
    cmd_yaw_norm: float = 1.5
    cmd_zero_prob: float = 0.25         # fraction of draws that are EXACTLY stand-still
    cmd_deadband: float = 0.12          # |v_cmd| below this snaps to 0 (a real stick has one too)
    cmd_yaw_deadband: float = 0.10
    cmd_resample_s: float = 4.0         # mean seconds between command changes (x U(0.7,1.3))
    # command-RANGE curriculum (train.CommandCurriculumCallback): widen the box only while the
    # policy tracks the TOP of it. Bidirectional — a box that proves too wide walks back down.
    cmd_curriculum_step: float = 0.1    # scale increment per widen/narrow
    cmd_curriculum_err_open: float = 0.22    # m/s mean top-of-range error that widens the box
    cmd_curriculum_err_close: float = 0.40   # m/s that narrows it (hysteresis: must exceed _open)
    cmd_curriculum_top_frac: float = 0.7     # "top of range" = |cmd| >= this x current max
    cmd_curriculum_gate_ep_len: float = 1200.0   # must also be surviving this long
    cmd_curriculum_min_samples: int = 200        # per-rollout top-range samples before judging
    # tracking reward. sigma is RELATIVE (see env._command_income): a fixed sigma systematically
    # under-rewards high-speed commands and the policy learns to prefer the slow end of the range.
    w_track_lin: float = 4.0
    w_track_yaw: float = 2.0
    track_sigma_rel: float = 0.25       # sigma = max(sigma_min, this * |cmd|)
    track_sigma_min: float = 0.20       # m/s floor (also the sigma used at cmd = 0)
    track_yaw_sigma_min: float = 0.25   # rad/s floor
    w_stand: float = 3.0                # stand-still income when the stick is centred
    stand_sigma: float = 0.15           # m/s width of the 'stationary' kernel
    w_stand_drift: float = 2.0          # penalty per m^2 of drift from where the stand began
    stand_drift_free_m: float = 0.10    # free drift radius (stepping in place moves the base a bit)

    # ----- steering (the L/R asymmetry channel; see fourier_gait.assemble) -----------------------
    # A mirror-symmetric gait can only run straight, so turning needs the mirror broken. Two knobs,
    # both part of the gait SPEC (a turn is a gait change, so coef_rate should bill it).
    # OPT-IN: enabling this adds 2 action dims (24 -> 26), so every checkpoint trained without it
    # is incompatible. Off by default, on in the teleop presets — the m1..m7 lineage is untouched.
    steer_enable: bool = False
    # MEASURED authority (open-loop, hand-built gait, roll+pitch railed, 2026-07-30):
    #   stride channel at scale 0.35 -> only 0.26 rad/s of yaw range, and NON-MONOTONIC
    #   width  channel at scale 0.12 -> 1.70 rad/s of yaw range
    # i.e. the hip-roll (width) channel is ~6.6x stronger per unit scale, and my original 0.35/0.12
    # split had it backwards. It also explains the trained policy's inability to turn right: the
    # stride channel's TOTAL authority (0.26) was smaller than the plant's intrinsic yaw bias
    # (~0.4 rad/s, see dr_loop_site), so the strong channel was throttled and the weak one could not
    # even cancel the bias. Physically the width channel wins because differential hip roll changes
    # the lateral moment arm of the ground reaction about the vertical axis, which is a direct yaw
    # couple, whereas differential stride mostly changes fore-aft foot placement.
    # NOTE the hip_roll joint axes are NOT mirrored L/R in this model (both +X), so in JOINT space
    # +u/-u = opposite physical rotations and +w/+w = the same physical rotation. See assemble().
    # Tuned so a full-scale (1.0 rad/s) yaw command needs ~half deflection: enough headroom to
    # cancel the intrinsic bias and still command a turn, without the whole range living in the
    # first 30% of the stick (which would make yaw twitchy and hard to hold).
    #   width 0.15 -> +-1.1 rad/s (92% deflection, no headroom)
    #   width 0.18 -> +-2.1 rad/s (47% deflection)  <- chosen
    #   width 0.22 -> +-3.3 rad/s (30% deflection, twitchy)
    steer_stride_scale: float = 0.20    # +-fraction of differential stride amplitude at full steer
    steer_width_scale: float = 0.18     # rad of differential hip roll at full steer

    # ----- observation -----
    # per-frame: motor pos/vel/torque (6+6+6) + gravity (3) + gyro (3) + base velocity (3,
    # PRIVILEGED sim state — the quantity the reward maximizes must be observable; needs a
    # velocity estimator for hardware later) + [sin, cos] gait phase (2) + task channel (2:
    # [run_flag, dist_to_go/100]) + prev action (24) = 55. Stacked over history_len frames.
    history_len: int = 5
    # PRIVILEGED base linear velocity in the obs. True for every sprint/speed milestone (that is
    # how they were trained); the command presets set False, because on hardware this number does
    # not exist without a state estimator, and a policy that has only ever seen ground truth has
    # never seen the signal it will actually be handed. With it off, velocity has to be inferred
    # from the strided history below — which is why that history gets longer at the same time.
    obs_base_vel: bool = True
    # History STRIDE: expose every k-th of the last (history_len-1)*k+1 frames. At 200 Hz a
    # contiguous 5-frame history spans 25 ms, far too short to infer body velocity from leg
    # kinematics; len=10 x stride=4 spans 200 ms for the same obs width. 1 = contiguous (legacy).
    history_stride: int = 1
    obs_delay_steps: int = 0            # extra staleness on the MEASUREMENT (0 = off; the action
    #                                     delay already models inference+CAN, so this would
    #                                     double-count unless action_delay_steps is reduced to match)
    obs_scales: dict = field(default_factory=lambda: dict(
        motor_pos=1.0, motor_vel=0.1, motor_torque=0.01, gravity=1.0, ang_vel=0.25, base_vel=1.0))

    # ----- domain randomization (per episode; see domain_rand.PlantRandomizer) -------------------
    # All ranges are RELATIVE (+-fraction of nominal) so 0 disables an axis exactly and the model
    # stays byte-identical. dr_enable=False makes the whole module inert -> every pre-existing
    # preset trains exactly as before.
    dr_enable: bool = False
    # Competence-gated ramp on the WIDTH of every dr_* range (0 -> full). 0 steps = no ramp, full
    # width from the first episode (the legacy behaviour, and what teleop v1..v3 all did). That was
    # a mistake: ablating teleop_v3 showed DR alone cost 20x MTBF (196 s clean -> 4.8 s), far more
    # than trips (1.8x), sensor noise (1.5x) or pushes (1.0x, no measurable effect), and it was the
    # only adversity applied at full strength from step 0. Uses the same gate + retreat as the gait
    # ramps, so an over-wide plant distribution walks itself back instead of pinning the policy.
    dr_curriculum_steps: int = 0
    dr_mass_global: float = 0.12        # whole-robot mass scale (build + payload tolerance)
    dr_mass_body: float = 0.15          # per-link mass jitter (CAD vs as-built)
    dr_inertia: float = 0.25            # per-link inertia jitter ON TOP of the mass scaling. Wide
    #                                     on purpose: the CAD link inertials are still placeholders,
    #                                     and the speed oracle says this is the axis the answer is
    #                                     most sensitive to. Narrow it once measured values land.
    dr_com_offset: float = 0.02         # m, uniform per-body CoM offset (assembly + harness)
    dr_friction: float = 1.0            # >0 enables the friction draw (the range below)
    dr_friction_range: tuple = (0.5, 1.3)   # absolute sliding friction (nominal model value is 1.0)
    dr_kp: float = 0.20                 # servo position-gain scale (moteus tuning + temperature)
    dr_kv: float = 0.25                 # servo damping-gain scale
    dr_torque: float = 0.12             # torque headroom scale (bus voltage sag, thermal derate)
    dr_joint_damping: float = 0.30      # leg joint damping/friction (grease, wear, temperature)
    dr_ankle_k: float = 0.20            # PASSIVE ANKLE SPRING stiffness scale. The single highest-
    #                                     value axis here: the whole m3 balance result hinges on
    #                                     k=350, and a physical spring will not be built to spec.
    dr_ankle_damping: float = 0.35
    dr_gravity_tilt: float = 3.0        # deg; rotating gravity == tilting the world (slope proxy),
    #                                     and also absorbs IMU mount misalignment
    dr_delay_steps_range: tuple = (3, 6)    # per-episode action delay, in control steps
    # metres of per-episode jitter on the 4-bar loop-closure sites. The as-exported model has
    # leg_anchor_L/R differing by ~1 mm in x and z (should be identical; only y mirrors), which
    # makes the two linkages geometrically different and produces a measured ~0.4 rad/s INTRINSIC
    # yaw bias under a perfectly symmetric gait. Randomizing it (rather than symmetrizing the model,
    # which the real robot will not be either) turns the bias into something the policy must cancel
    # from the yaw-rate error every episode instead of a constant it can bake in. 0 = off.
    dr_loop_site: float = 0.0015        # +-1.5 mm, ~the size of the as-built asymmetry

    # ----- sensor noise (see domain_rand.SensorNoise) --------------------------------------------
    # Per-episode CONSTANTS (calibration you get wrong once and live with) + per-step white noise.
    obs_noise_enable: bool = False
    noise_encoder: float = 0.003        # rad, per-step encoder noise
    noise_encoder_offset: float = 0.01  # rad, per-episode zero offset per joint
    noise_motor_vel: float = 0.15       # rad/s, per-step (velocity is differentiated -> noisy)
    noise_motor_vel_bias: float = 0.05
    noise_torque: float = 1.5           # N*m, per-step (torque is estimated from phase current)
    noise_torque_bias: float = 1.0
    noise_torque_gain: float = 0.08     # multiplicative Kt error
    noise_grav: float = 0.02            # per-step noise on the unit gravity vector
    noise_grav_bias: float = 0.02       # per-episode IMU mount misalignment (~1.1 deg)
    noise_gyro: float = 0.02            # rad/s per-step
    noise_gyro_bias: float = 0.02       # rad/s per-episode
    noise_gyro_walk: float = 0.0002     # rad/s per step of bias random walk
    # Accelerometer leak into the gravity estimate — the one that matters most for this robot. On
    # hardware "down" comes from an attitude filter whose accelerometer cannot distinguish gravity
    # from body acceleration, so during push-off the measured gravity swings by several degrees
    # with no actual rotation. The FIXED pitch reflex reads grav_x directly, so a policy trained on
    # exact gravity has been trained on a signal the robot cannot produce. Drawn per episode in
    # [0, this] as a fraction of a/g.
    noise_accel_leak: float = 0.15

    # ----- trip / unseen-obstacle disturbance ----------------------------------------------------
    # A brief force opposing a SWING foot: "my toe caught on something". Trains recovery as a
    # behaviour rather than relying on a hand-written detector that has to fire correctly.
    trip_prob: float = 0.0              # per control step (0 = off); ~0.001 at 200 Hz = one per 5 s
    trip_force_range: tuple = (30.0, 90.0)  # N opposing travel
    trip_duration_s: float = 0.05

    # ----- action: per-step Fourier gait spec + residuals (see fourier_gait.py) -----
    # The policy re-emits the whole gait spec (cam+thigh Fourier coeffs + frequency + abduction
    # reflex gains) at EVERY 50 Hz control step — it can rewrite the gait instantly, mid-cycle
    # rewrites priced by the phase-gated w_coef_rate penalty (free at the cycle boundary) — plus
    # 6 per-step residual target corrections (the PMTG fast-feedback channel; NOT coef_rate-billed).
    n_harmonics: int = 3                # Fourier harmonics per joint (coeffs/joint = 1 + 2N)
    gait_freq_hz: tuple = (0.5, 3.0)    # learnable cadence range (Hz)
    # max Fourier DEVIATION (rad) of cam/thigh from the nominal stance posture. Raised from the
    # walking-era 0.30/0.35 toward the empirically-valid ctrl band (cam ~[-0.6,0.6]): top speed is
    # ~ 2 * step_length * cadence, and the Cassie 100 m result says the speed headroom lives in
    # STRIDE LENGTH (long strides + flight), not cadence. The env's ctrl-range clip is the final
    # guard that keeps the coupled 4-bar assemblable.
    cam_amp: float = 0.45
    thigh_amp: float = 0.45
    reflex_kp_scale: float = 0.5        # abduction reflex: hip_roll += kp*roll (rad per rad)
    reflex_kd_scale: float = 0.1        #                 + kd*roll_rate (rad per rad/s)
    reflex_bias_scale: float = 0.2      #                 + bias (rad, lateral stance offset)
    # FIXED pitch-stabilizing reflex (fourier_gait.assemble): a symmetric fore-aft foot shift
    # thigh_L += u, thigh_R -= u with u = -clip(kp*grav_x + kd*pitch_rate + bias, +-clip). grav_x
    # ~ sin(pitch), + = nose-down; -sign moves both feet toward the fall to catch the CoM (capture-
    # point / CoP matching: kp~=h_CoM/leg~=1, kd~=sqrt(h/g)~=0.3). ALWAYS active + pure numpy, so it
    # also stabilizes the real robot from the IMU. Inert on m1/m2 (pitch railed -> grav_x~=0), so
    # obs/action dims are unchanged and those checkpoints still load. clip=0 disables it entirely.
    # gains tuned 2026-07-21 (gain grid on the m3 plant + m2 policy): kd>=0.3 oscillates through
    # the 1-step delay + EMA filter; clip>0.35 fights the gait. kp=2/kd=0.2/clip=0.25 arrests a
    # pitch kick ~2.5-7x better than no reflex (both directions) without exciting oscillation.
    # The reflex is a stabilizing PRIOR, not a standalone controller — the plant needs an active
    # gait for height too, so it can't stand passively; it takes early-retrain episodes from
    # ~18 to ~80 steps of survival, giving the policy signal to learn a pitch-neutral gait.
    pitch_kp: float = 2.0               # rad thigh offset per unit grav_x (~sin pitch)
    pitch_kd: float = 0.2               # rad thigh offset per rad/s pitch rate (gyro y)
    pitch_clip: float = 0.25            # authority cap (rad); 0 = reflex off
    pitch_bias: float = 0.0             # static fore-aft trim added inside the clip (rad)
    pitch_cam_gain: float = 0.0         # reserved: cam coupling of the pitch reflex (keep 0)
    # pitch-reflex rate low-pass (2026-07-29): measured on m7_freq, the reflex D-term (kd*pitch_rate)
    # DOMINATES the P-term 2.2x and saturates the reflex 81% of the time, flipping at 6.4 Hz = the
    # footfall. The body barely tilts (7.6 deg) but wobbles fast (pitch_rate rms 2.87) and the D-term
    # rectifies that gait-bob into a 6 Hz foot oscillation (a limit cycle no reward/torque lever
    # touches). EMA-filter the pitch_rate the reflex uses so it keeps its slow real-tilt damping but
    # stops responding to the fast bob. 0 = raw (off). alpha 0.9 @200Hz ~= 3.5 Hz cutoff.
    pitch_reflex_rate_lp: float = 0.0   # EMA alpha on the pitch_rate feeding the reflex (0 = off)
    # ANKLE-TORQUE reflex (2026-07-23): the sagittal-balance experiment. The robot's ankle (foot)
    # joints are PASSIVE SPRINGS (no actuator) -> no ankle pitch torque -> foot-placement is the only
    # balance channel, and it plateaus at ~1 s (see hz200-reactive-stack). This is a FIXED PD that
    # applies a pitch-restoring TORQUE at the ankle joints (only while the foot is GROUNDED = ankle
    # strategy), driven by base pitch/pitch-rate -- i.e. it emulates an ACTUATED ankle. Pure numpy
    # (ships to a Pi with a real ankle motor). Mirrored L/R axes -> +u on L, -u on R. kp=0 = off.
    ankle_kp: float = 0.0               # N*m ankle torque per rad of base pitch (grav_x)
    ankle_kd: float = 0.0               # N*m per rad/s of base pitch rate
    ankle_clip: float = 0.0             # N*m authority cap per ankle (plausible motor limit)
    # lower-CoM experiment (2026-07-23): the CHEAPEST candidate hardware fix for the m3 pitch-balance
    # failure (vs adding ankle motors). Shifts the base body's inertial CoM down by this many metres
    # at model load (bottom-heavy -> longer fall time-constant -> foot placement can balance it).
    # 0 = unchanged. Tests, in sim, how much CoM-lowering the current-actuator plant needs to balance.
    com_lower: float = 0.0              # metres to lower the base-body CoM (body_ipos z)
    # ankle-spring STIFFNESS experiment (2026-07-24): the ankle (foot) joints are passive springs
    # (k=28.65 N*m/rad) and provide almost no stance pitch-restoring torque -> all foot-placement
    # authority plateaued m3 at ~1 s. A STIFFER spring resists ankle deflection harder = more
    # passive pitch-restoring moment in stance (the balance channel the plant lacks). CRUCIAL: the
    # standing ankle sits at +-0.25 rad but springref is +-0.7, so the spring is PRELOADED ~12.8 N*m
    # at stance; naively raising k balloons that preload (-> foot slams to springref, robot topples,
    # verified). So env.py ALSO shifts springref to PRESERVE the standing preload -> only the
    # restoring GAIN (stiffness) rises, posture unchanged. 0 = keep the model's 28.65.
    ankle_stiffness: float = 0.0        # N*m/rad on the passive ankle springs (0 = keep model 28.65)
    ankle_damping: float = 0.0          # N*m*s/rad ankle joint damping (0 = keep model 0.3); raise
    #                                     with stiffness to keep the stiffer spring ~critically damped
    # ----- ankle-spring STUDY (2026-08-03) ------------------------------------------------------
    # The 2026-07-24 sweep only ever compared SOFTER-vs-STIFFER passive springs, warm-started, one
    # seed, and (as m7 later found) under-damped throughout -- so it never answered whether the
    # spring is needed at all, whether an ACTUATED ankle would beat it, or where the optimum sits.
    # This axis makes the plant's ankle a first-class experimental variable:
    #   "passive"      k = ankle_stiffness (or the model's 28.65 when 0). The status quo.
    #   "free"         k = 0 EXACTLY -- a floppy, undriven ankle. The null that ankle_stiffness
    #                  alone cannot express, because 0 there means "keep the model value".
    #   "rigid"        ankle welded at its stance angle via the lock_ankle_L/R equalities. The
    #                  other null: no compliance, no actuation, just a stiff foot.
    #   "active"       policy-driven ankle servo, NO spring (k=0). Needs the +2-actuator plant
    #                  (model/dash01_measured_active.xml) -- action/obs widen, own lineage.
    #   "active_spring" policy-driven ankle servo IN PARALLEL with the spring (parallel-elastic).
    # NOTE "rigid" needs the lock_ankle equalities and the active modes need the 8-actuator model;
    # env.py raises if model_path lacks them, rather than silently running the wrong arm.
    ankle_mode: str = "passive"
    # Damping as a DAMPING RATIO instead of an absolute number. The 2026-07-24 sweep hand-picked
    # ankle_damping per arm (1.6 at k=350 = ~9% of critical) and the resulting spring ring became
    # the m7 6 Hz cadence bug -- i.e. the k-curve was measured with damping varying independently
    # of k, confounding the very thing being swept. With ankle_zeta > 0 the env computes
    # damping = 2*zeta*sqrt(k*I) from the ankle's EFFECTIVE inertia, so every k in the sweep sits
    # at the same point on the damping curve and k is the only thing that changes. Overrides
    # ankle_damping. 1.0 = critical; 0.7 is the usual "fast without ringing" choice.
    ankle_zeta: float = 0.0
    # Active-ankle command range, rad about the settled stance angle. The servo's ctrlrange is the
    # joint's full +-1.047, but handing the policy the whole range makes early exploration slam the
    # foot; this scales the action into a sane authority band, mirroring residual_scale.
    ankle_action_scale: float = 0.5
    # Bill the ankle motors in the torque/energy penalty like any other actuator. Off would let the
    # active arm buy stability with unpriced energy and win the comparison for the wrong reason.
    ankle_torque_billed: bool = True
    # ----- action mode: which gait generator turns the action into PD targets ---------------
    # "fourier" = fourier_gait.py (the m1..m7 lineage). "cpg" = cpg_gait.py, the Ijspeert-school
    # alternative: two coupled amplitude-controlled phase oscillators whose parameters (amplitude,
    # frequency, inter-leg phase bias) are what the policy emits, with a FIXED measured foot-IK
    # mapping from oscillator to joints. The two modes have different action AND obs widths, so
    # checkpoints never cross between them — every comparison is same-generator warm-start only.
    action_mode: str = "fourier"        # "fourier" | "cpg"
    cpg_a: float = 50.0                 # amplitude convergence rate (r -> mu, critically damped).
    #                                     ~5x the fastest gait frequency: fast enough to change
    #                                     stride within a step, slow enough to filter action chatter
    cpg_coupling: float = 8.0           # inter-leg phase coupling weight (rad/s per unit amplitude)
    cpg_psi_range: float = 0.6          # rad the policy may shift the antiphase target by
    cpg_mu_min: float = 0.0             # amplitude setpoint range; r=0 is a valid "stand still"
    cpg_mu_max: float = 1.0
    cpg_stride: float = 0.28            # m of fore-aft foot travel at r=1 (LUT box is +-0.30)
    # m of swing-phase foot lift at r=1. Bounded by the MEASURED workspace, not by taste: the
    # standing leg is near-straight, so at mid-stride (dx ~ 0, where the swing bump peaks) only
    # ~0.083 m of lift exists at all. 0.06 keeps the whole trajectory inside the reachable shell.
    cpg_clearance: float = 0.06
    cpg_substeps: int = 4               # oscillator integration substeps per control step
    # which measured foot-IK table (training/model/) the mapping uses. Per-preset so a new arm can
    # ship a wider-stride table without changing the mapping under an arm already training.
    cpg_lut: str = "cpg_foot_lut.npz"
    cpg_residual: bool = True           # False = the no-residual ablation arm (pure CPG output)
    residual_scale: float = 0.08        # rad of per-step correction authority on each PD target
    action_scale: float = 0.5           # normalization for the action_rate term's motor_cmd units
    action_filter: float = 0.2          # EMA smoothing of targets (0 = off); helps sim2real
    action_delay_steps: int = 1         # fixed actuation delay in control steps (Pi+CAN plant truth)
    # ----- motor velocity / acceleration limits (2026-07-24; sim2real + cadence) -----
    # The position servos have NO velocity or acceleration cap in the model (only kv damping + a
    # SOFT w_motor_vel reward), so the 200 Hz policy can crank the legs arbitrarily fast -> the k350
    # winner balanced m3 by pattering ONE foot at ~11 Hz (peak thigh 23 rad/s, at the 210 RPM motor
    # ceiling). These slew-limit the COMMANDED target so joint velocity <= motor_vel_limit and its
    # rate of change <= motor_accel_limit -- a velocity/accel-bounded position servo, exactly the
    # real moteus position-mode limits. 0 = off (plant byte-identical to before). NOTE 22 rad/s is
    # the MOTOR ceiling; the cam/thigh see it through the linkage reduction, so this is a loose upper
    # bound (only trims the worst spikes) -- the accel limit + cadence penalty do the real slowing.
    motor_vel_limit: float = 0.0        # rad/s cap on commanded joint velocity (210 RPM motor=22)
    motor_accel_limit: float = 0.0      # rad/s^2 cap on commanded joint acceleration (0 = off)

    # ----- sim2real control-timing randomization (2026-07-23; models the Pi inference loop) -----
    # The real Pi control loop has ms-scale timing jitter and occasionally MISSES an inference
    # deadline (the moteus then holds the last command). We randomize both in sim so the policy is
    # robust to them. Curriculum: HOLD at 0 until the gait is competent (GatedRampCallback on
    # ep_len), THEN ramp jitter/drop in. sim_dt = 1 ms, so jitter_ms == jitter in sim substeps.
    ctrl_jitter_ms: float = 0.0         # current +-uniform jitter (ms) on the substeps/control step
    ctrl_drop_prob: float = 0.0         # current prob a control step is DROPPED (hold last action)
    ctrl_jitter_ms_final: float = 0.0   # ramp target for the jitter (0 = off)
    ctrl_drop_prob_final: float = 0.0   # ramp target for the drop prob (0 = off)
    jitter_curriculum_gate_ep_len: float = 0.0   # ep_len that opens the jitter/drop ramp
    jitter_curriculum_steps: int = 0    # env steps to ramp jitter/drop 0 -> final after the gate

    # ----- pitch-assist curriculum (m2->m3 bridge, 2026-07-22) -----
    # The balance-first easing sweep plateaued at ep_len ~67 (the passive-collapse time): even with
    # NO flight demand + efficiency OFF, the m3 policy can't discover a pitch-stable height-holding
    # gait from the m2 (pitch-LOCKED) warm-start. This is a soft, DECAYING training-wheel: an
    # external PD torque on the base pitch JOINT (spring-damper toward level, applied to
    # qfrc_applied) holds the body near upright early — like m2's lock but compliant — so the robot
    # can keep/relearn the height-holding gait, then the SCALE fades 1->0 over pitch_assist_ramp_steps
    # so the final policy balances pitch itself (assist=0 => hardware-valid; the assist is sim-only).
    # Only meaningful when pitch is FREE. kp=0 (default) disables it entirely -> every other preset
    # and eval is byte-identical.
    pitch_assist_kp: float = 0.0        # N*m per rad of base pitch angle (restoring toward level)
    pitch_assist_kd: float = 0.0        # N*m per rad/s of base pitch rate (damping)
    pitch_assist_ramp_steps: int = 0    # env steps to linearly fade the assist SCALE 1 -> 0; 0 = off
    # anti-crutch (round 4, 2026-07-23): a decaying assist ALONE breeds crutch-dependence — the
    # policy maxes out the wheel at every fade level (assist=0 eval collapsed to ep_len 14-50) since
    # nothing rewards NOT needing it. This penalizes the assist torque the policy provokes: balance
    # well (pitch ~ 0 -> assist exerts ~0 N*m) and pay nothing; lean on the wheel (large restoring
    # torque) and pay. Decouples the dense fall-catching safety net from a reward for SELF-balance,
    # so the policy learns pitch control WHILE the net is still there. Squared, _pen-capped. 0 = off.
    w_assist_penalty: float = 0.0       # reward penalty per (N*m)^2 of assist torque the policy causes
    # pitch SLOW-MOTION curriculum (round 5, 2026-07-23): every assist above HOLDS the body up ->
    # crutch (rounds 2-4 all collapsed at assist=0). This instead adds extra ARMATURE (rotor inertia)
    # to the base pitch DOF: the fall becomes sluggish (longer time constant) so the 50 Hz policy has
    # time to learn foot-placement balance, then the extra inertia fades to 0 (real dynamics). It
    # NEVER holds position -> NO crutch: the robot balances itself throughout, so the in-training
    # ep_len is already the honest unaided survival. Ramp fades the extra armature 1 -> 0. kp... = 0
    # off. Base pitch inertia ~0.62 kg m^2, so pitch_armature ~2-3 roughly halves the fall rate.
    pitch_armature: float = 0.0         # extra kg*m^2 on the base pitch DOF at scale 1
    pitch_armature_ramp_steps: int = 0  # env steps to fade the extra armature 1 -> 0; 0 = off

    # ----- reward: caps & terminal -----
    # Two-level suicide-proofing (reward normalization is OFF — raw scales reach PPO directly):
    # each penalty term is floored at -penalty_term_cap (keeps per-term gradients), AND the
    # per-step TOTAL is floored at -step_reward_floor before terminal bonuses. Per-term caps
    # alone are not enough: the SUM of always-on run-phase penalties for a standing robot was
    # ~-2.2/step, whose discounted value (-220 at gamma 0.99) made diving (-100) value-optimal
    # for every pre-locomotion policy (verified, review 2026-07-17). With the total floored at
    # -1.0/step, living forever (~-100) never loses to dying (-100), and any income tips it.
    penalty_term_cap: float = 2.0
    step_reward_floor: float = 1.0      # per-step total floored at -this, pre-terminal
    fall_penalty: float = 100.0

    # ----- reward: anti-skate gait shaping (sim-side, reward-only) -----
    w_foot_slip: float = 8.0            # horizontal toe speed while grounded, quadratic > deadband
    slip_deadband: float = 0.05         # m/s of tolerated in-contact toe motion
    w_stance_time: float = 0.5          # per-foot stance-time cap: forces every foot to cycle
    stance_cap_s: float = 0.7
    stance_cap_slow_s: float = 1.0
    stance_slow_speed: float = 0.4
    w_clearance: float = 0.4            # fresh-swing height credit (gradient bridge for lift-off)
    clearance_dead_m: float = 0.02
    clearance_scale_m: float = 0.03
    swing_fresh_s: float = 0.45
    gait_cmd_gate: float = 0.25         # m/s of commanded speed above which gait terms engage
    w_air_time: float = 2.0             # one-sided capped touchdown credit
    foot_air_time_min: float = 0.25
    air_credit_cap_s: float = 0.45
    grounded_h: float = 0.005           # sphere-bottom height that still counts as grounded
    # foot-placement-ahead-of-CoM reward (2026-07-24): reward planting the stance foot AHEAD of the
    # whole-robot CoM (in the +x heading; yaw is locked m3..m5) at TOUCHDOWN -- the capture step
    # that catches a forward fall and the natural running footfall. Credited ONLY on the air->ground
    # transition (a transient EVENT, not a held pose) so it can't be farmed by standing with both
    # feet forward (which would breed a backward lean). Forward offset capped at foot_ahead_cap_m.
    # Logged as reward_terms/foot_ahead so the plots show whether the policy actually uses it. 0 = off.
    w_foot_ahead: float = 0.0           # reward per metre the touchdown foot lands ahead of the CoM
    foot_ahead_cap_m: float = 0.20      # cap on the rewarded forward offset (m)
    # cadence / anti-chatter (2026-07-24): penalize foot contact-state CHANGES per control step ->
    # fewer, longer steps (minimise stepping frequency). Balanced against phase_contact (which
    # DEMANDS swing) so it slows cadence without collapsing to a skate. Logged as reward_terms/
    # step_rate. 0 = off.
    w_contact_switch: float = 0.0       # penalty per foot that flips grounded<->airborne this step
    # duty-SYMMETRY / anti-one-legged (2026-07-31): the slow_gait lineage still converged to a ONE-
    # LEGGED gait (m3_slow_gait: left foot duty 0%, right foot patters 6.4 Hz, asymmetry 0.99) -- the
    # freq-map + ankle-damping + gentle contact-switch fixes slowed the CLOCK but did not stop the
    # policy from using one leg as its whole support+propulsion+balance actuator while the other is
    # just carried. This penalizes any foot whose STANCE-DUTY (grounded fraction, EMA over
    # duty_sym_tau_s) sinks below duty_floor -> a foot that never bears load pays, forcing a two-legged
    # gait. Linear in the deficit (non-vanishing gradient), _pen-capped. Logged as reward_terms/
    # duty_sym. 0 = off (byte-identical to every pre-existing preset). Pairs with a stronger
    # w_contact_switch (symmetry alone could be met by both feet chattering fast; the switch penalty
    # keeps it slow).
    w_duty_sym: float = 0.0             # penalty weight per unit of summed per-foot duty deficit
    duty_floor: float = 0.30           # min per-foot grounded-fraction (duty) before the penalty bites
    duty_sym_tau_s: float = 1.0        # EMA time constant (s) for the per-foot duty estimate
    # workspace-KILL termination (2026-08-04): the one-legged gait parks a leg FOLDED, with the toe
    # OUTSIDE the measured real-robot foot workspace (visible in the slow_gait videos) — a sim-only
    # exploit the physical 4-bar cannot do. Terminate the episode (fall_penalty) when either foot's
    # toe leaves the measured reachable box, sustained for workspace_grace_s. The box is in the BASE
    # frame relative to the LUT nominal_toe (cpg_foot_lut.npz): dx = fore-aft toe travel, dz = lift
    # (dz>0 = foot up / shorter leg). Measured envelope is dx +-0.30 m, dz 0..0.10 m (folded branch
    # past dz>0.40). Unlike the soft duty/cadence penalties this makes the exploit IMPOSSIBLE rather
    # than merely taxed. 0/False = off (byte-identical to every existing preset).
    workspace_kill: bool = False
    workspace_dx_max: float = 0.34      # fore-aft toe travel ceiling (m); measured box 0.30 + margin
    workspace_dz_max: float = 0.14      # lift ceiling (m); measured box 0.10, fold at 0.40
    workspace_dz_min: float = -0.18     # extension floor (m); generous, don't kill push-off
    workspace_grace_s: float = 0.10     # a foot must be outside continuously this long to terminate

    # ----- reward: phase-gated contact schedule (Siekmann-style; NEW) -----
    # The gait clock the ACTION uses is also the reward's contact schedule: each foot pays for
    # being grounded during its expected SWING window (left window = phase in [0, 2pi*sr); right
    # = antiphase). stance_ratio ramps DOWN over training: below 0.5 the two swing windows
    # overlap -> both feet penalized for ground contact at once -> a flight phase is demanded.
    # This is the one term that explicitly asks for RUNNING rather than fast walking.
    w_phase_contact: float = 1.0
    stance_ratio_start: float = 0.65    # walking duty factor (curriculum start)
    stance_ratio_final: float = 0.42    # running duty factor (< 0.5 = flight window exists)
    gait_curriculum_steps: int = 60_000_000   # env steps to ramp stance_ratio over; 0 = hold start
    # COMPETENCE-GATED curriculum (2026-07-22): the stance_ratio + efficiency ramps are otherwise
    # CLOCK-driven from step 0 — they harden toward flight-phase running + full efficiency whether
    # or not the robot ever learned to balance the freed pitch DOF, so an m3 policy that can't yet
    # hold height is perpetually asked to run and collapses at the passive-fall time (~60 steps).
    # Same failure class as the entropy-gate deadlock. With this > 0, BOTH ramps HOLD at their easy
    # start (stance_ratio_start, eff_scale 0) until rollout ep_len_mean exceeds it for `patience`
    # rollouts, THEN ramp to target over gait_curriculum_steps / efficiency_ramp_steps from the
    # open step. 0 = the old clock-driven behavior. A gate that never opens = easy regime forever
    # (the correct failure mode: keep it upright rather than force it to run and fall).
    curriculum_gate_ep_len: float = 0.0
    # BIDIRECTIONAL ramp (2026-07-31). Competence-gating the ramp's OPEN but not its ADVANCE only
    # delays the failure it was meant to prevent: teleop_v2's gate opened correctly at ep_len 1855,
    # then the ramp advanced on a clock for 215 M more steps while the policy collapsed to 725 —
    # nothing was watching. With this > 0, progress along start->target ADVANCES only while
    # ep_len >= gate, HOLDS in the band, and RETREATS while ep_len < this * gate, so a curriculum
    # that turns out to be too ambitious walks itself back instead of grinding the policy down.
    # 0 = the legacy one-way clock ramp (every m1..m7 preset keeps it, bit for bit).
    curriculum_retreat_frac: float = 0.0
    # Ramp target for the efficiency terms. Was hard-coded to 1.0 in train.py; the sprint lineage
    # wants full efficiency pressure, a walking demo does not (teleop_v2 peaked at eff_scale 0.00
    # and degraded monotonically as this climbed).
    efficiency_target: float = 1.0

    # ----- reward: efficiency (Cassie-100m recipe; annealed IN over efficiency_ramp_steps) -----
    # Weighted to comparable magnitudes at running speed so the optimizer trades them off, which
    # is what produced long-stride human-like running instead of frantic-cadence thrash. Ramped
    # from 0 so they cannot smother gait emergence early (a known failure mode).
    w_torque: float = 1.0e-4            # torque ABOVE the standing baseline, squared (stance is free)
    w_motor_vel: float = 1.0e-4         # sum motor qvel^2 (the motor-velocity cost)
    w_energy: float = 2.0e-4            # sum positive mechanical power |tau*qvel|_+ (CoT proxy)
    efficiency_ramp_steps: int = 60_000_000   # linear 0 -> 1 multiplier on the three terms above

    # ----- torque-budget curriculum (2026-07-28; cadence fix, take 2) -----
    # The m3 controller uses only ~18% of the available actuator torque on average, so it has huge
    # headroom to waste on high-frequency foot dither. This curriculum SHRINKS the torque budget
    # (scales the actuator forcerange down) while the policy stays competent (ep_len gate), forcing
    # a torque-efficient = smoother, lower-cadence gait. It STOPS training when the actuators are
    # WELL SATURATED (mean |tau|/limit >= torque_util_target) -- once the budget is the binding
    # constraint there is no point tightening further. Torque is a SOFTER, physically-grounded lever
    # than the velocity/accel HARD caps (m3_cad*) that collapsed the warm-started policy. 0 = off.
    torque_util_target: float = 0.0     # target/stop mean torque utilization (0 = off; 0.70 = the ask)
    torque_limit_gate_ep_len: float = 0.0   # only tighten the budget while rollout ep_len >= this
    torque_limit_step: float = 0.01     # forcerange-scale decrement per tighten (cooldown-paced)
    torque_limit_floor: float = 0.15    # minimum torque scale (safety floor)

    # ----- reward: smoothness & posture -----
    w_action_rate: float = 0.1          # on the reconstructed normalized motor targets
    w_coef_rate: float = 0.5            # phase-gated gait-SPEC change penalty: billed *sin^2(phi/2),
    #                                     free at the cycle boundary (spec dims only, not residuals)
    w_residual: float = 0.1             # keep the Fourier prior dominant: residuals are for
    #                                     corrections, not for becoming the controller
    # residual RATE penalty (2026-07-29): the freq fix slowed the CPG clock to ~1 Hz but the FOOT-
    # FALL stayed ~6 Hz -- the policy steps through the fast residual channel (thigh residual flips
    # sign ~8 Hz), not the clock. Penalize the per-step residual CHANGE (Delta-residual^2) so a
    # high-frequency residual oscillation is expensive while a one-off correction stays cheap. This
    # is the "re-bill the residual as a rate" fix, targeted at the chatter the freq fix left behind.
    w_residual_rate: float = 0.0        # penalty per sum((residual - prev_residual)^2); 0 = off
    w_upright: float = 5.0
    w_height: float = 2.5               # only when Z is free (neutralized on the rail)
    w_vz: float = 0.5
    w_lat_vel: float = 1.0              # body-frame lateral velocity (go straight)
    w_angvel_xy: float = 0.05
    # centroidal angular-momentum regulation (the "impulses average out" idea = capture-point /
    # MPC-style momentum control): penalize whole-robot angular momentum about the CoM
    # (mj_subtreeVel -> subtree_angmom), pitch component only while yaw/roll are locked (m3..m5).
    # Sim-only reward (no hardware estimator needed). I_pitch(CoM) ~ 0.62 kg m^2: a 2 rad/s nose-
    # dive is L_y ~ 1.3 (costs ~0.34 at w=0.2), healthy leg-swing residual ~0.5-1.0 stays < 0.2.
    # It also prices cyclic swing momentum, so if plots show it fighting air_time, halve it.
    # 0 = off (m1/m2 pay nothing and skip the mj_subtreeVel call); m3..m6 presets set 0.2.
    w_angmom: float = 0.0
    w_no_cross: float = 50.0            # one-sided stance-width penalty (legs must not scissor)
    stance_min_sep: float = 0.25        # m; nominal stance is ~0.40
    w_hip_roll: float = 3.0             # keep abduction near neutral

    # ----- episode / termination -----
    episode_s: float = 60.0
    term_height: float = 0.45
    term_gravity_z: float = -0.5        # tipped past ~60 deg
    reset_joint_noise: float = 0.03
    push_interval_s: float = 0.0        # random base shoves (training disturbance); 0 = off
    push_dv: float = 0.4

    # ----- PPO -----
    n_envs: int = 8
    total_steps: int = 240_000_000
    n_steps: int = 1024
    batch_size: int = 4096
    n_epochs: int = 4
    # gamma 0.99 (2 s horizon at 50 Hz): long enough to price falling + the dash structure,
    # short enough that farming the stop-phase income (w_stop_vel/(1-gamma) = 200 gross, cut off
    # by the 1 s hold-detector termination anyway) cannot out-value the finish bonus.
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3.0e-4       # linearly annealed to lr_final over the run
    lr_final: float = 1.0e-4
    clip_range: float = 0.2
    target_kl: float = 0.03
    # entropy: hold ent_coef until stepping has emerged (reward_terms/air_time > gate), THEN anneal
    # to ent_final. A fixed low value collapses std onto a skating optimum; a fixed high value
    # farms the clipped-Gaussian entropy bonus — which is also why log_std is clamped.
    ent_coef: float = 0.01
    ent_final: float = 0.002
    ent_anneal_steps: int = 20_000_000
    ent_gate_air_time: float = 0.02
    ent_anneal_deadline_steps: int = 0  # hard fallback: begin the ent_coef anneal by this many env
    #                                     steps even if the air_time competence gate never opens
    #                                     (num_timesteps-based; 0 = disabled, gate-only). Without it
    #                                     a stuck gate pins std at max_log_std forever (m3 deadlock).
    max_log_std: float = 0.0            # std <= 1.0: beyond the clipped range is pure farming
    seed: int = 0
    policy_hidden: List[int] = field(default_factory=lambda: [256, 256])

    # ----- warm-start VecNormalize rejuvenation (milestone hops only; see train.py) -----
    # A milestone that frees a base DOF makes a previously-CONSTANT obs dim start varying. The
    # warm-started VecNormalize carries near-zero variance on that dim (var ~ 1e-8) and a giant
    # running count (~ source total_steps), so the new signal normalizes to O(1e3), clips at +-10
    # (binarized), and the stats adapt only glacially. On --warm-start we therefore cap the count
    # (fresh data reaches equal weight within ~count_cap samples) and floor the variance (a rail-
    # locked dim becomes readable from batch one: pitch 0.2 rad -> normalized 2.0, not clipped 10).
    warmstart_obs_count_cap: float = 50_000.0   # 0 = leave obs_rms.count untouched
    warmstart_var_floor: float = 1.0e-2         # 0 = leave obs_rms.var untouched


# ----- presets ---------------------------------------------------------------------------
def _sprint(**kw) -> Config:
    base = dict(objective="sprint", w_alive=0.0)
    base.update(kw)
    return Config(**base)


def _speed(**kw) -> Config:
    """Endless max-speed debug objective: no line, no clock, small alive bonus, 20 s episodes."""
    base = dict(objective="speed", w_alive=0.5, episode_s=20.0,
                sprint_curriculum_steps=0)
    base.update(kw)
    return Config(**base)


LOCKS = {                       # [X, Y, Z, roll, pitch, yaw], 1 = locked
    "m1": (0, 1, 1, 1, 1, 1),   # only X free; Z railed (random ride height)
    "m2": (0, 1, 0, 1, 1, 1),   # + Z: carries its own ride height
    "m3": (0, 1, 0, 1, 0, 1),   # + pitch: fore/aft attitude is live
    "m4": (0, 0, 0, 1, 0, 1),   # + Y (lat_vel keeps it straight)
    "m5": (0, 0, 0, 0, 0, 1),   # + roll: the abduction reflex becomes live
    "m6": (0, 0, 0, 0, 0, 0),   # fully free
}


def _extras(m):
    """Per-milestone extra kwargs on top of the base_lock: m1 rails Z at a random ride height;
    milestones with pitch FREE (m3..m6, lock[4]==0) turn on the angular-momentum regulation term."""
    kw = {}
    if m == "m1":
        kw["z_rail_randomize"] = True
    if LOCKS[m][4] == 0:                 # pitch free (m3..m6): angmom + anti-topple/anti-deadlock
        kw["w_angmom"] = 0.2             # regulate centroidal angular momentum
        kw["ent_anneal_deadline_steps"] = 25_000_000   # break the entropy-gate deadlock (2026-07-22)
        kw["speed_upright_gate"] = True                # kill the lunge-for-speed exploit
    return kw


def _mk_sprint(m):
    lock = LOCKS[m]
    return lambda: _sprint(base_lock=lock, **_extras(m))


def _mk_speed(m):
    lock = LOCKS[m]
    return lambda: _speed(base_lock=lock, **_extras(m))


PRESETS = {**{f"{m}_sprint": _mk_sprint(m) for m in LOCKS},
           **{f"{m}_speed": _mk_speed(m) for m in LOCKS},
           "default": Config}


# ----- m3 anti-topple sweep (2026-07-22) -------------------------------------------------
# m3 (X+Z+pitch free) keeps collapsing at the passive-fall time (ep_len ~60 = 1.2 s) with a hard
# forward lunge (peak vx ~3.7 m/s): the moment pitch is freed the robot must learn active balance,
# but the clock-driven curricula simultaneously demand a flight-phase running gait (stance_ratio
# 0.42) + full efficiency + high speed, all of which fight "extend legs, hold height, don't tip".
# m2 (same plant, pitch LOCKED) trains fine — so the fix is: establish a height-holding, pitch-
# balanced gait in an EASY regime first, then harden. Each preset below attacks one lever; all
# warm-start from m2_sprint/ppo_180000000_steps.zip (dims are identical across milestones).
def _m3_speed(**kw):
    """m3 endless-speed variant: m3 lock + the standard m3 extras (angmom, ent deadline, upright
    gate), then the sweep overrides on top."""
    base = dict(base_lock=LOCKS["m3"])
    base.update(_extras("m3"))
    base.update(kw)
    return _speed(**base)


PRESETS.update({
    # R1 — balance-first WALKING: never demand a flight phase (stance >= 0.5), modest top speed so
    # it can't lunge, efficiency held near 0 the whole run. The cleanest "just walk and stay up".
    "m3_walk": lambda: _m3_speed(
        stance_ratio_start=0.65, stance_ratio_final=0.60, gait_curriculum_steps=40_000_000,
        v_ceiling=1.2, efficiency_ramp_steps=250_000_000),
    # R2 — competence-GATED curriculum toward the full running target: hold easy until it survives,
    # then harden. The principled, scalable fix (reused for m4+ if it works).
    "m3_gated": lambda: _m3_speed(
        curriculum_gate_ep_len=400.0, v_ceiling=2.0,
        stance_ratio_start=0.65, stance_ratio_final=0.42, gait_curriculum_steps=40_000_000,
        efficiency_ramp_steps=40_000_000),
    # R3 — kill the LUNGE only, keep the running/flight demand: low speed cap + sharp uprightness
    # gate. Isolates whether the forward lunge (not the gait demand) is the killer.
    "m3_slow": lambda: _m3_speed(
        v_ceiling=1.0, w_fwd_speed=1.5, speed_upright_k=3.0, speed_upright_c0=0.7),
    # R4 — more corrective AUTHORITY: 2x residual fast-feedback + stronger/roomier pitch reflex,
    # cheaper residuals, mild easing so the authority can show.
    "m3_authority": lambda: _m3_speed(
        residual_scale=0.16, pitch_clip=0.40, pitch_kp=2.5, w_residual=0.05,
        stance_ratio_final=0.50, efficiency_ramp_steps=150_000_000),
    # R5 — remove the EFFICIENCY tax + the flight demand entirely (ablation): does the energy/torque
    # penalty smother the leg work that holds height?
    "m3_noeff": lambda: _m3_speed(
        stance_ratio_final=0.50, w_torque=0.0, w_motor_vel=0.0, w_energy=0.0),
    # R6 — emphasize STAYING UP: stronger height/vz/upright shaping + modest speed + walking duty.
    "m3_height": lambda: _m3_speed(
        w_height=6.0, w_vz=1.5, w_upright=8.0, v_ceiling=1.5, stance_ratio_final=0.55),
    # R7 — COMBO of the most promising levers: gated hardening + low speed + more authority +
    # gentle final gait. Best single shot at a working policy overnight.
    "m3_combo": lambda: _m3_speed(
        curriculum_gate_ep_len=300.0, v_ceiling=1.5, residual_scale=0.14, pitch_clip=0.35,
        w_residual=0.05, stance_ratio_start=0.65, stance_ratio_final=0.50,
        gait_curriculum_steps=40_000_000, efficiency_ramp_steps=40_000_000),
})

# ----- round 2 (2026-07-22, ~23M): the easing sweep plateaued at ep_len ~67; add the pitch-assist
# m2->m3 bridge (decaying training-wheel on base pitch) so the robot keeps m2's height-holding gait
# while it learns to balance pitch, then the assist fades to 0. m3_cold (= m3_walk preset launched
# WITHOUT --warm-start) is the orthogonal control for "is the m2 lunge prior the trap?".
PRESETS.update({
    # pitch-assist + gentle gait: bridge the height-holding gait across the pitch release.
    "m3_assist": lambda: _m3_speed(
        pitch_assist_kp=150.0, pitch_assist_kd=15.0, pitch_assist_ramp_steps=40_000_000,
        stance_ratio_final=0.55, v_ceiling=1.5, efficiency_ramp_steps=150_000_000),
    # pitch-assist + competence-gated hardening to full running: assist bootstraps balance, the gate
    # only hardens the gait once it can actually survive. Best single shot at a running policy.
    "m3_assist_gated": lambda: _m3_speed(
        pitch_assist_kp=150.0, pitch_assist_kd=15.0, pitch_assist_ramp_steps=40_000_000,
        curriculum_gate_ep_len=300.0, v_ceiling=1.5, residual_scale=0.14, pitch_clip=0.35,
        w_residual=0.05, stance_ratio_start=0.65, stance_ratio_final=0.42,
        gait_curriculum_steps=40_000_000, efficiency_ramp_steps=40_000_000),
})

# ----- round 3 (2026-07-23): round-2 assist FAILED via crutch-dependence. assist=0 eval of the
# m3_assist* runs gave ep_len 39-50 (WORSE than the 67 plateau) with NEGATIVE mean vx -- the
# in-training ep_len ~100+ was entirely the wheel. Mechanism: SPEED mode pays a per-step `alive`
# bonus (0.5) with NO penalty for standing still, so once the assist removes the fall risk, "stand
# still and bank alive" is optimal -> the policy learned neither balance nor locomotion. (m2 worked
# because SPRINT's clock cost makes standing strictly negative.) Fix: close the stand-still
# loophole so the assist help goes toward balancing WHILE MOVING, plus a slower/weaker assist so it
# can't be a full crutch. All warm-start from m2_sprint/ppo_180000000_steps.zip.
def _m3_sprint(**kw):
    """m3 SPRINT variant (the clock cost forbids standing still) + the standard m3 extras."""
    base = dict(base_lock=LOCKS["m3"])
    base.update(_extras("m3"))
    base.update(kw)
    return _sprint(**base)


PRESETS.update({
    # PRIMARY: sprint (forward pressure via the clock cost) + a MODERATE, slower-fading assist
    # (kp 100 vs 150, fade 60M vs 40M) so balance help is spent while RUNNING, not while idling,
    # and the policy must supply more of the balance itself. Warm from m2_sprint (objective matches).
    "m3_assist_sprint": lambda: _m3_sprint(
        pitch_assist_kp=100.0, pitch_assist_kd=10.0, pitch_assist_ramp_steps=60_000_000,
        stance_ratio_final=0.55, residual_scale=0.14, pitch_clip=0.35, w_residual=0.05),
    # CONTROL: sprint + competence-gated curriculum, NO assist — isolates whether the SPRINT
    # objective's forward pressure alone (vs the speed-mode plateau) helps m3 balance.
    "m3_sprint_gated": lambda: _m3_sprint(
        curriculum_gate_ep_len=300.0, stance_ratio_start=0.65, stance_ratio_final=0.42,
        gait_curriculum_steps=40_000_000, efficiency_ramp_steps=40_000_000,
        residual_scale=0.14, pitch_clip=0.35, w_residual=0.05),
    # HEDGE: speed mode but with the alive bonus REMOVED (w_alive=0) so standing still is no longer
    # rewarded, + a weak assist (kp 70) that only softens falls. Tests the stand-still fix in the
    # simpler endless-speed setting.
    "m3_assist_move": lambda: _m3_speed(
        w_alive=0.0, pitch_assist_kp=70.0, pitch_assist_kd=7.0, pitch_assist_ramp_steps=60_000_000,
        stance_ratio_final=0.55, v_ceiling=1.5, residual_scale=0.14, pitch_clip=0.35, w_residual=0.05),
})

# ----- round 4 (2026-07-23): anti-crutch. Round-3 assist=0 eval at 16.6M collapsed to ep_len 14
# (0 m travelled) -- the decaying assist alone still bred full crutch-dependence. Add w_assist_penalty
# so the policy PAYS for the assist torque it provokes: it must keep its own pitch near level (assist
# ~0 N*m) to avoid the penalty, while the assist still catches real falls (dense safety net). Same
# sprint + fade-60M base as m3_assist_sprint; two penalty strengths. Warm from m2_sprint.
PRESETS.update({
    "m3_assist_pen": lambda: _m3_sprint(
        pitch_assist_kp=100.0, pitch_assist_kd=10.0, pitch_assist_ramp_steps=60_000_000,
        w_assist_penalty=0.005, stance_ratio_final=0.55, residual_scale=0.14, pitch_clip=0.35,
        w_residual=0.05),
    "m3_assist_pen_hi": lambda: _m3_sprint(
        pitch_assist_kp=100.0, pitch_assist_kd=10.0, pitch_assist_ramp_steps=60_000_000,
        w_assist_penalty=0.02, stance_ratio_final=0.55, residual_scale=0.14, pitch_clip=0.35,
        w_residual=0.05),
})

# ----- round 5 (2026-07-23): the assist family (rounds 2-4) is exhausted -- every position-holding
# assist bred crutch-dependence (assist=0 eval 20-34, no better than the no-assist control). Try a
# NON-holding idea: pitch slow-motion via extra base-pitch armature that fades to 0. No crutch, so
# the in-training ep_len is already the honest survival -- WIN = it climbs well past ~67 and HOLDS
# as the armature fades. Warm from m2_sprint; more residual/reflex authority to exploit the slow fall.
PRESETS.update({
    # moderate slow-mo (~1.8x slower fall): the robot still visibly falls, must balance itself.
    "m3_slowmo": lambda: _m3_sprint(
        pitch_armature=1.5, pitch_armature_ramp_steps=20_000_000,
        stance_ratio_final=0.55, residual_scale=0.16, pitch_clip=0.40, pitch_kp=2.5,
        w_residual=0.05),
    # strong slow-mo (~2.6x): easier early learning, more risk it doesn't transfer as inertia fades.
    "m3_slowmo_hi": lambda: _m3_sprint(
        pitch_armature=4.0, pitch_armature_ramp_steps=20_000_000,
        stance_ratio_final=0.55, residual_scale=0.16, pitch_clip=0.40, pitch_kp=2.5,
        w_residual=0.05),
})

# ----- 200 Hz reactive-stepping stack (2026-07-23) --------------------------------------------
# Move the control loop 50 -> 200 Hz (physics sim stays 1 kHz, decimation 20 -> 5) so the policy can
# do FAST reactive foot-placement stepping -- the balance channel the passive-ankle plant depends on.
# Because the control RATE changed, several 50 Hz-tuned constants are rescaled here (the reward is
# made rate-invariant IN THE ENV by scaling the per-step sum by control_dt/0.02, a no-op at 50 Hz):
#   gamma 0.99 -> 0.9975 (= 0.99^(1/4): same ~2 s horizon); action_delay 1 -> 4 steps (same ~20 ms
#   Pi+CAN latency); action_filter 0.2 -> 0.4 (less smoothing = reactive, still sim2real-safe);
#   ent_gate_air_time 0.02 -> 0.005 (per-step-mean of an event term scales with dt); every *_steps
#   curriculum/anneal count x4 (an episode is 4x more env steps, so schedules must x4 for equal
#   robot-time). gait_freq ceiling -> 50 Hz (from the 210 RPM motor + partial-arc moves; effectively
#   non-binding). residual_scale 0.08 -> 0.20 + w_residual 0.1 -> 0.02 (reactive-stepping authority).
_HZ200 = dict(
    control_decimation=5, gamma=0.9975, action_delay_steps=4, action_filter=0.4,
    gait_freq_hz=(0.5, 50.0), residual_scale=0.20, w_residual=0.02, ent_gate_air_time=0.005,
    total_steps=800_000_000, gait_curriculum_steps=240_000_000, efficiency_ramp_steps=240_000_000,
    sprint_curriculum_steps=120_000_000, ent_anneal_steps=80_000_000,
    warmstart_obs_count_cap=200_000.0,
)


def _sprint200(m, **kw):
    """A 200 Hz sprint preset for milestone m: base_lock + m-extras + the _HZ200 rate rescalings,
    then per-preset overrides. Obs/action DIMS are unchanged vs 50 Hz, so m2->m3 warm-start works."""
    base = dict(base_lock=LOCKS[m])
    base.update(_extras(m))          # m3+: w_angmom, ent_anneal_deadline_steps, speed_upright_gate
    base.update(_HZ200)
    base.update(kw)
    return _sprint(**base)


PRESETS.update({
    # 200 Hz PRIOR: m2 (X+Z free, pitch LOCKED) trained FROM SCRATCH at the new control rate so its
    # gait is expressed in the 200 Hz action semantics m3 will inherit. No jitter (clean fast prior).
    "m2_reactive": lambda: _sprint200("m2"),
    # 200 Hz TARGET: m3 (pitch FREE), warm-started from m2_reactive. The reactive authority (via
    # _HZ200) + fast control give the policy a real foot-placement balance channel; the sim2real
    # timing curriculum (jitter +-4 ms, drop 5%, competence-gated) hardens it once it can survive.
    "m3_reactive": lambda: _sprint200(
        "m3", ent_anneal_deadline_steps=100_000_000,
        ctrl_jitter_ms_final=4.0, ctrl_drop_prob_final=0.05,
        jitter_curriculum_gate_ep_len=1600.0, jitter_curriculum_steps=80_000_000),
    # m3_reactive base (residual 0.2) plateaued at ep_len ~206 (~1 s, passive fall) by 13M -> more
    # reactive AUTHORITY: 1.5x residual, much less target smoothing (action_filter 0.4->0.25 =
    # crisper fast corrections), longer thigh swings (0.55) + roomier pitch reflex clip (0.40).
    "m3_reactive_hi": lambda: _sprint200(
        "m3", ent_anneal_deadline_steps=100_000_000,
        residual_scale=0.30, action_filter=0.25, thigh_amp=0.55, pitch_clip=0.40,
        ctrl_jitter_ms_final=4.0, ctrl_drop_prob_final=0.05,
        jitter_curriculum_gate_ep_len=1600.0, jitter_curriculum_steps=80_000_000),
    # hi (residual 0.3) also plateaued ~196 -> a bigger jump: 2x residual authority + near-raw targets
    # (action_filter 0.10 = maximal reactivity) + bigger swings + roomier reflex. Last authority lever
    # before concluding the reactive/software route can't fix free-pitch m3 (see m3-antitopple-sweep).
    "m3_reactive_x": lambda: _sprint200(
        "m3", ent_anneal_deadline_steps=100_000_000,
        residual_scale=0.40, action_filter=0.10, thigh_amp=0.60, pitch_clip=0.45,
        ctrl_jitter_ms_final=4.0, ctrl_drop_prob_final=0.05,
        jitter_curriculum_gate_ep_len=1600.0, jitter_curriculum_steps=80_000_000),
    # ALL foot-placement authority (residual 0.2/0.3/0.4) plateaued ~1 s -> the plant lacks ANKLE
    # pitch torque (passive-spring feet). This adds a fixed ankle-torque reflex (emulated actuated
    # ankle, stance-gated) on top of the base reactive stack. If m3 now BALANCES (ep_len climbs),
    # it validates the actuated-ankle hardware recommendation. Warm-startable from m2 (dims same).
    "m3_ankle": lambda: _sprint200(
        "m3", ent_anneal_deadline_steps=100_000_000,
        ankle_kp=100.0, ankle_kd=20.0, ankle_clip=12.0,   # gentle: 40 N*m destabilized the gait (ep_len 28)
        ctrl_jitter_ms_final=4.0, ctrl_drop_prob_final=0.05,
        jitter_curriculum_gate_ep_len=1600.0, jitter_curriculum_steps=80_000_000),
    # lower-CoM: the cheapest candidate hardware fix. Bottom-heavy plant -> slower pitch fall ->
    # the EXISTING actuators (foot placement) may finally balance it. Warm-startable from m2.
    "m3_lowcom": lambda: _sprint200(
        "m3", ent_anneal_deadline_steps=100_000_000, com_lower=0.15,
        ctrl_jitter_ms_final=4.0, ctrl_drop_prob_final=0.05,
        jitter_curriculum_gate_ep_len=1600.0, jitter_curriculum_steps=80_000_000),
    "m3_lowcom_hi": lambda: _sprint200(
        "m3", ent_anneal_deadline_steps=100_000_000, com_lower=0.30,
        ctrl_jitter_ms_final=4.0, ctrl_drop_prob_final=0.05,
        jitter_curriculum_gate_ep_len=1600.0, jitter_curriculum_steps=80_000_000),
})


# ----- ankle-stiffness + foot-ahead experiment (2026-07-24) -------------------------------------
# Two cheap levers the reactive/software route hadn't tried, both warm-started from m2_reactive
# (WARM=training/runs/m2_reactive/ppo_33999660_steps.zip at launch): (a) a STIFFER, preload-
# preserving passive ankle spring = a firmer foot lever, more passive pitch-restoring torque in
# stance; (b) a foot-placement-ahead-of-CoM reward = the capture step to catch a forward fall (the
# user's read of the failure videos: the policy never tried stepping ahead of the CoM). Same 200 Hz
# reactive stack as m3_reactive (residual 0.20, jitter/drop curriculum) so ankle_stiffness /
# w_foot_ahead is the only changed variable vs the known ep_len~206 plateau.
def _m3_react(**kw):
    return _sprint200(
        "m3", ent_anneal_deadline_steps=100_000_000,
        ctrl_jitter_ms_final=4.0, ctrl_drop_prob_final=0.05,
        jitter_curriculum_gate_ep_len=1600.0, jitter_curriculum_steps=80_000_000, **kw)


PRESETS.update({
    # ankle-stiffness sweep (foot-ahead OFF): isolate the stiffer-spring plant change. Preload is
    # preserved in env.py so the standing posture is unchanged; only the restoring gain rises.
    "m3_stiff_lo":    lambda: _m3_react(ankle_stiffness=90.0,  ankle_damping=0.8),   # ~3x
    "m3_stiff":       lambda: _m3_react(ankle_stiffness=200.0, ankle_damping=1.2),   # ~7x
    "m3_stiff_hi":    lambda: _m3_react(ankle_stiffness=350.0, ankle_damping=1.6),   # ~12x
    # follow-up (2026-07-24): the sweep showed a clean monotonic gain with stiffness (stiff_hi/k350
    # climbing past 330 while lo/mid flatten ~245) -> chase the gradient with two stiffer points.
    "m3_stiff_xhi":   lambda: _m3_react(ankle_stiffness=550.0, ankle_damping=2.2),   # ~19x
    "m3_stiff_xxhi":  lambda: _m3_react(ankle_stiffness=750.0, ankle_damping=2.8),   # ~26x
    # foot-ahead reward (default ankle): isolate the capture-step reward.
    "m3_ahead":       lambda: _m3_react(w_foot_ahead=3.0),
    # COMBO: stiffer ankle + capture-step reward -- the best single shot at a balancing m3.
    "m3_stiff_ahead": lambda: _m3_react(ankle_stiffness=200.0, ankle_damping=1.2, w_foot_ahead=3.0),
})


# ----- cadence / motor-limit experiment (2026-07-24) --------------------------------------------
# The k350 winner (m3_stiff_hi) balances m3 but via a degenerate gait: it patters ONE foot at ~11 Hz
# (peak joint vel 23 rad/s, at the 210 RPM motor ceiling) while holding the other leg up -- exploiting
# the fact that the plant has NO velocity/acceleration cap (only torque + range are limited). Add the
# missing motor limits (vel 22 rad/s = 210 RPM as-is; accel) + a contact-switch penalty to force a
# slower, realistic, two-legged gait ("minimise stepping frequency"). Warm-started from stiff_hi (it
# already balances), same k350 ankle so only the limits + cadence penalty change.
def _m3_cad(**kw):
    return _m3_react(ankle_stiffness=350.0, ankle_damping=1.6, **kw)


PRESETS.update({
    # limits + moderate cadence penalty (the recommended combo).
    "m3_cad":        lambda: _m3_cad(motor_vel_limit=22.0, motor_accel_limit=300.0, w_contact_switch=0.15),
    # stronger: tighter accel + heavier cadence penalty (push cadence down harder).
    "m3_cad_hi":     lambda: _m3_cad(motor_vel_limit=22.0, motor_accel_limit=200.0, w_contact_switch=0.30),
    # CONTROL: motor limits ONLY (no cadence penalty) -- isolates whether the limits alone slow the
    # 11 Hz pattering, or whether the explicit penalty is needed.
    "m3_cad_limonly": lambda: _m3_cad(motor_vel_limit=22.0, motor_accel_limit=300.0, w_contact_switch=0.0),
    # HEDGE (2026-07-24): m3_cad* stalled at ep_len ~80 (12M, no recovery) -- the accel=300 cap
    # blocks the reactive foot-placement balance needs. Near-unlimited accel (~torque-limited) +
    # only the velocity cap + cadence penalty: does loosening accel let it re-balance while the
    # penalty still slows the cadence?
    "m3_cad_a1000":  lambda: _m3_cad(motor_vel_limit=22.0, motor_accel_limit=1000.0, w_contact_switch=0.15),
})


# ----- milestone advance with the stiff ankle (2026-07-25, weekend) ----------------------------
# m3 is SOLVED by the k350 preload-preserving stiff ankle (m3_stiff_hi: ep_len 1720, rew +725 at
# 80M, and the least-extreme / most hardware-realistic of the winners). Advance the base-DOF ladder
# with the SAME recipe, each warm-started from the m3 winner: m4 frees Y (lateral; w_lat_vel keeps
# it straight), m5 also frees roll (the abduction reflex becomes the lateral balancer), m6 is fully
# free (+yaw). Same 200 Hz reactive stack + jitter/drop curriculum as the m3 runs; _extras(m) turns
# on angmom / ent-deadline / upright-gate for every pitch-free milestone. NO cadence limits (those
# collapsed training; the chatter is a separate axis to fix later via a ramp curriculum).
def _react(m, **kw):
    return _sprint200(m, ent_anneal_deadline_steps=100_000_000,
                      ctrl_jitter_ms_final=4.0, ctrl_drop_prob_final=0.05,
                      jitter_curriculum_gate_ep_len=1600.0, jitter_curriculum_steps=80_000_000, **kw)


PRESETS.update({
    "m4_stiff": lambda: _react("m4", ankle_stiffness=350.0, ankle_damping=1.6),
    "m5_stiff": lambda: _react("m5", ankle_stiffness=350.0, ankle_damping=1.6),
    "m6_stiff": lambda: _react("m6", ankle_stiffness=350.0, ankle_damping=1.6),
})


# ----- m7: cadence fix, take 2 (2026-07-28) -----------------------------------------------------
# NOT a new DOF -- the run name for the take-2 cadence work on the solved m3 base. Root cause found
# by instrumenting the m6 policy (channel_authority.py): the 200 Hz rescale widened gait_freq_hz to
# (0.5, 50) but the freq_raw->Hz map is linear, so a NEUTRAL action (~0) maps to ~25 Hz -- the
# policy left that output near zero and inherited a ~24 Hz gait clock (the CPG thigh output flips
# sign 35x/s = dither, ±0.45 rad through a ~31 Hz-cutoff EMA -> would destroy the real motors). The
# external velocity/accel/torque limits all failed because they fought this broken INTERNAL default.
# Fix: remap gait_freq_hz -> (0.5, 4.0) so neutral ~= 2 Hz. m7 adds the requested torque-budget
# curriculum ON TOP (now that the default cadence is sane, torque efficiency can refine it); m7_freq
# isolates the freq fix alone (the analysis predicts "this alone may fix the chatter"). Both warm-
# start from m3_stiff_hi (the 7806-ep_len winner); k350 ankle; residual re-bill + pitch-reflex
# headroom deliberately deferred until we re-measure after the freq fix.
PRESETS.update({
    "m7": lambda: _react("m3", ankle_stiffness=350.0, ankle_damping=1.6,
                         gait_freq_hz=(0.5, 4.0),
                         torque_util_target=0.70, torque_limit_gate_ep_len=1500.0),
    "m7_freq": lambda: _react("m3", ankle_stiffness=350.0, ankle_damping=1.6,
                             gait_freq_hz=(0.5, 4.0)),
    # 2026-07-29: re-measured m7_freq @48M -> gait clock fell to 1 Hz but FOOTFALL stayed ~6 Hz:
    # the policy steps through the fast residual channel, not the CPG clock. Add the residual-rate
    # penalty (the analysis's deferred fix) to make that chatter expensive. Warm-start from the
    # freq-adapted m7_freq checkpoint (only needs to unlearn the residual chatter). Two strengths.
    "m7_res":    lambda: _react("m3", ankle_stiffness=350.0, ankle_damping=1.6,
                               gait_freq_hz=(0.5, 4.0), w_residual_rate=5.0),
    "m7_res_hi": lambda: _react("m3", ankle_stiffness=350.0, ankle_damping=1.6,
                               gait_freq_hz=(0.5, 4.0), w_residual_rate=15.0),
    # 2026-07-29 (2nd re-measure): residual-rate + torque did NOT drop the footfall (stuck ~5.5 Hz).
    # Confirmed driver = the fixed pitch reflex: u_p saturated 81%, flips at 6.4 Hz, D-term (kd*rate)
    # DOMINATES P-term 2.2x. Two fixes for the D-term rectifying the 6 Hz gait-bob, warm from m7_freq:
    "m7_reflex_lp": lambda: _react("m3", ankle_stiffness=350.0, ankle_damping=1.6,
                                  gait_freq_hz=(0.5, 4.0), pitch_reflex_rate_lp=0.9),  # ~3.5Hz cutoff
    "m7_reflex_kd": lambda: _react("m3", ankle_stiffness=350.0, ankle_damping=1.6,
                                  gait_freq_hz=(0.5, 4.0), pitch_kd=0.05),             # D-term 4x lower
    # 2026-07-29 (ROOT CAUSE): the 6.3 Hz footfall lives in the ANKLE JOINT, not any control channel
    # -- the k=350 spring is ~9% of critical damping for the leg/body rocking mode (c_crit~18, set to
    # 1.6), so it RINGS at 6.3 Hz on every contact. No reward/reflex/torque lever touched it because
    # it's the spring ringing. Fix = damp the ankle: raise ankle_damping toward critical. Warm from
    # m7_freq. (Keeps the k=350 stiffness that solved m3 balance; only adds damping.)
    "m7_damp":    lambda: _react("m3", ankle_stiffness=350.0, ankle_damping=10.0,
                                gait_freq_hz=(0.5, 4.0)),   # ~55% critical
    "m7_damp_hi": lambda: _react("m3", ankle_stiffness=350.0, ankle_damping=18.0,
                                gait_freq_hz=(0.5, 4.0)),   # ~critical
})


# ----- CPG vs Fourier: the Ijspeert-school A/B (2026-07-29) -------------------------------------
# Question: is a central pattern generator — coupled amplitude-controlled phase oscillators whose
# parameters the policy tunes (Bellegarda & Ijspeert 2022) — better or worse on this robot than the
# per-step Fourier gait spec the whole m1..m7 lineage uses? See cpg_gait.py for the formulation.
#
# What makes this an experiment and not two unrelated runs: every arm below shares ONE plant and
# ONE reward. Same k=350/damping 10 ankle (the m7 root-cause fix), same (0.5, 4.0) frequency map
# (the m7 freq-map bug fix), same 200 Hz reactive stack, same jitter/drop sim2real curriculum, same
# pitch reflex, same abduction reflex, same residual authority. Only the generator differs.
#
# The lineages are trained the way the Fourier one actually was — cold at m2 (pitch locked), then
# warm-started into m3 (pitch free) — because comparing a cold CPG against the 162 M-step warm
# m7_freq checkpoint would measure warm-start budget, not the generator. `_cpgN_` presets are the
# cold m2 stage; `_cpg3_` are the m3 stage that warm-starts from it. The `_fN_`/`_f3_` pair is the
# identical schedule for the Fourier arm, so the baseline is built under the same budget.
#
# ARMS
#   cpg      : CPG + the 6-dim PMTG residual channel  (matched to the Fourier arm's authority)
#   cpg_nr   : CPG with NO residual channel — the pure oscillator. Tests Ijspeert's claim that the
#              limit-cycle dynamics alone give a stable gait, and probes the m7 6 Hz footfall
#              directly: the residual channel is the only fast open path from action to target.
#   fourier  : the m1..m7 generator, same schedule and budget = the honest baseline.
def _ab(action_mode="fourier", milestone="m2", **kw):
    """One arm of the CPG-vs-Fourier A/B: the m7 plant + the 200 Hz reactive stack at a milestone,
    with only the generator swapped. Kept separate from _react so tuning the A/B can never
    retroactively change the m1..m7 presets."""
    return _sprint200(
        milestone, ent_anneal_deadline_steps=100_000_000,
        ankle_stiffness=350.0, ankle_damping=10.0, gait_freq_hz=(0.5, 4.0),
        ctrl_jitter_ms_final=4.0, ctrl_drop_prob_final=0.05,
        jitter_curriculum_gate_ep_len=1600.0, jitter_curriculum_steps=80_000_000,
        action_mode=action_mode, **kw)


PRESETS.update({
    # stage 1 — cold at m2 (X+Z free, pitch LOCKED)
    "ab_f_m2":       lambda: _ab("fourier", "m2"),
    "ab_cpg_m2":     lambda: _ab("cpg", "m2"),
    "ab_cpg_nr_m2":  lambda: _ab("cpg", "m2", cpg_residual=False),
    # stage 2 — m3 (pitch FREE), warm-started from the matching stage-1 run
    "ab_f_m3":       lambda: _ab("fourier", "m3"),
    "ab_cpg_m3":     lambda: _ab("cpg", "m3"),
    "ab_cpg_nr_m3":  lambda: _ab("cpg", "m3", cpg_residual=False),
})

# STRIDE-ENVELOPE CONTROL (added 2026-07-29 after the first 9 M steps). The Fourier arm was running
# ~5 m/s while the CPG lagged, and part of that gap is an artefact of MY parameter choice, not of
# the generator: cpg_stride 0.28 m only used |joint| <= 0.371, while the Fourier arm is free to use
# the full +-0.45 cam/thigh clip, which measurably reaches dx -0.457..+0.592 m. Since the reward is
# speed-dominated, that is a real handicap on the metric that decides the verdict.
#
# So this arm matches the two generators on the constraint the Fourier arm actually feels — the
# JOINT amplitude envelope, not a foot-space number. Measured: with the wider IK table, stride 0.315
# puts max |joint| at 0.449, i.e. exactly saturating the same +-0.45 clip. A separate LUT file keeps
# this out of the way of the arms already training.
PRESETS.update({
    "ab_cpg_wide_m2": lambda: _ab("cpg", "m2", cpg_lut="cpg_foot_lut_wide.npz", cpg_stride=0.315),
    "ab_cpg_wide_m3": lambda: _ab("cpg", "m3", cpg_lut="cpg_foot_lut_wide.npz", cpg_stride=0.315),
})


# ----- m1..m6_CPG: the full base-DOF-unlocking curriculum on the CPG generator (2026-07-30) ------
# A from-scratch CPG lineage that walks the whole milestone ladder, m1 (only X free) -> m6 (fully
# free), each stage warm-started from the previous one. The A/B (see above) said the CPG is at least
# competitive and is the better cold learner, so this is the "does it go all the way" run.
#
# CURRICULA ARE COMPRESSED TO THE PER-STAGE BUDGET, and that is a deliberate departure from the
# m1..m7 presets. train.py restarts a warm-started stage's ramps from zero, so with the _HZ200
# defaults (gait ramp 240 M, ent anneal deadline 100 M) a ~60 M-step stage would only ever traverse
# a quarter of its own stance-ratio ramp and would never leave maximum exploration. That is exactly
# what happened to the A/B arms: policy std sat at ~1.0 and ent_coef never left 0.01 in every run
# that stopped under 80 M steps. Sized to 60 M per stage, every ramp completes inside the stage:
#   gait/efficiency 30 M, sprint line 20 M, entropy anneal 12 M with a 20 M gate deadline.
# The trade is that each stage sees a faster-moving curriculum than the historical presets did.
_CPG_STAGE_STEPS = 60_000_000
_CPG_CHAIN = dict(
    action_mode="cpg",
    # plant knowledge carried over from m7 (see m7-cadence-freq-bug): the k=350 ankle spring that
    # solved m3 balance, damping near critical so it cannot ring at 6.3 Hz, and the frequency
    # remap without which a neutral action means a 25 Hz gait clock
    ankle_stiffness=350.0, ankle_damping=10.0, gait_freq_hz=(0.5, 4.0),
    total_steps=_CPG_STAGE_STEPS,
    gait_curriculum_steps=30_000_000,
    efficiency_ramp_steps=30_000_000,
    sprint_curriculum_steps=20_000_000,
    ent_anneal_steps=12_000_000,
    ent_anneal_deadline_steps=20_000_000,
)


def _cpg_stage(m):
    """One rung of the CPG milestone ladder. Deliberately does NOT reuse _ab/_react: those pass
    ent_anneal_deadline_steps=100_000_000, which is the override that kept the A/B arms pinned at
    full exploration for their whole run."""
    kw = dict(_CPG_CHAIN)
    if LOCKS[m][4] == 0:
        # PITCH-FREE RUNGS KEEP THE UNCOMPRESSED _HZ200 CURRICULA. Compressing them was a mistake
        # and cost three failed m3 attempts, each killed by a different clock-driven ramp reaching
        # its target while the policy was still falling at ~1.3 s:
        #   attempt 1: stance hit 0.42 (flight demanded) at 19.6 M with ep_len 257
        #   attempt 2: gated stance, but the warm-start prior was itself a flight-phase runner
        #   attempt 3: gated stance AND a grounded prior, but the sprint line -- which the gate does
        #             NOT cover -- had already ramped to 77 m by 14 M, vs 34 m for the run that
        #             worked, so a robot falling in 1.3 s was being asked to run 77 m.
        # The A/B's ab_cpg_m3warm_s0 reached ep_len 3772 and crossed the line on exactly these
        # (long) schedules, so use them rather than keep guessing which ramp bites next. m1/m2 keep
        # the compressed schedules, which demonstrably work there (m2_CPG ran the full 100 m dash).
        # Consequence to be honest about: within a 60 M stage these ramps barely move, so m3..m6
        # produce a WALKING controller on a free plant, not a flight-phase runner.
        for k in ("gait_curriculum_steps", "efficiency_ramp_steps", "sprint_curriculum_steps",
                  "ent_anneal_steps", "ent_anneal_deadline_steps"):
            kw.pop(k, None)                      # fall through to the _HZ200 defaults
        kw.update(ent_anneal_deadline_steps=100_000_000)
        # pitch free (m3..m6): add the sim2real timing curriculum, competence
        kw.update(           # gated, as the Fourier lineage did — m1/m2 stay a clean fast prior
            ctrl_jitter_ms_final=4.0, ctrl_drop_prob_final=0.05,
            jitter_curriculum_gate_ep_len=1600.0, jitter_curriculum_steps=80_000_000,
            # NO curriculum_gate_ep_len. It was added here as a safety fix and measurably BACKFIRED:
            # the gate holds BOTH the stance and the EFFICIENCY ramps at their easy start, so with a
            # gate the policy never reaches, eff_scale stays pinned at exactly 0 and the torque /
            # motor-velocity / energy penalties are switched off entirely. The successful reference
            # run (ab_cpg_m3warm_s0, ep_len 3772) had them ramping in — eff 0.058 by 14 M — so it was
            # being regularized against exactly the high-frequency thrash this plant is prone to,
            # while the gated runs were not. A config diff against that run left the gate as the ONLY
            # substantive difference. It is also now redundant: on the restored 240 M curricula the
            # stance ramp only reaches ~0.63 within a 60 M stage, so it cannot demand a flight phase
            # early the way the compressed schedule did, which is what the gate was guarding against.
            # (History: the gate was introduced because the FIRST run's compressed clock ramp hit
            # stance 0.42 at 19.6 M while m3's ep_len was 257. Restoring the long curricula fixes
            # that cause directly, so the gate is no longer needed and its eff_scale side effect
            # makes it actively harmful. Both facts are measured, see the note above.)
        )
    return _sprint200(m, **kw)


PRESETS.update({f"{m}_CPG": (lambda m=m: _cpg_stage(m)) for m in LOCKS})


# ----- teleop: joystick command + the full sim2real package (2026-07-29) ------------------------
# The demo target: a robot you can DRIVE — stand still on a centred stick (stepping in place is
# fine, the plant cannot stand passively), walk and slow-run forward, back up, and steer left/right
# — that survives contact with a real floor. Not a speed record; the commandable maximum is the
# oracle's CONTINUOUS-torque ceiling (1.8 m/s) so nothing the joystick can ask for will cook a
# motor during a long demo.
#
# This is a FROM-SCRATCH lineage, not another milestone hop: the action gained 2 steering dims
# (24 -> 26), the observation lost the privileged base velocity and gained the command channel and
# a strided history, so no m1..m7 checkpoint can be loaded into it. What carries over is the plant
# knowledge, and all of it is baked into the defaults below:
#   * k=350 ankle spring          — the m3 balance result (m3-ankle-stiffness-foot-ahead)
#   * ankle_damping=10.0          — the m7 root cause: at 1.6 the k=350 spring is ~9% of critical
#                                   and RINGS at 6.3 Hz on every contact, which is what every
#                                   cadence lever failed to fix because it was never a control
#                                   problem (commit 0f22b8d)
#   * gait_freq_hz=(0.5, 4.0)     — the m7 freq-map fix; at (0.5, 50) a neutral action meant 25 Hz
#   * the 200 Hz reactive stack + the jitter/drop timing curriculum
#
# Stage it: teleop_easy (does the command/steering machinery learn at all?) -> teleop (the real
# thing). Do not debug a failure to turn and a failure to survive domain randomization at once.
_TELEOP_PLANT = dict(
    ankle_stiffness=350.0, ankle_damping=10.0, gait_freq_hz=(0.5, 4.0),
)


def _teleop(**kw):
    """A joystick-command preset on the fully-free base (m6): yaw MUST be free to turn."""
    base = dict(objective="command", base_lock=LOCKS["m6"])
    base.update(_extras("m6"))          # w_angmom, ent deadline, speed_upright_gate
    base.update(_HZ200)
    base.update(_TELEOP_PLANT)
    base.update(dict(
        # no clock in this objective, so survival has to be paid for directly
        w_alive=0.5, w_time=0.0,
        steer_enable=True,              # the L/R asymmetry channel; without it it cannot turn
        # walking / slow running, not sprinting: keep a little double support at the top of the
        # range instead of demanding a flight phase the demo does not need
        stance_ratio_start=0.70, stance_ratio_final=0.50,
        # the command curriculum replaces the sprint-distance one
        sprint_curriculum_steps=0,
        # observation: privileged velocity OFF, longer strided history to make it inferable
        obs_base_vel=False, history_len=10, history_stride=4,
        # sim2real package
        dr_enable=True, obs_noise_enable=True,
        push_interval_s=6.0, push_dv=0.35,
        trip_prob=0.0008,               # ~one trip per 6 s at 200 Hz
        ent_anneal_deadline_steps=100_000_000,
        # competence gates: everything hard waits until the robot can survive ~8 s
        curriculum_gate_ep_len=1600.0,
        jitter_curriculum_gate_ep_len=1600.0, jitter_curriculum_steps=80_000_000,
        ctrl_jitter_ms_final=4.0, ctrl_drop_prob_final=0.05,
    ))
    base.update(kw)
    return Config(**base)


# The ladder below is deliberately WARM-STARTABLE end to end: teleop_easy -> teleop_nodist ->
# teleop all share obs and action dims, so each stage loads the previous one's checkpoint and only
# has to absorb the new adversity instead of relearning to walk. That is why teleop_easy does NOT
# turn the privileged velocity back on, tempting as it is — doing so changes the obs width and
# breaks the chain. The privileged variant is teleop_oracle, a diagnostic, not a rung.
PRESETS.update({
    # THE TARGET.
    "teleop": lambda: _teleop(),
    # Rung 1: same task and same tensor shapes, no adversity. Plant fixed, sensors clean, no
    # pushes/trips/jitter. If standing and steering do not emerge HERE, the problem is the command
    # design, not the randomization — debug it here where the signal is clean and it trains fast.
    "teleop_easy": lambda: _teleop(
        dr_enable=False, obs_noise_enable=False,
        push_interval_s=0.0, trip_prob=0.0,
        ctrl_jitter_ms_final=0.0, ctrl_drop_prob_final=0.0),
    # Rung 2: full observation/plant realism, but no external disturbances. Isolates "can it track
    # commands through sensor noise and a randomized plant" from "can it take a shove".
    "teleop_nodist": lambda: _teleop(push_interval_s=0.0, trip_prob=0.0),
    # DIAGNOSTIC, not a rung: the privileged base velocity back in the observation. Its obs is 3
    # dims wider, so it does not warm-start into anything above. Run it only to answer one
    # question — if teleop_easy stalls, does it stall because velocity is unobservable (this one
    # trains fine) or because the task/reward is wrong (this one stalls too)?
    "teleop_oracle": lambda: _teleop(
        obs_base_vel=True, dr_enable=False, obs_noise_enable=False,
        push_interval_s=0.0, trip_prob=0.0,
        ctrl_jitter_ms_final=0.0, ctrl_drop_prob_final=0.0),
    # Conservative demo variant: half the command box (0.9 m/s / 0.5 rad/s). Use for the first
    # tethered hardware runs — same policy interface, smaller envelope.
    "teleop_slow": lambda: _teleop(
        cmd_v_fwd_max=0.9, cmd_v_back_max=0.35, cmd_yaw_max=0.5,
        stance_ratio_final=0.60),
    # v2 (2026-07-30): the 175M-step teleop policy could not turn RIGHT AT ALL — measured yaw
    # response slope 0.19 with a +0.31 rad/s intercept, i.e. it turned left whatever you asked.
    # Diagnosed to two causes, both fixed in the defaults above and both carried by this preset:
    #   (a) the model's L/R four-bar linkages differ by ~1 mm -> a ~0.4 rad/s INTRINSIC yaw bias
    #       under a symmetric gait. Now randomized per episode (dr_loop_site) so cancelling it is a
    #       learned feedback behaviour rather than a constant.
    #   (b) the two steering channels were scaled inversely to their measured authority. Corrected.
    # Warm-startable from the v1 teleop run: dims are unchanged, only scales and DR.
    "teleop_v2": lambda: _teleop(dr_loop_site=0.0015),
    # v3 (2026-07-31): v2 PEAKED at ep_len 1855 / rew 512 at 113 M and then declined monotonically
    # for the remaining 215 M steps, ending at 725 / 26. Nothing was broken — the curriculum simply
    # kept demanding more (stance_ratio 0.70 -> 0.52, eff_scale 0 -> 0.90) with no feedback path,
    # and the policy was ground down against targets it could not hold. Two changes, no new physics:
    #   * curriculum_retreat_frac=0.7 -> the stance/efficiency/jitter ramps now RETREAT when ep_len
    #     falls below 70% of the gate, so an over-ambitious target self-corrects.
    #   * gentler targets. The peak lived at stance 0.70 / eff 0.00; 0.50/1.0 was inherited from the
    #     SPRINT lineage and is the wrong reference for "clean walking and slow running, robust".
    #     0.62 keeps real double support; eff 0.35 still buys smoothness without dominating.
    # Warm-starts from the v2 PEAK checkpoint, not its final one.
    "teleop_v3": lambda: _teleop(
        dr_loop_site=0.0015,
        curriculum_retreat_frac=0.7,
        stance_ratio_final=0.62,
        efficiency_target=0.35),
    # v4 (2026-08-03): v3's retreat fix worked — no collapse, reward held ~500 to the end instead of
    # falling to 26 — but on a FIXED eval task v1/v2/v3 were indistinguishable (MTBF 9.6-11.2 s over
    # 20 seeded episodes). v3 only scored better because the retreat correctly kept it at an easier
    # curriculum point; it never got harder, so it never got better. Ablation found why nothing could
    # progress: the plant randomization alone costs 20x MTBF (196 s clean -> 4.8 s full DR), while
    # trips cost 1.8x, sensor noise 1.5x and pushes nothing measurable. Every other adversity was
    # ramped; DR was not, so the policy spent every run pinned just under the competence gate.
    # v4 = v3 + a gated, retreating ramp on the DR width. Everything else identical.
    "teleop_v4": lambda: _teleop(
        dr_loop_site=0.0015,
        curriculum_retreat_frac=0.7,
        stance_ratio_final=0.62,
        efficiency_target=0.35,
        dr_curriculum_steps=120_000_000,
        push_dv=0.6),                  # pushes at 0.35 had no measurable effect; make them real
    # Wider randomization, for the robustness-vs-performance ablation the eval grid scores.
    "teleop_hard": lambda: _teleop(
        dr_mass_global=0.20, dr_mass_body=0.25, dr_inertia=0.40, dr_com_offset=0.03,
        dr_friction_range=(0.35, 1.5), dr_kp=0.30, dr_kv=0.35, dr_torque=0.20,
        dr_ankle_k=0.35, dr_gravity_tilt=5.0, dr_delay_steps_range=(2, 8),
        trip_prob=0.0015, push_dv=0.5),
})


# ----- mX_slow_gait: retrain the WHOLE m1..m6 curriculum with the cadence fixes baked in ---------
# Every prior milestone lineage (m2_reactive, m3_stiff, m4/m5/m6_stiff, m7*) carried at least one of
# the two cadence root causes, discovered only after they had trained:
#   (1) the gait_freq_hz MAP BUG — the 200 Hz rescale set the range to (0.5, 50) Hz, but the
#       freq_raw->Hz map is LINEAR, so a neutral action (~0) inherited a ~25 Hz gait clock. Fixed by
#       remapping to (0.5, 4.0) so neutral ~= 2 Hz (see m7-cadence-freq-bug).
#   (2) the ANKLE-SPRING RING — the k=350 spring that SOLVES m3 pitch balance is only ~9% of critical
#       damping at the shipped ankle_damping (1.6), so it rings at 6.3 Hz on every contact. NO reward
#       /reflex/torque lever touched it because it was never a control problem; ankle_damping=10.0
#       (~55% critical) is the fix, now the standing plant for the teleop / CPG-A/B lineages too.
# Warm-starting a fast-stepping policy CANNOT unlearn the chatter (every m7 warm variant stayed
# trapped in the fast basin), so the only real fix is to retrain from step 0 with the corrected plant
# — which is what the user asked for. This lineage bakes BOTH fixes into ONE identical plant across
# the entire base-DOF ladder (so each warm-start transfers cleanly), plus a GENTLE contact-switch
# penalty as a from-scratch nudge toward fewer, longer steps:
#   * gait_freq_hz=(0.5, 4.0)   — neutral ~2 Hz, not 25 Hz
#   * ankle_stiffness=350.0     — the preload-preserving stiff spring that solved m3 balance
#   * ankle_damping=10.0        — ~55% critical; kills the 6.3 Hz contact ring
#   * w_contact_switch=0.05     — mild, well below the 0.15/0.30 that collapsed the warm m3_cad runs
#                                 (those also fought the 24 Hz internal clock; from scratch at ~2 Hz
#                                 with a damped ankle the equilibrium is already slow, so this only
#                                 biases it, it doesn't have to force it)
# Chain: m1 COLD, then m2<-m1, m3<-m2 (the pitch release), m4<-m3, m5<-m4, m6<-m5 (afterok SLURM).
# Same obs/action dims at every rung (fourier generator, no steering, privileged vel on) so the whole
# ladder warm-starts end to end. _extras(m) still adds z-rail (m1) + angmom/upright-gate (m3..m6).
_SLOW_PLANT = dict(
    ankle_stiffness=350.0, ankle_damping=10.0, gait_freq_hz=(0.5, 4.0),
    w_contact_switch=0.05,
)


def _mk_slow(m):
    return lambda: _react(m, **_SLOW_PLANT)


PRESETS.update({f"{m}_slow_gait": _mk_slow(m) for m in LOCKS})


# ----- mX_sym_gait: slow_gait + the anti-one-legged fix (2026-07-31) -----------------------------
# slow_gait solved balance/performance but converged ONE-LEGGED (m3: left duty 0%, right patters
# 6.4 Hz, asymmetry 0.99) — the freq/ankle/contact-switch levers slowed the CLOCK but did not force
# two-leggedness. This lineage adds the duty-symmetry penalty (w_duty_sym: a foot that never bears
# load is expensive) and a 4x stronger contact-switch (0.05 -> 0.20; symmetry alone could be met by
# both feet chattering fast, so the switch penalty keeps it slow). Everything else is the slow_gait
# plant, so this is a clean A/B vs the one-legged baseline. Same obs/action dims -> warm-chains end
# to end. Reward-only (no obs change) on purpose — watch m1/m2_sym cadence early; if the duty signal
# proves too delayed to learn, the fallback is to add the duty EMA to the observation.
_SYM_PLANT = dict(_SLOW_PLANT)
_SYM_PLANT.update(w_contact_switch=0.20, w_duty_sym=8.0, duty_floor=0.30, duty_sym_tau_s=1.0)


def _mk_sym(m):
    return lambda: _react(m, **_SYM_PLANT)


PRESETS.update({f"{m}_sym_gait": _mk_sym(m) for m in LOCKS})


# ================================================================================================
# ANKLE-SPRING STUDY (2026-08-03) -- is the passive foot spring useful, must it be actuated, and
# is there an optimal stiffness?
# ================================================================================================
# What the record actually contained before this, and why none of it answers the question:
#   * 2026-07-24 swept ankle_stiffness 90/200/350/550/750 and found a threshold at k>=350. But it
#     was ONE seed, WARM-STARTED from a policy trained on the soft spring, and -- as m7 later
#     discovered -- under-damped at every point, because ankle_damping was hand-picked per arm
#     instead of tracking k. So the sweep varied two things at once and never located an optimum;
#     it only showed ">=350 beats <=200".
#   * The only "actuated ankle" ever tested was a FIXED, phase-blind PD on base pitch (ankle_kp/kd).
#     Strong destabilized the gait (ep_len 28), gentle was marginal (284 vs 212). A policy-controlled
#     ankle was recommended twice and never built -- it needed a hardware decision from the user.
#   * k=0 was never runnable at all: ankle_stiffness=0 means "keep the model's 28.65".
#
# So the study is 11 arms on ONE fixed control stack (_SYM_PLANT: the current best -- Fourier
# generator, (0.5,4.0) freq map, contact-switch 0.20, duty-symmetry 8.0), COLD-started, 3 seeds:
#
#   rigid              ankle welded             -- null: is compliance worth anything?
#   free               k=0, floppy              -- null: is it the SPRING or just the joint?
#   k29 .. k1100       7-point passive sweep    -- where is the optimum, if there is one?
#   active             servo, no spring         -- would an actuated angle have been better?
#   active_k350        servo + spring, parallel  -- or is parallel-elastic the real answer?
#
# Three things make it a fair test, all of which the 2026-07-24 sweep lacked:
#   1. ankle_zeta=0.7 -- damping is DERIVED from k against the measured stance inertia, so every
#      arm sits at the same damping ratio and k is genuinely the only variable. (Measured: the
#      ankle's stance inertia is 0.316 kg*m^2, 53x its swing inertia. That is why the old fixed
#      ankle_damping=1.6 was ~7.6% of critical at k=350 and rang at 5.3 Hz -- the m7 6 Hz footfall,
#      re-derived from the plant instead of from a training curve.)
#   2. The MEASURED-MASS plant (dash01_measured.xml, 15.14 kg vs the CAD placeholder's 12.83, with
#      a 2.6x heavier shin). The spring's job is distal energy storage, so getting distal mass
#      wrong would answer the question about a robot that does not exist.
#   3. The active arms carry their motors' real mass (0.75 kg AK60-39 per side, welded AT the
#      ankle) and their torque is billed in the efficiency reward, so "active" cannot win on free
#      energy or free mass.
_STUDY_PLANT = {k: v for k, v in _SYM_PLANT.items()
                if k not in ("ankle_stiffness", "ankle_damping")}   # the study sets these per arm
_STUDY_PLANT.update(
    model_path="model/dash01_measured.xml",
    ankle_zeta=0.7,          # derived damping; overrides ankle_damping. See _setup_ankle.
)
_ACTIVE_MODEL = "model/dash01_measured_active.xml"

# The passive stiffness grid, log-spaced so an optimum shows up as a peak rather than a plateau
# edge. k=28.65 is the real spring the robot has today; 350 is the 2026-07-24 winner, kept as the
# anchor that lets this study be compared against the old one; 1100 extends past it because that
# sweep never established an upper bound (750 was still climbing when it was cut).
STUDY_K = (28.65, 90.0, 200.0, 350.0, 550.0, 750.0, 1100.0)

STUDY_ARMS = {
    "rigid": dict(ankle_mode="rigid"),
    "free":  dict(ankle_mode="free"),
    **{f"k{k:g}".replace(".", "_"): dict(ankle_mode="passive", ankle_stiffness=k)
       for k in STUDY_K},
    "active":     dict(ankle_mode="active", model_path=_ACTIVE_MODEL),
    "active_k350": dict(ankle_mode="active_spring", ankle_stiffness=350.0,
                        model_path=_ACTIVE_MODEL),
}


def _study(m, arm, **kw):
    """One study arm at milestone m. Everything except the ankle is _SYM_PLANT, identical in every
    arm -- that is the whole point, so any difference in the result is attributable to the ankle."""
    if arm not in STUDY_ARMS:
        raise KeyError(f"unknown study arm {arm!r}; have {sorted(STUDY_ARMS)}")
    return _react(m, **{**_STUDY_PLANT, **STUDY_ARMS[arm], **kw})


# m3 (pitch free) is where the ankle decides the outcome -- it is the rung the whole k=350 result
# came from, and the cheapest rung that is still sensitive. m6 (fully free) re-runs only the top
# arms, to check the ranking survives on the real 6-DOF plant.
PRESETS.update({f"study_m3_{a}": (lambda a=a: _study("m3", a)) for a in STUDY_ARMS})
PRESETS.update({f"study_m6_{a}": (lambda a=a: _study("m6", a)) for a in STUDY_ARMS})

# Control arm: the 2026-07-24 winner on the OLD placeholder-mass plant. Not part of the sweep --
# it exists to measure how much the mass correction alone moved the answer, i.e. whether any of the
# pre-2026-08 results transfer to the real robot at all.
PRESETS["study_m3_k350_oldmass"] = lambda: _react(
    "m3", **{**_STUDY_PLANT, **STUDY_ARMS["k350"], "model_path": "model/dash01.xml"})

# Sensitivity check, run ONLY if the active arm loses: does it still lose when its energy is free?
# If yes, active is dead on the merits; if no, active is dead on its energy budget, which is a
# different (and more fixable) conclusion.
PRESETS["study_m3_active_freeenergy"] = lambda: _study("m3", "active", ankle_torque_billed=False)


# ----- mX_wskill_gait: hard workspace-kill on top of sym (2026-08-04) ----------------------------
# The one-legged gait parks a leg FOLDED with the toe OUTSIDE the measured real-robot foot workspace
# (a sim-only exploit, visible in the slow_gait videos). sym_gait only TAXED one-leggedness (duty
# term) and still lost at the DOF releases (m4/m5/m6_sym collapsed to ep_len ~230). This makes the
# exploit IMPOSSIBLE: `workspace_kill` terminates the episode (fall_penalty) the moment a foot leaves
# the measured reachable box. Recipe = the sym plant (duty term kept as belt-and-suspenders) + the
# kill. Meant to be WARM-STARTED from m3_sym (already two-legged, so the kill barely bites) up the
# m4->m6 ladder that collapsed — the test of whether removing the one-legged crutch lets the
# lateral/roll/yaw releases finally survive. Same obs/action dims, so it warm-chains from m3_sym.
_WSKILL_PLANT = dict(_SYM_PLANT)
_WSKILL_PLANT.update(workspace_kill=True)


def _mk_wskill(m):
    return lambda: _react(m, **_WSKILL_PLANT)


PRESETS.update({f"{m}_wskill_gait": _mk_wskill(m) for m in LOCKS})


def get_config(name: str = "default") -> Config:
    return PRESETS[name]()


# ----- serialization (resolved_config.json round-trip) -----------------------------------
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
    """Rebuild a Config from a JSON-loaded dict. Unknown keys are warned about and dropped
    (forward compatibility with configs written by newer code)."""
    from dataclasses import fields as _f
    known = {x.name for x in _f(Config)}
    clean = {}
    for k, v in d.items():
        if k not in known:
            print(f"[config] WARNING: dropping unknown Config field '{k}'")
            continue
        clean[k] = tuple(v) if k in _tuple_fields() and isinstance(v, list) else v
    return Config(**clean)
