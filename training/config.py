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
from typing import Any, List


@dataclass
class Config:
    # ----- model & timing -----
    # The one plant: measured masses (15.14 kg), ankle-lock equalities, settled loaded stance.
    # Resolved relative to this package if not found from CWD. Only the active-ankle arms override
    # it (dash01_active.xml, 8 actuators) -- mass is not an experimental variable, see
    # model/apply_measured_masses.py.
    model_path: str = "model/dash01.xml"
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
    # ASYMMETRIC actor-critic (2026-08-10). Appends DashEnv.PRIV_DIM (=6) sim-only ground-truth
    # entries after the history block: true body-frame base velocity (3), per-foot contact (2),
    # height error (1). Only the CRITIC and the supervised velocity-estimator target read them —
    # the actor's input is the slice obs[:frame_dim*history_len], enforced by asym_policy.py.
    # Rationale: obs_base_vel=False is right for the ACTOR (the hardware has no velocimeter) but
    # SB3 shares one observation between actor and critic, so it also blinded the VALUE FUNCTION
    # of a velocity-tracking task to velocity. Every SOTA velocity-command stack (Rudin et al.,
    # RMA, DreamWaQ) gives the critic privileged state; the critic never runs on the robot.
    # Changes the obs contract (550 -> 556): orphans every earlier checkpoint, which is why it
    # ships with the cold 2026-08-10 ladder and not before.
    obs_privileged_critic: bool = False
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
    # ----- MEASURED-ON-ROBOT sim2real axes (2026-08-06) -----------------------------------------
    # HOMING ERROR. noise_encoder_offset (0.01 rad = 0.57 deg) is the encoder's own zero jitter and
    # is far too small for what actually happens: the user re-homes before every run and calibrates
    # to 1-2 deg, so 5 deg is the safe envelope. Crucially this is NOT just an observation error --
    # a wrong zero means you READ theta-delta and the drive parks at theta+delta, so it must be
    # applied to the COMMAND as well, which noise_encoder_offset never was. Only the pair is
    # physically right; obs-only would teach the policy a bias that does not exist in the plant.
    # NOTE only ~45% of the (cam,thigh) box is mechanically assemblable, so a large homing error can
    # push commands out of the band on the real robot -- this trains for it, it does not fix it.
    dr_joint_zero_deg: float = 0.0
    # IMU MOUNTING ROTATION. noise_grav_bias adds a vector offset to gravity and renormalises, which
    # approximates a small tilt but leaves the GYRO untouched -- physically impossible, a rotated
    # IMU rotates both. This applies one per-episode random rotation to gravity AND angular velocity
    # together. Distinct from dr_gravity_tilt, which rotates the WORLD (a slope): there the robot
    # really is tilted, here it is level and the sensor lies.
    dr_imu_rot_deg: float = 0.0
    # IMU DROPOUT. Freeze the IMU channels for a window -- stale frames, a hung driver, or the
    # tethered rig where the reading stops reflecting free-body dynamics. The point is NOT to make
    # the policy IMU-blind (for a free-base biped the IMU is the only balance sensor, and blind
    # means it can never reject a disturbance); it is to stop it LUNGING when the IMU goes
    # uninformative. The gait generator is feedforward, so the legs should keep cycling.
    dr_imu_dropout_prob: float = 0.0    # per-step probability of entering a stale window
    dr_imu_dropout_s: float = 0.25      # how long a stale window lasts
    # BUS-VOLTAGE SAG. dr_torque is a per-EPISODE constant; a real pack droops as current is drawn
    # and recovers when it is not. First-order droop on delivered power. The EVE 50PL cells are not
    # expected to sag much -- this is insurance, and it costs nothing.
    dr_torque_sag: float = 0.0          # peak fractional torque loss at sustained full power
    dr_torque_sag_tau_s: float = 2.0    # droop/recovery time constant
    # Put pushes and trips on the dr_scale ramp as well. OFF by default so every existing preset
    # trains bit-identically. Measured on walk_fwd_easy_s0 (paired seeds, 12 episodes, clean 0.8 Hz
    # plant): the walker survives 3/12 episodes clean, and 0/12 with pushes alone, 0/12 with trips
    # alone -- each as expensive as full-width plant DR, which already had a ramp. Those two were
    # the last adversity axes still applied at full width from step 0, and walk_fwd2 sat at 2.0 s
    # for 300 M steps because of it.
    adversity_curriculum: bool = False

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
    #                  (model/dash01_active.xml) -- action/obs widen, own lineage.
    #   "active_spring" policy-driven ankle servo IN PARALLEL with the spring (parallel-elastic).
    #   "bar"          NO spring. A rigid TENSION-ONLY strut (2026-08-05, user's v2 candidate):
    #                  it takes unlimited load in traction, so it is a hard stop at the flat-foot
    #                  angle in the direction the ground loads it; in compression it BUCKLES past
    #                  ankle_bar_buckle_nm and carries nothing more. Removes the 249 g spring
    #                  assembly from each shin (see ankle_spring_mass_kg) -- distal mass being the
    #                  axis the speed ceiling is most sensitive to.
    # NOTE "rigid" needs the lock_ankle equalities and the active modes need the 8-actuator model;
    # env.py raises if model_path lacks them, rather than silently running the wrong arm.
    ankle_mode: str = "passive"
    # PRELOAD. The shipped spring is preloaded ~14.3 N*m at stance (springref -+0.7 vs a standing
    # ankle at -+0.20), and _setup_ankle's k-change is preload-PRESERVING by default: raising k
    # alone balloons that preload and flips the robot (verified 2026-07-24). "zero" instead puts
    # springref AT the flat-foot standing angle, so the spring makes no torque on an unloaded flat
    # foot and only resists deflection from it -- the physically-real "spring with no preload".
    # This is a much WEAKER ankle than the same k with preload, so the two are separate arms.
    ankle_preload: str = "preserve"     # "preserve" | "zero"
    # Compressive torque the tension-only strut carries before it buckles ("bar" mode). The user's
    # spec is a 10 N buckling FORCE; the plant has no spring geometry at all (build_model.py models
    # the spring as pure joint stiffness -- the preload breakaway is an explicit TODO there), so
    # there is no lever arm to convert it. Realistic ankle lever arms of 20-60 mm put 10 N at
    # 0.2-0.6 N*m, and the foot's own gravity torque is ~0.13 N*m, so the strut holds the foot near
    # flat through swing across that whole band -- the conclusion is insensitive to the lever arm.
    # Swept rather than guessed; see the ankle2 study arms.
    ankle_bar_buckle_nm: float = 0.4
    # Mass REMOVED from each shin (LegLeft/RightNCS-v1) when the spring assembly is deleted, with
    # that body's inertia tensor scaled by the mass ratio -- the same approximation
    # apply_measured_masses.py makes. The measured shin is 573 g = 324 g of link + 249 g of spring
    # hardware folded in (they are parallel and very close), so 0.249 restores the bare shin.
    # 0 = keep the plant as built. Distal mass costs ~0.93 m/s per kg, so this is not a detail.
    ankle_spring_mass_kg: float = 0.0
    # Re-settle the `stand` keyframe against THIS arm's ankle law before training (env
    # ._resettle_keyframe). The shipped keyframe is an equilibrium of the k=28.65 PRELOADED spring,
    # so a softer/preload-free/strut ankle starts off equilibrium and lurches at every reset -- a
    # handicap falling on exactly the arms under test, and a bias in _stand_torque and
    # height_target too. Default OFF so every m1..m7 / slow / sym / wskill / ankle-study preset is
    # bit-for-bit unchanged; the 2026-08-05 arms all turn it on, INCLUDING their k350 control, so
    # every arm in that comparison gets identical treatment.
    ankle_resettle: bool = False
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
    # ----- ankle MOTOR envelope (active modes only) ---------------------------------------------
    # The active arm's motor is MASSLESS with an AKE90-8's performance. That is deliberate: we do
    # not yet know whether an ankle motor is worth anything, nor what performance it would need, and
    # both are what the study is for. Charging it a specific motor's mass and rotor inertia would
    # answer the narrower "is THIS motor worth it" and would let a loss be blamed on the hardware
    # pick rather than on the idea. So the active arm is an UPPER BOUND -- real torque, real speed,
    # no mass, no reflected inertia. If it still loses, an actuated ankle is dead on the merits. If
    # it wins, the telemetry below IS the spec, and only then is paying the mass worth simulating.
    #
    # The torque-speed CURVE is enforced (env._apply_ankle_torque_speed, re-evaluated every substep):
    # available torque falls linearly to zero at the no-load speed, as a real motor's does. Without
    # it the "ideal" ankle would deliver peak torque at any speed, and a win could just mean an
    # impossible actuator wins. peak_nm lives in the model's forcerange; these two shape the curve
    # and the reporting.
    ankle_motor_noload_rads: float = 22.0   # output-side no-load speed (AKE90-8). 0 = no speed limit
    # BACK-EMF on the six GAIT actuators: joint-side no-load speed per actuator, in the model's
    # actuator order (hip_roll_L, cam_L, thigh_L, hip_roll_R, cam_R, thigh_R). Empty = disabled,
    # which is what every preset before 2026-08-10 trained with -- a flat forcerange delivering
    # peak torque at any speed, which no motor does. Only the optional ANKLE motor ever had a
    # torque-speed curve, because the ankle study needed it to avoid "an impossible actuator wins".
    #
    # Measured/derived at the robot's 48 V bus:
    #   AK60-39 V3.0, KV80, 39:1 (hip-roll)  -> 80 rpm/V x 48 V = 3840 rpm motor
    #                                        -> 402 rad/s / 39 = 10.31 rad/s at the joint
    #   AKE90-8, 8:1 (cam + thigh)           -> 22.0 rad/s at the joint (same spec the ankle
    #                                           study uses; implies motor KV ~35 rpm/V at 48 V)
    # A real drive is CURRENT-limited at low speed and only VOLTAGE-limited past the corner:
    #     tau(w) = min( tau_peak ,  Kt_j * (V_bus - Kt_j*w_joint) / R )
    # (back-EMF at the joint is Kt_j*w_joint exactly, because Kt_motor = Kt_j/G and w_motor = w*G).
    # The linear-from-zero derate the ANKLE uses is the wrong shape for this: it ignores the flat
    # branch and under-reports torque precisely where the robot operates.
    #
    # At the robot's 48 V bus the corner sits far above anything reached so far:
    #     hip_roll  corner 4.7-10.2 rad/s (R 2.0 -> 0.05) vs 1.48 observed
    #     cam/thigh corner 6.8-20.5 rad/s (R 0.5 -> 0.05) vs 5.40 observed
    # and R is bounded by tau_peak being reachable at all: R <= Kt_j*V/tau_peak = 0.72 ohm (AKE90),
    # 3.65 ohm (AK60). So for ANY consistent R both motors sit in the flat branch and a constant
    # forcerange is already correct. This exists for the fast-running case, where it will bite.
    #
    # Per actuator, model order (hip_roll_L, cam_L, thigh_L, hip_roll_R, cam_R, thigh_R).
    # Leave motor_r_ohm empty to disable -- R is NOT measured on this robot yet.
    motor_bus_volts: float = 48.0
    motor_kt_joint: tuple = (4.655, 2.176, 2.176, 4.655, 2.176, 2.176)   # Nm/A, output side
    motor_r_ohm: tuple = ()             # phase resistance; empty = torque-speed curve OFF
    ankle_motor_cont_nm: float = 55.0       # continuous rating; telemetry-only thermal proxy
    #                                         (fraction of time above it -> can it run continuously?)
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
    # ----- THE MEASURED DRIVE (2026-08-05 swept-sine Bode, identical on both legs) ---------------
    # The plant models the motors as <position kp=200 kv=5>, a critically damped servo at ~13 Hz.
    # The REAL drive's internal cascaded position loop is a first-order roll-off at 0.8 Hz plus a
    # ~25 ms transport delay:
    #       f Hz   0.8   1.1   1.4   2.0   2.5   2.9
    #       gain  0.71  0.60  0.53  0.38  0.30  0.28
    #       lag    -48   -61   -71   -86   -94  -104
    # (0.71 at -48 deg IS a first-order corner, so one pole fits it.) That is a >10x bandwidth
    # mismatch against a gait_freq_hz range topping out at 4 Hz, where the drive executes ~1/4 of
    # the commanded amplitude ~100 deg late. Every policy trained before this assumed authority the
    # hardware cannot deliver.
    #
    # Set drive_bandwidth_hz and the env DERIVES action_filter from the control rate as
    # exp(-control_dt / tau), tau = 1/(2*pi*f). That is the point of expressing it in Hz rather than
    # as a filter coefficient: the same PHYSICAL drive maps to 0.975 at 200 Hz and 0.904 at 50 Hz,
    # so a 50 Hz and a 200 Hz arm are the same robot rather than two different ones. Likewise
    # drive_delay_ms converts to whole control steps. 0 = legacy (use the raw coefficients above),
    # so every m1..m7 / slow / sym / wskill / ankle-study preset is bit-for-bit unchanged.
    drive_bandwidth_hz: float = 0.0
    drive_delay_ms: float = 0.0
    # DRIVE-BANDWIDTH CURRICULUM. Going from the stack's effective ~35 Hz target filter to the
    # measured 0.8 Hz in one step is the largest plant change in this project, and it MEASURABLY
    # destroys a warm start: teleop_v5 ran at ep_len 1448 / reward +511, and walk_fwd_easy -- same
    # policy, no DR, no sensor noise, no pushes, ONLY the ankle and drive changed -- collapsed to
    # 251 / -78. Same failure family as DR-at-full-width-from-step-0, which is what pinned
    # teleop_v3 until dr_curriculum_steps was added.
    # Ramps in LOG10(Hz), not Hz: the filter coefficient is wildly nonlinear in frequency
    # (alpha 0.4 -> 35 Hz, 0.6875 -> 11.9 Hz, 0.975 -> 0.8 Hz), so a linear-in-Hz ramp would do
    # almost nothing for most of its length and then fall off a cliff at the end.
    # 0 = no curriculum (start at drive_bandwidth_hz immediately), so every existing preset is
    # bit-for-bit unchanged.
    drive_bandwidth_start_hz: float = 0.0
    drive_curriculum_steps: int = 0
    # ----- motor velocity / acceleration limits (2026-07-24; sim2real + cadence) -----
    # The position servos have NO velocity or acceleration cap in the model (only kv damping + a
    # SOFT w_motor_vel reward), so the 200 Hz policy can crank the legs arbitrarily fast -> the k350
    # winner balanced m3 by pattering ONE foot at ~11 Hz (peak thigh 23 rad/s, at the 210 RPM motor
    # ceiling). These slew-limit the COMMANDED target so joint velocity <= motor_vel_limit and its
    # rate of change <= motor_accel_limit -- a velocity/accel-bounded position servo, exactly the
    # real moteus position-mode limits. 0 = off (plant byte-identical to before). NOTE 22 rad/s is
    # the MOTOR ceiling; the cam/thigh see it through the linkage reduction, so this is a loose upper
    # bound (only trims the worst spikes) -- the accel limit + cadence penalty do the real slowing.
    # Scalar applies to all six; a 6-tuple is PER JOINT in actuator order
    # (hip_roll_L, cam_L, thigh_L, hip_roll_R, cam_R, thigh_R). Per-joint matters here because the
    # two motor families differ by more than 2x: no-load OUTPUT speed is 1261 deg/s = 22.01 rad/s
    # for the AKE90-8 (cam, thigh) and 590 deg/s = 10.30 rad/s for the AK60-39 (hip-roll) -- the
    # same pair of numbers the webui's excitation sizing already uses. A single scalar has to be
    # wrong for one family or the other, and applying the AKE90's 22 rad/s to hip-roll hands the
    # policy 2.1x the lateral speed the hardware has.
    motor_vel_limit: Any = 0.0          # rad/s cap on commanded joint velocity (0 = off)
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
    # SCALE-FREE competence gate (preferred; 0 = disabled, use the air_time term above).
    #
    # ent_gate_air_time keys off reward_terms/air_time, which is a WEIGHTED, dt-SCALED per-step mean
    # -- so its numeric scale moves whenever w_air_time or the control rate moves, and the gate
    # silently stops meaning what it meant. It has already drifted twice: once at the 50->200 Hz
    # move (0.02 -> 0.005, see the m7 note below) and again since. MEASURED on m2drv_d3_s0, a policy
    # that unmistakably walks (20.75 m at 0.69 m/s, right foot airborne 0.32 s at a time) produced
    # reward_terms/air_time = 0.0002 against a 0.005 gate -- 25x short. The gate could never open,
    # ent_coef sat at 0.01 for the whole 60 M run, and log_std stayed pinned at exactly 1.000.
    #
    # swing_frac is the physical thing the gate was always trying to ask: the fraction of feet off
    # the ground, averaged over the rollout. Standing is 0.0; a walking gait is 0.3-0.4. It is
    # invariant to reward weights, to dt and to the control rate, so it cannot drift like this.
    ent_gate_swing_frac: float = 0.0
    ent_anneal_deadline_steps: int = 0  # hard fallback: begin the ent_coef anneal by this many env
    #                                     steps even if the air_time competence gate never opens
    #                                     (num_timesteps-based; 0 = disabled, gate-only). Without it
    #                                     a stuck gate pins std at max_log_std forever (m3 deadlock).
    # A deadline or an anneal span LONGER THAN THE RUN is the same failure with extra steps: the
    # walk_fwd lineage carries deadline 100 M and ent_anneal_steps 80 M (sized for the preset's
    # nominal total_steps=800 M) while the ladder actually trains 60 M stages, so neither could ever
    # fire. train.py now scales both against the REAL budget and says so -- see _size_entropy_schedule.
    ent_schedule_autoscale: bool = True
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
    # v5 (2026-08-04): v4 never got a node (25 h pending), so nothing was burned on the stale plant —
    # and in the meantime the MEASURED-mass model landed. v1..v4 all trained on dash01.xml at
    # 12.83 kg with a 0.222 kg shin; the real robot is 15.14 kg with a 0.573 kg shin (2.6x). Distal
    # leg mass is the axis the speed oracle says dominates, so that is not a detail — every teleop
    # policy so far is tuned to a robot that does not exist. v5 = v4 on the corrected plant, which
    # since 2026-08-04 is just the default dash01.xml (no model_path override needed, and none
    # possible: the mass correction is baked in).
    #
    # Mass DR narrowed because the masses are now measured; inertia DR deliberately NOT narrowed:
    # checking the two models body by body, every inertia tensor scaled by EXACTLY its body's mass
    # ratio (1.000 in all 13 bodies), i.e. the measured model has real masses with the CAD inertia
    # tensors rescaled proportionally. The mass DISTRIBUTION inside each link is still a placeholder,
    # so the radius of gyration is exactly as uncertain as it was before and dr_inertia stays wide.
    "teleop_v5": lambda: _teleop(
        dr_loop_site=0.0015,
        curriculum_retreat_frac=0.7,
        stance_ratio_final=0.62,
        efficiency_target=0.35,
        dr_curriculum_steps=120_000_000,
        push_dv=0.6,
        dr_mass_global=0.06,           # was 0.12 — masses measured, payload placement still varies
        dr_mass_body=0.08),            # was 0.15 — ditto
        # dr_inertia stays 0.25: measured masses did NOT bring measured inertia tensors
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
# The active arms use a MASSLESS motor with an AKE90-8's performance envelope (170 N*m peak,
# no-load 22 rad/s, enforced as a torque-SPEED curve every substep). That is an upper bound by
# design: we do not know yet whether an ankle motor is worth anything or what performance it would
# need, so charging it one specific motor's mass and rotor inertia would answer the narrower "is
# THIS motor worth it" and would let a loss be blamed on the hardware pick. If the upper bound
# still loses, an actuated ankle is dead on the merits. If it wins, the ankle/* telemetry (peak
# torque, peak speed, peak power, fraction of time above the 55 N*m continuous rating) IS the spec,
# and paying the real mass is worth simulating only then.
#
# Three things make it a fair test, all of which the 2026-07-24 sweep lacked:
#   1. ankle_zeta=0.7 -- damping is DERIVED from k against the measured stance inertia, so every
#      arm sits at the same damping ratio and k is genuinely the only variable. (Measured: the
#      ankle's stance inertia is 0.316 kg*m^2, 53x its swing inertia. That is why the old fixed
#      ankle_damping=1.6 was ~7.6% of critical at k=350 and rang at 5.3 Hz -- the m7 6 Hz footfall,
#      re-derived from the plant instead of from a training curve.)
#   2. The MEASURED-MASS plant (15.14 kg vs the CAD placeholder's 12.83, with a 2.6x heavier shin).
#      The spring's job is distal energy storage, so getting distal mass wrong would answer the
#      question about a robot that does not exist. Since 2026-08-04 that is just dash01.xml -- the
#      correction is baked into the one plant, so no arm can opt out of it by accident.
#   3. The active arms' torque is billed in the efficiency reward like any other actuator, and they
#      are held to a real torque-speed curve, so "active" cannot win on free energy or on an
#      actuator that delivers peak torque at any speed. (It IS given free mass — deliberately; see
#      the upper-bound note above.)
_STUDY_PLANT = {k: v for k, v in _SYM_PLANT.items()
                if k not in ("ankle_stiffness", "ankle_damping")}   # the study sets these per arm
_STUDY_PLANT.update(
    ankle_zeta=0.7,          # derived damping; overrides ankle_damping. See _setup_ankle.
)
_ACTIVE_MODEL = "model/dash01_active.xml"

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

# REMOVED 2026-08-04: study_m3_k350_oldmass, the control arm on the CAD placeholder masses. It only
# made sense while two plant files existed; now that dash01.xml IS the measured plant, its
# model_path override pointed at the same model as study_m3_k350 and it would have run as a silent
# duplicate -- the exact "clean, plausible, wrong curve" failure test_ankle_study guards against.
# The question it asked (do pre-2026-08 results transfer to the real robot?) is a question about the
# old runs, not about the ankle; answer it by re-running an old preset off a git checkout if needed.

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


# ================================================================================================
# ANKLE-2 STUDY (2026-08-05) -- can the REAL ankle, or no ankle spring at all, be stabilised?
# ================================================================================================
# The whole m1..m7 record was obtained on a k=350 ankle spring that WILL NOT BE BUILT, and the
# 2026-08-04 force map showed the real spring is 8x short of the 3.5 BW running requirement. So the
# open question is no longer "which k is best" -- it is a MECHANICAL DESIGN decision with exactly
# two candidates the user can actually build:
#
#   Q1  the real spring: k = 41.4 N*m/rad (user-measured, replaces the 28.65 spec) with NO preload.
#       Can any controller keep the robot from falling on that?
#   Q2  no spring at all: a rigid TENSION-ONLY strut (unlimited in traction, buckles past 10 N in
#       compression), which also deletes the 249 g spring assembly from each shin. Can that
#       stabilise? Distal mass costs ~0.93 m/s per kg, so this arm is cheaper AND lighter if it
#       works.
#
# Everything except the ankle is _WSKILL_PLANT, identical in every arm: the sym stack (Fourier,
# freq (0.5,4.0), contact-switch 0.20, duty-symmetry 8.0) PLUS workspace_kill. The kill is on at
# the user's call because this is a build decision and a false pass is expensive -- without it an
# arm can "succeed" one-legged with a foot parked outside the real robot's reachable box, which is
# a sim exploit, not a stable robot.
#
# Every arm sets ankle_resettle: the shipped keyframe is an equilibrium of the PRELOADED 28.65
# spring, so without it the soft arms would start every episode out of equilibrium and be measured
# against a stance they cannot hold. The k350 control gets it too, so the treatment is identical.
_ANKLE2_PLANT = {k: v for k, v in _WSKILL_PLANT.items()
                 if k not in ("ankle_stiffness", "ankle_damping")}
_ANKLE2_PLANT.update(ankle_zeta=0.7, ankle_resettle=True)

SHIN_SPRING_KG = 0.249      # spring assembly folded into the measured 573 g shin (bare shin 324 g)

ANKLE2_ARMS = {
    # --- controls, so a failure below can be read ------------------------------------------------
    # the known-good, unbuildable spring. If THIS fails on the corrected plant (144.5 N*m torque,
    # measured masses, workspace_kill) then the whole screen is uninterpretable and nothing else
    # here means anything. It is the first curve to look at.
    "k350":     dict(ankle_mode="passive", ankle_stiffness=350.0),
    # today's robot as modelled: the 28.65 spec spring, preload preserved.
    "k28_65":   dict(ankle_mode="passive", ankle_stiffness=28.65),
    # welded ankle, spring mass still aboard. Upper bound on stiffness, and the reference the
    # tension strut is measured against -- bar vs rigid isolates the cost of being UNILATERAL.
    "rigid":    dict(ankle_mode="rigid"),
    # --- Q1: the real spring -----------------------------------------------------------------
    # real k, preload PRESERVED. Not the question, but it separates "k is too low" from "removing
    # the preload is what kills it" -- without this arm a k41_4_np failure is unattributable.
    "k41_4":    dict(ankle_mode="passive", ankle_stiffness=41.4),
    # THE Q1 ARM: real k, no preload.
    "k41_4_np": dict(ankle_mode="passive", ankle_stiffness=41.4, ankle_preload="zero"),
    # --- Q2: the tension-only strut ----------------------------------------------------------
    # THE Q2 ARM: strut + the shin lightened by the deleted spring assembly.
    "bar":      dict(ankle_mode="bar", ankle_spring_mass_kg=SHIN_SPRING_KG),
    # strut with the shin mass left on: separates "the strut works" from "the lighter shin works".
    "bar_heavy": dict(ankle_mode="bar"),
    # buckling-load sensitivity. The user's spec is a 10 N buckling FORCE and the plant carries no
    # spring geometry to convert it (build_model.py models the spring as pure joint stiffness), so
    # the lever arm is unknown to within 20-60 mm = 0.2-0.6 N*m. These two bracket it. If they
    # agree, the lever arm does not matter and the answer is robust; if they disagree, the user
    # needs to tape-measure the ankle lever arm before believing either.
    "bar_lo":   dict(ankle_mode="bar", ankle_spring_mass_kg=SHIN_SPRING_KG,
                     ankle_bar_buckle_nm=0.2),
    "bar_hi":   dict(ankle_mode="bar", ankle_spring_mass_kg=SHIN_SPRING_KG,
                     ankle_bar_buckle_nm=0.6),
    # 2026-08-06: `rigid` (welded BOTH ways) completes the 57 m dash at 2.55 m/s in 5 of 6 greedy
    # episodes while `bar` (tension-only) manages 0.6 m and stands one-legged -- and the ONLY
    # difference between them is that the strut gives way in compression past 0.4 N*m. So the
    # tension-only-ness may be the whole cost, not the ankle compliance. This arm is a strut that
    # also takes real COMPRESSION (5 N*m before buckling = a proper link rather than a thin rod),
    # bracketing bar -> rigid. If it recovers rigid's performance, the build answer changes from
    # "tension-only strut" to "just make it a stiff link", which is a trivial mechanical change.
    "bar_comp": dict(ankle_mode="bar", ankle_spring_mass_kg=SHIN_SPRING_KG,
                     ankle_bar_buckle_nm=5.0),
}


def _ankle2(m, arm, **kw):
    if arm not in ANKLE2_ARMS:
        raise KeyError(f"unknown ankle2 arm {arm!r}; have {sorted(ANKLE2_ARMS)}")
    return _react(m, **{**_ANKLE2_PLANT, **ANKLE2_ARMS[arm], **kw})


# m3 (pitch free) is the screen: the cheapest rung that is still ankle-sensitive, and where every
# prior ankle result lives. m6 (all six base DOF free) is what "does not fall down" really means and
# is deliberately NOT chained -- which arms earn it is a human call on the m3 curves.
PRESETS.update({f"ankle2_m3_{a}": (lambda a=a: _ankle2("m3", a)) for a in ANKLE2_ARMS})
PRESETS.update({f"ankle2_m6_{a}": (lambda a=a: _ankle2("m6", a)) for a in ANKLE2_ARMS})

# ----- the same arms THROUGH THE MEASURED DRIVE -------------------------------------------------
# The arms above run against the plant's <position kp=200 kv=5> servo (~13 Hz) behind an
# action_filter whose equivalent bandwidth is 34.7 Hz. The real drive is 0.8 Hz + 25 ms. Which
# ankle is best is one question; whether ANY of them can be stabilised through the actuator that
# exists is a different and more decisive one, and it is the one the build decision rests on.
#
# Run at BOTH control rates on purpose. Because drive_bandwidth_hz is specified in Hz and converted
# at the control rate, the 200 Hz and 50 Hz arms are the SAME PHYSICAL ROBOT -- so the pair
# isolates "does the control rate matter" from "does the drive matter". If they tie, 200 Hz buys
# nothing through this drive and the Pi only ever needs 50 Hz; if 200 Hz wins, the rate is real and
# the drive is the thing to fix. Compute is not the constraint either way: the policy is a 275->64
# ->64->24 MLP (46 kFLOP, 9.3 MFLOP/s at 200 Hz, ~0.1% of a Pi 4) and CAN is 26% of a 1 Mbps bus.
_DRIVE = dict(drive_bandwidth_hz=0.8, drive_delay_ms=25.0)

PRESETS.update({
    f"ankle2drv_m3_{a}": (lambda a=a: _ankle2("m3", a, **_DRIVE))
    for a in ("k350", "bar", "k28_65")})
def _at_50hz(c):
    """Take a 200 Hz preset back to 50 Hz *at equal ROBOT TIME*, not equal step count.

    _HZ200 multiplied every step-denominated schedule by 4 when the control rate went 50 -> 200 Hz,
    and set gamma to 0.99^(1/4) to hold the ~2 s horizon. Running those same numbers at 50 Hz would
    make every curriculum advance 4x slower in seconds-of-robot, and shorten the discount horizon to
    0.5 s -- so a 50 Hz arm would lose to a 200 Hz arm for reasons that have nothing to do with the
    control rate, which is the one thing the pair exists to measure. Everything counted in control
    steps has to come back down by 4, including the ep_len competence GATES (1600 steps is 8 s at
    200 Hz but 32 s at 50 Hz, which nothing ever reaches, so the gated curricula would never open).

    action_filter / action_delay_steps are deliberately NOT touched: drive_bandwidth_hz and
    drive_delay_ms re-derive them from the control rate, which is the whole point of specifying the
    drive in Hz."""
    c.control_decimation = 20
    c.gamma = 0.99                                   # 0.9975^4 — same ~2 s horizon
    c.ent_gate_air_time = 0.02                       # per-step mean of an event term scales with dt
    for f in ("total_steps", "gait_curriculum_steps", "efficiency_ramp_steps",
              "sprint_curriculum_steps", "ent_anneal_steps", "ent_anneal_deadline_steps",
              "jitter_curriculum_steps", "dr_curriculum_steps", "pitch_armature_ramp_steps"):
        v = getattr(c, f, 0)
        if v:
            setattr(c, f, int(v // 4))
    for f in ("curriculum_gate_ep_len", "jitter_curriculum_gate_ep_len"):
        v = getattr(c, f, 0.0)
        if v:
            setattr(c, f, float(v) / 4.0)
    return c


PRESETS.update({
    f"ankle2drv50_m3_{a}": (lambda a=a: _at_50hz(_ankle2("m3", a, **_DRIVE)))
    for a in ("k350", "bar", "k28_65")})


# ================================================================================================
# walk_fwd (2026-08-06) -- THE UNTETHERED-WALKING TARGET
# ================================================================================================
# Goal: walk forward, untethered, on a real floor. Not a speed record, not teleop -- steering is
# deferred but PROVISIONED (see below).
#
# This is an ADAPTATION run, not a from-scratch one. teleop_v5 reached 321 M steps as an m6
# walking policy with the full sim2real package, and none of the changes below touch obs or action
# width, so that checkpoint loads directly. Growing a walker from zero would not fit the schedule;
# adapting one does.
#
# What changes vs teleop_v5, all of it measured on the actual robot rather than assumed:
#   ankle       -> the `bar_comp` strut: the ankle2 screen's winner (ep_len 2307 at 29 M vs the
#                  tension-only strut's 0.6 m dash) AND 249 g/leg lighter. Note the user's as-built
#                  3 mm PETG rod buckles at 0.27 N, ~35x below spec -- this arm assumes the carbon
#                  replacement.
#   drive       -> the MEASURED 0.8 Hz + 25 ms position loop, in place of a fictional ~35 Hz filter.
#   friction    -> (0.35, 0.60). Measured slip angle was 40 deg = mu 0.84; training well below it
#                  is deliberate, since the failure mode is planting a foot that then slides.
#   homing      -> 5 deg per-joint zero error on BOTH obs and command (user calibrates to 1-2 deg
#                  and re-records the safe workspace each run, so 5 is the safety envelope).
#   IMU         -> 10 deg mounting rotation on gravity AND gyro, plus stale-frame dropout.
#   voltage     -> within-episode torque droop.
#
# STEERING IS PROVISIONED, NOT ENABLED: steer_enable stays True so the action keeps its 26 dims and
# every checkpoint in this lineage stays loadable, but cmd_yaw_max = 0 so nothing is asked of it
# yet. Turning steering on later is then a curriculum widening rather than a reshape that orphans
# the run. Enabling it later WITHOUT this would break every checkpoint -- the reason it is here.
#
# The drive bandwidth rides the DR curriculum rather than being switched on at step 0. Going from a
# ~35 Hz filter to 0.8 Hz is the largest single plant change in this project's history and would
# otherwise destroy the warm start on contact.
def _walk_fwd(**kw):
    base = dict(
        # --- the ankle2 winner ---
        ankle_mode="bar", ankle_bar_buckle_nm=5.0, ankle_spring_mass_kg=SHIN_SPRING_KG,
        ankle_resettle=True, ankle_damping=0.0, ankle_zeta=0.0,
        # --- the measured drive ---
        drive_bandwidth_hz=0.8, drive_delay_ms=25.0,
        # ease in from an optimistic drive; applying 0.8 Hz at step 0 measurably destroys the
        # teleop_v5 warm start (ep_len 1448 -> 251 with nothing else changed)
        drive_bandwidth_start_hz=12.0, drive_curriculum_steps=120_000_000,
        # --- measured sim2real ---
        dr_friction=1.0, dr_friction_range=(0.35, 0.60),
        dr_joint_zero_deg=5.0,
        dr_imu_rot_deg=10.0, dr_imu_dropout_prob=0.0015, dr_imu_dropout_s=0.25,
        dr_torque_sag=0.15, dr_torque_sag_tau_s=2.0,
        # --- forward only, steering provisioned ---
        cmd_v_fwd_max=1.0, cmd_v_back_max=0.3, cmd_yaw_max=0.0,
        # --- keep the v3 lessons: gated+retreating curricula, DR ramped not slammed ---
        curriculum_retreat_frac=0.7, efficiency_target=0.35,
        stance_ratio_start=0.70, stance_ratio_final=0.62,
        dr_curriculum_steps=80_000_000,
    )
    base.update(kw)
    return _teleop(**base)


PRESETS.update({
    "walk_fwd": lambda: _walk_fwd(),
    # bring-up rig: base clamped, so balance is provided by the clamp and the IMU truthfully reads
    # a constant. Short fine-tune off the same trunk so the rig has a policy that cycles its legs
    # without needing to balance. NOT the deployment policy.
    "walk_fwd_rig": lambda: _walk_fwd(push_interval_s=0.0, trip_prob=0.0, dr_imu_dropout_prob=0.0),
    # diagnostic: no adversity at all. If walking does not survive the ankle+drive change HERE,
    # the problem is the plant change, not the randomisation.
    "walk_fwd_easy": lambda: _walk_fwd(
        dr_enable=False, obs_noise_enable=False, push_interval_s=0.0, trip_prob=0.0,
        ctrl_jitter_ms_final=0.0, ctrl_drop_prob_final=0.0),
    # --- v2 (2026-08-08): walk_fwd DEADLOCKED. Measured, both seeds, 530 M steps ------------------
    # walk_fwd_s0/s1 sat at ep_len ~500 with dr_scale EXACTLY 0.00 and drive_bw pinned at its 12 Hz
    # start for the entire run. Cause: curriculum_gate_ep_len=1600 gates ALL SIX ramps at once
    # (stance, drive_bw, eff, dr, jitter, drop), the policy never reached 1600, so no curriculum
    # ever moved -- while the sim2real calibration axes (5 deg homing, 10 deg IMU rotation) were
    # NOT gated and stood at full width from step 0. Full sensor adversity, zero dynamics
    # randomisation, zero efficiency regularisation, and a fake 12 Hz drive: the worst of both
    # worlds, with the only exit gated on the thing it could not do. This is the SAME failure the
    # _teleop comment above already documents ("measurably BACKFIRED", eff_scale pinned at 0) --
    # the gate was re-introduced here anyway.
    #
    # Three changes, all pointed at that one mechanism:
    #   (a) warm-start from walk_fwd_easy_s0, which FINISHED the drive curriculum and holds
    #       ep_len 4938 (peak 6833) at the real 0.8 Hz -> start above any sane gate instead of
    #       under it, and start on the real drive rather than needing to ramp onto it.
    #   (b) drive curriculum OFF (start_hz=0) -- the parent already lives at 0.8 Hz, so re-easing
    #       to 12 Hz would only teach it a drive that does not exist.
    #   (c) gates 1600 -> 400, so the ramps advance from step 0 at the parent's competence and
    #       retreat (retreat_frac 0.7 -> floor 280) if it collapses. The ramp is the safety, not
    #       the gate.
    # Paired with SensorNoise.scale (domain_rand.py): homing/IMU/dropout/sag now ride dr_scale.
    # Order inverted on purpose: learn the hard PLANT first, then add adversity.
    "walk_fwd2": lambda: _walk_fwd(
        drive_bandwidth_start_hz=0.0, drive_curriculum_steps=0,
        curriculum_gate_ep_len=400.0, jitter_curriculum_gate_ep_len=400.0),
    # --- v3 (2026-08-09): walk_fwd2 plateaued at 2.0 s for 300 M steps ---------------------------
    # The curricula were all moving this time, so it was not another deadlock. Two paired-seed
    # ablations on the walk_fwd_easy_s0 walker (12 episodes, common random numbers, clean 0.8 Hz
    # plant -- unpaired samples are worthless here, outcomes are bimodal and a 5-episode draw of
    # the SAME condition gave 35.6 s and 1.8 s) found two separate causes:
    #
    # (1) COMMANDS THE PLANT CANNOT DELIVER. Survival vs the forced command:
    #        -0.10 m/s  39.3 s med  5/12      +0.00 m/s  27.4 s med  2/12
    #        +0.20 m/s  24.9 s med  3/12      +0.40 m/s   3.1 s med  0/12
    #        -0.20 m/s   2.7 s med  5/12      +0.60 m/s   1.5 s med  0/12
    #     Backward is FINE -- it is the best case. Forward above ~0.3 m/s is fatal, and the policy
    #     only ever achieves ~0.14 m/s anyway. But cmd_v_fwd_start is 0.6, so the EASIEST command
    #     the curriculum can ask for at cmd_scale=0 is already 2-3x beyond the plant, and cmd_scale
    #     sat at 0.0 (its 1200-step gate never opened) so the box never narrowed. That value was
    #     inherited from the teleop lineage, whose drive was effectively instantaneous; it was never
    #     re-tuned when the drive dropped to the measured 0.8 Hz. Start at 0.25 m/s -- inside what
    #     the robot demonstrably does -- and let cmd_scale earn its way up to 1.0.
    #
    # (2) PUSHES AND TRIPS AT FULL WIDTH FROM STEP 0. Each takes the walker from 3/12 surviving
    #     episodes to 0/12, the same cost as full-width plant DR, which already had a ramp. They
    #     were the last two adversity axes without one. adversity_curriculum=True puts them on
    #     dr_scale with everything else.
    #
    # NOT changed, deliberately: cmd_v_back_max stays non-zero. The measurement says backing up is
    # the one thing this policy is good at, so removing it would delete a working skill to fix a
    # problem it does not have -- which is exactly what the first hypothesis here would have done.
    "walk_fwd3": lambda: _walk_fwd3(),
    # STANDING STILL OUT-SCORES WALKING, measured on ladder_m2_s0 at 10 M steps: it reached
    # ep_len 10796 (54.0 s) and reward +9752 without locomoting -- both feet grounded 86-99% of
    # the time, 7.6 footfalls/s of chatter, 0.2-1.2 m covered in a MINUTE (~0.01 m/s). At 12 Hz
    # drive with a ZERO command it moves 0.207 m/s and falls, so it had learned to emit large
    # actions and use the 0.8 Hz filter as a brake.
    #
    # The arithmetic, at cmd +0.30 m/s (w_track_lin 4.0, sigma_min 0.2, w_alive 0.5, fall 100):
    #     stand still, survive 60 s   0.922/step  ->  ~2766 per episode
    #     track perfectly, fall at 5s 4.500/step  ->   1125 - 100 = 1025
    # Standing wins by 2.7x. Two causes, both fixed here:
    #   * sigma_min 0.2 pays 0.42 for ZERO velocity against a 0.3 command -- partial credit for
    #     doing nothing. 0.15 cuts that to 0.07 while keeping a usable gradient (e=2, not e=3:
    #     tightening further zeroes the gradient too and the policy simply never learns to move).
    #   * w_alive 0.5 is a survival wage payable for idling. The sprint lineage that actually ran
    #     at 2.55 m/s kept w_alive=0 and paid a CLOCK COST instead. A clock cost is wrong for a
    #     command objective (standing is the correct response to cmd=0), so just drop the wage.
    # Suicide-safe without it: step_reward_floor bounds per-step loss at -0.25 (dt-scaled) while
    # dying costs fall_penalty 100, so living always out-values diving.
    # Applied to the LADDER only -- walk_fwd3 keeps the old reward so it stays a clean control.
    # That does mean walk_fwd_m6 is no longer byte-identical to walk_fwd3.
    # --- THE LADDER (2026-08-09) ----------------------------------------------------------------
    # The whole teleop -> walk_fwd lineage trains at m6, ALL SIX base DOFs free, from step 0 --
    # verified across every resolved_config in the archive: teleop, teleop_easy, v2, v3, v5,
    # walk_fwd, walk_fwd_easy, walk_fwd2 are all base_lock (0,0,0,0,0,0). It is the one lineage in
    # this project that skipped the base-DOF ladder, and the one that will not converge: three
    # plateaus, ~1.4 G steps, dr_scale self-limiting at 0.27.
    #
    # Everything that ever worked on this robot used the ladder -- 97 archived runs sit at m3, and
    # the only controller to reach 2.55 m/s on the MEASURED plant (ankle2_m3_rigid) was m3, with
    # roll and yaw locked. It got there in 73 M steps; m6 has spent 300 M going nowhere.
    #
    # Each rung differs from the next by base_lock and NOTHING else, so the reward, obs (550) and
    # action (26) are identical all the way up and every stage warm-starts the one below. m5 is the
    # rung to watch: it frees ROLL, and the CPG A/B walled there at matched budget.
    # m2 is the SPEED prior: x and z free, y/roll/pitch/yaw locked, so the robot cannot fall and
    # the only thing left to learn is how the gait converts into forward velocity — the exact
    # deficit measured on the m6 policy (0.14 m/s achieved against a 0.6 m/s command, and 0/12
    # survival at +0.40). Deliberately kept SHORT: with pitch locked a policy can lunge with no
    # consequence, and m2 -> m3 is the historically hard transition here (the whole m3 anti-topple
    # sweep). We want a gait prior, not a policy overfitted to a plant that cannot topple.
    # --- DRIVE-BANDWIDTH BRACKET on m2 (2026-08-10) ---------------------------------------------
    # Is the 0.8 Hz drive what stops it walking? A first-order lag keeps, of commanded amplitude:
    #     0.5 Hz gait 85%   0.8 Hz 71%   1.5 Hz 47%   3.8 Hz 21%
    # and the finished m2 policy runs a 3.8 Hz gait (7.6 footfalls/s), so only a FIFTH of the
    # commanded swing survives the drive, 78 deg late. That is a very plausible reason the feet
    # never leave the ground. It cannot be settled by replaying a 0.8 Hz-trained policy at higher
    # bandwidth (tried: it moves faster and falls) -- the policy has to be TRAINED there.
    #
    # Three cold m2 arms, identical but for drive_bandwidth_hz, all keeping the measured 25 ms
    # transport delay and with workspace_kill ON. d08 re-runs the baseline because the original
    # ladder_m2_s0 predates workspace_kill and is no longer a fair control.
    #   d08  the measured robot
    #   d3   above walking cadence, below the policy's preferred 3.8 Hz limit cycle
    #   d12  effectively unlimited -- the regime the 2.55 m/s runner trained in
    # If d3/d12 walk and d08 does not, the drive is the binding constraint and the fix is
    # HARDWARE (control mode / bandwidth), not more training.
    "walk_fwd_m2_d08": lambda: _walk_fwd3(base_lock=LOCKS["m2"], drive_bandwidth_hz=0.8, **_LADDER_RWD),
    "walk_fwd_m2_d3":  lambda: _walk_fwd3(base_lock=LOCKS["m2"], drive_bandwidth_hz=3.0, **_LADDER_RWD),
    "walk_fwd_m2_d12": lambda: _walk_fwd3(base_lock=LOCKS["m2"], drive_bandwidth_hz=12.0, **_LADDER_RWD),
    # ----- THE LADDER, on the RETUNED drive and the REBUILT ankle (2026-08-10) --------------------
    # ankle_mode "rigid", not "bar". The ankle has been rebuilt as a carbon tube: stiff, symmetric
    # in push AND pull, with no buckling limit. "bar" models a TENSION-ONLY strut that saturates at
    # ankle_bar_buckle_nm in compression, which is what the old 3 mm PETG rod was (measured to
    # buckle at 0.27 N, 35x below spec). That asymmetry is now gone from the hardware, so carrying
    # it in the plant would be modelling a part that no longer exists. "rigid" is also the arm that
    # produced the only 2.55 m/s controller this project has ever had (ankle2_m3_rigid).
    # The bracket above settled it: at the measured 0.8 Hz the robot cannot walk (d08, duty 1.00,
    # falls), at 3 Hz it does (d3: 20.75 m at 0.69 m/s, per-foot air 0.269/0.184, 60 M steps), and
    # 12 Hz is WORSE than 3 (d12 folds a leg into workspace_kill). So the fix was hardware, and it
    # has been made: position-loop kp 0.003 -> 0.009 on the sagittal boards moved left.thigh
    # 0.86 -> 3.35 Hz and left.cam 0.85 -> 2.88 Hz, with left.abd (untouched, 0.74 -> 0.76) as the
    # in-run control. The ladder therefore trains at the drive that now exists, not the old one.
    #
    # ASSUMPTION, and the one to watch: 3.0 Hz UNIFORM across all six joints. Measured so far only
    # on the LEFT cam+thigh. Abduction is a different actuator (AK60-39 at 39:1 vs AKE90-8 at 8:1)
    # and at the same kp sits at 0.76 Hz; f_c ~ kp/gear predicts it needs kp ~ 0.036, which is an
    # extrapolation ACROSS motor families and is not yet confirmed by a chirp. If abduction comes
    # back short, this preset is optimistic exactly where roll matters -- m5 and m6 -- and those two
    # rungs need re-baselining against a per-joint bandwidth. m2..m4 lock roll and are unaffected.
    "walk_fwd_m2": lambda: _walk_fwd3(base_lock=LOCKS["m2"], drive_bandwidth_hz=3.0,
                                      ankle_mode="rigid",
                                      **_LADDER_RWD),  # x,z free — cannot fall
    "walk_fwd_m3": lambda: _walk_fwd3(base_lock=LOCKS["m3"], drive_bandwidth_hz=3.0,
                                      ankle_mode="rigid",
                                      **_LADDER_RWD),  # + pitch
    "walk_fwd_m4": lambda: _walk_fwd3(base_lock=LOCKS["m4"], drive_bandwidth_hz=3.0,
                                      ankle_mode="rigid",
                                      **_LADDER_RWD),  # + lateral translation
    "walk_fwd_m5": lambda: _walk_fwd3(base_lock=LOCKS["m5"], drive_bandwidth_hz=3.0,
                                      ankle_mode="rigid",
                                      **_LADDER_RWD),  # + ROLL <- the wall
    "walk_fwd_m6": lambda: _walk_fwd3(base_lock=LOCKS["m6"], drive_bandwidth_hz=3.0,
                                      ankle_mode="rigid",
                                      **_LADDER_RWD),  # + yaw
    # ----- THE MIT-DRIVE RUNGS (2026-08-26) -------------------------------------------------------
    # Force control over the extended-id frame (robot/MIT_PROTOCOL.md) bypasses the drive's internal
    # position planner -- the 0.8 -> 3.0 Hz roll-off above IS that planner. Measured on both motor
    # families: ~6-7 ms command->response delay at 200 Hz command rate, inertia-independent (servo
    # SET_POS was ~200 ms). The law the drive runs in this mode, tau = Kp(qd-q) - Kd*qdot + tau_ff,
    # is the same law the plant's <position kp kv> actuator already computes, so the sim change is
    # lag only: 12 Hz stands in for the UNIDENTIFIED MIT closed-loop bandwidth (kept below the
    # plant servo's own ~13 Hz pole; replace with the chirp number once mit_identify.py is ported
    # to the real wire format and run on the stiff leg), and 25 -> 7 ms delay (1 step at 200 Hz).
    # Command START box 0.25 -> 0.50 fwd: 0.25 was sized for the 0.8 Hz planner, the cmd gate has
    # never opened anywhere in this lineage, and the d3 walker already travels at 0.69 m/s.
    "walk_fwd_m2_mit": lambda: _walk_fwd3(base_lock=LOCKS["m2"], drive_bandwidth_hz=12.0,
                                          drive_delay_ms=7.0, ankle_mode="rigid",
                                          cmd_v_fwd_start=0.50, cmd_v_back_start=0.15,
                                          **_LADDER_RWD),
    "walk_fwd_m3_mit": lambda: _walk_fwd3(base_lock=LOCKS["m3"], drive_bandwidth_hz=12.0,
                                          drive_delay_ms=7.0, ankle_mode="rigid",
                                          cmd_v_fwd_start=0.50, cmd_v_back_start=0.15,
                                          **_LADDER_RWD),
    "walk_fwd_m4_mit": lambda: _walk_fwd3(base_lock=LOCKS["m4"], drive_bandwidth_hz=12.0,
                                          drive_delay_ms=7.0, ankle_mode="rigid",
                                          cmd_v_fwd_start=0.50, cmd_v_back_start=0.15,
                                          **_LADDER_RWD),
})

# ----- THE FOOT-SHAPE ARMS (2026-08-11) ----------------------------------------------------------
# Every controller this project has produced walks on a 25 mm point toe, which has EXACTLY ZERO
# centre-of-pressure authority IN ANY DIRECTION: it can carry no moment about the contact, so the
# only way to arrest a lean is to move the foot, and moving the foot needs a step the 4-bar cannot
# deliver in time (feet planted ~8 cm BEHIND the CoM while falling forward, 0.0 N on the swung foot
# through an entire fall -- [[m3-ankle-stiffness-foot-ahead]], [[scripted-walk-controller]]).
# The finished ladder3 chain says where that costs the most: m2 clears (one seed WALKS), m3 drops
# to ~2 s, and m5 -- the rung that unlocks ROLL -- is a flat wall, both seeds at ~400 ep_len for
# 80 M with no slope while the retreating curriculum backed off and bought them nothing.
# A foot with a FOOTPRINT changes that premise, and the two arms below split it by axis.
#
#   _blade  25 mm radius cylinder, 100 mm long, axis ACROSS the robot. The sagittal profile is
#           bit-identical to the shipped ball -- it still rolls fore-aft carrying no pitch moment --
#           and the point contact becomes a 100 mm line. Buys LATERAL CoP only. Also the running-
#           blade geometry: curved in the sagittal plane, wide across it.
#   _flat   30 x 100 x 10 mm plate, levelled at the stance. Lateral CoP +-50 mm as the blade, plus
#           +-15 mm of SAGITTAL CoP the blade does not have.
#
# So the pair is a decomposition, not two goes at the same idea: sphere->blade isolates roll,
# blade->flat isolates the pitch increment. Everything else -- reward, obs (550), action (26),
# drive, ankle, curricula -- is the ladder rung it is named after, so ladder3_* IS the control and
# each rung warm-starts the one below it exactly as before.
#
# The ANKLE is what makes the plate mean anything: ankle_mode "rigid" welds it, so a moment about
# the contact patch is reacted into the shin instead of just folding the foot flat. On the shipped
# passive spring a plate would be a floppy paddle and the study would measure the spring.
_FOOT_MODEL = {"flat": "model/dash01_flat.xml", "blade": "model/dash01_blade.xml"}

PRESETS.update({
    f"walk_fwd_{_m}_{_k}": (
        lambda _m=_m, _k=_k: _walk_fwd3(base_lock=LOCKS[_m], drive_bandwidth_hz=3.0,
                                        ankle_mode="rigid", model_path=_FOOT_MODEL[_k],
                                        **_LADDER_RWD))
    for _m in ("m2", "m3", "m4", "m5", "m6") for _k in ("flat", "blade")
})


# workspace_kill ON: terminate when a toe leaves the MEASURED reachable box. walk_fwd inherited
# it OFF from _teleop, so nothing has been enforcing reachability -- the policy was free to use toe
# positions the physical 4-bar cannot produce, and its characteristic failure (sinking) is exactly
# the shape an out-of-workspace exploit would take. The 2.55 m/s runner (ankle2_m3_rigid) had it ON.
# Needs ankle_resettle (already set) so _ws_ref is re-referenced to THIS arm's settled stance.
# The ladder grades entropy competence on the PHYSICAL per-foot airborne fraction (WORSE foot), not
# on the weighted air_time term whose scale drifted 25x out from under the old gate.
# 0.13 is set from the measured bracket, sampling actions the way training does:
#     d08 (shuffles, never walks)          min-foot 0.019   -> shut, 7x margin
#     d12 (parks a leg: 0.092/0.582)       min-foot 0.092   -> shut, correctly rejected
#     d3  (walks 20.75 m at 0.69 m/s)      min-foot 0.184   -> OPEN, 1.4x margin
# Taking the WORSE foot is what rejects d12: its mean swing is the HIGHEST of the three purely
# because it folds one leg up until workspace_kill fires. A mean-based gate would have opened for
# the one arm that cannot walk.
# Every one of those reads "shut" on the old air_time gate, including the walker -- which is the bug.
# NO-LOAD OUTPUT SPEED per actuator, in actuator order. Until 2026-08-10 the position servos had
# NO velocity cap at all (motor_vel_limit defaulted to 0 = off) for this whole lineage, so the
# policy could crank a joint arbitrarily fast for free -- which is exactly what it did: the k350 m3
# winner "balanced" by pattering ONE foot at ~11 Hz with peak thigh 23 rad/s, past the motor
# ceiling. Torque was always capped; speed never was.
#   AK60-39  (hip-roll)   590 deg/s = 10.30 rad/s
#   AKE90-8  (cam, thigh) 1261 deg/s = 22.01 rad/s
# Same numbers the webui excitation sizing uses. This slew-limits the COMMANDED target, i.e. models
# a velocity-bounded position servo; it does not stop the joint being back-driven faster, which is
# correct -- gravity can outrun a motor.
_NOLOAD_RADS = (10.30, 22.01, 22.01, 10.30, 22.01, 22.01)

# The rebuilt ankle is a 40 g CARBON TUBE. dash01.xml bakes the spring's 249 g into the measured
# 573 g shin (bare shin 324 g), and ankle_spring_mass_kg is the amount REMOVED from that shin -- so
# the tube is a NET removal of 249 - 40 = 209 g, leaving 364 g of shin. Not 40, and not 249.
# This is the single most valuable number in the rebuild: distal leg mass costs ~9x what torso mass
# costs on this robot, so 209 g off each shin is worth more than 1.8 kg off the base.
_ANKLE_TUBE_KG = 0.040
_SHIN_TUBE_REMOVAL_KG = SHIN_SPRING_KG - _ANKLE_TUBE_KG      # 0.209

_LADDER_RWD = dict(w_alive=0.0, track_sigma_min=0.15, workspace_kill=True,
                   ent_gate_swing_frac=0.13, motor_vel_limit=_NOLOAD_RADS,
                   ankle_spring_mass_kg=_SHIN_TUBE_REMOVAL_KG,
                   # asymmetric critic + velocity-estimator head (see obs_privileged_critic)
                   obs_privileged_critic=True,
                   # anti-limp: the d3 walker measured duty 0.64 L / 0.31 R. w_duty_sym is the
                   # purpose-built counter-term (a foot that never bears load is expensive) and it
                   # was sitting at 0. Values from the sym_gait lineage, which was built against
                   # exactly this failure: 8.0 paired with contact_switch 0.20, because duty
                   # symmetry alone can be met by both feet CHATTERING fast — the switch penalty
                   # keeps it slow.
                   w_duty_sym=8.0, w_contact_switch=0.20)


def _walk_fwd3(**kw):
    """walk_fwd2 + the three 2026-08-09 fixes. Every ladder rung is this, with only base_lock
    changed, so a stage's checkpoint always loads into the stage above it."""
    base = dict(
        drive_bandwidth_start_hz=0.0, drive_curriculum_steps=0,
        curriculum_gate_ep_len=400.0, jitter_curriculum_gate_ep_len=400.0,
        adversity_curriculum=True,
        cmd_v_fwd_start=0.25, cmd_v_back_start=0.10,
        # (3) cmd_yaw_start was 0.3 while cmd_yaw_max is 0.0, so the yaw "curriculum" ramped DOWN
        # and at cmd_scale=0 -- where it sat for every run in this lineage -- the policy was being
        # commanded +/-0.3 rad/s of steering the whole time. walk_fwd_easy_s0's curriculum.json
        # records cmd_yaw 0.3, so the run described as "forward only, steering provisioned but
        # disabled" was in fact steering-commanded from step 0. This is an internal-consistency
        # fix (start must not exceed max), NOT a measured win -- unlike (1) and (2) above.
        cmd_yaw_start=0.0,
        cmd_curriculum_gate_ep_len=400.0,
    )
    base.update(kw)
    return _walk_fwd(**base)


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
