"""RobotDaemon — the single thread that owns both CAN buses and runs the 200 Hz control loop.

Mirrors the proven loop structure of fixed_gait/play_trajectory.py:286-373 (drain -> compute ALL
targets -> workspace-check up front -> command -> safety sweep) and the limp discipline of
fixed_gait/record_trajectory.py:142-144 (limp = STREAM SET_CURRENT 0, never just stop sending).

Frames: raw motor degrees only at the CAN boundary; the workspace check, recording buffers and
published telemetry use the normalized zero-pose frame (see calibration.py).

Modes:
    LIMP            zero current streamed; telemetry runs; backdrivable
    MANUAL          per-actuator position hold (sliders) and/or per-actuator sine — SET_POS
    RECORD_GAIT     limp + sample takes for the gait recorder
    RECORD_WS       limp + sample segments for the workspace sweep
    PLAYBACK        trajectory replay, 'position' or 'current' control law
    POLICY          a trained policy bundle, in force control, under the deploy safety governor
    ESTOPPED        latched; zero current streamed until cleared
HTTP handlers never touch the buses — they only post requests into this object and read snapshots.
"""
import os
import threading
import time
import traceback

import numpy as np

import paths
import blackbox
import canio
import ringbuffer

import play_trajectory as pt              # fixed_gait/ — Motor, drain, reconstruct-side helpers
import trajectory as traj                 # fixed_gait/
import thermal_excite                     # robot/deploy/ — the burst law + its envelope
# robot/deploy/, all three pure (numpy / struct / json) and importing nothing from the webui: the
# force-control wire format, the model<->motor joint map, and the policy safety governor. The
# dependency runs one way on purpose, so the deployed control law stays reviewable on its own.
import mit
import jointmap as JM
import safety as SAFE

# ===================================================================== operator bypasses
# Four software limits the operator can switch off deliberately. They exist because every one of
# them is tuned for a plant that keeps changing: the workspace polygon erodes as the robot is
# rebuilt, the playback current cap was chosen for a different leg, and a tracking threshold that
# is right for homing is wrong for a chirp. Refusing to let the person who owns the machine
# override them just means they run the procedure somewhere the recorder cannot see it.
#
# What a bypass does NOT touch, ever:
#   * the joints' own hard limits -- _validate_pose checks those BEFORE the workspace, so a
#     workspace bypass cannot walk a joint into a stop
#   * the drive's own firmware phase-current limit, which is not ours to raise
#   * the E-STOP, the fall/tilt kill, drive error codes, over-temperature, and telemetry staleness
#
# Bypasses are NEVER persisted. A daemon restart re-arms every one of them, because the state a
# bypass leaves behind outlives the reason for it.
BYPASS_NAMES = ("workspace", "speed", "torque", "tracking")
# A torque bypass raises the software cap; it does not remove it. Unbounded is not a setting.
BYPASS_CURRENT_CEILING_A = 40.0

MODES = ("LIMP", "MANUAL", "RECORD_GAIT", "RECORD_WS", "PLAYBACK", "MEASURE", "THERMAL",
         "IDENTIFY", "POLICY",
         "ESTOPPED")
MODE_CODE = {m: i for i, m in enumerate(MODES)}
blackbox.register_modes(MODES)            # so a recorded mode byte is decodable offline
TICK_HZ = 200.0
TELEMETRY_DIV = 10                        # ring/snapshot update every Nth tick (=> 20 Hz)
MAX_TEMP_C = 80                           # run_hardware.py:102
MAX_TRACK_ERR_DEG = 25.0                  # run_hardware.py:101 (position-command modes)
DEFAULT_SLEW_DPS = 60.0
# base never-exceed clamps, normalized deg (model joint ranges + small margin; cam +-1.5 rad,
# thigh +-1.047 rad, abduction +-0.785 rad — mujoco/dash01/build_model.py J dict).
# NOTE: the URDF ranges are CAD guesses (calibrate_workspace.py docstring) and the real cam is
# multi-turn — a recorded workspace can legitimately exceed these, so _hard_bounds() widens the
# net to the demonstrated envelope (+10 deg) whenever a workspace is loaded for that leg.
HARD_CLAMP = {"abd": 48.0, "cam": 88.0, "thigh": 62.0}
HARD_WIDEN_DEG = 10.0

# ===================================================================== pre-move safety (2026-08-10)
# On 2026-08-10 a joint destroyed itself: left.cam was commanded absolutely against a calibration
# whose raw origin had moved underneath it, and the drive happily wound the joint to ~678 deg
# normalized (~1.9 output turns) on a +-88 deg axis. Four independent mechanisms now bound that,
# and NONE of them adds a step to the normal zero -> home workflow (see BLACKBOX_TASK.md #4).
#
# (a) HOLD BEFORE MOVING. The first CAN command after leaving LIMP is set_pos(pos_raw as measured
#     right now) for HOLD_BEFORE_MOVE_S. It never consults calibration.offsets, so it is correct
#     even if the zero is nonsense, and it catches the legs sagging under gravity immediately —
#     which is what the operator actually wants after a capture. Only then do we slew anywhere.
HOLD_BEFORE_MOVE_S = 0.35
# (b) RAW-AT-REST. Compare live pos_raw against the fingerprint captured at the last successful
#     zero. Decisive only when we cannot vouch for continuity ourselves (a calibration restored
#     from disk, never re-captured in this process) — after a power cycle the drives re-randomise
#     their raw origin, so a stale file is exactly the trap. Within a live session the continuity
#     watchdog below is used instead, because gravity sag legitimately moves the joints between the
#     capture and the move and must not be mistaken for an origin jump.
RAW_AT_REST_TOL_DEG = 10.0
# ...and the sanity check that is decisive in every case: does calibration + reported raw imply a
# pose the joint can physically be in? 678 deg on a 88 deg axis fails this by 7x.
RANGE_SANITY_SLACK_DEG = 30.0
# (c) Homing is a slow guided slew and should track well, so the 25 deg playback threshold is far
#     too loose for it — a stale zero shows up as tracking error within the first few degrees.
MAX_TRACK_ERR_HOMING_DEG = 8.0
# (d) TRAVEL BUDGET, the backstop for when everything else has been fooled: a guided move that has
#     travelled this multiple of a joint's own range without arriving is not going to arrive.
TRAVEL_BUDGET_FACTOR = 1.3
TRAVEL_ARRIVED_DEG = 2.0

# Continuity watchdog: pos_raw stepping this fast while the joint is LIMP and the drive itself
# reports it is not turning is not motion — it is the driver board's multi-turn origin moving on
# its own (the open question from 2026-08-10). Rate-based, not per-tick, because the loop slips
# ~12% and a doubled dt must not read as a jump.
RAW_JUMP_DEG = 4.0
RAW_JUMP_DPS = 800.0
RAW_JUMP_SPD_MAX = 200.0                  # |ERPM| under which we believe "not turning"

# Warn levels: BELOW the trip thresholds, so a Tier B dump catches the approach and not only the
# fall. These never stop the robot — they only ask the black box to preserve the window.
WARN_TRACK_ERR_DEG = 12.0
WARN_SPD_ERPM = 12000.0
WARN_TEMP_C = 70

# Playback cycle period, seconds. Two independent reasons for the floor:
#
#   1. phase = play_t / period, so a 0 from the PATCH endpoint (no schema, takes what it is handed)
#      is a ZeroDivisionError in the CAN thread -- a daemon death with the motors mid-gait.
#   2. HARDWARE (2026-08-04, after a part failed during a high-cadence run). 0.4 s = 2.5 Hz is a
#      HARD ceiling now, not a slider preference. Measured off gait_drawn_tuesday.npz, the cam is
#      the binding joint: 68.8 deg of travel means a peak of 359 deg/s per Hz of cadence, so it
#      reaches the AKE90-8's 1261 deg/s no-load speed at 3.5 Hz. Past that the command is not
#      merely hard to follow, it is kinematically impossible -- the drive saturates current chasing
#      a target it cannot reach, which is a mechanism for breaking things rather than a tracking
#      nuisance. At 2.5 Hz the cam sits at 71% of no-load (so ~29% of stall torque left) and the
#      thigh at 44%. That is already the aggressive end; it is a limit, not a target.
#
# Raising this needs a hardware argument, not a UI one. See scripts in the commit message.
PERIOD_MIN = 0.4
PERIOD_MAX = 600.0

# ===================================================================== thermal bursts (2026-08-28)
# A burst deliberately saturates ONE motor's torque to deposit a known amount of copper loss in a
# few seconds, so the lumped thermal model can be fitted from the temperature rise that follows.
# Everything about it is bounded twice: once by thermal_excite.Envelope (the law cannot ask for
# more) and once here (the daemon will not accept a spec that asks for more).
# The current here has to be large enough to deposit a READABLE amount of heat -- see the note in
# thermal_excite: 12 A for 30 s moves the case by 0.7 degC, which no handheld probe resolves. The
# real bound on a burst is its predicted WINDING rise, checked in thermal_start; these are the
# outer walls. The drive's own configured phase limit is a separate, lower ceiling and it wins.
THERMAL_MAX_AMPS = 30.0
THERMAL_MAX_DURATION_S = 180.0   # the operator-facing knob; the panel's slider is 1-30 s by default
THERMAL_WINDOW_DEG = 8.0         # half-width of the dither window, before the safe-room intersect
THERMAL_SPEED_ERPM = 600.0       # the PRIMARY bound: reversing on speed keeps the motion small
THERMAL_ABORT_TEMP_C = 70        # well under MAX_TEMP_C (80): the reported temperature LAGS
THERMAL_COOLDOWN_S = 3.0         # forced limp after every burst before the mode will do anything else
# FREE-ROTOR sine defaults (motor off the robot, nothing on the shaft): the run tracks a
# position-mode sine so the current comes from the rotor fighting its own inertia. The knob
# bounds live in thermal_excite (FREE_SINE_*); these are just where the panel opens.
THERMAL_FREE_FREQ_HZ = 6.0
THERMAL_FREE_AMP_DEG = 20.0

# ===================================================================== joint identify (2026-08-28)
# "Is left.thigh the joint I think it is?" precedes every measurement on this robot, and it has
# been answered wrong before -- can0/can1 and the 104/105/106 ids are easy to transpose, and the
# abduction axis has never been mapped by anything in this repo at all. A 5 deg, 2 s sine answers
# it by eye, but the half that actually settles it is MEASURED: every other joint is HELD at the
# raw pose it started in, so the per-joint excursion over the wiggle is a direct read of the
# mapping. Exactly one number should be large. If two are, the joints are coupled or two drives
# share an id; if the wrong one is, the map is wrong.
#
# Excursion is accumulated in RAW encoder degrees, so the verdict does not depend on the
# calibration being right -- which is the whole point of being able to run it.
IDENT_AMP_DEG = 5.0              # a gentle, unmistakable 5 deg
IDENT_MAX_AMP_DEG = 10.0
IDENT_DURATION_S = 2.0
IDENT_MAX_DURATION_S = 10.0
IDENT_MIN_AMP_DEG = 1.5          # below this the wiggle is neither visible nor conclusive
IDENT_FREQ_HZ = 1.0              # 5 deg at 1 Hz peaks at 31 deg/s: obvious, and gentle
IDENT_RAMP_S = 0.25              # raised-cosine in/out -- no velocity step at either end
# Tracking is judged against the amplitude, not against a fixed number of degrees. The command
# is always centre +- amp where centre is the joint's MEASURED position, so calib.raw(centre)
# is exactly where the joint already is: a wrong zero cannot make this slew. What it CAN do is
# invert the direction -- and a joint running the wrong way tracks at 2x amplitude, while a
# joint that simply does not move never exceeds 1x. That gap is what this threshold sits in,
# so it catches a flipped calibration sign (which this project has shipped before) instead of
# duplicating the excursion verdict.
IDENT_TRACK_MARGIN_DEG = 3.0     # abort above amp + this
IDENT_MOVED_DEG = 1.5            # excursion at or above this counts as "this joint moved"


# ===================================================================== policy runs (2026-09-01)
# Running a learned policy is the only mode here whose command is not a shape the operator drew.
# Four things bound it, in this order, and none of them replaces the others:
#
#   1. this block          -- what may be ASKED for (run length, approach speed, watchdog windows)
#   2. safety.SafetyGovernor -- what may be SENT (position, rate, torque, gains; clamp then kill)
#   3. the daemon's own _validate_pose -- the recorded workspace polygon, which the governor cannot
#      see because it bounds each joint independently and self-collision is a joint COMBINATION
#   4. the drive's own firmware phase-current limit, which is not ours to raise
#
# The approach numbers come straight from robot/deploy/run_policy.py, which is the version of this
# that has actually been run. They are deliberately crawling: the first policy tick is only legal
# if the legs are already AT the stance, because the controller's filter starts there, and a step
# at the policy's own 200 N*m/rad is not a gait, it is an impact.
POLICY_DEFAULT_SECONDS = 10.0
POLICY_MAX_SECONDS = 60.0
POLICY_APPROACH_DPS = 25.0        # joint travel during the approach -- deliberately crawling
POLICY_APPROACH_KP = 40.0         # N*m/rad: enough to carry a leg, far too little to hurt anything
POLICY_APPROACH_KD = 2.0
POLICY_APPROACH_TRACK_ERR_DEG = 12.0   # abort: the map is wrong, the zero is stale, or it is stuck
POLICY_APPROACH_ARRIVE_DEG = 3.0
POLICY_APPROACH_SLACK_S = 5.0     # grace on top of the computed travel time before giving up
POLICY_APPROACH_MAX_S = 30.0      # only used to size the log buffer
# The dead-man is the panel saying "a human is looking at this", refreshed only while the page is
# VISIBLE. A status poll deliberately does not refresh it: a poll proves a browser is alive, which
# is not the same claim. 1.5 s is ~7 missed 200 ms refreshes.
POLICY_DEADMAN_S = 1.5
# The drives broadcast at exactly 200.0 Hz (measured, 0.14 ms sd), so 50 ms is 10 missed frames.
POLICY_TELEMETRY_STALE_S = 0.05
POLICY_IMU_STALE_S = 0.2          # gravity is the ONLY fall detector; stale gravity is blind
# A workspace refusal is frozen-then-killed rather than killed outright, on the governor's own
# clamp-now-kill-if-persistent rule: 100 ms at 200 Hz, longer than a footfall and far shorter than
# the 0.31 s fall timescale.
POLICY_WS_PERSIST_TICKS = 20
# A drive holds its last force-control frame, and SET_CURRENT 0 is not the release for that mode.
# After the last force frame, keep streaming the zero-gain frame for this long.
POLICY_FORCE_RELEASE_S = 1.0
POLICY_UPRIGHT_GRAV = np.array([0.0, 0.0, -1.0])   # --no-imu fallback: faked upright, never a fall
# ===================================================================== can this Pi run it?
# THE ROBOT'S PI 3B CANNOT HOLD 200 Hz FOR A POLICY. Measured 2026-09-01, imp_m3d bundle, per tick,
# single-threaded so the numbers are work and not GIL contention:
#
#     controller.step()     4.2 ms           (~1.8 ms of it is BLAS and cannot be removed)
#     SafetyGovernor.step() 1.9 ms           (includes the winding observer)
#     send prep + 6 frames  0.4 ms
#     --------------------------------
#     policy tick core      6.0 ms           against a 5.0 ms budget
#
# Those are standalone medians. The arm-time probe measures the same call INSIDE this process,
# where the IMU thread, Flask and the recorder are competing for the GIL, and reads ~6.4 ms -- which
# is the number the gate should use, because it is the one the robot actually gets.
#
# The first measurement was 14 ms, because Debian's numpy was linked against the REFERENCE netlib
# BLAS: a 593x256 float32 gemv ran at 52 MFLOPS. Installing libopenblas0-pthread (one 3.5 MB
# package, no dependencies; Debian's alternatives system repoints libblas.so.3 and numpy picks it
# up with no rebuild) took that gemv to 519 MFLOPS -- 10x -- and the whole tick from 14 ms to
# 8-9 ms. Worth doing, and not enough on its own.
#
# What is left is not BLAS. It is numpy's per-call overhead on SIX-ELEMENT arrays, where the
# dispatch costs more than the arithmetic. Per tick, counted: the governor makes 177 Python calls
# (14 ufunc reductions, 7 np.all, 6 np.any, 6 np.clip) and the controller 150 (9 np.clip alone).
# Folding the thermal observer INTO the governor (2026-09-01) plus two constant-hoists in
# thermal.step took the core from 8-9 ms to 6.5. Closing the last 1.3x means rewriting
# safety.step, controller.step and fourier_gait in scalar or preallocated form -- and they are the
# bit-exactness-verified deploy path, so every change has to be re-proven with verify_export.py
# against the torch policy.
#
# WHY A SLOW LOOP IS NOT MERELY "A BIT SLOW". control_dt is a CONSTANT inside the control law, and
# the loop free-runs when it slips (next_t is reset, never caught up). So a loop that manages 100 Hz
# while the law still believes 200 Hz produces:
#
#   * a gait clock advancing at half real time -- the whole gait plays in slow motion
#   * a velocity channel of (pos - prev) * 200 computed over a 10 ms interval, so every joint
#     velocity the policy observes is inflated 2x: an observation off the training manifold
#   * a thermal observer integrating 5 ms of heating per 10 ms of real current, which is the one
#     error here that is NOT conservative
#
# The filter, the slew limit and the governor's rate cap are all per-step too, but those err
# toward commanding LESS, so they are not what this guard is about.
#
# Two gates, because a bench measurement and a live loop answer different questions: a probe at arm
# time (is this bundle plausible on this machine at all) and the realised rate during the run (is
# it holding up right now, with the IMU thread, Flask and the recorder all competing). Both can be
# acknowledged away with `allow_slow_loop` -- bringing the drives up to watch them move is a real
# reason to run a knowingly-wrong control law, as long as it is knowingly. Until the hot paths are
# optimised, that acknowledgement is how this robot runs a policy at all.
# The probe times controller.step ALONE, so its ceiling is the tick budget minus everything else
# in the tick -- and that is now measured rather than assumed: governor+observer 2.3 ms, send prep
# 0.4, plus drain, measure, log and the black-box sample. Call it 3.0 ms of non-controller work.
POLICY_MAX_STEP_MS = 2.0
POLICY_PROBE_TICKS = 40           # enough to see past the first-call allocations
POLICY_RATE_WINDOW = 200          # realised-rate window, ticks (1 s at nominal)
POLICY_MIN_RATE_FRAC = 0.90       # kill below 180 Hz sustained

# ===================================================================== the runaway that got out
# 2026-09-01, first policy run on the real drives. Four of six drives faulted simultaneously with
# error code 3, across BOTH buses, at 46-52 degC -- so not thermal, and not one bad drive:
#
#     r.abd 12430 ERPM   r.cam 9060 ERPM   r.thigh 15570 ERPM at -19.8 A   (left leg ~1500)
#
# A joint at 15 570 ERPM being braked with 19.8 A of OPPOSING current is pumping energy back into
# the bus. Nothing on this robot can sink that -- there is no brake resistor, and the drives report
# no bus voltage (see the CAN feedback protocol note), so the first evidence of the spike is every
# drive on the bus tripping at once. CubeMars' AK fault table calls code 3 over-voltage, which is
# the only reading consistent with four drives, two buses, one instant and cold motors.
#
# WHAT LET IT HAPPEN: POLICY had no measured-speed kill at all. PLAYBACK has had one since the
# beginning (max_speed 16000 ERPM, warn at 9000) and MEASURE bounds its own excitation, but the
# policy path only ever checked the COMMAND -- the governor's rate clamp bounds how fast the target
# may move, which is a different quantity from how fast the joint is actually turning. So the only
# thing that stopped a runaway was the drives' own over-voltage protection. That is the wrong layer
# to be relying on: it is a hardware fault trip, it takes the whole bus down with it, and it leaves
# the robot limp at speed.
#
# The ceiling is well under playback's 16 000 because a walking policy has no business anywhere
# near no-load speed: in the same recording the leg that was NOT running away sat at ~1500 ERPM.
# The kill is HARD on purpose. A soft stop keeps commanding, and it is the commanded BRAKING that
# generates the over-voltage -- going limp lets the joint spin down through friction instead.
POLICY_MAX_ERPM = 8000.0

POLICY_LOG_COLS = 64
POLICY_LOG_COLUMNS = ("t | pos6 | vel6 | tau6 | amps6 | temp6 | grav3 | gyro3 | target6 | kp6 | "
                      "kd6 | t_winding6 | gait_phase | gait_freq | stop  "
                      "(every six-vector is in MODEL actuator order: "
                      "hip_roll_L, cam_L, thigh_L, hip_roll_R, cam_R, thigh_R)")


def _clamp_period(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return PLAYBACK_DEFAULTS["period"]
    if v != v:                                  # NaN: json.loads happily produces one
        return PLAYBACK_DEFAULTS["period"]
    return min(PERIOD_MAX, max(PERIOD_MIN, v))


# track_err_estop: does exceeding max_track_err latch the E-STOP, or only warn?
#
# Tracking error conflates two different things. A JAM (the leg is against a hard stop, the drive is
# winding up against it) must stop the robot. LAG (the commanded cadence is faster than the drive can
# follow) is not a fault -- it grows with period^-1 and is exactly what you see when deliberately
# pushing playback speed. The trip cannot tell them apart, so at high cadence it fires on the benign
# one and the run is unusable.
#
# Setting this False decouples ONLY this check. Everything else still latches: workspace violation,
# runaway ERPM (max_speed), over-temperature, motor error codes, and the user E-STOP. The peak error
# is tracked and published either way, so "warn" is not "ignore" -- the UI shows how far behind the
# robot actually is, which is the number you want when deciding whether the cadence is realistic.
PLAYBACK_DEFAULTS = dict(period=8.0, mode="position", current_limit=3.0, kp=0.8, ki=0.4, kd=0.02,
                         ramp=3.0, max_track_err=30.0, speed_limit=9000.0, max_speed=16000.0,
                         track_err_estop=True,
                         left_phase=None, legs="both", abd_right=None, abd_left=None)

# System-ID excitation (position mode: stream SET_POS chirps, the drive tracks, we log the current).
# One leg at a time; per-role amplitude + a linear frequency chirp f0->f1 (f0==f1 => pure sine) with
# a per-role phase so the joints decorrelate. 'quasi_static' is the SAME machinery run very slowly
# (velocity ~ 0 -> every sample is a gravity measurement, for Kt/CoM/Coulomb); 'dynamic' chirps
# faster to excite inertia + viscous friction. The WHOLE planned trajectory is workspace-validated
# up front (envelope pre-check) so the run cannot start into unsafe space. See the plan.
MEASURE_DEFAULTS = dict(
    leg="right", profile="dynamic", duration=20.0, ramp=2.0, max_track_err=30.0,
    hold_other=True, override=False,
    amp=dict(abd=8.0, cam=12.0, thigh=12.0),
    f0=dict(abd=0.05, cam=0.05, thigh=0.05),
    f1=dict(abd=0.40, cam=0.50, thigh=0.50),
    phase=dict(abd=0.0, cam=1.5708, thigh=3.1416),
)
MEASURE_ENVELOPE_SAMPLES = 480          # trajectory samples validated before a run may start

# No-load OUTPUT speed per role, deg/s: abduction is an AK60-39 (10.3 rad/s), cam and thigh are
# AKE90-8 (22 rad/s) — the same 1261 deg/s the PERIOD_MIN argument above is built on.
NO_LOAD_DPS = {"abd": 590.0, "cam": 1261.0, "thigh": 1261.0}

# What a DEFAULT excitation should claim of the hardware, on both axes at once: `frac` of the safe
# travel the leg actually has around its current pose, and `frac` of the motor's no-load speed.
# Amplitude and frequency are not independent — a sine of amplitude A at f Hz peaks at 2*pi*f*A
# deg/s — so the two couple, and asking for more range costs you frequency. Measurements taken at
# 0.6 Hz / 9 deg put only 0.7-4.8% of the drive current into the qddot column (the rest was Coulomb
# friction), which is why the identified inertia was garbage: excitation, not estimator.
MEASURE_FRAC = 0.8
# Ceiling on chirp frequency, and the SINGLE source of truth: _merge_measure_spec clamps f0/f1 to
# it. It used to be 3.0 mirrored by a hard-coded 3.0 in that clamp, so a 9 Hz sweep typed into the
# web UI was silently executed as a 3 Hz one (2026-08-19: two runs saved as *_9hz are bit-for-bit
# 0.05->3 Hz chirps -- 0.12% of the command energy sat above 3 Hz, i.e. none).
# 15 Hz because the drive now closes at ~4.7-5.0 Hz on both legs, so the interesting part of the
# response -- gain crossover and the phase margin that goes with it -- lives at 5-12 Hz and was
# entirely outside the old window. The measurement CANNOT see a margin it never excites.
# Not higher than 15: the command stream and the log both run at ~200 Hz, so 15 Hz is already only
# ~13 samples per cycle and the commanded "sine" degrades into a staircase above that.
# NOTE this raises what a HAND-TYPED spec may ask for. measure_defaults() still sizes amplitude
# against no-load speed and predicted tracking error, but a manual spec bypasses that sizing --
# at 12 Hz, 15 deg peaks at 1131 deg/s, which is 90% of no-load for cam/thigh and ~192% for
# abduction. Type the amplitude DOWN when you raise the frequency; the max_speed and max_track_err
# trips are the net, and a latching trip costs the whole run.
MEASURE_F_MAX = 15.0
MEASURE_F_STATIC = 0.03                 # quasi-static: velocity ~ 0, every sample is gravity

# Measured closed-loop response of the drive's POSITION loop (swept-sine on all six joints,
# 2026-08-05 measure_*_dyn_2.npz). It is a first-order roll-off at 0.8 Hz plus ~25 ms of transport
# delay, and it is the same on both legs to two decimals -- 0.71 gain / -48 deg at 0.8 Hz, 0.50/-71
# at 1.4, 0.37/-86 at 2.0, 0.28/-104 at 2.9. This is the REAL ceiling on excitation: above ~1 Hz the
# commanded chirp is mostly not executed, and the un-executed part IS the tracking error.
POS_LOOP_BW_HZ = 0.8
POS_LOOP_DELAY_S = 0.025
# Keep predicted tracking error to this fraction of max_track_err. The trip is latching and costs a
# whole run, so leave real headroom: a 32.4 deg abduction sweep reached 30.0 deg of error at 1.2 Hz
# and e-stopped the right leg mid-run.
MEASURE_TRACK_FRAC = 0.55


def _pos_loop_error_ratio(f):
    """|1 - G(f)| for the position loop: the fraction of a commanded sine amplitude that shows up as
    tracking error. ~0 when the drive tracks, ->1 once it has given up and stopped following."""
    if f <= 0.0:
        return 0.0
    g = 1.0 / np.hypot(1.0, f / POS_LOOP_BW_HZ)
    phi = -np.arctan2(f, POS_LOOP_BW_HZ) - 2.0 * np.pi * f * POS_LOOP_DELAY_S
    return float(abs(1.0 - g * np.exp(1j * phi)))


class RobotDaemon(threading.Thread):
    def __init__(self, interface="socketcan", mock=False, calib=None, wstore=None, fklut=None,
                 bb=None):
        super().__init__(daemon=True, name="RobotDaemon")
        self.interface = interface
        self.mock = mock
        self.calib = calib
        self.wstore = wstore
        self.fklut = fklut
        # The flight recorder. None is a fully supported configuration: every call site goes
        # through the tiny _bb_* helpers below, so the robot runs identically without it.
        self.bb = bb

        self.lock = threading.Lock()
        self.estop_event = threading.Event()
        self.stop_event = threading.Event()

        # -------- requests (web -> daemon, guarded by self.lock) --------
        self._req_mode = None
        self._req_clear_estop = False
        self._manual_targets = {}          # name -> desired normalized deg
        self._manual_override = False
        self._slew_dps = DEFAULT_SLEW_DPS
        self._home_active = False           # slow guided move engaged (feasibility-net checked)
        self._home_kind = "zero"            # "zero" (Home) or "center" (max-room pose) — label only
        self._home_relax = False            # guided move started from a pose the band net rejects
        self._home_slew = 20.0
        self._sine = {n: dict(enabled=False, a=-10.0, b=10.0, freq=0.3, _blend0=None)
                      for n in paths.MOTOR_NAMES}
        self._playback_req = None          # dict: params + 'data'
        self._playback_patch = None
        self._record_cmds = []             # (cmd, leg) tuples
        self._thermal_req = None           # dict: burst spec (motor, amps, duration_s)
        self._thermal_stop = False
        self._therm = None                 # live burst state; see thermal_start/_tick_thermal
        self._wiggle_req = None            # dict: identify spec (motor, amp_deg, duration_s)
        self._wiggle_stop = False
        # NOT _ident: RobotDaemon is a threading.Thread and Thread._set_ident() assigns
        # self._ident = get_ident(). That collision cost an afternoon on 2026-08-28 -- the
        # daemon crashed publishing an int as if it were the wiggle state.
        self._wiggle = None                # live identify state; see identify_start
        self._plan_cache = None            # (key, t_mono, result) -- see IDENT_PLAN_CACHE_S
        # deliberately not restored from anywhere: see BYPASS_NAMES
        self.bypass = {n: False for n in BYPASS_NAMES}
        self._pol_req = None               # dict: a request already validated by policy_arm
        self._pol_stop = None              # "soft" | "hard", consumed by _tick_policy
        self._pol = None                   # live/finished policy run; see _start_policy
        self._pol_deadman = 0.0            # monotonic stamp of the last panel keepalive
        self._measure_req = None           # dict: excitation spec (MEASURE_DEFAULTS merged)
        self._measure_stop = False         # request to end the active excitation (keep the log)

        # -------- daemon-internal state --------
        self.mode = "LIMP"
        self.estop_reason = ""
        self.loop_error = None
        self.buses = {}
        self.motors = []                   # pt.Motor list in paths.MOTOR_NAMES order
        self.by_name = {}
        self.motors_by_bus = {}
        self.side_groups = {}
        self._held = {}                    # name -> commanded normalized target (MANUAL)
        self._last_cmd_raw = {}            # name -> last raw SET_POS (tracking-error guard)
        self._pb = None                    # playback state dict
        self._meas = None                  # MEASURE excitation + high-rate log buffers
        self._rec = dict(kind=None, active=False, leg=None, buf_t=[], buf_p=[], t0=0.0,
                         takes={"right": [], "left": []}, segments={"right": [], "left": []},
                         centers={"right": None, "left": None}, outside=False)
        self._last_reject = ""             # last workspace-check refusal (surfaced in snapshot)
        # The Sense HAT is created by the server AFTER the daemon (so a wedged I2C bus can never
        # delay motor bring-up), so it is attached rather than passed in. None is supported
        # everywhere; only POLICY needs it, and it refuses to arm without one unless told to.
        self.sense = None
        self._rx_at = {}                   # motor name -> t_mono of its last status frame
        self._tick_mono = 0.0              # this tick's monotonic stamp (read by _stream_limp)
        self._force_until = 0.0            # keep streaming the zero-gain force frame until here
        self._tick_count = 0
        self._slip_count = 0
        self.ring = ringbuffer.TelemetryRing()
        self.snapshot = {"daemon_alive": False, "mode": "LIMP"}
        self._cap = None                   # latest plain-data capture from the control loop
        self._cap_ver = 0
        self._snap_ver = -1                # version self.snapshot was formatted from
        self._started_ok = threading.Event()

        # -------- black box + pre-move guard state --------
        self._bb_prev_mono = None
        self._bb_buf = [[0.0] * paths.N_MOTORS for _ in range(8)]   # reused: zero alloc per tick
        self._zero_epoch_at_start = getattr(calib, "zero_epoch", 0) if calib else 0
        self._cmd_zero_epoch = None        # zero_epoch in force when we last commanded absolutely
        self._raw_prev = [None] * paths.N_MOTORS      # continuity watchdog
        self._raw_jumps = []               # discontinuities seen since the last zero capture
        self._jump_epoch = self._zero_epoch_at_start
        self._hold_until = 0.0             # (a) hold-before-move deadline
        self._hold_raw = {}                # raw pose measured at the instant we left LIMP
        self._travel = {}                  # (d) |raw| travelled since the guided move started
        self._travel_prev = {}
        self._guard_latched = ""           # why the last activation was refused (UI + snapshot)
        self._guard_detail = {}
        self._guard_logged = (None, -1e9)
        self._was_homing = False
        self._warn_last = {}               # warn-level trigger debounce
        self._bb_status = {"alive": False}
        self._can_err_seen = 0             # canio send-failure counter, sampled per tick
        self._can_err_last = ("", "")
        self._can_err_logged = -1e9
        self._can_bad_ticks = 0

    # ================================================================= web-facing API
    def request_mode(self, mode):
        with self.lock:
            self._req_mode = mode

    def estop(self, reason="user e-stop"):
        with self.lock:
            self.estop_reason = reason
        self.estop_event.set()

    def clear_estop(self):
        with self.lock:
            self._req_clear_estop = True

    def _hard_bounds(self, side, role):
        """(lo, hi) never-exceed clamp: model limits, widened by the demonstrated workspace."""
        lo, hi = -HARD_CLAMP[role], HARD_CLAMP[role]
        leg = self.wstore.legs.get(side) if self.wstore else None
        if leg:
            if role == "abd":
                olo, ohi = leg["abd_observed"]
                lo, hi = min(lo, olo - HARD_WIDEN_DEG), max(hi, ohi + HARD_WIDEN_DEG)
            elif "knee_grid" in leg:
                o = leg["knee_cam_origin"] if role == "cam" else leg["knee_thigh_origin"]
                n_cells = leg["knee_grid"].shape[0 if role == "cam" else 1]
                r = leg["knee_grid_deg"]
                lo = min(lo, o - HARD_WIDEN_DEG)
                hi = max(hi, o + n_cells * r + HARD_WIDEN_DEG)
        return lo, hi

    def _validate_pose(self, targets, override):
        """Check a FULL normalized pose {name: deg} (hard clamps + workspace / feasibility).
        Called from HTTP threads at request time so refusals are immediate and the daemon never
        chases an out-of-band target."""
        for n, v in targets.items():
            side, role = paths.split_name(n)
            lo, hi = self._hard_bounds(side, role)
            if not lo <= v <= hi:
                return False, f"{n} {v:+.1f} deg exceeds the hard limit [{lo:+.0f}, {hi:+.0f}]"
        # NB the hard-limit loop above has already run and is not bypassable -- only the
        # workspace/feasibility half below can be switched off.
        limits = (self.wstore.limits
                  if (self.wstore and not override and not self.bypass["workspace"]) else None)
        if limits is not None:
            for side in paths.SIDES:
                if not limits.has_leg(side):
                    continue
                ok, reason = limits.validate(side, targets[f"{side}.abd"],
                                             targets[f"{side}.cam"], targets[f"{side}.thigh"])
                if not ok:
                    return False, reason
        if override and self.fklut is not None and self.fklut.available:
            for side in paths.SIDES:
                ok, reason = self.fklut.feasible_check(side, targets[f"{side}.cam"],
                                                       targets[f"{side}.thigh"])
                if not ok:
                    return False, reason
        return True, ""

    def _merged_pose(self, updates):
        """Current held/desired pose merged with an update dict -> full 6-name pose."""
        with self.lock:
            base = {n: self._manual_targets.get(
                        n, self._held.get(n, self.calib.norm(n, self.by_name[n].pos)
                                          if self.by_name and self.by_name[n].pos is not None
                                          else 0.0))
                    for n in paths.MOTOR_NAMES}
        base.update({k: float(v) for k, v in (updates or {}).items() if k in base})
        return base

    def manual_update(self, targets=None, override=None, slew_dps=None):
        """Returns (ok, reason). Targets are validated as a full pose BEFORE being accepted;
        ENTERING manual additionally requires the current pose itself to be inside the safe set
        (else the hold command would be refused every tick) — backdrive the leg into the green
        region first, extend the workspace, or use override."""
        with self.lock:
            eff_override = self._manual_override if override is None else bool(override)
        if self.mode != "MANUAL" and self.by_name \
                and all(m.pos is not None for m in self.motors):
            pose_now = {n: self.calib.norm(n, m.pos) for n, m in self.by_name.items()}
            ok, reason = self._validate_pose(pose_now, eff_override)
            if not ok:
                reason = ("cannot enter manual hold: the leg's CURRENT pose is outside the safe "
                          "workspace (" + reason + ") — backdrive it inside the green region, "
                          "extend the workspace in the editor, or use override")
                with self.lock:
                    self._last_reject = reason
                return False, reason
        if targets:
            pose = self._merged_pose(targets)
            ok, reason = self._validate_pose(pose, eff_override)
            if not ok:
                with self.lock:
                    self._last_reject = reason
                return False, reason
        with self.lock:
            if targets:
                self._manual_targets.update({k: float(v) for k, v in targets.items()
                                             if k in self.by_name})
                self._home_active = False           # user jogging cancels a homing move
            if override is not None:
                self._manual_override = bool(override)
            if slew_dps is not None:
                self._slew_dps = float(np.clip(slew_dps, 5.0, 240.0))
            self._req_mode = "MANUAL"
        return True, ""

    def sine_update(self, actuator, enabled=None, a=None, b=None, freq=None):
        if actuator not in self._sine:
            return False, f"unknown actuator {actuator}"
        with self.lock:
            s = dict(self._sine[actuator])
            override = self._manual_override
        if a is not None:
            s["a"] = float(a)
        if b is not None:
            s["b"] = float(b)
        will_enable = s["enabled"] if enabled is None else bool(enabled)
        if will_enable:
            # both endpoints must be reachable with the other joints where they are now
            for v in (s["a"], s["b"]):
                ok, reason = self._validate_pose(self._merged_pose({actuator: v}), override)
                if not ok:
                    with self.lock:
                        self._last_reject = reason
                    return False, f"sine endpoint {v:+.1f}: {reason}"
        with self.lock:
            st = self._sine[actuator]
            st["a"], st["b"] = s["a"], s["b"]
            if freq is not None:
                st["freq"] = float(np.clip(freq, 0.01, 3.0))
            if enabled is not None:
                if enabled and not st["enabled"]:
                    st["_blend0"] = time.time()
                st["enabled"] = bool(enabled)
            self._home_active = False               # touching sine cancels a homing move
            self._req_mode = "MANUAL"
        return True, ""

    def _pose_rejected_by_band(self):
        """Does the CURRENT pose already fail the assembly-band net? A guided move that starts here
        has to be allowed to leave it (see _tick_manual), or the robot is pinned where it stands."""
        if self.fklut is None or not self.fklut.available or not self.by_name:
            return False
        if any(m.pos is None for m in self.motors):
            return False
        pose = {n: self.calib.norm(n, m.pos) for n, m in self.by_name.items()}
        for side in paths.SIDES:
            ok, _ = self.fklut.feasible_check(side, pose[f"{side}.cam"], pose[f"{side}.thigh"])
            if not ok:
                return True
        return False

    def home(self, slew_dps=None):
        """Slowly drive every joint back to the URDF zero pose (normalized 0 = the stance we
        manually zero to). Trusts the CAD zero: it slews under the physical-feasibility net (like
        override) rather than the eroded gait polygon, so it can still reach 0 when 0 sits a
        degree or so outside the hand-drawn safe region."""
        ok, why = self._activation_allowed()
        if not ok:
            with self.lock:
                self._last_reject = why
            return False, why
        relax = self._pose_rejected_by_band()
        with self.lock:
            self._manual_targets = {n: 0.0 for n in paths.MOTOR_NAMES}
            self._home_active = True
            self._home_kind = "zero"
            self._home_relax = relax
            self._home_slew = float(np.clip(slew_dps if slew_dps else 20.0, 5.0, 120.0))
            for s in self._sine.values():
                s["enabled"] = False
            self._req_mode = "MANUAL"
        return True, ""

    # ---------------------------------------------------------------- centering (max room around)
    @staticmethod
    def _largest_safe_square(grid, res, cam_origin, thigh_origin):
        """Centre of the LARGEST axis-aligned square of safe (cam, thigh) cells, as
        (cam_deg, thigh_deg, half_width_deg), or None if the grid holds no safe cell.

        A square (not a disc) because the thing that has to fit inside is a box: the excitation
        moves cam and thigh independently, so the swept set is [c±A_cam] x [t±A_thigh], and the
        half-width returned here is exactly the amplitude that is guaranteed to stay safe on both
        axes at once. Integral image, no scipy (this runs on the Pi). Ties go to the square nearest
        the zero pose, so the leg travels as little as possible to get there."""
        g = np.asarray(grid, bool)
        h, w = g.shape
        integ = np.zeros((h + 1, w + 1), np.int64)
        integ[1:, 1:] = g.cumsum(0).cumsum(1)
        hits, half = None, 0
        for r in range(min(h, w) // 2 + 1):
            n = 2 * r + 1
            # block sum over every n x n window, indexed by its top-left cell
            blocks = integ[n:, n:] - integ[:-n, n:] - integ[n:, :-n] + integ[:-n, :-n]
            found = np.argwhere(blocks == n * n)
            if not len(found):
                break
            hits, half = found, r
        if hits is None:
            return None
        # cell centres: a safe cell is safe across its whole width (the check floors into the grid),
        # so the room around the centre of the middle cell is (half + 0.5) cells on every side
        cam = cam_origin + (hits[:, 0] + half + 0.5) * res
        thigh = thigh_origin + (hits[:, 1] + half + 0.5) * res
        k = int(np.argmin(np.hypot(cam, thigh)))
        return float(cam[k]), float(thigh[k]), float((half + 0.5) * res)

    def workspace_center(self, sides=None):
        """Per-leg pose with the MOST room around it, as ({name: deg}, {side: info}).

        This is the pose a system-ID chirp is most likely to fit in: near a hard limit or a
        workspace edge every amplitude is refused at t=0, which is the usual reason a measurement
        will not start. `room` is the symmetric amplitude each joint can take FROM that pose:
        cam/thigh share the inscribed-square half-width (both move at once), abduction is half its
        safe range. A leg with no workspace falls back to the zero pose, like Home."""
        targets, info = {}, {}
        for side in (sides or paths.SIDES):
            leg = self.wstore.legs.get(side) if self.wstore else None
            got = None
            if leg is not None and leg.get("knee_grid") is not None:
                got = self._largest_safe_square(leg["knee_grid"], leg["knee_grid_deg"],
                                                leg["knee_cam_origin"], leg["knee_thigh_origin"])
            if got is None:
                for role in paths.ROLES:
                    targets[f"{side}.{role}"] = 0.0
                info[side] = dict(source="zero pose (no workspace for this leg)",
                                  room={r: HARD_CLAMP[r] for r in paths.ROLES})
                continue
            cam, thigh, room = got
            lo, hi = leg["abd_safe"]
            abd = 0.5 * (lo + hi)
            targets[f"{side}.abd"] = round(abd, 1)
            targets[f"{side}.cam"] = round(cam, 1)
            targets[f"{side}.thigh"] = round(thigh, 1)
            info[side] = dict(source="workspace centre",
                              room=dict(abd=round(min(abd - lo, hi - abd), 1),
                                        cam=round(room, 1), thigh=round(room, 1)))
        return targets, info

    def center(self, sides=None, slew_dps=None):
        """Slew to workspace_center(). Uses the same guided move as home(): the feasibility net
        rather than the eroded polygon, so it also works when the leg is currently parked OUTSIDE
        the safe region — which is exactly when you reach for this button.
        Returns (targets, info, "") or (None, None, reason)."""
        ok, why = self._activation_allowed()
        if not ok:
            with self.lock:
                self._last_reject = why
            return None, None, why
        targets, info = self.workspace_center(sides)
        relax = self._pose_rejected_by_band()
        with self.lock:
            self._manual_targets.update(targets)
            self._home_active = True
            self._home_kind = "center"
            self._home_relax = relax
            self._home_slew = float(np.clip(slew_dps if slew_dps else 20.0, 5.0, 120.0))
            for s in self._sine.values():
                s["enabled"] = False
            self._req_mode = "MANUAL"
        return targets, info, ""

    def sine_defaults(self, frac=0.7):
        """Per-actuator sine endpoints = `frac` of the contiguous SAFE travel around the CURRENT
        pose (the other joints held where they are now). So the caller gets start/stop presets that
        stay inside the safe workspace without having to think about angles. Returns
        ({name: {a, b, center, room_up, room_down}}, "") or (None, reason)."""
        if not self.by_name or any(m.pos is None for m in self.motors):
            return None, "not all motors are reporting yet"
        frac = float(np.clip(frac, 0.05, 0.98))
        pose = {n: self.calib.norm(n, m.pos) for n, m in self.by_name.items()}
        out = {}
        for n in paths.MOTOR_NAMES:
            side, role = paths.split_name(n)
            c = pose[n]
            lo, hi = self._hard_bounds(side, role)
            room_up = self._safe_room(pose, n, +1.0, hi - c)
            room_dn = self._safe_room(pose, n, -1.0, c - lo)
            out[n] = dict(a=round(c - frac * room_dn, 1), b=round(c + frac * room_up, 1),
                          center=round(c, 1), room_up=round(room_up, 1),
                          room_down=round(room_dn, 1))
        return out, ""

    def measure_defaults(self, leg="right", profile="dynamic", frac=MEASURE_FRAC):
        """Excitation sized to what the hardware actually offers from the CURRENT pose.

        Per role: amplitude = `frac` of the contiguous safe travel (the smaller of the two
        directions — the chirp is a sine centred on the pose, so it needs symmetric room), and the
        chirp's end frequency = the one whose peak sine velocity 2*pi*f*A reaches `frac` of that
        motor's no-load speed at that amplitude, capped at MEASURE_F_MAX.

        Per-role room does NOT compose: cam and thigh share the 4-bar assembly band, so three
        individually-safe amplitudes can still leave the workspace when swept together. The whole
        envelope is therefore validated here and the amplitudes backed off uniformly until it fits,
        which is the same check measure_start will apply. Returns ({amp, f0, f1, room}, "") or
        (None, reason)."""
        if not self.by_name or any(m.pos is None for m in self.motors):
            return None, "not all motors are reporting yet"
        if leg not in paths.SIDES:
            return None, f"leg must be right|left (got {leg})"
        frac = float(np.clip(frac, 0.05, 0.98))
        pose = {n: self.calib.norm(n, m.pos) for n, m in self.by_name.items()}
        room = {}
        for role in paths.ROLES:
            n = f"{leg}.{role}"
            c = pose[n]
            lo, hi = self._hard_bounds(leg, role)
            room[role] = min(self._safe_room(pose, n, +1.0, hi - c),
                             self._safe_room(pose, n, -1.0, c - lo))

        def freq_for(role, a):
            if profile == "quasi_static" or a < 0.1:
                return MEASURE_F_STATIC
            return min(MEASURE_F_MAX, frac * NO_LOAD_DPS[role] / (2.0 * np.pi * a))

        def size(role, a_room):
            """Trade amplitude against frequency until BOTH the no-load speed and the position
            loop's tracking error are satisfied. A and f fight each other -- peak speed is 2*pi*f*A,
            and error is A*|1-G(f)| -- so solve by iterating rather than in closed form. Quasi-static
            is untouched: at 0.03 Hz the loop tracks perfectly and amplitude is free."""
            a = a_room
            for _ in range(6):
                f = freq_for(role, a)
                room_for_err = MEASURE_TRACK_FRAC * meas["max_track_err"]
                a_track = room_for_err / max(_pos_loop_error_ratio(f), 1e-6)
                a_new = min(a_room, a_track)
                if abs(a_new - a) < 0.05:
                    a = a_new
                    break
                a = a_new
            return round(a, 1), round(freq_for(role, a), 2)

        meas = self._merge_measure_spec(dict(leg=leg, profile=profile))
        meas["base"] = pose
        scale, amp, f0, f1 = 1.0, {}, {}, {}
        for _ in range(12):
            sized = {r: size(r, frac * room[r] * scale) for r in paths.ROLES}
            amp = {r: sized[r][0] for r in paths.ROLES}
            f1 = {r: sized[r][1] for r in paths.ROLES}
            f0 = {r: (MEASURE_F_STATIC if profile == "quasi_static"
                      else MEASURE_DEFAULTS["f0"][r]) for r in paths.ROLES}
            meas["amp"], meas["f0"], meas["f1"] = amp, f0, f1
            T = meas["duration"]
            if all(self._validate_pose(self._measure_pose(meas, T * k / (MEASURE_ENVELOPE_SAMPLES - 1)),
                                       meas["override"])[0]
                   for k in range(MEASURE_ENVELOPE_SAMPLES)):
                break
            scale *= 0.8
        else:
            return None, "no excitation amplitude fits here — centre the leg first"
        return dict(amp=amp, f0=f0, f1=f1, frac=frac, scale=round(scale, 3),
                    room={r: round(room[r], 1) for r in paths.ROLES}), ""

    def _safe_room(self, pose, name, direction, max_reach):
        """Largest contiguous safe displacement of `name` from its current value in `direction`
        (+1/-1), up to `max_reach` deg, judged by the safe-workspace check with the other joints
        held at `pose`. Coarse outward scan then a bisection on the boundary."""
        if max_reach <= 0.5:
            return max(0.0, float(max_reach))
        base = pose[name]
        trial = dict(pose)

        def ok_at(d):
            trial[name] = base + direction * d
            ok, _ = self._validate_pose(trial, override=False)
            return ok

        if not ok_at(0.0):
            return 0.0                                     # current pose already outside safe set
        step = 2.0
        last_ok, d = 0.0, step
        while d <= max_reach:
            if not ok_at(d):
                break
            last_ok, d = d, d + step
        else:
            return float(max_reach)                        # safe all the way to the hard bound
        lo, hi = last_ok, min(d, max_reach)
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if ok_at(mid):
                lo = mid
            else:
                hi = mid
        return lo

    def playback_start(self, data, params):
        p = {**PLAYBACK_DEFAULTS, **params}
        p["period"] = _clamp_period(p["period"])
        # coerce explicitly: JSON "false" is a truthy string, and this flag disables a safety check
        p["track_err_estop"] = p["track_err_estop"] not in (False, 0, "false", "False", "0", "", None)
        with self.lock:
            self._playback_req = {"data": data, **p}

    def playback_patch(self, patch):
        with self.lock:
            self._playback_patch = dict(patch)

    def record_command(self, cmd, leg, kind=None):
        """cmd: start_mode|take_start|take_stop|undo|center|reset ; kind: gait|workspace."""
        with self.lock:
            self._record_cmds.append((cmd, leg, kind))

    def get_recording(self):
        """Copy of accumulated takes/segments/centers for the finish endpoints."""
        with self.lock:
            r = self._rec
            return dict(kind=r["kind"],
                        takes={s: list(r["takes"][s]) for s in ("right", "left")},
                        segments={s: list(r["segments"][s]) for s in ("right", "left")},
                        centers=dict(r["centers"]))

    # ---------------------------------------------------------------- system-ID (MEASURE)
    @staticmethod
    def _merge_measure_spec(spec):
        """Merge a web spec over MEASURE_DEFAULTS and clamp every field to a sane/safe range."""
        m = {k: (dict(v) if isinstance(v, dict) else v) for k, v in MEASURE_DEFAULTS.items()}
        for k, v in (spec or {}).items():
            if k in ("amp", "f0", "f1", "phase") and isinstance(v, dict):
                m[k].update({r: float(v[r]) for r in paths.ROLES if v.get(r) is not None})
            elif k in m and k not in ("amp", "f0", "f1", "phase"):
                m[k] = v
        m["leg"] = str(m["leg"])
        m["profile"] = str(m["profile"])
        m["duration"] = float(np.clip(m["duration"], 1.0, 600.0))
        m["ramp"] = float(np.clip(m["ramp"], 0.0, m["duration"] / 2.0))
        m["max_track_err"] = float(np.clip(m["max_track_err"], 5.0, 60.0))
        m["hold_other"] = bool(m["hold_other"])
        m["override"] = bool(m["override"])
        for r in paths.ROLES:
            m["amp"][r] = float(np.clip(m["amp"][r], 0.0, 60.0))
            m["f0"][r] = float(np.clip(m["f0"][r], 0.0, MEASURE_F_MAX))
            m["f1"][r] = float(np.clip(m["f1"][r], 0.0, MEASURE_F_MAX))
            m["phase"][r] = float(m["phase"][r])
        return m

    @staticmethod
    def _measure_ramp(t, T, ramp):
        """Amplitude envelope: ease in over `ramp` s and ease back out near the end (clean stop)."""
        if ramp <= 0:
            return 1.0
        return float(np.clip(min(t / ramp, (T - t) / ramp), 0.0, 1.0))

    @staticmethod
    def _measure_phase(meas, role, t):
        """Instantaneous chirp phase: 2*pi*integral(f) with f linear f0->f1 over the run."""
        T = max(meas["duration"], 1e-6)
        f0, f1 = meas["f0"][role], meas["f1"][role]
        return 2.0 * np.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * T)) + meas["phase"][role]

    def _measure_pose(self, meas, t):
        """Full 6-name normalized pose at time t: the selected leg tracks its per-role chirp about
        the captured base pose; the other leg stays at base."""
        pose = dict(meas["base"])
        leg = meas["leg"]
        a = self._measure_ramp(t, meas["duration"], meas["ramp"])
        for role in paths.ROLES:
            n = f"{leg}.{role}"
            pose[n] = meas["base"][n] + a * meas["amp"][role] * np.sin(
                self._measure_phase(meas, role, t))
        return pose

    def measure_start(self, spec):
        """Validate the FULL excitation envelope against the safe workspace, then arm a MEASURE run.
        Returns (ok, reason). Refuses if any point of the planned trajectory (or the current pose)
        leaves the safe set — the run can never start into unsafe space (mirrors sine_update's
        both-endpoints pre-check, extended to the whole chirp)."""
        ok, why = self._activation_allowed()
        if not ok:
            return False, why
        if not self.by_name or any(m.pos is None for m in self.motors):
            return False, "not all motors are reporting yet"
        meas = self._merge_measure_spec(spec)
        if meas["leg"] not in paths.SIDES:
            return False, f"leg must be right|left (got {meas['leg']})"
        meas["base"] = {n: self.calib.norm(n, m.pos) for n, m in self.by_name.items()}
        T = meas["duration"]
        for k in range(MEASURE_ENVELOPE_SAMPLES):
            t = T * k / (MEASURE_ENVELOPE_SAMPLES - 1)
            ok, reason = self._validate_pose(self._measure_pose(meas, t), meas["override"])
            if not ok:
                return False, (f"excitation would leave the safe workspace at t={t:.1f}s "
                               f"({reason}) — reduce amplitude, re-center the leg first, "
                               f"or enable override")
        with self.lock:
            self._measure_req = meas
        return True, ""

    def measure_stop(self):
        with self.lock:
            self._measure_stop = True

    def get_measurement(self):
        """Copy of the high-rate log + run metadata for the finish/export endpoints (or None)."""
        with self.lock:
            m = self._meas
            if m is None:
                return None
            n = len(m["buf_t"])

            def arr(key):
                return (np.array(m[key], float) if n else
                        np.zeros((0, paths.N_MOTORS), float))
            run = dict(t=np.array(m["buf_t"], float), cmd_norm=arr("buf_cmd"),
                       pos_norm=arr("buf_posn"), pos_raw=arr("buf_posr"),
                       spd=arr("buf_spd"), cur=arr("buf_cur"))
            meta = dict(leg=m["leg"], profile=m["profile"], duration=m["duration"],
                        ramp=m["ramp"], amp=dict(m["amp"]), f0=dict(m["f0"]), f1=dict(m["f1"]),
                        phase=dict(m["phase"]), hold_other=m["hold_other"],
                        override=m["override"], base=dict(m["base"]),
                        running=m["running"], done=m.get("done", False),
                        started=m.get("started"))
        return run, meta

    def latest_raw_positions(self):
        return {n: (self.by_name[n].pos if self.by_name else None) for n in paths.MOTOR_NAMES}

    def mock_drag(self, name, norm_target=None):
        """Test hook: pull a limp mock joint toward a normalized angle (None = release)."""
        if not self.mock or name not in self.by_name:
            return False
        m = self.by_name[name]
        if norm_target is None:
            m.bus.drag_release(m.cid)
        else:
            m.bus.drag(m.cid, self.calib.raw(name, float(norm_target)))
        return True

    # ================================================================= lifecycle
    def run(self):
        try:
            self._setup()
            self._started_ok.set()
            self._loop()
        except Exception:
            self.loop_error = traceback.format_exc()
            print(f"!! RobotDaemon crashed:\n{self.loop_error}")
            self._bb_event("daemon.crash", traceback=self.loop_error, mode=self.mode)
            self._bb_dump("daemon_crash")
        finally:
            self._bb_event("daemon.loop_exit", mode=self.mode, ticks=self._tick_count,
                           slip=self._slip_count, loop_error=self.loop_error)
            self._release_and_close()
            snap = self.get_snapshot()
            with self.lock:
                snap["daemon_alive"] = False
                snap["loop_error"] = self.loop_error
                snap["blackbox"] = (self.bb.status() if self.bb is not None
                                    else {"alive": False, "error": "not configured"})
                snap["can_errors"] = canio.send_errors()
                snap["can_bus"] = canio.send_stats()
                self.snapshot = snap

    def _setup(self):
        canio.install_send_error_hook(self._can_error)
        channels = sorted(set(paths.SIDE_CHANNEL.values()))
        self.buses = canio.open_buses(self.interface, channels, mock=self.mock)
        for side in paths.SIDES:
            ch = paths.SIDE_CHANNEL[side]
            self.motors_by_bus.setdefault(ch, {})
            for col, role in enumerate(paths.ROLES):
                m = pt.Motor(self.buses[ch], paths.ROLE_ID[role], side, col)
                self.motors.append(m)
                self.by_name[f"{side}.{role}"] = m
                self.motors_by_bus[ch][m.cid] = m
        self.side_groups = pt.group_by_side(self.motors)
        self._name_of = {id(m): n for n, m in self.by_name.items()}
        # preflight: limp-stream + wait for every motor to report (play_trajectory.preflight)
        t_end = time.time() + 2.0
        while time.time() < t_end and any(m.pos is None for m in self.motors):
            for m in self.motors:
                canio.set_current(m.bus, m.cid, 0.0)
            # _drain rather than pt.drain so _rx_at is populated from the very first frame: the
            # telemetry watchdog reports "no drive has ever spoken" as infinitely stale, and the
            # preflight is exactly where a drive first speaks.
            self._drain(time.monotonic(), 0.0)
            time.sleep(0.005)
        silent = [self._name_of[id(m)] for m in self.motors if m.pos is None]
        raw = {n: m.pos for n, m in self.by_name.items()}
        self._bb_event("daemon.preflight", mock=self.mock, interface=self.interface,
                       silent=silent, raw=raw,
                       calibration=self.calib.snapshot() if self.calib else None,
                       raw_vs_last_zero=self.calib.compare_raw(raw) if self.calib else None,
                       note="raw_vs_last_zero with the robot untouched answers 'does pos_raw "
                            "still match the last zero capture'")
        if silent:
            print(f"!! preflight: no status from {silent} — motion blocked until they report")
            self._bb_event("can.silent", motors=silent,
                           note="no status frames — the bus, the wiring or the drives are down")

    def _release_and_close(self):
        for m in self.motors:
            try:
                canio.set_current(m.bus, m.cid, 0.0)
            except Exception:
                pass
        time.sleep(0.02)
        for b in self.buses.values():
            try:
                b.shutdown()
            except Exception:
                pass
        print("RobotDaemon: motors released (0 A), buses closed.")

    # ================================================================= main loop
    def _loop(self):
        dt = 1.0 / TICK_HZ
        next_t = time.time()
        prev_mono = time.monotonic()
        while not self.stop_event.is_set():
            now = time.time()
            t_mono = time.monotonic()
            dt_actual = t_mono - prev_mono
            prev_mono = t_mono
            self._tick_mono = t_mono

            # 1) e-stop first — latch (limp is streamed by the mode body below)
            if self.estop_event.is_set() and self.mode != "ESTOPPED":
                self._set_mode("ESTOPPED", self.estop_reason or "e-stop latched")
                self._pb = None
                if self._meas:
                    self._meas["running"] = False       # keep the partial log; stop exciting
                if self._pol is not None and self._pol["phase"] != "done":
                    # end it rather than drop it: an e-stop during a policy run is exactly when
                    # the 200 Hz log is worth keeping
                    self._policy_end(self._pol, now, self.estop_reason or "e-stop")
                with self.lock:
                    self._manual_targets = {}
                    self._manual_override = False

            # 2) always drain feedback (telemetry works even limp)
            self._drain(t_mono, dt)

            # 3) consume web requests
            self._consume_requests(now)

            # 4) mode body
            if self.mode in ("LIMP", "RECORD_GAIT", "RECORD_WS", "ESTOPPED"):
                self._stream_limp()
                if self.mode in ("RECORD_GAIT", "RECORD_WS"):
                    self._tick_recording(now)
            elif self.mode == "MANUAL":
                self._tick_manual(now, dt)
            elif self.mode == "PLAYBACK":
                self._tick_playback(now, dt)
            elif self.mode == "MEASURE":
                self._tick_measure(now, dt)
            elif self.mode == "THERMAL":
                self._tick_thermal(now, dt)
            elif self.mode == "IDENTIFY":
                self._tick_identify(now, dt)
            elif self.mode == "POLICY":
                self._tick_policy(now, dt)

            # 5) safety sweep (err flag / temp in every mode; run_hardware.safety_check semantics)
            for m in self.motors:
                if m.pos is None:
                    continue
                if m.err:
                    self._trip(f"{self._name_of[id(m)]} error code {m.err}")
                elif m.temp >= MAX_TEMP_C:
                    self._trip(f"{self._name_of[id(m)]} temp {m.temp}C >= {MAX_TEMP_C}")

            # 6) black box: EVERY tick at the full rate (a bounded deque append, no I/O)
            self._bb_tick(t_mono, now, dt_actual)

            # 7) telemetry + snapshot at 20 Hz
            self._tick_count += 1
            if self._tick_count % TELEMETRY_DIV == 0:
                self._publish(now)

            next_t += dt
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                self._slip_count += 1
                next_t = time.time()

    def _set_mode(self, mode, reason, **fields):
        """The ONE place the mode changes, so every transition is on the timeline with its reason
        and gets a Tier B dump of the 200 Hz window around it."""
        old = self.mode
        self.mode = mode
        if old == mode:
            return
        for i in range(paths.N_MOTORS):
            self._raw_prev[i] = None            # continuity chains do not span a mode change
        raw = {n: m.pos for n, m in self.by_name.items()}
        self._bb_event("mode", **{"from": old, "to": mode, "reason": reason, "raw": raw,
                                  "raw_vs_last_zero": self.calib.compare_raw(raw),
                                  "zero_epoch": self.calib.zero_epoch,
                                  "slip": self._slip_count, **fields})
        # routine, so a long cooldown: an operator toggling LIMP/MANUAL must not evict the
        # continuous Tier A history one 1.7 MB dump at a time
        self._bb_dump(f"mode_{old}_to_{mode}", cooldown_s=blackbox.ROUTINE_COOLDOWN_S, why=reason)

    def _trip(self, reason):
        if not self.estop_event.is_set():
            with self.lock:
                self.estop_reason = reason
            self.estop_event.set()
            print(f"!! SAFETY STOP: {reason}")
            self._bb_event("trip", reason=reason, mode=self.mode,
                           raw={n: m.pos for n, m in self.by_name.items()},
                           cmd_raw=dict(self._last_cmd_raw),
                           norm={n: (None if m.pos is None else round(self.calib.norm(n, m.pos), 3))
                                 for n, m in self.by_name.items()},
                           temp={n: m.temp for n, m in self.by_name.items()},
                           err={n: m.err for n, m in self.by_name.items()},
                           cur={n: m.cur for n, m in self.by_name.items()},
                           spd={n: m.spd for n, m in self.by_name.items()},
                           slip=self._slip_count, homing=self._home_active)
            self._bb_dump("trip", why=reason)

    def _drain(self, t_mono, dt):
        """play_trajectory.drain, plus a per-motor arrival stamp.

        pt.drain returns nothing, so a caller cannot tell a live drive from a dead one -- and the
        question that matters is not "did any frame arrive" but "has ONE drive gone quiet while its
        five neighbours keep talking". One dict store per frame buys that; the POLICY governor's
        telemetry watchdog is the first thing that needs it, and the answer it gets is the age of
        the OLDEST motor rather than of the newest frame."""
        rx, name_of = self._rx_at, self._name_of
        for ch, bus in self.buses.items():
            by_id = self.motors_by_bus[ch]
            while True:
                msg = bus.recv(timeout=0.0)
                if msg is None:
                    break
                if not msg.is_extended_id:
                    continue
                m = by_id.get(msg.arbitration_id & 0xFF)
                if m is None:
                    continue
                st = canio.parse_status(msg.data)
                if st:
                    m.update_from(st, dt)
                    rx[name_of[id(m)]] = t_mono

    def _stream_limp(self):
        for m in self.motors:
            canio.set_current(m.bus, m.cid, 0.0)
        # A force-control run leaves the drive holding its last kp/kd frame, and SET_CURRENT 0 is
        # not the release for THAT mode. So for a second after the last force frame, stream the
        # zero-gain force frame alongside -- streamed, because "stop sending" is never "stop
        # commanding". Outside that window this is the same six frames it has always been.
        if self._tick_mono < self._force_until:
            payload = mit.limp_payload()
            for m in self.motors:
                canio.force_control(m.bus, m.cid, payload)
        self._last_cmd_raw.clear()

    def _motion_allowed(self):
        if not self.calib.complete:
            return False, "calibration incomplete — finish the zero/direction wizard first"
        silent = [self._name_of[id(m)] for m in self.motors if m.pos is None]
        if silent:
            return False, f"motor(s) silent: {', '.join(silent)} — never commanding blind"
        return True, ""

    # ================================================================= black box
    def _bb_event(self, kind, /, **fields):
        bb = self.bb
        if bb is not None:
            bb.log_event(kind, **fields)

    def _bb_dump(self, reason, /, cooldown_s=None, **fields):
        bb = self.bb
        return bb.trigger_dump(reason, cooldown_s=cooldown_s, **fields) if bb is not None else None

    def _bb_tick(self, t_mono, t_wall, dt_actual):
        """One 200 Hz sample into the recorder, plus the two watchdogs that need every tick.

        Everything here is O(6) arithmetic on preallocated lists and one deque append — no locks,
        no allocation beyond the outgoing tuple, no file I/O. The tick already slips ~12%; this
        must not make that worse.
        """
        pr, pn, cr, cn, sp, cu, tp, er = self._bb_buf
        calib, nan = self.calib, float("nan")
        for i, n in enumerate(paths.MOTOR_NAMES):
            m = self.by_name[n]
            p = m.pos
            if p is None:
                pr[i] = pn[i] = sp[i] = cu[i] = tp[i] = nan
                er[i] = 0.0
            else:
                pr[i] = p
                pn[i] = calib.norm(n, p)
                sp[i], cu[i], tp[i], er[i] = m.spd, m.cur, m.temp, m.err
            c = self._last_cmd_raw.get(n)
            if c is None:
                cr[i] = cn[i] = nan
            else:
                cr[i] = c
                cn[i] = calib.norm(n, c)

        bb = self.bb
        if bb is not None:
            bb.push_sample((t_mono, t_wall, dt_actual, MODE_CODE.get(self.mode, 0),
                            1.0 if self.mode == "ESTOPPED" else 0.0, self._slip_count,
                            *pr, *pn, *cr, *cn, *sp, *cu, *tp, *er))

        self._watch_continuity(t_mono, dt_actual, pr, sp)
        self._watch_warnings(t_mono, pr, cr, sp, tp, er)
        self._can_watch(t_mono)

    def _watch_continuity(self, t_mono, dt_actual, pr, sp):
        """Did a driver board's multi-turn origin move on its own? (question 5 of the brief)

        Only meaningful while we are commanding nothing: then any change in pos_raw is either the
        joint physically moving — which the drive reports as speed — or the encoder origin being
        rewritten underneath us, which it does not. A jump with the drive claiming standstill is
        the latter, and it invalidates the calibration instantly. This lives in the daemon rather
        than in blackbox.py because the pre-move guard must keep working even if the recorder dies.
        """
        if self.calib.zero_epoch != self._jump_epoch:      # a fresh zero starts a fresh history
            self._jump_epoch = self.calib.zero_epoch
            self._raw_jumps = []
        prev = self._raw_prev
        if self.mode not in ("LIMP", "ESTOPPED", "RECORD_GAIT", "RECORD_WS"):
            for i in range(paths.N_MOTORS):
                prev[i] = None
            return
        thr = max(RAW_JUMP_DEG, RAW_JUMP_DPS * dt_actual)
        for i in range(paths.N_MOTORS):
            p, q = pr[i], prev[i]
            prev[i] = None if p != p else p                # NaN (silent motor) breaks the chain
            if q is None or p != p:
                continue
            d = p - q
            if abs(d) <= thr or abs(sp[i]) >= RAW_JUMP_SPD_MAX:
                continue
            n = paths.MOTOR_NAMES[i]
            self._raw_jumps.append({"motor": n, "t_mono": round(t_mono, 4),
                                    "before": round(q, 3), "after": round(p, 3),
                                    "delta": round(d, 3), "spd": round(sp[i], 1),
                                    "mode": self.mode})
            del self._raw_jumps[:-64]
            self._bb_event("raw.jump", motor=n, before=q, after=p, delta_deg=d, spd=sp[i],
                           mode=self.mode, dt=dt_actual, threshold_deg=thr,
                           zero_epoch=self.calib.zero_epoch,
                           note="pos_raw stepped while limp and the drive reported no motion — "
                                "the encoder origin moved, the joint did not. The calibration is "
                                "now stale and the pre-move guard will refuse to leave LIMP.")
            self._bb_dump("raw_origin_jump", motor=n, delta_deg=d)

    def _can_error(self, channel, exc):
        """canio's send-failure hook. Called from the control thread, so it does nothing but count
        and (rarely) log — the escalation decision is made in _can_watch, once per tick."""
        self._can_err_last = (channel, f"{type(exc).__name__}: {exc}")
        t = time.monotonic()
        if t - self._can_err_logged > 30.0:
            self._can_err_logged = t
            self._bb_event("can.error", channel=channel, error=str(exc),
                           exc_type=type(exc).__name__, mode=self.mode,
                           stats=canio.send_stats(),
                           note="the bus would not accept the frame. With the motor power off "
                                "nothing ACKs, socketcan's TX queue fills and every send fails — "
                                "expected at boot, a fault if it happens mid-motion")

    def _can_watch(self, t_mono):
        """A bus that will not take frames is survivable while we are commanding nothing, and a
        fault the moment we are. Never a daemon death: the telemetry loop and the recorder have to
        keep running so the operator can see WHY the robot is not responding."""
        total = sum(canio.send_errors().values())
        grew = total > self._can_err_seen
        self._can_err_seen = total
        self._can_bad_ticks = self._can_bad_ticks + 1 if grew else 0
        if (self._can_bad_ticks > 0.5 * TICK_HZ
                and self.mode in ("MANUAL", "PLAYBACK", "MEASURE")):
            ch, err = self._can_err_last
            self._trip(f"CAN bus {ch} is not accepting frames ({err}) — commands are not "
                       f"reaching the drives")
            self._can_bad_ticks = 0

    def _watch_warnings(self, t_mono, pr, cr, sp, tp, er):
        """Warn levels sit BELOW the trip thresholds so a dump catches the approach, not only the
        fall. These never stop the robot."""
        if self.bb is None:
            return
        for i in range(paths.N_MOTORS):
            n = paths.MOTOR_NAMES[i]
            hit = None
            if cr[i] == cr[i] and pr[i] == pr[i] and abs(pr[i] - cr[i]) > WARN_TRACK_ERR_DEG:
                hit = ("track", f"{n} tracking error {pr[i] - cr[i]:+.1f} deg "
                                f"(warn {WARN_TRACK_ERR_DEG})")
            elif abs(sp[i]) > WARN_SPD_ERPM:
                hit = ("speed", f"{n} speed {sp[i]:.0f} ERPM (warn {WARN_SPD_ERPM:.0f})")
            elif tp[i] == tp[i] and tp[i] >= WARN_TEMP_C:
                hit = ("temp", f"{n} temp {tp[i]:.0f}C (warn {WARN_TEMP_C})")
            elif er[i]:
                hit = ("err", f"{n} drive error code {int(er[i])}")
            if hit is None:
                continue
            key = f"{hit[0]}:{n}"
            if t_mono - self._warn_last.get(key, -1e9) < 30.0:
                continue
            self._warn_last[key] = t_mono
            self._bb_event("warn", kind=hit[0], motor=n, message=hit[1], mode=self.mode,
                           pos_raw=pr[i], cmd_raw=cr[i], spd=sp[i], temp=tp[i], err=int(er[i]))
            self._bb_dump(f"warn_{hit[0]}", motor=n, message=hit[1])

    # ================================================================= pre-move guard
    def _premove_guard(self):
        """Is it safe to command absolute positions right now? (ok, reason, detail) — pure.

        Three checks, none of which fires on the normal zero -> home workflow:

          (i)   RANGE SANITY, always decisive. calibration + the reported raw must imply a pose the
                joint can physically be in. On 2026-08-10 left.cam implied 678 deg on a +-88 deg
                axis; this alone would have refused the move.
          (ii)  CONTINUITY, decisive whenever we have watched every tick since the zero capture. A
                logged origin jump means the calibration is stale, full stop.
          (iii) RAW-AT-REST, decisive only when we CANNOT vouch for continuity — a calibration
                restored from disk and never re-captured in this process. The drives re-randomise
                their raw origin on every power cycle, so a stale file is precisely the trap; and
                since gravity sag also moves the joints while we were dead, a mismatch is
                genuinely ambiguous and must be resolved by a re-zero rather than by guessing.
        """
        raw_now = {n: m.pos for n, m in self.by_name.items()}
        detail = {"raw_now": {n: (None if v is None else round(v, 3)) for n, v in raw_now.items()},
                  "raw_at_last_zero": self.calib.raw_at_rest(),
                  "compare": self.calib.compare_raw(raw_now),
                  "zero_epoch": self.calib.zero_epoch,
                  "zeroed_this_session": self.calib.zero_epoch != self._zero_epoch_at_start,
                  "moved_since_zero": self._cmd_zero_epoch == self.calib.zero_epoch,
                  "origin_jumps_since_zero": list(self._raw_jumps)}

        # (i) range sanity
        for n, m in self.by_name.items():
            if m.pos is None:
                continue
            side, role = paths.split_name(n)
            lo, hi = self._hard_bounds(side, role)
            norm = self.calib.norm(n, m.pos)
            if not (lo - RANGE_SANITY_SLACK_DEG <= norm <= hi + RANGE_SANITY_SLACK_DEG):
                return False, (f"{n}: calibration says this joint is at {norm:+.1f} deg, but its "
                               f"range is [{lo:.0f}, {hi:.0f}] — the raw origin and the zero do "
                               f"not agree. Re-zero before moving."), detail

        # (ii) an origin jump seen with our own eyes since the last capture
        if self._raw_jumps:
            j = self._raw_jumps[-1]
            return False, (f"{j['motor']}: pos_raw jumped {j['delta']:+.1f} deg while LIMP and not "
                           f"turning ({j['before']:.1f} -> {j['after']:.1f}) — the encoder origin "
                           f"moved, so the zero is stale. Re-zero before moving."), detail

        # (iii) raw-at-rest, when continuity is unknowable
        if not detail["zeroed_this_session"] and detail["raw_at_last_zero"]:
            off = {n: c for n, c in detail["compare"].items()
                   if abs(c["delta"]) > RAW_AT_REST_TOL_DEG}
            if off:
                worst = max(off, key=lambda n: abs(off[n]["delta"]))
                return False, (
                    f"{worst}: pos_raw is {off[worst]['now']:.1f} but the last zero capture "
                    f"recorded {off[worst]['then']:.1f} ({off[worst]['delta']:+.1f} deg, "
                    f"{len(off)} joint(s) off). This calibration was restored from disk and has "
                    f"not been re-captured since — the drives re-randomise their raw origin on "
                    f"every power cycle. Re-zero before moving."), detail
        return True, "", detail

    def _activation_allowed(self):
        """_motion_allowed() plus the pre-move guard. Every transition out of LIMP goes through
        here; refusals are latched, logged with BOTH raw poses, and dumped."""
        ok, why = self._motion_allowed()
        if not ok:
            return False, why
        ok, why, detail = self._premove_guard()
        if ok:
            self._guard_latched, self._guard_detail = "", {}
            return True, ""
        self._guard_latched, self._guard_detail = why, detail
        self._last_reject = why
        last_why, last_t = self._guard_logged
        now = time.monotonic()
        if why != last_why or (now - last_t) > 30.0:      # do not spam on a polling UI
            self._guard_logged = (why, now)
            # one dict, for the reason spelled out at the thermal.done call site: a key
            # appearing in both halves is a TypeError at binding, not a merge
            self._bb_event("premove.refused",
                           **dict(detail, reason=why, mode=self.mode))
            self._bb_dump("premove_guard_refused", why=why)
        return False, why

    # ----------------------------------------------------------------- requests
    def _consume_requests(self, now):
        with self.lock:
            req_mode = self._req_mode
            self._req_mode = None
            clear = self._req_clear_estop
            self._req_clear_estop = False
            pb_req = self._playback_req
            self._playback_req = None
            pb_patch = self._playback_patch
            self._playback_patch = None
            rec_cmds = self._record_cmds
            self._record_cmds = []
            meas_req = self._measure_req
            self._measure_req = None
            meas_stop = self._measure_stop
            self._measure_stop = False
            th_req = self._thermal_req
            self._thermal_req = None
            th_stop = self._thermal_stop
            self._thermal_stop = False
            id_req = self._wiggle_req
            self._wiggle_req = None
            id_stop = self._wiggle_stop
            self._wiggle_stop = False
            pol_req = self._pol_req
            self._pol_req = None

        if clear and self.mode == "ESTOPPED":
            self.estop_event.clear()
            with self.lock:
                was = self.estop_reason
                self.estop_reason = ""
            self._bb_event("estop.clear", previous_reason=was)
            self._set_mode("LIMP", "e-stop cleared by operator")

        if self.mode == "ESTOPPED":
            return                                          # latched: ignore everything else

        for cmd, leg, kind in rec_cmds:
            self._handle_record_cmd(cmd, leg, kind, now)

        if pb_req is not None:
            ok, why = self._activation_allowed()
            if ok:
                self._start_playback(pb_req, now)
            else:
                self._last_reject = why

        if pb_patch is not None and self._pb is not None:
            self._apply_playback_patch(pb_patch, now)

        if meas_stop and self._meas is not None:
            self._meas["running"] = False               # user-ended; keep the log for saving
            self._bb_event("measure.stop", leg=self._meas.get("leg"),
                           n_samples=len(self._meas.get("buf_t", [])))
        if meas_req is not None:
            ok, why = self._activation_allowed()
            if ok:
                self._start_measure(meas_req, now)
            else:
                self._last_reject = why

        if th_stop and self._therm is not None:
            self._therm["running"] = False
            self._therm["abort"] = self._therm["abort"] or "stopped by operator"
            self._bb_event("thermal.stop", motor=self._therm["motor"])
        if th_req is not None:
            ok, why = self._activation_allowed()
            if ok:
                self._start_thermal(th_req, now)
            else:
                self._last_reject = why

        if id_stop and self._wiggle is not None:
            self._wiggle["running"] = False
            self._wiggle["abort"] = self._wiggle["abort"] or "stopped by operator"
            self._bb_event("identify.stop", motor=self._wiggle["motor"])
        if id_req is not None:
            ok, why = self._activation_allowed()
            if ok:
                self._start_identify(id_req, now)
            else:
                self._last_reject = why

        if pol_req is not None:
            ok, why = self._activation_allowed()
            if ok:
                self._start_policy(pol_req, now)
            else:
                self._last_reject = why

        if req_mode and req_mode != self.mode:
            if req_mode == "LIMP":
                if self._pol is not None and self._pol["phase"] != "done":
                    # end it, do NOT drop it: the run log is the whole point of having run it
                    self._policy_end(self._pol, now, "operator requested LIMP")
                self._pb = None
                self._meas = None                          # finished/aborted run is cleared here
                self._therm = None
                self._wiggle = None
                self._end_active_record()
                self._enter_hold(None)
                with self.lock:
                    self._manual_targets = {}
                    self._manual_override = False          # override never survives a mode change
                self._set_mode("LIMP", "operator requested LIMP")
            elif req_mode == "MANUAL":
                ok, why = self._activation_allowed()
                if ok:
                    if self.mode != "MANUAL":
                        # enter without a jump: hold the current pose; keep targets that were
                        # posted with this request, fill the rest from the held pose
                        self._held = {n: self.calib.norm(n, m.pos)
                                      for n, m in self.by_name.items()}
                        with self.lock:
                            for n, v in self._held.items():
                                self._manual_targets.setdefault(n, v)
                            for s in self._sine.values():
                                s["enabled"] = False
                        self._enter_hold(now)
                    self._set_mode("MANUAL", "operator requested MANUAL")
                else:
                    self._last_reject = why

    def _enter_hold(self, now):
        """(a) hold-before-move: latch the raw pose to hold, and reset the travel budget.

        `now is None` cancels it (going limp). The held raw is the MEASURED encoder value, not
        anything derived from the calibration, which is what makes this safe when the zero is
        wrong: whatever the offsets say, set_pos(where you already are) cannot slew anywhere."""
        if now is None:
            self._hold_until, self._hold_raw = 0.0, {}
            self._travel, self._travel_prev = {}, {}
            return
        self._hold_raw = {n: m.pos for n, m in self.by_name.items() if m.pos is not None}
        self._hold_until = now + HOLD_BEFORE_MOVE_S
        self._travel = {n: 0.0 for n in self._hold_raw}
        self._travel_prev = dict(self._hold_raw)
        self._bb_event("premove.hold", hold_s=HOLD_BEFORE_MOVE_S, raw=self._hold_raw,
                       note="first command after enable is set_pos(current raw) — "
                            "calibration-independent, stops the sag, cannot slew")

    # ----------------------------------------------------------------- MANUAL (hold + sine)
    def _tick_manual(self, now, dt):
        ok, why = self._motion_allowed()
        if not ok:
            self._trip(why)
            return

        # (a) hold-before-move. For the first HOLD_BEFORE_MOVE_S after enabling we command each
        # joint to the raw angle it was measured at — NOT through the calibration. This catches
        # the sag instantly and is safe even against a completely wrong zero; only after it do we
        # start slewing toward absolute targets.
        if now < self._hold_until:
            for n, m in self.by_name.items():
                raw = self._hold_raw.get(n)
                if raw is None:
                    continue
                canio.set_pos(m.bus, m.cid, raw)
                self._last_cmd_raw[n] = raw
            self._held = {n: self.calib.norm(n, m.pos) for n, m in self.by_name.items()
                          if m.pos is not None}
            return

        with self.lock:
            desired = dict(self._manual_targets)
            homing = self._home_active
            override = self._manual_override or homing   # homing trusts the feasibility net, not
            slew = self._home_slew if homing else self._slew_dps   # the eroded gait polygon
            sine = {n: dict(s) for n, s in self._sine.items()}

        held_before = dict(self._held)
        targets_norm = {}
        for n, m in self.by_name.items():
            cur = self._held.get(n, self.calib.norm(n, m.pos))
            s = sine[n]
            if s["enabled"]:
                mid, amp = (s["a"] + s["b"]) / 2.0, (s["b"] - s["a"]) / 2.0
                want = mid + amp * np.sin(2.0 * np.pi * s["freq"] * (now - (s["_blend0"] or now)))
            else:
                want = desired.get(n, cur)
            step = slew * dt                              # slew applies to sine too: no jumps ever
            tgt = cur + float(np.clip(want - cur, -step, step))
            self._held[n] = tgt
            side, role = paths.split_name(n)
            lo, hi = self._hard_bounds(side, role)
            targets_norm[n] = float(np.clip(tgt, lo, hi))

        # workspace check on the FULL tick before sending anything (play_trajectory.py:299-309)
        if not override:
            limits = self.wstore.limits if self.wstore else None
            tgt_by_motor = {self.by_name[n]: targets_norm[n] for n in targets_norm}
            ok, reason = pt.check_workspace(self.side_groups, tgt_by_motor, limits)
            if not ok:
                self._last_reject = reason
                self._held = held_before                   # do NOT advance through refused space
                return                                     # hold previous commands, don't send
        note = ""
        if override and self.fklut is not None and self.fklut.available:
            bad = ""
            for side in paths.SIDES:                       # physically-assemblable band net
                ok, reason = self.fklut.feasible_check(side, targets_norm[f"{side}.cam"],
                                                       targets_norm[f"{side}.thigh"])
                if not ok:
                    bad = reason
                    break
            if not bad:
                self._home_relax = False                   # back in the band: re-arm the net
            elif not (homing and self._home_relax):
                self._last_reject = bad
                self._held = held_before
                return
            else:
                # The move STARTED from a pose the band net rejects, so enforcing it here would
                # pin the robot in that pose forever — and getting out of it is the whole point of
                # Home / ⌖ Centre. The destination was validated before the move was accepted, the
                # slew is slow, and the hard clamps, the tracking-error trip and the E-STOP all
                # still apply. The net re-arms the moment the leg is back inside the band (above).
                note = f"recovering toward a checked target — {bad}"
        self._last_reject = note

        if homing and not self._was_homing:              # (d) each new guided move: fresh budget
            self._travel, self._travel_prev = {}, {}
        self._was_homing = homing
        # (c) homing is a slow guided slew and should track well — a stale zero shows up as
        # tracking error within the first few degrees, long before the 25 deg playback threshold.
        track_lim = MAX_TRACK_ERR_HOMING_DEG if homing else MAX_TRACK_ERR_DEG
        self._cmd_zero_epoch = self.calib.zero_epoch     # we are commanding ABSOLUTE positions now
        for n, m in self.by_name.items():
            raw = self.calib.raw(n, targets_norm[n])
            canio.set_pos(m.bus, m.cid, raw)
            self._last_cmd_raw[n] = raw
            if not self.bypass["tracking"] and abs(m.pos - raw) > track_lim:
                self._trip(f"{n} tracking error {m.pos - raw:+.1f} deg (> {track_lim}"
                           f"{' while homing' if homing else ''})")
                return
        if homing and self._over_travel_budget(desired):
            return

    # ----------------------------------------------------------------- PLAYBACK
    def _over_travel_budget(self, desired):
        """(d) the backstop. A guided move that has swept more than TRAVEL_BUDGET_FACTOR times a
        joint's own range without arriving is not going to arrive — something between the command
        and the joint is lying, and the only safe thing left is to stop.

        Scoped to guided moves (Home / Centre) because only there is "arriving" well defined; free
        manual sliding legitimately accumulates unlimited travel. On 2026-08-10 left.cam reached
        ~1.9 output turns on a +-88 deg joint: this cuts that at ~0.6 of a turn whatever the
        calibration claims.
        """
        for n, m in self.by_name.items():
            if m.pos is None:
                continue
            prev = self._travel_prev.get(n)
            self._travel_prev[n] = m.pos
            if prev is None:
                self._travel[n] = 0.0
                continue
            self._travel[n] = self._travel.get(n, 0.0) + abs(m.pos - prev)
            side, role = paths.split_name(n)
            lo, hi = self._hard_bounds(side, role)
            budget = TRAVEL_BUDGET_FACTOR * (hi - lo)
            if self._travel[n] <= budget:
                continue
            want = desired.get(n)
            # `desired` is the FINAL target of the guided move, never the per-tick slew step —
            # comparing against the step would make every joint look permanently "arrived".
            if want is None or abs(self.calib.norm(n, m.pos) - want) <= TRAVEL_ARRIVED_DEG:
                self._travel[n] = 0.0                    # arrived; a fresh move gets a fresh budget
                continue
            self._trip(f"{n} travel budget: swept {self._travel[n]:.0f} deg without arriving "
                       f"(> {budget:.0f} = {TRAVEL_BUDGET_FACTOR}x its {hi - lo:.0f} deg range) — "
                       f"the commanded target does not correspond to a reachable pose")
            return True
        return False

    def _start_playback(self, req, now):
        data = req["data"]
        sides = [s for s in (("right", "left") if req["legs"] == "both" else (req["legs"],))
                 if data.get(s) is not None]
        if not sides:
            self._last_reject = f"trajectory has no data for legs={req['legs']}"
            return
        if req["left_phase"] is not None and data.get("left") is not None:
            data["left"]["phase_shift"] = float(req["left_phase"])
        for m in self.motors:
            m.integ = 0.0
            m.prev_target = None
            m.tvel = 0.0
        self._pb = dict(req, data=data, sides=sides, t0=now,
                        track_err_peak=0.0, track_err_worst=None, track_err_over=False,
                        start_pos={id(m): self.calib.norm(self._name_of[id(m)], m.pos)
                                   for m in self.motors})
        # no _enter_hold here: playback already ramps out of start_pos, whose first command is
        # calib.raw(calib.norm(pos)) == pos exactly, i.e. it holds where the robot already is.
        self._set_mode("PLAYBACK", "playback started", period=req.get("period"),
                       mode_law=req.get("mode"), sides=list(sides),
                       max_track_err=req.get("max_track_err"),
                       track_err_estop=req.get("track_err_estop"))

    def _apply_playback_patch(self, patch, now):
        pb = self._pb
        old_period = pb["period"]
        elapsed = now - pb["t0"]
        play_t = max(0.0, elapsed - pb["ramp"])
        phase = (play_t / old_period) % 1.0
        for k in ("period", "current_limit", "kp", "ki", "kd", "speed_limit",
                  "max_speed", "max_track_err"):
            if k in patch and patch[k] is not None:
                pb[k] = float(patch[k])
        pb["period"] = _clamp_period(pb["period"])   # live patch: never let phase divide by 0
        if patch.get("track_err_estop") is not None:
            pb["track_err_estop"] = bool(patch["track_err_estop"])
        if patch.get("reset_track_err"):
            # the peak is a per-attempt statistic: let the UI zero it when re-arming or retuning
            pb["track_err_peak"], pb["track_err_worst"] = 0.0, None
            pb["track_err_over"] = False
        if "left_phase" in patch and patch["left_phase"] is not None \
                and pb["data"].get("left") is not None:
            pb["data"]["left"]["phase_shift"] = float(patch["left_phase"])
        if pb["period"] != old_period and elapsed > pb["ramp"]:
            # keep phase continuity: re-anchor t0 so phase is unchanged at the new period
            pb["t0"] = now - pb["ramp"] - phase * pb["period"]

    def _tick_playback(self, now, dt):
        pb = self._pb
        if pb is None:
            self._set_mode("LIMP", "playback state vanished")
            return
        ok, why = self._motion_allowed()
        if not ok:
            self._trip(why)
            return
        self._cmd_zero_epoch = self.calib.zero_epoch    # commanding absolute positions
        # === re-implementation of play_trajectory.py:286-373 in the normalized frame ===
        elapsed = now - pb["t0"]
        ramp = min(1.0, elapsed / pb["ramp"]) if pb["ramp"] > 0 else 1.0
        play_t = max(0.0, elapsed - pb["ramp"])
        phase = (play_t / pb["period"]) % 1.0
        lim = ramp * (BYPASS_CURRENT_CEILING_A if self.bypass["torque"]
                      else pb["current_limit"])
        abd_override = {"right": pb["abd_right"], "left": pb["abd_left"]}
        active = [m for m in self.motors if m.side in pb["sides"]]

        targets_norm = {}
        for m in active:
            tgt_full = traj.reconstruct(pb["data"], m.side, phase,
                                        abduction_override=abd_override[m.side])[m.col]
            targets_norm[m] = (1 - ramp) * pb["start_pos"][id(m)] + ramp * float(tgt_full)

        limits = self.wstore.limits if self.wstore else None
        ok, reason = pt.check_workspace(pt.group_by_side(active), targets_norm, limits)
        if not ok:
            self._trip(f"playback workspace: {reason}")
            return

        pb["phase"] = phase
        for m in active:
            n = self._name_of[id(m)]
            target_raw = self.calib.raw(n, targets_norm[m])
            if pb["mode"] == "position":
                canio.set_pos(m.bus, m.cid, target_raw)
                self._last_cmd_raw[n] = target_raw
                err_deg = target_raw - m.pos
                if abs(err_deg) > pb["track_err_peak"]:
                    pb["track_err_peak"] = abs(err_deg)
                    pb["track_err_worst"] = n
                if abs(err_deg) > pb["max_track_err"]:
                    if pb["track_err_estop"]:
                        self._trip(f"{n} tracking err {err_deg:+.0f} deg "
                                   f"(> {pb['max_track_err']:.0f}) — hitting a stop?")
                        return
                    # override: keep running, but say so. The other trips are untouched.
                    pb["track_err_over"] = True
            else:
                # current mode: software PID in RAW frame -> SET_CURRENT, hard torque cap
                # (play_trajectory.py:325-345 verbatim, incl. governor + runaway cut)
                if m.prev_target is not None:
                    m.tvel = 0.3 * (target_raw - m.prev_target) / dt + 0.7 * m.tvel
                m.prev_target = target_raw
                err = target_raw - m.pos
                if ramp >= 1.0 and pb["ki"] > 0:
                    m.integ += err * dt
                    m.integ = float(np.clip(m.integ, -lim / pb["ki"], lim / pb["ki"]))
                curr = pb["kp"] * err + pb["ki"] * m.integ + pb["kd"] * (m.tvel - m.vel)
                curr = float(np.clip(curr, -lim, lim))
                if (not self.bypass["speed"] and pb["speed_limit"] > 0
                        and curr * m.spd > 0 and abs(m.spd) > pb["speed_limit"]):
                    band = 0.3 * pb["speed_limit"]
                    curr *= float(np.clip((pb["speed_limit"] + band - abs(m.spd)) / band, 0.0, 1.0))
                canio.set_current(m.bus, m.cid, curr)
                if not self.bypass["speed"] and abs(m.spd) > pb["max_speed"]:
                    self._trip(f"{n} runaway {m.spd:.0f} ERPM (> {pb['max_speed']:.0f})")
                    return

    # ----------------------------------------------------------------- MEASURE (system-ID)
    # ---------------------------------------------------------------- thermal bursts
    def thermal_start(self, spec):
        """Validate a burst request and queue it. Returns (ok, why) -- the CAN thread starts it.

        Every bound is computed HERE, against the pose the joint is actually in, rather than being
        trusted from the web request. The window is intersected with the joint's hard band and with
        the safe-workspace room it has right now, so a burst can never dither a leg into a stop or
        into the other leg."""
        if self._wiggle is not None and self._wiggle["running"]:
            return False, "an identification wiggle is still running"
        # The free-joint dither this panel shipped with self-excited on right.cam at 30 A and
        # destroyed its envelope in 0.21 s (2026-08-28); see the retraction at the top of
        # thermal_excite.py. There is no current-mode law that is both worth depositing and
        # gentle on an unloaded JOINT, so a leg on the robot has to be held by something other
        # than software.
        # Two ways a burst is allowed, and neither is the retracted free-JOINT dither:
        #   blocked -- the joint is clamped and must not move; unidirectional saturated current.
        #   free    -- the motor is off the robot with NOTHING on the shaft. Tracks a
        #              position-mode sine so the current comes from the rotor fighting its own
        #              inertia (thermal_excite.step_sine) -- unidirectional current cannot heat a
        #              free rotor (back-EMF), and the sine only works unloaded.
        rotor = str(spec.get("rotor_mode", "blocked" if spec.get("rotor_is_blocked") else ""))
        if rotor not in ("blocked", "free"):
            return False, ("a burst needs the rotor declared: 'blocked' (joint clamped, the "
                           "experiment that works) or 'free' (motor off the robot, nothing on the "
                           "shaft). The free-JOINT dither -- a leg on a spinning shaft -- is "
                           "retracted: it self-excited on right.cam at 30 A and left its window "
                           "in 0.21 s.")
        name = str(spec.get("motor", ""))
        if name not in paths.MOTOR_NAMES:
            return False, "unknown motor {!r}".format(name)
        m = self.by_name[name]
        if m.pos is None:
            return False, "{} is not reporting -- never energising a drive we cannot watch".format(name)
        try:
            amps = float(spec.get("amps", 6.0))
            duration = float(spec.get("duration_s", 5.0))
        except (TypeError, ValueError):
            return False, "amps and duration_s must be numbers"
        if not (amps == amps and duration == duration):            # NaN: json.loads makes them
            return False, "amps and duration_s must be numbers"
        if amps <= 0 or amps > THERMAL_MAX_AMPS:
            return False, "amps must be in (0, {:.0f}]".format(THERMAL_MAX_AMPS)
        # Predicted-energy gate. The parameters this uses are placeholders until the campaign is
        # finished, so it is a floor on the risk rather than a measurement -- but it is the only
        # thing standing between a 30 s burst at 30 A and a winding nobody can see cooking.
        ok, why, pred = thermal_excite.check_burst(self._thermal_params(name), amps, duration)
        if not ok:
            return False, why
        if not (thermal_excite.MIN_DURATION_S <= duration <= THERMAL_MAX_DURATION_S):
            return False, "duration must be {:.0f}-{:.0f} s".format(
                thermal_excite.MIN_DURATION_S, THERMAL_MAX_DURATION_S)

        side, role = paths.split_name(name)
        centre = self.calib.norm(name, m.pos)
        if rotor == "free":
            # Off the robot there is no workspace to intersect and no hard stop to respect --
            # the drift band in step_sine is the position bound. The sine is centred on the
            # MEASURED position, so like the wiggle it cannot slew on a wrong zero.
            try:
                freq = float(spec.get("freq_hz", THERMAL_FREE_FREQ_HZ))
                amp = float(spec.get("amp_deg", THERMAL_FREE_AMP_DEG))
            except (TypeError, ValueError):
                return False, "freq_hz and amp_deg must be numbers"
            if not (freq == freq and amp == amp):                  # NaN survives float()
                return False, "freq_hz and amp_deg must be numbers"
            if not (thermal_excite.FREE_SINE_FREQ_MIN_HZ <= freq
                    <= thermal_excite.FREE_SINE_FREQ_MAX_HZ):
                return False, "sine frequency must be {:.1f}-{:.0f} Hz".format(
                    thermal_excite.FREE_SINE_FREQ_MIN_HZ, thermal_excite.FREE_SINE_FREQ_MAX_HZ)
            if not (1.0 <= amp <= thermal_excite.FREE_SINE_AMP_MAX_DEG):
                return False, "sine amplitude must be 1-{:.0f} deg".format(
                    thermal_excite.FREE_SINE_AMP_MAX_DEG)
            try:
                env = thermal_excite.Envelope(
                    amps=amps, duration_s=duration, centre_deg=centre,
                    window_deg=amp, speed_erpm=THERMAL_SPEED_ERPM,
                    temp_abort_c=THERMAL_ABORT_TEMP_C, blocked=True, mode="free",
                    freq_hz=freq, sine_amp_deg=amp)
            except ValueError as e:
                return False, str(e)
            with self.lock:
                self._thermal_req = {"motor": name, "env": env,
                                     "ambient_c": spec.get("ambient_c")}
            return True, ""
        lo, hi = self._hard_bounds(side, role)
        # how far this joint can actually travel from here without leaving the safe workspace
        pose = {n: self.calib.norm(n, mm.pos) for n, mm in self.by_name.items()
                if mm.pos is not None}
        room_up = self._safe_room(pose, name, +1.0, THERMAL_WINDOW_DEG) if len(pose) == paths.N_MOTORS else 0.0
        room_dn = self._safe_room(pose, name, -1.0, THERMAL_WINDOW_DEG) if len(pose) == paths.N_MOTORS else 0.0
        window = min(THERMAL_WINDOW_DEG, max(room_up, room_dn))
        try:
            env = thermal_excite.Envelope(
                amps=amps, duration_s=duration, centre_deg=centre, window_deg=max(window, 1.0),
                speed_erpm=THERMAL_SPEED_ERPM, temp_abort_c=THERMAL_ABORT_TEMP_C,
                pos_lo=max(lo, centre - room_dn), pos_hi=min(hi, centre + room_up),
                blocked=True, mode=rotor)
        except ValueError as e:
            return False, str(e)
        with self.lock:
            self._thermal_req = {"motor": name, "env": env,
                                 "ambient_c": spec.get("ambient_c")}
        return True, ""

    def _thermal_params(self, motor_name):
        """Thermal parameters for one motor, fitted if we have them and placeholders otherwise.

        Deliberately re-read per call rather than cached: the whole point of the campaign is that
        these numbers change while the robot is powered, and a burst gated on a stale copy of them
        is gated on nothing."""
        import thermal as _th
        role = paths.split_name(motor_name)[1]
        kind = "AK60-39" if role == "abd" else "AKE90-8"
        try:
            fitted = _th.load_params(os.path.join(paths.DEPLOY, "thermal_params.json"))
            if kind in fitted:
                return fitted[kind]
        except (OSError, ValueError, KeyError):
            pass
        return _th.DEFAULT_PARAMS[kind]

    def thermal_stop(self):
        with self.lock:
            self._thermal_stop = True

    def get_thermal(self):
        """The finished burst, for the HTTP layer to persist. None while one is still running."""
        t = self._therm
        if t is None or t["running"]:
            return None
        return {"motor": t["motor"], "envelope": t["env"].as_dict(),
                "summary": t["ex"].summary(), "abort": t["abort"],
                "drive_t_start_c": t["t_start"], "drive_t_peak_c": t["t_peak"],
                "ambient_c": t["ambient_c"],
                "elapsed_since_end_s": (time.monotonic() - t["t_end"]) if t["t_end"] else 0.0,
                "log": {k: list(v) for k, v in t["buf"].items()}}

    def _start_thermal(self, req, now):
        m = self.by_name[req["motor"]]
        free = req["env"].free_rotor
        self._therm = {
            "motor": req["motor"], "env": req["env"],
            # the sine gets a longer ramp: its first cycles are the position loop settling, and a
            # velocity step into a bare rotor is exactly the impulse the raised cosine removes
            "ex": thermal_excite.BurstExciter(req["env"], ramp_s=1.0 if free else 0.25),
            "t0": now, "t_end": None, "running": True, "abort": None,
            # hold-before-move for the position-mode sine, same rule as manual/identify: the first
            # commands are set_pos(measured raw), which cannot slew whatever the zero says
            "t_move": now + (HOLD_BEFORE_MOVE_S if free else 0.0),
            "raw0": m.pos,
            "t_start": m.temp, "t_peak": m.temp, "ambient_c": req.get("ambient_c"),
            "buf": {"t": [], "amps_cmd": [], "amps_meas": [], "pos": [], "pos_cmd": [],
                    "spd": [], "temp": []},
            "last_rx": now,
        }
        self._last_reject = ""
        self._set_mode("THERMAL", "thermal burst started", motor=req["motor"],
                       **req["env"].as_dict())

    def _tick_thermal(self, now, dt):
        """One tick of a burst -- or of the WAIT that follows it.

        The wait is not idle time. The case temperature peaks roughly a winding time constant
        AFTER the current stops, so the drive's own peak is still ahead of us when the burst ends;
        the mode therefore stays in THERMAL, streaming limp, tracking that peak, until the operator
        saves. Dropping to LIMP at the end of the burst would throw away the measurement."""
        t = self._therm
        if t is None:
            self._set_mode("LIMP", "thermal state vanished")
            return
        m = self.by_name[t["motor"]]
        if m.pos is not None:
            t["last_rx"] = now
        # the peak keeps being tracked through the whole wait, not just the burst
        if m.temp is not None:
            t["t_peak"] = max(t["t_peak"] if t["t_peak"] is not None else m.temp, m.temp)

        if not t["running"]:
            self._stream_limp()
            return

        ok, why = self._motion_allowed()
        if not ok:
            t["running"] = False
            t["abort"] = why
            t["t_end"] = now
            self._stream_limp()
            self._trip(why)
            return

        # EVERY other drive is actively held limp for the whole burst -- unaddressed is not the
        # same as commanded, and only one motor is supposed to be moving here
        for n, mm in self.by_name.items():
            if n != t["motor"]:
                canio.set_current(mm.bus, mm.cid, 0.0)

        pos_norm = self.calib.norm(t["motor"], m.pos) if m.pos is not None else t["env"].centre
        if t["env"].free_rotor:
            # position-mode sine: the drive's own loop closes the fast loop (stable against the
            # actuation delay, unlike the retracted current-mode dither), and the heat comes from
            # the rotor fighting its own inertia. Hold-before-move first, like manual/identify.
            if now < t["t_move"]:
                if t["raw0"] is not None:
                    canio.set_pos(m.bus, m.cid, t["raw0"])
                    self._last_cmd_raw[t["motor"]] = t["raw0"]
                return
            el = now - t["t_move"]
            want, done, abort = t["ex"].step_sine(
                el, pos_norm, m.spd, m.temp, m.err, now - t["last_rx"],
                i_meas=None if m.pos is None else m.cur)
            amps_cmd = None
            if not done:
                raw = self.calib.raw(t["motor"], float(want))
                canio.set_pos(m.bus, m.cid, raw)
                self._last_cmd_raw[t["motor"]] = raw
        else:
            el = now - t["t0"]
            amps, done, abort = t["ex"].step(
                el, pos_norm, m.spd, m.temp, m.err, now - t["last_rx"],
                i_meas=None if m.pos is None else m.cur)
            canio.set_current(m.bus, m.cid, float(amps))
            amps_cmd, want = float(amps), None
        b = t["buf"]
        b["t"].append(round(el, 4))
        b["amps_cmd"].append(None if amps_cmd is None else round(amps_cmd, 3))
        b["amps_meas"].append(None if m.pos is None else round(float(m.cur), 3))
        b["pos"].append(None if m.pos is None else round(self.calib.norm(t["motor"], m.pos), 3))
        b["pos_cmd"].append(None if want is None else round(float(want), 3))
        b["spd"].append(None if m.pos is None else round(float(m.spd), 1))
        b["temp"].append(None if m.pos is None else int(m.temp))
        if done:
            t["running"] = False
            t["abort"] = abort
            t["t_end"] = now
            canio.set_current(m.bus, m.cid, 0.0)
            self._stream_limp()
            # Merge into ONE dict rather than passing abort= alongside **summary(): the
            # exciter's summary already carries `abort`, so the two collided and TypeError'd at
            # argument binding -- inside the branch that handles an ABORT, killing the control
            # loop at the exact moment it was shutting a runaway burst down (2026-08-28). Argument
            # binding happens before the callee runs, so no try/except in _bb_event could have
            # caught it; the fix has to be here, at the call.
            self._bb_event("thermal.done", **dict(t["ex"].summary(),
                                                  motor=t["motor"], abort=abort))
            self._bb_dump("thermal_burst", why=abort or "burst complete")

    # ---------------------------------------------------------------- joint identify
    # A plan costs two _safe_room scans -- about 40 _validate_pose evaluations -- and it is
    # served from the process that runs the 200 Hz CAN loop, on the same GIL. It is also POLLED by
    # the panel, and on 2026-08-29 a stale browser tab polled it 4x a second. The client is rate
    # limited now, but a client is not a safety boundary: memoise the answer briefly so no caller
    # can turn a poll into control-loop jitter. Keyed on the pose, so it still tracks a joint that
    # is actually being moved by hand.
    IDENT_PLAN_CACHE_S = 0.5

    def set_bypass(self, name, on, note=""):
        """Turn one safety limit off (or back on). Every change is an event in the black box.

        Returns (ok, message). The message is what the UI shows: it names what is now unguarded,
        because a bypass nobody remembers enabling is the failure mode this whole feature has."""
        if name not in BYPASS_NAMES:
            return False, "unknown bypass {!r} (have: {})".format(name, ", ".join(BYPASS_NAMES))
        on = bool(on)
        was = self.bypass[name]
        self.bypass[name] = on
        if was != on:
            self._bb_event("bypass", name=name, on=on, note=str(note)[:200], mode=self.mode,
                           active=[k for k, v in self.bypass.items() if v])
            if on:
                # worth a dump: whatever happens next, the record should say this was off
                self._bb_dump("bypass_enabled", why="{} limit bypassed by the operator".format(name))
        return True, ""

    def bypass_active(self):
        return [n for n in BYPASS_NAMES if self.bypass[n]]

    def identify_plan(self, spec):
        """Validate a wiggle and report the amplitude it would actually use. Queues NOTHING.

        Split out of identify_start so the panel can ask "would this work?" without energising a
        drive, and so a refusal can say which bound produced it. Returns (ok, why, plan).

        Bounds are computed HERE against the pose the joint is in right now -- the same treatment
        thermal_start gives a burst -- so a wiggle can never be argued into a hard stop by the web
        request."""
        if self._therm is not None and self._therm["running"]:
            return False, "a thermal burst is still running", None
        name = str(spec.get("motor", ""))
        if name not in paths.MOTOR_NAMES:
            return False, "unknown motor {!r}".format(name), None
        m = self.by_name[name]
        if m.pos is None:
            return False, "{} is not reporting -- never energising a drive we cannot watch".format(name), None
        try:
            want_amp = float(spec.get("amp_deg", IDENT_AMP_DEG))
            duration = float(spec.get("duration_s", IDENT_DURATION_S))
        except (TypeError, ValueError):
            return False, "amp_deg and duration_s must be numbers", None
        if not (want_amp == want_amp and duration == duration):   # NaN survives float()
            return False, "amp_deg and duration_s must be numbers", None
        if not (0.0 < want_amp <= IDENT_MAX_AMP_DEG):
            return False, "amplitude must be in (0, {:.0f}] deg".format(IDENT_MAX_AMP_DEG), None
        if not (0.5 <= duration <= IDENT_MAX_DURATION_S):
            return False, "duration must be 0.5-{:.0f} s".format(IDENT_MAX_DURATION_S), None

        side, role = paths.split_name(name)
        centre = self.calib.norm(name, m.pos)
        lo, hi = self._hard_bounds(side, role)
        pose = {n: self.calib.norm(n, mm.pos) for n, mm in self.by_name.items()
                if mm.pos is not None}
        if len(pose) != paths.N_MOTORS:
            return False, "not every joint is reporting -- cannot bound the wiggle", None

        # pose rounded to 0.5 deg: fine enough that a hand-moved joint re-plans, coarse enough
        # that encoder dither does not defeat the cache
        ckey = (name, round(want_amp, 2), round(duration, 2), self.calib.zero_epoch,
                tuple(round(pose[n] * 2.0) for n in paths.MOTOR_NAMES))
        hit = self._plan_cache
        if hit is not None and hit[0] == ckey and (time.monotonic() - hit[1]) < self.IDENT_PLAN_CACHE_S:
            return hit[2]

        # The joint's OWN never-exceed band. This is the bound that actually stops a joint hitting
        # an end stop, and it always applies.
        hard_up, hard_dn = max(0.0, hi - centre), max(0.0, centre - lo)

        # The gait-feasibility polygon, which is a different question: can the LEGS be in this
        # configuration. It can only advise from inside itself -- _safe_room scans outward from the
        # current pose and returns 0.0 for every direction when that pose is already outside the
        # recorded set. A robot hanging limp on a stand, sagging, is routinely outside it, and
        # taking that 0.0 at face value refused every wiggle on the real robot (2026-08-28) with a
        # message about being near a stop that was not true.
        #
        # Homing already resolves exactly this: "_manual_override = homing -- homing trusts the
        # feasibility net, not the eroded gait polygon". A +-5 deg dither about a joint's MEASURED
        # position, clamped to its hard band, is in the same category. So the polygon SHRINKS the
        # amplitude when it can speak, and is skipped -- loudly, in the returned plan -- when it
        # cannot.
        pose_ok, pose_why = self._validate_pose(pose, override=False)
        if pose_ok:
            amp = min(want_amp, self._safe_room(pose, name, +1.0, want_amp),
                      self._safe_room(pose, name, -1.0, want_amp), hard_up, hard_dn)
            bound = "safe workspace"
            note = ""
        else:
            amp = min(want_amp, hard_up, hard_dn)
            bound = "hard limits"
            note = ("the current pose is outside the recorded safe workspace ({}), so the wiggle "
                    "is bounded by {}'s own hard limits instead".format(pose_why, name))

        # The pre-move guard, asked here rather than only on the CAN thread. _activation_allowed
        # is the authority and still runs when the request is consumed, but it runs AFTER the HTTP
        # call has already returned 200 -- so a guard refusal reached the operator as a button that
        # did nothing. _premove_guard is documented pure, so asking it early costs nothing and
        # turns "nothing happened" into the sentence that says what to do about it.
        guard_ok, guard_why = self._motion_allowed()
        if guard_ok:
            guard_ok, guard_why, _detail = self._premove_guard()

        plan = {"motor": name, "amp_deg": float(amp), "duration_s": duration,
                "centre_deg": centre, "lo": lo, "hi": hi,
                "requested_amp_deg": want_amp, "bound": bound, "note": note,
                "hard_room_deg": round(min(hard_up, hard_dn), 2),
                "pose_in_workspace": bool(pose_ok),
                "guard_ok": bool(guard_ok), "guard_why": guard_why}
        if not guard_ok:
            return False, guard_why, plan
        if amp < IDENT_MIN_AMP_DEG:
            out = (False, ("{} has only {:.1f} deg of room here (bounded by {}) and the wiggle "
                           "needs {:.1f} -- move it away from the stop first"
                           .format(name, amp, bound, IDENT_MIN_AMP_DEG)), plan)
        else:
            out = (True, "", plan)
        self._plan_cache = (ckey, time.monotonic(), out)
        return out

    def identify_start(self, spec):
        """identify_plan, plus queueing it for the CAN thread."""
        ok, why, plan = self.identify_plan(spec)
        if not ok:
            return False, why
        with self.lock:
            self._wiggle_req = plan
        return True, ""

    def identify_stop(self):
        with self.lock:
            self._wiggle_stop = True

    def get_identify(self):
        """The finished wiggle and its verdict. None while one is still running."""
        i = self._wiggle
        if i is None or i["running"]:
            return None
        return self._identify_result(i)

    def _start_identify(self, req, now):
        self._wiggle = {
            "motor": req["motor"], "amp_deg": req["amp_deg"], "duration_s": req["duration_s"],
            "centre_deg": req["centre_deg"], "lo": req["lo"], "hi": req["hi"],
            "bound": req.get("bound", ""), "note": req.get("note", ""),
            "t0": now, "t_move": now + HOLD_BEFORE_MOVE_S, "t_end": None,
            "running": True, "abort": None,
            # raw pose at entry: what the other five are commanded to for the whole run, and the
            # datum every excursion is measured from. Raw, so none of this trusts the calibration.
            "raw0": {n: mm.pos for n, mm in self.by_name.items() if mm.pos is not None},
            "raw_lo": {}, "raw_hi": {}, "track_err": 0.0,
        }
        for n, r in self._wiggle["raw0"].items():
            self._wiggle["raw_lo"][n] = r
            self._wiggle["raw_hi"][n] = r
        self._last_reject = ""
        self._set_mode("IDENTIFY", "joint identification wiggle", motor=req["motor"],
                       amp_deg=round(req["amp_deg"], 2), duration_s=req["duration_s"])

    def _wiggle_envelope(self, u):
        """Raised-cosine in/out over IDENT_RAMP_S at each end, as a fraction of full amplitude.

        The sine already starts and ends at the centre after whole cycles; the envelope is what
        removes the velocity step, so the joint eases in and eases out instead of snapping."""
        i = self._wiggle
        ramp = min(IDENT_RAMP_S, 0.4 * i["duration_s"])
        if ramp <= 0.0:
            return 1.0
        if u < ramp:
            return 0.5 * (1.0 - np.cos(np.pi * u / ramp))
        if u > i["duration_s"] - ramp:
            return 0.5 * (1.0 - np.cos(np.pi * max(0.0, i["duration_s"] - u) / ramp))
        return 1.0

    def _tick_identify(self, now, dt):
        """One tick of the wiggle: sine on the selected joint, position hold on the other five.

        The other five are commanded to the RAW angle they were measured at when the run started,
        never through the calibration -- identical to the hold-before-move rule and for the same
        reason. It also makes the measurement clean: a LIMP joint can be back-driven through the
        4-bar by the joint that is moving, which would read as a second joint responding and would
        answer the mapping question wrongly."""
        i = self._wiggle
        if i is None:
            self._set_mode("LIMP", "identify state vanished")
            return

        # excursions are tracked across the whole run, including the hold and the tail
        for n, mm in self.by_name.items():
            if mm.pos is None or n not in i["raw_lo"]:
                continue
            i["raw_lo"][n] = min(i["raw_lo"][n], mm.pos)
            i["raw_hi"][n] = max(i["raw_hi"][n], mm.pos)

        if not i["running"]:
            self._stream_limp()                    # finished: hold limp until the panel clears it
            return

        ok, why = self._motion_allowed()
        if not ok:
            i["running"], i["abort"], i["t_end"] = False, why, now
            self._stream_limp()
            self._trip(why)
            return

        # every joint that is not under test holds where it started, for the whole run
        for n, mm in self.by_name.items():
            if n == i["motor"]:
                continue
            raw = i["raw0"].get(n)
            if raw is not None:
                canio.set_pos(mm.bus, mm.cid, raw)
                self._last_cmd_raw[n] = raw

        m = self.by_name[i["motor"]]
        if now < i["t_move"]:                      # (a) hold-before-move, on the test joint too
            raw = i["raw0"].get(i["motor"])
            if raw is not None:
                canio.set_pos(m.bus, m.cid, raw)
                self._last_cmd_raw[i["motor"]] = raw
            return

        u = now - i["t_move"]
        want = i["centre_deg"] + (i["amp_deg"] * self._wiggle_envelope(u)
                                 * float(np.sin(2.0 * np.pi * IDENT_FREQ_HZ * u)))
        want = float(np.clip(want, i["lo"], i["hi"]))
        if m.pos is not None:
            err = abs(self.calib.norm(i["motor"], m.pos) - want)
            i["track_err"] = max(i["track_err"], err)
            if err > i["amp_deg"] + IDENT_TRACK_MARGIN_DEG:
                i["running"] = False
                i["abort"] = ("{} is {:.1f} deg off a +-{:.1f} deg command -- more than a joint "
                              "that simply refuses to move could be. Most likely the calibration "
                              "sign is inverted and it is being driven away from the target."
                              .format(i["motor"], err, i["amp_deg"]))
                i["t_end"] = now
                self._stream_limp()
                self._bb_dump("identify_track_error", why=i["abort"])
                return
        raw = self.calib.raw(i["motor"], want)
        canio.set_pos(m.bus, m.cid, raw)
        self._last_cmd_raw[i["motor"]] = raw

        if u >= i["duration_s"]:                   # the envelope has already returned it to centre
            i["running"], i["t_end"] = False, now
            res = self._identify_result(i)
            self._bb_event("identify.done", motor=i["motor"], verdict=res["verdict"],
                           excursions=res["excursions"])
            self._bb_dump("identify", why="identify complete: " + res["verdict"])

    def _identify_result(self, i):
        """Per-joint excursion over the run, and what it says about the mapping.

        Excursion is max-min of the RAW encoder angle. calib.norm is sign*(raw-offset), so this is
        the same number of degrees the normalized axis moved -- but it is computed without the
        calibration, which is exactly what lets it be used to CHECK a calibration."""
        exc = {n: round(float(i["raw_hi"][n] - i["raw_lo"][n]), 2) for n in i["raw_lo"]}
        moved = sorted([n for n, v in exc.items() if v >= IDENT_MOVED_DEG], key=lambda n: -exc[n])
        sel = i["motor"]
        if i["abort"]:
            verdict, detail = "aborted", i["abort"]
        elif moved == [sel]:
            verdict = "confirmed"
            detail = "{} moved {:.1f} deg, and nothing else moved more than {:.1f}".format(
                sel, exc.get(sel, 0.0), IDENT_MOVED_DEG)
        elif not moved:
            verdict = "no-motion"
            detail = ("nothing moved more than {:.1f} deg. That drive is not following position "
                      "commands, or it is not the drive you think it is.".format(IDENT_MOVED_DEG))
        elif sel not in moved:
            verdict = "mismatch"
            detail = ("you selected {} but it barely moved ({:.1f} deg) -- {} moved instead. The "
                      "motor map is wrong.".format(sel, exc.get(sel, 0.0), " and ".join(moved)))
        else:
            others = [n for n in moved if n != sel]
            verdict = "coupled"
            detail = ("{} moved {:.1f} deg as commanded, but so did {}. Either they are "
                      "mechanically coupled, or two drives answer to the same id.".format(
                          sel, exc.get(sel, 0.0),
                          " and ".join("{} ({:.1f} deg)".format(n, exc[n]) for n in others)))
        return {"motor": sel, "amp_deg": round(i["amp_deg"], 2), "duration_s": i["duration_s"],
                "abort": i["abort"], "track_err_deg": round(i["track_err"], 2),
                "bound": i.get("bound", ""), "note": i.get("note", ""),
                "excursions": exc, "moved": moved, "verdict": verdict, "detail": detail,
                "threshold_deg": IDENT_MOVED_DEG}

    def _identify_pub(self, now):
        """Live wiggle state for the panel. None when none has been run this session."""
        i = self._wiggle
        if i is None:
            return None
        out = {"motor": i["motor"], "running": bool(i["running"]),
               "amp_deg": round(i["amp_deg"], 2), "duration_s": i["duration_s"],
               "elapsed_s": round(max(0.0, now - i["t_move"]), 2),
               "holding": now < i["t_move"], "abort": i["abort"]}
        if not i["running"]:
            out.update(self._identify_result(i))
        return out

    # ================================================================= POLICY (learned control)
    # A trained policy running on the drives, from the web UI, in this thread.
    #
    # WHY IT LIVES HERE AND NOT IN A SUBPROCESS
    # -----------------------------------------
    # robot/deploy/run_policy.py does exactly this from a terminal, and it refuses to start while
    # this daemon is up -- correctly, because the CAN bus has exactly ONE owner and two writers
    # race, the loser being a motor holding whichever frame arrived last. So "run a policy from the
    # panel" cannot mean "launch run_policy.py"; it has to mean running the control law inside the
    # loop that already owns the buses, the way THERMAL and IDENTIFY do.
    #
    # Everything that decides behaviour is still the deploy package, imported, not reimplemented:
    #   bundle.Bundle          the .npz that IS the control law (nets, stance, scales, gait cfg)
    #   controller.PolicyController   the 200 Hz law, byte-identical to what verify_export.py
    #                          proved against the torch policy inside MuJoCo
    #   safety.SafetyGovernor  clamp-then-kill on position/rate/torque/gains + the watchdogs
    #   thermal.MotorThermalModel     the winding observer that derates the torque budget
    #   jointmap.JointMap      model radians <-> normalized degrees, per joint, verified
    #   mit.pack               the force-control frame (kp-first, DLC 8 always)
    # This module contributes the parts that need the robot: the phase machine, the daemon's own
    # pre-move guard and workspace polygon, the flight-recorder events, and the dead-man.
    #
    # WHAT IT STILL DOES NOT DO: balance. Every bundle that can be deployed at all is railed in
    # roll and yaw in training (a policy that reads the privileged base velocity cannot run here at
    # all, and is refused at arm time). SUPPORT THE TORSO -- that is the `supported` acknowledgement
    # and it is the one hazard no check in this file can see.
    def policy_arm(self, spec):
        """Validate a run request end to end and queue it. Returns (ok, why, info).

        Runs on the HTTP thread on purpose: loading a bundle, building the nets and scanning the
        workspace for the stance are tens of milliseconds of work, and none of it may happen inside
        the 200 Hz tick. What crosses into the CAN thread is finished objects only."""
        info = {}
        busy = self._policy_busy_with()
        if busy:
            return False, busy, info
        if self._pol is not None and self._pol["phase"] not in ("done",):
            return False, "a policy run is already active", info

        # ---- the acknowledgement no software check can replace ---------------------------------
        if spec.get("supported") is not True:
            return False, ("confirm the torso is physically supported. Every deployable bundle was "
                           "trained with the base's roll and yaw RAILED -- it has never experienced "
                           "them free and nothing in it stabilises them. On a free-standing robot "
                           "it is open-loop there and a fall is the expected outcome."), info

        # ---- the bundle -----------------------------------------------------------------------
        fname = os.path.basename(str(spec.get("file", "")))
        if not fname.endswith(".npz") or not fname[:-4]:
            return False, "a policy bundle is a .npz file", info
        # both bundle directories: export_policy.py writes to deploy/bundles/, the panel's upload
        # writes to data/policies/, and a name resolves against whichever has it
        path = paths.find_policy_bundle(fname)
        if path is None:
            return False, "no bundle {!r} in data/policies/ or deploy/bundles/".format(fname), info
        try:
            from bundle import Bundle
            from controller import PolicyController
            import thermal as TH
            b = Bundle.load(path)
        except Exception as e:                              # noqa: BLE001 -- surfaced as data
            return False, "not a loadable policy bundle: {}".format(e), info

        hz = 1.0 / float(b.control_dt)
        if abs(hz - TICK_HZ) > 1.0:
            return False, ("this bundle wants {:.0f} Hz control and the daemon runs at {:.0f}. The "
                           "action filter, the actuation delay and the slew limit are all per-step "
                           "constants -- running it at the wrong rate is a different control law."
                           .format(hz, TICK_HZ)), info
        if bool(b.meta.get("obs_base_vel")):
            return False, ("this bundle was trained with the PRIVILEGED base velocity in its "
                           "observation (obs_base_vel=True). No robot can produce that number, so "
                           "the policy cannot be deployed at all -- re-train or export a checkpoint "
                           "with obs_base_vel=False."), info

        # ---- the joint map: a sign error here drives corrections the WRONG WAY at 200 N*m/rad ---
        if tuple(JM.MOTOR_NAMES) != tuple(paths.MOTOR_NAMES):
            return False, ("jointmap and paths disagree about motor ordering -- refusing to guess "
                           "which column is which joint"), info
        jm_path = os.path.join(paths.DEPLOY, "deploy_map.json")
        try:
            jm = JM.JointMap.load(jm_path)
        except OSError:
            return False, ("robot/deploy/deploy_map.json does not exist -- the model->motor joint "
                           "map has never been built on this robot. Run "
                           "robot/deploy/make_deploy_map.py after verifying fklut."), info
        except (ValueError, KeyError) as e:
            return False, "deploy_map.json is unreadable: {}".format(e), info
        jm_ok, jm_why = jm.check_ready()
        if not jm_ok and spec.get("skip_jointmap_check") is not True:
            return False, jm_why, info

        # ---- the IMU: gravity is the only thing that can say the robot has fallen ---------------
        sh = self.sense
        no_imu = spec.get("no_imu") is True
        if not no_imu:
            if sh is None:
                return False, ("no Sense HAT: the fall detector reads gravity and there is none. "
                               "Start the server without --no-sensors, or acknowledge no_imu and "
                               "keep the robot physically restrained."), info
            if sh.fast() is None:
                return False, ("the IMU has not produced a sample yet -- refusing to run a "
                               "balance-relevant controller blind"), info
            if not self.mock and not getattr(getattr(sh, "mount", None), "calibrated", False):
                return False, ("the IMU mount rotation has not been calibrated, so 'up' is in CHIP "
                               "axes rather than body axes. Every gravity reading the policy sees "
                               "would be rotated. Run the mount calibration first."), info

        # ---- the thermal observer ---------------------------------------------------------------
        try:
            amb = float(spec.get("ambient_c", 25.0))
        except (TypeError, ValueError):
            return False, "ambient_c must be a number", info
        chain = [self._thermal_params(n) for n in JM.MODEL_TO_MOTOR]
        uncal = sorted({p.name for p in chain if not p.calibrated})
        if uncal and spec.get("allow_uncalibrated_thermal") is not True:
            return False, ("the thermal parameters for {} are UNCALIBRATED placeholders, so the "
                           "winding-temperature estimate that derates every torque budget is a "
                           "guess. Fit them (thermal panel), or acknowledge "
                           "allow_uncalibrated_thermal.".format(", ".join(uncal))), info
        try:
            thermal = TH.MotorThermalModel(chain, dt=1.0 / TICK_HZ, t_amb=amb,
                                           names=list(JM.MODEL_ACTUATORS),
                                           allow_uncalibrated=True)
        except ValueError as e:
            return False, str(e), info

        # ---- the governor's envelope ------------------------------------------------------------
        # Start from what the policy was TRAINED against (the model's own ctrlrange and force
        # ranges) and narrow it with THIS robot's hard bounds. Narrowing is the only direction that
        # is ever safe, and it is the direction the daemon's own limits point.
        lo_norm = np.empty(paths.N_MOTORS)
        hi_norm = np.empty(paths.N_MOTORS)
        for i, n in enumerate(paths.MOTOR_NAMES):
            lo_norm[i], hi_norm[i] = self._hard_bounds(*paths.split_name(n))
        e0, e1 = jm.to_model_rad(lo_norm), jm.to_model_rad(hi_norm)
        hard_lo, hard_hi = np.minimum(e0, e1), np.maximum(e0, e1)

        i_cont = np.array([p.i_continuous(amb) for p in chain])
        peak = np.asarray(b["forcerange"], float)
        tau_cont = np.minimum(peak, jm.kt_joint * jm.kt_efficiency * i_cont)
        try:
            amp_cap = spec.get("drive_amp_limit")
            amp_cap = None if amp_cap in (None, "") else float(amp_cap)
        except (TypeError, ValueError):
            return False, "drive_amp_limit must be a number", info
        if amp_cap:
            peak = np.minimum(peak, jm.kt_joint * jm.kt_efficiency * amp_cap)
        try:
            max_s = float(spec.get("max_seconds", POLICY_DEFAULT_SECONDS))
        except (TypeError, ValueError):
            return False, "max_seconds must be a number", info
        if not (max_s == max_s) or not (1.0 <= max_s <= POLICY_MAX_SECONDS):
            return False, "max_seconds must be 1-{:.0f}".format(POLICY_MAX_SECONDS), info

        limits = SAFE.Limits.from_bundle(b, hard_lo=hard_lo, hard_hi=hard_hi, tau_cont=tau_cont,
                                         deadman_s=POLICY_DEADMAN_S,
                                         telemetry_stale_s=POLICY_TELEMETRY_STALE_S)
        limits.tau_peak = peak
        gov = SAFE.SafetyGovernor(limits, 1.0 / TICK_HZ, thermal=thermal,
                                  names=list(JM.MODEL_ACTUATORS))

        # ---- the stance has to be somewhere this robot can actually stand ----------------------
        stance = np.asarray(b["nominal_ctrl"], float)
        stance_norm = jm.to_norm_deg(stance)
        pose = {n: float(stance_norm[i]) for i, n in enumerate(paths.MOTOR_NAMES)}
        ok, why = self._validate_pose(pose, override=False)
        if not ok:
            return False, ("the stance this policy centres its gait on is not a pose this robot "
                           "may hold: {}. The approach would refuse on its first tick. Extend the "
                           "recorded workspace, or switch the workspace guard off in the top bar "
                           "and accept that self-collision is no longer checked.".format(why)), info
        for n, v in pose.items():
            lo, hi = self._hard_bounds(*paths.split_name(n))
            if not lo <= v <= hi:
                return False, ("the policy's stance puts {} at {:+.1f} deg, outside its hard limit "
                               "[{:+.0f}, {:+.0f}] -- the joint map or the calibration is wrong"
                               .format(n, v, lo, hi)), info

        # ---- the command, inside the box this checkpoint was trained to ------------------------
        try:
            v_want = float(spec.get("v_cmd", 0.0))
            yaw_want = float(spec.get("yaw_cmd", 0.0))
        except (TypeError, ValueError):
            return False, "v_cmd and yaw_cmd must be numbers", info
        if not (v_want == v_want and yaw_want == yaw_want):
            return False, "v_cmd and yaw_cmd must be numbers", info
        v_cmd = float(np.clip(v_want, -float(b.cmd_v_back_trained), float(b.cmd_v_fwd_trained)))
        yaw_lim = float(b.cmd_yaw_trained)
        yaw_cmd = float(np.clip(yaw_want, -yaw_lim, yaw_lim))

        ctrl = PolicyController(b)
        # Can this machine actually run it? Timed HERE, on the HTTP thread, with the real bundle
        # and the real numpy -- not estimated. ctrl.step mutates the controller, which is fine:
        # _tick_policy calls ctrl.start() when it enters RUN and start() reallocates everything.
        step_ms = self._policy_probe(ctrl, b)
        slow = step_ms > POLICY_MAX_STEP_MS
        if slow and spec.get("allow_slow_loop") is not True:
            budget_ms = 1000.0 / TICK_HZ
            return False, (
                "this machine needs {:.1f} ms per control tick for this bundle and the loop period "
                "is {:.1f} ms. The control law's dt is a CONSTANT, so running it anyway plays the "
                "gait at about {:.2f}x speed and inflates every joint velocity the policy observes "
                "by about {:.1f}x -- an observation it was never trained on. Acknowledge "
                "allow_slow_loop to bring the drives up anyway and watch them move; the lasting "
                "fix is a numpy linked against OpenBLAS, since most of that time is three matrix "
                "multiplies running on the reference BLAS.".format(
                    step_ms, budget_ms, min(1.0, budget_ms / max(step_ms, 1e-6)),
                    max(1.0, step_ms / budget_ms))), info
        # preallocated, because a 200 Hz loop must not allocate: approach budget + run + slack
        rows = int((max_s + POLICY_APPROACH_MAX_S + 2.0) * TICK_HZ) + 8
        req = {
            "file": fname, "bundle": b, "ctrl": ctrl, "gov": gov, "thermal": thermal, "jm": jm,
            "stance": stance, "v_cmd": v_cmd, "yaw_cmd": yaw_cmd, "max_seconds": max_s,
            "ambient_c": amb, "no_imu": no_imu, "log": np.zeros((rows, POLICY_LOG_COLS), np.float32),
            "jm_verified": jm_ok, "thermal_uncalibrated": bool(uncal),
            "step_ms": step_ms, "slow_loop": slow,
        }
        info = {"file": fname, "run": b.meta.get("run"), "checkpoint": b.meta.get("checkpoint"),
                "v_cmd": v_cmd, "yaw_cmd": yaw_cmd, "max_seconds": max_s,
                "v_cmd_clamped": v_cmd != v_want, "yaw_cmd_clamped": yaw_cmd != yaw_want,
                "stance_norm_deg": {k: round(v, 2) for k, v in pose.items()},
                "tau_peak": np.round(peak, 1).tolist(),
                "tau_cont": np.round(tau_cont, 1).tolist(),
                "thermal_uncalibrated": bool(uncal), "jointmap_verified": jm_ok,
                "no_imu": no_imu, "log_rows": rows,
                "step_ms": round(step_ms, 2), "slow_loop": slow,
                "tick_budget_ms": round(1000.0 / TICK_HZ, 2)}
        self._pol_deadman = time.monotonic()          # arm the dead-man before the mode change
        with self.lock:
            self._pol_req = req
        return True, "", info

    def _policy_probe(self, ctrl, b):
        """Median milliseconds per ctrl.step() on THIS machine, with THIS bundle. Measured, never
        assumed: the same numpy on the same Pi is 90x off the figure the port was estimated at,
        because whether it found OpenBLAS is not something the code can see."""
        pos = np.asarray(b["nominal_ctrl"], float)
        zero6, grav, gyro = np.zeros(6), POLICY_UPRIGHT_GRAV.copy(), np.zeros(3)
        ctrl.start(pos, zero6, zero6, grav, gyro)
        for _ in range(5):                                   # first calls allocate; discard them
            ctrl.step(pos, zero6, zero6, grav, gyro)
        ts = []
        for _ in range(POLICY_PROBE_TICKS):
            t0 = time.perf_counter()
            ctrl.step(pos, zero6, zero6, grav, gyro)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return float(ts[len(ts) // 2]) * 1e3                 # median: one scheduler hiccup is not it

    def _policy_busy_with(self):
        """The other exclusive activities. One thread, one bus, one thing at a time."""
        if self.mode == "ESTOPPED":
            return "the e-stop is latched -- clear it first"
        if self._therm is not None and self._therm["running"]:
            return "a thermal burst is still running"
        if self._wiggle is not None and self._wiggle["running"]:
            return "an identification wiggle is still running"
        if self._meas is not None and self._meas["running"]:
            return "a system-ID excitation is still running"
        if self.mode == "PLAYBACK":
            return "a gait playback is running -- stop it first"
        return ""

    def policy_stop(self, hard=False):
        """Ask for a stop. SOFT freezes the target and bleeds the gains out over ~0.3 s, which
        puts the robot down under control; HARD zeroes them this tick."""
        with self.lock:
            self._pol_stop = "hard" if hard else "soft"

    def policy_keepalive(self):
        """The dead-man. The panel refreshes this while the run is visible on screen; if it stops
        arriving -- tab closed, page hidden, Wi-Fi gone -- the governor soft-stops within
        POLICY_DEADMAN_S. It is deliberately NOT refreshed by the status poll: a poll proves a
        browser is alive, not that anyone is watching.

        Lock-free on purpose, and _deadman_age below reads it the same way: a single float store
        is atomic under the GIL, and taking self.lock here would be a bug rather than caution --
        _publish() calls _policy_pub() while ALREADY holding it, threading.Lock is not reentrant,
        and the deadlock lands in the thread that owns both CAN buses."""
        self._pol_deadman = time.monotonic()

    def get_policy(self):
        """The finished run, for the HTTP layer to persist. None while one is still active."""
        p = self._pol
        if p is None or p["phase"] != "done":
            return None
        g = p["gov"].status()
        return {
            "file": p["file"], "run": p["run_name"], "checkpoint": p["checkpoint"],
            "v_cmd": p["v_cmd"], "yaw_cmd": p["yaw_cmd"], "max_seconds": p["max_seconds"],
            "ambient_c": p["ambient_c"], "exit_reason": p["exit_reason"],
            "reached_run": p["reached_run"], "run_seconds": round(p["run_seconds"], 2),
            "governor": g, "no_imu": p["no_imu"],
            "step_ms": round(float(p["step_ms"]), 2), "slow_loop": bool(p["slow_loop"]),
            "realised_hz": round(float(p["rate_hz"]), 1), "nominal_hz": float(TICK_HZ),
            "thermal_uncalibrated": p["thermal_uncalibrated"],
            "jointmap_verified": p["jm_verified"],
            "peak_winding_c": np.round(p["thermal"].peak_w, 1).tolist(),
            "ticks": int(p["n"]), "late_ticks": int(self._slip_count - p["slip0"]),
            "columns": POLICY_LOG_COLUMNS,
            "log": p["log"][:int(p["n"])],
        }

    def _start_policy(self, req, now):
        p = dict(req)
        b = req["bundle"]
        p.update(phase="hold", t0=now, t_move=now + HOLD_BEFORE_MOVE_S, t_phase=now,
                 t_end=None, run_t0=None, run_seconds=0.0, reached_run=False,
                 exit_reason=None, n=0, slip0=self._slip_count, ws_block=0, ws_blocked_total=0,
                 last_ws_target=None, prev_pos=None, approach_total=0.0, approach_f=0.0,
                 freq=0.0, gait_phase=0.0, saturated=0,
                 rate_t0=None, rate_n=0, rate_hz=float(TICK_HZ),
                 run_name=b.meta.get("run"), checkpoint=b.meta.get("checkpoint"),
                 imu_age=0.0, tel_age=0.0)
        self._pol = p
        with self.lock:
            # a stop posted after the previous run had already ended is still sitting there, and
            # it would kill this one on its first tick
            self._pol_stop = None
        self._last_reject = ""
        self._enter_hold(now)
        # the observer must start at the drives' reported temperature, never at ambient: a robot
        # switched on after a run is not cold, and this is the one estimate that must never be
        # optimistic
        t_now = np.array([float(self.by_name[n].temp) for n in JM.MODEL_TO_MOTOR])
        p["thermal"].reset(t_now, t_amb=p["ambient_c"])
        self._set_mode("POLICY", "policy run armed", file=p["file"], run=p["run_name"],
                       checkpoint=p["checkpoint"], v_cmd=p["v_cmd"], yaw_cmd=p["yaw_cmd"],
                       max_seconds=p["max_seconds"], no_imu=p["no_imu"],
                       step_ms=round(p["step_ms"], 2), slow_loop=p["slow_loop"],
                       jointmap_verified=p["jm_verified"],
                       thermal_uncalibrated=p["thermal_uncalibrated"],
                       bypass=[k for k, v in self.bypass.items() if v])

    # ---------------------------------------------------------------- POLICY: the tick
    def _policy_imu(self):
        """(gravity_body, gyro_body, age_s) in the MuJoCo convention the controller expects.

        The Sense HAT publishes world UP in body axes; gravity is world DOWN, so grav = -up."""
        sh = self.sense
        if sh is None:
            return POLICY_UPRIGHT_GRAV.copy(), np.zeros(3), 0.0
        f = sh.fast()
        if f is None:
            return POLICY_UPRIGHT_GRAV.copy(), np.zeros(3), 1e9
        t, up, gyr = f
        return -np.asarray(up, float), np.asarray(gyr, float), max(0.0, time.time() - t)

    def _policy_measure(self, p):
        """Everything the controller and the governor read, in the MODEL frame."""
        jm = p["jm"]
        norm = np.array([self.calib.norm(n, self.by_name[n].pos) for n in paths.MOTOR_NAMES])
        amps = np.array([self.by_name[n].cur for n in paths.MOTOR_NAMES])
        temp = np.array([float(self.by_name[n].temp) for n in JM.MODEL_TO_MOTOR])
        err = np.array([int(self.by_name[n].err) for n in JM.MODEL_TO_MOTOR])
        return jm.to_model_rad(norm), amps, temp, err

    def _policy_send(self, p, target_model, kp_model, kd_model):
        """Model-frame command -> six force-control frames. Returns the clamped wire fields.

        Force control, not SET_POS: kp and kd are per-joint per-tick outputs of the policy's
        impedance channel, and the drive computes tau = kp*(p_des - p) - kd*v, which is exactly the
        MuJoCo position actuator the policy trained against. v_des and tau_ff stay at zero because
        their wire spans have never been identified for these motors (see mit.py)."""
        jm = p["jm"]
        norm_deg = jm.to_norm_deg(target_model)
        kp_m = np.asarray(kp_model, float)[jm.motor_from_model]
        kd_m = np.asarray(kd_model, float)[jm.motor_from_model]
        clamped = set()
        self._cmd_zero_epoch = self.calib.zero_epoch          # commanding absolute positions
        for i, n in enumerate(paths.MOTOR_NAMES):
            m = self.by_name[n]
            raw = self.calib.raw(n, float(norm_deg[i]))
            payload, cl = mit.pack(np.radians(raw), float(kp_m[i]), float(kd_m[i]))
            clamped.update(cl)
            canio.force_control(m.bus, m.cid, payload)
            self._last_cmd_raw[n] = raw
        self._force_until = self._tick_mono + POLICY_FORCE_RELEASE_S
        return clamped

    def _policy_workspace(self, p, target_model):
        """(ok, why) for a MODEL-frame command, against the recorded safe workspace.

        This is the check the governor cannot make. The governor bounds each joint independently,
        and self-collision is a property of the COMBINATION -- two individually-legal angles can
        put the legs through each other. It is also the one guard here the operator can switch off
        (the `workspace` bypass in the top bar), and _validate_pose already honours that."""
        norm = p["jm"].to_norm_deg(target_model)
        pose = {n: float(norm[i]) for i, n in enumerate(paths.MOTOR_NAMES)}
        return self._validate_pose(pose, override=False)

    def _policy_hold_frame(self, p):
        """Hold the pose the joints were MEASURED at, in raw encoder degrees, at approach gains.

        The same hold-before-move discipline MANUAL uses, in the force-control frame: the command
        never passes through calibration.offsets, so it is correct even against a completely wrong
        zero. Whatever the offsets say, commanding where the joint already is cannot slew."""
        for n, m in self.by_name.items():
            raw = self._hold_raw.get(n)
            if raw is None:
                continue
            payload, _ = mit.pack(np.radians(raw), POLICY_APPROACH_KP, POLICY_APPROACH_KD)
            canio.force_control(m.bus, m.cid, payload)
            self._last_cmd_raw[n] = raw
        self._force_until = self._tick_mono + POLICY_FORCE_RELEASE_S

    def _policy_limp(self):
        """Zero-gain force frame AND SET_CURRENT 0, streamed. Both, because the drive holds its
        last command in whichever mode it is in and 'stop sending' is not 'stop commanding'."""
        payload = mit.limp_payload()
        for m in self.motors:
            canio.force_control(m.bus, m.cid, payload)
            canio.set_current(m.bus, m.cid, 0.0)
        self._last_cmd_raw.clear()

    def _policy_end(self, p, now, reason):
        p["phase"] = "done"
        p["t_end"] = now
        p["exit_reason"] = reason
        self._policy_limp()
        self._force_until = self._tick_mono + POLICY_FORCE_RELEASE_S
        g = p["gov"].status()
        self._bb_event("policy.done", file=p["file"], run=p["run_name"],
                       checkpoint=p["checkpoint"], reason=reason, reached_run=p["reached_run"],
                       run_seconds=round(p["run_seconds"], 2), ticks=int(p["n"]),
                       late_ticks=int(self._slip_count - p["slip0"]), governor=g,
                       ws_blocked_ticks=int(p["ws_blocked_total"]),
                       peak_winding_c=np.round(p["thermal"].peak_w, 1).tolist())
        self._bb_dump("policy_run", why=reason)

    def _policy_log(self, p, t, pos, vel, tau, amps, temp, grav, gyro, v, stop_code):
        """One row into the preallocated buffer. `amps` arrives in MOTOR order and every other
        six-vector here is in MODEL actuator order, so it is reindexed -- a log whose columns are
        in two different orders is a log that will be read wrong once."""
        n = int(p["n"])
        if n >= p["log"].shape[0]:
            return
        row = p["log"][n]
        row[0] = t
        row[1:7] = pos
        row[7:13] = vel
        row[13:19] = tau
        row[19:25] = np.asarray(amps, float)[p["jm"].model_from_motor]
        row[25:31] = temp
        row[31:34] = grav
        row[34:37] = gyro
        row[37:43] = v.target
        row[43:49] = v.kp
        row[49:55] = v.kd
        row[55:61] = p["thermal"].t_winding
        row[61] = p["gait_phase"]
        row[62] = p["freq"]
        row[63] = stop_code
        p["n"] = n + 1

    def _tick_policy(self, now, dt):
        p = self._pol
        if p is None:
            self._set_mode("LIMP", "policy state vanished")
            return
        if p["phase"] == "done":
            self._policy_limp()
            return

        ok, why = self._motion_allowed()
        if not ok:
            self._policy_end(p, now, why)
            self._trip(why)
            return

        gov = p["gov"]
        stale = self._telemetry_age()
        deadman = self._deadman_age()
        grav, gyro, imu_age = self._policy_imu()
        p["tel_age"], p["imu_age"] = stale, imu_age

        with self.lock:
            req_stop = self._pol_stop
            self._pol_stop = None
        if req_stop:
            gov.kill("stopped by the operator", hard=(req_stop == "hard"))

        # ---- watchdogs that apply in EVERY phase, not just under the policy --------------------
        if stale > POLICY_TELEMETRY_STALE_S:
            gov.kill("telemetry stale by {:.0f} ms -- never command a joint we cannot see"
                     .format(stale * 1e3), hard=True)
        if deadman > POLICY_DEADMAN_S:
            gov.kill("dead-man not refreshed for {:.2f} s -- the panel stopped saying it is "
                     "watching".format(deadman), hard=False)
        if not p["no_imu"] and imu_age > POLICY_IMU_STALE_S:
            gov.kill("IMU sample is {:.0f} ms old -- gravity is the only fall detector there is"
                     .format(imu_age * 1e3), hard=True)

        # ---- measured speed, which is NOT what the governor's rate clamp bounds -------------
        # The governor limits how fast the TARGET may move. This is how fast the joint is actually
        # turning, and on 2026-09-01 the difference between those two was four faulted drives.
        if not self.bypass["speed"]:
            for n, m in self.by_name.items():
                if m.pos is not None and abs(m.spd) > POLICY_MAX_ERPM:
                    why = ("{} runaway {:.0f} ERPM (> {:.0f}) -- going limp rather than braking, "
                           "because braking a joint at this speed is what puts the bus into "
                           "over-voltage".format(n, m.spd, POLICY_MAX_ERPM))
                    self._policy_end(p, now, why)
                    self._trip(why)
                    return

        pos, amps, temp, err = self._policy_measure(p)
        if p["prev_pos"] is None:
            p["prev_pos"] = pos.copy()
        # Joint velocity from the DIFFERENTIATED model position at the NOMINAL period, exactly as
        # run_policy.py does: the drive reports ERPM and nothing in this repo has ever measured the
        # ERPM-to-joint-speed scale. Noisy but unambiguous, and inside the noise band the policy
        # trained against.
        vel = (pos - p["prev_pos"]) * TICK_HZ
        p["prev_pos"] = pos.copy()
        tau = p["jm"].torque_to_model(amps)
        amps_model = np.abs(amps)[p["jm"].model_from_motor]

        # ---- HOLD: the first commands are where the joints already are --------------------------
        if p["phase"] == "hold":
            if gov.stop != SAFE.STOP_NONE:
                self._policy_end(p, now, "; ".join(gov.reasons) or "stopped before the approach")
                return
            if now < p["t_move"]:
                gov.observe(amps_model, omega=np.abs(vel), drive_temp=temp, t_amb=p["ambient_c"])
                self._policy_hold_frame(p)
                return
            p["phase"] = "approach"
            p["t_phase"] = now
            p["approach_start"] = pos.copy()
            travel = float(np.max(np.abs(p["stance"] - pos)))
            p["approach_total"] = travel / np.radians(POLICY_APPROACH_DPS)
            self._bb_event("policy.approach", file=p["file"],
                           travel_deg=round(float(np.degrees(travel)), 1),
                           seconds=round(p["approach_total"], 2), dps=POLICY_APPROACH_DPS)
            return

        # ---- APPROACH: crawl to the stance at a gain that can carry a leg and nothing more ------
        if p["phase"] == "approach":
            if gov.stop != SAFE.STOP_NONE:
                self._policy_end(p, now, "; ".join(gov.reasons) or "stopped during the approach")
                return
            if np.any(err):
                bad = ", ".join(JM.MODEL_ACTUATORS[i] for i in np.flatnonzero(err))
                self._policy_end(p, now, "drive error flag on {} during the approach".format(bad))
                return
            el = now - p["t_phase"]
            f = 1.0 if p["approach_total"] <= 0 else min(1.0, el / p["approach_total"])
            p["approach_f"] = f
            tgt = p["approach_start"] + (p["stance"] - p["approach_start"]) * f
            # The endpoints were both checked -- the stance at arm time, the measured pose is
            # where the robot already is -- but a straight line between two safe poses is not
            # itself safe, and this one sweeps both legs at once. Aborted rather than frozen: the
            # approach is a deterministic ramp, so a refusal here is permanent, not transient.
            ws_ok, ws_why = self._policy_workspace(p, tgt)
            if not ws_ok:
                p["ws_blocked_total"] += 1
                self._policy_end(p, now, "the approach path leaves the safe workspace: {} -- move "
                                         "the legs closer to the policy's stance by hand, or "
                                         "extend the recorded workspace".format(ws_why))
                return
            p["last_ws_target"] = tgt.copy()
            miss = np.abs(pos - tgt)
            if float(np.max(miss)) > np.radians(POLICY_APPROACH_TRACK_ERR_DEG):
                i = int(np.argmax(miss))
                self._policy_end(p, now, (
                    "{} is {:.1f} deg from its approach target -- either the joint map is wrong, "
                    "the zero is stale, or the leg is obstructed".format(
                        JM.MODEL_ACTUATORS[i], float(np.degrees(miss[i])))))
                return
            # the observer tracks through the approach as well -- the drive's case temperature
            # is what corrects it, and that correction should not wait for the policy to start
            gov.observe(amps_model, omega=np.abs(vel), drive_temp=temp, t_amb=p["ambient_c"])
            self._policy_send(p, tgt, np.full(6, POLICY_APPROACH_KP),
                              np.full(6, POLICY_APPROACH_KD))
            if f >= 1.0 and float(np.max(np.abs(pos - p["stance"]))) < np.radians(
                    POLICY_APPROACH_ARRIVE_DEG):
                p["phase"] = "run"
                p["t_phase"] = now
                p["run_t0"] = now
                p["reached_run"] = True
                p["ctrl"].start(pos, np.zeros(6), tau, grav, gyro,
                                v_cmd=p["v_cmd"], yaw_cmd=p["yaw_cmd"])
                self._bb_event("policy.run", file=p["file"], v_cmd=p["v_cmd"],
                               yaw_cmd=p["yaw_cmd"], max_seconds=p["max_seconds"])
                return
            if el > p["approach_total"] + POLICY_APPROACH_SLACK_S:
                self._policy_end(p, now, "the approach did not converge within its budget")
            return

        # ---- RUN --------------------------------------------------------------------------------
        t = now - p["run_t0"]
        p["run_seconds"] = t
        if t >= p["max_seconds"]:
            gov.kill("max run time {:.0f} s reached".format(p["max_seconds"]), hard=False)

        # Realised rate, measured over a rolling window. The arm-time probe says the machine
        # CAN; this says it IS -- with the IMU thread, Flask, the recorder and the GC all competing
        # for the same four cores. See the POLICY_MAX_STEP_MS block for why a slow loop is not a
        # slightly-degraded control law but a different one.
        if p["rate_t0"] is None:
            p["rate_t0"], p["rate_n"] = now, 0
        p["rate_n"] += 1
        if p["rate_n"] >= POLICY_RATE_WINDOW:
            el = max(now - p["rate_t0"], 1e-9)
            p["rate_hz"] = p["rate_n"] / el
            p["rate_t0"], p["rate_n"] = now, 0
            if p["rate_hz"] < POLICY_MIN_RATE_FRAC * TICK_HZ and not p["slow_loop"]:
                gov.kill("the control loop is running at {:.0f} Hz, not {:.0f} -- the control law's "
                         "dt is a constant, so the gait is playing at {:.2f}x and every observed "
                         "joint velocity is inflated {:.1f}x".format(
                             p["rate_hz"], TICK_HZ, p["rate_hz"] / TICK_HZ,
                             TICK_HZ / max(p["rate_hz"], 1e-6)), hard=False)

        cmd = p["ctrl"].step(pos, vel, tau, grav, gyro)
        p["gait_phase"], p["freq"] = float(cmd.phase), float(cmd.freq)
        p["saturated"] += 1 if cmd.saturated else 0
        # the governor steps the winding observer itself, off `current` -- see
        # safety.SafetyGovernor.observe for why that is not the caller's job any more
        v = gov.step(cmd.target, cmd.kp, cmd.kd, pos, vel, grav, gyro,
                     telemetry_age=stale, deadman_age=deadman, drive_temp=temp, drive_err=err,
                     current=amps_model, t_amb=p["ambient_c"])

        # ---- the daemon's own workspace polygon, which the governor knows nothing about ---------
        # The governor bounds each joint independently; only the recorded workspace knows that two
        # individually-legal joint angles can put the legs through each other. It cannot be
        # clamped (it is a polygon, not an interval), so: FREEZE at the last pose that passed, and
        # kill if the policy is still asking for the same forbidden place 100 ms later. Same
        # clamp-now-kill-if-persistent rule the governor uses, for the same reason -- dropping a
        # standing robot on one transient is worse than the transient.
        if not v.limp:
            ws_ok, ws_why = self._policy_workspace(p, v.target)
            if ws_ok:
                p["ws_block"] = 0
                p["last_ws_target"] = v.target.copy()
            elif p["last_ws_target"] is not None:
                p["ws_block"] += 1
                p["ws_blocked_total"] += 1
                v.target = p["last_ws_target"].copy()
                self._last_reject = ws_why
                if p["ws_block"] >= POLICY_WS_PERSIST_TICKS:
                    gov.kill("the policy has been commanding outside the safe workspace for "
                             "{:.0f} ms: {}".format(p["ws_block"] * 1000.0 / TICK_HZ, ws_why),
                             hard=False)
            else:
                # nothing safe to fall back to: the very first commanded pose is already outside
                gov.kill("the first commanded pose is outside the safe workspace: {}".format(
                    ws_why), hard=True)

        if v.limp:
            self._policy_limp()
        else:
            self._policy_send(p, v.target, v.kp, v.kd)

        self._policy_log(p, t, pos, vel, tau, amps, temp, grav, gyro, v, float(v.stop))
        if v.stop != SAFE.STOP_NONE and v.limp:
            self._policy_end(p, now, "; ".join(v.reasons) or "stopped")

    def _policy_pub(self, now):
        """Live run state for the panel. None when no run has been armed this session."""
        p = self._pol
        if p is None:
            return None
        g = p["gov"].status()
        out = {
            "file": p["file"], "run": p["run_name"], "checkpoint": p["checkpoint"],
            "phase": p["phase"], "running": p["phase"] not in ("done",),
            "v_cmd": p["v_cmd"], "yaw_cmd": p["yaw_cmd"], "max_seconds": p["max_seconds"],
            "elapsed_s": round(p["run_seconds"], 2),
            "approach_frac": round(float(p["approach_f"]), 3),
            "gait_freq_hz": round(p["freq"], 2),
            "rate_hz": round(float(p["rate_hz"]), 1), "step_ms": round(float(p["step_ms"]), 2),
            "slow_loop": bool(p["slow_loop"]),
            "gait_phase": round(p["gait_phase"], 3),
            "stop": g["stop"], "reasons": g["reasons"], "clamps": g["clamp_counts"],
            "saturated_now": [k for k, n in g["saturated_now"].items() if n],
            "ramp": round(float(g["ramp"]), 3),
            "ticks": int(p["n"]), "late_ticks": int(self._slip_count - p["slip0"]),
            "ws_blocked_ticks": int(p["ws_blocked_total"]),
            "target_clip_ticks": int(p["saturated"]),
            "winding_c": [round(float(x), 1) for x in p["thermal"].t_winding],
            "peak_winding_c": [round(float(x), 1) for x in p["thermal"].peak_w],
            "winding_names": list(JM.MODEL_ACTUATORS),
            "thermal_uncalibrated": p["thermal_uncalibrated"],
            "jointmap_verified": p["jm_verified"], "no_imu": p["no_imu"],
            "telemetry_age_ms": round(p["tel_age"] * 1e3, 1),
            "imu_age_ms": round(p["imu_age"] * 1e3, 1),
            "deadman_age_s": round(self._deadman_age(), 2),
            "exit_reason": p["exit_reason"], "reached_run": p["reached_run"],
        }
        return out

    def _deadman_age(self):
        return max(0.0, time.monotonic() - self._pol_deadman)   # lock-free: see policy_keepalive

    def _telemetry_age(self):
        """Seconds since the OLDEST motor last said anything.

        Not the newest frame: a bus-wide 'did anything arrive' would miss the case that actually
        matters, which is ONE drive going quiet while its five neighbours keep talking."""
        if len(self._rx_at) < paths.N_MOTORS:
            return 1e9
        return max(0.0, self._tick_mono - min(self._rx_at.values()))

    def _start_measure(self, meas, now):
        meas["t0"] = now
        meas["running"] = True
        meas["done"] = False
        meas["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        for k in ("buf_t", "buf_cmd", "buf_posn", "buf_posr", "buf_spd", "buf_cur"):
            meas[k] = []
        self._meas = meas
        self._last_reject = ""
        self._set_mode("MEASURE", "measurement excitation started", leg=meas["leg"],
                       profile=meas["profile"], duration=meas["duration"], base=dict(meas["base"]))

    def _tick_measure(self, now, dt):
        meas = self._meas
        if meas is None:
            self._set_mode("LIMP", "measurement state vanished")
            return
        if not meas["running"]:
            self._stream_limp()                     # completed/stopped: hold limp until saved+cleared
            return
        ok, why = self._motion_allowed()
        if not ok:
            self._trip(why)
            return
        self._cmd_zero_epoch = self.calib.zero_epoch    # commanding absolute positions
        t = now - meas["t0"]
        if t >= meas["duration"]:
            meas["running"] = False
            meas["done"] = True
            self._stream_limp()
            return

        leg = meas["leg"]
        pose = self._measure_pose(meas, t)
        for n in paths.MOTOR_NAMES:                  # never-exceed hard clamp
            side, role = paths.split_name(n)
            lo, hi = self._hard_bounds(side, role)
            pose[n] = float(np.clip(pose[n], lo, hi))
        # live workspace net (the envelope was already validated at start; this catches drift)
        if not meas["override"]:
            limits = self.wstore.limits if self.wstore else None
            tgt_by_motor = {self.by_name[n]: pose[n] for n in paths.MOTOR_NAMES}
            ok, reason = pt.check_workspace(self.side_groups, tgt_by_motor, limits)
            if not ok:
                self._trip(f"measure workspace: {reason}")
                return
        elif self.fklut is not None and self.fklut.available:
            for side in paths.SIDES:
                ok, reason = self.fklut.feasible_check(side, pose[f"{side}.cam"],
                                                       pose[f"{side}.thigh"])
                if not ok:
                    self._trip(f"measure feasibility: {reason}")
                    return
        self._last_reject = ""

        cmd_row = [np.nan] * paths.N_MOTORS
        for i, n in enumerate(paths.MOTOR_NAMES):
            side, _ = paths.split_name(n)
            m = self.by_name[n]
            if side == leg or meas["hold_other"]:       # excited leg tracks; other leg holds or limps
                raw = self.calib.raw(n, pose[n])
                canio.set_pos(m.bus, m.cid, raw)
                self._last_cmd_raw[n] = raw
                cmd_row[i] = pose[n]
                if m.pos is not None and abs(m.pos - raw) > meas["max_track_err"]:
                    self._trip(f"{n} tracking error {m.pos - raw:+.1f} deg "
                               f"(> {meas['max_track_err']:.0f}) — hitting a stop?")
                    return
            else:
                canio.set_current(m.bus, m.cid, 0.0)
        self._measure_log(t, cmd_row)

    def _measure_log(self, t, cmd_row):
        """Append one high-rate (200 Hz) row: t + per-motor commanded/measured pos, speed, current."""
        meas = self._meas
        posn = [np.nan] * paths.N_MOTORS
        posr = [np.nan] * paths.N_MOTORS
        spd = [np.nan] * paths.N_MOTORS
        cur = [np.nan] * paths.N_MOTORS
        for i, n in enumerate(paths.MOTOR_NAMES):
            m = self.by_name[n]
            if m.pos is not None:
                posr[i] = m.pos
                posn[i] = self.calib.norm(n, m.pos)
                spd[i] = m.spd
                cur[i] = m.cur
        meas["buf_t"].append(t)
        meas["buf_cmd"].append(cmd_row)
        meas["buf_posn"].append(posn)
        meas["buf_posr"].append(posr)
        meas["buf_spd"].append(spd)
        meas["buf_cur"].append(cur)

    # ----------------------------------------------------------------- RECORDING
    def _handle_record_cmd(self, cmd, leg, kind, now):
        r = self._rec
        if cmd == "start_mode":
            ok, why = ((True, "") if self.calib.complete
                       else (False, "calibration incomplete — recordings must be normalized"))
            if not ok:
                self._last_reject = why
                return
            r["kind"] = kind
            self._set_mode("RECORD_GAIT" if kind == "gait" else "RECORD_WS",
                           f"recording mode: {kind}")
            self._pb = None
        elif cmd == "take_start" and not r["active"]:
            r.update(active=True, leg=leg, buf_t=[], buf_p=[], t0=now)
        elif cmd == "take_stop" and r["active"]:
            self._end_active_record()
        elif cmd == "undo" and not r["active"]:
            store = r["takes"] if r["kind"] == "gait" else r["segments"]
            if store[leg]:
                store[leg].pop()
        elif cmd == "center":
            r["centers"][leg] = [self.calib.norm(f"{leg}.{role}",
                                                 self.by_name[f"{leg}.{role}"].pos)
                                 for role in paths.ROLES]
        elif cmd == "reset":
            r["takes"] = {"right": [], "left": []}
            r["segments"] = {"right": [], "left": []}
            r["centers"] = {"right": None, "left": None}
            r["active"] = False

    def _end_active_record(self):
        r = self._rec
        if not r["active"]:
            return
        r["active"] = False
        leg = r["leg"]
        if r["kind"] == "gait" and len(r["buf_t"]) > 20:      # record_trajectory.py:158 min length
            r["takes"][leg].append((np.array(r["buf_t"]), np.array(r["buf_p"])))
        elif r["kind"] == "workspace" and len(r["buf_p"]) > 10:
            r["segments"][leg].append(np.array(r["buf_p"], float))
        r["buf_t"], r["buf_p"] = [], []

    def _tick_recording(self, now):
        r = self._rec
        leg = r["leg"]
        if r["active"] and leg:
            names = [f"{leg}.{role}" for role in paths.ROLES]
            if all(self.by_name[n].pos is not None for n in names):
                r["buf_t"].append(now - r["t0"])
                r["buf_p"].append([self.calib.norm(n, self.by_name[n].pos) for n in names])
        # live outside-workspace warning (record_trajectory.py:180-185)
        limits = self.wstore.limits if self.wstore else None
        r["outside"] = False
        if leg and limits is not None and limits.has_leg(leg):
            vals = [self.by_name[f"{leg}.{role}"].pos for role in paths.ROLES]
            if all(v is not None for v in vals):
                normed = [self.calib.norm(f"{leg}.{r_}", v) for r_, v in zip(paths.ROLES, vals)]
                ok, _ = limits.validate(leg, *normed)
                r["outside"] = not ok

    # ----------------------------------------------------------------- publish
    def _publish(self, now):
        raw = np.full(paths.N_MOTORS, np.nan)
        spd = np.full(paths.N_MOTORS, np.nan)
        cur = np.full(paths.N_MOTORS, np.nan)
        temp = np.full(paths.N_MOTORS, np.nan)
        err = np.zeros(paths.N_MOTORS)
        # commanded target alongside the measurement: NaN wherever nothing is being commanded (limp,
        # e-stopped, or a mode that does not stream SET_POS), so the chart simply has no target line
        # there rather than a stale one held flat. _last_cmd_raw is cleared on every mode change.
        cmd_raw = np.full(paths.N_MOTORS, np.nan)
        for i, n in enumerate(paths.MOTOR_NAMES):
            m = self.by_name[n]
            if m.pos is not None:
                raw[i], spd[i], cur[i], temp[i], err[i] = m.pos, m.spd, m.cur, m.temp, m.err
            c = self._last_cmd_raw.get(n)
            if c is not None:
                cmd_raw[i] = c
        norm = self.calib.norm_array(raw)
        cmd_norm = self.calib.norm_array(cmd_raw)
        self.ring.push(now, dict(pos_raw=raw, pos_norm=norm, spd=spd, cur=cur, temp=temp, err=err,
                                 cmd_raw=cmd_raw, cmd_norm=cmd_norm))
        self._bb_status = self.bb.status() if self.bb is not None else {"alive": False,
                                                                       "error": "not configured"}

        # ---- CAPTURE, not FORMAT ------------------------------------------------------------
        # Everything below reads live daemon state, so it has to happen on this thread -- but it
        # only READS. The rounding, the nesting and the six-motor comprehension that used to sit
        # here are 1.24 ms of pure Python per call, at 20 Hz, on the thread that owes a CAN frame
        # every 5 ms. They now happen in _format_snapshot(), on whichever Flask thread asked, and
        # only when one actually does.
        #
        # The capture holds PLAIN DATA ONLY: arrays this method already built, scalars, and dicts
        # that are copied here rather than referenced. That is the whole thread-safety argument --
        # a formatter running later on another thread can never observe a half-written _pb or
        # iterate a dict the control loop is still mutating.
        r = self._rec
        with self.lock:
            cap = dict(
                mode=self.mode, mock=self.mock, estop_reason=self.estop_reason,
                slip=self._slip_count, ticks=self._tick_count, loop_error=self.loop_error,
                last_reject=self._last_reject, blackbox=self._bb_status,
                can_errors=canio.send_errors(), can_bus=canio.send_stats(),
                bypass=dict(self.bypass),
                # these three already return plain dicts of scalars, and only when their mode has
                # ever run -- so they cost nothing in the common case and are safe to hand over
                thermal=self._thermal_pub(now),
                identify=self._identify_pub(now),
                policy=self._policy_pub(now),
                raw=raw, norm=norm, spd=spd, cur=cur, temp=temp, err=err,
                alive=[self.by_name[n].pos is not None for n in paths.MOTOR_NAMES],
                guard_refused=self._guard_latched,
                guard_raw_now=self._guard_detail.get("raw_now"),
                guard_raw_at_zero=self._guard_detail.get("raw_at_last_zero"),
                guard_compare=self._guard_detail.get("compare"),
                origin_jumps=len(self._raw_jumps), holding=now < self._hold_until,
                manual_targets=dict(self._manual_targets), override=self._manual_override,
                slew_dps=self._slew_dps, homing=self._home_active, homing_kind=self._home_kind,
                # copied, not referenced: the sine dicts are mutated by sine_update()
                sine={n: dict(v) for n, v in self._sine.items()},
                pb=None if self._pb is None else {
                    k: self._pb.get(k) for k in
                    ("phase", "period", "mode", "sides", "current_limit", "max_track_err",
                     "track_err_estop", "track_err_peak", "track_err_worst", "track_err_over")},
                meas=None if self._meas is None else dict(
                    running=self._meas["running"], done=self._meas.get("done", False),
                    leg=self._meas["leg"], profile=self._meas["profile"],
                    duration=self._meas["duration"], t0=self._meas.get("t0", now),
                    n_samples=len(self._meas["buf_t"]), now=now),
                rec=dict(kind=r["kind"], active=r["active"], leg=r["leg"],
                         outside=r["outside"], n_samples=len(r["buf_p"]),
                         takes={k: len(v) for k, v in r["takes"].items()},
                         segments={k: len(v) for k, v in r["segments"].items()},
                         centers={k: (None if v is None else list(v))
                                  for k, v in r["centers"].items()}),
            )
            self._cap = cap
            self._cap_ver += 1

    @staticmethod
    def _format_snapshot(cap):
        """Capture -> the JSON-shaped dict the UI reads. PURE: no daemon state, no locks, no live
        objects, so it is safe on any thread and cheap to test."""
        raw, norm = cap["raw"], cap["norm"]
        spd, cur, temp, err = cap["spd"], cap["cur"], cap["temp"], cap["err"]
        meas, pb = cap["meas"], cap["pb"]
        return dict(
            daemon_alive=True, mode=cap["mode"], mock=cap["mock"],
            estop=dict(latched=cap["mode"] == "ESTOPPED", reason=cap["estop_reason"]),
            # ticks is here so the ACTUAL loop rate is observable without the recorder.
            # The 2026-08-10 lesson twice over: nothing measured the tick period, so a
            # loop running at a third of its rate looked exactly like a healthy one.
            loop=dict(hz=TICK_HZ, slip=cap["slip"], ticks=cap["ticks"]),
            thermal=cap["thermal"], identify=cap["identify"], policy=cap["policy"],
            bypass=cap["bypass"], loop_error=cap["loop_error"],
            can_errors=cap["can_errors"], can_bus=cap["can_bus"],
            last_reject=cap["last_reject"],
            # the recorder's own health, so a dead writer thread is visible in the UI rather
            # than being discovered after the next incident
            blackbox=cap["blackbox"],
            # the pre-move guard: why the last activation was refused, with both raw poses
            premove=dict(refused=cap["guard_refused"], raw_now=cap["guard_raw_now"],
                         raw_at_last_zero=cap["guard_raw_at_zero"],
                         compare=cap["guard_compare"], origin_jumps=cap["origin_jumps"],
                         holding=cap["holding"]),
            motors={n: dict(alive=cap["alive"][i],
                            pos_raw=None if np.isnan(raw[i]) else round(float(raw[i]), 2),
                            pos_norm=None if np.isnan(norm[i]) else round(float(norm[i]), 2),
                            spd=None if np.isnan(spd[i]) else round(float(spd[i]), 0),
                            cur=None if np.isnan(cur[i]) else round(float(cur[i]), 2),
                            temp=None if np.isnan(temp[i]) else int(temp[i]),
                            err=int(err[i]))
                    for i, n in enumerate(paths.MOTOR_NAMES)},
            manual=dict(targets=cap["manual_targets"], override=cap["override"],
                        slew_dps=cap["slew_dps"],
                        sine={n: {k: v for k, v in sv.items() if not k.startswith("_")}
                              for n, sv in cap["sine"].items()},
                        homing=cap["homing"], homing_kind=cap["homing_kind"]),
            playback=(None if pb is None else dict(
                running=cap["mode"] == "PLAYBACK", phase=round(pb.get("phase") or 0.0, 3),
                period=pb["period"], mode=pb["mode"], sides=pb["sides"],
                current_limit=pb["current_limit"], max_track_err=pb["max_track_err"],
                track_err_estop=pb["track_err_estop"],
                track_err_peak=round(pb.get("track_err_peak") or 0.0, 1),
                track_err_worst=pb.get("track_err_worst"),
                track_err_over=pb.get("track_err_over") or False)),
            measure=(None if meas is None else dict(
                running=meas["running"], done=meas["done"], leg=meas["leg"],
                profile=meas["profile"], duration=meas["duration"],
                elapsed=round(float(np.clip(meas["now"] - meas["t0"],
                                            0.0, meas["duration"])), 2),
                n_samples=meas["n_samples"])),
            recording=dict(kind=cap["rec"]["kind"], active=cap["rec"]["active"],
                           leg=cap["rec"]["leg"], outside_workspace=cap["rec"]["outside"],
                           n_samples=cap["rec"]["n_samples"],
                           takes={s: cap["rec"]["takes"][s] for s in ("right", "left")},
                           segments={s: cap["rec"]["segments"][s] for s in ("right", "left")},
                           centers={s: (None if cap["rec"]["centers"][s] is None
                                        else [round(v, 1) for v in cap["rec"]["centers"][s]])
                                    for s in ("right", "left")}),
        )

    def _thermal_pub(self, now):
        """Live burst state for the panel. None when no burst has been run this session."""
        t = self._therm
        if t is None:
            return None
        el = now - t["t0"]
        d = {"motor": t["motor"], "running": bool(t["running"]),
             "elapsed_s": round(el, 2),
             "duration_s": t["env"].duration_s,
             "amps": t["env"].amps,
             "since_end_s": round(now - t["t_end"], 1) if t["t_end"] else None,
             "drive_t_start_c": t["t_start"], "drive_t_peak_c": t["t_peak"],
             "abort": t["abort"], "reversals": t["ex"].n_reversals,
             "i_rms": round(t["ex"].summary()["i_rms"], 2),
             "travel_deg": t["ex"].summary()["travel_deg"],
             "peak_erpm": t["ex"].spd_peak,
             "free_rotor": t["env"].free_rotor}
        if t["env"].free_rotor:
            d.update(freq_hz=t["env"].freq_hz, sine_amp_deg=t["env"].sine_amp)
        return d

    def get_snapshot(self):
        """The UI-shaped state. Built HERE, on the calling thread, from the control loop's latest
        capture -- and memoised on the capture version, so the many _ok() envelopes inside one
        request format it once and a quiet robot formats it never."""
        with self.lock:
            cap, ver = self._cap, self._cap_ver
            if ver == self._snap_ver and self.snapshot is not None:
                return dict(self.snapshot)
        if cap is None:
            # Before the first capture, hand back the seed -- NOT a freshly-read self.mode. A
            # snapshot that reports a live mode it has no data for lets a caller believe the loop
            # has published a state it never published; the identify tests catch exactly that.
            with self.lock:
                return dict(self.snapshot)
        built = self._format_snapshot(cap)      # outside the lock: this is the expensive part
        with self.lock:
            # last writer wins, and both wrote the same thing unless a newer capture landed
            # mid-format, in which case the next call rebuilds
            self.snapshot, self._snap_ver = built, ver
        return dict(built)
