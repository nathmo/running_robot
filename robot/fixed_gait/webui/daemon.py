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
    ESTOPPED        latched; zero current streamed until cleared
HTTP handlers never touch the buses — they only post requests into this object and read snapshots.
"""
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

MODES = ("LIMP", "MANUAL", "RECORD_GAIT", "RECORD_WS", "PLAYBACK", "MEASURE", "ESTOPPED")
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
MEASURE_F_MAX = 3.0                     # ceiling from _merge_measure_spec's clamp
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
        self._tick_count = 0
        self._slip_count = 0
        self.ring = ringbuffer.TelemetryRing()
        self.snapshot = {"daemon_alive": False, "mode": "LIMP"}
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
        limits = self.wstore.limits if (self.wstore and not override) else None
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
            m["f0"][r] = float(np.clip(m["f0"][r], 0.0, 3.0))
            m["f1"][r] = float(np.clip(m["f1"][r], 0.0, 3.0))
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
            with self.lock:
                snap = dict(self.snapshot)
                snap["daemon_alive"] = False
                snap["loop_error"] = self.loop_error
                self.snapshot = snap

    def _setup(self):
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
            pt.drain(self.buses, self.motors_by_bus, 0.0)
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

            # 1) e-stop first — latch (limp is streamed by the mode body below)
            if self.estop_event.is_set() and self.mode != "ESTOPPED":
                self._set_mode("ESTOPPED", self.estop_reason or "e-stop latched")
                self._pb = None
                if self._meas:
                    self._meas["running"] = False       # keep the partial log; stop exciting
                with self.lock:
                    self._manual_targets = {}
                    self._manual_override = False

            # 2) always drain feedback (telemetry works even limp)
            pt.drain(self.buses, self.motors_by_bus, dt)

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
        self._bb_dump(f"mode_{old}_to_{mode}", why=reason)

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

    def _stream_limp(self):
        for m in self.motors:
            canio.set_current(m.bus, m.cid, 0.0)
        self._last_cmd_raw.clear()

    def _motion_allowed(self):
        if not self.calib.complete:
            return False, "calibration incomplete — finish the zero/direction wizard first"
        silent = [self._name_of[id(m)] for m in self.motors if m.pos is None]
        if silent:
            return False, f"motor(s) silent: {', '.join(silent)} — never commanding blind"
        return True, ""

    # ================================================================= black box
    def _bb_event(self, kind, **fields):
        bb = self.bb
        if bb is not None:
            bb.log_event(kind, **fields)

    def _bb_dump(self, reason, **fields):
        bb = self.bb
        return bb.trigger_dump(reason, **fields) if bb is not None else None

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
            self._bb_event("premove.refused", reason=why, mode=self.mode, **detail)
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

        if req_mode and req_mode != self.mode:
            if req_mode == "LIMP":
                self._pb = None
                self._meas = None                          # finished/aborted run is cleared here
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

        # (c) homing is a slow guided slew and should track well — a stale zero shows up as
        # tracking error within the first few degrees, long before the 25 deg playback threshold.
        if homing and not self._was_homing:              # each new guided move: fresh budget
            self._travel, self._travel_prev = {}, {}
        self._was_homing = homing
        track_lim = MAX_TRACK_ERR_HOMING_DEG if homing else MAX_TRACK_ERR_DEG
        self._cmd_zero_epoch = self.calib.zero_epoch     # we are commanding ABSOLUTE positions now
        for n, m in self.by_name.items():
            raw = self.calib.raw(n, targets_norm[n])
            canio.set_pos(m.bus, m.cid, raw)
            self._last_cmd_raw[n] = raw
            if abs(m.pos - raw) > track_lim:
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
        lim = ramp * pb["current_limit"]
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
                if pb["speed_limit"] > 0 and curr * m.spd > 0 and abs(m.spd) > pb["speed_limit"]:
                    band = 0.3 * pb["speed_limit"]
                    curr *= float(np.clip((pb["speed_limit"] + band - abs(m.spd)) / band, 0.0, 1.0))
                canio.set_current(m.bus, m.cid, curr)
                if abs(m.spd) > pb["max_speed"]:
                    self._trip(f"{n} runaway {m.spd:.0f} ERPM (> {pb['max_speed']:.0f})")
                    return

    # ----------------------------------------------------------------- MEASURE (system-ID)
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

        r = self._rec
        with self.lock:
            sine_pub = {n: {k: v for k, v in s.items() if not k.startswith("_")}
                        for n, s in self._sine.items()}
            override = self._manual_override
            manual_targets = dict(self._manual_targets)
            self.snapshot = dict(
                daemon_alive=True, mode=self.mode, mock=self.mock,
                estop=dict(latched=self.mode == "ESTOPPED", reason=self.estop_reason),
                loop=dict(hz=TICK_HZ, slip=self._slip_count),
                loop_error=self.loop_error,
                last_reject=self._last_reject,
                # the recorder's own health, so a dead writer thread is visible in the UI rather
                # than being discovered after the next incident
                blackbox=self._bb_status,
                # the pre-move guard: why the last activation was refused, with both raw poses
                premove=dict(refused=self._guard_latched,
                             raw_now=self._guard_detail.get("raw_now"),
                             raw_at_last_zero=self._guard_detail.get("raw_at_last_zero"),
                             compare=self._guard_detail.get("compare"),
                             origin_jumps=len(self._raw_jumps),
                             holding=now < self._hold_until),
                motors={n: dict(alive=self.by_name[n].pos is not None,
                                pos_raw=None if np.isnan(raw[i]) else round(float(raw[i]), 2),
                                pos_norm=None if np.isnan(norm[i]) else round(float(norm[i]), 2),
                                spd=None if np.isnan(spd[i]) else round(float(spd[i]), 0),
                                cur=None if np.isnan(cur[i]) else round(float(cur[i]), 2),
                                temp=None if np.isnan(temp[i]) else int(temp[i]),
                                err=int(err[i]))
                        for i, n in enumerate(paths.MOTOR_NAMES)},
                manual=dict(targets=manual_targets, override=override, slew_dps=self._slew_dps,
                            sine=sine_pub, homing=self._home_active,
                            homing_kind=self._home_kind),
                playback=(None if self._pb is None else dict(
                    running=self.mode == "PLAYBACK", phase=round(self._pb.get("phase", 0.0), 3),
                    period=self._pb["period"], mode=self._pb["mode"], sides=self._pb["sides"],
                    current_limit=self._pb["current_limit"],
                    max_track_err=self._pb["max_track_err"],
                    track_err_estop=self._pb["track_err_estop"],
                    track_err_peak=round(self._pb.get("track_err_peak", 0.0), 1),
                    track_err_worst=self._pb.get("track_err_worst"),
                    track_err_over=self._pb.get("track_err_over", False))),
                measure=(None if self._meas is None else dict(
                    running=self._meas["running"], done=self._meas.get("done", False),
                    leg=self._meas["leg"], profile=self._meas["profile"],
                    duration=self._meas["duration"],
                    elapsed=round(float(np.clip(now - self._meas.get("t0", now),
                                                0.0, self._meas["duration"])), 2),
                    n_samples=len(self._meas["buf_t"]))),
                recording=dict(kind=r["kind"], active=r["active"], leg=r["leg"],
                               outside_workspace=r["outside"],
                               n_samples=len(r["buf_p"]),
                               takes={s: len(r["takes"][s]) for s in ("right", "left")},
                               segments={s: len(r["segments"][s]) for s in ("right", "left")},
                               centers={s: (None if r["centers"][s] is None
                                            else [round(v, 1) for v in r["centers"][s]])
                                        for s in ("right", "left")}),
            )

    def get_snapshot(self):
        with self.lock:
            return dict(self.snapshot)
