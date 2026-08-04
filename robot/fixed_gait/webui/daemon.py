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
import canio
import ringbuffer

import play_trajectory as pt              # fixed_gait/ — Motor, drain, reconstruct-side helpers
import trajectory as traj                 # fixed_gait/

MODES = ("LIMP", "MANUAL", "RECORD_GAIT", "RECORD_WS", "PLAYBACK", "MEASURE", "ESTOPPED")
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

# Playback cycle period, seconds. The floor is a real limit, not a slider preference: phase is
# play_t / period, so a 0 sails straight into a ZeroDivisionError in _tick_playback, and the PATCH
# endpoint takes whatever number it is handed. 0.2 s = 5 Hz is the fastest the UI offers; at the
# daemon's 200 Hz that is still 40 control samples per cycle. Going faster is not blocked by this
# clamp being generous -- it is blocked by max_speed / max_track_err tripping first, which is the
# guard that should be doing the work.
PERIOD_MIN = 0.2
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


class RobotDaemon(threading.Thread):
    def __init__(self, interface="socketcan", mock=False, calib=None, wstore=None, fklut=None):
        super().__init__(daemon=True, name="RobotDaemon")
        self.interface = interface
        self.mock = mock
        self.calib = calib
        self.wstore = wstore
        self.fklut = fklut

        self.lock = threading.Lock()
        self.estop_event = threading.Event()
        self.stop_event = threading.Event()

        # -------- requests (web -> daemon, guarded by self.lock) --------
        self._req_mode = None
        self._req_clear_estop = False
        self._manual_targets = {}          # name -> desired normalized deg
        self._manual_override = False
        self._slew_dps = DEFAULT_SLEW_DPS
        self._home_active = False           # slow return-to-zero engaged (feasibility-net checked)
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

    def home(self, slew_dps=None):
        """Slowly drive every joint back to the URDF zero pose (normalized 0 = the stance we
        manually zero to). Trusts the CAD zero: it slews under the physical-feasibility net (like
        override) rather than the eroded gait polygon, so it can still reach 0 when 0 sits a
        degree or so outside the hand-drawn safe region."""
        ok, why = self._motion_allowed()
        if not ok:
            with self.lock:
                self._last_reject = why
            return False, why
        with self.lock:
            self._manual_targets = {n: 0.0 for n in paths.MOTOR_NAMES}
            self._home_active = True
            self._home_slew = float(np.clip(slew_dps if slew_dps else 20.0, 5.0, 120.0))
            for s in self._sine.values():
                s["enabled"] = False
            self._req_mode = "MANUAL"
        return True, ""

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
        ok, why = self._motion_allowed()
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
        finally:
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
        if silent:
            print(f"!! preflight: no status from {silent} — motion blocked until they report")

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
        while not self.stop_event.is_set():
            now = time.time()

            # 1) e-stop first — latch (limp is streamed by the mode body below)
            if self.estop_event.is_set() and self.mode != "ESTOPPED":
                self.mode = "ESTOPPED"
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

            # 6) telemetry + snapshot at 20 Hz
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

    def _trip(self, reason):
        if not self.estop_event.is_set():
            with self.lock:
                self.estop_reason = reason
            self.estop_event.set()
            print(f"!! SAFETY STOP: {reason}")

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
                self.estop_reason = ""
            self.mode = "LIMP"

        if self.mode == "ESTOPPED":
            return                                          # latched: ignore everything else

        for cmd, leg, kind in rec_cmds:
            self._handle_record_cmd(cmd, leg, kind, now)

        if pb_req is not None:
            ok, why = self._motion_allowed()
            if ok:
                self._start_playback(pb_req, now)
            else:
                self._last_reject = why
        if pb_patch is not None and self._pb is not None:
            self._apply_playback_patch(pb_patch, now)

        if meas_stop and self._meas is not None:
            self._meas["running"] = False               # user-ended; keep the log for saving
        if meas_req is not None:
            ok, why = self._motion_allowed()
            if ok:
                self._start_measure(meas_req, now)
            else:
                self._last_reject = why

        if req_mode and req_mode != self.mode:
            if req_mode == "LIMP":
                self.mode = "LIMP"
                self._pb = None
                self._meas = None                          # finished/aborted run is cleared here
                self._end_active_record()
                with self.lock:
                    self._manual_targets = {}
                    self._manual_override = False          # override never survives a mode change
            elif req_mode == "MANUAL":
                ok, why = self._motion_allowed()
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
                    self.mode = "MANUAL"
                else:
                    self._last_reject = why

    # ----------------------------------------------------------------- MANUAL (hold + sine)
    def _tick_manual(self, now, dt):
        ok, why = self._motion_allowed()
        if not ok:
            self._trip(why)
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
        elif self.fklut is not None and self.fklut.available:
            for side in paths.SIDES:                       # physically-assemblable band net
                ok, reason = self.fklut.feasible_check(side, targets_norm[f"{side}.cam"],
                                                       targets_norm[f"{side}.thigh"])
                if not ok:
                    self._last_reject = reason
                    self._held = held_before
                    return
        self._last_reject = ""

        for n, m in self.by_name.items():
            raw = self.calib.raw(n, targets_norm[n])
            canio.set_pos(m.bus, m.cid, raw)
            self._last_cmd_raw[n] = raw
            if abs(m.pos - raw) > MAX_TRACK_ERR_DEG:
                self._trip(f"{n} tracking error {m.pos - raw:+.1f} deg (> {MAX_TRACK_ERR_DEG})")
                return

    # ----------------------------------------------------------------- PLAYBACK
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
        self.mode = "PLAYBACK"

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
            self.mode = "LIMP"
            return
        ok, why = self._motion_allowed()
        if not ok:
            self._trip(why)
            return
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
        self.mode = "MEASURE"

    def _tick_measure(self, now, dt):
        meas = self._meas
        if meas is None:
            self.mode = "LIMP"
            return
        if not meas["running"]:
            self._stream_limp()                     # completed/stopped: hold limp until saved+cleared
            return
        ok, why = self._motion_allowed()
        if not ok:
            self._trip(why)
            return
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
            self.mode = "RECORD_GAIT" if kind == "gait" else "RECORD_WS"
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
                motors={n: dict(alive=self.by_name[n].pos is not None,
                                pos_raw=None if np.isnan(raw[i]) else round(float(raw[i]), 2),
                                pos_norm=None if np.isnan(norm[i]) else round(float(norm[i]), 2),
                                spd=None if np.isnan(spd[i]) else round(float(spd[i]), 0),
                                cur=None if np.isnan(cur[i]) else round(float(cur[i]), 2),
                                temp=None if np.isnan(temp[i]) else int(temp[i]),
                                err=int(err[i]))
                        for i, n in enumerate(paths.MOTOR_NAMES)},
                manual=dict(targets=manual_targets, override=override, slew_dps=self._slew_dps,
                            sine=sine_pub, homing=self._home_active),
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
