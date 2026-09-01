"""The layer between a learned policy and 144 N*m of joint torque.

DESIGN: CAP THE COMMAND, KILL THE RUN, AND DO NOT CONFUSE THE TWO
------------------------------------------------------------------
A policy asks for something unsafe in two very different ways, and they need different answers.

  MOMENTARY overshoot -- a target a few degrees past a joint limit on one tick, a torque spike on
  a hard footfall. Killing the run for this drops a standing robot on the floor, which is worse
  than the thing being prevented. These are CLAMPED, and the clamp is counted.

  PERSISTENT demand -- the policy pushing against the same limit tick after tick. That is not a
  transient, it is the control law asking for a machine it does not have, and continuing means
  running a policy whose actual output is no longer the output that was verified. These KILL, on
  a per-limit dwell timer (`persist_ticks`).

So every limit has both: clamp now, and kill if the clamp is still working N ticks later. That is
what "either kill the controller or cap it to the safe limit" means in a loop that runs every 5 ms.

THE LADDER, in the order it is applied
--------------------------------------
  0  SANITY      non-finite target, gain or measurement          -> immediate HARD kill
  1  POSITION    clip to the intersection of the MuJoCo ctrlrange, the mechanical hard bounds,
                 and the calibration-verified band
  2  RATE        slew-limit the target to the joint velocity limit
  2b OBSERVE     advance the winding-temperature observer on the current just measured. It is
                 stepped HERE rather than by the caller because it is the input to step 3 and to
                 the thermal kill below, and a governor holding an observer nobody remembered to
                 step has infinite thermal headroom and no thermal protection at all -- silently.
                 One owner, one call, and `current` means what its name says.
  3  TORQUE      cap the position ERROR so that kp*e stays inside the thermal + rating budget.
                 Capping the error rather than kp is deliberate: kp and kd are what the policy
                 chose and what the sim ran, so scaling them changes the closed-loop dynamics into
                 something untested, whereas bounding the error only saturates the command exactly
                 as a torque-limited actuator does -- which the sim ALSO did (forcerange).
  4  GAINS       clamp kp/kd into the drive's force-control wire ranges (0-500, 0-5)
  5  WATCHDOGS   the conditions that are never clamped, only killed

TWO STOP FLAVOURS
-----------------
  SOFT  freeze the last good target, ramp kp/kd to zero over `soft_stop_s`, then limp. Puts the
        robot down under control. For a released dead-man, a lost command link, a thermal trip.
  HARD  zero current immediately. For anything where continuing to command is itself the hazard:
        a drive fault, a non-finite command, a fall, loss of telemetry.

A fall is HARD on purpose. Once gravity is winning, holding a stance target is just driving the
legs into the floor.

This module is PURE: it takes numbers and returns numbers plus a verdict. No CAN, no threads, no
clock of its own (the caller supplies dt). That is what makes it testable without a robot, and it
is why the tests can assert on the kill conditions rather than on a mock.
"""
import numpy as np

STOP_NONE, STOP_SOFT, STOP_HARD = 0, 1, 2
_STOP_NAME = {STOP_NONE: "running", STOP_SOFT: "soft-stop", STOP_HARD: "HARD-STOP"}


class Limits:
    """Everything the governor is allowed to enforce. Values are per-joint unless noted."""

    def __init__(self, pos_lo, pos_hi, vel_max, tau_peak, tau_cont,
                 kp_max=500.0, kd_max=5.0, kp_min=0.0, kd_min=0.0,
                 tilt_kill=-0.5, gyro_max=12.0, track_err_max=0.35, temp_case_max=80.0,
                 persist_ticks=40, telemetry_stale_s=0.05, deadman_s=0.5,
                 soft_stop_s=0.30, jerk_max=None):
        self.pos_lo = np.asarray(pos_lo, float)
        self.pos_hi = np.asarray(pos_hi, float)
        self.vel_max = np.asarray(vel_max, float)
        self.tau_peak = np.asarray(tau_peak, float)
        self.tau_cont = np.asarray(tau_cont, float)
        self.kp_max, self.kd_max = float(kp_max), float(kd_max)
        self.kp_min, self.kd_min = float(kp_min), float(kd_min)
        # -grav_z below which the robot counts as fallen. The sim terminated an episode here
        # (term_gravity_z), so past this point the policy is outside every state it was trained
        # on, quite apart from being on its way to the floor.
        self.tilt_kill = float(tilt_kill)
        self.gyro_max = float(gyro_max)              # rad/s on any body axis
        self.track_err_max = float(track_err_max)    # rad between commanded and measured joint
        self.temp_case_max = float(temp_case_max)    # the DRIVE's own reported temperature
        # ticks a limit may stay saturated before it stops being a transient. 40 at 200 Hz = 0.2 s,
        # comfortably longer than a footfall spike and far shorter than the 0.31 s fall timescale.
        self.persist_ticks = int(persist_ticks)
        self.telemetry_stale_s = float(telemetry_stale_s)
        self.deadman_s = float(deadman_s)
        self.soft_stop_s = float(soft_stop_s)
        self.jerk_max = jerk_max

    @classmethod
    def from_bundle(cls, b, hard_lo=None, hard_hi=None, tau_cont=None, **kw):
        """Start from what the policy was TRAINED against, then narrow.

        The joint ranges and the torque limits come out of the model the policy ran in, so the
        default governor is exactly as permissive as the simulator was and no more. `hard_lo`/
        `hard_hi` narrow it further with the mechanism's real bounds, which is the direction that
        is always safe."""
        lo = np.asarray(b["ctrl_lo"], float).copy()
        hi = np.asarray(b["ctrl_hi"], float).copy()
        if hard_lo is not None:
            lo = np.maximum(lo, np.asarray(hard_lo, float))
        if hard_hi is not None:
            hi = np.minimum(hi, np.asarray(hard_hi, float))
        peak = np.asarray(b["forcerange"], float)
        cont = np.asarray(tau_cont, float) if tau_cont is not None else 0.35 * peak
        return cls(lo, hi, np.asarray(b["motor_vel_limit"], float), peak, cont,
                   tilt_kill=float(b.term_gravity_z), **kw)


class Verdict:
    __slots__ = ("target", "kp", "kd", "stop", "reasons", "clamped", "limp", "ramp")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def ok(self):
        return self.stop == STOP_NONE

    def __repr__(self):
        return "<Verdict {} clamped={} reasons={}>".format(
            _STOP_NAME[self.stop], sorted(self.clamped), self.reasons)


class SafetyGovernor:
    def __init__(self, limits, dt, thermal=None, n=6, names=None):
        self.L = limits
        self.dt = float(dt)
        self.thermal = thermal
        self.n = int(n)
        self.names = list(names) if names else ["j{}".format(i) for i in range(self.n)]
        self.reset()

    def reset(self):
        self.stop = STOP_NONE
        self.reasons = []
        self._sat = {}                    # limit name -> consecutive saturated ticks
        self._last_good = None            # target frozen at the moment of a soft stop
        self._ramp = 1.0                  # kp/kd multiplier during a soft stop
        self._t_since_rx = 0.0
        self._t_since_deadman = 0.0
        self.n_clamped = {}
        self.n_ticks = 0

    # ------------------------------------------------------------------ kills
    def kill(self, reason, hard=True):
        """Latch a stop. Never un-latches: clearing one is an operator action, via reset()."""
        level = STOP_HARD if hard else STOP_SOFT
        if level > self.stop:
            self.stop = level
            if hard:
                self._ramp = 0.0
        if reason not in self.reasons:
            self.reasons.append(reason)
        return self.stop

    def _saturate(self, name, active, detail):
        """Count consecutive ticks a limit is doing work, and kill once it stops being a blip."""
        if not active:
            self._sat[name] = 0
            return False
        k = self._sat.get(name, 0) + 1
        self._sat[name] = k
        self.n_clamped[name] = self.n_clamped.get(name, 0) + 1
        if k >= self.L.persist_ticks:
            self.kill("{} clamped for {} consecutive ticks ({:.0f} ms): {}".format(
                name, k, k * self.dt * 1e3, detail), hard=False)
            return True
        return False

    # ------------------------------------------------------------------ the tick
    def observe(self, current, omega=None, drive_temp=None, t_amb=None):
        """Advance the winding observer one tick and return its estimate (None if there is none).

        Public because the approach phase of a run wants the observer tracking before there is any
        verdict to ask for -- the drive's reported case temperature is what corrects the estimate,
        and that correction should not wait for the policy to start. `step` calls this itself, so
        a caller that uses `step` must NOT also call this."""
        if self.thermal is None or current is None:
            return None
        return self.thermal.step(np.abs(np.asarray(current, float)), omega=omega,
                                 t_reported=drive_temp, t_amb=t_amb)

    def step(self, target, kp, kd, motor_pos, motor_vel, grav, gyro,
             telemetry_age=0.0, deadman_age=0.0, drive_temp=None, drive_err=None,
             current=None, t_amb=None):
        """Filter one controller output. Returns a Verdict whose target/kp/kd are what to send.

        telemetry_age : seconds since the newest status frame from ANY motor
        deadman_age   : seconds since the operator last refreshed the run permission
        drive_temp    : per-motor reported temperature, degC (the drive's own, not the winding)
        drive_err     : per-motor error code from the status frame; nonzero is a fault
        current       : per-motor current, A, in the same joint order as everything else. Supplying
                        it steps the winding observer (see `observe`); it used to be accepted here
                        and silently ignored while the caller was expected to step the model
                        itself, which is a parameter that lies and a guard that can go missing.
        t_amb         : ambient air temperature, degC, if something is measuring it
        """
        self.n_ticks += 1
        L = self.L
        target = np.asarray(target, float).copy()
        kp = np.asarray(kp, float).copy()
        kd = np.asarray(kd, float).copy()
        pos = np.asarray(motor_pos, float)
        vel = np.asarray(motor_vel, float)
        clamped = set()

        # -- 0. SANITY. A NaN reaching the wire encodes to a bit pattern that means something, and
        # whatever it means, nobody chose it. This one can never be a clamp.
        for label, arr in (("target", target), ("kp", kp), ("kd", kd),
                           ("motor_pos", pos), ("motor_vel", vel),
                           ("gravity", np.asarray(grav, float)), ("gyro", np.asarray(gyro, float))):
            if not np.all(np.isfinite(arr)):
                self.kill("non-finite {}: {}".format(label, np.asarray(arr).tolist()), hard=True)
                return self._verdict(target, kp, kd, clamped)

        # -- 5a. WATCHDOGS that must fire before anything is commanded ---------------------------
        if telemetry_age > L.telemetry_stale_s:
            self.kill("telemetry stale by {:.0f} ms -- never command a joint we cannot see"
                      .format(telemetry_age * 1e3), hard=True)
        if drive_err is not None:
            bad = [self.names[i] for i, e in enumerate(np.asarray(drive_err).ravel()) if e]
            if bad:
                self.kill("drive error flag on {}".format(", ".join(bad)), hard=True)
        if drive_temp is not None:
            t = np.asarray(drive_temp, float)
            hot = [self.names[i] for i in np.flatnonzero(np.isfinite(t) & (t >= L.temp_case_max))]
            if hot:
                self.kill("drive case temperature at/over {:.0f} C on {}".format(
                    L.temp_case_max, ", ".join(hot)), hard=True)
        gz = float(np.asarray(grav, float)[2])
        if gz > L.tilt_kill:
            # grav is world-DOWN in body axes, so upright is -1 and 0 is on its side
            self.kill("fallen: gravity_z {:+.2f} above the {:+.2f} termination the policy was "
                      "trained to (tilt {:.0f} deg)".format(
                          gz, L.tilt_kill, np.degrees(np.arccos(min(1.0, -gz)))), hard=True)
        gmax = float(np.max(np.abs(np.asarray(gyro, float))))
        if gmax > L.gyro_max:
            self.kill("body rate {:.1f} rad/s over the {:.1f} limit".format(gmax, L.gyro_max),
                      hard=True)
        if deadman_age > L.deadman_s:
            self.kill("dead-man not refreshed for {:.2f} s".format(deadman_age), hard=False)

        # -- 2b. OBSERVE. Before the torque budget, which reads the estimate this produces, and
        # before the thermal kill at the bottom. `vel` is the joint speed the iron-loss term wants
        # and we already have it, so it is not a separate argument.
        self.observe(current, omega=np.abs(vel), drive_temp=drive_temp, t_amb=t_amb)

        if self.stop == STOP_HARD:
            return self._verdict(target, kp, kd, clamped)

        # -- 1. POSITION ---------------------------------------------------------------------
        lim = np.clip(target, L.pos_lo, L.pos_hi)
        if np.any(lim != target):
            i = int(np.argmax(np.abs(lim - target)))
            clamped.add("position")
            self._saturate("position", True, "{} wanted {:+.3f} rad, band [{:+.3f}, {:+.3f}]"
                           .format(self.names[i], target[i], L.pos_lo[i], L.pos_hi[i]))
        else:
            self._saturate("position", False, "")
        target = lim

        # -- 2. RATE -------------------------------------------------------------------------
        if self._last_good is not None:
            dmax = L.vel_max * self.dt
            lim = np.clip(target, self._last_good - dmax, self._last_good + dmax)
            if np.any(lim != target):
                i = int(np.argmax(np.abs(lim - target)))
                clamped.add("rate")
                self._saturate("rate", True, "{} wanted {:.1f} rad/s, limit {:.1f}".format(
                    self.names[i], (target[i] - self._last_good[i]) / self.dt, L.vel_max[i]))
            else:
                self._saturate("rate", False, "")
            target = lim

        # -- 3. TORQUE via the position error -------------------------------------------------
        # The drive computes tau = kp*(p_des - p) - kd*v. We cannot see that torque before it
        # happens, but we know every term, so bound the one we command: |p_des - p| is capped so
        # that the PROPORTIONAL contribution stays inside the budget. The kd term is left alone --
        # it is dissipative, it opposes motion, and clamping damping is how a system oscillates.
        tau_max = self._torque_budget()
        err = target - pos
        err_max = tau_max / np.maximum(kp, 1e-6)
        lim_err = np.clip(err, -err_max, err_max)
        if np.any(lim_err != err):
            i = int(np.argmax(np.abs(lim_err - err)))
            clamped.add("torque")
            self._saturate("torque", True,
                           "{} demanded {:.0f} N*m (kp {:.0f} x {:.3f} rad), budget {:.0f}".format(
                               self.names[i], kp[i] * err[i], kp[i], err[i], tau_max[i]))
        else:
            self._saturate("torque", False, "")
        target = pos + lim_err

        # -- 4. GAINS into the wire ranges ----------------------------------------------------
        kpc = np.clip(kp, L.kp_min, L.kp_max)
        kdc = np.clip(kd, L.kd_min, L.kd_max)
        if np.any(kpc != kp) or np.any(kdc != kd):
            clamped.add("gains")
            # NOT a persistence kill: for the deployed bundle the impedance channel spans exactly
            # kp 40-500 and kd 1.0-5.0, so a clamp here means the bundle changed, not that the
            # policy is fighting the machine. It is recorded and surfaced, loudly, and the run
            # continues on a command that is still inside the drive's own range.
            self.n_clamped["gains"] = self.n_clamped.get("gains", 0) + 1
        kp, kd = kpc, kdc

        # -- 5b. TRACKING: the joint is not where it was told to be ----------------------------
        # After the torque cap this is the honest test of "is the machine doing what we asked".
        terr = np.abs(pos - target)
        if np.any(terr > L.track_err_max):
            i = int(np.argmax(terr))
            self._saturate("tracking", True, "{} is {:.3f} rad from its target (limit {:.3f})"
                           .format(self.names[i], terr[i], L.track_err_max))
        else:
            self._saturate("tracking", False, "")

        # -- thermal trip (the observer is stepped by the caller; we read its verdict) ----------
        if self.thermal is not None:
            over = np.flatnonzero(self.thermal.t_winding >= self.thermal.t_trip)
            if over.size:
                self.kill("estimated winding temperature {:.0f} C at/over the {:.0f} C limit on {}"
                          .format(float(self.thermal.t_winding[over[0]]),
                                  float(self.thermal.t_trip[over[0]]),
                                  ", ".join(self.names[i] for i in over)), hard=False)

        self._last_good = target.copy()
        return self._verdict(target, kp, kd, clamped)

    # ------------------------------------------------------------------ helpers
    def _torque_budget(self):
        """Per-joint |torque| ceiling: the rated peak, derated by the thermal observer."""
        if self.thermal is None:
            return self.L.tau_peak
        return self.thermal.torque_budget(self.L.tau_peak, self.L.tau_cont)

    def _verdict(self, target, kp, kd, clamped):
        """Apply whatever stop is latched, and hand back what should actually go on the wire."""
        limp = False
        if self.stop == STOP_HARD:
            # zero gains AND zero target: a hard stop must not depend on the drive interpreting
            # a position it may or may not still be holding
            kp = np.zeros(self.n)
            kd = np.zeros(self.n)
            target = self._last_good if self._last_good is not None else np.asarray(target, float)
            self._ramp = 0.0
            limp = True
        elif self.stop == STOP_SOFT:
            # freeze the target where it was and bleed the gains out over soft_stop_s
            if self._last_good is not None:
                target = self._last_good.copy()
            self._ramp = max(0.0, self._ramp - self.dt / max(self.L.soft_stop_s, 1e-6))
            kp = kp * self._ramp
            kd = kd * self._ramp
            limp = self._ramp <= 0.0
        return Verdict(target=target, kp=kp, kd=kd, stop=self.stop, reasons=list(self.reasons),
                       clamped=clamped, limp=limp, ramp=self._ramp)

    def status(self):
        return {"stop": _STOP_NAME[self.stop], "reasons": list(self.reasons),
                "clamp_counts": dict(self.n_clamped), "saturated_now": dict(self._sat),
                "ramp": self._ramp, "ticks": self.n_ticks,
                "clamp_rate": {k: v / max(self.n_ticks, 1) for k, v in self.n_clamped.items()}}
