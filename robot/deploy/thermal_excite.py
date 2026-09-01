"""Torque-saturating excitation for a thermal burst: heat one motor hard, move it as little as
possible, and never leave the safe envelope.

THE LAW
-------
Bang-bang current with hysteresis on speed and position:

    I = sign * I_amp,  and sign flips when the joint reaches the speed limit OR the window edge

|I| is therefore ALWAYS I_amp -- the torque output is saturated 100% of the time, which is the
point: the run has to deposit a known, large amount of copper loss in a short window, and anything
that lets the current sag makes the deposited energy depend on the mechanics instead of on the
command.

RETRACTED 2026-08-29: THE FREE-JOINT DITHER DOES NOT WORK
---------------------------------------------------------
This module used to reverse on measured SPEED, on the argument that the current would flip as soon
as the joint was really moving, giving a small, fast dither. MEASURED on right.cam at 30 A, that
law self-excited and destroyed its own envelope in 0.21 s:

    t=0.017 s   730 ERPM     t=0.071 s   6130 ERPM
    t=0.127 s 15380 ERPM     t=0.207 s -25510 ERPM, 9.5 deg out, aborted

The mechanism is transport delay. The drive takes ~10 ms to reverse its current, so after every
velocity zero-crossing the winding is still pushing the OLD way for 2-3 control ticks. At the
~20 Hz the joint settles into, 10 ms is ~72 degrees of phase lag, which turns the intended damping
term negative: 11 of 31 ticks in the recording were pumping energy IN. Velocity-triggered
bang-bang with actuation delay is an oscillator, not a dither.

Nor does a fixed-frequency open-loop square wave rescue it, which was the obvious next idea. The
arithmetic is unforgiving on a joint with no load:

    30 A -> 35160 deg/s^2 -> crosses an 8 deg window in 21 ms -> needs 47 Hz to stay under 2 deg
     8 A ->  9376 deg/s^2 -> needs 24 Hz, but delivers 7% of the power (hours per burst)

and the same ~10 ms delay caps any renderable half-period near 20 ms, i.e. ~25 Hz. There is no
current that is both depositable and gentle, because with no load the energy has nowhere to go
except kinetic. A FREE joint cannot absorb a current-mode burst.

So the burst is now a BLOCKED-ROTOR experiment: the joint is mechanically clamped, the current is
unidirectional, and motion is an ABORT rather than a reversal -- if a clamped joint starts moving,
the clamp is slipping and the run must stop, not oscillate.

Net mechanical work per cycle is ~zero -- the energy pumped into the rotor accelerating is taken
back out decelerating -- so essentially all the electrical input becomes I^2 R, which is exactly
what the thermal model integrates. That is what makes this experiment as clean as a blocked rotor
while distributing the loss across all three phases instead of parking it in one.

FREE ROTOR: TRACK A SINE, FIGHT YOUR OWN INERTIA (2026-09-01)
-------------------------------------------------------------
A motor OFF the robot with nothing on the shaft gets a different law entirely. Unidirectional
current cannot heat it (back-EMF at no-load speed removes the voltage headroom and the current
collapses -- that was the previous free-rotor mode, and it existed only to self-diagnose). What
CAN draw current with no load attached is acceleration: the run streams a POSITION-mode sine, and
every half-cycle the drive has to decelerate and re-accelerate its own rotor inertia,
I = J*alpha/Kt. Net mechanical work per cycle is ~zero -- the kinetic energy pumped in
accelerating is taken back out decelerating -- so what the winding integrates is I^2 R, which is
exactly what the thermal model wants.

Why this is NOT the retracted dither: the dither was current-mode bang-bang reversing on measured
speed, and the ~10 ms actuation delay turned its damping term negative. The sine is
position-mode -- the drive's own position loop closes the fast loop, the same stable arrangement
as the joint-identify wiggle, and the trajectory is open-loop in time so nothing this module
computes can be phase-flipped by delay into an oscillator.

Two honest limitations, both self-diagnosing rather than assumed away:
  * The current is set by physics (J, Kt, the sine), not by the amps knob. The knob is the
    TARGET the safety gate was sized with; the run integrates the MEASURED current, and it stops
    early with the achieved number if the sine cannot draw a useful fraction of the target --
    raise the frequency or amplitude, or re-size the burst for the current it actually draws.
  * Position mode has no current cap this module controls, so the deposited-energy integral is
    watched instead: the moment measured I^2 dt exceeds what the winding-rise gate approved, the
    run aborts. The run can therefore never deposit more heat than it was cleared for, no matter
    what the drive's loop does.

Nothing may be attached to the shaft. A leg on the shaft turns the sine into a pendulum drive:
gravity adds a load torque the current budget was not sized for, and the moving limb is exactly
the hazard the blocked/free declaration exists to rule out.

SAFETY
------
This module is PURE: it takes state and returns a commanded current plus a verdict. It opens no
bus and owns no clock. Everything it can do is bounded by the envelope it is constructed with, and
`step()` returns an abort reason rather than a current whenever the envelope is violated -- so a
caller that ignores the reason still gets 0 A.
"""
import numpy as np

# Hard ceilings this module will not exceed regardless of what it is asked for. Raising any of
# them needs a hardware argument. The 62.5 A incident on 2026-08-26 is the reference point for how
# little margin a stalled winding has.
# WHY THESE NUMBERS, AND WHY THE FIRST GUESS WAS TOO SMALL. The energy a burst has to deposit is
# set by the instrument, not by the motor: a handheld probe resolves ~0.5 degC, so a useful CASE
# rise is >= 3 degC, and the case rise is E / (C_w + C_c). For a ~1 kg servo that is C ~ 1150 J/K,
# so E >= 3400 J. At 12 A and k_cu ~ 0.18 W/A^2 that is 26 W -- 130 seconds, not 10.
#
# MEASURED in thermal_fit --campaign-self-test: 12 A for 5-30 s produces 0.1-0.7 degC at the case,
# i.e. nothing a person can read. The first version of this file capped duration at 30 s and
# current at 20 A, which made every burst in range unmeasurable.
#
# The binding safety constraint is the WINDING, not the case: the same 3400 J that moves the case
# 3 degC moves the copper E / C_w ~ 36 degC, and it does it in seconds. So the envelope is bounded
# by predicted winding rise (see predict()), which is what the caller must check -- these are only
# the outer walls.
MAX_AMPS = 40.0
MAX_DURATION_S = 180.0
MIN_DURATION_S = 1.0
MAX_SPEED_ERPM = 3000.0
MAX_WINDOW_DEG = 25.0
# A clamped joint that moves is a clamp that is failing -- but "moves" has to mean MOVES, not
# "twitched once". MEASURED 2026-08-29: the first run on a genuinely static joint aborted 26 ms in
# on a 0.3 deg twitch (750 ERPM ~ 14 deg/s) while the soft-start was still at 0.04 A rms. Real
# fixtures have compliance and gearboxes have backlash, so first torque always produces a small
# step, and a single-sample threshold at 5.5 deg/s cannot tell that from a clamp letting go.
#
# So: the condition is DEBOUNCED -- it has to hold for BLOCKED_SLIP_TICKS consecutive control ticks
# before it counts -- and the thresholds are set where a failing clamp lives, not where backlash
# does.
BLOCKED_SLIP_DEG = 8.0
BLOCKED_SLIP_ERPM = 2000.0
BLOCKED_SLIP_TICKS = 20            # 0.1 s at 200 Hz

# FREE-ROTOR mode: the motor is off the robot with nothing on the shaft. The run tracks a
# position-mode sine so the current comes from fighting the rotor's own inertia (see the module
# docstring). Position is only a soft bound (nothing to hit), speed and the measured current are
# the real ones.
FREE_SPEED_ERPM = 12000.0
# The sine cannot COMMAND a current -- I = J*alpha/Kt is set by physics -- so the amps knob is a
# target, and a run whose measured current averages below this fraction of it is depositing far
# less heat than the operator sized the burst for. Stop early and say the achieved number, instead
# of leaving them watching a case that never warms.
FREE_MIN_CURRENT_FRAC = 0.35
FREE_CURRENT_GRACE_S = 3.0         # the ramp-in and the first cycles live in here
FREE_SINE_FREQ_MIN_HZ = 0.5
FREE_SINE_FREQ_MAX_HZ = 25.0       # the drive's ~10 ms actuation delay makes faster unrenderable
FREE_SINE_AMP_MAX_DEG = 60.0
# The rotor started HERE and the sine is centred on it; drifting out of the band means net
# rotation -- something dragging the shaft, or a wrong zero making set_pos slew -- and that is not
# the declared experiment. Debounced like the blocked slip check, and for the same reason.
FREE_DRIFT_MARGIN_DEG = 15.0
FREE_DRIFT_TICKS = 20              # 0.1 s at 200 Hz
# Position mode has no current cap this module controls. The winding-rise gate approved
# amps^2 * duration of I^2 dt; the run aborts when the MEASURED integral exceeds that with this
# much headroom, so it can never deposit more heat than it was cleared for.
FREE_ENERGY_HEADROOM = 1.2


class Envelope:
    """The bounds a burst runs inside. All angles are NORMALIZED degrees (the web UI frame)."""

    def __init__(self, amps, duration_s, centre_deg, window_deg=8.0, speed_erpm=600.0,
                 temp_abort_c=70.0, pos_lo=None, pos_hi=None, stale_s=0.25, blocked=True,
                 mode="blocked", freq_hz=6.0, sine_amp_deg=20.0):
        self.amps = float(np.clip(abs(amps), 0.0, MAX_AMPS))
        self.duration_s = float(np.clip(duration_s, MIN_DURATION_S, MAX_DURATION_S))
        self.centre = float(centre_deg)
        self.window = float(np.clip(abs(window_deg), 1.0, MAX_WINDOW_DEG))
        self.speed = float(np.clip(abs(speed_erpm), 50.0, MAX_SPEED_ERPM))
        self.temp_abort_c = float(temp_abort_c)
        self.stale_s = float(stale_s)
        # blocked=False is the retracted free-joint dither; it is kept only so the recorded
        # runaway stays reproducible in the tests, and nothing in the daemon constructs it.
        self.blocked = bool(blocked)
        # free-rotor: nothing attached to the shaft. The run tracks a position-mode sine and the
        # heat comes from the rotor fighting its own inertia; see the module docstring.
        self.free_rotor = bool(blocked) and str(mode) == "free"
        self.freq_hz = float(np.clip(freq_hz, FREE_SINE_FREQ_MIN_HZ, FREE_SINE_FREQ_MAX_HZ))
        self.sine_amp = float(np.clip(abs(sine_amp_deg), 1.0, FREE_SINE_AMP_MAX_DEG))
        # what the sine asks of the mechanism, for the panel to show before anyone presses start
        self.sine_peak_dps = 2.0 * np.pi * self.freq_hz * self.sine_amp
        if self.free_rotor:
            self.speed = FREE_SPEED_ERPM
        # the joint's own hard band, intersected with the window -- the window is relative to
        # wherever the joint happens to be, so on its own it can still walk into a stop
        lo = self.centre - self.window
        hi = self.centre + self.window
        self.lo = lo if pos_lo is None else max(lo, float(pos_lo))
        self.hi = hi if pos_hi is None else min(hi, float(pos_hi))
        if self.hi - self.lo < 1.0:
            raise ValueError(
                "the safe window for this joint is only {:.1f} deg wide at this pose ({:.1f} to "
                "{:.1f}). Move the joint away from its limit before running a burst."
                .format(self.hi - self.lo, self.lo, self.hi))

    def as_dict(self):
        d = {"amps": self.amps, "duration_s": self.duration_s, "centre_deg": self.centre,
             "window_deg": self.window, "speed_erpm": self.speed, "lo_deg": self.lo,
             "hi_deg": self.hi, "temp_abort_c": self.temp_abort_c, "blocked": self.blocked,
             "free_rotor": self.free_rotor}
        if self.free_rotor:
            d.update(freq_hz=self.freq_hz, sine_amp_deg=self.sine_amp,
                     sine_peak_dps=round(self.sine_peak_dps, 1))
        return d


class BurstExciter:
    """One burst. Feed it telemetry at the control rate; it returns the current to command."""

    def __init__(self, env, ramp_s=0.25):
        self.env = env
        self.ramp_s = float(ramp_s)
        self.sign = 1.0
        self.abort = None
        self.n_reversals = 0
        self.i_sq_dt = 0.0            # integral of I^2 dt -- the deposited-energy proxy
        self.t_last = None
        self.pos_min = None
        self.pos_max = None
        self.spd_peak = 0.0
        self.slip_ticks = 0           # consecutive ticks the slip condition has held
        self.i_meas_sum = 0.0
        self.i_meas_n = 0

    def step(self, t, pos_deg, spd_erpm, temp_c, err, telemetry_age, i_meas=None):
        """Return (amps, done, abort_reason). amps is 0.0 whenever anything is wrong."""
        e = self.env
        if self.abort:
            return 0.0, True, self.abort
        if t >= e.duration_s:
            return 0.0, True, None
        if telemetry_age > e.stale_s:
            self.abort = "no status frame for {:.2f} s".format(telemetry_age)
            return 0.0, True, self.abort
        if err:
            self.abort = "drive error code {}".format(err)
            return 0.0, True, self.abort
        if temp_c is not None and temp_c >= e.temp_abort_c:
            self.abort = "reported temperature {:.0f} C at/over the {:.0f} C abort".format(
                temp_c, e.temp_abort_c)
            return 0.0, True, self.abort

        pos = float(pos_deg)
        spd = float(spd_erpm)
        self.pos_min = pos if self.pos_min is None else min(self.pos_min, pos)
        self.pos_max = pos if self.pos_max is None else max(self.pos_max, pos)
        self.spd_peak = max(self.spd_peak, abs(spd))
        # a joint that has left the band entirely is not something to keep driving, in either
        # direction -- that is the backstop, not a reversal
        # ...except on a free rotor, where nothing is attached to the shaft and an angle is not a
        # bound at all. Position only means something when it can hit something.
        if not e.free_rotor and (pos < e.lo - 3.0 or pos > e.hi + 3.0):
            self.abort = ("joint left its window: {:.1f} deg is outside [{:.1f}, {:.1f}]"
                          .format(pos, e.lo, e.hi))
            return 0.0, True, self.abort

        if e.blocked:
            if e.free_rotor:
                raise ValueError("free-rotor runs are position-mode: drive them with step_sine()")
            slipping = (abs(pos - e.centre) > BLOCKED_SLIP_DEG
                        or abs(spd) > BLOCKED_SLIP_ERPM)
            self.slip_ticks = self.slip_ticks + 1 if slipping else 0
            if self.slip_ticks >= BLOCKED_SLIP_TICKS:
                self.abort = ("the joint has moved {:+.1f} deg and is turning at {:.0f} ERPM, "
                              "held for {:.0f} ms -- the clamp is slipping"
                              .format(pos - e.centre, spd, BLOCKED_SLIP_TICKS * 5.0))
                return 0.0, True, self.abort
            scale = min(1.0, t / self.ramp_s) if self.ramp_s > 0 else 1.0
            amps = e.amps * scale
            if self.t_last is not None:
                self.i_sq_dt += (amps ** 2) * max(0.0, t - self.t_last)
            self.t_last = t
            return amps, False, None

        # hysteresis: reverse on the speed limit (the usual case) or the window edge (backstop)
        if self.sign > 0 and (spd >= e.speed or pos >= e.hi):
            self.sign = -1.0
            self.n_reversals += 1
        elif self.sign < 0 and (spd <= -e.speed or pos <= e.lo):
            self.sign = 1.0
            self.n_reversals += 1

        # soft-start: a step to full current is an impulse into the mechanism, and it also puts a
        # transient into the very first part of the deposited-energy integral
        scale = min(1.0, t / self.ramp_s) if self.ramp_s > 0 else 1.0
        amps = self.sign * e.amps * scale
        if self.t_last is not None:
            self.i_sq_dt += (amps ** 2) * max(0.0, t - self.t_last)
        self.t_last = t
        return amps, False, None

    def step_sine(self, t, pos_deg, spd_erpm, temp_c, err, telemetry_age, i_meas=None):
        """One tick of a FREE-ROTOR run. Returns (target_deg, done, abort_reason).

        The command is a POSITION target, not a current: the caller streams it in position mode
        and the drive's own loop closes the fast loop -- the same stable arrangement as the
        joint-identify wiggle, and the reason this is not the retracted current-mode dither. The
        heat comes from the rotor decelerating and re-accelerating its own inertia every
        half-cycle; the deposited-energy integral therefore uses the MEASURED current, because a
        position sine cannot command one."""
        e = self.env
        if self.abort:
            return e.centre, True, self.abort
        if t >= e.duration_s:
            return e.centre, True, None
        if telemetry_age > e.stale_s:
            self.abort = "no status frame for {:.2f} s".format(telemetry_age)
            return e.centre, True, self.abort
        if err:
            self.abort = "drive error code {}".format(err)
            return e.centre, True, self.abort
        if temp_c is not None and temp_c >= e.temp_abort_c:
            self.abort = "reported temperature {:.0f} C at/over the {:.0f} C abort".format(
                temp_c, e.temp_abort_c)
            return e.centre, True, self.abort

        pos = float(pos_deg)
        spd = float(spd_erpm)
        self.pos_min = pos if self.pos_min is None else min(self.pos_min, pos)
        self.pos_max = pos if self.pos_max is None else max(self.pos_max, pos)
        self.spd_peak = max(self.spd_peak, abs(spd))
        if abs(spd) > e.speed:
            self.abort = ("the rotor reached {:.0f} ERPM, over the {:.0f} limit"
                          .format(spd, e.speed))
            return e.centre, True, self.abort
        # Net rotation out of the sine's band is not the declared experiment: something is
        # dragging the shaft, or a wrong zero is making set_pos slew. Debounced -- the first
        # cycles of a lagging position loop overshoot briefly and mean nothing.
        drifting = abs(pos - e.centre) > e.sine_amp + FREE_DRIFT_MARGIN_DEG
        self.slip_ticks = self.slip_ticks + 1 if drifting else 0
        if self.slip_ticks >= FREE_DRIFT_TICKS:
            self.abort = ("the rotor is {:+.1f} deg from the sine's centre, outside the "
                          "+-{:.0f} deg band -- net rotation means something is on the shaft, "
                          "or the zero is wrong"
                          .format(pos - e.centre, e.sine_amp + FREE_DRIFT_MARGIN_DEG))
            return e.centre, True, self.abort

        if i_meas is not None:
            if self.t_last is not None:
                self.i_sq_dt += float(i_meas) ** 2 * max(0.0, t - self.t_last)
            # Position mode has no current cap this module controls, so the INTEGRAL is the wall:
            # the run may never deposit more than the amps x duration the winding gate approved.
            if self.i_sq_dt > FREE_ENERGY_HEADROOM * e.amps ** 2 * e.duration_s:
                self.abort = ("measured I^2*dt reached {:.0f} A^2*s -- the whole deposit the "
                              "{:.0f} A x {:.0f} s gate approved, {:.0f} s early. The drive is "
                              "drawing more than the run was sized for; not risking the winding."
                              .format(self.i_sq_dt, e.amps, e.duration_s, e.duration_s - t))
                return e.centre, True, self.abort
            if t > FREE_CURRENT_GRACE_S:
                self.i_meas_sum += abs(float(i_meas))
                self.i_meas_n += 1
                avg = self.i_meas_sum / max(1, self.i_meas_n)
                if avg < FREE_MIN_CURRENT_FRAC * e.amps:
                    self.abort = (
                        "the sine at +-{:.0f} deg / {:.1f} Hz draws {:.1f} A on average, against "
                        "the {:.0f} A this run was sized for -- fighting its own rotor inertia "
                        "cannot draw more here. Raise the frequency or amplitude, or re-size the "
                        "burst for the current it actually draws."
                        .format(e.sine_amp, e.freq_hz, avg, e.amps))
                    return e.centre, True, self.abort
        self.t_last = t

        # raised-cosine ramp at both ends: no velocity step going in, and the final cycles ease
        # back to the centre instead of stopping mid-swing
        ramp = min(self.ramp_s, 0.4 * e.duration_s)
        if ramp <= 0.0:
            g = 1.0
        elif t < ramp:
            g = 0.5 * (1.0 - float(np.cos(np.pi * t / ramp)))
        elif t > e.duration_s - ramp:
            g = 0.5 * (1.0 - float(np.cos(np.pi * max(0.0, e.duration_s - t) / ramp)))
        else:
            g = 1.0
        want = e.centre + e.sine_amp * g * float(np.sin(2.0 * np.pi * e.freq_hz * t))
        return want, False, None

    def summary(self):
        """What the fit needs from the burst, independent of how the log was sampled."""
        d = self.env.duration_s
        return {"i_sq_dt": self.i_sq_dt,
                "i_rms": float(np.sqrt(self.i_sq_dt / d)) if d > 0 else 0.0,
                "reversals": self.n_reversals,
                "travel_deg": (None if self.pos_min is None else self.pos_max - self.pos_min),
                "peak_erpm": self.spd_peak,
                "abort": self.abort}


def expected_peak_delay_s(params):
    """When the CASE temperature is expected to peak after a short burst, seconds.

    This is the number that makes the manual reading trustworthy. The burst dumps its energy into
    the copper in seconds; the case only sees it as that heat diffuses across R_wc, so a probe on
    the case peaks roughly one winding time constant later. Read the thermometer too early and you
    record a number that is systematically low -- and low in a way that looks like a smaller heat
    input rather than like a mistimed reading, so it biases k_cu instead of showing up as scatter.
    """
    return float(3.0 * params.tau_w / (1.0 + params.tau_w / max(params.tau_c, 1e-9)))


def expected_peak_rise_c(params, i_sq_dt, probe="case"):
    """Rough predicted rise for a burst, degC -- shown in the UI so a run that is going to measure
    nothing can be recognised BEFORE spending three minutes on it.

    Energy in is k_cu * integral(I^2 dt). For a burst short against the case time constant almost
    none of it has escaped to ambient yet, so the case settles near E / (C_w + C_c) while the
    winding peaks near E / C_w. The gap between those two is exactly the thing a case-mounted
    probe cannot see, and it is why the winding node exists."""
    e = params.k_cu * float(i_sq_dt)
    if probe == "winding":
        return e / max(params.c_w, 1e-9)
    return e / max(params.c_w + params.c_c, 1e-9)


def predict(params, amps, duration_s):
    """Predicted temperature rise for a burst, degC, at BOTH nodes.

    This is the number that decides whether a run is worth doing and whether it is safe, and it is
    cheap enough to recompute on every keystroke in the panel:

      case rise    ~ E / (C_w + C_c)   -- what the operator will actually be able to read
      winding rise ~ E / C_w           -- what decides whether the burst is safe

    The gap between them is the whole reason the model has two nodes, and it is also the trap in
    this experiment: the energy needed to move the case by a readable amount moves the copper by
    ten times as much, in seconds, with no sensor watching. Both numbers come from the CURRENT
    parameter set, so before calibration they are placeholder-quality -- which is exactly why the
    caller should treat the winding figure as a floor on the risk, not as a measurement."""
    e = params.k_cu * float(amps) ** 2 * float(duration_s)
    return {"energy_j": e,
            "case_c": e / max(params.c_w + params.c_c, 1e-9),
            "winding_c": e / max(params.c_w, 1e-9)}


def check_burst(params, amps, duration_s, min_case_rise_c=3.0, max_winding_rise_c=60.0):
    """(ok, reason, prediction). Refuses a burst that cannot be measured or should not be run."""
    pr = predict(params, amps, duration_s)
    if pr["winding_c"] > max_winding_rise_c:
        return False, ("predicted winding rise {:.0f} degC exceeds the {:.0f} degC burst limit -- "
                       "shorten it or drop the current. The case will only move {:.1f} degC, so "
                       "nothing you can read would warn you about this."
                       .format(pr["winding_c"], max_winding_rise_c, pr["case_c"])), pr
    if pr["case_c"] < min_case_rise_c:
        return False, ("predicted CASE rise is only {:.2f} degC -- below what a handheld probe can "
                       "resolve, so this run would produce no usable measurement. Needs about "
                       "{:.0f}x more energy: more current, or a longer burst."
                       .format(pr["case_c"], min_case_rise_c / max(pr["case_c"], 1e-6))), pr
    return True, "", pr


def suggest(params, target_rise_c=6.0, amps=None, i_limit=12.0):
    """A burst that should produce ~target_rise_c of CASE rise. Returns (amps, duration_s).

    Deliberately solves for duration at a fixed current rather than the other way round: current
    is the axis with the damage risk, so it stays at whatever the operator has already accepted,
    and the knob that grows is time -- which the abort watches every tick."""
    amps = float(i_limit if amps is None else amps)
    per_s = params.k_cu * amps * amps / max(params.c_w + params.c_c, 1e-9)
    if per_s <= 0:
        return amps, MAX_DURATION_S
    return amps, float(np.clip(target_rise_c / per_s, MIN_DURATION_S, MAX_DURATION_S))
