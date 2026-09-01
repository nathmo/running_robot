"""Winding-temperature observer for motors that do not measure winding temperature.

THE PROBLEM
-----------
The AK drives report ONE temperature in their status frame (int8, 1 degC resolution). Whatever
that sensor is bonded to -- driver board or stator iron -- it is not the copper. The copper is
where the damage happens, it has by far the shortest thermal time constant, and it is invisible.

Measured on this robot: a stalled joint at 62 A held its reported temperature at 43 degC while the
event was already a damage path (see the 2026-08-26 cmd-8 incident). The reported number was not
wrong, it was just answering a slower question than the one being asked.

THE MODEL
---------
Two lumped nodes plus ambient, which is the standard motor-protection model (it is what an IEC
motor-protection relay and every servo drive with an I2t limiter implements underneath):

    C_w dTw/dt = P(t)              - (Tw - Tc)/R_wc
    C_c dTc/dt = (Tw - Tc)/R_wc    - (Tc - Ta)/R_ca

    P(t) = k_cu * I^2 * (1 + alpha*(Tw - T_ref))   copper loss, with the copper tempco: at a
                                                   100 degC rise the resistance is 39% higher, so
                                                   ignoring it under-predicts a hot motor exactly
                                                   when the prediction matters
         + k_fe * |omega|                          speed-proportional iron/friction loss
         + p_idle                                  standing losses (logic, holding current bias)

`k_cu` is in W/A^2 and is fitted, NOT computed from a datasheet phase resistance: it absorbs the
3/2 factor, the drive's definition of the reported current, and the winding configuration, all of
which are places to be quietly wrong by 50%.

The two-node structure is what buys the safety margin. A single node fitted to case data has the
case time constant (many minutes), so it completely misses the winding transient -- the 10-30 s
excursion that a burst of walking actually produces.

CLOSING THE LOOP ON THE ONE SENSOR WE DO HAVE
---------------------------------------------
The model integrates I^2, so any error in k_cu integrates too, and an open-loop observer drifts.
The drive's reported temperature is used as a measurement of the CASE node through a slow
Luenberger correction (`obs_gain`). Slow on purpose: fast enough to stop multi-minute drift,
far too slow to hide a winding transient the case has not felt yet. With the correction on, a 20%
error in k_cu shows up as a bounded offset instead of an unbounded ramp.

If the reported temperature is unavailable or implausible the correction switches itself off and
the observer runs open-loop, which is the conservative direction (no correction can pull the
estimate DOWN).

INTEGRATION
-----------
Exact discretisation (matrix exponential of the augmented system) computed ONCE per parameter set,
so the 200 Hz update is four multiplies and two adds per motor and is unconditionally stable for
any time step -- including the 20 ms the fitter uses and the multi-second steps a "what if I walk
for 10 minutes" projection wants.

UNITS: temperatures degC, currents A (as reported by the drive), speeds rad/s at the JOINT,
resistances K/W, capacitances J/K, times s.
"""
import json

import numpy as np

# Copper temperature coefficient of resistivity, per K, referenced to T_REF. This is physics, not
# a fit parameter -- it is 0.00393 for annealed copper and fitting it would just absorb model
# error into a constant everyone can look up.
ALPHA_CU = 0.00393
T_REF = 20.0


class ThermalParams:
    """Per-motor-type lumped parameters. `calibrated` gates deployment -- see MotorThermalModel."""

    __slots__ = ("name", "k_cu", "k_fe", "p_idle", "c_w", "c_c", "r_wc", "r_ca",
                 "t_trip", "t_derate", "t_warn", "calibrated", "source", "fit_rms_c")

    def __init__(self, name, k_cu, c_w, c_c, r_wc, r_ca, k_fe=0.0, p_idle=0.0,
                 t_trip=120.0, t_derate=90.0, t_warn=100.0,
                 calibrated=False, source="default", fit_rms_c=None):
        self.name = name
        self.k_cu = float(k_cu)
        self.k_fe = float(k_fe)
        self.p_idle = float(p_idle)
        self.c_w = float(c_w)
        self.c_c = float(c_c)
        self.r_wc = float(r_wc)
        self.r_ca = float(r_ca)
        self.t_trip = float(t_trip)
        self.t_derate = float(t_derate)
        self.t_warn = float(t_warn)
        self.calibrated = bool(calibrated)
        self.source = str(source)
        self.fit_rms_c = fit_rms_c
        if min(self.c_w, self.c_c, self.r_wc, self.r_ca) <= 0.0:
            raise ValueError("thermal capacitances and resistances must be positive")
        if not (self.t_derate < self.t_warn <= self.t_trip):
            raise ValueError("need t_derate < t_warn <= t_trip, got {} / {} / {}".format(
                self.t_derate, self.t_warn, self.t_trip))

    # -------------------------------------------------------------- derived, physical readouts
    @property
    def tau_w(self):
        """Winding time constant with the case held fixed, s. The number that says how long a
        burst can last."""
        return self.c_w * self.r_wc

    @property
    def tau_c(self):
        """Case-to-ambient time constant, s -- how long the machine takes to actually cool down."""
        return self.c_c * self.r_ca

    @property
    def r_total(self):
        """Winding-to-ambient steady-state thermal resistance, K/W."""
        return self.r_wc + self.r_ca

    def i_continuous(self, t_amb=25.0, t_limit=None, omega=0.0):
        """The current this motor can hold FOREVER without passing t_limit. Solves the steady
        state including the copper tempco, which is what makes this a fixed point rather than an
        algebraic rearrangement: hotter copper burns more power, which makes it hotter."""
        t_limit = self.t_trip if t_limit is None else t_limit
        budget = (t_limit - t_amb) / self.r_total - self.p_idle - self.k_fe * abs(omega)
        if budget <= 0.0:
            return 0.0
        return float(np.sqrt(budget / (self.k_cu * (1.0 + ALPHA_CU * (t_limit - T_REF)))))

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        return cls(**{k: d[k] for k in cls.__slots__ if k in d})


# ------------------------------------------------------------------------------------------------
# UNCALIBRATED PLACEHOLDERS. Order-of-magnitude values for a ~1 kg outrunner-class servo. They are
# here so the code runs and the tests have something to exercise; they are NOT a thermal model of
# these motors, and MotorThermalModel refuses to arm a policy run with calibrated=False unless the
# caller explicitly says it accepts that. Replace via thermal_fit.py.
# ------------------------------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "AKE90-8": ThermalParams("AKE90-8", k_cu=0.15, c_w=120.0, c_c=900.0, r_wc=0.9, r_ca=1.6,
                             k_fe=0.02, p_idle=1.0, t_trip=120.0, t_derate=90.0, t_warn=100.0),
    "AK60-39": ThermalParams("AK60-39", k_cu=0.25, c_w=90.0, c_c=700.0, r_wc=1.1, r_ca=2.0,
                             k_fe=0.02, p_idle=1.0, t_trip=120.0, t_derate=90.0, t_warn=100.0),
}


def _expm(M, terms=18):
    """Matrix exponential by scaling and squaring with a Taylor series. No scipy on the Pi.

    Squaring 2^s times after scaling the norm below 0.5 makes the truncated series accurate to
    float64 round-off for the small, well-conditioned matrices here. Called once per parameter
    set, never in the control loop."""
    M = np.asarray(M, float)
    nrm = float(np.abs(M).sum(axis=1).max())
    s = max(0, int(np.ceil(np.log2(nrm / 0.5))) if nrm > 0.5 else 0)
    A = M / (2.0 ** s)
    E = np.eye(A.shape[0])
    T = np.eye(A.shape[0])
    for k in range(1, terms + 1):
        T = T @ A / k
        E = E + T
    for _ in range(s):
        E = E @ E
    return E


def discretize(p, dt):
    """Exact zero-order-hold discretisation of the two-node system for one motor.

    State is (Tw - Ta, Tc - Ta): working in RISE-ABOVE-AMBIENT makes ambient an input rather than
    a state, so a slowly drifting room temperature never has to be integrated.

    Returns (Ad 2x2, Bd 2x1) with  x[k+1] = Ad x[k] + Bd P[k]."""
    a = np.array([[-1.0 / (p.r_wc * p.c_w), 1.0 / (p.r_wc * p.c_w)],
                  [1.0 / (p.r_wc * p.c_c), -(1.0 / p.r_wc + 1.0 / p.r_ca) / p.c_c]])
    b = np.array([[1.0 / p.c_w], [0.0]])
    # van Loan: expm([[A, B], [0, 0]] * dt) = [[Ad, Bd], [0, I]]
    m = np.zeros((3, 3))
    m[:2, :2] = a
    m[:2, 2:] = b
    e = _expm(m * float(dt))
    return e[:2, :2].copy(), e[:2, 2:3].copy()


class MotorThermalModel:
    """Bank of N independent two-node observers, stepped together at the control rate.

    Vectorised by hand rather than with einsum: it is a 2x2 per motor, and at 200 Hz the loop
    budget is 5 ms shared with CAN and inference."""

    def __init__(self, params, dt, t_amb=25.0, obs_gain=None, allow_uncalibrated=False,
                 names=None):
        self.params = list(params)
        self.n = len(self.params)
        self.dt = float(dt)
        self.names = list(names) if names else [p.name for p in self.params]
        uncal = [p.name for p in self.params if not p.calibrated]
        if uncal and not allow_uncalibrated:
            raise ValueError(
                "thermal parameters for {} are UNCALIBRATED placeholders. Run "
                "robot/deploy/thermal_calibrate.py on the robot and fit them with "
                "robot/deploy/thermal_fit.py, or pass allow_uncalibrated=True and accept that "
                "the winding-temperature estimate is a guess.".format(", ".join(sorted(set(uncal)))))
        self.uncalibrated = bool(uncal)

        ad, bd = [], []
        for p in self.params:
            a, b = discretize(p, self.dt)
            ad.append(a)
            bd.append(b[:, 0])
        ad = np.asarray(ad)
        bd = np.asarray(bd)
        self._a00, self._a01 = ad[:, 0, 0].copy(), ad[:, 0, 1].copy()
        self._a10, self._a11 = ad[:, 1, 0].copy(), ad[:, 1, 1].copy()
        self._b0, self._b1 = bd[:, 0].copy(), bd[:, 1].copy()

        self.k_cu = np.array([p.k_cu for p in self.params])
        self.k_fe = np.array([p.k_fe for p in self.params])
        # k_fe never changes, so asking np.any about it 200 times a second is pure dispatch cost
        self._has_iron_loss = bool(np.any(self.k_fe))
        self.p_idle = np.array([p.p_idle for p in self.params])
        self.t_trip = np.array([p.t_trip for p in self.params])
        self.t_derate = np.array([p.t_derate for p in self.params])
        self.t_warn = np.array([p.t_warn for p in self.params])
        self.r_total = np.array([p.r_total for p in self.params])

        # Luenberger correction on the CASE node only, expressed as a time constant so it can be
        # reasoned about physically: default 120 s, i.e. far slower than any winding transient
        # (tau_w is 30-120 s here) and far faster than an hour of accumulated k_cu drift.
        self.obs_tau = 120.0 if obs_gain is None else float(obs_gain)
        self._obs_k = self.dt / max(self.obs_tau, self.dt)

        self.t_amb = float(t_amb)
        self.reset(t_amb)

    # ------------------------------------------------------------------ state
    def reset(self, t_start=None, t_amb=None):
        """Start the observer. t_start is per-motor or scalar; use the drive's reported
        temperature at power-on -- a robot switched on after a run is NOT at ambient, and starting
        the estimate cold is the one direction the observer must never be optimistic in."""
        if t_amb is not None:
            self.t_amb = float(t_amb)
        t0 = self.t_amb if t_start is None else t_start
        t0 = np.broadcast_to(np.asarray(t0, float), (self.n,)).astype(float)
        # both nodes start at the measured body temperature: the winding cannot be assumed cooler
        self.x = np.stack([t0 - self.t_amb, t0 - self.t_amb], axis=1)
        self.p_last = np.zeros(self.n)
        self.n_steps = 0
        self.peak_w = t0.copy()

    @property
    def t_winding(self):
        return self.x[:, 0] + self.t_amb

    @property
    def t_case(self):
        return self.x[:, 1] + self.t_amb

    # ------------------------------------------------------------------ update
    def power(self, current, omega=None, t_w=None):
        """Instantaneous dissipation per motor, W."""
        i = np.asarray(current, float)
        t_w = self.t_winding if t_w is None else np.asarray(t_w, float)
        p = self.k_cu * i * i * (1.0 + ALPHA_CU * (t_w - T_REF)) + self.p_idle
        if omega is not None and self._has_iron_loss:
            p = p + self.k_fe * np.abs(np.asarray(omega, float))
        return p

    def step(self, current, omega=None, t_reported=None, t_amb=None, dt=None):
        """One control-rate update.

        current     : per-motor current, A, as the drive reports it (sign irrelevant, squared)
        omega       : per-motor joint speed, rad/s (optional; only used if k_fe is nonzero)
        t_reported  : per-motor drive temperature, degC, or None/NaN where unavailable. Used ONLY
                      to correct the case node, and only when it is plausible.
        t_amb       : ambient air temperature, degC (the Sense HAT's SHTC3 publishes it)
        dt          : override the step (for replay of a log with a different sample rate; the
                      discretisation is only exact at the dt the model was built with, so this
                      re-discretises and is not for the control loop)
        """
        if dt is not None and abs(float(dt) - self.dt) > 1e-12:
            raise ValueError("this model is discretised at dt={} s; build a second model for {} s "
                             "rather than integrating it at the wrong step".format(self.dt, dt))
        if t_amb is not None and np.isfinite(t_amb):
            self.t_amb = float(t_amb)
        p = self.power(current, omega)
        self.p_last = p
        x0, x1 = self.x[:, 0], self.x[:, 1]
        n0 = self._a00 * x0 + self._a01 * x1 + self._b0 * p
        n1 = self._a10 * x0 + self._a11 * x1 + self._b1 * p
        if t_reported is not None:
            t_rep = np.asarray(t_reported, float)
            # plausibility gate: an int8 field that reads -1, 0 or 200 is a missing/garbled sample,
            # not a cold motor, and letting it into the correction would drag the estimate DOWN --
            # the one direction a safety observer must never be pulled without evidence.
            ok = np.isfinite(t_rep) & (t_rep > self.t_amb - 15.0) & (t_rep < 200.0)
            innov = np.where(ok, (t_rep - self.t_amb) - n1, 0.0)
            n1 = n1 + self._obs_k * innov
            # the winding node follows the correction, because a case that is hotter than modelled
            # means the whole machine is hotter, not that the gradient reversed
            n0 = n0 + self._obs_k * innov
        self.x[:, 0] = n0
        self.x[:, 1] = n1
        self.n_steps += 1
        # t_winding is a property that builds a fresh array; it was being evaluated three times per
        # step. Once.
        t_w = self.t_winding
        np.maximum(self.peak_w, t_w, out=self.peak_w)
        return t_w

    # ------------------------------------------------------------------ budget / limiting
    def headroom(self):
        """1.0 below the derate knee, 0.0 at the trip point, linear between. This is the number
        the torque cap is scaled by."""
        span = np.maximum(self.t_trip - self.t_derate, 1e-6)
        return np.clip((self.t_trip - self.t_winding) / span, 0.0, 1.0)

    def torque_budget(self, tau_peak, tau_cont):
        """Per-motor allowed |torque|, N*m: full peak while cool, decaying to the continuous
        rating at the trip point. Never returns 0 -- a leg that is suddenly allowed no torque
        drops the robot, which is a worse outcome than a hot winding. Zero torque is the KILL
        path's job, and the kill path also puts the machine on the floor deliberately."""
        tau_peak = np.asarray(tau_peak, float)
        tau_cont = np.asarray(tau_cont, float)
        return tau_cont + (tau_peak - tau_cont) * self.headroom()

    def state(self):
        return {"t_winding": self.t_winding.tolist(), "t_case": self.t_case.tolist(),
                "t_amb": self.t_amb, "power_w": self.p_last.tolist(),
                "headroom": self.headroom().tolist(), "peak_winding": self.peak_w.tolist(),
                "over_warn": (self.t_winding >= self.t_warn).tolist(),
                "over_trip": (self.t_winding >= self.t_trip).tolist(),
                "uncalibrated": self.uncalibrated}

    def seconds_to_trip(self, current, omega=None, horizon_s=600.0, coarse_dt=0.5):
        """How long the CURRENT dissipation could be held before the winding reaches t_trip.

        Integrated forward on a coarse grid with a second, coarsely discretised copy of the same
        system -- the exact discretisation means a 0.5 s step is as valid as a 5 ms one, so a
        10-minute projection costs 1200 iterations of 2x2 arithmetic and can be run at 1 Hz for
        the operator display. inf means the load is sustainable indefinitely."""
        p = self.power(current, omega)
        ad, bd = [], []
        for pp in self.params:
            a, b = discretize(pp, coarse_dt)
            ad.append(a)
            bd.append(b[:, 0])
        ad, bd = np.asarray(ad), np.asarray(bd)
        x = self.x.copy()
        out = np.full(self.n, np.inf)
        n = int(horizon_s / coarse_dt)
        for k in range(n):
            x0 = ad[:, 0, 0] * x[:, 0] + ad[:, 0, 1] * x[:, 1] + bd[:, 0] * p
            x1 = ad[:, 1, 0] * x[:, 0] + ad[:, 1, 1] * x[:, 1] + bd[:, 1] * p
            x[:, 0], x[:, 1] = x0, x1
            hit = (x[:, 0] + self.t_amb >= self.t_trip) & ~np.isfinite(out)
            out = np.where((x[:, 0] + self.t_amb >= self.t_trip) & np.isinf(out),
                           (k + 1) * coarse_dt, out)
            if np.all(np.isfinite(out)):
                break
        return out


# ------------------------------------------------------------------------------------------------
# offline simulation, shared by thermal_fit.py so the fit and the robot run the SAME equations
# ------------------------------------------------------------------------------------------------
def simulate(p, t, current, omega=None, t_amb=25.0, t0=None):
    """Open-loop response of one motor to a measured current trace. No observer correction --
    the fit must be told by the data, not by a correction term that would hide the model error
    it is supposed to expose.

    t may be non-uniform; each interval is discretised exactly. Returns (Tw, Tc), both degC."""
    t = np.asarray(t, float)
    i = np.asarray(current, float)
    w = None if omega is None else np.asarray(omega, float)
    t_amb_arr = np.broadcast_to(np.asarray(t_amb, float), t.shape)
    n = len(t)
    tw = np.empty(n)
    tc = np.empty(n)
    start = float(t_amb_arr[0] if t0 is None else t0)
    x0 = x1 = start - float(t_amb_arr[0])
    tw[0], tc[0] = start, start
    # SCALAR inner loop, and one discretisation for a uniform grid. The fitter calls this once per
    # Jacobian column per LM iteration, so a 2x2 numpy matmul per sample (~4 us of dispatch) turns
    # a 20-second fit into a 20-minute one. Plain floats are ~40x faster here.
    dts = np.diff(t)
    uniform = n > 1 and float(np.max(np.abs(dts - dts[0]))) < 1e-9
    cache = {}
    if uniform:
        ad, bd = discretize(p, float(dts[0]))
        a00, a01, a10, a11 = ad[0, 0], ad[0, 1], ad[1, 0], ad[1, 1]
        b0, b1 = bd[0, 0], bd[1, 0]
    i2 = i * i
    kcu, pidle, kfe = p.k_cu, p.p_idle, p.k_fe
    for k in range(1, n):
        if not uniform:
            dt = float(t[k] - t[k - 1])
            if dt <= 0:
                tw[k], tc[k] = tw[k - 1], tc[k - 1]
                continue
            key = round(dt, 9)
            if key not in cache:
                a, b = discretize(p, dt)
                cache[key] = (a[0, 0], a[0, 1], a[1, 0], a[1, 1], b[0, 0], b[1, 0])
            a00, a01, a10, a11, b0, b1 = cache[key]
        pw = kcu * i2[k - 1] * (1.0 + ALPHA_CU * (tw[k - 1] - T_REF)) + pidle
        if w is not None and kfe:
            pw += kfe * abs(w[k - 1])
        x0, x1 = a00 * x0 + a01 * x1 + b0 * pw, a10 * x0 + a11 * x1 + b1 * pw
        tw[k] = x0 + t_amb_arr[k]
        tc[k] = x1 + t_amb_arr[k]
    return tw, tc


def load_params(path):
    """Read a fitted parameter file (written by thermal_fit.py). Returns {name: ThermalParams}."""
    with open(path, "r", encoding="utf-8-sig") as f:
        d = json.load(f)
    return {k: ThermalParams.from_dict(v) for k, v in d.get("motors", {}).items()}


def save_params(path, by_name, meta=None):
    d = {"meta": dict(meta or {}),
         "motors": {k: v.to_dict() for k, v in by_name.items()}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
