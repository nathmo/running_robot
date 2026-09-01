"""Real AKE90-8 (cam/thigh) motor envelope, joint-side (after the 8:1 gearbox), for leg2d/sweep.py.

Numbers, all from memory/spiderbot-hardware.md (measured 2026-08-04/26):
  PEAK_NM   144.5 N*m   delivered peak torque. The datasheet quotes 170 N*m; the measured effective
                        gearbox efficiency is 0.85 (known-mass static test, robot/identification),
                        so 170*0.85 = 144.5 is what the joint can actually deliver -- this is the
                        SAME number already baked into dash01.xml's actuator forcerange.
  NO_LOAD_RAD_S  22.0   no-load joint speed (rad/s). Not efficiency-derated (a kinematic quantity).
  CONT_NM    55.0       continuous (thermal) torque rating, used directly as in
                        memory/hardware-speed-ceiling.md ("1.8 m/s sustained on the 55 N*m
                        continuous rating") rather than re-deriving a further-derated number --
                        robot/deploy/thermal.py's two-node RC model exists but is explicitly
                        UNCALIBRATED for this motor, so it is not used here.

TORQUE-SPEED CLAMP. dash01.xml's <position> actuators enforce a CONSTANT forcerange (144.5 N*m)
regardless of joint speed -- a real brushless motor cannot do that; deliverable torque falls off
linearly to zero at the no-load speed. That falloff is exactly the mechanism behind memory's
"swing is power-limited" finding, so leg2d drives cam_L/thigh_L as direct-torque <motor> actuators
and applies this clamp itself every control step instead of relying on MuJoCo's actuator model.

The clamp only throttles MOTORING (torque and velocity the same sign -- delivering mechanical
power). BRAKING (torque opposing velocity -- dissipating power) gets the full peak torque: a real
drive is current-limited, not power-limited, when it's pulling energy OUT of the load.

PEAK POWER. peak_power_w() = PEAK_NM * NO_LOAD_RAD_S / 4 = 794.75 W at the (144.5, 22) pairing used
consistently here. memory/hardware-speed-ceiling.md quotes 935 W because that ad-hoc calculation
paired the RAW 170 N*m stall torque with the same no-load speed -- an inconsistency inherited from
mixing a derated and an un-derated number. 794.75 W (both numbers on the same, delivered/measured
basis) is used throughout this package; the two are close enough (15%) not to change any
conclusion, but the discrepancy is noted here rather than silently resolved.
"""
import numpy as np

PEAK_NM = 144.5
NO_LOAD_RAD_S = 22.0
CONT_NM = 55.0


def peak_power_w(peak=PEAK_NM, omega0=NO_LOAD_RAD_S):
    return peak * omega0 / 4.0


def clamp_torque(tau_cmd, omega, peak=PEAK_NM, omega0=NO_LOAD_RAD_S):
    """Real single-joint torque-speed envelope. `tau_cmd`, `omega` are scalars or same-shape
    arrays (N*m, rad/s); returns the actually-deliverable torque, same shape."""
    tau_cmd = np.asarray(tau_cmd, dtype=float)
    omega = np.asarray(omega, dtype=float)
    motoring = (np.sign(tau_cmd) == np.sign(omega)) & (omega != 0) & (tau_cmd != 0)
    cap = np.where(motoring, peak * np.clip(1.0 - np.abs(omega) / omega0, 0.0, 1.0), peak)
    return np.clip(tau_cmd, -cap, cap)


class ThermalTracker:
    """Running RMS torque per motor -- a simple I^2t-style continuous-rating check, not the full
    winding/case RC model (robot/deploy/thermal.py is explicitly uncalibrated for this motor, see
    module docstring). `feasible` compares the RMS over everything recorded so far to CONT_NM;
    treat this as a necessary, not sufficient, thermal check."""

    def __init__(self):
        self._sumsq = 0.0
        self._n = 0

    def add(self, tau):
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        self._sumsq += float(np.sum(tau ** 2))
        self._n += tau.size

    @property
    def rms(self):
        return float(np.sqrt(self._sumsq / self._n)) if self._n else 0.0

    def feasible(self, cont=CONT_NM):
        return self.rms <= cont
