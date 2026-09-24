"""Stage-2 Raibert prior on the relationship knobs (artifact §09), pure numpy, per commit.

    o_cam <- o_theta,cam - k_p * (v_hat - v_ref) - k_i * I_v,   I_v <- clip(I_v + (v_hat - v_ref) * T, +-I_max)
    o_hip <- o_theta,hip - k_y * v_hat_y - k_r * roll_lp

The step-to-step stabilizer: foot placement as a function of the measured velocity error, so any
theta in the library box converges to a periodic orbit of the body. The integral term is what
absorbs a constant force (wind) that a proportional law would lean into forever. The knobs it
modulates are the ones proven exact in workspace (o_cam 1.3 % spread, o_hip 6.6 %), and it writes
into the SAME latched spec the generator reads, so the once-block carries its output as the live
spec. Signs: o enters both legs (+,+); the FK rig showed o_cam > 0 moves BOTH feet the same way in
x, and a foot placed further forward decelerates the body (Raibert) -- hence the minus signs. They
are verified by the smoke test on the env's own rig, never assumed.

On the robot v_hat is the estimator's output; in sim the env hands the law the true base velocity
(optionally noised) as a stand-in. The law runs identically in both places.
"""
import numpy as np


class RaibertPrior:
    def __init__(self, kp, ki, imax, ky, kr, o_max, roll_tau_s=0.3):
        self.kp, self.ki, self.imax, self.ky, self.kr = float(kp), float(ki), float(imax), \
            float(ky), float(kr)
        self.o_max = np.asarray(o_max, dtype=float)      # (o_cam, o_thigh, o_hip) rad
        self.roll_tau_s = float(roll_tau_s)
        self.reset()

    def reset(self):
        self.I_v = 0.0
        self.roll_lp = 0.0

    def filter_roll(self, roll, dt):
        """Per-tick EMA of the base roll the o_hip term reads (gait wobble is not a lean)."""
        a = float(np.exp(-dt / max(self.roll_tau_s, 1e-6)))
        self.roll_lp = a * self.roll_lp + (1.0 - a) * float(roll)
        return self.roll_lp

    def commit(self, o_theta_raw, v_hat, v_ref, T):
        """Return the raw (unit-scaled) offset triple for the next cycle.

        o_theta_raw : the library entry's own (o_cam, o_thigh, o_hip) in [-1, 1]
        v_hat       : body-frame velocity estimate (vx, vy, ...)
        v_ref       : commanded forward speed (m/s)
        T           : cycle period (s), the integrator's sample time
        """
        ev = float(v_hat[0]) - float(v_ref)
        self.I_v = float(np.clip(self.I_v + ev * T, -self.imax, self.imax))
        o = np.asarray(o_theta_raw, dtype=float) * self.o_max        # to rad
        o = o.copy()
        o[0] -= self.kp * ev + self.ki * self.I_v
        o[2] -= self.ky * float(v_hat[1]) + self.kr * self.roll_lp
        return np.clip(o / np.maximum(self.o_max, 1e-9), -1.0, 1.0)
