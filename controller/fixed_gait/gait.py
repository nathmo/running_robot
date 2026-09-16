"""Hard-coded straight-walking gait for DASH-01 — the single source of truth.

PURE NUMPY, NO MUJOCO. This module runs unchanged on the Raspberry Pi (whose runtime is
onnxruntime/numpy/python-can only — see requirements-rpi.txt) and on the desktop sim. The sim
viewer and the CAN hardware runner both import `GaitGenerator` from here so the two can never
drift apart.

The robot base is FIXED IN SPACE (in the air). This is a demonstrator: it proves all six
motors move in a coordinated walking pattern; there is no balance to keep.

Joint convention (sim actuator / ctrl order, angles in radians):
    index 0: hip_roll_L (abduction, motor 104 on can1)   -> held at home (straight walking)
    index 1: cam_L      (cam,        motor 105 on can1)   -> knee bend via the pushrod loop
    index 2: thigh_L    (hip,        motor 106 on can1)   -> fore/aft swing
    index 3: hip_roll_R (abduction, motor 104 on can0)
    index 4: cam_R      (cam,        motor 105 on can0)
    index 5: thigh_R    (hip,        motor 106 on can0)

Measured from the MuJoCo model (fixed base, gravity off, on-branch continuous probe):
    +thigh_L swings the LEFT foot forward (+X);  the RIGHT leg is the geometric mirror, so its
    joint signs are negated for the same body-frame motion. The two legs run half a cycle apart
    (antiphase): while one swings forward, the other sweeps back. Straight walking keeps both
    hip-roll (abduction) joints at their home angle.

The gait is a smooth periodic joint trajectory. Each leg:
    thigh(phi) = thigh_center +  swing_sign * A_thigh * sin(phi)
    cam(phi)   = cam_center   +  A_cam * clearance(phi)     # retract (knee up) during swing
Amplitudes are bounded well inside the joint limits (thigh +-1.047, cam +-1.5, hip_roll +-0.785),
so every commanded target is in range by construction.
"""
from dataclasses import dataclass, field
import numpy as np

# actuator/ctrl indices (order matches mujoco/dash01/dash01.xml <actuator>)
HIP_ROLL_L, CAM_L, THIGH_L, HIP_ROLL_R, CAM_R, THIGH_R = range(6)
N_ACT = 6

# joint limits (rad) from the model, for the in-range assertion
CTRL_LIMIT = np.array([0.785, 1.5, 1.047, 0.785, 1.5, 1.047])


@dataclass
class GaitParams:
    period_s: float = 1.4          # seconds per full stride cycle (both legs); bigger = slower
    # gait CENTER pose (rad). The cam is centred high (knee bent) so the whole cycle stays in the
    # well-conditioned part of the workspace: below cam ~= 0.3 (with the thigh forward) the 4-bar
    # crosses its dead-centre and the foot folds/whips. The gait ellipse below keeps cam in
    # [center-amp, center+amp] = [0.42, 0.78], clear of that fold. Mirror: right = -left.
    thigh_center: float = 0.20     # left thigh centred slightly forward; right centred at -0.20
    cam_center: float = 0.60       # left knee bent so the foot sits mid-height; right at -0.60
    hip_roll_center: float = 0.0   # abduction stays home for straight walking
    # swing amplitudes (rad). The reachability study found the foot workspace is a THIN diagonal
    # band (thigh and cam both move the foot along nearly the same up-forward line — the 2-DOF
    # Jacobian is near-singular everywhere, so a fat foot ellipse is mechanically impossible).
    # The gait therefore sweeps the foot back and forth ALONG that band; running thigh and cam a
    # quarter-cycle apart (sin vs cos) opens the small swing/stance loop the band allows. Both
    # amplitudes stay well inside the joint limits.
    thigh_amp: float = 0.25        # fore/aft swing of the thigh (the visible stepping motion)
    cam_amp: float = 0.18          # knee lift/plant over the cycle (swing up, stance down)
    ramp_s: float = 1.5            # soft-start: ease amplitudes 0 -> full over this many seconds
    settle_center_s: float = 0.0   # (hardware) hold the center pose this long before stepping


class GaitGenerator:
    """Time -> 6 PD position targets (rad), in sim actuator order. Deterministic and stateless
    apart from the start-time reference passed in, so sim and hardware get identical output."""

    def __init__(self, params: GaitParams = None):
        self.p = params or GaitParams()

    # --- per-leg signal helpers -------------------------------------------------
    def _phase(self, t):
        return 2.0 * np.pi * (t / self.p.period_s)

    def _ramp(self, t):
        if self.p.ramp_s <= 0:
            return 1.0
        return float(np.clip(t / self.p.ramp_s, 0.0, 1.0))

    def targets(self, t):
        """Return the 6 joint position targets (rad) at time t seconds since gait start.
        t < 0 (or during settle) holds the center pose."""
        p = self.p
        c = self.center_pose()
        if t <= p.settle_center_s:
            return c.copy()
        te = t - p.settle_center_s
        phi = self._phase(te)
        r = self._ramp(te)

        # Foot ellipse per leg: thigh = sin (fore/aft), cam = cos (lift), a quarter-cycle apart,
        # so the foot lifts as it swings forward and plants as it sweeps back.
        # LEFT leg
        thigh_L = p.thigh_center + r * p.thigh_amp * np.sin(phi)
        cam_L = p.cam_center + r * p.cam_amp * np.cos(phi)
        # RIGHT leg: geometric mirror (negate) + half-cycle antiphase (phi + pi)
        phiR = phi + np.pi
        thigh_R = -(p.thigh_center + r * p.thigh_amp * np.sin(phiR))
        cam_R = -(p.cam_center + r * p.cam_amp * np.cos(phiR))

        out = c.copy()
        out[THIGH_L] = thigh_L
        out[CAM_L] = cam_L
        out[THIGH_R] = thigh_R
        out[CAM_R] = cam_R
        return self._clip(out)

    def center_pose(self):
        p = self.p
        c = np.zeros(N_ACT)
        c[HIP_ROLL_L] = p.hip_roll_center
        c[HIP_ROLL_R] = -p.hip_roll_center
        c[THIGH_L] = p.thigh_center
        c[THIGH_R] = -p.thigh_center
        c[CAM_L] = p.cam_center
        c[CAM_R] = -p.cam_center
        return c

    @staticmethod
    def _clip(x):
        return np.clip(x, -CTRL_LIMIT, CTRL_LIMIT)


if __name__ == "__main__":
    # quick self-check: print a cycle and confirm everything stays in range
    g = GaitGenerator()
    print("center pose (rad):", np.round(g.center_pose(), 3))
    ok = True
    for t in np.linspace(0, g.p.period_s, 9):
        tg = g.targets(t + 5.0)  # past the ramp
        inrange = np.all(np.abs(tg) <= CTRL_LIMIT + 1e-9)
        ok &= inrange
        print(f"t={t:4.2f}  {np.round(tg,3)}  in-range={inrange}")
    print("ALL IN RANGE" if ok else "OUT OF RANGE!")
