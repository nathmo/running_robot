"""Reusable hard-limit / workspace safety check for DASH-01's legs.

PURE NUMPY -- no matplotlib/scipy import here. This module is imported by the Pi-deployed hardware
runners (run_hardware.py, play_trajectory.py) as well as the teach recorder (record_trajectory.py),
so it must stay inside the onnxruntime/numpy/python-can Pi runtime (requirements-rpi.txt).

Built by fixed_gait/calibrate_workspace.py from hand-backdriven recordings:
  - abduction (single DOF, motor 104): a plain [min, max], eroded a bit inside what was physically
    demonstrated.
  - knee (cam=105, thigh=106 -- coupled through the parallel pushrod/knee loop): the valid set is a
    2-D region, not a box (see the dash01-hardware 4-bar assembly-band findings), so it's stored
    as a boolean occupancy GRID over the swept (cam, thigh) samples. calibrate_workspace.py already
    dilates (fills sampling gaps) then erodes (safety margin) the grid at build time; this module
    only does the O(1) cell lookup at check time.

All angles are RAW absolute motor degrees -- the same frame the CAN status frames already report --
so callers don't need to reconcile this with whatever home/offset convention they use internally.
"""
import os

import numpy as np

DEFAULT_PATH = "fixed_gait/calibration/joint_limits.npz"
LEGS = ("left", "right")


class JointLimits:
    def __init__(self, legs):
        self.legs = legs   # leg -> dict (see load())

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        z = np.load(path)
        legs = {}
        for leg in LEGS:
            if f"{leg}_abd_safe_min" not in z.files:
                continue
            legs[leg] = dict(
                abd_safe=(float(z[f"{leg}_abd_safe_min"]), float(z[f"{leg}_abd_safe_max"])),
                abd_observed=(float(z[f"{leg}_abd_observed_min"]), float(z[f"{leg}_abd_observed_max"])),
                abd_zero=float(z[f"{leg}_abd_zero"]),                    # NaN if never captured
                knee_grid=z[f"{leg}_knee_grid"].astype(bool),
                knee_cam_origin=float(z[f"{leg}_knee_cam_origin"]),
                knee_thigh_origin=float(z[f"{leg}_knee_thigh_origin"]),
                knee_grid_deg=float(z[f"{leg}_knee_grid_deg"]),
                knee_zero=z[f"{leg}_knee_zero"].copy(),                  # [cam,thigh], NaN if uncaptured
            )
        return cls(legs)

    def has_leg(self, leg):
        return leg in self.legs

    def check_abduction(self, leg, abd_deg):
        """(ok, reason) for a RAW abduction motor angle (deg)."""
        if leg not in self.legs:
            return False, f"no workspace calibration for {leg} leg"
        lo, hi = self.legs[leg]["abd_safe"]
        if lo <= abd_deg <= hi:
            return True, ""
        return False, f"{leg} abduction {abd_deg:+.1f} deg outside safe range [{lo:+.1f}, {hi:+.1f}]"

    def check_knee(self, leg, cam_deg, thigh_deg):
        """(ok, reason) for a RAW (cam, thigh) motor angle pair (deg) -- coupled parallel-loop DOF,
        so this is a grid lookup against the demonstrated-safe (cam, thigh) region, not two ranges."""
        if leg not in self.legs:
            return False, f"no workspace calibration for {leg} leg"
        L = self.legs[leg]
        grid, res = L["knee_grid"], L["knee_grid_deg"]
        i = int(np.floor((cam_deg - L["knee_cam_origin"]) / res))
        j = int(np.floor((thigh_deg - L["knee_thigh_origin"]) / res))
        if not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]) or not grid[i, j]:
            return False, (f"{leg} knee (cam={cam_deg:+.1f}, thigh={thigh_deg:+.1f} deg) outside "
                           f"calibrated safe workspace")
        return True, ""

    def validate(self, leg, abd_deg, cam_deg, thigh_deg):
        """(ok, reason) -- abduction AND knee must both be inside their calibrated safe region."""
        ok, reason = self.check_abduction(leg, abd_deg)
        if not ok:
            return ok, reason
        return self.check_knee(leg, cam_deg, thigh_deg)


def load_or_warn(path=DEFAULT_PATH):
    """Load JointLimits, or return None (+ a one-line warning) if no calibration exists yet.
    Callers should treat None as "skip the check" so scripts keep working before calibration."""
    if not os.path.exists(path):
        print(f"(no workspace calibration at {path} -- skipping safety check; "
              f"run fixed_gait/calibrate_workspace.py to build one)")
        return None
    limits = JointLimits.load(path)
    print(f"Loaded workspace safety limits from {path}: legs={sorted(limits.legs)}")
    return limits


if __name__ == "__main__":
    # ---- self-test: synthetic diagonal-band calibration, no hardware needed ----
    rng = np.random.default_rng(0)
    grid_deg = 1.0
    cam = np.linspace(-40, 40, 4000)
    thigh = 30 + 0.5 * cam + rng.normal(0, 3, cam.size)      # a noisy diagonal band, like the
                                                              # real cam/thigh 4-bar assembly band
    samples = np.stack([cam, thigh], axis=1)

    cam_lo, cam_hi = samples[:, 0].min() - 2, samples[:, 0].max() + 2
    th_lo, th_hi = samples[:, 1].min() - 2, samples[:, 1].max() + 2
    nc = int(np.ceil((cam_hi - cam_lo) / grid_deg)) + 1
    nt = int(np.ceil((th_hi - th_lo) / grid_deg)) + 1
    grid = np.zeros((nc, nt), bool)
    ic = np.clip(((samples[:, 0] - cam_lo) / grid_deg).astype(int), 0, nc - 1)
    jt = np.clip(((samples[:, 1] - th_lo) / grid_deg).astype(int), 0, nt - 1)
    grid[ic, jt] = True

    path = "fixed_gait/calibration/_selftest/joint_limits.npz"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path,
             left_abd_observed_min=-44.0, left_abd_observed_max=44.0,
             left_abd_safe_min=-41.0, left_abd_safe_max=41.0, left_abd_zero=0.0,
             left_knee_grid=grid, left_knee_cam_origin=cam_lo, left_knee_thigh_origin=th_lo,
             left_knee_grid_deg=grid_deg, left_knee_zero=np.array([0.0, 30.0]))
    limits = JointLimits.load(path)

    checks = [
        ("left abduction 20 deg (inside)",   limits.check_abduction("left", 20.0),   True),
        ("left abduction 44.5 deg (past safe margin)", limits.check_abduction("left", 44.5), False),
        ("left knee on the band (0, 30)",    limits.check_knee("left", 0.0, 30.0),    True),
        ("left knee off the band (0, -20)",  limits.check_knee("left", 0.0, -20.0),   False),
        ("right leg uncalibrated",           limits.validate("right", 0.0, 0.0, 0.0), False),
    ]
    ok = True
    for name, (got, reason), want in checks:
        passed = got is want
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: ok={got}  ({reason or 'no reason'})")
    print("has_leg('left') =", limits.has_leg("left"), " has_leg('right') =", limits.has_leg("right"))
    print("RESULT:", "PASS" if ok else "FAIL")
