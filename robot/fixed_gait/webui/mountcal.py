#!/usr/bin/env python3
"""Where the Sense HAT's IMU sits on DASH-01, and how its axes relate to the robot's.

Two independent things live here, both persisted to `data/sensehat_mount.json`:

**1. The mount ROTATION** — chip axes -> body axes (X forward, Y left, Z up). Measured, not
declared, because the HAT is bolted UNDER the robot and reads gravity on chip -Z.

    Level capture   robot hung upright and still  ->  up_chip = the measured specific force.
    Forward capture robot tipped nose-down, still ->  the fore-aft axis and its sign.

The second capture is not optional bookkeeping: **gravity fixes only two of the three rotation
DOF.** Rotation *about* the vertical is invisible to an accelerometer at rest, and that is exactly
the DOF separating pitch from roll — the axis every balance question on this robot is about. A
declared axis (`set_declared`) can stand in for the tilt capture, and when both exist they are
cross-checked: a disagreement means the HAT is not bolted on square.

What the level capture does NOT do is separate accelerometer bias from mount misalignment — a robot
tilted 1 deg and a sensor with a 17 mg cross-axis bias produce the identical reading. It does not
need to: both are absorbed into the frame in which the reference pose reads roll = pitch = 0. A
true per-axis bias+scale calibration needs a 6-orientation tumble, which is not happening with a
15 kg robot for a sensor already within 1% of 1 g.

**2. The LEVER ARM** — the IMU's position relative to the base centre. An accelerometer offset from
the point you care about measures the rotational terms too:

    a_imu = a_base + alpha x r + omega x (omega x r)

At rest those vanish (so they never disturb the calibration above), but while the robot is running
they bias roll/pitch exactly when it matters. Two sources, deliberately kept side by side:
`lever_cad` typed in from CAD, and `lever_fit` regressed from a rocking excitation.

**Read the fit's reference point before comparing them.** A single IMU cannot observe its position
relative to the base centre: `a_base` above is unknown, so `r` is not separable. It becomes
identifiable only when the motion is rotation about a FIXED PIVOT, where `a_base` is itself a
function of the pivot and the fit returns `r_pivot->imu`. Hung on the test rig, the pivot is the
hang point — so the fit equals the CAD vector only insofar as the base centre sits at the pivot,
and the two otherwise differ by exactly the pivot offset. `lever_fit["about"]` says so out loud.
"""
import json
import math
import os
import threading
import time

import numpy as np

import paths

MOUNT_FILE = os.path.join(paths.DATA, "sensehat_mount.json")
G0 = 9.80665                    # m/s^2 per g, the unit the accelerometer is reported in

# Body frame: X forward, Y left, Z up (same convention as the MuJoCo model).
AXIS_VECTORS = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
                "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
                "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0)}


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotation_from(up_chip, fwd_chip):
    """Rotation taking a vector from CHIP axes to BODY axes (X forward, Y left, Z up).

    Its rows are the body axes written in chip coordinates. `fwd_chip` is orthogonalised against
    `up_chip` rather than trusted: the tilt capture's horizontal projection is only as clean as the
    hand that tipped the robot, and the two captures need not be exactly perpendicular."""
    z = _unit(up_chip)
    x = np.asarray(fwd_chip, float)
    x = x - np.dot(x, z) * z
    if np.linalg.norm(x) < 1e-6:                # forward parallel to up: unusable, keep it honest
        return None
    x = _unit(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def angle_between(a, b):
    """Degrees between two vectors (0 if either is degenerate)."""
    a, b = _unit(a), _unit(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def estimate_lever(samples, min_omega=0.6):
    """Least-squares fit of the IMU's position relative to the pivot the motion rotated about.

    `samples`: iterable of (f_body [m/s^2], omega [rad/s], alpha [rad/s^2], up_body [unit]), all in
    BODY axes. Rearranging the rigid-body relation with a_pivot = 0:

        a_imu = f_imu + g  =  (skew(alpha) + skew(omega)^2) r

    which is linear in r — three rows per sample, three unknowns. Samples too slow to carry
    information (|omega| under `min_omega` rad/s) are dropped rather than diluting the fit with rows
    of near-zero.

    Returns a dict with the fit AND what it is worth: the residual, the condition number, and how
    much of a SECOND rotation axis the excitation actually contained. A fit from rocking about one
    axis is rank-deficient along that axis — the number that comes back looks perfectly reasonable
    and means nothing — so `axis_coverage` is reported for the caller to judge."""
    rows, rhs, axes, wmax = [], [], [], 0.0
    for f, w, al, up in samples:
        w = np.asarray(w, float)
        wn = np.linalg.norm(w)
        wmax = max(wmax, wn)
        if wn < min_omega:
            continue
        A = skew(np.asarray(al, float)) + skew(w) @ skew(w)
        a_imu = np.asarray(f, float) + (-G0) * _unit(up)      # g points opposite the measured up
        rows.append(A)
        rhs.append(a_imu)
        axes.append(w / wn)
    n = len(rows)
    if n < 60:
        return {"ok": False, "error": f"only {n} samples had enough rotation "
                                      f"(peak |omega| {math.degrees(wmax):.0f} deg/s) — rock the "
                                      f"robot harder, about two different axes"}
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    r, _, _, sv = np.linalg.lstsq(A, b, rcond=None)
    resid = float(np.sqrt(np.mean((A @ r - b) ** 2)))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf")

    # Did the excitation contain a SECOND rotation axis? Measured sign-invariantly, as the second
    # eigenvalue of the axis scatter matrix: an axis and its negative are the same axis, so a
    # rocking oscillation must not be allowed to score as "two directions" just by reversing.
    # 0 = every rotation was about one line, 1 = isotropic.
    ax = np.array(axes)
    lam = np.linalg.eigvalsh(ax.T @ ax / len(ax))[::-1]
    coverage = float(lam[1] / lam[0]) if lam[0] > 1e-12 else 0.0
    return {"ok": True, "r": [float(v) for v in r], "samples": n,
            "residual_ms2": resid, "cond": cond, "axis_coverage": coverage,
            "peak_omega_dps": math.degrees(wmax),
            "weak": coverage < 0.05 or cond > 200.0}


class MountCal:
    """The persisted mount calibration. Thread-safe: the poll thread reads `R`/`lever` every tick
    while Flask handlers write captures."""

    def __init__(self):
        self._lock = threading.Lock()
        self.up_chip = None             # measured at the level capture (unit, chip axes)
        self.fwd_chip = None            # from the tilt capture (unit, chip axes)
        self.fwd_declared = None        # "+x" / "-x" / "+y" / "-y", the asserted forward axis
        self.captures = {}              # kind -> {when, spread, mag, n}
        self.lever_cad = None           # [x,y,z] m, base centre -> IMU, typed in from CAD
        self.lever_fit = None           # estimate_lever() result + {"about": ...}
        self.lever_use = "cad"          # which one feeds the live compensation: cad | fit | none
        self.reference = "hung on the test rig"
        self.updated = None
        self._R = np.eye(3)             # cached; rebuilt on every mutation
        # Bumped whenever the rotation changes. An attitude filter's state is expressed in the
        # frame it was integrated in, so it is meaningless the instant that frame moves — the poll
        # thread watches this counter and restarts the filter rather than slowly (or never)
        # converging from an arbitrarily wrong attitude.
        self.version = 0

    # ------------------------------------------------------------------ derived
    def _rebuild(self):
        self.version += 1
        fwd = self.fwd_chip
        if fwd is None and self.fwd_declared and self.up_chip is not None:
            # A declared body axis is a statement about the CHIP axis that points forward, so it is
            # already a chip-frame vector — it only needs squaring up against measured gravity.
            fwd = AXIS_VECTORS[self.fwd_declared]
        if self.up_chip is None or fwd is None:
            self._R = np.eye(3)
            return
        R = rotation_from(self.up_chip, fwd)
        self._R = np.eye(3) if R is None else R

    @property
    def R(self):
        """chip -> body rotation; identity while uncalibrated (i.e. values stay in chip axes)."""
        with self._lock:
            return self._R.copy()

    @property
    def calibrated(self):
        return self.up_chip is not None and (self.fwd_chip is not None or self.fwd_declared)

    def lever(self):
        """The lever arm actually used for compensation, in body axes, or None."""
        with self._lock:
            if self.lever_use == "cad" and self.lever_cad is not None:
                return np.asarray(self.lever_cad, float)
            if self.lever_use == "fit" and self.lever_fit and self.lever_fit.get("ok"):
                return np.asarray(self.lever_fit["r"], float)
            return None

    def cross_check(self):
        """Measured vs declared forward axis — the one number that says whether the HAT is square.
        None when only one of the two exists."""
        if self.fwd_chip is None or not self.fwd_declared:
            return None
        return angle_between(self.fwd_chip, AXIS_VECTORS[self.fwd_declared])

    def lever_disagreement(self):
        """Distance between the CAD lever and the fitted one, in metres."""
        if self.lever_cad is None or not (self.lever_fit and self.lever_fit.get("ok")):
            return None
        return float(np.linalg.norm(np.asarray(self.lever_cad) - np.asarray(self.lever_fit["r"])))

    # ------------------------------------------------------------------ mutations
    def set_level(self, acc_chip, meta):
        with self._lock:
            self.up_chip = [float(v) for v in _unit(acc_chip)]
            self.captures["level"] = {"when": time.time(), **meta}
            self._rebuild()
        self.save()

    def set_forward_from_tilt(self, acc_tilted, meta):
        """The tilt capture. Tipping the robot NOSE-DOWN swings the measured up-vector backwards in
        body axes, so the horizontal part of (tilted - level) points AFT and forward is its
        negative. Only the direction is used — the tilt angle never enters, which is why the step
        does not need to be a measured angle."""
        with self._lock:
            if self.up_chip is None:
                return False, "capture the level reference first"
            up = np.asarray(self.up_chip, float)
            d = np.asarray(acc_tilted, float) - up
            horiz = d - np.dot(d, up) * up
            tilt_deg = angle_between(acc_tilted, up)
            if np.linalg.norm(horiz) < 0.05:            # < ~3 deg of tilt: direction is all noise
                return False, (f"only {tilt_deg:.1f} deg of tilt — tip the robot nose-down further "
                               f"(10-20 deg) and hold it still")
            self.fwd_chip = [float(v) for v in _unit(-horiz)]
            self.captures["forward"] = {"when": time.time(), "tilt_deg": tilt_deg, **meta}
            self._rebuild()
        self.save()
        return True, None

    def set_declared(self, axis):
        axis = (axis or "").lower().strip()
        if axis in ("", "none", "null"):
            with self._lock:
                self.fwd_declared = None
                self._rebuild()
            self.save()
            return True, None
        if axis not in AXIS_VECTORS:
            return False, f"forward axis must be one of {', '.join(sorted(AXIS_VECTORS))}"
        with self._lock:
            self.fwd_declared = axis
            self._rebuild()
        self.save()
        return True, None

    def set_lever_cad(self, xyz):
        if xyz is None:
            with self._lock:
                self.lever_cad = None
        else:
            try:
                v = [float(x) for x in xyz]
            except (TypeError, ValueError):
                return False, "lever arm must be three numbers [x, y, z] in metres"
            if len(v) != 3 or not all(math.isfinite(x) for x in v):
                return False, "lever arm must be three finite numbers [x, y, z] in metres"
            if max(abs(x) for x in v) > 1.0:
                return False, "lever arm looks wrong: over 1 m from the base centre (units are metres)"
            with self._lock:
                self.lever_cad = v
        self.save()
        return True, None

    def set_lever_fit(self, fit, about="the pivot the excitation rotated about"):
        with self._lock:
            self.lever_fit = dict(fit, about=about) if fit else None
        self.save()
        return True, None

    def set_lever_use(self, which):
        if which not in ("cad", "fit", "none"):
            return False, "lever source must be cad, fit or none"
        with self._lock:
            self.lever_use = which
        self.save()
        return True, None

    def reset(self):
        with self._lock:
            self.up_chip = self.fwd_chip = self.fwd_declared = None
            self.captures = {}
            self.lever_fit = None
            self._rebuild()
        self.save()

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load_or_new(cls, path=MOUNT_FILE):
        c = cls()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8-sig") as f:
                    d = json.load(f)
                c.up_chip = d.get("up_chip")
                c.fwd_chip = d.get("fwd_chip")
                c.fwd_declared = d.get("fwd_declared")
                c.captures = d.get("captures") or {}
                c.lever_cad = d.get("lever_cad")
                c.lever_fit = d.get("lever_fit")
                c.lever_use = d.get("lever_use", "cad")
                c.reference = d.get("reference", c.reference)
                c.updated = d.get("updated")
                c._rebuild()
            except (ValueError, OSError) as e:
                print(f"(could not read {path}: {e} — starting with an uncalibrated IMU mount)")
        return c

    def save(self, path=MOUNT_FILE):
        with self._lock:
            self.updated = time.time()
            d = {"up_chip": self.up_chip, "fwd_chip": self.fwd_chip,
                 "fwd_declared": self.fwd_declared, "captures": self.captures,
                 "lever_cad": self.lever_cad, "lever_fit": self.lever_fit,
                 "lever_use": self.lever_use, "reference": self.reference,
                 "updated": self.updated,
                 # written for humans reading the file, never read back
                 "_R_chip_to_body": [[float(v) for v in row] for row in self._R]}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        os.replace(tmp, path)

    def snapshot(self):
        with self._lock:
            R = self._R.copy()
            snap = {
                "calibrated": self.up_chip is not None and (self.fwd_chip is not None or bool(self.fwd_declared)),
                "up_chip": self.up_chip, "fwd_chip": self.fwd_chip,
                "fwd_declared": self.fwd_declared, "captures": self.captures,
                "R_chip_to_body": [[round(float(v), 6) for v in row] for row in R],
                "lever_cad": self.lever_cad, "lever_fit": self.lever_fit,
                "lever_use": self.lever_use, "reference": self.reference,
                "updated": self.updated,
            }
        snap["cross_check_deg"] = self.cross_check()
        snap["lever_disagreement_m"] = self.lever_disagreement()
        lv = self.lever()
        snap["lever_active"] = None if lv is None else [float(v) for v in lv]
        return snap
