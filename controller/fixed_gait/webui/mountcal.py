#!/usr/bin/env python3
"""How the Sense HAT's IMU is rotated on DASH-01: chip axes -> body axes (X forward, Y left, Z up).

Measured, not declared, because the HAT is bolted UNDER the robot and reads gravity on chip -Z.
Persisted to `data/sensehat_mount.json`.

    Upright capture  robot hung upright and still  ->  up_chip = the measured specific force.
    Tilt sequence    forward, left, right, back    ->  the heading of the mount about the vertical.

The tilts are not optional bookkeeping: **gravity fixes only two of the three rotation DOF.**
Rotation *about* the vertical is invisible to an accelerometer at rest, and that is exactly the DOF
separating pitch from roll — the axis every balance question on this robot is about.

Each tilt contributes one direction: the HORIZONTAL part of (tilted up - upright up), in chip axes.
Tipping the robot nose-down swings the measured up-vector toward body -X, leaning it to the left
swings it toward body -Y, and so on (TILT_TARGET). Two captures would pin the heading; four make it
checkable. Forward and back must point opposite ways, so must left and right, and the fore-aft pair
must sit 90 deg from the lateral pair ON THE RIGHT-HANDED SIDE. A tilt done in the wrong direction
shows up as a pair that disagrees, instead of as a quietly wrong frame.

Flips. The operator can negate what either pair is taken to mean (the robot's front was the other
end, or left and right were swapped). A rotation cannot flip one horizontal axis alone — that is a
mirror, and no bolted-on HAT is a mirror — so flipping ONE pair of a consistent calibration puts the
two pairs in conflict. The conflict is shown, never averaged: the fore-aft pair then sets the frame
on its own and the lateral pair is reported as disagreeing until it is flipped too.

What the upright capture does NOT do is separate accelerometer bias from mount misalignment — a
robot tilted 1 deg and a sensor with a 17 mg cross-axis bias produce the identical reading. It does
not need to: both are absorbed into the frame in which the reference pose reads roll = pitch = 0.
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

# The tilt sequence, in the order the operator is asked for it.
TILTS = ("fwd", "left", "right", "back")
TILT_LABEL = {"fwd": "forward (nose down)", "left": "left (left side down)",
              "right": "right (right side down)", "back": "backward (nose up)"}
# Where the up-vector's horizontal swing points, in BODY axes, for each tilt. Leaning toward +X
# (nose down) is a positive rotation about +Y, which carries world-up toward body -X.
TILT_TARGET = {"fwd": (-1.0, 0.0, 0.0), "back": (1.0, 0.0, 0.0),
               "left": (0.0, -1.0, 0.0), "right": (0.0, 1.0, 0.0)}
PAIR_OF = {"fwd": "fore_aft", "back": "fore_aft", "left": "lateral", "right": "lateral"}

PAIR_OK_DEG = 15.0              # a pair (or the two pairs' right angle) this far off is flagged


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def rotation_from(up_chip, fwd_chip):
    """Rotation taking a vector from CHIP axes to BODY axes (X forward, Y left, Z up).

    Its rows are the body axes written in chip coordinates. `fwd_chip` is orthogonalised against
    `up_chip` rather than trusted: the tilts are only as clean as the hands that made them."""
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


def nearest_chip_axis(v):
    """('+x' | ... , degrees off it) — which chip axis a chip-frame direction is closest to."""
    v = _unit(v)
    k = int(np.argmax(np.abs(v)))
    name = ("+" if v[k] >= 0 else "-") + "xyz"[k]
    return name, angle_between(v, AXIS_VECTORS[name])


class MountCal:
    """The persisted mount calibration. Thread-safe: the poll thread reads `R` every tick while
    Flask handlers write captures."""

    # A small tilt is the trap in this step. The heading is the HORIZONTAL part of the gravity
    # change, so an unintended sideways lean while tipping rotates it by roughly
    # atan(lean / tilt): at a 4 deg tilt, 1 deg of accidental lean is 14 deg of heading error,
    # while at 15 deg it is under 4. Shallow tilts are accepted but flagged.
    TILT_MIN_DEG = 3.0
    TILT_GOOD_DEG = 8.0

    def __init__(self):
        self._lock = threading.Lock()
        self.up_chip = None             # measured at the upright capture (unit, chip axes)
        # direction -> {"acc": raw mean chip accel (g) | None, "d": legacy horizontal dir | None,
        #               "tilt_deg", "weak", "when", ...}
        self.tilts = {}
        self.flip = {"fore_aft": False, "lateral": False}
        self.captures = {}              # "level" -> {when, spread, mag, n}
        self.reference = "hung on the test rig"
        self.updated = None
        self._R = np.eye(3)             # cached; rebuilt on every mutation
        self._check = {}                # consistency report of the last rebuild
        # Bumped whenever the rotation changes. An attitude filter's state is expressed in the
        # frame it was integrated in, so it is meaningless the instant that frame moves — the poll
        # thread watches this counter and restarts the filter rather than slowly (or never)
        # converging from an arbitrarily wrong attitude.
        self.version = 0

    # ------------------------------------------------------------------ derived
    def _tilt_dir(self, t):
        """The horizontal swing of the up-vector for one tilt, chip axes (unit), or None."""
        if t.get("acc") is not None and self.up_chip is not None:
            up = np.asarray(self.up_chip, float)
            d = np.asarray(t["acc"], float) - up
            h = d - np.dot(d, up) * up
            return _unit(h) if np.linalg.norm(h) > 1e-4 else None
        if t.get("d") is not None:                  # migrated from the old one-tilt calibration
            return _unit(t["d"])
        return None

    def _solve(self):
        """(fwd_chip or None, check dict). Each tilt says where one body axis points in chip axes;
        the fore-aft pair gives +X directly, the lateral pair gives +Y and hence +X = Y x Z."""
        z = _unit(self.up_chip)
        flat = lambda v: v - np.dot(v, z) * z           # noqa: E731
        sgn = {p: (-1.0 if self.flip[p] else 1.0) for p in self.flip}
        dirs = {k: self._tilt_dir(t) for k, t in self.tilts.items()}
        dirs = {k: d for k, d in dirs.items() if d is not None}
        # each tilt's vote for body +X (fore-aft) or body +Y (lateral), in chip axes
        vote = {}
        for k, d in dirs.items():
            tgt = np.asarray(TILT_TARGET[k]) * sgn[PAIR_OF[k]]
            vote[k] = d * (tgt[0] + tgt[1])            # target is +-X or +-Y: undo its sign
        chk = {"pairs": {}, "right_angle_deg": None, "conflict": False, "used": []}
        axes = {}
        for pair, (a, b) in (("fore_aft", ("fwd", "back")), ("lateral", ("left", "right"))):
            have = [k for k in (a, b) if k in vote]
            if not have:
                continue
            if len(have) == 2:
                # both of a pair should vote for the SAME axis; how far apart they are is the check
                dis = angle_between(vote[a], vote[b])
                chk["pairs"][pair] = {"disagree_deg": dis, "n": 2, "excluded": dis > 90.0}
                if dis > 90.0:
                    # one of the two was tilted the wrong way: their sum is noise, and which one
                    # is wrong cannot be told from the pair alone — leave the pair out entirely
                    continue
            else:
                chk["pairs"][pair] = {"disagree_deg": None, "n": 1, "excluded": False}
            axes[pair] = _unit(flat(sum(vote[k] for k in have)))
        x_fa = axes.get("fore_aft")
        x_lat = None if "lateral" not in axes else _unit(np.cross(axes["lateral"], z))
        if x_fa is not None and x_lat is not None:
            # 0 = the pairs are exactly at right angles on the right-handed side; 180 = mirrored
            chk["right_angle_deg"] = angle_between(x_fa, x_lat)
            if chk["right_angle_deg"] > 90.0:
                chk["conflict"] = True                  # never average a mirror into the frame
                chk["used"] = ["fore_aft"]
                return x_fa, chk
            chk["used"] = ["fore_aft", "lateral"]
            return _unit(x_fa + x_lat), chk
        if x_fa is not None:
            chk["used"] = ["fore_aft"]
            return x_fa, chk
        if x_lat is not None:
            chk["used"] = ["lateral"]
            return x_lat, chk
        return None, chk

    def _rebuild(self):
        self.version += 1
        self._check = {}
        if self.up_chip is None:
            self._R = np.eye(3)
            return
        fwd, self._check = self._solve()
        R = None if fwd is None else rotation_from(self.up_chip, fwd)
        self._R = np.eye(3) if R is None else R
        if R is not None:
            # per-tilt residual: where the fitted frame puts each tilt vs where it should be
            sgn = {p: (-1.0 if self.flip[p] else 1.0) for p in self.flip}
            res = {}
            for k, t in self.tilts.items():
                d = self._tilt_dir(t)
                if d is not None:
                    res[k] = angle_between(R @ d, np.asarray(TILT_TARGET[k]) * sgn[PAIR_OF[k]])
            self._check["residual_deg"] = res

    @property
    def R(self):
        """chip -> body rotation; identity while uncalibrated (i.e. values stay in chip axes)."""
        with self._lock:
            return self._R.copy()

    @property
    def R_flat(self):
        """The same rotation as a flat 9-tuple of floats. The IMU poll thread applies it 200 times
        a second in scalar arithmetic, where indexing a numpy array costs more than the multiply."""
        with self._lock:
            R = self._R
            return (float(R[0][0]), float(R[0][1]), float(R[0][2]),
                    float(R[1][0]), float(R[1][1]), float(R[1][2]),
                    float(R[2][0]), float(R[2][1]), float(R[2][2]))

    @property
    def calibrated(self):
        with self._lock:
            return self.up_chip is not None and bool(self._check.get("used"))

    @property
    def conflict(self):
        """True while the fore-aft and lateral pairs describe a mirror (one is mislabelled)."""
        with self._lock:
            return bool(self._check.get("conflict"))

    def tilt_from_upright(self, acc_chip):
        """Degrees between a raw chip-frame accel reading and the upright reference (None before
        the upright capture). Cheap: the sequence calls it on every IMU tick."""
        up = self.up_chip
        if up is None:
            return None
        ax, ay, az = acc_chip
        n = math.sqrt(ax * ax + ay * ay + az * az)
        if n < 1e-6:
            return None
        c = (ax * up[0] + ay * up[1] + az * up[2]) / n
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    # ------------------------------------------------------------------ mutations
    def set_level(self, acc_chip, meta):
        with self._lock:
            self.up_chip = [float(v) for v in _unit(acc_chip)]
            self.captures["level"] = {"when": time.time(), **meta}
            # the migrated one-tilt direction was measured against the OLD upright and cannot be
            # re-derived; raw tilts can, and are (see _tilt_dir)
            self._rebuild()
        self.save()

    def set_tilt(self, which, acc_tilted, meta):
        """One tilt of the sequence. Only the DIRECTION of the swing is used — the tilt angle never
        enters the result, it only says how much the direction is worth."""
        if which not in TILTS:
            return False, f"unknown tilt '{which}' (one of {', '.join(TILTS)})"
        with self._lock:
            if self.up_chip is None:
                return False, "capture the upright reference first"
            up = np.asarray(self.up_chip, float)
            acc = np.asarray(acc_tilted, float)
            d = acc - up
            horiz = d - np.dot(d, up) * up
            tilt_deg = angle_between(acc, up)
            if tilt_deg < self.TILT_MIN_DEG or np.linalg.norm(horiz) < 1e-3:
                return False, (f"only {tilt_deg:.1f}° of tilt — the direction would be mostly "
                               f"noise. Tilt the robot {TILT_LABEL[which]} 10-20° and hold it.")
            weak = tilt_deg < self.TILT_GOOD_DEG
            self.tilts[which] = {"acc": [float(v) for v in acc], "tilt_deg": tilt_deg, "weak": weak,
                                 "when": time.time(), **meta}
            self._rebuild()
        self.save()
        if weak:
            return True, (f"only {tilt_deg:.1f}° of tilt: 1° of unintended lean rotates this "
                          f"direction by ~{math.degrees(math.atan2(1.0, tilt_deg)):.0f}°. "
                          f"Redo it at 10-20°.")
        return True, None

    def set_flip(self, pair, on):
        if pair not in self.flip:
            return False, f"flip must be one of {', '.join(self.flip)}"
        with self._lock:
            self.flip[pair] = bool(on)
            self._rebuild()
        self.save()
        return True, None

    def clear_tilts(self):
        with self._lock:
            self.tilts = {}
            self.flip = {"fore_aft": False, "lateral": False}
            self._rebuild()
        self.save()

    def reset(self):
        with self._lock:
            self.up_chip = None
            self.tilts = {}
            self.flip = {"fore_aft": False, "lateral": False}
            self.captures = {}
            self._rebuild()
        self.save()

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load_or_new(cls, path=None):
        path = path or MOUNT_FILE
        c = cls()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8-sig") as f:
                    d = json.load(f)
                c.up_chip = d.get("up_chip")
                c.tilts = d.get("tilts") or {}
                c.flip = {"fore_aft": False, "lateral": False, **(d.get("flip") or {})}
                c.captures = d.get("captures") or {}
                c.reference = d.get("reference", c.reference)
                c.updated = d.get("updated")
                if not c.tilts:
                    c._migrate(d)
                c._rebuild()
            except (ValueError, OSError) as e:
                print(f"(could not read {path}: {e} — starting with an uncalibrated IMU mount)")
        return c

    def _migrate(self, d):
        """The old format had one nose-down tilt (`fwd_chip` = the forward axis in chip axes) or a
        declared forward chip axis. Either becomes a forward tilt, so an existing calibration keeps
        working until the sequence is re-run."""
        old = self.captures.pop("forward", {})
        if d.get("fwd_chip"):
            fwd = d["fwd_chip"]
            src = "old single nose-down capture"
        elif d.get("fwd_declared") in AXIS_VECTORS:
            fwd = AXIS_VECTORS[d["fwd_declared"]]
            src = f"old declared forward axis {d['fwd_declared']}"
        else:
            return
        self.tilts["fwd"] = {"acc": None, "d": [-float(v) for v in fwd], "legacy": src,
                             "tilt_deg": old.get("tilt_deg"), "weak": bool(old.get("weak")),
                             "when": old.get("when")}

    def save(self, path=None):
        path = path or MOUNT_FILE
        with self._lock:
            self.updated = time.time()
            d = {"up_chip": self.up_chip, "tilts": self.tilts, "flip": self.flip,
                 "captures": self.captures, "reference": self.reference,
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
            chk = dict(self._check)
            cal = self.up_chip is not None and bool(chk.get("used"))
            tilts = {k: {kk: vv for kk, vv in t.items() if kk not in ("acc", "d")}
                     for k, t in self.tilts.items()}
            snap = {
                "calibrated": cal,
                "up_chip": self.up_chip, "tilts": tilts, "flip": dict(self.flip),
                "captures": self.captures, "check": chk,
                "R_chip_to_body": [[round(float(v), 6) for v in row] for row in R],
                "reference": self.reference, "updated": self.updated,
            }
        if cal:
            # body X / Y written as the chip axis they sit nearest to — the line an operator can
            # check against the HAT's silkscreen
            snap["axes"] = {"fwd": nearest_chip_axis(R[0]), "left": nearest_chip_axis(R[1]),
                            "up": nearest_chip_axis(R[2])}
        return snap
