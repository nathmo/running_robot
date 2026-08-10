"""Session calibration: per-motor encoder offset + sign, and the normalized<->raw conversion.

The motors report a RANDOM raw angle (and possibly inverted direction) after every power cycle, so
nothing recorded in raw degrees survives a reboot. The web UI therefore works in a NORMALIZED
frame anchored to the URDF/CAD zero pose:

    norm_deg = sign * (raw_deg - offset_deg)        raw_deg = offset_deg + sign * norm_deg

offset = the raw angle captured while the user physically holds the robot in the zero pose;
sign   = confirmed per motor in the guided direction check (flipped in the UI if wrong).

Everything above the daemon's CAN boundary (workspace grids, gait files, plots, sliders) lives in
normalized degrees; conversion happens only right after parse_status and right before set_pos.

Legacy files from fixed_gait/ (calibrate_workspace.py / joint_limits.npz) store RAW degrees plus
the raw zero captured in-session with the 'z' key — convert_legacy_limits() translates them into
the normalized frame so yesterday's workspace keeps working after any number of reboots.
"""
import json
import os
import threading
import time

import numpy as np

import paths
import blackbox

# Measured-correct direction-check signs (booth calibration, 2026-07-14): saves re-flipping every
# card each session. Physical motor wiring can still invert across a power cycle — the wizard
# always makes the user re-confirm each sign, this is only the starting guess.
DEFAULT_SIGNS = {
    "right.abd": 1.0, "right.cam": 1.0, "right.thigh": -1.0,
    "left.abd": -1.0, "left.cam": -1.0, "left.thigh": 1.0,
}


class Calibration:
    def __init__(self):
        self._lock = threading.Lock()
        self.offsets = {n: 0.0 for n in paths.MOTOR_NAMES}
        self.signs = {n: DEFAULT_SIGNS.get(n, 1.0) for n in paths.MOTOR_NAMES}
        self.confirmed = {n: False for n in paths.MOTOR_NAMES}
        self.stage = "none"                    # none | zero_set | complete
        self.created = None
        self.restored_from_disk = False        # UI shows a "re-zero if unsure" banner when True
        # RAW-AT-REST FINGERPRINT: exactly the pos_raw of all six at the moment of the last
        # successful zero capture. It is numerically the same as `offsets`, but it is kept and
        # persisted separately because it means something different: offsets is a conversion
        # constant, this is EVIDENCE about where the encoders read when the robot was known to be
        # in the zero pose. daemon._premove_guard compares live pos_raw against it — that is the
        # check that would have caught 2026-08-10. `zero_epoch` bumps on every capture so the
        # daemon can tell "we have not moved since this zero" from "we have".
        self.zero_raw = {}
        self.zero_epoch = 0

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load_or_new(cls, path=paths.CALIB_FILE):
        c = cls()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8-sig") as f:   # -sig: tolerate a BOM
                    d = json.load(f)
                for n, m in d.get("motors", {}).items():
                    if n in c.offsets:
                        c.offsets[n] = float(m.get("offset_deg", 0.0))
                        c.signs[n] = 1.0 if float(m.get("sign", 1)) >= 0 else -1.0
                        c.confirmed[n] = bool(m.get("confirmed", False))
                c.stage = d.get("stage", "none")
                c.created = d.get("created")
                c.restored_from_disk = c.stage != "none"
                c.zero_epoch = int(d.get("zero_epoch", 0))
                c.zero_raw = {n: float(v) for n, v in (d.get("zero_raw") or {}).items()
                              if n in c.offsets}
                if not c.zero_raw and c.stage != "none":
                    # a file written before the fingerprint existed: offsets ARE that raw pose
                    c.zero_raw = dict(c.offsets)
            except (ValueError, OSError) as e:
                print(f"(could not read {path}: {e} — starting uncalibrated)")
        blackbox.log_event("calib.load", stage=c.stage, created=c.created,
                           zero_epoch=c.zero_epoch, zero_raw=c.zero_raw,
                           offsets=c.offsets, signs=c.signs,
                           restored_from_disk=c.restored_from_disk,
                           note="restored from disk: the drives re-randomise their raw origin on "
                                "every power cycle, so this zero is only valid if the boards have "
                                "not been power-cycled since it was captured")
        return c

    def save(self, path=paths.CALIB_FILE):
        with self._lock:
            d = {
                "created": self.created,
                "stage": self.stage,
                "zero_epoch": self.zero_epoch,
                "zero_raw": dict(self.zero_raw),
                "motors": {n: {"offset_deg": self.offsets[n], "sign": int(self.signs[n]),
                               "confirmed": self.confirmed[n]} for n in paths.MOTOR_NAMES},
            }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        os.replace(tmp, path)
        blackbox.log_event("calib.save", stage=d["stage"], created=d["created"],
                           zero_epoch=d["zero_epoch"], path=os.path.basename(path))
        blackbox.note_config_change("calibration")

    # ------------------------------------------------------------------ conversion
    @property
    def complete(self):
        return self.stage == "complete"

    def norm(self, name, raw_deg):
        return self.signs[name] * (raw_deg - self.offsets[name])

    def raw(self, name, norm_deg):
        return self.offsets[name] + self.signs[name] * norm_deg

    def norm_array(self, raw_by_index):
        """Vectorized: raw array in MOTOR_NAMES order -> normalized array."""
        off = np.array([self.offsets[n] for n in paths.MOTOR_NAMES])
        sgn = np.array([self.signs[n] for n in paths.MOTOR_NAMES])
        return sgn * (np.asarray(raw_by_index, float) - off)

    # ------------------------------------------------------------------ wizard steps
    def set_zero(self, raw_positions):
        """raw_positions: {motor_name: raw_deg} for ALL motors (refuse partial capture)."""
        missing = [n for n in paths.MOTOR_NAMES if raw_positions.get(n) is None]
        if missing:
            blackbox.log_event("calib.set_zero.refused", missing=missing,
                               raw=raw_positions, reason="partial capture")
            return False, f"no position from {', '.join(missing)} — cannot set zero"
        before = dict(self.offsets)
        before_raw = dict(self.zero_raw)
        with self._lock:
            for n in paths.MOTOR_NAMES:
                self.offsets[n] = float(raw_positions[n])
                self.confirmed[n] = False
            self.zero_raw = {n: float(raw_positions[n]) for n in paths.MOTOR_NAMES}
            self.zero_epoch += 1
            self.stage = "zero_set"
            self.created = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.restored_from_disk = False
        self.save()
        blackbox.log_event(
            "calib.set_zero", zero_epoch=self.zero_epoch, created=self.created,
            offsets_before=before, offsets_after=dict(self.offsets),
            raw_at_last_zero_before=before_raw, raw_captured=dict(self.zero_raw),
            moved_since_last_zero={n: round(float(raw_positions[n]) - before_raw[n], 3)
                                   for n in paths.MOTOR_NAMES if n in before_raw},
            note="raw_captured IS the raw-at-rest fingerprint the pre-move guard compares against")
        blackbox.trigger_dump("calibration_zero_capture")
        return True, ""

    def set_sign(self, name, sign):
        if name not in self.signs:
            return False, f"unknown motor {name}"
        with self._lock:
            before = self.signs[name]
            self.signs[name] = 1.0 if sign >= 0 else -1.0
            after = self.signs[name]
            self.confirmed[name] = False
        self.save()
        blackbox.log_event("calib.set_sign", motor=name, before=before, after=after)
        if before != after:
            blackbox.trigger_dump("calibration_sign_flip")
        return True, ""

    def confirm(self, name):
        if name not in self.confirmed:
            return False, f"unknown motor {name}"
        if self.stage == "none":
            return False, "set zero first"
        with self._lock:
            self.confirmed[name] = True
            if all(self.confirmed.values()):
                self.stage = "complete"
            stage = self.stage
        self.save()
        blackbox.log_event("calib.confirm", motor=name, stage=stage)
        return True, ""

    def complete_now(self):
        if self.stage == "none":
            return False, "set zero first"
        not_conf = [n for n in paths.MOTOR_NAMES if not self.confirmed[n]]
        if not_conf:
            return False, f"direction not confirmed for: {', '.join(not_conf)}"
        with self._lock:
            self.stage = "complete"
        self.save()
        blackbox.log_event("calib.complete", zero_epoch=self.zero_epoch,
                           raw_at_last_zero=self.raw_at_rest(), signs=dict(self.signs))
        return True, ""

    def reset(self):
        before = dict(self.offsets)
        with self._lock:
            self.offsets = {n: 0.0 for n in paths.MOTOR_NAMES}
            self.signs = {n: DEFAULT_SIGNS.get(n, 1.0) for n in paths.MOTOR_NAMES}
            self.confirmed = {n: False for n in paths.MOTOR_NAMES}
            self.stage = "none"
            self.restored_from_disk = False
            self.zero_raw = {}
            self.zero_epoch += 1
        self.save()
        blackbox.log_event("calib.reset", offsets_before=before, zero_epoch=self.zero_epoch)

    # ------------------------------------------------------------------ raw-at-rest comparison
    def raw_at_rest(self):
        """{name: pos_raw at the last successful zero capture}, or {} if never zeroed."""
        with self._lock:
            return dict(self.zero_raw)

    def compare_raw(self, raw_now):
        """Compare live pos_raw against the raw-at-rest fingerprint.

        Returns {name: {"then", "now", "delta"}} for every motor present in both. `delta` is only
        meaningful while the robot has NOT been commanded to move since that capture — the daemon
        decides that (see daemon._premove_guard); this method just does the arithmetic so the guard,
        the snapshot and any postmortem all read the same numbers.
        """
        then = self.raw_at_rest()
        out = {}
        for n in paths.MOTOR_NAMES:
            a, b = then.get(n), raw_now.get(n)
            if a is None or b is None:
                continue
            out[n] = {"then": round(float(a), 3), "now": round(float(b), 3),
                      "delta": round(float(b) - float(a), 3)}
        return out

    def snapshot(self):
        with self._lock:
            return {
                "stage": self.stage,
                "created": self.created,
                "restored_from_disk": self.restored_from_disk,
                "zero_epoch": self.zero_epoch,
                "raw_at_last_zero": {n: round(v, 3) for n, v in self.zero_raw.items()},
                "motors": {n: {"offset_deg": round(self.offsets[n], 2),
                               "sign": int(self.signs[n]),
                               "confirmed": self.confirmed[n]} for n in paths.MOTOR_NAMES},
            }


# ===================================================================== legacy conversion
def _convert_axis(origin, n_cells, res, zero, sign):
    """Normalized origin of a grid axis whose raw cell k spans [origin+k*res, origin+(k+1)*res].
    With norm = sign*(raw - zero):  sign=+1 -> origin-zero (no flip);
    sign=-1 -> cells reverse order and the new origin is -(origin + n*res - zero)."""
    if sign >= 0:
        return origin - zero, False
    return -(origin + n_cells * res - zero), True


def convert_legacy_limits(npz, signs=None):
    """Convert a RAW-frame joint_limits npz (calibrate_workspace.py output) into normalized-frame
    per-leg dicts using each leg's stored in-session zero.

    npz: an opened np.load() result. signs: optional {leg: {"abd":±1,"cam":±1,"thigh":±1}} if the
    recording session's motor directions differed from the normalized convention (default all +1).
    Returns {leg: dict} in the workspace.WorkspaceStore leg format. Raises ValueError when a leg
    never captured its zero (then it cannot be normalized).
    """
    signs = signs or {}
    legs = {}
    for leg in ("left", "right"):
        if f"{leg}_abd_safe_min" not in npz.files:
            continue
        s = {**{"abd": 1.0, "cam": 1.0, "thigh": 1.0}, **signs.get(leg, {})}
        abd_zero = float(npz[f"{leg}_abd_zero"])
        knee_zero = np.asarray(npz[f"{leg}_knee_zero"], float)
        if np.isnan(abd_zero) or np.isnan(knee_zero).any():
            raise ValueError(f"{leg} leg: this legacy file never captured its zero pose "
                             f"('z' during calibrate_workspace) — cannot normalize it")

        a, b = s["abd"] * (float(npz[f"{leg}_abd_observed_min"]) - abd_zero), \
               s["abd"] * (float(npz[f"{leg}_abd_observed_max"]) - abd_zero)
        obs = (min(a, b), max(a, b))
        a, b = s["abd"] * (float(npz[f"{leg}_abd_safe_min"]) - abd_zero), \
               s["abd"] * (float(npz[f"{leg}_abd_safe_max"]) - abd_zero)
        safe = (min(a, b), max(a, b))

        grid = npz[f"{leg}_knee_grid"].astype(bool)
        res = float(npz[f"{leg}_knee_grid_deg"])
        cam_o, flip_c = _convert_axis(float(npz[f"{leg}_knee_cam_origin"]), grid.shape[0], res,
                                      float(knee_zero[0]), s["cam"])
        th_o, flip_t = _convert_axis(float(npz[f"{leg}_knee_thigh_origin"]), grid.shape[1], res,
                                     float(knee_zero[1]), s["thigh"])
        if flip_c:
            grid = grid[::-1, :]
        if flip_t:
            grid = grid[:, ::-1]

        legs[leg] = dict(abd_observed=obs, abd_safe=safe,
                         knee_grid=np.ascontiguousarray(grid),
                         knee_cam_origin=cam_o, knee_thigh_origin=th_o, knee_grid_deg=res,
                         samples=None)
    if not legs:
        raise ValueError("file contains no leg calibration (not a joint_limits npz?)")
    return legs


def convert_legacy_raw_segments(npz, signs=None):
    """Convert a RAW workspace-sweep file (calibrate_workspace.save_raw: p0..pn + zero) into
    normalized segments {leg, segments:[...]} for re-processing. Needs the stored zero."""
    leg = str(npz["leg"])
    if not int(npz["has_zero"]):
        raise ValueError(f"{leg} raw sweep file has no zero pose captured — cannot normalize")
    zero = np.asarray(npz["zero"], float)          # [abd, cam, thigh] raw
    s = signs or {"abd": 1.0, "cam": 1.0, "thigh": 1.0}
    sgn = np.array([s["abd"], s["cam"], s["thigh"]])
    segments = [sgn * (npz[f"p{i}"] - zero) for i in range(int(npz["n"]))]
    return leg, segments
