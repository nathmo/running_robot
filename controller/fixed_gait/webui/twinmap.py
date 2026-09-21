"""Sign map for the browser-side digital twin (static/twin3d.js).

    mjcf_qpos_rad = radians(sign * norm_deg)

No offset term, on purpose: the twin renders dash-01CAD/homing/SpiderBotInitPos, the pose in which
the drives are zeroed, so normalized 0 deg IS MJCF qpos 0 on every joint. What a re-zero changes is
the offset, and the calibration wizard already owns that; the axis direction does not flip between
zeroings (calibration.DEFAULT_SIGNS), so a sign set once stays right.

Defaults are DERIVED, not fitted: the calibration wizard's direction check fixes what + means
physically, the same on both legs (index.html step 2: thigh + swings the leg FORWARD, cam + moves
the crank DOWN, abduction + lifts the foot = outward), and the MJCF's joint axes turn that into a
qpos sign. x is forward, y is left, z up.
  left.thigh  -1   axis +y: +q swings the knee BACKWARD            right.thigh +1   axis -y
  left.cam    +1   axis +y: +q moves the crank pin down (and fwd)  right.cam   -1   axis -y
  left.abd    +1   axis +x: +q moves the foot to +y = outward      right.abd   -1   axis +x = inward
The left leg agrees with the sign map fklut fitted against recorded workspace data (cam +1,
thigh -1, data/model_map.json) and abd with deploy_map.json (L +1 / R -1, verified 2026-08-29).
Right is the mirror of left: its sagittal axes are -y, so the SAME physical motion is the opposite
qpos. Stored in data/twin_map.json; the panel's selects write it.
"""
import json
import os

import paths

TWIN_MAP_FILE = os.path.join(paths.DATA, "twin_map.json")
DEFAULT_SIGNS = {"left.abd": 1, "left.cam": 1, "left.thigh": -1,
                 "right.abd": -1, "right.cam": -1, "right.thigh": 1}


def load(path=None):
    path = path or TWIN_MAP_FILE
    signs = dict(DEFAULT_SIGNS)
    try:
        with open(path) as f:
            saved = json.load(f).get("signs", {})
        for k, v in saved.items():
            if k in signs:
                signs[k] = 1 if float(v) >= 0 else -1
    except (OSError, ValueError, AttributeError):
        pass                                   # missing / corrupt file -> defaults
    return {"signs": signs, "defaults": dict(DEFAULT_SIGNS)}


def save(signs, path=None):
    """Merge `signs` ({motor: ±1}) into the stored map. Raises ValueError on an unknown motor or a
    value that is not ±1, so a typo never silently becomes a default."""
    path = path or TWIN_MAP_FILE
    cur = load(path)["signs"]
    for k, v in (signs or {}).items():
        if k not in cur:
            raise ValueError(f"unknown motor '{k}'")
        try:
            v = int(v)
        except (TypeError, ValueError):
            raise ValueError(f"{k}: sign must be +1 or -1")
        if v not in (1, -1):
            raise ValueError(f"{k}: sign must be +1 or -1")
        cur[k] = v
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"signs": cur}, f, indent=2)
    os.replace(tmp, path)
    return load(path)
