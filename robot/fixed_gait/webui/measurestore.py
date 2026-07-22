"""System-ID measurement runs: the high-rate (200 Hz) capture the daemon accumulates during a
MEASURE excitation, persisted for the offline estimator (robot/identification/).

One start->stop excitation = one run = one `.npz` (+ the same metadata mirrored into a sidecar
`.json` so a run is human-inspectable without numpy). All angles are NORMALIZED degrees (zero pose =
0), the same frame the rest of the web UI uses; the estimator maps them to model radians via the
saved calibration + the FK map.

Array layout (columns follow paths.MOTOR_NAMES = right.abd,right.cam,right.thigh,left.*):
    t         [N]      seconds since the run started (daemon clock; jitter is real, keep it)
    cmd_norm  [N, 6]   commanded normalized deg (what SET_POS asked for, drive-frame back-converted)
    pos_norm  [N, 6]   measured normalized deg
    pos_raw   [N, 6]   measured raw motor deg
    spd       [N, 6]   measured speed (ERPM, as the servo reports it)
    cur       [N, 6]   measured q-axis current (A) — the torque proxy (tau = Kt * cur)
The metadata dict carries: leg, profile, the excitation spec, base pose, calibration offsets/signs,
the drive PID gains + weighed masses + any prior Kt in effect at capture, and timestamps.
"""
import io
import json
import os
import time

import numpy as np

import paths

ARRAYS = ("t", "cmd_norm", "pos_norm", "pos_raw", "spd", "cur")


def _path(name):
    keep = "".join(c for c in name if c.isalnum() or c in "-_ .").strip().replace(" ", "_")
    if not keep:
        raise ValueError("empty measurement name")
    if not keep.endswith(".npz"):
        keep += ".npz"
    return os.path.join(paths.MEASURE_DIR, keep)


def list_files():
    return sorted(f for f in os.listdir(paths.MEASURE_DIR) if f.endswith(".npz"))


def save(name, run, meta):
    """run: {array_name: np.ndarray}; meta: JSON-able dict. Writes name.npz + name.json."""
    for k in ARRAYS:
        if k not in run:
            raise ValueError(f"measurement run missing '{k}' array")
    n = len(run["t"])
    if n < 5:
        raise ValueError(f"measurement too short ({n} samples) — nothing to save")
    p = _path(name)
    meta = dict(meta, name=os.path.basename(p), n_samples=int(n),
                duration_s=float(run["t"][-1] - run["t"][0]) if n else 0.0,
                motor_names=list(paths.MOTOR_NAMES), saved=time.strftime("%Y-%m-%dT%H:%M:%S"))
    np.savez(p, meta_json=json.dumps(meta), **{k: np.asarray(run[k], np.float32) for k in ARRAYS})
    with open(os.path.splitext(p)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load(name):
    z = np.load(_path(name), allow_pickle=False)
    run = {k: z[k] for k in ARRAYS}
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    return run, meta


def summary(name):
    """Lightweight listing entry (reads the JSON sidecar, not the arrays when possible)."""
    side = os.path.splitext(_path(name))[0] + ".json"
    if os.path.exists(side):
        with open(side, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    _, meta = load(name)
    return meta


def list_summaries():
    out = []
    for f in list_files():
        try:
            m = summary(f)
        except (ValueError, OSError, KeyError):
            m = {"name": f}
        out.append({"name": f, "leg": m.get("leg"), "profile": m.get("profile"),
                    "n_samples": m.get("n_samples"), "duration_s": m.get("duration_s"),
                    "saved": m.get("saved")})
    return out


def delete(name):
    p = _path(name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"measurement '{name}' not found")
    os.remove(p)
    side = os.path.splitext(p)[0] + ".json"
    if os.path.exists(side):
        os.remove(side)
    return os.path.basename(p)


def export_bytes(name):
    with open(_path(name), "rb") as f:
        return f.read(), os.path.basename(_path(name))
