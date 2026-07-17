"""Trajectory (gait) files for the web UI — record, hand-draw, import/export, list.

Files use the standard fixed_gait/trajectory.py per-leg npz format, so the CLI tools
(play_trajectory.py, view_trajectory.py) can open web-made gaits unchanged — but all angles are
in the NORMALIZED zero-pose frame. Hand-drawn paths are turned into a synthetic 'take' and pushed
through the SAME traj.process pipeline as hand-recorded takes (FFT smoothing, loop closing,
hip-turning-point re-timing), so drawn and taught gaits behave identically downstream.
"""
import io
import os

import numpy as np

import paths
import trajectory as traj                  # fixed_gait/

DRAW_TAKE_SECONDS = 4.0


def _path(name):
    keep = "".join(c for c in name if c.isalnum() or c in "-_ .").strip().replace(" ", "_")
    if not keep:
        raise ValueError("empty trajectory name")
    if not keep.endswith(".npz"):
        keep += ".npz"
    return os.path.join(paths.TRAJ_DIR, keep)


def list_files():
    return sorted(f for f in os.listdir(paths.TRAJ_DIR) if f.endswith(".npz"))


def load(name):
    return traj.load(_path(name))          # raises ValueError on the old single-canonical format


def save(name, data):
    p = _path(name)
    traj.save(p, data)
    return p


def delete(name):
    p = _path(name)                        # _path sanitizes the name (no path traversal)
    if not os.path.exists(p):
        raise FileNotFoundError(f"gait '{name}' not found")
    os.remove(p)
    return os.path.basename(p)


def data_to_json(data, fk=None):
    """JSON-ready trajectory (+ EE-projected paths when the FK LUT side is verified)."""
    out = dict(N=int(data["N"]), split=float(data["split"]))
    ph = np.linspace(0, 1, int(data["N"]), endpoint=False)
    for side in ("right", "left"):
        cal = data.get(side)
        if cal is None:
            out[side] = None
            continue
        rec = np.array([traj.reconstruct(data, side, p) for p in ph])   # [N,3] abd,cam,hip
        out[side] = dict(
            canonical=np.round(cal["canonical"], 3).tolist(),
            center=np.round(np.asarray(cal["center"], float), 2).tolist(),
            phase_shift=float(cal["phase_shift"]),
            lo=np.round(np.asarray(cal["lo"], float), 2).tolist(),
            hi=np.round(np.asarray(cal["hi"], float), 2).tolist(),
            path=np.round(rec[:, 1:3], 2).tolist(),                     # (cam, thigh) loop
            abd_hold=float(cal["abduction_hold"]),
        )
        if fk is not None and fk.side_verified(side):
            ee = fk.project_ee(side, rec[:, 1], rec[:, 2])
            out[side]["ee_path"] = [None if not np.isfinite(p).all() else
                                    [round(float(p[0]), 4), round(float(p[1]), 4)] for p in ee]
    return out


def finish_recording(rec, name, harmonics=8, split=0.5, left_phase=0.5):
    """Process accumulated web-recorded takes (normalized frame) into a saved trajectory."""
    right, left = rec["takes"]["right"], rec["takes"]["left"]
    if not right and not left:
        raise ValueError("no takes recorded yet")
    data = traj.process(right, left,
                        right_center=rec["centers"]["right"], left_center=rec["centers"]["left"],
                        harmonics=harmonics, split=split, left_phase=left_phase, verbose=False)
    save(name, data)
    return data


def draw_to_trajectory(name, leg, points, abd_hold=0.0, center=None,
                       harmonics=8, split=0.5, left_phase=0.5, reverse=False):
    """Hand-drawn (cam, thigh) polyline (normalized deg) -> synthetic take -> traj.process.

    Merges into an existing file of the same name (the OTHER leg's data is kept), so a gait can be
    drawn one leg at a time.
    """
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 8:
        raise ValueError("need a drawn path of at least 8 (cam, thigh) points")
    if reverse:
        pts = pts[::-1]
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.vstack([pts, pts[0]])                     # close the loop explicitly
    n = len(pts)
    t = np.linspace(0.0, DRAW_TAKE_SECONDS, n)
    pos = np.column_stack([np.full(n, float(abd_hold)), pts[:, 0], pts[:, 1]])
    take = (t, pos)
    ctr = None if center is None else np.asarray([abd_hold, center[0], center[1]], float)

    kw = dict(harmonics=harmonics, split=split, left_phase=left_phase, verbose=False)
    if leg == "right":
        new = traj.process([take], [], right_center=ctr, **kw)
    else:
        new = traj.process([], [take], left_center=ctr, **kw)

    # merge with the existing other-leg calibration, if any
    other = "left" if leg == "right" else "right"
    if os.path.exists(_path(name)):
        try:
            old = load(name)
            if old.get(other) is not None:
                new[other] = old[other]
        except ValueError:
            pass                                           # old-format file: overwrite entirely
    save(name, new)
    return new


def mirror(name, from_leg, to_leg, left_phase=0.5):
    """Copy one leg's gait onto the other within a trajectory file (normalized frames coincide
    between legs). The destination keeps the conventional dephasing: left plays `left_phase`
    (default 0.5 = 180°) ahead, right plays at 0."""
    if from_leg == to_leg:
        raise ValueError("source and destination leg are the same")
    data = load(name)
    src = data.get(from_leg)
    if src is None:
        raise ValueError(f"{name} has no {from_leg}-leg data to copy")
    dst = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in src.items()}
    dst["phase_shift"] = float(left_phase) if to_leg == "left" else 0.0
    data[to_leg] = dst
    save(name, data)
    return data


def export_bytes(name):
    with open(_path(name), "rb") as f:
        return f.read(), os.path.basename(_path(name))


def import_bytes(blob, filename):
    """Import a trajectory npz. It must be loadable by traj.load (new per-leg format).
    NOTE: legacy files recorded before the normalization scheme are in RAW degrees of a dead
    session — only import them if that session had motors zeroed at the zero pose."""
    buf = io.BytesIO(blob)
    z = np.load(buf, allow_pickle=False)
    if "canonical" in z.files and "right_canonical" not in z.files:
        raise ValueError(f"{filename} is an OLD-format trajectory (single shared canonical) — "
                         "re-record it; it cannot be converted")
    name = os.path.splitext(os.path.basename(filename))[0]
    p = _path(name)
    with open(p, "wb") as f:
        f.write(blob)
    try:
        load(name)                                         # validate through traj.load
    except Exception:
        os.remove(p)
        raise
    return name + ".npz"
