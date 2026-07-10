"""Safe-workspace store for the web UI — NORMALIZED zero-pose frame everywhere.

Wraps the proven building blocks of fixed_gait/calibrate_workspace.py (grid rasterize + dilate +
erode) and fixed_gait/joint_limits.py (JointLimits check object), but keeps all angles in the
normalized frame (zero pose == 0), so saved workspaces survive motor power cycles.

Per-leg data format (the `legs` dict):
    abd_observed (lo, hi)   abd_safe (lo, hi)          # normalized deg
    knee_grid bool[nc, nt]  knee_cam_origin  knee_thigh_origin  knee_grid_deg
    samples float[N, 3] | None                          # backdriven scatter (abd, cam, thigh)
"""
import base64
import io
import os
import threading

import numpy as np

import paths
import calibration as calib_mod
import calibrate_workspace as cw          # fixed_gait/ — grid morphology + processing
import joint_limits                        # fixed_gait/ — the check object

MAX_SCATTER = 4000


class WorkspaceStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.legs = {}                    # leg -> dict (see module docstring)
        self.source = None                # filename this came from (display only)
        self.limits = None                # joint_limits.JointLimits in NORMALIZED frame
        self._active_path = os.path.join(paths.DATA, "workspace_active.npz")
        if os.path.exists(self._active_path):
            try:
                with np.load(self._active_path) as z:
                    self._set_legs(_legs_from_npz(z), str(z["source"]) if "source" in z.files
                                   else "workspace_active.npz")
                print(f"restored active workspace ({sorted(self.legs)}) from {self._active_path}")
            except (ValueError, OSError, KeyError) as e:
                print(f"(could not restore active workspace: {e})")

    # ------------------------------------------------------------------ core state
    def _rebuild_limits(self):
        """JointLimits over normalized angles. zero == 0 by construction of the frame."""
        legs = {}
        for leg, d in self.legs.items():
            legs[leg] = dict(
                abd_safe=tuple(d["abd_safe"]), abd_observed=tuple(d["abd_observed"]),
                abd_zero=0.0,
                knee_grid=d["knee_grid"], knee_cam_origin=d["knee_cam_origin"],
                knee_thigh_origin=d["knee_thigh_origin"], knee_grid_deg=d["knee_grid_deg"],
                knee_zero=np.zeros(2),
            )
        self.limits = joint_limits.JointLimits(legs) if legs else None

    def _set_legs(self, legs, source):
        with self._lock:
            self.legs.update(legs)
            self.source = source
            self._rebuild_limits()

    def _persist_active(self):
        try:
            np.savez(self._active_path, **_legs_to_npz(self.legs, self.source or "active"))
        except OSError as e:
            print(f"(could not persist active workspace: {e})")

    # ------------------------------------------------------------------ mutations
    def apply_grid(self, leg, grid, cam_origin, thigh_origin, grid_deg):
        if leg not in self.legs:
            self.legs[leg] = dict(abd_observed=(-1.0, 1.0), abd_safe=(-1.0, 1.0), samples=None)
        with self._lock:
            d = self.legs[leg]
            d["knee_grid"] = np.ascontiguousarray(grid.astype(bool))
            d["knee_cam_origin"] = float(cam_origin)
            d["knee_thigh_origin"] = float(thigh_origin)
            d["knee_grid_deg"] = float(grid_deg)
            self._rebuild_limits()
        self._persist_active()

    def apply_abduction(self, leg, safe_lo, safe_hi):
        if leg not in self.legs:
            return False, f"no workspace for {leg} leg yet"
        with self._lock:
            self.legs[leg]["abd_safe"] = (float(safe_lo), float(safe_hi))
            self._rebuild_limits()
        self._persist_active()
        return True, ""

    def process_segments(self, leg, segments, margin_deg=3.0, grid_deg=1.0, dilate_deg=2.0):
        """Build a leg workspace from normalized backdriven segments — the exact pipeline of
        calibrate_workspace.process_and_export (:227-277), minus file/plot I/O."""
        samples = np.concatenate(segments, axis=0)         # [N,3] abd, cam, thigh (normalized)
        abd = samples[:, 0]
        lo, hi = float(abd.min()), float(abd.max())
        safe_lo, safe_hi = lo + margin_deg, hi - margin_deg
        warn = ""
        if safe_lo >= safe_hi:
            warn = (f"margin {margin_deg:g} deg empties the abduction range "
                    f"[{lo:.1f},{hi:.1f}] — reduce it or sweep wider")
        knee = cw._knee_grid(samples[:, 1], samples[:, 2], grid_deg, dilate_deg, margin_deg)
        if not knee["safe_grid"].any():
            warn = (warn + "; " if warn else "") + \
                "eroded knee safe-region is EMPTY (margin/dilate too aggressive or too few samples)"
        legs = {leg: dict(abd_observed=(lo, hi), abd_safe=(safe_lo, safe_hi),
                          knee_grid=knee["safe_grid"],
                          knee_cam_origin=knee["cam_origin"], knee_thigh_origin=knee["thigh_origin"],
                          knee_grid_deg=knee["grid_deg"],
                          samples=samples[:: max(1, len(samples) // (MAX_SCATTER * 4))])}
        self._set_legs(legs, f"web sweep ({leg})")
        self._persist_active()
        return warn

    # ------------------------------------------------------------------ files
    def save(self, name):
        safe = _safe_name(name)
        path = os.path.join(paths.WORKSPACE_DIR, safe + ".npz")
        np.savez(path, **_legs_to_npz(self.legs, safe))
        with self._lock:
            self.source = safe + ".npz"
        self._persist_active()
        return path

    def export_bytes(self, name=None):
        if name:
            path = os.path.join(paths.WORKSPACE_DIR, _safe_name(name) + ".npz")
            with open(path, "rb") as f:
                return f.read(), os.path.basename(path)
        buf = io.BytesIO()
        np.savez(buf, **_legs_to_npz(self.legs, self.source or "export"))
        return buf.getvalue(), "workspace_export.npz"

    def import_bytes(self, blob, filename, legacy_signs=None):
        """Auto-detect and import: normalized webui workspace / legacy joint_limits.npz /
        legacy raw sweep (raw_{leg}.npz with zero). Returns a description string."""
        z = np.load(io.BytesIO(blob), allow_pickle=False)
        if "frame" in z.files:                                  # webui normalized format
            self._set_legs(_legs_from_npz(z), filename)
            self._persist_active()
            return f"imported normalized workspace ({sorted(self.legs)})"
        if any(k.endswith("_abd_safe_min") for k in z.files):    # legacy joint_limits.npz
            legs = calib_mod.convert_legacy_limits(z, signs=legacy_signs)
            self._set_legs(legs, filename + " (legacy, normalized via stored zero)")
            self._persist_active()
            return f"converted legacy joint_limits ({sorted(legs)}) via stored zero pose"
        if "has_zero" in z.files and "p0" in z.files:            # legacy raw sweep segments
            leg, segments = calib_mod.convert_legacy_raw_segments(z, signs=legacy_signs and
                                                                  legacy_signs.get("any"))
            warn = self.process_segments(leg, segments)
            return f"rebuilt {leg} workspace from legacy raw sweep" + (f" ({warn})" if warn else "")
        raise ValueError("unrecognized npz: not a webui workspace, joint_limits, or raw sweep file")

    def list_files(self):
        return sorted(f for f in os.listdir(paths.WORKSPACE_DIR) if f.endswith(".npz"))

    # ------------------------------------------------------------------ serialization for the UI
    def leg_json(self, leg):
        d = self.legs.get(leg)
        if d is None:
            return None
        grid = d["knee_grid"]
        out = dict(
            abd_observed=list(np.round(d["abd_observed"], 2)),
            abd_safe=list(np.round(d["abd_safe"], 2)),
            knee=dict(cam_origin=d["knee_cam_origin"], thigh_origin=d["knee_thigh_origin"],
                      res_deg=d["knee_grid_deg"], shape=list(grid.shape),
                      grid_b64=base64.b64encode(np.packbits(grid)).decode()),
        )
        s = d.get("samples")
        if s is not None and len(s):
            step = max(1, len(s) // MAX_SCATTER)
            out["samples"] = np.round(s[::step, 1:3], 2).tolist()      # (cam, thigh) scatter
            out["abd_samples"] = np.round(s[::step, 0], 2).tolist()
        return out


# ===================================================================== npz (de)serialization
def _safe_name(name):
    keep = "".join(c for c in name if c.isalnum() or c in "-_ .").strip()
    return keep.replace(" ", "_") or "workspace"


def _legs_to_npz(legs, source):
    flat = {"frame": "normalized", "source": source}
    for leg, d in legs.items():
        flat[f"{leg}_abd_observed_min"], flat[f"{leg}_abd_observed_max"] = d["abd_observed"]
        flat[f"{leg}_abd_safe_min"], flat[f"{leg}_abd_safe_max"] = d["abd_safe"]
        flat[f"{leg}_abd_zero"] = 0.0
        flat[f"{leg}_knee_grid"] = d["knee_grid"]
        flat[f"{leg}_knee_cam_origin"] = d["knee_cam_origin"]
        flat[f"{leg}_knee_thigh_origin"] = d["knee_thigh_origin"]
        flat[f"{leg}_knee_grid_deg"] = d["knee_grid_deg"]
        flat[f"{leg}_knee_zero"] = np.zeros(2)
        if d.get("samples") is not None:
            flat[f"{leg}_samples"] = d["samples"]
    return flat


def _legs_from_npz(z):
    legs = {}
    for leg in ("left", "right"):
        if f"{leg}_abd_safe_min" not in z.files:
            continue
        legs[leg] = dict(
            abd_observed=(float(z[f"{leg}_abd_observed_min"]), float(z[f"{leg}_abd_observed_max"])),
            abd_safe=(float(z[f"{leg}_abd_safe_min"]), float(z[f"{leg}_abd_safe_max"])),
            knee_grid=z[f"{leg}_knee_grid"].astype(bool),
            knee_cam_origin=float(z[f"{leg}_knee_cam_origin"]),
            knee_thigh_origin=float(z[f"{leg}_knee_thigh_origin"]),
            knee_grid_deg=float(z[f"{leg}_knee_grid_deg"]),
            samples=z[f"{leg}_samples"] if f"{leg}_samples" in z.files else None,
        )
    return legs
