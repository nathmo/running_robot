"""FK lookup table runtime — pure numpy, no mujoco (Pi-safe).

fk_lut.npz is generated OFFLINE on the desktop by mujoco/spiderbot/gen_fk_lut.py from the real
closed-loop model (plot_reachability.Leg.fk + linkage_points). Contents:
    cam[nc], thigh[nt]          grid axes, radians (model joint space, LEFT leg)
    nodes[nc, nt, 7, 2] f32     XZ of linkage nodes per cell: cam, thigh, push, knee, ank, ptip, ee
    valid[nc, nt] bool          assembles & no self-collision & ankle in range
    feas[nc, nt] bool           assembles (the physical never-exceed band)

Normalized motor degrees map into model radians per side via a sign map (model is the LEFT leg;
the right leg mirrors). The map lives in data/model_map.json and is VERIFIED against a recorded
workspace band by IoU — the (cam, thigh) assembly band is a thin diagonal, so a wrong sign mirrors
it off the band and loses badly. EE displays stay disabled per side until verified.
"""
import json
import math
import os

import numpy as np

import paths
import calibrate_workspace as cw          # _binary_dilate for the feasibility margin

NODE_NAMES = ("cam", "thigh", "push", "knee", "ank", "ptip", "ee")
FEAS_MARGIN_DEG = 5.0

_DEFAULT_MAP = {"left": {"cam": 1, "thigh": 1}, "right": {"cam": 1, "thigh": 1},
                "verified": {"left": False, "right": False}}


class FkLut:
    def __init__(self, lut_path=paths.FK_LUT_FILE, map_path=paths.MODEL_MAP_FILE):
        self.available = False
        self.map_path = map_path
        self.model_map = dict(_DEFAULT_MAP)
        if os.path.exists(map_path):
            try:
                with open(map_path, "r", encoding="utf-8-sig") as f:   # -sig: tolerate a BOM
                    m = json.load(f)
                for k in ("left", "right", "verified"):
                    if k in m:
                        self.model_map[k] = m[k]
            except (ValueError, OSError):
                pass
        if not os.path.exists(lut_path):
            print(f"(no FK LUT at {lut_path} — EE animation disabled; "
                  f"generate it with mujoco/spiderbot/gen_fk_lut.py)")
            return
        z = np.load(lut_path)
        self.cam = z["cam"]
        self.thigh = z["thigh"]
        self.nodes = z["nodes"]
        self.valid = z["valid"].astype(bool)
        self.feas = z["feas"].astype(bool)
        self._dc = float(self.cam[1] - self.cam[0])
        self._dt = float(self.thigh[1] - self.thigh[0])
        r = max(1, int(round(math.radians(FEAS_MARGIN_DEG) / min(self._dc, self._dt))))
        self._feas_dilated = cw._binary_dilate(self.feas, r)
        # smooth[i,j]: the cell whose corners are (i..i+1, j..j+1) is interpolable — cells
        # straddling the fold (dead-center singularity) have corner EE spreads of ~0.5 m and
        # bilinear interpolation there is meaningless; they are masked like invalid cells.
        ee = self.nodes[:, :, 6, :]                               # [nc, nt, 2]
        c4 = np.stack([ee[:-1, :-1], ee[:-1, 1:], ee[1:, :-1], ee[1:, 1:]])   # [4, nc-1, nt-1, 2]
        spread = np.linalg.norm(c4 - c4.mean(axis=0), axis=-1).max(axis=0)
        with np.errstate(invalid="ignore"):
            self._smooth = np.isfinite(spread) & (spread < 0.03)  # [nc-1, nt-1] per-cell
        self.available = True

    def save_map(self):
        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump(self.model_map, f, indent=2)

    def side_verified(self, side):
        return self.available and bool(self.model_map["verified"].get(side))

    # ------------------------------------------------------------------ deg <-> rad
    def to_rad(self, side, cam_deg, thigh_deg):
        s = self.model_map[side]
        return (s["cam"] * np.radians(cam_deg), s["thigh"] * np.radians(thigh_deg))

    # ------------------------------------------------------------------ lookups
    def _cell(self, cam_rad, thigh_rad):
        i = (cam_rad - self.cam[0]) / self._dc
        j = (thigh_rad - self.thigh[0]) / self._dt
        return i, j

    def interp_nodes(self, side, cam_deg, thigh_deg):
        """Bilinear linkage-node positions at one pose. Returns (nodes[7,2] list, valid bool)."""
        if not self.available:
            return None, False
        cr, tr = self.to_rad(side, float(cam_deg), float(thigh_deg))
        i, j = self._cell(cr, tr)
        i0 = int(np.clip(np.floor(i), 0, len(self.cam) - 2))
        j0 = int(np.clip(np.floor(j), 0, len(self.thigh) - 2))
        fi = float(np.clip(i - i0, 0.0, 1.0))
        fj = float(np.clip(j - j0, 0.0, 1.0))
        corners = self.nodes[i0:i0 + 2, j0:j0 + 2]                # [2,2,7,2]
        vmask = self.valid[i0:i0 + 2, j0:j0 + 2]
        ok = bool(vmask.all()) and bool(self._smooth[i0, j0])
        if not np.isfinite(corners).all():
            fin = np.isfinite(corners[..., 0, 0])
            if not fin.any():
                return None, False
            ii, jj = np.argwhere(fin)[0]
            return corners[ii, jj].tolist(), False
        w = np.array([[(1 - fi) * (1 - fj), (1 - fi) * fj], [fi * (1 - fj), fi * fj]])
        out = (corners * w[:, :, None, None]).sum(axis=(0, 1))
        return np.round(out, 4).tolist(), ok

    def project_ee(self, side, cam_deg, thigh_deg):
        """Vectorized EE XZ for arrays of normalized degrees. NaN where not assemblable."""
        if not self.available:
            return np.full((np.size(cam_deg), 2), np.nan)
        cr, tr = self.to_rad(side, np.asarray(cam_deg, float), np.asarray(thigh_deg, float))
        i, j = self._cell(cr, tr)
        i0 = np.clip(np.floor(i).astype(int), 0, len(self.cam) - 2)
        j0 = np.clip(np.floor(j).astype(int), 0, len(self.thigh) - 2)
        fi = np.clip(i - i0, 0.0, 1.0)[:, None]
        fj = np.clip(j - j0, 0.0, 1.0)[:, None]
        ee = self.nodes[:, :, 6, :]                               # [nc,nt,2]
        out = ((1 - fi) * (1 - fj) * ee[i0, j0] + (1 - fi) * fj * ee[i0, j0 + 1]
               + fi * (1 - fj) * ee[i0 + 1, j0] + fi * fj * ee[i0 + 1, j0 + 1])
        oob = (i < 0) | (i > len(self.cam) - 1) | (j < 0) | (j > len(self.thigh) - 1)
        out[oob | ~self._smooth[i0, j0]] = np.nan                 # fold cells not interpolable
        return out

    def feasible_check(self, side, cam_deg, thigh_deg):
        """(ok, reason): is the pose inside the physically-assemblable band (+margin)?
        Used as the never-exceed net when the user overrides the workspace check."""
        if not self.available:
            return True, ""
        cr, tr = self.to_rad(side, float(cam_deg), float(thigh_deg))
        i, j = self._cell(cr, tr)
        i, j = int(round(i)), int(round(j))
        if not (0 <= i < len(self.cam) and 0 <= j < len(self.thigh)) \
                or not self._feas_dilated[i, j]:
            return False, (f"{side} (cam={cam_deg:+.1f}, thigh={thigh_deg:+.1f}) is outside the "
                           f"physically assemblable linkage band — refused even with override")
        return True, ""

    # ------------------------------------------------------------------ projections for the UI
    def ee_region(self, side, leg_ws, max_points=3000):
        """Project a leg's safe-workspace cells into EE space -> decimated [[x,z],...]."""
        if not self.available or leg_ws is None:
            return []
        grid = leg_ws["knee_grid"]
        ci, tj = np.nonzero(grid)
        if not len(ci):
            return []
        cam_deg = leg_ws["knee_cam_origin"] + (ci + 0.5) * leg_ws["knee_grid_deg"]
        thigh_deg = leg_ws["knee_thigh_origin"] + (tj + 0.5) * leg_ws["knee_grid_deg"]
        pts = self.project_ee(side, cam_deg, thigh_deg)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) > max_points:
            pts = pts[:: len(pts) // max_points + 1]
        return np.round(pts, 4).tolist()

    def ee_zero(self, side):
        nodes, _ = self.interp_nodes(side, 0.0, 0.0)
        return None if nodes is None else nodes[6]

    # ------------------------------------------------------------------ sign-map verification
    def verify_against_workspace(self, wstore):
        """Try all 4 sign combos per side; the one whose normalized workspace band lands on the
        LUT feasibility band (max IoU) wins. Marks 'verified' when the winner is unambiguous."""
        if not self.available:
            return {"error": "no FK LUT loaded"}
        report = {}
        for side in paths.SIDES:
            leg = wstore.legs.get(side)
            if leg is None:
                report[side] = {"error": "no workspace for this leg"}
                continue
            grid = leg["knee_grid"]
            ci, tj = np.nonzero(grid)
            cam_deg = leg["knee_cam_origin"] + (ci + 0.5) * leg["knee_grid_deg"]
            thigh_deg = leg["knee_thigh_origin"] + (tj + 0.5) * leg["knee_grid_deg"]
            scores = {}
            for sc in (1, -1):
                for st in (1, -1):
                    cr = sc * np.radians(cam_deg)
                    tr = st * np.radians(thigh_deg)
                    i = np.round((cr - self.cam[0]) / self._dc).astype(int)
                    j = np.round((tr - self.thigh[0]) / self._dt).astype(int)
                    ok = (i >= 0) & (i < len(self.cam)) & (j >= 0) & (j < len(self.thigh))
                    inside = np.zeros(len(cam_deg), bool)
                    inside[ok] = self.feas[i[ok], j[ok]]
                    scores[f"{sc:+d},{st:+d}"] = round(float(inside.mean()), 4)
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            best, second = ranked[0], ranked[1]
            sc, st = (int(v) for v in best[0].split(","))
            decisive = best[1] > 0.5 and best[1] > 1.5 * max(second[1], 1e-6)
            report[side] = {"scores": scores, "best": best[0], "coverage": best[1],
                            "decisive": decisive}
            if decisive:
                self.model_map[side] = {"cam": sc, "thigh": st}
                self.model_map["verified"][side] = True
        self.save_map()
        return report
