"""FK lookup table runtime — pure numpy, no mujoco (Pi-safe).

fk_lut.npz is generated OFFLINE on the desktop by mujoco/spiderbot/gen_fk_lut.py from the real
closed-loop model (plot_reachability.Leg.fk + linkage_points). Contents:
    cam[nc], thigh[nt]          grid axes, radians (model joint space, LEFT leg)
    nodes[nc, nt, 7, 2] f32     XZ of linkage nodes per cell: cam, thigh, push, knee, ank, ptip, ee
    valid[nc, nt] bool          assembles & no self-collision & ankle in range
    feas[nc, nt] bool           assembles (the physical never-exceed band)

Normalized motor degrees map into model radians per side via a SIGN + OFFSET map (model is the
LEFT leg; the right leg mirrors):  model_deg = sign * norm_deg + off_deg.
The offset is essential: the robot's captured zero pose is NOT the MJCF qpos-0 pose. The zero is
captured with the leg near-extended, i.e. right at the 4-bar dead-center where large cam
rotations barely move the leg, so the zero lands far from model zero (measured ~ +135 deg of cam
on real data). The map lives in data/model_map.json and is VERIFIED against a recorded workspace
band: for every sign combo the offsets are fitted by maximizing the fraction of workspace cells
inside the LUT assembly band (thin diagonal ribbon -> the fit is sharp). EE displays stay
disabled per side until verified.
"""
import json
import math
import os
import zipfile

import numpy as np

import paths
import calibrate_workspace as cw          # _binary_dilate for the feasibility margin

NODE_NAMES = ("cam", "thigh", "push", "knee", "ank", "ptip", "ee")
FEAS_MARGIN_DEG = 5.0

# flip_view: pure DISPLAY x-mirror of the EE canvas (viewing convention, like plot_reachability's
# VIEW_X) — it never changes which LUT cell is looked up, only how the drawing is projected.
# Defaults to True so the canvas matches plot_reachability's mirrored map out of the box.
# cam_off_deg / thigh_off_deg: model_deg = sign * norm_deg + off_deg (see module docstring).
_DEFAULT_MAP = {"left": {"cam": 1, "thigh": 1, "cam_off_deg": 0.0, "thigh_off_deg": 0.0,
                         "flip_view": True},
                "right": {"cam": 1, "thigh": 1, "cam_off_deg": 0.0, "thigh_off_deg": 0.0,
                          "flip_view": True},
                "verified": {"left": False, "right": False}}

# offset-fit search bounds: cam zero is captured at the fold (huge uncertainty, and the LUT cam
# axis spans a full turn) -> search the whole circle; thigh zero is directly observable -> small.
OFF_CAM_MAX_DEG = 180.0
OFF_THIGH_MAX_DEG = 25.0


class FkLut:
    def __init__(self, lut_path=paths.FK_LUT_FILE, map_path=paths.MODEL_MAP_FILE):
        self.available = False
        self.lut_path = lut_path
        self.map_path = map_path
        self.model_map = {k: dict(v) for k, v in _DEFAULT_MAP.items()}
        if os.path.exists(map_path):
            try:
                with open(map_path, "r", encoding="utf-8-sig") as f:   # -sig: tolerate a BOM
                    m = json.load(f)
                for k in ("left", "right", "verified"):
                    if k in m:
                        self.model_map[k].update(m[k])     # merge: old files lack e.g. flip_view
            except (ValueError, OSError):
                pass
        if not self.try_reload():
            print(f"(no FK LUT at {lut_path} — EE animation disabled; generate it with "
                  f"mujoco/spiderbot/gen_fk_lut.py — it hot-loads once the file appears)")

    def try_reload(self):
        """Load (or hot-load) fk_lut.npz. Safe to call repeatedly — cheap when absent or already
        loaded. Lets a LUT scp'd onto the robot AFTER server start be picked up without a restart."""
        if self.available or not os.path.exists(self.lut_path):
            return self.available
        try:
            z = np.load(self.lut_path)
            cam, thigh = z["cam"], z["thigh"]
            nodes = z["nodes"]
            valid = z["valid"].astype(bool)
            feas = z["feas"].astype(bool)
        except (ValueError, OSError, KeyError, EOFError, zipfile.BadZipFile) as e:
            print(f"(FK LUT at {self.lut_path} unreadable ({e}) — still disabled; "
                  f"incomplete scp copy?)")
            return False
        self.cam, self.thigh, self.nodes, self.valid, self.feas = cam, thigh, nodes, valid, feas
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
        print(f"FK LUT loaded: {len(self.cam)}x{len(self.thigh)} grid from {self.lut_path}")
        return True

    def save_map(self):
        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump(self.model_map, f, indent=2)

    def side_verified(self, side):
        return self.available and bool(self.model_map["verified"].get(side))

    # ------------------------------------------------------------------ deg <-> rad
    def to_rad(self, side, cam_deg, thigh_deg):
        """Normalized degrees -> model radians:  model = sign * norm + offset."""
        s = self.model_map[side]
        return (np.radians(s["cam"] * np.asarray(cam_deg, float) + s.get("cam_off_deg", 0.0)),
                np.radians(s["thigh"] * np.asarray(thigh_deg, float)
                           + s.get("thigh_off_deg", 0.0)))

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
        """EE at the robot's CALIBRATED zero pose (normalized 0,0 mapped through sign+offset)."""
        nodes, _ = self.interp_nodes(side, 0.0, 0.0)
        return None if nodes is None else nodes[6]

    def ee_model_zero(self):
        """EE at the MODEL zero pose (MJCF/URDF qpos 0) — side-independent, no sign/offset.
        Distinct from ee_zero(): the robot's captured zero is NOT the model zero pose."""
        if not self.available:
            return None
        i0 = int(np.argmin(np.abs(self.cam)))
        j0 = int(np.argmin(np.abs(self.thigh)))
        ee = self.nodes[i0, j0, 6]
        return None if not np.isfinite(ee).all() else np.round(ee, 4).tolist()

    # ------------------------------------------------------------------ sign+offset verification
    def _fit_side(self, cam_deg, thigh_deg):
        """For each sign combo, fit the (cam, thigh) OFFSET that maximizes the fraction of the
        given normalized workspace cells landing inside the LUT assembly band (coarse grid
        search, then a 1-cell refine). Returns {(sc, st): (coverage, cam_off_deg, thigh_off_deg)}.
        Offsets are needed because the captured zero pose is far from model zero (module doc)."""
        dcd, dtd = np.degrees(self._dc), np.degrees(self._dt)
        cam0d, th0d = np.degrees(self.cam[0]), np.degrees(self.thigh[0])
        nc, nt = len(self.cam), len(self.thigh)
        ki_max = int(round(OFF_CAM_MAX_DEG / dcd))
        kj_max = int(round(OFF_THIGH_MAX_DEG / dtd))
        out = {}
        for sc in (1, -1):
            for st in (1, -1):
                bi = (sc * cam_deg - cam0d) / dcd          # fractional LUT cell at zero offset
                bj = (st * thigh_deg - th0d) / dtd

                def cov(ki, kj):
                    i = np.round(bi + ki).astype(int)
                    j = np.round(bj + kj).astype(int)
                    ok = (i >= 0) & (i < nc) & (j >= 0) & (j < nt)
                    if not ok.any():
                        return 0.0
                    inside = np.zeros(len(i), bool)
                    inside[ok] = self.feas[i[ok], j[ok]]
                    return float(inside.mean())

                best = (-1.0, 0, 0)
                for ki in range(-ki_max, ki_max + 1, 4):   # coarse: every 4th cell
                    for kj in range(-kj_max, kj_max + 1, 4):
                        c = cov(ki, kj)
                        if c > best[0]:
                            best = (c, ki, kj)
                _, bki, bkj = best
                for ki in range(bki - 4, bki + 5):         # refine +-4 cells at full resolution
                    for kj in range(bkj - 4, bkj + 5):
                        c = cov(ki, kj)
                        if c > best[0]:
                            best = (c, ki, kj)
                out[(sc, st)] = (best[0], best[1] * dcd, best[2] * dtd)
        return out

    def verify_against_workspace(self, wstore):
        """Per side: fit sign + offset against the recorded workspace band (see _fit_side) and
        mark 'verified' when the winner is unambiguous. The thin diagonal assembly band makes a
        correct fit land ~100% of cells inside; wrong signs plateau visibly lower."""
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
            fits = self._fit_side(cam_deg, thigh_deg)
            scores = {f"{sc:+d},{st:+d}": {"coverage": round(c, 4),
                                           "cam_off_deg": round(oc, 1),
                                           "thigh_off_deg": round(ot, 1)}
                      for (sc, st), (c, oc, ot) in fits.items()}
            ranked = sorted(fits.items(), key=lambda kv: -kv[1][0])
            (sc, st), (cbest, oc, ot) = ranked[0]
            second = ranked[1][1][0]
            decisive = cbest >= 0.95 and (cbest - second) >= 0.03
            report[side] = {"scores": scores, "best": f"{sc:+d},{st:+d}",
                            "coverage": round(cbest, 4),
                            "cam_off_deg": round(oc, 1), "thigh_off_deg": round(ot, 1),
                            "decisive": decisive}
            if decisive:
                self.model_map[side].update({"cam": sc, "thigh": st,          # keep flip_view etc.
                                             "cam_off_deg": round(oc, 1),
                                             "thigh_off_deg": round(ot, 1)})
                self.model_map["verified"][side] = True
        self.save_map()
        return report
