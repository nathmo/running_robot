"""End-effector static force map over the ROBOT'S RECORDED SAFE WORKSPACE: the maximum
pure-vertical and pure-horizontal toe force the two sagittal actuators can exert at each
reachable toe position.

DOMAIN -- the hand-recorded workspace, not a sim scan. The robot's workspace is recorded by
backdriving the leg (outline), then manually closed and filled in the web UI; the active filled
region ships in robot/fixed_gait/webui/data/workspace_active.npz (normalized frame, 1 deg
knee-grid cells). Cells are mapped to model joint angles through the VERIFIED sign+offset map
(data/model_map.json, left leg) and evaluated on the closed-loop FK lookup table
(webui/fk_lut.npz, built offline from the MuJoCo model): toe XZ per (cam, thigh) cell plus the
LUT's own assembly/validity mask.

FORCE. The toe Jacobian J = d(ee)/d(cam, thigh) comes from finite differences of the FK table.
Joint torque for a toe force F is tau = J^T F with both joints capped at the measured
144.5 N*m, so

    Fz_max = min_j  tau_peak / |dz/dq_j|      (pushing straight down, Fx = 0)
    Fx_max = min_j  tau_peak / |dx/dq_j|      (pushing straight fore-aft, Fz = 0)

in body weights (15.14 kg), clipped at 3.5 BW (near full extension the leg is a strut and the
Jacobian alone would allow arbitrary vertical force -- a structural limit, not an actuation one).
Static, leg-weight torque neglected.

Run:  .venv/Scripts/python.exe leg2d/plot_force_map.py   ->  results/force_map.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
sys.path.insert(0, str(PKG))
import rail_bound as rb  # noqa: E402  (RESULTS dir + PEAK)

WS_FILE = REPO / "robot" / "fixed_gait" / "webui" / "data" / "workspace_active.npz"
MAP_FILE = REPO / "robot" / "fixed_gait" / "webui" / "data" / "model_map.json"
LUT_FILE = REPO / "robot" / "fixed_gait" / "webui" / "fk_lut.npz"
OUT = rb.RESULTS / "force_map.png"

M, G = 15.14, 9.81
BW = M * G
TAU_PEAK = rb.PEAK
CLIP_BW = 3.5
CMAP_Z = LinearSegmentedColormap.from_list(
    "seqblue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
CMAP_X = LinearSegmentedColormap.from_list(
    "seqorange", ["#fbe3d2", "#f5b78f", "#eb6834", "#b34a1f", "#7a2f12"])


def main():
    ws = np.load(WS_FILE, allow_pickle=True)
    grid = ws["left_knee_grid"]                            # bool [n_cam_deg, n_thigh_deg]
    c0, t0 = float(ws["left_knee_cam_origin"]), float(ws["left_knee_thigh_origin"])
    step = float(ws["left_knee_grid_deg"])
    mm = json.loads(MAP_FILE.read_text())["left"]
    assert json.loads(MAP_FILE.read_text())["verified"]["left"]

    lut = np.load(LUT_FILE)
    cam, thigh = lut["cam"], lut["thigh"]                  # model radians
    ee = lut["nodes"][:, :, 6, :]                          # toe XZ per cell
    valid = lut["valid"]

    # LUT cell -> normalized degrees -> workspace grid cell
    cam_norm = (np.degrees(cam) - mm["cam_off_deg"]) / mm["cam"]
    thigh_norm = (np.degrees(thigh) - mm["thigh_off_deg"]) / mm["thigh"]
    ic = np.round((cam_norm - c0) / step).astype(int)
    it = np.round((thigh_norm - t0) / step).astype(int)
    IC, IT = np.meshgrid(ic, it, indexing="ij")
    inside = (IC >= 0) & (IC < grid.shape[0]) & (IT >= 0) & (IT < grid.shape[1])
    in_ws = np.zeros_like(valid)
    in_ws[inside] = grid[IC[inside], np.clip(IT, 0, grid.shape[1] - 1)[inside]]
    mask = in_ws & valid & np.isfinite(ee).all(axis=2)
    # sanity: the verified map should land most recorded cells on the LUT assembly band
    print(f"[force-map] workspace cells on LUT-valid poses: "
          f"{mask.sum()}/{in_ws.sum()} LUT cells ({mask.sum() / max(in_ws.sum(), 1) * 100:.0f}%)")

    # Jacobian by finite differences of the FK table (NaN outside assembly -> dropped by mask)
    dee_dc = np.gradient(ee, cam, axis=0)                  # d(x,z)/d cam
    dee_dt = np.gradient(ee, thigh, axis=1)                # d(x,z)/d thigh
    with np.errstate(invalid="ignore", divide="ignore"):
        fz = TAU_PEAK / np.maximum(np.maximum(np.abs(dee_dc[..., 1]), np.abs(dee_dt[..., 1])),
                                   1e-3) / BW
        fx = TAU_PEAK / np.maximum(np.maximum(np.abs(dee_dc[..., 0]), np.abs(dee_dt[..., 0])),
                                   1e-3) / BW
    fz, fx = np.minimum(fz, CLIP_BW), np.minimum(fx, CLIP_BW)
    ok = mask & np.isfinite(fz) & np.isfinite(fx)
    pts = ee[ok]
    # LUT (hip) frame -> torso frame: fitted against the workspace-scan poses (x residual
    # 6.8 mm), so the knee-fold boundary from the sweep analysis applies directly
    pts = np.column_stack([pts[:, 0] + 0.021, 1.028 * pts[:, 1] + 0.030])
    vz, vx = fz[ok], fx[ok]
    # cut the knee-over-centre (fold) region: the hand-backdriven recording contains poses past
    # the 4-bar dead centre (a human can guide it there quasi-statically), but force there is
    # not usable -- the sweep analysis put the boundary at x = +0.426 of the torso
    infold = pts[:, 0] > 0.426
    print(f"[force-map] cut {infold.sum()}/{len(pts)} poses past the knee-fold boundary "
          f"(x_torso > +0.426 m)")
    pts, vz, vx = pts[~infold], vz[~infold], vx[~infold]
    print(f"[force-map] {len(pts)} poses; Fz median {np.median(vz):.2f} BW "
          f"(clip {CLIP_BW}), Fx median {np.median(vx):.2f} BW, "
          f"anisotropy ~{np.median(vz) / np.median(vx):.1f}:1")

    # dense fill: inverse-distance interpolation onto a fine grid, masked near the data
    gx = np.arange(pts[:, 0].min() - 0.01, pts[:, 0].max() + 0.01, 0.004)
    gz = np.arange(pts[:, 1].min() - 0.01, pts[:, 1].max() + 0.01, 0.004)
    GX, GZ = np.meshgrid(gx, gz)
    Vz = np.full(GX.shape, np.nan)
    Vx = np.full(GX.shape, np.nan)
    for r in range(GX.shape[0]):                           # chunked: rows vs full point set
        d2 = (GX[r][:, None] - pts[:, 0]) ** 2 + (GZ[r][:, None] - pts[:, 1]) ** 2
        near = np.sqrt(d2.min(axis=1)) <= 0.02
        k6 = np.argsort(d2, axis=1)[:, :6]
        wk = 1.0 / (np.take_along_axis(d2, k6, axis=1) + 1e-5)
        Vz[r] = np.where(near, (wk * vz[k6]).sum(1) / wk.sum(1), np.nan)
        Vx[r] = np.where(near, (wk * vx[k6]).sum(1) / wk.sum(1), np.nan)

    fig, axs = plt.subplots(1, 2, figsize=(12.6, 6.2), dpi=130, sharey=True)
    for ax, V, val, cmap, title in ((axs[0], Vz, vz, CMAP_Z, "max vertical force  $F_z$"),
                                    (axs[1], Vx, vx, CMAP_X, "max horizontal force  $F_x$")):
        lo, hi = np.percentile(val, 10), np.percentile(val, 90)
        pc = ax.pcolormesh(GX, GZ, np.ma.masked_invalid(V), cmap=cmap, vmin=lo, vmax=hi,
                           shading="auto")
        ax.plot([0], [0], marker="+", ms=12, mew=2, color="#333")
        ax.annotate("torso origin", (0, 0), xytext=(8, 4), textcoords="offset points",
                    fontsize=9, color="#333")
        ax.axvline(0.426, color="#8a8f98", lw=1.0, ls=":")
        ax.annotate("knee fold", (0.426, -0.35), fontsize=8.5, color="#666",
                    rotation=90, ha="right", va="top", xytext=(-4, 0),
                    textcoords="offset points")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("toe x, fore-aft [m]")
        ax.set_aspect("equal")
        ax.grid(True, lw=0.4, alpha=0.3)
        cb = fig.colorbar(pc, ax=ax, shrink=0.85, pad=0.02, extend="both")
        cb.set_label("[body weights]", fontsize=9)
    axs[0].set_ylabel("toe z, relative to torso [m]")
    fig.suptitle("Static end-effector force capability over the recorded safe workspace "
                 f"(values capped at {CLIP_BW} BW; colour scaled to each map's "
                 "10th--90th percentile)", fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
