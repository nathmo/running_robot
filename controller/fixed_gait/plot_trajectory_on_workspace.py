#!/usr/bin/env python3
"""Overlay the hand-recorded gait trajectory on top of the calibrated safe workspace.

Combines two independently-captured artifacts that live in the same raw-motor-degree frame:
    fixed_gait/joint_limits.npz    -- calibrate_workspace.py: per-leg safe (cam, thigh) region
    fixed_gait/gait_recorded.npz   -- record_trajectory.py:   per-leg taught (cam, hip) cycle

(cam=105, thigh/hip=106 are the same motor; "thigh" and "hip" are just the two scripts' names for
it.) Plots each leg's safe-workspace grid + abduction range, with the recorded trajectory loop
drawn on top, so you can see at a glance whether the taught gait stays inside the demonstrated-safe
envelope.

    python fixed_gait/plot_trajectory_on_workspace.py
    python fixed_gait/plot_trajectory_on_workspace.py --workspace fixed_gait/joint_limits.npz \\
        --trajectory fixed_gait/gait_recorded.npz --out fixed_gait/trajectory_on_workspace.png
"""
import argparse

import numpy as np

import trajectory as traj

DEFAULT_WORKSPACE = "fixed_gait/joint_limits.npz"
DEFAULT_TRAJECTORY = "fixed_gait/gait_recorded.npz"
DEFAULT_OUT = "fixed_gait/trajectory_on_workspace.png"


def load_workspace(path):
    z = np.load(path)
    legs = {}
    for leg in ("left", "right"):
        if f"{leg}_abd_safe_min" not in z.files:
            continue
        legs[leg] = dict(
            abd_observed=(float(z[f"{leg}_abd_observed_min"]), float(z[f"{leg}_abd_observed_max"])),
            abd_safe=(float(z[f"{leg}_abd_safe_min"]), float(z[f"{leg}_abd_safe_max"])),
            abd_zero=float(z[f"{leg}_abd_zero"]),
            knee_grid=z[f"{leg}_knee_grid"].astype(bool),
            knee_cam_origin=float(z[f"{leg}_knee_cam_origin"]),
            knee_thigh_origin=float(z[f"{leg}_knee_thigh_origin"]),
            knee_grid_deg=float(z[f"{leg}_knee_grid_deg"]),
            knee_zero=z[f"{leg}_knee_zero"],
        )
    return legs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    ap.add_argument("--trajectory", default=DEFAULT_TRAJECTORY)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--points", type=int, default=400, help="phase samples drawn along the loop")
    args = ap.parse_args()

    ws = load_workspace(args.workspace)
    data = traj.load(args.trajectory)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    legs = [l for l in ("left", "right") if l in ws]
    ph = np.linspace(0, 1, args.points, endpoint=False)

    fig, axes = plt.subplots(len(legs), 2, figsize=(11, 4.8 * len(legs)), squeeze=False)
    for row, leg in enumerate(legs):
        w = ws[leg]
        ax_abd, ax_knee = axes[row]

        lo, hi = w["abd_observed"]
        slo, shi = w["abd_safe"]
        ax_abd.hlines(0, lo, hi, color="0.7", lw=10, label="observed")
        ax_abd.hlines(0, slo, shi, color="#2c9e3f", lw=10, label="safe (calibrated)")
        ax_abd.axvline(w["abd_zero"], color="k", ls="--", lw=1, label="workspace zero")
        if data[leg] is not None:
            ax_abd.axvline(data[leg]["abduction_hold"], color="darkorange", ls="-", lw=2,
                           label="trajectory hold")
        ax_abd.set_yticks([])
        ax_abd.set_xlabel("abduction (raw motor deg)")
        ax_abd.set_title(f"{leg} abduction")
        ax_abd.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
        ax_abd.grid(alpha=0.3, axis="x")

        grid = w["knee_grid"]
        ci, tj = np.nonzero(grid)
        cam_cells = w["knee_cam_origin"] + (ci + 0.5) * w["knee_grid_deg"]
        thigh_cells = w["knee_thigh_origin"] + (tj + 0.5) * w["knee_grid_deg"]
        ax_knee.scatter(cam_cells, thigh_cells, s=3, c="#2c9e3f", alpha=0.3, lw=0,
                        label="safe workspace (eroded)")
        ax_knee.plot(*w["knee_zero"], "ks", ms=9, label="workspace zero")

        if data[leg] is not None:
            R = np.array([traj.reconstruct(data, leg, p) for p in ph])
            cam, hip = R[:, traj.COL_CAM], R[:, traj.COL_HIP]
            cam_closed = np.append(cam, cam[0])
            hip_closed = np.append(hip, hip[0])
            ax_knee.plot(cam_closed, hip_closed, color="darkorange", lw=2.2,
                        label="recorded trajectory", zorder=5)
            ax_knee.plot(data[leg]["center"][traj.COL_CAM], data[leg]["center"][traj.COL_HIP],
                        "*", color="darkorange", ms=14, mec="k", mew=0.5,
                        label="trajectory center", zorder=6)
            outside = ~np.array([
                (w["knee_cam_origin"] <= c < w["knee_cam_origin"] + grid.shape[0] * w["knee_grid_deg"]
                 and w["knee_thigh_origin"] <= t < w["knee_thigh_origin"] + grid.shape[1] * w["knee_grid_deg"]
                 and grid[int((c - w["knee_cam_origin"]) / w["knee_grid_deg"]),
                          int((t - w["knee_thigh_origin"]) / w["knee_grid_deg"])])
                for c, t in zip(cam, hip)])
            if outside.any():
                ax_knee.scatter(cam[outside], hip[outside], s=18, c="red", marker="x", zorder=7,
                                label=f"outside safe region ({outside.sum()}/{len(cam)} pts)")

        ax_knee.set_xlabel("cam (raw motor deg)")
        ax_knee.set_ylabel("thigh / hip (raw motor deg)")
        ax_knee.set_title(f"{leg} knee (cam, thigh) — workspace + recorded trajectory")
        ax_knee.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        ax_knee.grid(alpha=0.3)
        ax_knee.set_aspect("equal", "box")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
