"""Figure: the rail-sweep extent, with the two extreme poses of the traversable workspace
superposed (ghost-blended MuJoCo renders) and the measured toe path drawn over them.

Reads results/rail_path.npz (the slow-verify record) + results/rail_bound.json; writes
results/rail_sweep_extent.png.

Run:  .venv/Scripts/python.exe leg2d/plot_rail_sweep.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import rail_bound as rb  # noqa: E402

OUT = rb.RESULTS / "rail_sweep_extent.png"

# traversable range from the fold-probe (printed by rail_bound.py's summary / json L_workspace)
X_LO, X_HI = -0.615, +0.426

INK = "#F5F5F0"
BOX = dict(boxstyle="round,pad=0.35", fc="#101418", ec="none", alpha=0.72)
C_PATH = "#5AA9F9"          # traversable toe path
C_FOLD = "#F97066"          # fold-marginal portion, excluded


def camera_axes(cam):
    az, el = np.deg2rad(cam.azimuth), np.deg2rad(cam.elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
    right = np.cross(fwd, up)
    pos = np.asarray(cam.lookat) - cam.distance * fwd
    return pos, fwd, up, right


def project(pts, cam, fovy_deg, W, H):
    pos, fwd, up, right = camera_axes(cam)
    d = np.atleast_2d(pts) - pos
    zc = d @ fwd
    xc = d @ right
    yc = d @ up
    t = np.tan(np.deg2rad(fovy_deg) / 2.0)
    aspect = W / H
    px = (1.0 + xc / (zc * t * aspect)) / 2.0 * W
    py = (1.0 - yc / (zc * t)) / 2.0 * H
    return np.column_stack([px, py])


def main():
    z = np.load(rb.PATH_NPZ)
    toe, qpos = z["toe"], z["qpos"]
    half = len(z["t"]) // 2
    toe, qpos = toe[:half], qpos[:half]
    meta = json.loads(rb.OUT_JSON.read_text())
    base_h, L = meta["base_h"], meta["L_workspace"]

    x_rel = toe[:, 0]
    i_lo = int(np.argmin(np.abs(x_rel - X_LO)))
    i_hi = int(np.argmin(np.abs(x_rel - X_HI)))

    # ---- render the two extreme poses with an identical camera, blend 50/50 ----
    model = mujoco.MjModel.from_xml_path(str(rb.MODEL_PATH))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    W, H = 1280, 720
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance, cam.elevation, cam.azimuth = 1.9, -12, 90
    cam.lookat[:] = [-0.07, 0.0, base_h * 0.55]

    data = mujoco.MjData(model)
    frames = []
    for i in (i_lo, i_hi):
        qp = qpos[i].copy()
        qp[2] = base_h
        data.qpos[:] = qp
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam)
        frames.append(renderer.render().astype(np.float32))
    img = (0.5 * frames[0] + 0.5 * frames[1]).astype(np.uint8)

    # toe y-offset (left leg) for projecting the recorded (x, z) path into the image
    y_toe = float(data.geom_xpos[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "foot_L_col")][1])
    fovy = float(model.vis.global_.fovy)

    order = np.argsort(x_rel)
    x_o, z_o = x_rel[order], toe[order, 1]
    trav = (x_o >= X_LO) & (x_o <= X_HI)
    p_trav = project(np.column_stack([x_o[trav], np.full(trav.sum(), y_toe),
                                      base_h + z_o[trav]]), cam, fovy, W, H)
    p_ends = project(np.array([[x_rel[i_lo], y_toe, base_h + toe[i_lo, 1]],
                               [x_rel[i_hi], y_toe, base_h + toe[i_hi, 1]]]), cam, fovy, W, H)
    # extent arrow drawn just above the floor plane, between the two extreme toe x positions
    p_arrow = project(np.array([[x_rel[i_lo], y_toe, 0.10],
                                [x_rel[i_hi], y_toe, 0.10]]), cam, fovy, W, H)

    # ---- compose ----
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=110)
    ax.imshow(img)
    ax.set_axis_off()

    ax.plot(p_trav[:, 0], p_trav[:, 1], color=C_PATH, lw=2.6, solid_capstyle="round")
    ax.plot(p_ends[:, 0], p_ends[:, 1], "o", color=C_PATH, ms=9,
            mec="#101418", mew=1.4, zorder=5)

    for k in range(2):                                   # droplines toe -> extent arrow
        ax.plot([p_ends[k, 0], p_arrow[k, 0]], [p_ends[k, 1], p_arrow[k, 1]],
                color=INK, lw=0.9, ls=":", alpha=0.7)
    ax.annotate("", xy=tuple(p_arrow[1]), xytext=tuple(p_arrow[0]),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.8,
                                shrinkA=0, shrinkB=0))
    mid = p_arrow.mean(axis=0)
    ax.text(mid[0], mid[1] + 26, f"traversable sweep  L = {L:.2f} m",
            ha="center", va="top", color=INK, fontsize=13, bbox=BOX)

    # x = 0 (below the hip) reference tick on the extent arrow
    p_zero = project(np.array([[0.0, y_toe, 0.10], [0.0, y_toe, 0.16]]), cam, fovy, W, H)
    ax.plot(p_zero[:, 0], p_zero[:, 1], color=INK, lw=1.6, alpha=0.9)
    ax.text(p_zero[1, 0], p_zero[1, 1] - 8, "x = 0 (hip)", ha="center", va="bottom",
            color=INK, fontsize=10, bbox=BOX)

    ax.text(p_ends[0, 0] - 10, p_ends[0, 1] - 26, f"x = {x_rel[i_lo]:+.2f} m",
            ha="center", va="bottom", color=INK, fontsize=11, bbox=BOX)
    ax.text(p_ends[1, 0] + 10, p_ends[1, 1] - 26, f"x = {x_rel[i_hi]:+.2f} m",
            ha="center", va="bottom", color=INK, fontsize=11, bbox=BOX)

    # legend (top-left), title (top-center)
    ax.text(0.015, 0.03,
            "Rail rig: torso welded at 1.10 m, contact off — two extreme poses superposed",
            transform=ax.transAxes, color=INK, fontsize=12.5, va="bottom", bbox=BOX)
    leg = [plt.Line2D([], [], color=C_PATH, lw=2.6, label="measured toe path (traversable)")]
    lg = ax.legend(handles=leg, loc="upper left", frameon=True, fontsize=11,
                   facecolor="#101418", edgecolor="none", framealpha=0.72,
                   labelcolor=INK, borderpad=0.8)
    lg.set_zorder(6)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, facecolor="#101418")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
