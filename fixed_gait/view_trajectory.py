#!/usr/bin/env python3
"""Inspect a recorded trajectory: save a PNG and/or show a live animation. No hardware needed.

    python fixed_gait/view_trajectory.py                       # save PNG (always works, headless)
    python fixed_gait/view_trajectory.py --live                # animated window (needs a display)
    python fixed_gait/view_trajectory.py --anim walk.gif       # save an animated GIF (headless)
    python fixed_gait/view_trajectory.py --file other.npz --period 3

The PNG shows cam & hip angle vs phase for both legs (dephased 180 deg) plus the cam-vs-hip loop.
The animation adds a moving phase marker and a schematic two-link stick figure of each leg. The
angles are real; the leg *geometry* is a schematic (cam drives the knee through a linkage, not 1:1),
so read it as rhythm/coordination, not exact kinematics.
"""
import argparse
import sys

import numpy as np

import trajectory as traj


def sampled(data, side, ph):
    """Absolute [abd, cam, hip] (deg) for a side over the phase grid ph."""
    return np.array([traj.reconstruct(data, side, p) for p in ph])


def world_shape(data, ph):
    """Mean-removed canonical (world) cam,hip over the phase grid — right at ph, left at ph+0.5."""
    N = data["N"]
    idx = (ph * N).astype(int) % N
    return data["canonical"][idx]                    # [len(ph), 2] cam,hip


def static_png(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ph = np.linspace(0, 1, 200, endpoint=False)
    R = sampled(data, "right", ph)
    L = sampled(data, "left", ph) if data["left"] is not None else None
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for col, name, a in ((traj.COL_CAM, "cam (105)", ax[0]), (traj.COL_HIP, "hip (106)", ax[1])):
        a.plot(ph, R[:, col], "b", lw=2, label="right")
        if L is not None:
            a.plot(ph, L[:, col], "r", lw=2, label="left (180° dephased)")
        a.set_title(name); a.set_xlabel("phase"); a.set_ylabel("deg"); a.grid(alpha=.3); a.legend()
    ax[2].plot(R[:, traj.COL_CAM], R[:, traj.COL_HIP], "b", lw=2, label="right")
    if L is not None:
        ax[2].plot(L[:, traj.COL_CAM], L[:, traj.COL_HIP], "r", lw=2, label="left")
    ax[2].set_title("cam–hip loop"); ax[2].set_xlabel("cam (deg)"); ax[2].set_ylabel("hip (deg)")
    ax[2].grid(alpha=.3); ax[2].axis("equal"); ax[2].legend()
    abd = f"abduction hold: right={data['right']['abduction_hold']:.1f}°"
    if data["left"] is not None:
        abd += f"  left={data['left']['abduction_hold']:.1f}°"
    fig.suptitle(abd, y=1.0, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"saved {path}")


def _leg_points(world_hip_deg, world_cam_deg, x0, Lt=1.0, Ls=1.0, knee_scale=0.9):
    """Schematic 2-link leg from world angles (deg). Returns [(hip),(knee),(foot)] xy."""
    thigh = np.deg2rad(world_hip_deg)                 # thigh pitch about the hip
    knee = np.deg2rad(world_cam_deg) * knee_scale     # knee bend (approx; cam->knee is a linkage)
    hip = np.array([x0, 0.0])
    kn = hip + Lt * np.array([np.sin(thigh), -np.cos(thigh)])
    ft = kn + Ls * np.array([np.sin(thigh + knee), -np.cos(thigh + knee)])
    return np.array([hip, kn, ft])


def animate(data, period, live, save_path):
    import matplotlib
    if not live:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    Ngrid = 240
    ph = np.linspace(0, 1, Ngrid, endpoint=False)
    R = sampled(data, "right", ph)
    L = sampled(data, "left", ph) if data["left"] is not None else None
    W = world_shape(data, ph)                          # canonical cam,hip (world), right convention
    Wl = world_shape(data, (ph + 0.5) % 1.0) if L is not None else None

    fig, (axc, axk, axs) = plt.subplots(1, 3, figsize=(15, 5))
    for a, col, name in ((axc, traj.COL_CAM, "cam"), (axk, traj.COL_HIP, "hip")):
        a.plot(ph, R[:, col], "b", lw=1.5, label="right")
        if L is not None:
            a.plot(ph, L[:, col], "r", lw=1.5, label="left")
        a.set_title(f"{name} (deg)"); a.set_xlabel("phase"); a.grid(alpha=.3); a.legend(fontsize=8)
    mrk_c, = axc.plot([], [], "ko", ms=8)
    mrk_k, = axk.plot([], [], "ko", ms=8)
    axs.set_title("schematic (angles real, geometry approx)")
    axs.set_xlim(-2.2, 2.2); axs.set_ylim(-2.4, 0.6); axs.set_aspect("equal"); axs.grid(alpha=.3)
    (lineR,) = axs.plot([], [], "b-o", lw=3, ms=6)
    (lineL,) = axs.plot([], [], "r-o", lw=3, ms=6)
    footR, footL = [], []
    (traceR,) = axs.plot([], [], "b.", ms=1, alpha=0.4)
    (traceL,) = axs.plot([], [], "r.", ms=1, alpha=0.4)

    def frame(i):
        p = ph[i]
        mrk_c.set_data([p], [R[i, traj.COL_CAM]])
        mrk_k.set_data([p], [R[i, traj.COL_HIP]])
        pr = _leg_points(W[i, 1], W[i, 0], x0=+0.9)
        lineR.set_data(pr[:, 0], pr[:, 1]); footR.append(pr[2])
        traceR.set_data([f[0] for f in footR[-Ngrid:]], [f[1] for f in footR[-Ngrid:]])
        if L is not None:
            pl = _leg_points(Wl[i, 1], Wl[i, 0], x0=-0.9)
            lineL.set_data(pl[:, 0], pl[:, 1]); footL.append(pl[2])
            traceL.set_data([f[0] for f in footL[-Ngrid:]], [f[1] for f in footL[-Ngrid:]])
        return mrk_c, mrk_k, lineR, lineL, traceR, traceL

    interval = 1000.0 * period / Ngrid
    anim = FuncAnimation(fig, frame, frames=Ngrid, interval=interval, blit=True)
    fig.tight_layout()
    if save_path:
        try:
            from matplotlib.animation import PillowWriter
            anim.save(save_path, writer=PillowWriter(fps=max(5, int(Ngrid / period))))
            print(f"saved animation -> {save_path}")
        except Exception as e:
            print(f"could not save animation ({e}); is Pillow installed?")
    if live:
        try:
            plt.show()
        except Exception as e:
            print(f"live window failed ({e}); on a headless machine use --anim out.gif instead.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="fixed_gait/trajectories/gait_recorded.npz")
    ap.add_argument("--png", default="fixed_gait/trajectories/trajectory.png",
                    help="static plot output (set '' to skip)")
    ap.add_argument("--anim", default=None, help="save an animated GIF/MP4 to this path")
    ap.add_argument("--live", action="store_true", help="show an interactive animation (needs display)")
    ap.add_argument("--period", type=float, default=4.0, help="animation seconds per cycle")
    args = ap.parse_args()

    try:
        data = traj.load(args.file)
    except FileNotFoundError:
        print(f"no trajectory at {args.file} — record one first (record_trajectory.py)."); sys.exit(1)

    if args.png:
        static_png(data, args.png)
    if args.anim or args.live:
        animate(data, args.period, args.live, args.anim)


if __name__ == "__main__":
    main()
