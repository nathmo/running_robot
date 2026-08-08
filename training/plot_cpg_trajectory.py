"""Plot the FIXED foot trajectory the CPG hard-codes, and what it costs.

    python training/plot_cpg_trajectory.py [--preset m5_CPG] [--out <png>]

Three panels, each answering one question about the mapping in cpg_gait.assemble:
  1. what the hard-coded loop IS      — the (dx, dz) path the oscillator drives, phase by phase
  2. how much of the leg it uses      — that loop drawn on the MEASURED reachable workspace
  3. what the motors actually see     — the cam/thigh commands the loop reconstructs to

The loop is Bellegarda & Ijspeert's CPG-RL foot trajectory: a sinusoid fore-aft plus a one-sided
lift that is nonzero only in swing. The one adaptation is that the lift window is slaved to the
reward's phase-gated contact schedule (fourier_gait.stance_indicator) instead of the paper's
sin(theta) > 0, so the generator and the reward agree on which phase is stance at any stance_ratio.
"""
import argparse
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cpg_gait
from config import get_config

# dataviz reference palette (light mode). Categorical slots 1-3 only: that subset is the documented
# all-pairs-validated set. Text/axis wear ink tokens, never a series colour.
SURFACE, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
STANCE, SWING, JOINT2 = "#2a78d6", "#eb6834", "#1baf7a"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=1.0)
    ax.grid(True, color=AXIS, lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="m5_CPG")
    ap.add_argument("--sr", type=float, default=0.5, help="stance ratio the loop is drawn at")
    ap.add_argument("--out", default=str(PKG_DIR / "runs_dl" / "cpg_trajectory.png"))
    args = ap.parse_args()

    c = get_config(args.preset)
    lut = cpg_gait.load_lut(c.cpg_lut)
    th = np.linspace(0, 2 * np.pi, 721)
    stance_mask = cpg_gait.swing_bump(th, args.sr) == 0.0

    def loop(r):
        return (c.cpg_stride * r * np.cos(th),
                c.cpg_clearance * r * cpg_gait.swing_bump(th, args.sr))

    dx, dz = loop(1.0)
    sw = ~stance_mask

    # Layout: the spatial panels keep TRUE aspect (a foot path drawn with a stretched z axis lies
    # about the gait), and the loop is genuinely a wide flat sliver -- 0.56 m of travel against
    # 0.06 m of lift -- so it gets its own full-width row instead of being squashed beside others.
    fig = plt.figure(figsize=(14.5, 9.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.45], hspace=0.60, wspace=0.22,
                          left=0.07, right=0.97, top=0.88, bottom=0.09)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # ---- 1. the hard-coded loop, at true scale --------------------------------------------
    ax = ax1
    style(ax)
    for r in (0.25, 0.5, 0.75):                 # r scales the SAME shape homothetically
        gx, gz = loop(r)
        ax.plot(gx, gz, color=MUTED, lw=1.0, alpha=0.5, zorder=1)
    ax.plot(dx[stance_mask], dz[stance_mask], color=STANCE, lw=2.0, solid_capstyle="round",
            zorder=3, label="Stance — foot on ground, travels back")
    ax.plot(dx[sw], dz[sw], color=SWING, lw=2.0, solid_capstyle="round", zorder=3,
            label="Swing — lifted, returns forward")
    for frac, col in ((0.30, STANCE), (0.80, SWING)):     # travel direction
        i = int(frac * len(th))
        ax.annotate("", xy=(dx[i + 10], dz[i + 10]), xytext=(dx[i], dz[i]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, mutation_scale=15))
    ax.scatter([0], [0], s=30, color=INK, zorder=5)
    ax.annotate("nominal foot", xy=(0, 0), xytext=(0.015, -0.017), fontsize=8, color=INK2)
    ax.annotate("r = 0.25 / 0.5 / 0.75\nsame shape, scaled", xy=(-0.21, 0.0225),
                xytext=(-0.30, 0.045), fontsize=8, color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("fore-aft   dx  [m]")
    ax.set_ylabel("lift   dz  [m]")
    ax.set_title("1.  The hard-coded loop, drawn at true scale   (r = 1, stance ratio %.2f)\n"
                 "dx = %.2f·r·cos θ      dz = %.2f·r·bump(θ, sr)"
                 % (args.sr, c.cpg_stride, c.cpg_clearance), fontsize=10.5, pad=12)
    ax.set_aspect("equal")
    ax.set_ylim(-0.028, 0.075)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2,
              handlelength=1.6, borderaxespad=0.2)

    # ---- 2. loop vs the measured reachable workspace ---------------------------------------
    ax = ax2
    style(ax)
    z = np.load(cpg_gait._lut_path(c.cpg_lut), allow_pickle=True)
    P = z["fwd"].reshape(-1, 2)
    P = P[P[:, 1] <= 0.40]                       # drop the 4-bar's folded branch
    ax.scatter(P[:, 0], P[:, 1], s=5, color=AXIS, alpha=0.8, linewidths=0, zorder=1,
               label="Reachable — measured, 3577 sims")
    ax.plot(dx[stance_mask], dz[stance_mask], color=STANCE, lw=2.2, zorder=3, label="Stance")
    ax.plot(dx[sw], dz[sw], color=SWING, lw=2.2, zorder=3, label="Swing")
    ax.set_xlabel("fore-aft   dx  [m]\n\nA 1-D curve inside a 2-D space —\n"
                  "and no lateral (dy) axis exists at all")
    ax.set_ylabel("lift   dz  [m]")
    ax.set_title("2.  What it uses of the leg's workspace", fontsize=10.5, pad=12)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=1.6)

    # ---- 3. the joint commands it reconstructs to ------------------------------------------
    ax = ax3
    style(ax)
    J = np.array([cpg_gait.foot_ik(x, zz, lut) for x, zz in zip(dx, dz)])
    ax.plot(th, J[:, 0], color=STANCE, lw=2.0, label="cam")
    ax.plot(th, J[:, 1], color=SWING, lw=2.0, label="thigh")
    ax.axhline(0, color=AXIS, lw=1.0, zorder=0)
    lo, hi = J.min() - 0.10, J.max() + 0.16
    ax.axvspan(2 * np.pi * args.sr, 2 * np.pi, color=AXIS, alpha=0.30, lw=0, zorder=0)
    ax.text(2 * np.pi * (args.sr + 1) / 2, hi - 0.035, "swing window", ha="center", va="top",
            fontsize=8.5, color=MUTED)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("gait phase  θ  [rad]\n\nThe policy can scale and re-time this,\n"
                  "but never reshape it")
    ax.set_ylabel("joint command, Δ from nominal  [rad]")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax.set_title("3.  What the motors are commanded", fontsize=10.5, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=1.6)

    fig.suptitle("The foot trajectory CPG-RL hard-codes  —  RL tunes the oscillator (μ, ω, ψ), never this shape",
                 fontsize=13, color=INK, y=0.965)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=145, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
