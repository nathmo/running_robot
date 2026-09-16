"""The three steady-state gait figures, drawn from a gait_shape.py --npz recording.

    python walk_v2/tools/gait_shape.py --run ... --npz rec.npz --seconds 25 --settle 5
    python walk_v2/tools/gait_figures.py --npz rec.npz --run ... --out walk_v2/results

  gait_cycle    the gait relative to the base: foot path over a cycle, contact bars, foot height
  gait_authors  feedforward / reflex / residual per joint over two strides, plus their amplitudes
  gait_fourier  the joint angle over time: the full command against the latched-Fourier part alone
  gait_actuator the same limit cycle in actuator coordinates
  gait_ee_ghost the foot paths in end-effector space over the translucent robot, side and rear view
                (needs a recording with qpos_full)

Colours are the dataviz reference palette's first three categorical slots (validated all-pairs,
light surface); the aqua slot is below 3:1 on this surface, so every series carries a direct label.
"""
import argparse
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gait
from config import config_from_dict
from gait import GaitParams

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a84"
GRID = "#e6e5e1"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # blue, orange, aqua
JOINTS = ["hip_roll_L", "cam_L", "thigh_L", "hip_roll_R", "cam_R", "thigh_R"]
TP = 2.0 * np.pi


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    })


def load(npz, run):
    z = np.load(npz)
    cfg = config_from_dict(json.loads((Path(run) / "resolved_config.json").read_text())["config"])
    gp = GaitParams.from_cfg(cfg)
    import mujoco
    from plant import resolve
    m = mujoco.MjModel.from_xml_path(resolve(cfg.model_path))
    nid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "nominal_ctrl")
    a = int(m.numeric_adr[nid])
    return z, cfg, gp, m.numeric_data[a:a + 6].copy()


def rebuild(z, cfg, gp, nominal, e, i0, i1):
    """feedforward / reflex / residual / total command, all (T,6) in rad."""
    phi, spec, res = z["phi"][i0:i1, e], z["spec"][i0:i1, e], z["residual"][i0:i1, e]
    grav, gyro, prate = z["grav"][i0:i1, e], z["gyro"][i0:i1, e], z["prate"][i0:i1, e]
    ff = np.stack([gait.feedforward(spec[i], phi[i], nominal, gp, xp=np) for i in range(len(phi))])
    rx = np.zeros_like(ff)
    for i in range(len(phi)):
        ur, up = gait.reflexes(spec[i], grav[i, 1], gyro[i, 0], grav[i, 0], prate[i], gp, xp=np)
        rx[i] = [ur, 0.0, up, ur, 0.0, -up]
    rs = cfg.residual_scale * np.clip(res, -1.0, 1.0)
    return ff, rx, rs, ff + rx + rs, phi


def cycle_profile(phi, sig, n=128):
    """Phase-average a signal over whole clock cycles -> (phase grid, mean, sd, n cycles)."""
    grid = np.linspace(0, TP, n, endpoint=False)
    wr = np.flatnonzero(np.diff(phi) < -np.pi) + 1
    P = [np.interp(grid, phi[a:b], sig[a:b], period=TP)
         for a, b in zip(wr[:-1], wr[1:]) if b - a >= 8]
    P = np.array(P)
    return grid / TP, P.mean(0), P.std(0), len(P)


def tag(ax, x, y, text, color):
    """Direct label: the relief rule for the low-contrast slot, and faster to read than a legend."""
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", color=color,
                fontsize=8, fontweight="bold", va="center", clip_on=False)


def spans_of(g):
    """Contiguous True runs of a boolean series."""
    out, s0 = [], (0 if g[0] else None)
    for i in np.flatnonzero(np.diff(g.astype(int))):
        if g[i]:
            out.append((s0 if s0 is not None else 0, i + 1))
            s0 = None
        else:
            s0 = i + 1
    if s0 is not None:
        out.append((s0, len(g)))
    return out


# --------------------------------------------------------------------- figure 1
def fig_cycle(z, e, i0, i1, dt, out):
    base, rot, toe = z["base"][i0:i1, e], z["rot"][i0:i1, e], z["toe"][i0:i1, e]
    grounded, phi = z["grounded"][i0:i1, e], z["phi"][i0:i1, e]
    fb = np.einsum("tij,tfi->tfj", rot, toe - base[:, None, :]) * 100.0     # cm, base frame
    fig = plt.figure(figsize=(12.4, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], hspace=0.6, wspace=0.22,
                          left=0.06, right=0.95, top=0.81, bottom=0.12)

    # --- sagittal foot path over one averaged cycle, zoomed to the path
    ax = fig.add_subplot(gs[:, 0])
    lo = min(cycle_profile(phi, fb[:, f, 2])[1].min() for f in (0, 1))
    for f, (c, nm) in enumerate([(S1, "left"), (S2, "right")]):
        _, mx, _, _ = cycle_profile(phi, fb[:, f, 0])
        _, mz, _, _ = cycle_profile(phi, fb[:, f, 2])
        _, mg, _, _ = cycle_profile(phi, grounded[:, f])
        mz = mz - lo
        st = mg > 0.5
        ax.plot(np.append(mx, mx[0]), np.append(mz, mz[0]), color=c, lw=1.8, alpha=0.9, zorder=2)
        xs, zs = mx.copy(), mz.copy()
        xs[~st], zs[~st] = np.nan, np.nan
        ax.plot(xs, zs, color=c, lw=6.0, alpha=0.95, solid_capstyle="round", zorder=3)
        i = int(0.16 * len(mx))
        ax.annotate("", (mx[i + 7], mz[i + 7]), (mx[i], mz[i]), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2, mutation_scale=17))
        offs = {0.0: (-8, 9), 0.25: (0, 11), 0.5: (10, 4), 0.75: (0, -13)}
        for frac in (0.0, 0.25, 0.5, 0.75):
            i = int(frac * len(mx))
            ax.plot([mx[i]], [mz[i]], "o", ms=6, mfc=SURFACE, mec=c, mew=1.9, zorder=5)
            if f == 0:
                ax.annotate("phase %g" % frac, (mx[i], mz[i]), xytext=offs[frac], zorder=6,
                            textcoords="offset points",
                            ha="right" if frac == 0.0 else ("left" if frac == 0.5 else "center"),
                            fontsize=7.5, color=MUTED)
        j = int(0.10 * len(mx))
        ax.annotate("%s foot" % nm, (mx[j], mz[j]), xytext=(14, 12 if f == 0 else -16),
                    textcoords="offset points", color=c, fontsize=9, fontweight="bold",
                    ha="left", zorder=7)
    ax.set_xlabel("fore-aft in the base frame (cm)      -> direction of travel")
    ax.set_ylabel("height above the lowest point (cm)")
    ax.set_title("Foot path relative to the base, averaged over one cycle")
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.16)
    ax.set_ylim(-3.5, 21.5)
    ax.annotate("base origin ~90 cm above", xy=(0.5, 0.995), xycoords="axes fraction",
                xytext=(0, -11), textcoords="offset points", ha="center", fontsize=7.5, color=MUTED)
    ax.text(0.015, 0.04, "thick = stance", transform=ax.transAxes, fontsize=8, color=INK2)

    # --- contact diagram, four strides
    n = int(round(4 / (4.0 * dt)))
    tt = np.arange(n) * dt
    ax = fig.add_subplot(gs[0, 1])
    for f, (c, nm) in enumerate([(S1, "left"), (S2, "right")]):
        y = 1 - f
        for a, b in spans_of(grounded[:n, f] > 0.5):
            ax.plot([tt[a], tt[b - 1]], [y, y], color=c, lw=13, solid_capstyle="butt")
        ax.text(-0.015, y, nm, transform=ax.get_yaxis_transform(), ha="right", va="center",
                color=c, fontsize=8.5, fontweight="bold")
    ax.set_ylim(-0.8, 1.8)
    ax.set_yticks([])
    ax.set_xlim(tt[0], tt[-1])
    ax.set_xlabel("time (s)")
    ax.set_title("Stance bars: duty 0.28, no double support, 44 % flight")
    ax.grid(axis="y", visible=False)

    # --- foot height over the same window
    ax = fig.add_subplot(gs[1, 1])
    h = fb[:n, :, 2] - fb[:n, :, 2].min()
    for f, c in [(0, S1), (1, S2)]:
        for a, b in spans_of(grounded[:n, f] > 0.5):
            ax.axvspan(tt[a], tt[b - 1], color=c, alpha=0.10, lw=0)
        ax.plot(tt, h[:, f], color=c, lw=2.0)
    tag(ax, tt[-1], h[-1, 0], " L", S1)
    tag(ax, tt[-1], h[-1, 1], " R", S2)
    ax.set_xlim(tt[0], tt[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("foot height (cm)")
    ax.set_title("Foot height, shaded = that foot's stance")

    fig.suptitle("DASH-01 v2: steady-state gait relative to the base", x=0.06, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.06, 0.885, "S2 runner at 88.5 M steps, nominal plant. 4.00 Hz clock, "
             "250.0 +/- 0.0 ms stride over 79 strides, 3.5 m/s",
             fontsize=9, color=INK2, ha="left")
    fig.savefig(out, dpi=160)
    print("wrote %s" % out)


# --------------------------------------------------------------------- figure 2
def fig_authors(ff, rx, rs, tot, dt, out):
    fig = plt.figure(figsize=(12.4, 7.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.62, wspace=0.26,
                          left=0.06, right=0.96, top=0.85, bottom=0.09)
    n = int(round(2 / (4.0 * dt)))
    tt = np.arange(n) * dt * 1000.0
    for j in range(6):
        ax = fig.add_subplot(gs[j // 3, j % 3])
        ax.axhline(0, color=GRID, lw=1)
        ax.plot(tt, np.degrees(tot[:n, j] - tot[:n, j].mean()), color=INK, lw=1.1,
                ls=(0, (4, 2)), zorder=4)
        ax.plot(tt, np.degrees(ff[:n, j] - ff[:n, j].mean()), color=S1, lw=2.0, zorder=3)
        ax.plot(tt, np.degrees(rx[:n, j] - rx[:n, j].mean()), color=S2, lw=2.0, zorder=2)
        ax.plot(tt, np.degrees(rs[:n, j] - rs[:n, j].mean()), color=S3, lw=2.0, zorder=2)
        ax.set_title(JOINTS[j])
        ax.set_xlim(0, tt[-1])
        if j >= 3:
            ax.set_xlabel("time (ms), two strides")
        if j % 3 == 0:
            ax.set_ylabel("deviation from mean (deg)")
    ax = fig.add_subplot(gs[2, :])
    w, xs = 0.26, np.arange(6)
    for k, (arr, c, nm) in enumerate([(ff, S1, "latched Fourier"), (rx, S2, "reflex"),
                                      (rs, S3, "residual")]):
        vals = [np.degrees(arr[:, j].max() - arr[:, j].min()) for j in range(6)]
        ax.bar(xs + (k - 1) * w, vals, width=w * 0.86, color=c, linewidth=0)
        for x, v in zip(xs, vals):
            if v > 0.4:
                ax.text(x + (k - 1) * w, v + 0.8, "%.0f" % v, ha="center", fontsize=7.5,
                        color=c, fontweight="bold")
        ax.plot([], [], color=c, lw=6, label=nm)
    ax.set_xticks(xs)
    ax.set_xticklabels(JOINTS)
    ax.set_ylabel("peak-to-peak (deg)")
    ax.set_title("How much motion each term contributes")
    ax.legend(loc="upper right", ncols=3, handlelength=1.2)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.18)
    fig.suptitle("DASH-01 v2: who writes the joint command", x=0.06, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    for x, c, nm in [(0.06, S1, "latched Fourier"), (0.20, S2, "reflex"),
                     (0.28, S3, "residual"), (0.37, INK, "total command (dashed)")]:
        fig.text(x, 0.878, nm, color=c, fontsize=8.5, fontweight="bold", ha="left")
    fig.text(0.06, 0.912, "target = feedforward(latched spec, phase) + reflexes + 0.1 x residual. "
             "The latched spec writes 75-85 % of the variance, the residual 3-12 %.",
             fontsize=9, color=INK2, ha="left")
    fig.savefig(out, dpi=160)
    print("wrote %s" % out)


# --------------------------------------------------------------------- figure 3
def fig_fourier(z, e, i0, i1, ff, tot, dt, out):
    q = z["q"][i0:i1, e]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.6), sharex=True)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.80, bottom=0.10, hspace=0.42, wspace=0.24)
    n = int(round(3 / (4.0 * dt)))
    tt = np.arange(n) * dt * 1000.0
    for j, ax in enumerate(axes.ravel()):
        ax.plot(tt, np.degrees(q[:n, j]), color=MUTED, lw=1.3, zorder=1)
        ax.plot(tt, np.degrees(ff[:n, j]), color=S1, lw=2.2, zorder=3)
        ax.plot(tt, np.degrees(tot[:n, j]), color=S2, lw=1.6, ls=(0, (5, 2)), zorder=2)
        ax.set_title(JOINTS[j])
        ax.set_xlim(0, tt[-1])
        if j >= 3:
            ax.set_xlabel("time (ms), three strides")
        if j % 3 == 0:
            ax.set_ylabel("joint angle (deg)")
    fig.suptitle("DASH-01 v2: the joint angle, full command against the latched Fourier alone",
                 x=0.06, ha="left", fontsize=13, fontweight="bold", color=INK)
    for x, c, nm in [(0.06, S1, "latched Fourier only"), (0.21, S2, "full command (dashed)"),
                     (0.37, MUTED, "measured joint")]:
        fig.text(x, 0.845, nm, color=c, fontsize=8.5, fontweight="bold", ha="left")
    fig.text(0.06, 0.885, "The gap between solid blue and dashed orange is the reflex plus the "
             "residual. The joint then does something smoother and smaller again: the plant filters "
             "the 2nd and 3rd harmonics,",
             fontsize=9, color=INK2, ha="left")
    fig.text(0.06, 0.862, "and the PD leaves 5-19 deg rms of tracking error - the hip-rolls execute "
             "only ~8 deg of a 22 deg command.", fontsize=9, color=INK2, ha="left")
    fig.savefig(out, dpi=160)
    print("wrote %s" % out)


# --------------------------------------------------------------------- figure 4
def fig_actuator(z, e, i0, i1, tot, dt, plant_range, out):
    """The same limit cycle drawn in actuator coordinates rather than task space."""
    q, qd, phi = z["q"][i0:i1, e], z["qd"][i0:i1, e], z["phi"][i0:i1, e]
    grounded = z["grounded"][i0:i1, e]
    fig = plt.figure(figsize=(12.4, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], hspace=0.62, wspace=0.20,
                          left=0.065, right=0.95, top=0.80, bottom=0.10)

    # --- configuration loop in the sagittal actuator pair (cam, thigh)
    ax = fig.add_subplot(gs[:, 0])
    for f, (c, nm, ic, it) in enumerate([(S1, "left", 1, 2), (S2, "right", 4, 5)]):
        _, mc, _, _ = cycle_profile(phi, np.degrees(q[:, ic]))
        _, mt, _, _ = cycle_profile(phi, np.degrees(q[:, it]))
        _, cc, _, _ = cycle_profile(phi, np.degrees(tot[:, ic]))
        _, ct, _, _ = cycle_profile(phi, np.degrees(tot[:, it]))
        _, mg, _, _ = cycle_profile(phi, grounded[:, f])
        st = mg > 0.5
        ax.plot(np.append(cc, cc[0]), np.append(ct, ct[0]), color=c, lw=1.2, ls=(0, (4, 2)),
                alpha=0.75, zorder=2)
        ax.plot(np.append(mc, mc[0]), np.append(mt, mt[0]), color=c, lw=2.0, zorder=3)
        xs, ys = mc.copy(), mt.copy()
        xs[~st], ys[~st] = np.nan, np.nan
        ax.plot(xs, ys, color=c, lw=6.5, alpha=0.95, solid_capstyle="round", zorder=4)
        i = int(0.16 * len(mc))
        ax.annotate("", (mc[i + 7], mt[i + 7]), (mc[i], mt[i]), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2, mutation_scale=17))
        offs = {0.0: (-6, 10), 0.25: (-12, -2), 0.5: (0, -14), 0.75: (10, 4)}
        for frac in (0.0, 0.25, 0.5, 0.75):
            i = int(frac * len(mc))
            ax.plot([mc[i]], [mt[i]], "o", ms=6, mfc=SURFACE, mec=c, mew=1.9, zorder=5)
            if f == 0:
                ax.annotate("phase %g" % frac, (mc[i], mt[i]), xytext=offs[frac], zorder=6,
                            textcoords="offset points", fontsize=7.5, color=MUTED,
                            ha="right" if frac == 0.25 else
                               ("left" if frac == 0.75 else "center"))
        j = int(0.40 * len(mc))
        ax.annotate("%s leg" % nm, (mc[j], mt[j]), xytext=(-14 if f == 0 else 12, 0),
                    textcoords="offset points", color=c, fontsize=9, fontweight="bold",
                    ha="right" if f == 0 else "left", va="center", zorder=7)
    ax.set_xlabel("cam angle (deg)")
    ax.set_ylabel("thigh angle (deg)")
    ax.set_title("Configuration loop in the sagittal actuator pair")
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.24)
    ax.text(0.015, 0.03, "solid = measured, dashed = commanded, thick = stance",
            transform=ax.transAxes, fontsize=8, color=INK2)
    # inset: the same loops inside the full actuator travel
    lo_c, hi_c = plant_range[1]
    lo_t, hi_t = plant_range[2]
    ins = ax.inset_axes([0.635, 0.635, 0.35, 0.35])
    for f, (c, ic, it) in enumerate([(S1, 1, 2), (S2, 4, 5)]):
        _, mc, _, _ = cycle_profile(phi, np.degrees(q[:, ic]))
        _, mt, _, _ = cycle_profile(phi, np.degrees(q[:, it]))
        ins.plot(np.append(mc, mc[0]), np.append(mt, mt[0]), color=c, lw=1.4)
    ins.add_patch(plt.Rectangle((lo_c, lo_t), hi_c - lo_c, hi_t - lo_t, fill=False,
                                ec=MUTED, ls=(0, (4, 3)), lw=1.1))
    ins.set_xlim(lo_c * 1.1, hi_c * 1.1)
    ins.set_ylim(lo_t * 1.15, hi_t * 1.15)
    ins.set_aspect("equal", adjustable="box")
    ins.set_xticks([]); ins.set_yticks([])
    ins.grid(False)
    for sp in ins.spines.values():
        sp.set_visible(False)
    ins.set_facecolor(SURFACE)
    ins.set_title("inside the full actuator travel", fontsize=7.5, color=MUTED,
                  loc="center", pad=3, fontweight="normal")

    # --- phase portraits, one per actuator family
    caps = [10.3, 22.01, 22.01]
    for row, (nm, il, ir) in enumerate([("hip_roll", 0, 3), ("cam", 1, 4), ("thigh", 2, 5)]):
        ax = fig.add_subplot(gs[row, 1])
        for c, i in [(S1, il), (S2, ir)]:
            _, ma, _, _ = cycle_profile(phi, np.degrees(q[:, i]))
            _, mv, _, _ = cycle_profile(phi, np.degrees(qd[:, i]))
            ax.plot(np.append(ma, ma[0]), np.append(mv, mv[0]), color=c, lw=1.9)
        cap = np.degrees(caps[row])
        ymax = max(abs(np.degrees(qd[:, [il, ir]])).max(), 1.0)
        if cap < 1.35 * ymax:
            ax.axhline(cap, color=MUTED, ls=(0, (4, 3)), lw=1.0)
            ax.axhline(-cap, color=MUTED, ls=(0, (4, 3)), lw=1.0)
            ax.annotate("no-load speed", (0.99, cap), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=7, color=MUTED)
        else:
            ax.annotate("no-load speed %.0f deg/s, off scale" % cap, (0.99, 0.06),
                        xycoords="axes fraction", ha="right", fontsize=7, color=MUTED)
        ax.set_title(nm)
        ax.set_ylabel("deg/s")
        if row == 2:
            ax.set_xlabel("joint angle (deg)")
        if row == 0:
            ax.text(0.02, 0.96, "left", transform=ax.transAxes, color=S1, fontsize=8,
                    fontweight="bold", va="top")
            ax.text(0.12, 0.96, "right", transform=ax.transAxes, color=S2, fontsize=8,
                    fontweight="bold", va="top")

    fig.suptitle("DASH-01 v2: the same limit cycle in actuator space", x=0.065, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.065, 0.885, "Left: measured loop (solid, thick = stance) against the commanded loop "
             "(dashed) in the cam-thigh plane. Right: phase portraits, one closed orbit per "
             "actuator.", fontsize=9, color=INK2, ha="left")
    fig.text(0.065, 0.855, "The gait uses 7-9 % of the hip-roll range, 16-18 % of the cam and "
             "34-36 % of the thigh: the actuators are nowhere near their travel limits.",
             fontsize=9, color=INK2, ha="left")
    fig.savefig(out, dpi=160)
    print("wrote %s" % out)



# --------------------------------------------------------------------- figure 5
def ghost_model(cfg):
    """The plant's MJCF for drawing only: flat sky in the surface colour, orthographic camera,
    legs tinted to their path colour and every mesh translucent."""
    import mujoco
    from plant import resolve
    s = mujoco.MjSpec.from_file(resolve(cfg.model_path))
    bg = [int(SURFACE[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    s.add_texture(name="ghost_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                  builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=bg, rgb2=bg, width=16, height=16)
    m = s.compile()
    m.vis.global_.orthographic = 1
    m.vis.global_.offwidth = m.vis.global_.offheight = 2600
    m.vis.headlight.ambient[:] = 0.45
    m.vis.headlight.diffuse[:] = 0.45
    m.vis.headlight.specular[:] = 0.0
    rgb = lambda h: [int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        c = S1 if "Left" in nm else (S2 if "Right" in nm else MUTED)
        m.geom_rgba[g] = rgb(c) + [0.13]            # closed meshes: every pixel is >= 2 faces
    return m


def render_ortho(m, d, azimuth, center, half_w, half_h, px_per_m):
    """Orthographic render looking horizontally from `azimuth`. Returns the RGBA image (sky made
    transparent) and its extent in camera-plane coordinates (along camera-right, along up), plus
    the two axes, so world points map as (p . right, p . up)."""
    import mujoco
    W, H = int(round(2 * half_w * px_per_m)), int(round(2 * half_h * px_per_m))
    m.vis.global_.fovy = 2.0 * half_h                   # orthographic: the vertical extent, metres
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.distance, cam.azimuth, cam.elevation = 3.0, azimuth, 0.0
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[2] = 1                                # visual meshes only: no floor, no spheres
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, camera=cam, scene_option=opt)
    img = r.render()
    # scene.camera holds the two stereo eyes, each ipd/2 (3.4 cm) off-centre; a mono render
    # looks through their mean
    g0, g1 = r.scene.camera[0], r.scene.camera[1]
    pos = 0.5 * (np.array(g0.pos) + np.array(g1.pos))
    fwd, up = np.array(g0.forward), np.array(g0.up)
    right = np.cross(fwd, up)
    a0 = pos @ right + 0.5 * (g0.frustum_center + g1.frustum_center)
    b0 = pos @ up
    hw = (g0.frustum_top - g0.frustum_bottom) * 0.5 * W / H
    ext = (a0 - hw, a0 + hw, b0 + g0.frustum_bottom, b0 + g0.frustum_top)
    sky = img[0, 0].astype(int)
    alpha = (np.abs(img.astype(int) - sky).max(-1) > 2).astype(np.uint8) * 255
    r.close()
    return np.dstack([img, alpha]), ext, right, up


def fig_ghost(z, e, i0, i1, cfg, out):
    """The gait in end-effector space: both foot paths in the base frame, over the robot itself
    drawn translucent in the same frame at left mid-stance. Side view and rear view."""
    import mujoco
    from matplotlib.lines import Line2D
    base, rot, toe = z["base"][i0:i1, e], z["rot"][i0:i1, e], z["toe"][i0:i1, e]
    grounded, phi, qfull = z["grounded"][i0:i1, e], z["phi"][i0:i1, e], z["qpos_full"][i0:i1, e]
    fb = np.einsum("tij,tfi->tfj", rot, toe - base[:, None, :])            # m, base frame
    dt = float(z["dt"])

    # averaged cycle per foot, and the pose to draw: left mid-stance, in the middle of the window
    prof = []
    for f in range(2):
        gph, mx, _, ncyc = cycle_profile(phi, fb[:, f, 0])
        prof.append(dict(x=mx, y=cycle_profile(phi, fb[:, f, 1])[1],
                         z=cycle_profile(phi, fb[:, f, 2])[1],
                         st=cycle_profile(phi, grounded[:, f])[1] > 0.5))
    ang = TP * gph[prof[0]["st"]]
    ph_pose = np.mod(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()), TP)
    mid = len(phi) // 2
    k = mid + int(np.argmin(np.abs(np.angle(np.exp(1j * (phi[mid:mid + 40] - ph_pose))))))

    m = ghost_model(cfg)
    d = mujoco.MjData(m)
    names = ["base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw"]
    base_q = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in names]
    d.qpos[:] = qfull[k]
    d.qpos[base_q] = 0.0                                # world frame == base frame
    mujoco.mj_forward(m, d)
    gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "foot_%s_col" % s) for s in "LR"]
    err = np.abs(d.geom_xpos[gids] - fb[k]).max() * 1000.0
    print("ghost pose: tick %d, phase %.2f; drawn toes vs recorded toes %.2f mm" % (k, ph_pose / TP, err))
    toe_r = float(m.geom_size[gids[0], 0])
    z_ground = float(np.mean(np.concatenate([fb[grounded[:, f] > 0.5, f, 2] for f in (0, 1)]))) - toe_r

    fig = plt.figure(figsize=(10.4, 8.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.12,
                          left=0.08, right=0.985, top=0.80, bottom=0.075)
    views = [(gs[0], 90.0, "Seen from the right side", "fore-aft, base frame (cm)   -> direction of travel"),
             (gs[1], 0.0, "Seen from behind", "lateral, base frame (cm)   + = robot's right")]
    for spec, az, title, xl in views:
        ax = fig.add_subplot(spec)
        img, ext, right, up = render_ortho(m, d, az, [0.0, 0.0, -0.45], 0.85, 0.62, 1500.0)
        cm = lambda p: (100.0 * (p @ right), 100.0 * (p @ up))
        ax.imshow(img, extent=[100.0 * v for v in ext], interpolation="antialiased", zorder=1)
        # the robot's own extent on screen, to crop to
        ys, xs = np.nonzero(img[..., 3])
        H, W = img.shape[:2]
        rx = 100.0 * (ext[0] + (ext[1] - ext[0]) * np.array([xs.min(), xs.max()]) / W)
        rz = 100.0 * (ext[3] - (ext[3] - ext[2]) * np.array([ys.max(), ys.min()]) / H)
        ax.axhline(100.0 * z_ground, color=INK2, lw=1.0, ls=(0, (5, 3)), zorder=2)
        for f, c in [(0, S1), (1, S2)]:
            P = np.stack([prof[f]["x"], prof[f]["y"], prof[f]["z"]], 1)
            a, b = cm(P)
            ax.plot(np.append(a, a[0]), np.append(b, b[0]), color=c, lw=2.0, zorder=4)
            sa, sb = a.copy(), b.copy()
            sa[~prof[f]["st"]], sb[~prof[f]["st"]] = np.nan, np.nan
            ax.plot(sa, sb, color=c, lw=6.0, solid_capstyle="round", zorder=5)
            i = int(0.16 * len(a))
            ax.annotate("", (a[i + 6], b[i + 6]), (a[i], b[i]), zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0, mutation_scale=15))
            for frac in (0.0, 0.25, 0.5, 0.75):
                i = int(frac * len(a))
                ax.plot([a[i]], [b[i]], "o", ms=6, mfc=SURFACE, mec=c, mew=1.8, zorder=6)
            pa, pb = cm(fb[k, f])
            ax.plot([pa], [pb], "o", ms=10, mfc=c, mec=SURFACE, mew=2.0, zorder=7)
        bx, bz = cm(np.zeros(3))
        ax.plot([bx], [bz], "+", ms=12, mew=2.0, color=INK, zorder=7)
        ax.annotate("base origin", (bx, bz), xytext=(0, -13), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color=INK, zorder=7)
        allx = np.concatenate([rx] + [cm(np.stack([p["x"], p["y"], p["z"]], 1))[0] for p in prof])
        allz = np.concatenate([rz, [100.0 * z_ground]])
        ax.set_xlim(allx.min() - 6.0, allx.max() + 6.0)
        ax.set_ylim(allz.min() - 5.0, allz.max() + 5.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axisbelow(True)
        ax.set_title(title)
        ax.set_xlabel(xl)
        if az == 90.0:
            ax.set_ylabel("height, base frame (cm)")
            ax.annotate("ground, stance average", (ax.get_xlim()[0], 100.0 * z_ground),
                        xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK2)

    handles = [Line2D([], [], color=S1, lw=2.0, label="left foot"),
               Line2D([], [], color=S2, lw=2.0, label="right foot"),
               Line2D([], [], color=MUTED, lw=6.0, label="stance"),
               Line2D([], [], ls="", marker="o", ms=6, mfc=SURFACE, mec=INK2, mew=1.8,
                      label="clock phase 0, .25, .5, .75"),
               Line2D([], [], ls="", marker="o", ms=10, mfc=INK2, mec=SURFACE, mew=2.0,
                      label="feet in the pose drawn")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.072, 0.862), ncols=5,
               handlelength=1.8, columnspacing=1.4, fontsize=8.5)
    v = float(np.mean(z["vbody"][i0:i1, e, 0]))
    sz = ["%s %.1f x %.1f cm" % (nm, 100 * np.ptp(prof[f]["x"]), 100 * np.ptp(prof[f]["z"]))
          for f, nm in [(0, "left"), (1, "right")]]
    fig.suptitle("DASH-01 v2: the running gait in end-effector space", x=0.08, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    for yy, line in [(0.925, "Foot paths in the base frame, averaged over %d cycles, over the robot "
                             "drawn translucent in the same frame at left mid-stance (phase %.2f)."
                      % (ncyc, ph_pose / TP)),
                     (0.903, "S2 runner at 88.5 M steps, nominal plant, %.2f m/s over %.0f s. "
                             "Averaged path, fore-aft x vertical: %s, %s."
                      % ((v, (i1 - i0) * dt) + tuple(sz))),
                     (0.881, "Legs are tinted like their path. The path is the centre of the foot's "
                             "collision sphere; the ground line sits one sphere radius below it.")]:
        fig.text(0.08, yy, line, fontsize=9, color=INK2, ha="left")
    fig.savefig(out, dpi=160)
    print("wrote %s" % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="walk_v2/results")
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--settle", type=float, default=5.0)
    ap.add_argument("--seconds", type=float, default=25.0)
    args = ap.parse_args()
    style()
    z, cfg, gp, nominal = load(args.npz, args.run)
    dt = float(z["dt"])
    i0, i1 = int(args.settle / dt), int(args.seconds / dt)
    ff, rx, rs, tot, phi = rebuild(z, cfg, gp, nominal, args.env, i0, i1)
    out = Path(args.out)
    fig_cycle(z, args.env, i0, i1, dt, out / "gait_cycle.png")
    fig_authors(ff, rx, rs, tot, dt, out / "gait_authors.png")
    fig_fourier(z, args.env, i0, i1, ff, tot, dt, out / "gait_fourier.png")
    import mujoco
    from plant import resolve
    m = mujoco.MjModel.from_xml_path(resolve(cfg.model_path))
    rng = [np.degrees(m.jnt_range[m.actuator_trnid[a, 0]]) for a in range(6)]
    fig_actuator(z, args.env, i0, i1, tot, dt, rng, out / "gait_actuator.png")
    if "qpos_full" in z.files:                          # recordings from before 2026-09-16 lack it
        fig_ghost(z, args.env, i0, i1, cfg, out / "gait_ee_ghost.png")


if __name__ == "__main__":
    main()
