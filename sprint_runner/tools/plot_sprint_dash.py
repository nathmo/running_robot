#!/usr/bin/env python3
"""Thesis figures: the learned runner (sprint_m3_mit_s0) on the simulated 100 m dash.

Two passes over ONE greedy, seeded episode (deterministic policy + seeded reset => bit-identical
trajectories), because rendering every control step of a 33 s dash to keep 8 frames is ~6 GB of
pixels:
  pass 1  rolls the episode headless and logs per-control-step time, base x, forward body speed
          and foot contacts (via env.on_control_step -- the only place raw.data is safe to read;
          after a terminal step SB3 has already auto-reset the env).
  pass 2  re-seeds the same episode and renders only the frame indices the filmstrip needs.

Outputs (results/):
  sprint_dash_run.{png,pdf}     filmstrip of one full stride at steady state + the foot-contact
                                raster around it (the "is it actually stepping" evidence)
  sprint_dash_stats.{png,pdf}   forward speed and distance vs time, line crossing marked
  sprint_dash_frame.png         one full-res still for slides

Usage:
  python tools/plot_sprint_dash.py                      # defaults to sprint_m3_mit_s0, seed 1000
  python tools/plot_sprint_dash.py --strip-t 20 --frames 8
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

# palette shared with the other thesis figures (plot_imu_noise.py)
C_SPEED, C_DIST, C_L, C_R = "#2a78d6", "#1baf7a", "#2a78d6", "#eb6834"
INK, INK2, GRID, CRIT = "#0b0b0b", "#52514e", "#e1e0d9", "#d03b3b"


def rollout(model_, venv, raw, seed, want_frames=None, cam_cfg=None):
    """One greedy episode. Logs stats every control step; if want_frames is a {step_index}
    set, renders those steps and returns them as {step_index: rgb}."""
    frames = {}
    renderer = None
    if want_frames:
        import mujoco
        raw.model.vis.global_.offwidth = max(raw.model.vis.global_.offwidth, 1600)
        raw.model.vis.global_.offheight = max(raw.model.vis.global_.offheight, 1200)
        renderer = mujoco.Renderer(raw.model, 1200, 1600)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(raw.model, cam)
        cam.distance, cam.elevation, cam.azimuth = cam_cfg
    log = dict(t=[], x=[], vx=[], cL=[], cR=[])
    n = [0]

    def on_ctrl():
        n[0] += 1
        log["t"].append(n[0] * raw.control_dt)
        log["x"].append(float(raw.data.qpos[0]))
        log["vx"].append(float(raw._vel_body()[0]))
        c = raw._foot_contacts()
        log["cL"].append(bool(c[0]))
        log["cR"].append(bool(c[1]))
        if renderer is not None and n[0] in want_frames:
            cam.lookat[:] = raw.data.qpos[:3]
            renderer.update_scene(raw.data, cam)
            frames[n[0]] = renderer.render().copy()

    raw.on_control_step = on_ctrl
    venv.seed(seed)
    obs = venv.reset()
    done, sprint = [False], None
    while not done[0]:
        a, _ = model_.predict(obs, deterministic=True)
        obs, r, done, info = venv.step(a)
        sprint = info[0].get("sprint", sprint)
    raw.on_control_step = None
    return {k: np.asarray(v) for k, v in log.items()}, sprint, frames


def touchdowns(contact, t):
    """Times of rising edges of a contact trace."""
    c = np.asarray(contact, bool)
    edges = np.flatnonzero(c[1:] & ~c[:-1]) + 1
    return t[edges]


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8.5, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.linewidth": 0.8, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    print(f"wrote {OUT / stem}.png/.pdf")


def fig_stats(log, sprint, avg_v):
    fig, (ax_v, ax_x) = plt.subplots(
        1, 2, figsize=(6.3, 2.35), dpi=300, gridspec_kw={"wspace": 0.32})
    for ax in (ax_v, ax_x):
        ax.grid(True, color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3)
        ax.set_xlabel("time (s)")
        ax.set_xlim(0, log["t"][-1])

    # forward speed: raw 200 Hz trace faint, 0.5 s rolling mean on top
    k = int(round(0.5 / (log["t"][1] - log["t"][0])))
    smooth = np.convolve(log["vx"], np.ones(k) / k, mode="same")
    ax_v.plot(log["t"], log["vx"], lw=0.4, color=C_SPEED, alpha=0.30)
    ax_v.plot(log["t"], smooth, lw=1.2, color=C_SPEED)
    ax_v.axhline(avg_v, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax_v.text(0.4, avg_v + 0.12, f"dash average {avg_v:.2f} m/s", fontsize=7, color=INK2)
    ax_v.set_ylabel("forward speed (m/s)")
    ax_v.set_ylim(bottom=0)

    # distance: the dash itself
    x0 = log["x"][0]
    ax_x.plot(log["t"], log["x"] - x0, lw=1.2, color=C_DIST)
    ax_x.axhline(100.0, color=INK2, lw=0.8, ls=(0, (4, 3)))
    if sprint and sprint.get("t_line") is not None:
        tl = sprint["t_line"]
        ax_x.plot([tl], [100.0], "o", ms=4, color=CRIT, zorder=5)
        ax_x.annotate(f"100 m in {tl:.1f} s", (tl, 100.0), xytext=(-8, -16),
                      textcoords="offset points", ha="right", fontsize=7.5, color=INK)
    ax_x.set_ylabel("distance (m)")
    ax_x.set_ylim(0, 112)
    return fig


def fig_filmstrip(log, frames, order, t0, t1, crop_frac, side, vcrop):
    """Top: the stride frames hstacked (single image). Bottom: contact raster around it."""
    imgs = []
    for idx in order:
        f = frames[idx]
        h, w = f.shape[:2]
        cw = int(w * crop_frac)
        lo = (w - cw) // 2
        imgs.append(f[int(h * vcrop[0]):int(h * vcrop[1]), lo:lo + cw])
    gap = np.full((imgs[0].shape[0], 6, 3), 255, np.uint8)
    strip = np.hstack([x for img in imgs for x in (img, gap)][:-1])

    fig = plt.figure(figsize=(6.3, 3.15), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.0], hspace=0.34)
    ax_s = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1])

    ax_s.imshow(strip)
    ax_s.set_yticks([])
    ax_s.spines[:].set_visible(False)
    fw = imgs[0].shape[1]
    centers = [i * (fw + 6) + fw / 2 for i in range(len(imgs))]
    dt_ms = [(log["t"][i - 1] - t0) * 1e3 for i in order]
    ax_s.set_xticks(centers)
    ax_s.set_xticklabels([f"{m:+.0f}" for m in dt_ms], fontsize=7)
    ax_s.tick_params(length=0)
    ax_s.set_xlabel(f"time after {side}-foot touchdown (ms)", fontsize=8)

    # contact raster: ~5 strides centered on the strip
    w0, w1 = t0 - 1.0, t1 + 1.0
    m = (log["t"] >= w0) & (log["t"] <= w1)
    tt = log["t"][m]
    for row, (c, col, lab) in enumerate((
            (log["cR"][m], C_R, "right foot"), (log["cL"][m], C_L, "left foot"))):
        on = np.flatnonzero(c)
        if on.size:
            splits = np.split(on, np.flatnonzero(np.diff(on) > 1) + 1)
            spans = [(tt[s[0]], tt[s[-1]] - tt[s[0]] + (tt[1] - tt[0])) for s in splits]
            ax_c.broken_barh(spans, (row + 0.12, 0.76), color=col, lw=0)
    ax_c.axvspan(t0, t1, color=GRID, alpha=0.55, zorder=0)
    for i in order:
        ax_c.axvline(log["t"][i - 1], color=INK2, lw=0.5, ls=(0, (2, 2)), alpha=0.7)
    ax_c.set_xlim(w0, w1)
    ax_c.set_ylim(0, 2)
    ax_c.set_yticks([0.5, 1.5])
    ax_c.set_yticklabels(["right", "left"], fontsize=7.5)
    ax_c.set_xlabel("time (s)  --  shaded span = the stride shown above", fontsize=8)
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.tick_params(length=3)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "walk_mit/runs/sprint_m3_mit_s0"))
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--frames", type=int, default=7)
    ap.add_argument("--strip-t", type=float, default=16.0,
                    help="pick the stride starting at the first left touchdown after this time")
    ap.add_argument("--crop", type=float, default=0.42, help="center width fraction kept per frame")
    ap.add_argument("--vcrop", default="0.30,0.99", help="kept height range per frame (sky cut)")
    ap.add_argument("--cam", default="2.3,-14,270",
                    help="distance,elevation,azimuth (270 = lit side, running left-to-right)")
    args = ap.parse_args()

    run = Path(args.run).resolve()
    pkg = "walk_mit" if "walk_mit" in run.parts else "training"
    sys.path.insert(0, str(ROOT / pkg))
    from evaluate import build                                    # noqa: E402

    model_, venv, raw = build(run, None, None)
    print(f"[dash] pass 1: stats rollout, seed {args.seed}")
    log, sprint, _ = rollout(model_, venv, raw, args.seed)
    avg_v = sprint["d"] / sprint["t"] if sprint else float(np.mean(log["vx"]))
    tl = sprint.get("t_line") if sprint else None
    print(f"[dash] {sprint['d']:.1f} m in {sprint['t']:.2f} s"
          f"   line {tl if tl is None else f'{tl:.2f} s'}   avg {avg_v:.2f} m/s")

    # one full stride at steady state, keyed on whichever foot actually carries the gait: the
    # learned solution is a strongly asymmetric hop (right foot ~0.20 contact duty, left ~0.01,
    # 80% of control steps fully airborne), so touchdowns of the busier foot define the cycle.
    n_td = {s: len(touchdowns(log[f"c{s[0].upper()}"], log["t"])) for s in ("left", "right")}
    side = max(n_td, key=n_td.get)
    duty = {s: float(np.mean(log[f"c{s[0].upper()}"])) for s in ("left", "right")}
    fly = float(np.mean(~log["cL"] & ~log["cR"]))
    print(f"[dash] contact duty L {duty['left']:.2f} / R {duty['right']:.2f}, "
          f"airborne {fly:.2f} -> stride keyed on the {side} foot")
    td = touchdowns(log[f"c{side[0].upper()}"], log["t"])
    td = td[td >= args.strip_t]
    t0, t1 = float(td[0]), float(td[1])
    print(f"[dash] stride {t0:.3f} -> {t1:.3f} s  (period {(t1 - t0) * 1e3:.0f} ms, "
          f"cadence {1 / (t1 - t0):.2f} Hz)")
    times = np.linspace(t0, t1, args.frames)
    dt = log["t"][1] - log["t"][0]
    order = [int(round(t / dt)) for t in times]          # step index n (1-based in the log)
    # slide hero frame: the fastest instant of this stride
    seg = slice(order[0] - 1, order[-1])
    hero = int(np.argmax(log["vx"][seg])) + order[0]
    cam_cfg = tuple(float(v) for v in args.cam.split(","))

    print(f"[dash] pass 2: rendering {len(set(order)) + 1} frames")
    log2, _, frames = rollout(model_, venv, raw, args.seed,
                              want_frames=set(order) | {hero}, cam_cfg=cam_cfg)
    assert np.array_equal(log2["cL"], log["cL"]), "replay diverged from the stats pass"

    style()
    vcrop = tuple(float(v) for v in args.vcrop.split(","))
    save(fig_stats(log, sprint, avg_v), "sprint_dash_stats")
    save(fig_filmstrip(log, frames, order, t0, t1, args.crop, side, vcrop), "sprint_dash_run")
    import imageio.v2 as imageio
    h = frames[hero]
    imageio.imwrite(OUT / "sprint_dash_frame.png", h[int(h.shape[0] * 0.22):])
    print(f"wrote {OUT / 'sprint_dash_frame.png'}")


if __name__ == "__main__":
    main()
