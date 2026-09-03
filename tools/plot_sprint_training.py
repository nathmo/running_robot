#!/usr/bin/env python3
"""Thesis figure: training curves of the sprint runner (sprint_m3_mit_s0).

Reads the run's progress*.csv (SB3 logger output; rotated segments are merged) and renders a
two-panel summary — episode return with the sprint-line curriculum overlaid, and the mean forward
speed of training episodes — instead of the 9-panel diagnostic sheet train.py maintains.

NOTE the speed panel shows the TRAINING-time stochastic-policy average (reward income / weight),
which sits below the 3.07 m/s greedy dash average: training episodes include the standing start,
exploration noise, and every fall.

  python tools/plot_sprint_training.py
  python tools/plot_sprint_training.py --run walk_mit/runs/sprint_m6_mit_s0 --out sprint6_training
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

C_MAIN, C_SPEED = "#2a78d6", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

FWD_SPEED_W = 2.0    # reward_terms/fwd_speed = w * vx (see plot_training.py's panel title)


def read_series(run_dir, keys):
    """{key: (steps, values)} merged from all progress*.csv segments, sorted by timesteps."""
    rows = []
    for p in sorted(Path(run_dir).glob("progress*.csv")):
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no progress*.csv in {run_dir}")
    out = {}
    for k in keys:
        pts = []
        for r in rows:
            try:
                x, y = float(r["time/total_timesteps"]), float(r.get(k, "") or "nan")
            except (ValueError, KeyError):
                continue
            if not (math.isnan(x) or math.isnan(y)):
                pts.append((x, y))
        pts.sort(key=lambda p: p[0])
        out[k] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return out


def smooth(y, n=151):
    if len(y) < 3 * n:
        return y
    pad = np.r_[np.full(n // 2, y[0]), y, np.full(n // 2, y[-1])]
    return np.convolve(pad, np.ones(n) / n, mode="valid")[:len(y)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "walk_mit/runs/sprint_m3_mit_s0"))
    ap.add_argument("--out", default="sprint_training")
    args = ap.parse_args()

    s = read_series(args.run, ["rollout/ep_rew_mean", "reward_terms/fwd_speed",
                               "curriculum/sprint_dist_m"])

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8.5, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.linewidth": 0.8, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, (ax_r, ax_v) = plt.subplots(
        1, 2, figsize=(6.3, 2.35), dpi=300, gridspec_kw={"wspace": 0.36})
    for ax in (ax_r, ax_v):
        ax.grid(True, color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3)
        ax.set_xlabel("environment steps ($\\times 10^8$)")

    # episode return + the sprint-line ramp that defines the task while it grows
    x, y = s["rollout/ep_rew_mean"]
    ax_r.plot(x / 1e8, y, lw=0.4, color=C_MAIN, alpha=0.25)
    l_ret, = ax_r.plot(x / 1e8, smooth(y), lw=1.2, color=C_MAIN, label="episode return")
    ax_r.set_ylabel("episode return")
    ax_d = ax_r.twinx()
    xd, yd = s["curriculum/sprint_dist_m"]
    l_cur, = ax_d.plot(xd / 1e8, yd, lw=0.9, color=INK2, ls=(0, (4, 3)),
                       label="sprint line (m, right)")
    ax_d.tick_params(axis="y", colors=INK2, length=3, labelcolor=INK2)
    ax_d.spines[["top", "left"]].set_visible(False)
    ax_d.spines["right"].set_color(INK2)
    ax_d.set_ylim(0, 110)
    ax_r.legend(handles=[l_ret, l_cur], loc="lower right", fontsize=7)

    # mean forward speed of training episodes (stochastic policy, includes starts and falls)
    xv, yv = s["reward_terms/fwd_speed"]
    v = yv / FWD_SPEED_W
    ax_v.plot(xv / 1e8, v, lw=0.4, color=C_SPEED, alpha=0.25)
    ax_v.plot(xv / 1e8, smooth(v), lw=1.2, color=C_SPEED)
    ax_v.set_ylabel("mean forward speed (m/s)")
    ax_v.set_ylim(bottom=min(0.0, float(np.percentile(v, 0.5))))

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{args.out}.{ext}", bbox_inches="tight")
    print(f"wrote {OUT / args.out}.png/.pdf   "
          f"(final smoothed: return {smooth(y)[-1]:.0f}, speed {smooth(v)[-1]:.2f} m/s)")


if __name__ == "__main__":
    main()
