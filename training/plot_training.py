"""Plot training curves (score + losses + reward-term breakdown + curricula) from progress.csv.

Stable-Baselines3 writes progress.csv in the run dir (enabled in train.py). Stdlib-only CSV read,
headless Agg backend (works on a cluster over SSH). Called automatically during training
(PlotCallback) and runnable by hand:

    python training/plot_training.py --run training/runs/m2_sprint
"""
import argparse
import csv
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")            # headless: render to file, never open a window
import matplotlib.pyplot as plt


def _read_progress(run_dir: Path):
    """Return {column: [(x_timesteps, value), ...]} merged from ALL progress*.csv segments
    (train.py rotates progress.csv to progress.<steps>.csv on every resume so a requeued
    cluster run keeps its full history), skipping blanks/NaNs."""
    paths = sorted(run_dir.glob("progress*.csv"))
    rows = []
    for csv_path in paths:
        with open(csv_path, newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        return None
    xkey = "time/total_timesteps"
    series = {}
    for row in rows:
        try:
            x = float(row.get(xkey, "") or "nan")
        except ValueError:
            continue
        if math.isnan(x):
            continue
        for k, v in row.items():
            if k == xkey or v is None or v == "":
                continue
            try:
                y = float(v)
            except ValueError:
                continue
            if math.isnan(y):
                continue
            series.setdefault(k, []).append((x, y))
    for k in series:                     # segments may interleave: keep curves monotone in x
        series[k].sort(key=lambda p: p[0])
    return series


def _plot_panel(ax, series, keys, title):
    plotted = False
    for k in keys:
        if k in series and len(series[k]) > 1:
            xs, ys = zip(*series[k])
            ax.plot(xs, ys, label=k.split("/")[-1], linewidth=1.2)
            plotted = True
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=7, loc="best")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)


def plot_run(run_dir) -> Optional[str]:      # NOT `str | None`: PEP 604 in an evaluated annotation
    # is a TypeError on Python 3.9, and the Izar venv is 3.9. Because train.py's PlotCallback wraps
    # plotting in a bare `except Exception` that only reports when verbose, that raised at IMPORT
    # time and silently produced NO training_plots.png for every cluster run ever done.
    """Render <run_dir>/training_plots.png from progress.csv. Returns the path, or None."""
    run_dir = Path(run_dir)
    series = _read_progress(run_dir)
    if not series:
        return None
    reward_terms = sorted(k for k in series if k.startswith("reward_terms/"))
    gait_terms = [k for k in reward_terms if k.split("/")[-1] in
                  ("foot_slip", "air_time", "clearance", "stance_time", "phase_contact")]
    other_terms = [k for k in reward_terms if k not in gait_terms]

    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    _plot_panel(axes[0, 0], series, ["rollout/ep_rew_mean"], "Episode reward (score)")
    _plot_panel(axes[0, 1], series, ["rollout/ep_len_mean"], "Episode length (control steps)")
    _plot_panel(axes[0, 2], series, ["reward_terms/fwd_speed"],
                "Speed income (fwd_speed = w * vx -> /2 for m/s)")
    _plot_panel(axes[1, 0], series, gait_terms, "Gait terms (stepping vs skating)")
    _plot_panel(axes[1, 1], series, other_terms, "Other reward terms")
    _plot_panel(axes[1, 2], series, ["curriculum/sprint_dist_m", "curriculum/stance_ratio",
                                     "curriculum/eff_scale", "curriculum/ent_coef"], "Curricula")
    _plot_panel(axes[2, 0], series, ["train/loss", "train/value_loss",
                                     "train/policy_gradient_loss", "train/entropy_loss"], "Losses")
    _plot_panel(axes[2, 1], series, ["train/approx_kl", "train/clip_fraction"],
                "approx_kl & clip_fraction (want clip < 0.2)")
    _plot_panel(axes[2, 2], series, ["train/std", "train/explained_variance"],
                "policy std & explained_variance")
    for ax in axes[-1]:
        ax.set_xlabel("timesteps")

    fig.tight_layout()
    out = run_dir / "training_plots.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing progress.csv")
    args = ap.parse_args()
    out = plot_run(args.run)
    print(f"wrote {out}" if out else f"no progress.csv data in {args.run}")


if __name__ == "__main__":
    main()
