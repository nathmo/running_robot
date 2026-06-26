"""Plot training curves (score + losses + reward-term breakdown) from a run's progress.csv.

Stable-Baselines3 writes a CSV logger file `progress.csv` in the run dir (enabled in train.py).
This reads it with the stdlib only (no pandas) and renders one PNG with all the key curves, using
the headless Agg backend so it works on a server over SSH. Called automatically during training
(PlotCallback) and runnable by hand:

    .venv/Scripts/python.exe -m rl.plot_training --run rl/runs/m2_walk_v3
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: render to file, never open a window
import matplotlib.pyplot as plt


def _read_progress(run_dir: Path):
    """Return {column: [(x_timesteps, value), ...]} from progress.csv (skipping blanks/NaNs)."""
    csv_path = run_dir / "progress.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
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


def plot_run(run_dir) -> str | None:
    """Render <run_dir>/training_plots.png from progress.csv. Returns the path, or None if no data."""
    run_dir = Path(run_dir)
    series = _read_progress(run_dir)
    if not series:
        return None
    reward_terms = sorted(k for k in series if k.startswith("reward_terms/"))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    _plot_panel(axes[0, 0], series, ["rollout/ep_rew_mean"], "Episode reward (score)")
    _plot_panel(axes[0, 1], series, ["rollout/ep_len_mean"], "Episode length (max 1000)")
    _plot_panel(axes[0, 2], series, ["train/loss", "train/value_loss",
                                     "train/policy_gradient_loss", "train/entropy_loss"], "Losses")
    _plot_panel(axes[1, 0], series, ["train/approx_kl", "train/clip_fraction"],
                "approx_kl & clip_fraction (want clip < 0.2)")
    _plot_panel(axes[1, 1], series, ["train/std", "train/explained_variance"],
                "policy std & explained_variance")
    _plot_panel(axes[1, 2], series, reward_terms, "Reward-term breakdown")
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
