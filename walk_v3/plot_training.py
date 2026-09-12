"""training_plots.png from a v2 run's progress.csv (+ eval.csv): the same reading the walk_mit
plots gave, with the v2 diagnostics (commit fraction, resync, frequency rails, residual
saturation, thermal, symmetry loss) alongside.

    python walk_v3/plot_training.py walk_v3/runs/<name>
"""
import sys
from pathlib import Path

import numpy as np


def _read(p):
    import csv
    rows = []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", None) else np.nan) for k, v in r.items()})
    return rows


def plot_run(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run)
    rows = []
    for p in sorted(run.glob("progress*.csv")):
        rows += _read(p)
    if not rows:
        raise FileNotFoundError(f"no progress.csv in {run}")
    keys = rows[0].keys()
    col = lambda k: np.array([r.get(k, np.nan) for r in rows])
    x = col("time/env_steps") / 1e6
    panels = [
        ("rollout/ep_len_mean", "episode length (ticks)"),
        ("rollout/ep_ret_mean", "episode return"),
        ("rollout/reward_mean", "reward / step"),
        ("rollout/finishes", "finishes / rollout"),
        ("rollout/t_line_mean", "dash time at the line (s)"),
        ("rollout/sprint_d_mean", "mean sprint distance (m)"),
        ("rollout/swing_frac_min", "worse-foot airborne fraction"),
        ("train/std_mean", "policy std"),
        ("train/approx_kl", "approx KL"),
        ("train/loss_sym", "symmetry loss"),
        ("est/vel_rmse", "estimator RMSE (norm units)"),
        ("diag/freq_hz_median", "latched f median (Hz)"),
        ("diag/res_sat_sampled", "residual saturation (sampled)"),
        ("rollout/commit_frac", "commit fraction"),
        ("rollout/resync_frac", "resync fraction"),
        ("rollout/thermal_max", "max dT/dT_max"),
        ("curriculum/dr_scale", "DR scale"),
        ("curriculum/sprint_dist_m", "sprint line (m)"),
        ("curriculum/stance_ratio", "stance ratio"),
        ("time/sps", "env steps / s"),
    ]
    terms = sorted(k for k in keys if k.startswith("reward_terms/"))
    n = len(panels) + 1
    cols = 4
    rws = (n + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(4.6 * cols, 3.0 * rws))
    axes = axes.ravel()
    for ax, (k, title) in zip(axes, panels):
        if k in keys:
            ax.plot(x, col(k), lw=1.2)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("M steps", fontsize=8)
    ax = axes[len(panels)]
    for k in terms:
        y = col(k)
        if np.nanmax(np.abs(y)) > 1e-4:
            ax.plot(x, y, lw=0.9, label=k.split("/")[1])
    ax.set_title("reward terms (per step)", fontsize=9)
    ax.legend(fontsize=5, ncol=2)
    ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    ev = run / "eval.csv"
    if ev.exists():
        er = _read(ev)
        if er:
            ax = axes[3]
            ax.plot(np.array([r["step"] for r in er]) / 1e6, [r["finishes"] / max(r["n"], 1) for r in er],
                    "o-", ms=3, lw=1, color="C3", label="greedy finish frac")
            ax.legend(fontsize=6)
    fig.suptitle(run.name, fontsize=11)
    fig.tight_layout()
    out = run / "training_plots.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot_run(sys.argv[1]))
