"""Full learning curves from tfevents, not progress.csv.

    python training/plot_tb.py --runs ladder_m2_s0 ladder_m3_s0 --out training/milestones/ladder_tb.png

progress.csv only holds the CURRENT chain link -- each 4 h job restarts the SB3 logger and rewrites
it, so a 100 M-step run can show a single point. The tfevents files are written one per link and
never overwritten, so concatenating them by wall-clock recovers the whole run. That is the only way
to see a chained run's actual learning curve.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HZ = 200.0
TAGS = {
    "rollout/ep_len_mean": ("survived (s)", 1.0 / HZ),
    "rollout/ep_rew_mean": ("reward", 1.0),
    "curriculum/dr_scale": ("dr_scale", 1.0),
    "curriculum/cmd_scale": ("cmd_scale", 1.0),
}


def load(run_dir):
    """Concatenate every tfevents in the run, ordered by file mtime (= link order)."""
    out = {t: {} for t in TAGS}
    files = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    for f in files:
        acc = EventAccumulator(str(f), size_guidance={"scalars": 0})
        acc.Reload()
        avail = set(acc.Tags().get("scalars", []))
        for tag in TAGS:
            if tag in avail:
                for e in acc.Scalars(tag):
                    out[tag][e.step] = e.value      # later links win on any overlap
    return {t: (np.array(sorted(d)), np.array([d[k] for k in sorted(d)]))
            for t, d in out.items() if d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--root", default="training/runs")
    ap.add_argument("--out", default="training/milestones/tb.png")
    args = ap.parse_args()
    root = Path(args.root)

    data = {}
    for r in args.runs:
        d = load(root / r)
        if not d:
            print(f"  [skip] {r}: no tfevents")
            continue
        data[r] = d
        n = len(d.get("rollout/ep_len_mean", ([],))[0])
        print(f"  {r}: {n} points, {int(d['rollout/ep_len_mean'][0][-1]):,} steps")

    fig, axes = plt.subplots(len(TAGS), 1, figsize=(12, 3 * len(TAGS)), squeeze=False, sharex=True)
    for ax, (tag, (label, scale)) in zip(axes[:, 0], TAGS.items()):
        for r, d in data.items():
            if tag not in d:
                continue
            x, y = d[tag]
            ax.plot(x / 1e6, y * scale, lw=1.3, label=r)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0][0].set_title("walk_fwd ladder — full curves recovered from tfevents", fontsize=12)
    axes[-1][0].set_xlabel("M steps")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
