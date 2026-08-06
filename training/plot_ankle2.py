"""Cross-arm learning curves for the ankle2 screen.

    python training/plot_ankle2.py                        # every live ankle2 arm
    python training/plot_ankle2.py --pattern 'ankle2_m3_*'
    python training/plot_ankle2.py --stale-hours 2        # widen the liveness filter

Four panels, all against training steps: episode length, episode reward, the DUTY-SYMMETRY penalty,
and forward speed.

duty_sym is on here as a first-class panel rather than buried in the reward terms because it is the
tell for the one failure mode that makes a good-looking curve meaningless on this plant: a policy
that survives by going ONE-LEGGED. That gait was what sank the slow_gait lineage, it is why
w_duty_sym and workspace_kill exist, and it still happens -- measured 2026-08-06, the `rigid` arm's
duty_sym penalty grows in lockstep with its episode length. So read the two together: ep_len rising
while duty_sym stays flat is progress; ep_len and |duty_sym| rising together is the exploit.

Runs whose progress.csv has not been touched recently are dropped, so a cancelled or crashed arm
cannot silently drag a common-step comparison down (the cancelled 50 Hz arms pulled the common
step count from ~45 M to 10.7 M before this filter existed).
"""
import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("rollout/ep_len_mean", "episode length (steps)", False),
    ("rollout/ep_rew_mean", "episode reward", False),
    ("reward_terms/duty_sym", "duty-symmetry penalty  (<0 = one-legged)", False),
    ("reward_terms/fwd_speed", "forward-speed reward", False),
]


def arm_of(name):
    """ankle2_m3_bar_s0 -> ('ankle2', 'bar', 's0')  — family, arm, seed."""
    base = name.split("_m3_")
    fam = base[0]
    rest = base[-1]
    seed = rest.rsplit("_", 1)[-1] if rest.rsplit("_", 1)[-1].startswith("s") else ""
    arm = rest[: -len(seed) - 1] if seed else rest
    return fam, arm, seed


def load(d, stale_s):
    f = os.path.join(d, "progress.csv")
    if not os.path.exists(f):
        return None
    if stale_s and (time.time() - os.path.getmtime(f)) > stale_s:
        return None
    rows = [r for r in csv.DictReader(open(f)) if r.get("rollout/ep_len_mean")]
    return rows or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(PKG_DIR / "runs"))
    ap.add_argument("--pattern", default="ankle2*_m3_*")
    ap.add_argument("--stale-hours", type=float, default=1.0,
                    help="drop runs whose progress.csv is older than this (0 = keep all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.runs, args.pattern)))
    stale_s = args.stale_hours * 3600.0
    data, dropped = {}, []
    for d in dirs:
        rows = load(d, stale_s)
        (data.__setitem__(os.path.basename(d), rows) if rows
         else dropped.append(os.path.basename(d)))
    if not data:
        raise SystemExit(f"no live runs matched {args.pattern!r} under {args.runs}")

    arms = sorted({arm_of(n)[:2] for n in data})
    cmap = plt.get_cmap("tab10")
    colors = {a: cmap(i % 10) for i, a in enumerate(arms)}
    styles = {"s0": "-", "s1": "--", "s2": ":", "": "-"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (key, label, logy) in zip(axes.ravel(), PANELS):
        for name, rows in sorted(data.items()):
            fam, arm, seed = arm_of(name)
            x = np.array([float(r["time/total_timesteps"]) for r in rows]) / 1e6
            y = np.array([float(r[key]) if r.get(key) else np.nan for r in rows])
            if np.all(np.isnan(y)):
                continue
            lbl = f"{fam.replace('ankle2', '')or'opt'}:{arm}{'/'+seed if seed else ''}"
            ax.plot(x, y, styles.get(seed, "-"), color=colors[(fam, arm)], lw=1.4,
                    label=lbl, alpha=0.9)
        ax.set_xlabel("steps (M)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale("log")
    axes[0, 0].legend(fontsize=7, ncol=2, loc="upper left")
    common = min(float(r[-1]["time/total_timesteps"]) for r in data.values())
    fig.suptitle(f"ankle2 screen — {len(data)} live runs, common step count "
                 f"{common/1e6:.1f} M"
                 + (f"   (dropped {len(dropped)} stale: {', '.join(dropped)})" if dropped else ""),
                 fontsize=11)
    fig.tight_layout()
    out = args.out or os.path.join(args.runs, "ankle2_compare.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    print(f"\n{'run':<28}{'steps':>11}{'ep_len':>9}{'peak':>8}{'duty_sym':>10}{'fwd':>8}")
    for name, rows in sorted(data.items(),
                             key=lambda kv: -max(float(r["rollout/ep_len_mean"]) for r in kv[1])):
        last = rows[-1]
        peak = max(float(r["rollout/ep_len_mean"]) for r in rows)
        g = lambda k: float(last[k]) if last.get(k) else float("nan")
        print(f"{name:<28}{g('time/total_timesteps'):>11,.0f}{g('rollout/ep_len_mean'):>9.0f}"
              f"{peak:>8.0f}{g('reward_terms/duty_sym'):>10.3f}{g('reward_terms/fwd_speed'):>8.2f}")
    if dropped:
        print(f"\ndropped as stale (>{args.stale_hours} h since last write): {', '.join(dropped)}")


if __name__ == "__main__":
    main()
