"""Compare the arms of the CPG-vs-Fourier generator A/B at a COMMON step count.

    python training/compare_ab.py --runs training/runs --pattern 'ab_*'
    python training/compare_ab.py --lineage cold          # only the cold-m3 jobs
    python training/compare_ab.py --at 60000000           # force the comparison step

Why a common step count and not "each run's final row": the arms are wall-clock bounded, so they
stop at different step counts, and a learning curve read at different x is not a comparison. By
default every arm is read at the largest step count ALL of them reached, which is the honest
apples-to-apples point. Runs are grouped by arm (fourier / cpg / cpg_nr) and lineage
(cold / m2 / m3warm) from the run-directory name, and seeds within an arm are shown as mean +- range
so a single lucky seed cannot carry a verdict.

Writes a table to stdout and a learning-curve PNG (ep_len_mean and ep_rew_mean vs steps, one colour
per arm, one line style per seed).
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = "time/total_timesteps"
ARM_COLORS = {"fourier": "#1f77b4", "cpg": "#d62728", "cpg_nr": "#2ca02c",
              "cpg_wide": "#ff7f0e"}


def read_progress(run_dir):
    """Load and concatenate progress*.csv (train.py rotates them on resume), sorted by step."""
    import csv
    rows = []
    for p in sorted(glob.glob(os.path.join(run_dir, "progress*.csv"))):
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    rows.append({k: (float(v) if v not in ("", None) else np.nan)
                                 for k, v in r.items()})
                except (TypeError, ValueError):
                    continue
    if not rows:
        return None
    rows.sort(key=lambda r: r.get(STEPS, 0.0))
    return {k: np.array([r.get(k, np.nan) for r in rows]) for k in rows[0]}


def parse_name(name):
    """ab_<arm>_<lineage>_s<seed>  ->  (arm, lineage, seed). Arm names contain underscores
    (cpg_nr), so match the known arms explicitly rather than splitting on '_'."""
    m = re.match(r"^ab_(cpg_nr|cpg_wide|cpg|f)_(cold|m2|m3warm)_s(\d+)$", name)
    if not m:
        return None
    arm = {"f": "fourier", "cpg": "cpg", "cpg_nr": "cpg_nr",
           "cpg_wide": "cpg_wide"}[m.group(1)]
    return arm, m.group(2), int(m.group(3))


def at_step(d, col, step):
    """Value of `col` at the last row with total_timesteps <= step (no interpolation across a
    curriculum change; the last logged row is what the run actually was at that point)."""
    if col not in d:
        return np.nan
    ok = d[STEPS] <= step
    if not ok.any():
        return np.nan
    return float(d[col][ok][-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(PKG_DIR / "runs"))
    ap.add_argument("--pattern", default="ab_*")
    ap.add_argument("--lineage", default=None, choices=["cold", "m2", "m3warm"])
    ap.add_argument("--at", type=float, default=None, help="comparison step count (default: the "
                                                           "largest all selected runs reached)")
    ap.add_argument("--out", default=None, help="PNG path (default: <runs>/ab_compare.png)")
    args = ap.parse_args()

    runs = {}
    for p in sorted(glob.glob(os.path.join(args.runs, args.pattern))):
        if not os.path.isdir(p):
            continue
        meta = parse_name(os.path.basename(p))
        if meta is None or (args.lineage and meta[1] != args.lineage):
            continue
        d = read_progress(p)
        if d is None or STEPS not in d:
            print(f"[compare] {os.path.basename(p)}: no progress rows yet, skipped")
            continue
        runs[os.path.basename(p)] = (meta, d)
    if not runs:
        raise SystemExit("[compare] no runs matched")

    reached = {n: float(d[STEPS].max()) for n, (_, d) in runs.items()}

    by_lineage = {}
    for name, (meta, d) in runs.items():
        by_lineage.setdefault(meta[1], []).append((meta[0], meta[2], name, d))

    # The comparison step is PER LINEAGE, not global. Each lineage is its own experiment, and they
    # do not start together (the m3warm stage only begins when its m2 stage ends), so a global
    # min() over every run collapses the whole table to the youngest run's step count and reads the
    # mature arms at ~0 progress. --at still forces one step for every lineage when that is wanted.
    lineage_step = {lin: (args.at if args.at is not None
                          else min(reached[n] for _, _, n, _ in rows))
                    for lin, rows in by_lineage.items()}
    print()
    for lin in sorted(by_lineage):
        # built with an explicit loop, not a nested f-string: the cluster venv is Python 3.9 and
        # reusing the same quote character inside an f-string expression is 3.12+ syntax
        parts = []
        for _, _, n, _ in sorted(by_lineage[lin]):
            short = n.replace("ab_", "")
            parts.append("{}={:.0f}M".format(short, reached[n] / 1e6))
        print("{:8s} comparison step = {:6.1f} M   (reached: {})".format(
            lin, lineage_step[lin] / 1e6, ", ".join(parts)))
    print()

    hdr = f"{'lineage':8s} {'arm':8s} {'seed':>4s} {'ep_len':>9s} {'ep_rew':>10s} " \
          f"{'fwd_speed':>10s} {'cadence':>8s} {'reached':>9s}"
    for lin in sorted(by_lineage):
        step = lineage_step[lin]
        print(f"=== {lin}  @ {step/1e6:.1f} M steps ===")
        print(hdr)
        print("-" * len(hdr))
        agg = {}
        for arm, seed, name, d in sorted(by_lineage[lin]):
            el = at_step(d, "rollout/ep_len_mean", step)
            er = at_step(d, "rollout/ep_rew_mean", step)
            fs = at_step(d, "reward_terms/fwd_speed", step)
            cs = at_step(d, "reward_terms/step_rate", step)
            print(f"{lin:8s} {arm:8s} {seed:4d} {el:9.1f} {er:10.1f} {fs:10.3f} {cs:8.3f} "
                  f"{reached[name]/1e6:8.0f}M")
            agg.setdefault(arm, []).append((el, er))
        print()
        for arm in sorted(agg):
            els = [a for a, _ in agg[arm]]
            ers = [b for _, b in agg[arm]]
            print(f"  {arm:8s} n={len(els)}  ep_len {np.mean(els):8.1f}"
                  f"{f' (range {min(els):.0f}..{max(els):.0f})' if len(els) > 1 else ''}"
                  f"   ep_rew {np.mean(ers):9.1f}"
                  f"{f' (range {min(ers):.0f}..{max(ers):.0f})' if len(ers) > 1 else ''}")
        print()

    # ----- learning curves ------------------------------------------------------------------
    lins = sorted(by_lineage)
    fig, axes = plt.subplots(2, len(lins), figsize=(6.5 * len(lins), 8), squeeze=False)
    for j, lin in enumerate(lins):
        for arm, seed, name, d in sorted(by_lineage[lin]):
            c = ARM_COLORS.get(arm, "gray")
            ls = ["-", "--", ":"][seed % 3]
            for i, col in enumerate(("rollout/ep_len_mean", "rollout/ep_rew_mean")):
                if col in d:
                    axes[i][j].plot(d[STEPS] / 1e6, d[col], color=c, ls=ls, lw=1.4,
                                    label=f"{arm} s{seed}")
        for i, lab in enumerate(("ep_len_mean (steps survived)", "ep_rew_mean")):
            ax = axes[i][j]
            ax.axvline(lineage_step[lin] / 1e6, color="k", lw=0.8, alpha=0.4)
            ax.set_xlabel("env steps (M)")
            ax.set_ylabel(lab)
            ax.set_title(f"{lin} — {lab.split(' ')[0]}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
    fig.suptitle("CPG (Ijspeert oscillator) vs Fourier gait generator — same plant, reward, schedule")
    fig.tight_layout()
    out = args.out or os.path.join(args.runs, "ab_compare.png")
    fig.savefig(out, dpi=130)
    print(f"[compare] wrote {out}")


if __name__ == "__main__":
    main()
