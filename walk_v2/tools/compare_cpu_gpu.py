"""Compare learning curves of the CPU arm (walk_mit, SB3 progress.csv) and the GPU arm (walk_v2
progress.csv) at matched env steps: episode length, return, finishes, and wall-clock throughput.

    python walk_v2/tools/compare_cpu_gpu.py --cpu jed/v2_s1_s0.csv jed/v2_s1_s1.csv \
        --gpu walk_v2/runs/v2_s1_planar_s0/progress.csv --png walk_v2/results/cpu_vs_gpu_s1.png
"""
from __future__ import annotations

import argparse
import csv
import math

KEYS = {  # name -> (cpu column, gpu column)
    "steps": ("time/total_timesteps", "time/env_steps"),
    "ep_len": ("rollout/ep_len_mean", "rollout/ep_len_mean"),
    "ep_ret": ("rollout/ep_rew_mean", "rollout/ep_ret_mean"),
    "fps": ("time/fps", "time/sps"),
    "elapsed": ("time/time_elapsed", None),
}


def load(path, side):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out = {}
            for k, (c, g) in KEYS.items():
                col = c if side == "cpu" else g
                v = r.get(col) if col else None
                try:
                    out[k] = float(v) if v not in (None, "") else math.nan
                except ValueError:
                    out[k] = math.nan
            if not math.isnan(out["steps"]):
                rows.append(out)
    if side == "gpu":       # walk_v2 logs iter_s, not elapsed
        t = 0.0
        with open(path) as f:
            for r, o in zip(csv.DictReader(f), rows):
                t += float(r.get("time/iter_s") or 0.0)
                o["elapsed"] = t
    return rows


def at(rows, target, k, window=5):
    i = min(range(len(rows)), key=lambda j: abs(rows[j]["steps"] - target))
    lo, hi = max(0, i - window), min(len(rows), i + window + 1)
    vals = [r[k] for r in rows[lo:hi] if not math.isnan(r[k])]
    return sum(vals) / len(vals) if vals else math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", nargs="*", default=[])
    ap.add_argument("--gpu", nargs="*", default=[])
    ap.add_argument("--at", type=float, nargs="*", default=[5e6, 10e6, 20e6, 50e6, 100e6, 200e6, 300e6])
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    runs = [(p, "cpu", load(p, "cpu")) for p in args.cpu] + [(p, "gpu", load(p, "gpu")) for p in args.gpu]
    for p, side, rows in runs:
        last = rows[-1]
        print(f"[{side}] {p}: {len(rows)} rows, {last['steps']/1e6:.1f} M steps, {last['elapsed']/3600:.2f} h, "
              f"mean {last['steps']/max(last['elapsed'],1e-9):,.0f} steps/s")
    print(f"{'steps':>8s} | " + " | ".join(f"{side}:{p.split('/')[-2] if '/' in p else p}"[:22].rjust(22) for p, side, _ in runs))
    for target in args.at:
        cells = []
        for p, side, rows in runs:
            if rows[-1]["steps"] < target * 0.9:
                cells.append(" " * 22)
            else:
                cells.append(f"len {at(rows, target, 'ep_len'):5.0f} ret {at(rows, target, 'ep_ret'):7.1f}".rjust(22))
        print(f"{target/1e6:6.0f} M | " + " | ".join(cells))
    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for p, side, rows in runs:
            x = [r["steps"] / 1e6 for r in rows]
            lab = f"{side} {p.split('/')[-2] if '/' in p else p}"
            ls = "-" if side == "gpu" else "--"
            axs[0].plot(x, [r["ep_len"] for r in rows], ls, label=lab, lw=1)
            axs[1].plot(x, [r["ep_ret"] for r in rows], ls, label=lab, lw=1)
            axs[2].plot(x, [r["fps"] for r in rows], ls, label=lab, lw=1)
        for ax, t in zip(axs, ("episode length", "episode return", "env steps / s")):
            ax.set_xlabel("env steps (M)"); ax.set_title(t); ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(args.png, dpi=130)
        print("wrote", args.png)


if __name__ == "__main__":
    main()
