"""One learning-curve panel per milestone in the walk_fwd lineage.

    python training/plot_milestones.py [--runs a b c] [--out training/milestones/milestones.png]

IMPORTANT — progress.csv only ever holds the CURRENT chain link. Each 4 h job restarts the SB3
logger and rewrites the file, so a 340 M-step run shows only its last ~55 M steps. The panels are
therefore windows, not full histories, and each is labelled with the window it covers. Do not read
a flat panel as "the run was always flat".

ep_len is converted to SECONDS. At 200 Hz the raw step counts are 4x what they look like, and
reading them as seconds is exactly the misreading that made a 2 s policy look like an 8 s one.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DEFAULT = ["teleop_v5_warm", "walk_fwd_easy_s0", "walk_fwd_s0", "walk_fwd2_s0", "walk_fwd2_s1"]
TITLE = {
    "teleop_v5_warm":   "teleop_v5 — k350 spring, legacy drive (the ancestor)",
    "walk_fwd_easy_s0": "walk_fwd_easy — bar_comp ankle, real 0.8 Hz, NO adversity",
    "walk_fwd_s0":      "walk_fwd — DEADLOCKED: dr_scale 0.00, drive stuck at 12 Hz",
    "walk_fwd2_s0":     "walk_fwd2 s0 — real 0.8 Hz + ramped adversity",
    "walk_fwd2_s1":     "walk_fwd2 s1 — real 0.8 Hz + ramped adversity",
}
COLOR = {"teleop_v5_warm": "#888888", "walk_fwd_easy_s0": "#1b9e77",
         "walk_fwd_s0": "#d95f02", "walk_fwd2_s0": "#7570b3", "walk_fwd2_s1": "#7570b3"}
HZ = 200.0


def load(run_dir):
    p = run_dir / "progress.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))

    def col(k):
        out = []
        for r in rows:
            try:
                out.append(float(r.get(k, "")))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.array(out)
    d = {"steps": col("time/total_timesteps"), "ep_len": col("rollout/ep_len_mean"),
         "rew": col("rollout/ep_rew_mean"), "dr": col("curriculum/dr_scale"),
         "bw": col("curriculum/drive_bw_log10")}
    ok = np.isfinite(d["steps"]) & np.isfinite(d["ep_len"])
    return {k: v[ok] for k, v in d.items()} if ok.any() else None


def drive_hz(d, run_dir):
    """The drive the run was ACTUALLY on: the curriculum value if it had a ramp, else the
    resolved config's fixed value (walk_fwd2 starts at 0.8 Hz, so it logs no curriculum key)."""
    if np.isfinite(d["bw"]).any():
        return 10 ** d["bw"][-1], True
    cfg = json.loads((run_dir / "resolved_config.json").read_text())
    cfg = cfg.get("config", cfg)
    return float(cfg.get("drive_bandwidth_hz", 0.0)), False


def smooth(y, w=25):
    return y if len(y) < w or w < 2 else np.convolve(y, np.ones(w) / w, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUNS_DEFAULT)
    ap.add_argument("--root", default="training/runs")
    ap.add_argument("--out", default="training/milestones/milestones.png")
    args = ap.parse_args()
    root = Path(args.root)

    data = {}
    for r in args.runs:
        d = load(root / r)
        if d is None:
            print(f"  [skip] {r}: no progress.csv")
            continue
        data[r] = d

    n = len(data)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), squeeze=False)
    fig.suptitle("walk_fwd lineage — each panel is the run's LAST chain link, not its full history",
                 fontsize=13)

    for ax, (r, d) in zip(axes[:, 0], data.items()):
        c = COLOR.get(r)
        x = d["steps"] / 1e6
        hz, ramped = drive_hz(d, root / r)
        ax.plot(x, smooth(d["ep_len"] / HZ), color=c, lw=1.5, label="episode length (s)")
        ax.axhline(60, color="k", ls=":", lw=0.8)
        ax.set_ylabel("survived (s)")
        ax.set_ylim(0, max(5.0, float(np.nanmax(d["ep_len"]) / HZ) * 1.25))
        ax2 = ax.twinx()
        ax2.plot(x, smooth(d["rew"]), color="k", lw=0.9, alpha=0.45, label="reward")
        ax2.set_ylabel("reward", color="k", alpha=0.6)
        dr = d["dr"][-1] if np.isfinite(d["dr"]).any() else float("nan")
        ax.set_title(f"{TITLE.get(r, r)}\n"
                     f"window {x[0]:.0f}–{x[-1]:.0f} M steps   |   drive {hz:.2f} Hz"
                     f"{' (ramped)' if ramped else ' (fixed)'}   |   dr_scale {dr:.3f}   |   "
                     f"best {np.nanmax(d['ep_len']) / HZ:.1f} s", fontsize=9, loc="left")
        ax.grid(alpha=0.25)
        ax.set_xlabel("M steps")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    print(f"\n{'run':<20} {'steps':>13} {'window':>16} {'ep_len':>8} {'best':>8} "
          f"{'reward':>9} {'drive':>9} {'dr':>7}")
    for r, d in data.items():
        hz, _ = drive_hz(d, root / r)
        dr = d["dr"][-1] if np.isfinite(d["dr"]).any() else float("nan")
        print(f"{r:<20} {int(d['steps'][-1]):>13,} "
              f"{d['steps'][0] / 1e6:>6.0f}-{d['steps'][-1] / 1e6:<9.0f} "
              f"{d['ep_len'][-1] / HZ:>7.1f}s {d['ep_len'].max() / HZ:>7.1f}s "
              f"{d['rew'][-1]:>9.1f} {hz:>8.2f}Hz {dr:>7.3f}")


if __name__ == "__main__":
    main()
