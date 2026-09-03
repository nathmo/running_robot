#!/usr/bin/env python3
"""Thesis figure: robustness margins of the sprint runner (perturbed greedy evaluations).

Reads the envelope JSON written by tools/eval_envelope.py (paired seeds, one parameter moved at a
time, DR/trips off) and renders the three axes the thesis discusses: total mass, actuation delay,
and fore-aft push impulses. Per panel: survival fraction (left axis) and mean forward speed of the
surviving portion (right axis), nominal operating point marked.

Delay is plotted in milliseconds (control steps x 5 ms at 200 Hz). The nominal stored in the JSON
is the env's resolved drive delay (7 ms -> 1 step), not resolved_config's pre-construction value.

  python tools/plot_sprint_margins.py
  python tools/plot_sprint_margins.py --json results/envelope_foo.json --out foo_margins
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

C_SURV, C_SPEED = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

STEP_MS = 5.0    # one 200 Hz control step

PANELS = [   # axis key -> x transform, xlabel
    ("mass",  lambda v: v,             "total mass ($\\times$ nominal)"),
    ("delay", lambda v: v * STEP_MS,   "action delay (ms)"),
    ("push",  lambda v: v,             "push impulse $\\Delta v$ (m/s)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(OUT / "envelope_sprint_m3_mit_s0.json"))
    ap.add_argument("--out", default="sprint_margins")
    args = ap.parse_args()
    j = json.loads(Path(args.json).read_text())

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8.5, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.linewidth": 0.8, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.15), dpi=300,
                             gridspec_kw={"wspace": 0.42})
    for (key, fx, xlabel), ax in zip(PANELS, axes):
        a = j["axes"][key]
        x = np.array([fx(r["value"]) for r in a["rows"]])
        surv = np.array([r["survival"] for r in a["rows"]]) * 100.0
        v = np.array([r["v_mean"] for r in a["rows"]])
        ax.grid(True, color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3)
        ax.axvline(fx(a["nominal"]), color=INK2, lw=0.8, ls=(0, (2, 2)))
        l_s, = ax.plot(x, surv, "-o", ms=3, lw=1.2, color=C_SURV, label="survival")
        ax.set_ylim(-4, 104)
        ax.set_xlabel(xlabel)
        ax2 = ax.twinx()
        l_v, = ax2.plot(x, v, "-s", ms=2.5, lw=1.0, color=C_SPEED, label="mean speed")
        ax2.set_ylim(0, 3.4)
        ax2.spines[["top", "left"]].set_visible(False)
        ax2.spines["right"].set_color(INK2)
        ax2.tick_params(axis="y", colors=INK2, length=3, labelsize=7)
        if ax is axes[0]:
            ax.set_ylabel("survival (%)", color=C_SURV)
            ax.legend(handles=[l_s, l_v], loc="lower left", fontsize=6.5,
                      handlelength=1.6, borderaxespad=0.2)
        if ax is axes[-1]:
            ax2.set_ylabel("mean speed (m/s)", color=C_SPEED, fontsize=8)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{args.out}.{ext}", bbox_inches="tight")
    print(f"wrote {OUT / args.out}.png/.pdf")


if __name__ == "__main__":
    main()
