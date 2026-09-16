#!/usr/bin/env python3
"""Slide figure: how the actuator bandwidth was measured -- a staged chirp (stepped sine).

The actuator is asked to track a sine of fixed amplitude; every stage lasts the same time
but runs at a higher frequency, so it has to track faster and faster. Frequency steps happen
at upward zero crossings, as in controller/tools/ak_bode_sweep.py, so the request is continuous.
Schematic only (no numbers): the real sweep is 1-30 Hz in 1 Hz steps, >= 5 s each.

Outputs results/staged_chirp_concept.{png,svg}.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "staged_chirp_concept"

FREQS = [1, 2, 3]             # one frequency per stage, held for the whole dwell
DWELL = 3.0                       # same hold time at every frequency (integer cycles each)
AMP = 1.0
BLUE, INK, GUIDE, BAND = "#2a78d6", "#0b0b0b", "#c9c8c1", "#eeede8"


def main():
    t, y = [], []
    for k, f in enumerate(FREQS):
        tt = np.linspace(0.0, DWELL, int(400 * f * DWELL), endpoint=False)
        t.append(k * DWELL + tt)
        y.append(AMP * np.sin(2 * np.pi * f * tt))
    t_end = len(FREQS) * DWELL
    t = np.concatenate(t + [[t_end]])
    y = np.concatenate(y + [[0.0]])

    # vertical layout: angle across, time running up the page
    fig, ax = plt.subplots(figsize=(4.8, 11), dpi=200)
    for k in range(len(FREQS)):                           # one shaded block per held frequency
        if k % 2 == 0:
            ax.axhspan(k * DWELL, (k + 1) * DWELL, color=BAND, lw=0, zorder=0)
    for s in (+1, -1):                                    # the fixed amplitude
        ax.axvline(s * AMP, color=GUIDE, lw=1.5, ls=(0, (5, 4)), zorder=1)
    ax.plot(y, t, color=BLUE, lw=2.6, solid_capstyle="round", zorder=3)

    ax.set_ylim(0, t_end * 1.03)
    ax.set_xlim(-1.35 * AMP, 1.35 * AMP)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(2.5)
        ax.spines[side].set_color(INK)
    # arrowheads on the axes
    ax.plot(1, 0, ">", color=INK, ms=13, transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^", color=INK, ms=13, transform=ax.transAxes, clip_on=False)

    ax.set_ylabel("Time", fontsize=26, fontweight="bold", color=INK, labelpad=10)
    ax.set_xlabel("Requested\nactuator angle", fontsize=26, fontweight="bold", color=INK,
                  labelpad=12, linespacing=1.1)

    OUT.parent.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}.png and .svg")


if __name__ == "__main__":
    main()
