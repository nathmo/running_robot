"""Figure for gait_diag JSONs: frequency histogram (log Hz) and residual saturation per channel,
one column per run.  python walk_mit/monitor/plot_gait_diag.py walk_mit/monitor/gait_diag_*.json"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CH = ["hrL", "camL", "thL", "hrR", "camR", "thR"]

files = [Path(p) for p in sys.argv[1:]] or sorted(Path(__file__).parent.glob("gait_diag_*.json"))
runs = {}
for f in files:
    runs.update(json.loads(f.read_text()))
n = len(runs)
fig, ax = plt.subplots(2, n, figsize=(4.2 * n, 6.2), squeeze=False)
for i, (name, s) in enumerate(runs.items()):
    fq = s["freq"]
    e = np.array(fq["hist_edges_hz"])
    h = 100 * np.array(fq["hist_frac"])
    a = ax[0, i]
    a.bar(e[:-1], h, width=np.diff(e), align="edge", color="#3b6ea5", edgecolor="white")
    a.set_xscale("log")
    a.set_xlim(e[0], e[-1])
    a.set_ylim(0, 100)
    a.set_title(f"{name}\nlo-rail {100*fq['lo_rail']:.0f}%  hi-rail {100*fq['hi_rail']:.0f}%  "
                f"re-warped {100*fq['rewrite_frac']:.0f}%/step", fontsize=9)
    a.set_xlabel("gait frequency (Hz, log)")
    if i == 0:
        a.set_ylabel("% of control steps")
    if "residual" not in s:
        ax[1, i].axis("off")
        continue
    r, au = s["residual"], s["authority"]
    b = ax[1, i]
    x = np.arange(6)
    b.bar(x - 0.2, 100 * np.array(r["sat_frac"]), 0.4, color="#c0392b", label="saturated (|r|≥0.95)")
    b.bar(x + 0.2, 100 * np.minimum(np.array(au["share"]), 1.5), 0.4, color="#7f8c8d",
          label="residual share of joint motion")
    b.set_xticks(x)
    b.set_xticklabels(CH)
    b.set_ylim(0, 150)
    b.axhline(100, color="k", lw=0.6, ls="--")
    b.set_title(f"residual ±{s['residual_scale']:.2f} rad\nsome channel saturated "
                f"{100*r['sat_frac_any']:.0f}% of steps, residual share {100*au['share_overall']:.0f}%",
                fontsize=9)
    for j in range(6):
        b.text(j - 0.2, 100 * r["sat_frac"][j] + 2, f"{r['sat_run_p90_ms'][j]:.0f}ms",
               ha="center", fontsize=7, rotation=90)
    if i == 0:
        b.set_ylabel("% of steps  /  % of joint-target RMS")
        b.legend(fontsize=7, loc="upper left", title="bar label = p90 same-sign run", title_fontsize=7)
fig.tight_layout()
out = Path(__file__).parent / "gait_diag_baseline.png"
fig.savefig(out, dpi=140)
print("wrote", out)
