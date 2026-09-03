#!/usr/bin/env python3
"""Thesis figure: sensitivity of the speed-ceiling estimates to +50% design changes.

One bar group per what-if (leg mass, rotor inertia, motor peak torque, motor no-load speed,
each x1.5 with everything else nominal), two bars per group -- the two actuated ceiling
estimates, in absolute m/s:

  grounded + 7 ms  the rail-sweep tracked bound at MIT-mode latency, re-run through
                   rail_sensitivity.best_v (full pass-2 search, fine k grid, 2-gain bracket)
                   per scenario. NOTE its self-consistent baseline is 6.26 m/s -- that
                   protocol -- not the 6.55 headline of rail_bound.py's coarser full search.
  flight closure   flight_bound.compute_v_star re-run per scenario: peak/no-load scaling enters
                   the force-at-speed map + V_TOE, the rig scaling enters the measured cycling
                   limit T_min, and the leg-mass case also raises the supported mass M by the
                   extra mass of BOTH legs (the leg2d rig is one leg; M is the whole robot).

(The pass-1 kinematic bound is deliberately NOT plotted: it assumes unlimited torque and a
massless leg, so it is exactly linear in no-load speed and blind to the other three knobs --
a 14.5 m/s bar that never moves adds confusion, not information.)

Run:  .venv/Scripts/python.exe leg2d/bound_sensitivity_bars.py [--replot]
Outputs: results/bound_sensitivity.json + results/bound_sensitivity_bars.{png,pdf} (repo root)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import rail_bound as rb            # noqa: E402
import flight_bound as fb          # noqa: E402

ROOT = PKG.parent
OUT_JSON = ROOT / "results" / "bound_sensitivity.json"
OUT_STEM = ROOT / "results" / "bound_sensitivity_bars"

S = 1.5
SCENARIOS = [                       # scenario -> Rig what-if kwargs
    ("leg mass +50%",      {"leg_mass_scale": S}),
    ("rotor inertia +50%", {"armature_scale": S}),
    ("peak torque +50%",   {"peak_scale": S}),
    ("no-load speed +50%", {"w0_scale": S}),
]

C_RAIL, C_FLY = "#eb6834", "#1baf7a"     # same entity colors as everywhere else in the thesis
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

# WIDER gain bracket than rail_sensitivity.GAINS: at x1.5 scalings the optimum drifts out of the
# 2-gain bracket (measured: peak torque x1.5 read 6.47 under {1000,4000}x{12,35} while x1.3 read
# 7.24 -- a monotone-resource decrease, i.e. a search failure, the rail bound's documented trap
# #5). The high-DAMPING corner is what recovers it: more torque authority deepens the 7 ms delay
# limit-cycle, and kp=4000 needs kv=60 (probed: 7.11 m/s) where kv=35 chatters into violations
# (4.52). The BASELINE is re-run under the same bracket so impacts stay protocol-consistent.
GAINS_WIDE = [(700.0, 8.0), (1000.0, 12.0), (2000.0, 20.0), (3000.0, 45.0), (4000.0, 60.0)]


def best_v_wide(p1, base_h, rig_kwargs):
    best = None
    for kp, kv in GAINS_WIDE:
        r = rb.pass2_tracked(p1, base_h, delay_ms=7, kp=kp, kv=kv, k_max=20.0,
                             rig_kwargs=rig_kwargs, n_k=48, refines=(0.97, 0.94))
        if r and (best is None or r["v_mean"] > best["v_mean"]):
            best = r
    return best


def leg_subtree_mass():
    """Mass of the rig's leg subtree (same walk as Rig's leg_mass_scale: everything under the
    hip-roll joint's body). The rig is ONE leg; the robot carries two."""
    rig = rb.Rig()
    m = rig.model
    b0 = int(m.jnt_bodyid[int(m.actuator_trnid[rig.a_hr, 0])])
    total = 0.0
    for b in range(m.nbody):
        bb = b
        while bb not in (0, b0):
            bb = int(m.body_parentid[bb])
        if bb == b0:
            total += float(m.body_mass[b])
    return total


def compute():
    rec, arc_p, p1, base_h = rb.prepare()
    J = fb.local_jacobians(p1)
    m_leg = leg_subtree_mass()
    print(f"[bars] leg subtree mass {m_leg:.2f} kg -> leg-mass case supports "
          f"M = {fb.M:.2f} + {2 * (S - 1.0) * m_leg:.2f} kg")

    print("[bars] grounded+7ms, baseline (wide gain bracket)")
    v_rail0 = best_v_wide(p1, base_h, None)["v_mean"]
    print("[bars] flight closure, baseline")
    v_fly0 = fb.compute_v_star(p1, J, base_h, v_max=16.0)["v_star"]
    print(f"[bars] baselines: grounded+7ms {v_rail0:.2f} (headline full-search 6.55), "
          f"flight {v_fly0:.2f} m/s")

    out = {"scale": S, "m_leg_subtree": m_leg,
           "base": {"grounded_7ms": v_rail0, "flight": v_fly0},
           "note_grounded": "grounded values = wide 4-gain bracket + fine k grid, baseline "
                            "recomputed under the same protocol (not rail_bound.py's 6.55 "
                            "full-search headline, not rail_sensitivity.json's 2-gain 6.26)",
           "scenarios": {}}
    for label, kw in SCENARIOS:
        print(f"[bars] grounded+7ms, {label}")
        r = best_v_wide(p1, base_h, kw)
        v_rail = r["v_mean"] if r else 0.0
        mass = fb.M + 2 * (S - 1.0) * m_leg if "leg_mass_scale" in kw else fb.M
        print(f"[bars] flight closure, {label}")
        v_fly = fb.compute_v_star(
            p1, J, base_h, mass=mass,
            peak=rb.PEAK * S if "peak_scale" in kw else None,
            no_load=rb.NO_LOAD * S if "w0_scale" in kw else None,
            rig_kwargs=kw, v_max=16.0)["v_star"]
        out["scenarios"][label] = {"grounded_7ms": v_rail, "flight": v_fly}
        print(f"[bars] {label}: rail {v_rail:.2f} ({v_rail / v_rail0 - 1:+.1%}), "
              f"flight {v_fly:.2f} ({v_fly / v_fly0 - 1:+.1%})")
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON}")
    return out


def plot(out):
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8.5, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.linewidth": 0.8, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    base = out["base"]
    series = [("grounded_7ms", C_RAIL, "grounded bound (torque envelope + 7 ms MIT latency)"),
              ("flight", C_FLY, "flight-phase closure")]
    groups = ["base\nreference"] + [g.replace(" +50%", "\n+50%") for g in out["scenarios"]]
    keys = list(out["scenarios"])
    fig, ax = plt.subplots(figsize=(6.3, 2.9), dpi=300)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)

    w = 0.34
    vmax = 0.0
    for si, (key, col, lab) in enumerate(series):
        vals = [base[key]] + [out["scenarios"][g][key] for g in keys]
        xpos = np.arange(len(groups)) + (si - 0.5) * w
        ax.bar(xpos, vals, width=w - 0.03, color=col, label=lab, zorder=3)
        vmax = max(vmax, max(vals))
        for gi, (x, v) in enumerate(zip(xpos, vals)):
            ax.text(x, v + 0.10, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color=INK)
            if gi > 0 and base[key] > 0:                 # impact vs base, under the value
                d = v / base[key] - 1.0
                ax.text(x, v + 0.62, f"{d:+.0%}", ha="center", va="bottom",
                        fontsize=5.8, color=INK2)
    ax.axhline(base["flight"], color=INK2, lw=0.7, ls=(0, (4, 3)), zorder=2)
    ax.text(len(groups) - 0.55, base["flight"] - 0.55, "base ceiling", fontsize=6,
            color=INK2, ha="right")
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, fontsize=7.5)
    ax.set_ylabel("speed ceiling (m/s)")
    ax.set_ylim(0, vmax * 1.22)
    ax.legend(fontsize=6.8, loc="upper left", borderaxespad=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_STEM}.{ext}", bbox_inches="tight")
    print(f"wrote {OUT_STEM}.png/.pdf")


if __name__ == "__main__":
    if "--replot" in sys.argv and OUT_JSON.exists():
        plot(json.loads(OUT_JSON.read_text()))
    else:
        plot(compute())
