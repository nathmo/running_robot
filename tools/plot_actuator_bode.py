#!/usr/bin/env python3
"""Thesis figure: measured position-loop frequency response of the four AKE90-8 joints
(cam + thigh, both legs), servo mode vs MIT impedance mode.

Data: robot/tools/ak_bode_sweep.py stepped-sine runs (1-30 Hz, 1 Hz steps, >= 5 s dwell,
legs attached and unloaded, robot homed on its stand, 2026-09-01). Both arms are POSITION
loops at the interface each mode is used through: servo = SET_POS (the drive's internal
loop), MIT = force-control impedance at the deployed gains kp=200, kd=5 -- the
<position kp=200 kv=5> actuator the policy trains against (walk_mit/config.py:497).
This supersedes the 2026-08 figure, whose MIT arm was a *velocity* loop on a free rotor
referenced to its own 1 Hz point because the velocity span was then unidentified.

Per frequency, the response is the least-squares projection of the reported position onto
the command's exact sine/cosine basis (plus mean and linear drift), over the dwell minus
the post-switch settle. Points whose response amplitude sinks below the drive's 0.1 deg
position-feedback LSB are masked rather than plotted as noise -- the high-frequency servo
points die there by design.

Outputs results/actuator_bode.{pdf,png} and prints -3 dB bandwidths + phase-slope delays.
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_GLOB = str(ROOT / "robot/fixed_gait/webui/data/measurements/*bode_*.npz")
OUT = ROOT / "results"

LSB_RAD = math.radians(0.1)          # position feedback resolution
MASK_BELOW = 0.7 * LSB_RAD           # response amplitude floor for an honest point
MASK_QUALITY = 0.45                  # residual/response above this is quantization, not response

SIDE = {"can0": "right", "can1": "left"}
ROLE = {105: "cam", 106: "thigh"}

# ---------------------------------------------------------------- palette (validated, light mode)
# color = joint, linestyle = side: the physics pairs by joint, the sides should overlay.
# The kp=500 stiffness probe (MIT, left thigh) gets its own hue -- it is a different loop,
# not a different joint: the corner is kp/kd, so stiffness moves it linearly.
C_JOINT = {"cam": "#2a78d6", "thigh": "#eb6834"}
C_KP500 = "#2a78d6"
LS_SIDE = {"left": "-", "right": (0, (4, 2))}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def load_run(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    if meta.get("abort"):
        print(f"!! {os.path.basename(path)} ABORTED ({meta['abort']}) -- skipping")
        return None
    t_cmd, phase, f_cmd = z["t_cmd"], z["phase"], z["f_cmd"]
    amp, ramp = z["amp"], z["gain_ramp"]
    t_fb = z["t_fb"] + meta.get("t_off_mono_minus_kernel", 0.0)
    pos = np.radians(z["pos_deg"])
    pos = pos - np.median(pos)
    # per-feedback-sample reference phase and frequency (nearest command tick)
    idx = np.clip(np.searchsorted(t_cmd, t_fb), 1, len(t_cmd) - 1)
    near = idx - (t_fb - t_cmd[idx - 1] < t_cmd[idx] - t_fb)
    ph_fb, f_fb = phase[near], f_cmd[near]

    freqs = np.arange(meta["f0"], meta["f1"] + 1e-9, meta["df"])
    settle = float(meta.get("settle_exclude_s", 1.2))
    rows = []
    for f in freqs:
        on = f_cmd == f
        if not on.any():
            continue
        t_on, t_off_ = t_cmd[on][0], t_cmd[on][-1]
        a_cmd = float(amp[on].max())
        m = (f_fb == f) & (t_fb >= t_on + settle) & (t_fb <= t_off_) & (a_cmd > 0)
        if m.sum() < 40:
            continue
        tt, pp, ph = t_fb[m], pos[m], ph_fb[m]
        basis = np.column_stack([np.sin(ph), np.cos(ph),
                                 np.ones_like(tt), tt - tt[0]])
        coef, *_ = np.linalg.lstsq(basis, pp, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        resp = math.hypot(a, b)
        resid = pp - basis @ coef
        rows.append((f, a_cmd, resp, math.atan2(b, a),
                     float(np.std(resid)) / max(resp, 1e-12)))
    r = np.array(rows)
    if not len(r):
        return None
    gain = r[:, 2] / r[:, 1]
    ph = np.degrees(np.unwrap(r[:, 3]))
    ph -= 360.0 * np.round(ph[0] / 360.0)
    good = (r[:, 2] >= MASK_BELOW) & (r[:, 4] <= MASK_QUALITY)
    return {"meta": meta, "f": r[:, 0], "gain_db": 20 * np.log10(np.maximum(gain, 1e-9)),
            "phase_deg": ph, "quality": r[:, 4], "good": good,
            "name": f"{SIDE.get(meta['channel'], meta['channel'])}."
                    f"{ROLE.get(meta['id'], meta['id'])}"}


def headline(f, mag, ph, good):
    f, mag, ph = f[good], mag[good], ph[good]
    if len(f) < 3:
        return float("nan"), float("nan")
    f3 = float(np.interp(-3.0, mag[::-1], f[::-1])) if mag.min() < -3.0 < mag.max() else \
        float("nan")
    hi = f >= max(4.0, f[0])
    tau = -np.polyfit(f[hi], ph[hi], 1)[0] / 360.0 if hi.sum() > 3 else float("nan")
    return f3, tau


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT / "actuator_bode"))
    args = ap.parse_args()
    files = args.files or sorted(glob.glob(DATA_GLOB))
    runs = [r for r in (load_run(p) for p in files) if r]
    if not runs:
        raise SystemExit(f"no usable runs (looked at {len(files)} files)")

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8.5, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.linewidth": 0.8, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(6.3, 4.9), dpi=300,
                             gridspec_kw={"height_ratios": [1.15, 1.0],
                                          "hspace": 0.10, "wspace": 0.26})
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.grid(True, which="both", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(which="both", length=3)
    (ax_ms, ax_mm), (ax_ps, ax_pm) = axes
    cols = {"servo": (ax_ms, ax_ps), "mit": (ax_mm, ax_pm)}

    print(f"{'run':<16} {'mode':<10} {'-3 dB':>8} {'delay':>8}")
    for r in runs:
        mode = r["meta"]["mode"]
        kp = r["meta"].get("kp")
        ax_m, ax_p = cols[mode]
        side, joint = r["name"].split(".")
        c = C_KP500 if (mode == "mit" and kp not in (None, 200.0)) else C_JOINT[joint]
        ls = LS_SIDE[side]
        g = r["good"]
        ax_m.plot(r["f"][g], r["gain_db"][g], color=c, ls=ls, lw=1.4,
                  marker="o", ms=2.6, markeredgecolor="white", markeredgewidth=0.5)
        ax_p.plot(r["f"][g], r["phase_deg"][g], color=c, ls=ls, lw=1.4,
                  marker="o", ms=2.6, markeredgecolor="white", markeredgewidth=0.5)
        f3, tau = headline(r["f"], r["gain_db"], r["phase_deg"], g)
        tag = mode if kp in (None, 200.0) else f"{mode}:kp{kp:.0f}"
        print(f"{r['name']:<16} {tag:<10} {f3:>6.1f}Hz {tau * 1e3:>6.1f}ms   "
              f"(masked below LSB: {int((~g).sum())} pts)")

    for mode, (ax_m, _ax_p) in cols.items():
        ax_m.axhline(-3.0, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax_m.set_title({"servo": "Servo mode (SET_POS)",
                        "mit": "MIT impedance"}[mode],
                       fontsize=8.5, color=INK)
    ax_mm.annotate("$k_p$=200", xy=(2.1, -6.2), fontsize=7.5,
                   color=C_JOINT["thigh"], ha="left", va="top")
    ax_mm.annotate("$k_p$=500", xy=(11.5, 0.9), fontsize=7.5,
                   color=C_KP500, ha="left", va="bottom")
    ax_ms.text(13.0, -2.6, "−3 dB", fontsize=7, color=INK2, va="bottom")
    ax_ms.set_ylabel("Magnitude (dB)")
    ax_ps.set_ylabel("Phase (deg)")
    lo = min(ax.get_ylim()[0] for ax in (ax_ms, ax_mm))
    for ax in (ax_ms, ax_mm):
        ax.set_ylim(max(lo, -42), 4)
    plo = min(ax.get_ylim()[0] for ax in (ax_ps, ax_pm))
    for ax in (ax_ps, ax_pm):
        ax.set_ylim(plo, 5)
    for ax in (ax_ps, ax_pm):
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0.9, 33)
        ax.set_xticks([1, 2, 5, 10, 20, 30])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "30"])
        ax.tick_params(axis="x", which="minor", labelbottom=False)

    present = sorted({tuple(r["name"].split(".")) for r in runs})
    handles = [plt.Line2D([], [], color=C_JOINT[j], ls=LS_SIDE[s], lw=1.4,
                          label=f"{s} {j}")
               for j in ("cam", "thigh") for s in ("left", "right") if (s, j) in present]
    if any(r["meta"].get("kp") not in (None, 200.0) for r in runs):
        handles.append(plt.Line2D([], [], color=C_KP500, lw=1.4,
                                  label="left thigh, $k_p$=500"))
    ax_ms.legend(handles=handles, loc="lower left", fontsize=7.5, handlelength=2.6,
                 borderaxespad=0.3, labelspacing=0.3)

    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
    print(f"wrote {args.out}.pdf and .png")


if __name__ == "__main__":
    main()
