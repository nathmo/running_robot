#!/usr/bin/env python3
"""Thesis figure: measured noise of the DASH-01 IMU (ICM-20948 on the Sense HAT B) at rest.

Data: a long still record captured on the Pi by
    sudo systemctl stop runningrobot-webui        # it owns the I2C bus
    python robot/fixed_gait/webui/tools/imu_bench.py --seconds 300 --save /tmp/imu_noise_300s.npz
    sudo systemctl start runningrobot-webui
then scp'd to robot/fixed_gait/webui/data/measurements/. The bench gates the record on a
stillness check and the verdict travels inside the npz; a record that failed the gate renders
with a CONTAMINATED stamp across the figure rather than quietly plotting sway as sensor noise.

Two views per channel (accel top row, gyro bottom row):
  excerpt  10 s of raw traces, mean-removed, vertically offset for legibility
  ASD      Welch amplitude spectral density, against the datasheet density and the DLPF corner

Outputs results/imu_noise.{pdf,png} and prints the headline numbers (RMS, spectral density,
Allan minimum — the Allan deviation is computed for stdout because it feeds IMU.md's
bias-instability row, but it gets no panel). Densities are read off the flat 1-30 Hz band of
the ASD, which is the honest estimate.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
NPZ_DEFAULT = ROOT / "robot/fixed_gait/webui/data/measurements/imu_noise_300s.npz"
OUT = ROOT / "results"

# Datasheet, FCHOICE=1: DLPF cfg -> 3 dB bandwidth in Hz (same table as imu_bench.py).
GYR_3DB = {0: 196.6, 1: 151.8, 2: 119.5, 3: 51.2, 4: 23.9, 5: 11.6, 6: 5.7}
ACC_3DB = {0: 246.0, 1: 246.0, 2: 111.4, 3: 50.4, 4: 23.9, 5: 11.5, 6: 5.7}
DS_ACC_UG = 230.0        # datasheet accel noise density, ug/sqrt(Hz)
DS_GYR_DPS = 0.015       # datasheet gyro noise density, dps/sqrt(Hz)

# ---------------------------------------------------------------- palette (validated, light mode)
# X/Y/Z = categorical slots 1-3, the trio documented to pass all-pairs validation.
C_AX = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a"}
INK, INK2, GRID, CRIT = "#0b0b0b", "#52514e", "#e1e0d9", "#d03b3b"


def allan_dev(x, fs, taus):
    """Overlapping Allan deviation of a rate signal x (any unit U) -> sigma(tau) in U."""
    theta = np.cumsum(x) / fs
    out = np.full(len(taus), np.nan)
    for i, tau in enumerate(taus):
        m = int(round(tau * fs))
        if m < 1 or 2 * m >= len(theta):
            continue
        d = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
        out[i] = np.sqrt(np.mean(d * d) / (2.0 * (m / fs) ** 2))
    return out


def asd(x, fs):
    """Welch amplitude spectral density (unit/sqrt(Hz)); linear detrend kills slow drift."""
    nseg = min(8192, len(x) // 6)
    f, p = signal.welch(x, fs, nperseg=nseg, noverlap=nseg // 2, detrend="linear")
    return f[1:], np.sqrt(p[1:])


def flat_density(f, a):
    """Median ASD over the flat 1-30 Hz band, well inside the DLPF corner."""
    return float(np.median(a[(f >= 1.0) & (f <= 30.0)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, default=NPZ_DEFAULT)
    args = ap.parse_args()

    z = np.load(args.npz)
    acc, gyr, ts = z["acc_g"], z["gyr_dps"], z["t_s"]
    still = bool(z["still"])
    cfg = int(z["dlpf_cfg"])
    fs = 1.0 / np.diff(ts).mean()
    T = ts[-1] - ts[0]
    acc0 = (acc - acc.mean(0)) * 1e3          # mg, mean-removed
    gyr0 = gyr - gyr.mean(0)                  # dps, mean-removed
    if not still:
        print("WARNING: record failed the stillness gate — this figure shows robot motion, "
              "not sensor noise. It will be stamped CONTAMINATED.")

    taus = np.unique(np.round(np.logspace(np.log10(2 / fs), np.log10(T / 5), 48) * fs)) / fs

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8.5, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.linewidth": 0.8, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.4), dpi=300,
                             gridspec_kw={"width_ratios": [1.0, 1.3],
                                          "hspace": 0.42, "wspace": 0.45})
    for ax in axes.flat:
        ax.grid(True, which="both", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(which="both", length=3)

    rows = (("accel", acc0, "mg", 1e3, ACC_3DB[cfg], DS_ACC_UG / 1e3, 12.0),
            ("gyro", gyr0, "dps", 1.0, GYR_3DB[cfg], DS_GYR_DPS, 0.5))
    headlines = {}
    for r, (name, sig0, unit, _scale, f3db, ds_dens, offset) in enumerate(rows):
        ax_t, ax_f = axes[r]

        # ---- 10 s excerpt, offset traces
        i0 = np.searchsorted(ts, ts[0] + T / 2)
        i1 = np.searchsorted(ts, ts[0] + T / 2 + 10.0)
        tt = ts[i0:i1] - ts[i0]
        for k, axn in enumerate("xyz"):
            ax_t.plot(tt, sig0[i0:i1, k] + (1 - k) * offset, color=C_AX[axn],
                      lw=0.45, solid_capstyle="round")
            ax_t.text(10.15, (1 - k) * offset, axn.upper(), color=C_AX[axn],
                      fontsize=7.5, va="center", ha="left")
        ax_t.set_xlim(0, 10)
        ax_t.set_xlabel("Time (s)")
        ax_t.set_ylabel(f"{name} ({unit}, offset)")

        # ---- amplitude spectral density
        dens, lo, hi = [], np.inf, 0.0
        for k, axn in enumerate("xyz"):
            f, a = asd(sig0[:, k], fs)
            ax_f.plot(f, a, color=C_AX[axn], lw=0.9)
            dens.append(flat_density(f, a))
            lo = min(lo, a.min())
            hi = max(hi, a.max())
        ax_f.set_xscale("log")
        ax_f.set_yscale("log")
        ax_f.set_xlim(0.03, fs / 2)
        lo = min(lo * 0.7, min(dens) * 0.2)
        top = max(ds_dens * 2.0, hi * 1.5)
        ax_f.set_ylim(lo, top)
        ax_f.axhline(ds_dens, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax_f.text(0.04, ds_dens * 1.12, "datasheet", fontsize=7, color=INK2, va="bottom")
        ax_f.axvline(f3db, color=INK2, lw=0.8, ls=(0, (1, 2)))
        ax_f.text(f3db * 1.12, top * 0.85, f"DLPF {f3db:.0f} Hz",
                  fontsize=7, color=INK2, rotation=90, va="top", ha="left")
        ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_ylabel(f"ASD ({unit}/$\\sqrt{{\\mathrm{{Hz}}}}$)")

        # ---- Allan statistics: stdout only (they feed IMU.md's bias-instability row)
        admin = []
        for k in range(3):
            ad = allan_dev(sig0[:, k], fs, taus)
            admin.append((np.nanmin(ad), taus[np.nanargmin(ad)]))
        headlines[name] = (sig0.std(0), dens, admin)

    handles = [plt.Line2D([], [], color=C_AX[axn], lw=1.4, label=axn.upper())
               for axn in "xyz"]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.005), handlelength=1.8, columnspacing=1.4)
    if not still:
        fig.text(0.5, 0.5, "CONTAMINATED BY MOTION", color=CRIT, fontsize=26,
                 fontweight="bold", ha="center", va="center", rotation=18, alpha=0.85)

    # ---- headline numbers
    print(f"record: {len(acc)} samples at {fs:.1f} Hz ({T:.0f} s), DLPF cfg {cfg}, "
          f"still={still}")
    for name, unit, dscale in (("accel", "mg", 1.0), ("gyro", "dps", 1.0)):
        rms, dens, admin = headlines[name]
        du = "ug/rtHz" if name == "accel" else "dps/rtHz"
        dv = [d * (1e3 if name == "accel" else 1.0) for d in dens]   # mg->ug for accel
        print(f"  {name}: RMS {np.array2string(rms, precision=3)} {unit}, "
              f"ASD flat band {np.array2string(np.array(dv), precision=4)} {du}")
        for (mn, tmn), axn in zip(admin, "XYZ"):
            print(f"    Allan {axn}: min {mn:.4g} {unit} at tau {tmn:.1f} s")

    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"imu_noise.{ext}", bbox_inches="tight")
    print(f"wrote {OUT / 'imu_noise.pdf'} and .png")


if __name__ == "__main__":
    main()
