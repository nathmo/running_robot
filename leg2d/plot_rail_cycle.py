"""One steady gait cycle of the rail-sweep bound WITH actuator bandwidth (7 ms MIT delay):
joint position / velocity / acceleration / torque for cam and thigh, and end-effector (toe)
position / velocity / acceleration / mechanical power. Reproduces the exact best cycle from
results/rail_bound.json (pass2.arc_constrained.mit_7ms) and writes results/rail_cycle_mit7ms.png.

Run:  .venv/Scripts/python.exe leg2d/plot_rail_cycle.py
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
import rail_bound as rb  # noqa: E402
import motor  # noqa: E402

OUT = rb.RESULTS / "rail_cycle_mit7ms.png"

C_CAM, C_THIGH, C_EE, C_REF, C_Z = "#2a78d6", "#eb6834", "#1baf7a", "#8a8f98", "#6b7280"


def smooth(a, w=9):
    k = np.ones(w) / w
    return np.convolve(np.pad(a, w // 2, mode="edge"), k, mode="valid")


def main():
    meta = json.loads(rb.OUT_JSON.read_text())
    cfg = meta["pass2"]["arc_constrained"]["mit_7ms"]
    rec, arc_p, p1, base_h = rb.prepare()
    r = rb.pass2_tracked(p1, base_h, delay_ms=int(cfg["delay_ms"]), kp=cfg["kp"], kv=cfg["kv"],
                         k_force=cfg["k"])
    lg = {k: np.asarray(v) for k, v in r["log"].items() if k != "qpos"}
    T = 1.0 / r["f_hz"]
    m = (lg["t"] >= 2 * T) & (lg["t"] < 3 * T)             # one steady cycle
    t = (lg["t"][m] - 2 * T) * 1000.0                       # ms
    dt = 1e-3
    q, dq, tau = lg["q"][m], lg["dq"][m], lg["tau"][m]
    x, z = lg["x"][m], lg["z"][m]

    ddq = np.column_stack([np.gradient(smooth(dq[:, j]), dt) for j in range(2)])
    vx = np.gradient(smooth(x), dt)
    ax_ee = np.gradient(smooth(vx), dt)
    cap = rb.PEAK * np.clip(1.0 - np.abs(dq) / rb.NO_LOAD, 0.0, 1.0)   # motoring cap per joint
    power = np.sum(tau * dq, axis=1)
    v_mean = float(np.sum(np.abs(np.diff(x))) / (t[-1] - t[0]) * 1000.0)

    fig, axs = plt.subplots(4, 3, figsize=(13.5, 10.0), dpi=120, sharex=True)
    fig.suptitle(f"Rail-sweep bound, one gait cycle — envelope + 7 ms MIT delay:  "
                 f"v = {r['v_mean']:.2f} m/s,  f = {r['f_hz']:.2f} Hz,  "
                 f"extent = {r['extent']:.2f} m", fontsize=13, y=0.995)

    def style(a, ylab=None, title=None):
        a.grid(True, lw=0.5, alpha=0.35)
        a.axvspan(0, T * 500.0, color="#000000", alpha=0.045, lw=0)
        if ylab:
            a.set_ylabel(ylab, fontsize=9.5)
        if title:
            a.set_title(title, fontsize=11)

    # ---- row 0: position ----
    axs[0, 0].plot(t, q[:, 0], color=C_CAM, lw=1.8)
    style(axs[0, 0], "angle [rad]", "cam joint")
    axs[0, 1].plot(t, q[:, 1], color=C_THIGH, lw=1.8)
    style(axs[0, 1], None, "thigh joint")
    axs[0, 2].plot(t, x, color=C_EE, lw=1.8, label="x (fore-aft)")
    axs[0, 2].plot(t, base_h + z, color=C_Z, lw=1.4, label="z (height)")
    style(axs[0, 2], "position [m]", "end effector (toe)")
    axs[0, 2].legend(fontsize=8.5, loc="upper right", framealpha=0.8)

    # ---- row 1: velocity ----
    for j, (a, c) in enumerate(((axs[1, 0], C_CAM), (axs[1, 1], C_THIGH))):
        a.plot(t, dq[:, j], color=c, lw=1.8)
        for s in (+1, -1):
            a.axhline(s * rb.NO_LOAD, color=C_REF, lw=1.0, ls="--")
        a.text(t[-1], rb.NO_LOAD, " no-load 22", fontsize=8, color=C_REF, va="bottom", ha="right")
        style(a, "velocity [rad/s]" if j == 0 else None)
    axs[1, 2].plot(t, vx, color=C_EE, lw=1.8)
    axs[1, 2].axhline(0, color=C_REF, lw=0.8)
    axs[1, 2].text(0.02, 0.95, f"mean |vx| = {v_mean:.2f} m/s", transform=axs[1, 2].transAxes,
                   fontsize=9, va="top")
    style(axs[1, 2], "vx [m/s]")

    # ---- row 2: acceleration (finite-difference of 9 ms-smoothed velocity) ----
    axs[2, 0].plot(t, ddq[:, 0], color=C_CAM, lw=1.5)
    style(axs[2, 0], "accel [rad/s$^2$]")
    axs[2, 1].plot(t, ddq[:, 1], color=C_THIGH, lw=1.5)
    style(axs[2, 1])
    axs[2, 2].plot(t, ax_ee, color=C_EE, lw=1.5)
    style(axs[2, 2], "ax [m/s$^2$]")

    # ---- row 3: torque (applied) inside the speed-dependent motoring cap; EE: mech power ----
    for j, (a, c) in enumerate(((axs[3, 0], C_CAM), (axs[3, 1], C_THIGH))):
        a.fill_between(t, -cap[:, j], cap[:, j], color=c, alpha=0.13, lw=0,
                       label="available (motoring cap)")
        a.plot(t, tau[:, j], color=c, lw=1.6, label="applied")
        for s in (+1, -1):
            a.axhline(s * rb.PEAK, color=C_REF, lw=1.0, ls="--")
        a.text(t[-1], rb.PEAK, " peak 144.5", fontsize=8, color=C_REF, va="bottom", ha="right")
        style(a, "torque [N·m]" if j == 0 else None)
        a.set_xlabel("time in cycle [ms]", fontsize=9.5)
        if j == 0:
            a.legend(fontsize=8.5, loc="lower right", framealpha=0.8)
    axs[3, 2].plot(t, power, color=C_EE, lw=1.6)
    axs[3, 2].axhline(2 * motor.peak_power_w(), color=C_REF, lw=1.0, ls="--")
    axs[3, 2].text(t[-1], 2 * motor.peak_power_w(), " 2 motors × 795 W", fontsize=8,
                   color=C_REF, va="bottom", ha="right")
    axs[3, 2].axhline(0, color=C_REF, lw=0.8)
    style(axs[3, 2], "mech. power [W]")
    axs[3, 2].set_xlabel("time in cycle [ms]", fontsize=9.5)

    axs[0, 0].text(0.02, 0.05, "shaded half = forward stroke", transform=axs[0, 0].transAxes,
                   fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT)
    print(f"wrote {OUT}  (v={r['v_mean']:.2f} m/s, f={r['f_hz']:.2f} Hz, "
          f"sat={r['sat_frac']:.2f}, rms={r['rms_tau']:.1f} N*m)")


if __name__ == "__main__":
    main()
