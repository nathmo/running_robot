"""Diagnose a policy's GAIT: stepping frequency, per-foot duty factor + L/R symmetry, joint
velocities vs the motor ceiling, and torque saturation. Built to check the cadence experiment
(are the motor limits + contact-switch penalty producing a slower, two-legged gait instead of the
k350 winner's ~11 Hz one-legged pattering?).

    python training/diagnose_cadence.py --run training/runs_dl/m3_stiff_hi --episodes 6

Reports per foot: touchdown rate (Hz), ground-contact duty (%); overall stride frequency and an
L/R asymmetry index; per joint: peak / 95pct |vel| vs the 22 rad/s (210 RPM) motor ceiling and the
fraction of steps over it; peak |torque| vs forcerange and the saturation fraction. Writes a
contact-raster + joint-velocity PNG.
"""
import argparse
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import build

MOTOR_RAD_S = 22.0        # 210 RPM = 22 rad/s (motor-side; joint sees it through the reduction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    model, venv, raw = build(run, args.preset, args.checkpoint)
    names = [mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a).replace("NCS", "")
             for a in range(raw.nu)]
    frng = raw.model.actuator_forcerange[:raw.nu, 1].copy()

    C, V, T, EP = [], [], [], []
    ep = {"i": 0}

    def hook():
        C.append(raw._foot_contacts().copy())
        V.append(raw.data.qvel[raw.act_dadr].copy())
        T.append(raw.data.actuator_force[:raw.nu].copy())
        EP.append(ep["i"])
    raw.on_control_step = hook
    for e in range(args.episodes):
        ep["i"] = e
        obs = venv.reset()
        done = [False]
        while not done[0]:
            a, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _, done, _ = venv.step(a)

    C = np.array(C); V = np.abs(np.array(V)); T = np.abs(np.array(T)); EP = np.array(EP)
    dt = raw.control_dt; total_s = len(C) * dt

    # per-foot touchdowns within episodes
    def foot_stats(fi):
        tds = 0
        for e in range(EP.max() + 1):
            g = C[EP == e, fi].astype(int)
            tds += int(np.sum((g[1:] == 1) & (g[:-1] == 0)))
        duty = 100.0 * C[:, fi].mean()
        return tds, tds / total_s, duty

    (tdL, rL, dutyL), (tdR, rR, dutyR) = foot_stats(0), foot_stats(1)
    stride_hz = (rL + rR) / 2.0                       # avg per-foot stepping frequency
    asym = abs(dutyL - dutyR) / max(dutyL + dutyR, 1e-9)   # 0 = symmetric, 1 = one-legged

    print(f"\n=== cadence / gait: {run.name} ===")
    print(f"episodes {EP.max()+1}  mean ep_len {len(C)/(EP.max()+1):.0f}  ({total_s:.1f} s total)")
    print(f"stepping freq: L {rL:.2f} Hz (duty {dutyL:.0f}%)  R {rR:.2f} Hz (duty {dutyR:.0f}%)  "
          f"-> avg {stride_hz:.2f} Hz/foot")
    print(f"L/R asymmetry index: {asym:.2f}  (0=even two-legged, 1=one-legged)  "
          f"{'*** ONE-LEGGED/CHATTER ***' if asym > 0.5 or stride_hz > 4 else 'ok'}")
    print(f"\n{'joint':11} peak|v| 95pct  over-22rad/s%%   peak|trq|/lim  sat%%")
    for i, nm in enumerate(names):
        v = V[:, i]; t = T[:, i]
        print(f"{nm:11} {v.max():6.1f} {np.percentile(v,95):6.1f}   {100*np.mean(v>MOTOR_RAD_S):5.1f}       "
              f"{t.max():6.0f}/{frng[i]:.0f}   {100*np.mean(t>0.95*frng[i]):4.1f}")

    # ---- plot: contact raster (first ~8 s of ep0) + a thigh-velocity trace ----
    m = EP == 0
    tt = np.arange(m.sum()) * dt
    n = min(len(tt), int(8.0 / dt))
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax[0].fill_between(tt[:n], 0, C[m][:n, 0], step="pre", alpha=0.7, label=f"L (duty {dutyL:.0f}%)")
    ax[0].fill_between(tt[:n], 1.2, 1.2 + C[m][:n, 1], step="pre", alpha=0.7,
                       label=f"R (duty {dutyR:.0f}%)")
    ax[0].set_yticks([0.5, 1.7]); ax[0].set_yticklabels(["L", "R"])
    ax[0].set_title(f"{run.name}: foot ground-contact raster  "
                    f"(stepping {stride_hz:.1f} Hz/foot, asymmetry {asym:.2f})")
    ax[0].legend(fontsize=8, loc="upper right"); ax[0].grid(True, alpha=0.3)
    thigh = [i for i, nm in enumerate(names) if "thigh" in nm]
    for i in thigh:
        ax[1].plot(tt[:n], np.array(V)[m][:n, i], lw=0.9, label=names[i])
    ax[1].axhline(MOTOR_RAD_S, color="r", ls="--", lw=0.8, label="22 rad/s (210 RPM)")
    ax[1].set_ylabel("|thigh vel| rad/s"); ax[1].set_xlabel("time [s]")
    ax[1].legend(fontsize=8, loc="upper right"); ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = args.out or str(run / "cadence_diag.png")
    fig.savefig(out, dpi=120); print(f"wrote {out}")


if __name__ == "__main__":
    main()
