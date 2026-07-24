"""Diagnose foot-placement vs CoM for a trained policy: does the stance foot land AHEAD of the
whole-robot CoM (the capture step that catches a forward fall), or does the policy step behind /
under the CoM and topple?  Rolls out the deterministic policy, records per control step the
forward offset (toe_x - com_x) of each foot, the ground contact, and the base pitch, then writes
a diagnostic PNG + prints touchdown statistics.

    python training/diagnose_foot_com.py --run training/runs_dl/m3_reactive_x_dbg --episodes 6
"""
import argparse
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions like training (these policies kept std=1, so the greedy "
                         "mean is off-distribution — stochastic is closer to the training curve)")
    ap.add_argument("--out", default=None, help="output PNG (default: <run>/foot_com_diag.png)")
    args = ap.parse_args()
    run = Path(args.run)
    model, venv, raw = build(run, args.preset, args.checkpoint)

    rec = []          # per control step: (ep, t, comx, toeL, toeR, gL, gR, pitch, vx)
    state = {"ep": 0}

    def on_ctrl():
        com_x = float(raw.data.subtree_com[0][0])
        toe = raw.data.geom_xpos[raw.foot_gids_arr, 0]
        g = raw._foot_contacts()
        rec.append((state["ep"], float(raw._elapsed_t), com_x, float(toe[0]), float(toe[1]),
                    bool(g[0]), bool(g[1]), float(raw._gravity_body()[0]), float(raw._vel_body()[0])))
    raw.on_control_step = on_ctrl

    for ep in range(args.episodes):
        state["ep"] = ep
        obs = venv.reset()
        done = [False]
        while not done[0]:
            a, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _, done, _ = venv.step(a)

    rec = np.array(rec, dtype=float)
    ep = rec[:, 0].astype(int)
    t = rec[:, 1]
    offL = rec[:, 3] - rec[:, 2]     # toe_x - com_x  (>0 = foot ahead of CoM in +x heading)
    offR = rec[:, 4] - rec[:, 2]
    gL = rec[:, 5].astype(bool)
    gR = rec[:, 6].astype(bool)
    pitch = rec[:, 7]                # grav_x: + = nose-down (falling forward)

    # touchdowns: contact rising edge, per foot, within an episode
    def touchdowns(g, off):
        td_off = []
        for e in range(int(ep.max()) + 1):
            m = ep == e
            gi, oi = g[m], off[m]
            for k in range(1, len(gi)):
                if gi[k] and not gi[k - 1]:
                    td_off.append(oi[k])
        return np.array(td_off)

    tdL, tdR = touchdowns(gL, offL), touchdowns(gR, offR)
    td_all = np.concatenate([tdL, tdR]) if len(tdL) + len(tdR) else np.array([])

    print(f"\n=== foot-placement vs CoM: {run.name} ===")
    print(f"control steps recorded: {len(rec)}   episodes: {int(ep.max())+1}   "
          f"mean ep_len: {len(rec)/(int(ep.max())+1):.0f}")
    if len(td_all):
        ahead = float((td_all > 0).mean()) * 100.0
        print(f"touchdowns: {len(td_all)}   landed AHEAD of CoM: {ahead:.0f}%   "
              f"mean offset at touchdown: {td_all.mean()*100:+.1f} cm  "
              f"(L {tdL.mean()*100:+.1f}, R {tdR.mean()*100:+.1f} cm)")
    # while grounded (stance), where is the foot relative to CoM on average?
    stance_off = np.concatenate([offL[gL], offR[gR]])
    if len(stance_off):
        print(f"stance (grounded) foot offset: mean {stance_off.mean()*100:+.1f} cm   "
              f"ahead-of-CoM fraction: {(stance_off>0).mean()*100:.0f}%")
    print(f"base pitch (grav_x, + nose-down): mean {pitch.mean():+.3f}  max {pitch.max():+.3f}  "
          f"(termination at grav_z>-0.5 ~ pitch~0.87)")

    # ---- plot the first few episodes ----
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    show = ep <= min(3, int(ep.max()))
    ax = axes[0]
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(t[show], offL[show] * 100, color="tab:blue", lw=1.0, label="L foot  (toe_x - CoM_x)")
    ax.plot(t[show], offR[show] * 100, color="tab:orange", lw=1.0, label="R foot")
    ax.scatter(t[show & gL], offL[show & gL] * 100, s=8, color="tab:blue", alpha=0.5,
               label="L grounded")
    ax.scatter(t[show & gR], offR[show & gR] * 100, s=8, color="tab:orange", alpha=0.5,
               label="R grounded")
    for e in range(int(ep[show].max()) + 1):       # episode reset lines
        te = t[ep == e]
        if len(te):
            ax.axvline(te[0], color="grey", ls=":", lw=0.8)
    ax.set_ylabel("foot ahead of CoM  [cm]\n(>0 = foot forward)")
    ax.set_title(f"{run.name}: foot contact point vs CoM  "
                 f"(ahead-of-CoM at touchdown: "
                 f"{(td_all>0).mean()*100 if len(td_all) else 0:.0f}%)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(t[show], pitch[show], color="tab:red", lw=1.0, label="base pitch (grav_x, + nose-down)")
    ax.set_ylabel("pitch  (grav_x)")
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = args.out or str(run / "foot_com_diag.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
