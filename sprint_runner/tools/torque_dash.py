#!/usr/bin/env python3
"""Per-actuator torque over the 100 m dash: the trace, the peak and the RMS.

  python tools/torque_dash.py                                   # sprint_m3_mit_s0, 8 greedy dashes
  python tools/torque_dash.py --run walk_mit/runs/sprint_m3_mit_s1 --episodes 4 --tag s1

Protocol is `evaluate.py`'s: the run's own resolved_config + curriculum, greedy actions, pushes
off (a shove is a TRAINING disturbance), one fresh seed per episode. So the speed printed here is
the same 3.06-3.08 m/s the run archive reports, and the torque numbers belong to THAT dash.

SAMPLING IS AT PHYSICS RATE, NOT CONTROL RATE. `mujoco.mj_step` is wrapped so every 1 ms substep
contributes a sample; reading `actuator_force` once per 5 ms control step aliases the contact
spikes and under-reads the peak. Substeps logged during the auto-reset that follows a terminal
step are discarded (the per-control-step marks are the truth).

Outputs, in results/:
  torque_dash_<tag>.{png,pdf}     6 panels, one per actuator, whole dash + peak/cont limits
  torque_stride_<tag>.{png,pdf}   the same six over one 1 s window at steady state, with the
                                  foot-contact raster underneath
  torque_bars_<tag>.{png,pdf}     peak and RMS per actuator against the motor's ratings
  torque_dash_<tag>.json          the numbers (per episode and pooled)
  torque_dash_<tag>.npz           the raw 1 kHz trace of the plotted episode
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
sys.path.insert(0, str(ROOT / "walk_mit"))
import evaluate                                    # noqa: E402  (walk_mit/evaluate.py)

# palette shared with the other thesis figures (plot_sprint_dash.py)
C_L, C_R = "#2a78d6", "#eb6834"
INK, INK2, GRID, CRIT, CONT = "#0b0b0b", "#52514e", "#e1e0d9", "#d03b3b", "#b08900"

# measured motor envelope (see the limits audit): AKE90-8 on cam+thigh, AK60-39 on hip_roll.
# peak = the model's forcerange, by construction; continuous is the thermal rating, and the
# AK60 number is an ESTIMATE -- its continuous torque was never measured on the bench.
CONT_NM = {"hip_roll_L": 23.3, "hip_roll_R": 23.3, "cam_L": 55.0, "cam_R": 55.0,
           "thigh_L": 55.0, "thigh_R": 55.0}
CONT_EST = ("hip_roll_L", "hip_roll_R")


def act_names(model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]


class SubstepTap:
    """Wraps mujoco.mj_step so every physics substep appends (tau, joint speed)."""

    def __init__(self, raw):
        self.raw, self.on = raw, False
        self.tau, self.w, self.marks = [], [], []
        self._real = mujoco.mj_step
        nu, dadr = raw.nu, raw.act_dadr

        def stepped(m, d, *a, **k):
            self._real(m, d, *a, **k)
            if self.on:
                self.tau.append(d.actuator_force[:nu].copy())
                self.w.append(d.qvel[dadr].copy())
        mujoco.mj_step = stepped

    def restore(self):
        mujoco.mj_step = self._real

    def clear(self):
        self.tau, self.w, self.marks = [], [], []

    def take(self):
        """Substeps up to the last completed control step, as (tau, w) arrays."""
        n = self.marks[-1] if self.marks else 0
        return np.array(self.tau[:n]), np.array(self.w[:n])


def stats(tau, peak_lim, cont_lim):
    a = np.abs(tau)
    rms = float(np.sqrt(np.mean(tau ** 2)))
    return dict(rms=rms, peak=float(a.max()), p95=float(np.percentile(a, 95)),
                mean_abs=float(a.mean()), peak_pct=float(a.max() / peak_lim * 100.0),
                rms_pct_cont=float(rms / cont_lim * 100.0),
                duty_95=float(np.mean(a >= 0.95 * peak_lim) * 100.0))


def rollout(model_, venv, raw, tap, seed):
    """One greedy dash. Returns the 1 kHz torque/speed trace + per-control-step log."""
    log = dict(t=[], x=[], vx=[], cL=[], cR=[])
    n = [0]

    def on_ctrl():
        n[0] += 1
        log["t"].append(n[0] * raw.control_dt)
        log["x"].append(float(raw.data.qpos[0]))
        log["vx"].append(float(raw._vel_body()[0]))
        c = raw._foot_contacts()
        log["cL"].append(bool(c[0]))
        log["cR"].append(bool(c[1]))
        tap.marks.append(len(tap.tau))

    raw.on_control_step = on_ctrl
    venv.seed(seed)
    tap.on = False
    obs = venv.reset()
    tap.clear()
    tap.on = True
    done, sprint, ep_ret = [False], None, 0.0
    while not done[0]:
        a, _ = model_.predict(obs, deterministic=True)
        obs, r, done, info = venv.step(a)
        ep_ret += float(r[0])
        sprint = info[0].get("sprint", sprint)
    tap.on = False
    tau, w = tap.take()
    raw.on_control_step = None
    return tau, w, {k: np.array(v) for k, v in log.items()}, sprint, ep_ret


def table(names, rows, peak_lim):
    hdr = (f"  {'actuator':<11} {'RMS':>8} {'peak':>8} {'P95':>8} {'peak lim':>9} {'%peak':>7} "
           f"{'cont':>7} {'RMS/cont':>9} {'sat%':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, nm in enumerate(names):
        s, c = rows[nm], CONT_NM[nm]
        star = "*" if nm in CONT_EST else " "
        print(f"  {nm:<11} {s['rms']:8.1f} {s['peak']:8.1f} {s['p95']:8.1f} {peak_lim[i]:9.1f} "
              f"{s['peak_pct']:6.0f}% {c:6.1f}{star} {s['rms_pct_cont']:8.0f}% {s['duty_95']:5.1f}%")
    print("  torque in N*m; 'sat%' = share of 1 kHz samples at >=95% of the peak limit")
    print("  * AK60-39 continuous rating is an ESTIMATE (never measured on the bench)")


def fig_dash(names, tau, dt, peak_lim, rows, tag, title):
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.2), sharex=True)
    t = np.arange(len(tau)) * dt
    for i, nm in enumerate(names):
        ax = axes[i % 3, i // 3]          # left column = left leg, right column = right leg
        col = C_L if nm.endswith("_L") else C_R
        cont = CONT_NM[nm]
        ax.axhspan(-cont, cont, color=CONT, alpha=0.08, lw=0, zorder=0)
        for s in (1, -1):
            ax.axhline(s * peak_lim[i], color=CRIT, lw=1.0, ls="--", zorder=1)
            ax.axhline(s * cont, color=CONT, lw=0.9, ls=":", zorder=1)
        ax.plot(t, tau[:, i], color=col, lw=0.35, rasterized=True, zorder=2)
        s = rows[nm]
        ax.axhline(s["rms"], color=INK, lw=1.0, zorder=3)
        ax.axhline(-s["rms"], color=INK, lw=1.0, zorder=3)
        ax.set_ylim(-1.18 * peak_lim[i], 1.18 * peak_lim[i])
        ax.set_title(f"{nm}   RMS {s['rms']:.1f}   peak {s['peak']:.1f} N*m "
                     f"({s['peak_pct']:.0f}% of {peak_lim[i]:.0f})",
                     fontsize=9.5, color=INK, loc="left", pad=4)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK2)
        if i // 3 == 0:
            ax.set_ylabel("torque  [N*m]", fontsize=9, color=INK2)
    for ax in axes[2]:
        ax.set_xlabel("time  [s]", fontsize=9, color=INK2)
    axes[0, 1].plot([], [], color=CRIT, ls="--", lw=1.0, label="peak limit (forcerange)")
    axes[0, 1].plot([], [], color=CONT, ls=":", lw=0.9, label="continuous rating")
    axes[0, 1].plot([], [], color=INK, lw=1.0, label="+/- RMS")
    axes[0, 1].legend(fontsize=7.5, frameon=False, ncol=3, loc="upper right",
                      bbox_to_anchor=(1.0, 1.32))
    fig.suptitle(title, fontsize=11, color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, f"torque_dash_{tag}")


def fig_stride(names, tau, dt, peak_lim, log, control_dt, t0, win, tag, title):
    i0, i1 = int(t0 / dt), int((t0 + win) / dt)
    t = np.arange(i0, i1) * dt
    fig, axes = plt.subplots(4, 1, figsize=(9, 8.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 1, 0.42]))
    pairs = [("hip_roll_L", "hip_roll_R"), ("cam_L", "cam_R"), ("thigh_L", "thigh_R")]
    for ax, (nl, nr) in zip(axes[:3], pairs):
        for nm, col in ((nl, C_L), (nr, C_R)):
            i = names.index(nm)
            ax.plot(t, tau[i0:i1, i], color=col, lw=1.4, label=nm)
            ax.axhline(peak_lim[i], color=CRIT, lw=0.9, ls="--")
            ax.axhline(-peak_lim[i], color=CRIT, lw=0.9, ls="--")
        ax.set_ylabel("torque  [N*m]", fontsize=9, color=INK2)
        ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK2)
    ras = axes[3]
    k0, k1 = int(t0 / control_dt), int((t0 + win) / control_dt)
    tc = np.arange(k0, k1) * control_dt
    for row, (key, col, lab) in enumerate((("cL", C_L, "left foot"), ("cR", C_R, "right foot"))):
        on = log[key][k0:k1]
        ras.fill_between(tc, row + 0.12, row + 0.88, where=on, color=col, lw=0, step="post")
        ras.text(t0 + 0.005, row + 0.5, lab, fontsize=8, color=INK2, va="center")
    ras.set_ylim(0, 2)
    ras.set_yticks([])
    ras.set_xlabel("time  [s]", fontsize=9, color=INK2)
    ras.set_xlim(t0, t0 + win)
    for sp in ("top", "right", "left"):
        ras.spines[sp].set_visible(False)
    ras.tick_params(labelsize=8, colors=INK2)
    ras.set_ylabel("stance", fontsize=9, color=INK2)
    fig.suptitle(title, fontsize=11, color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, f"torque_stride_{tag}")


def fig_bars(names, rows, peak_lim, tag, title):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(names))
    rms = [rows[n]["rms"] for n in names]
    pk = [rows[n]["peak"] for n in names]
    cols = [C_L if n.endswith("_L") else C_R for n in names]
    ax.bar(x - 0.19, rms, 0.36, color=cols, edgecolor="white", lw=2)
    ax.bar(x + 0.19, pk, 0.36, color=cols, alpha=0.42, edgecolor="white", lw=2)
    for xi, (r, p) in enumerate(zip(rms, pk)):
        ax.text(xi - 0.19, r + 2.5, f"{r:.0f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi + 0.19, p + 2.5, f"{p:.0f}", ha="center", fontsize=8.5, color=INK)
    for xi, nm in enumerate(names):
        ax.plot([xi - 0.42, xi + 0.42], [peak_lim[xi]] * 2, color=CRIT, lw=1.6, ls="--")
        ax.plot([xi - 0.42, xi + 0.42], [CONT_NM[nm]] * 2, color=CONT, lw=1.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, color=INK)
    ax.set_ylabel("torque  [N*m]", fontsize=9, color=INK2)
    ax.plot([], [], color=CRIT, ls="--", lw=1.6, label="peak limit")
    ax.plot([], [], color=CONT, ls=":", lw=1.6, label="continuous rating")
    ax.bar([], [], color=INK2, label="RMS (solid) / peak (pale)")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK2)
    ax.set_ylim(0, max(peak_lim) * 1.22)
    fig.suptitle(title, fontsize=11, color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, f"torque_bars_{tag}")


def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/{stem}.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="walk_mit/runs/sprint_m3_mit_s0")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--trace-ep", type=int, default=0, help="which episode the figures plot")
    ap.add_argument("--stride-t", type=float, default=16.0, help="start of the 1 s zoom window")
    ap.add_argument("--stride-win", type=float, default=1.0)
    ap.add_argument("--settle-s", type=float, default=2.0,
                    help="launch transient excluded from the steady-run table")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    tag = args.tag or run.name.replace("sprint_m3_mit_", "")
    model_, venv, raw = evaluate.build(run, None, None)
    names = act_names(raw.model)
    peak_lim = raw.model.actuator_forcerange[:raw.nu, 1].copy()
    dt, cdt = raw.sim_dt, raw.control_dt
    tap = SubstepTap(raw)

    per_ep, keep, speeds = [], None, []
    try:
        for e in range(args.episodes):
            tau, w, log, sprint, ep_ret = rollout(model_, venv, raw, tap, args.seed0 + e)
            rows = {nm: stats(tau[:, i], peak_lim[i], CONT_NM[nm]) for i, nm in enumerate(names)}
            d = sprint["d"] if sprint else float(log["x"][-1])
            t_line = sprint["t_line"] if sprint else None
            # the steady run: launch transient and the post-line stop are not "running torque"
            j0 = int(args.settle_s / dt)
            j1 = int(t_line / dt) if t_line else len(tau)
            steady = {nm: stats(tau[j0:j1, i], peak_lim[i], CONT_NM[nm])
                      for i, nm in enumerate(names)}
            per_ep.append(dict(seed=args.seed0 + e, steps=len(log["t"]), dash_s=float(log["t"][-1]),
                               dist_m=float(d), t_line=t_line, ret=ep_ret,
                               vx_mean=float(log["vx"].mean()), vx_peak=float(log["vx"].max()),
                               samples=int(len(tau)), torque=rows, torque_steady=steady,
                               steady_window_s=[args.settle_s, float(j1 * dt)],
                               peak_mech_w=float(np.abs(tau * w).sum(axis=1).max())))
            speeds.append(float(log["vx"].mean()))
            line_s = f"line {t_line:.2f} s" if t_line else "DNF"
            print(f"  ep{e} seed {args.seed0 + e}: {d:6.1f} m  {line_s}  "
                  f"{log['vx'].mean():.2f} m/s mean  {len(tau)} torque samples")
            if e == args.trace_ep:
                keep = (tau, w, log, sprint)
    finally:
        tap.restore()

    # pooled over every dash: peak = worst sample anywhere, RMS = root-mean-square over all of them
    def pool(key):
        out = {}
        for i, nm in enumerate(names):
            rmss = np.array([ep[key][nm]["rms"] for ep in per_ep])
            out[nm] = dict(rms=float(np.sqrt(np.mean(rmss ** 2))), rms_sd=float(rmss.std()),
                           peak=float(max(ep[key][nm]["peak"] for ep in per_ep)),
                           p95=float(np.mean([ep[key][nm]["p95"] for ep in per_ep])),
                           duty_95=float(np.mean([ep[key][nm]["duty_95"] for ep in per_ep])))
            out[nm]["peak_pct"] = out[nm]["peak"] / peak_lim[i] * 100.0
            out[nm]["rms_pct_cont"] = out[nm]["rms"] / CONT_NM[nm] * 100.0
        return out

    pooled, pooled_steady = pool("torque"), pool("torque_steady")

    dist = np.array([ep["dist_m"] for ep in per_ep])
    secs = np.array([ep["dash_s"] for ep in per_ep])
    print(f"\n  {args.episodes} greedy dashes: {dist.mean():.1f}+/-{dist.std():.1f} m in "
          f"{secs.mean():.2f}+/-{secs.std():.2f} s, mean body speed {np.mean(speeds):.2f} m/s")
    print(f"\n  WHOLE DASH, pooled over {args.episodes} episodes "
          f"({sum(ep['samples'] for ep in per_ep)} samples at {1 / dt:.0f} Hz)\n")
    table(names, pooled, peak_lim)
    print(f"\n  STEADY RUN ONLY (t = {args.settle_s:.0f} s to the line; launch and the post-line "
          f"stop excluded)\n")
    table(names, pooled_steady, peak_lim)

    tau, w, log, sprint = keep
    ep = per_ep[args.trace_ep]
    rows = ep["torque"]
    sub = (f"{run.name} - greedy 100 m dash, seed {ep['seed']}, {ep['dash_s']:.2f} s at "
           f"{ep['vx_mean']:.2f} m/s (1 kHz per-substep torque)")
    print()
    fig_dash(names, tau, dt, peak_lim, rows, tag, "Per-actuator torque over the dash\n" + sub)
    fig_stride(names, tau, dt, peak_lim, log, cdt, args.stride_t, args.stride_win, tag,
               f"One second at steady state (t = {args.stride_t:.0f}-"
               f"{args.stride_t + args.stride_win:.0f} s)\n" + sub)
    fig_bars(names, pooled_steady, peak_lim, tag,
             f"Torque demand vs motor ratings - steady run, pooled over {args.episodes} dashes"
             f"\n{run.name}, t = {args.settle_s:.0f} s to the line")

    OUT.mkdir(exist_ok=True)
    (OUT / f"torque_dash_{tag}.json").write_text(json.dumps(
        dict(run=str(run), episodes=args.episodes, seed0=args.seed0, sample_hz=1.0 / dt,
             actuators=names, peak_limit_nm=peak_lim.tolist(),
             cont_rating_nm=[CONT_NM[n] for n in names], cont_estimated=list(CONT_EST),
             dist_mean_m=float(dist.mean()), dash_s_mean=float(secs.mean()),
             vx_mean=float(np.mean(speeds)), settle_s=args.settle_s, pooled=pooled,
             pooled_steady=pooled_steady, per_episode=per_ep), indent=1))
    np.savez_compressed(OUT / f"torque_dash_{tag}.npz", tau=tau.astype(np.float32),
                        w=w.astype(np.float32), sim_dt=dt, control_dt=cdt,
                        names=np.array(names), peak_limit=peak_lim, **log)
    print(f"  wrote results/torque_dash_{tag}.json, results/torque_dash_{tag}.npz")


if __name__ == "__main__":
    main()
