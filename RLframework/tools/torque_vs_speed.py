"""What does each motor SPEND, as a function of commanded speed?

The joystick policy has one gait and one stick, and the interesting engineering question about it
is not "does it track?" (command_sweep.py answers that) but "what does the machine pay for each
rung of the stick?". This holds a stick position, runs the SHIPPING control law
(controller/deploy/controller_v2.PolicyControllerV2, the numpy path the Pi executes) against CPU
MuJoCo, and reports per-actuator peak and RMS torque at each speed.

    python RLframework/tools/torque_vs_speed.py \
        --bundle controller/deploy/bundles/dash_joy3_lr_s2_180M.npz

Three things decide whether the numbers mean anything:

  1. **Sampling is at physics rate, not control rate.** `mujoco.mj_step` is wrapped so every 1 ms
     substep contributes a sample. Reading `actuator_force` once per 10 ms control tick aliases
     the touchdown spikes badly: measured on this bundle 2026-09-24, the control-rate peak is 0 to
     49% below the 1 kHz peak, worst on the thighs at low speed (thigh_L at 0.59 m/s reads 55.4
     against the true 108.4 N*m). The whole torque budget would be wrong at that sampling.
  2. **The warm-up is cut.** The robot starts from the stand keyframe at whatever the stick says,
     so the first seconds are a launch transient, not "running torque at v". Only the settled
     window is pooled, and a rung that falls inside it is reported as FELL rather than averaged.
  3. **The speed on the axis is the ACHIEVED speed, not the command.** This policy under-delivers
     at the top of the stick (2.50 commanded -> ~1.9 m/s), and a torque-vs-speed curve plotted
     against a command the robot did not obey is a curve about nothing.

Every constant -- peak torque (forcerange), continuous rating (the thermal node's tau_cont), Kt,
bus voltage, the 12 ms actuation delay -- comes out of the BUNDLE, so this cannot drift from what
the policy was trained against.

Outputs, in RLframework/results/:
  torque_vs_speed_<tag>.{png,pdf}   six panels, one per actuator, peak + RMS at each speed
  torque_vs_speed_<tag>.json        the numbers
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

TOOLS = Path(__file__).resolve().parent
PKG = TOOLS.parents[0]
ROOT = TOOLS.parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from play_joystick import Sim                                  # noqa: E402

OUT = PKG / "results"

# PEAK IS RED, RMS IS BLUE -- the measure, not the leg, carries the hue here (the other torque
# figures colour by leg; this one compares two measures inside every panel, so leg identity moves
# to the panel title). The pair passes the CVD check at dE 23.8 (protan) / 31.6 normal, and the
# two bars are always adjacent in a fixed order, so the pair is also readable by position alone.
C_PEAK, C_RMS = "#d03b3b", "#2a78d6"
# the limit lines are therefore NOT red: red now means "peak bar", and a red rule above a red bar
# would read as the same quantity. Peak limit = dark rule, continuous rating = gold, different
# dash patterns, both named in the panel title.
INK, INK2, GRID, LIM, CONT = "#0b0b0b", "#52514e", "#e1e0d9", "#3d3d3a", "#b08900"


class SubstepTap:
    """Wraps mujoco.mj_step so every 1 ms physics substep appends (tau, joint speed).

    play_joystick.Sim calls `mujoco.mj_step` by attribute, so replacing the module attribute
    reaches it -- the control law itself is untouched.
    """

    def __init__(self, sim):
        self.sim, self.on = sim, False
        self._real = mujoco.mj_step
        nu, dadr = sim.model.nu, sim.act_d
        self.tau, self.w, self.marks = [], [], []

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


def stats(tau, peak_lim, cont_lim):
    """One actuator's signed torque over the settled window."""
    a = np.abs(tau)
    rms = float(np.sqrt(np.mean(tau ** 2)))
    return dict(rms=rms, peak=float(a.max()), p95=float(np.percentile(a, 95)),
                mean_abs=float(a.mean()),
                peak_pct=float(a.max() / peak_lim * 100.0),
                rms_pct_cont=float(rms / cont_lim * 100.0),
                duty_95=float(np.mean(a >= 0.95 * peak_lim) * 100.0))


def run_rung(sim, tap, frac, ticks, warm_ticks):
    """Hold one stick position; return the settled 1 kHz trace, the speed and whether it survived."""
    sim.reset()
    sim.set_stick(frac)
    sim.set_run(frac > 0.0)                 # on a RUN/STOP bundle the 0% rung IS the stop command
    sim.ctrl.set_speed(sim.v_cmd, immediate=True)
    tap.clear()
    tap.on = True
    vs, ys, alive = [], [], True
    for t in range(ticks):
        sim.control_tick()
        tap.marks.append(len(tap.tau))       # substep index at the END of this control tick
        if sim.fallen():
            alive = False
            break
        vs.append(sim.speed())
        ys.append(float(sim.data.qpos[sim.i_y]))
    tap.on = False

    n_ticks = len(tap.marks)
    i0 = tap.marks[warm_ticks - 1] if n_ticks >= warm_ticks else 0
    i1 = tap.marks[-1] if tap.marks else 0
    tau = np.array(tap.tau[i0:i1])
    w = np.array(tap.w[i0:i1])
    v = float(np.mean(vs[warm_ticks:])) if len(vs) > warm_ticks else float("nan")
    return dict(tau=tau, w=w, v=v, alive=alive, live_s=n_ticks * sim.control_dt,
                y_absmax=float(np.max(np.abs(ys))) if ys else float("nan"),
                yaw_end=sim.heading_deg())


def load_reference(path, names, peak_lim, cont):
    """A rung from an ARCHIVED torque run, for scale: THE RUNNER's 100 m dash.

    `sprint_runner/tools/torque_dash.py` measured sprint_m3_mit_s0 under the same protocol --
    1 kHz per-substep sampling, greedy, pushes off -- and on the same drives (its forcerange is
    the identical [61.2, 144.5, ...]), so the bars are directly comparable. What is NOT the same
    is the policy, the plant revision and the task: that is a 3.06 m/s sprint on the m3 runner,
    not the flat-foot joystick walker, which is exactly why it is drawn set apart.

    The numbers are read from the archived JSON rather than typed in, so they cannot drift from
    the run that produced them. `pooled` = the whole dash; `pooled_steady` excludes the launch
    and the post-line stop.
    """
    d = json.loads(Path(path).read_text())
    src = d["pooled"]
    ref_peak = np.asarray(d["peak_limit_nm"], float)
    if not np.allclose(ref_peak, peak_lim, rtol=0.02):
        print(f"  [ref] WARNING: {Path(path).name} was measured on a different forcerange "
              f"{ref_peak.tolist()} vs {peak_lim.tolist()} -- the bars are NOT comparable")
    missing = [n for n in names if n not in src]
    if missing:
        raise KeyError(f"{path} has no torque row for {missing}")
    tq = {n: dict(src[n], p95=src[n].get("p95", float("nan")),
                  peak_pct=src[n]["peak"] / peak_lim[i] * 100.0,
                  rms_pct_cont=src[n]["rms"] / cont[i] * 100.0)
          for i, n in enumerate(names)}
    run = Path(d["run"]).name
    return dict(stick=float("nan"), cmd=float("nan"), v=float(d["vx_mean"]), tau=True, torque=tq,
                alive=True, reference=True, source=str(path), ref_run=run,
                ref_label=f"{run} - 100 m dash, {d['episodes']} episodes (different policy)",
                live_s=float(d["dash_s_mean"]), samples=None)


def table(names, rungs, peak_lim, cont):
    hdr = (f"  {'stick':>6} {'cmd':>6} {'v':>6}  {'actuator':<11} {'RMS':>7} {'peak':>7} "
           f"{'P95':>7} {'%peak lim':>10} {'RMS/cont':>9} {'sat%':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rungs:
        if not r.get("torque"):
            print(f"  {r['stick']*100:>5.0f}% {r['cmd']:>6.2f} {'--':>6}  FELL at "
                  f"{r['live_s']:.1f} s -- no settled window")
            continue
        for i, nm in enumerate(names):
            s = r["torque"][nm]
            if i:
                head = " " * 21
            elif r.get("reference"):
                head = f"  {'ref':>6} {'--':>6} {r['v']:>6.2f}"
            else:
                head = f"  {r['stick']*100:>5.0f}% {r['cmd']:>6.2f} {r['v']:>6.2f}"
            print(f"{head}  {nm:<11} {s['rms']:7.1f} {s['peak']:7.1f} {s['p95']:7.1f} "
                  f"{s['peak_pct']:9.0f}% {s['rms_pct_cont']:8.0f}% {s['duty_95']:5.1f}%")
        print()
    print("  torque in N*m, sampled at 1 kHz; 'sat%' = share of samples at >=95% of the peak limit")
    print(f"  peak limit = the model's forcerange {np.round(peak_lim, 1).tolist()}")
    print(f"  continuous = the bundle's thermal tau_cont {np.round(cont, 1).tolist()}")


def fig_bars(names, rungs, peak_lim, cont, tag, title):
    """Six small multiples -- one per actuator -- of RMS and peak against speed.

    One panel per motor rather than one crowded 6x5x2 group: the question is per-motor headroom,
    and each panel can then carry its own two limit lines at the motor's own rating.
    """
    live = [r for r in rungs if r.get("torque")]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.6))
    # the reference rung is a DIFFERENT policy on the same drives, so it gets a gap and a shaded
    # band rather than sitting flush in the ladder -- it is a comparison, not a sixth stick rung.
    ref_i = [j for j, r in enumerate(live) if r.get("reference")]
    x = np.array([j + (0.45 if ref_i and j >= ref_i[0] else 0.0) for j in range(len(live))])
    labels = [f"{r['v']:.2f}" for r in live]
    order = ["hip_roll_L", "cam_L", "thigh_L", "hip_roll_R", "cam_R", "thigh_R"]

    for k, nm in enumerate(order):
        ax = axes[k // 3, k % 3]
        i = names.index(nm)
        pk = np.array([r["torque"][nm]["peak"] for r in live])
        rms = np.array([r["torque"][nm]["rms"] for r in live])

        if ref_i:
            ax.axvspan(x[ref_i[0]] - 0.47, x[-1] + 0.47, color=GRID, alpha=0.55, lw=0, zorder=0)

        # peak left, red; RMS right, blue.
        ax.bar(x - 0.185, pk, 0.33, color=C_PEAK, edgecolor="white", lw=1.4, zorder=2)
        ax.bar(x + 0.185, rms, 0.33, color=C_RMS, edgecolor="white", lw=1.4, zorder=2)
        for j, xi in enumerate(x):
            # a peak sitting ON the limit is the drive CLIPPING, not a measurement: the control
            # law caps every substep at min(forcerange, back-EMF envelope), so the bar cannot
            # exceed the rule and the honest reading is "demand >= this". Mark it.
            clipped = pk[j] >= 0.99 * peak_lim[i]
            ax.text(xi - 0.185, pk[j] + 0.02 * peak_lim[i],
                    f"{pk[j]:.0f}*" if clipped else f"{pk[j]:.0f}", ha="center",
                    fontsize=7.5, color=C_PEAK if clipped else INK,
                    fontweight="bold" if clipped else "normal")
            ax.text(xi + 0.185, rms[j] + 0.02 * peak_lim[i], f"{rms[j]:.0f}", ha="center",
                    fontsize=7.5, color=INK)

        ax.axhline(peak_lim[i], color=LIM, lw=1.4, ls="--", zorder=3)
        ax.axhline(cont[i], color=CONT, lw=1.4, ls=":", zorder=3)
        ax.set_title(f"{nm}   peak limit {peak_lim[i]:.0f}   continuous {cont[i]:.0f} N*m",
                     fontsize=9.5, color=INK, loc="left", pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5, color=INK)
        ax.set_xlim(-0.6, x[-1] + 0.6)
        ax.set_ylim(0, peak_lim[i] * 1.20)
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK2)
        if k % 3 == 0:
            ax.set_ylabel("torque  [N*m]", fontsize=9, color=INK2)
        if k // 3 == 1:
            ax.set_xlabel("achieved forward speed  [m/s]", fontsize=9, color=INK2)

    h = [plt.Rectangle((0, 0), 1, 1, facecolor=C_PEAK),
         plt.Rectangle((0, 0), 1, 1, facecolor=C_RMS),
         plt.Line2D([], [], color=LIM, ls="--", lw=1.4),
         plt.Line2D([], [], color=CONT, ls=":", lw=1.4)]
    lab = ["peak |tau| (worst 1 kHz sample)", "RMS torque",
           "peak limit (forcerange)", "continuous rating (thermal)"]
    if ref_i:
        h.append(plt.Rectangle((0, 0), 1, 1, facecolor=GRID, alpha=0.55))
        lab.append(live[ref_i[0]]["ref_label"])
    fig.legend(h, lab, fontsize=8.5, frameon=False, ncol=len(h), loc="lower center",
               bbox_to_anchor=(0.5, 0.022))
    fig.text(0.5, -0.004, "* the drive clipped at the limit on this rung -- demand is at least "
                          "the bar height, not equal to it", fontsize=7.5, color=INK2, ha="center")
    fig.suptitle(title, fontsize=11, color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"torque_vs_speed_{tag}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote RLframework/results/torque_vs_speed_{tag}.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="controller/deploy/bundles/dash_joy3_lr_s2_180M.npz")
    ap.add_argument("--model", default=None)
    ap.add_argument("--sticks", default="0.2,0.4,0.6,0.8,1.0",
                    help="stick positions as fractions of v_max")
    ap.add_argument("--seconds", type=float, default=16.0, help="held per rung")
    ap.add_argument("--warm", type=float, default=4.0, help="launch transient discarded per rung")
    ap.add_argument("--lock", choices=("none", "yaw", "rail"), default="none",
                    help="DIAGNOSTIC: the real machine has no such constraint")
    ap.add_argument("--reference", default="sprint_runner/results/torque_dash_s0.json",
                    help="archived torque_dash JSON drawn as a set-apart comparison rung; "
                         "'none' to omit it")
    ap.add_argument("--delay-ms", type=float, default=None,
                    help="default = the bundle's drive_delay_ms (what the policy trained with)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    bundle = Path(args.bundle)
    if not bundle.is_absolute():
        bundle = ROOT / bundle
    sim = Sim(bundle, args.model)
    if args.delay_ms is not None:
        sim.delay_ms = float(args.delay_ms)
    sim.lock = args.lock
    meta = sim.bundle.meta
    tag = args.tag or bundle.stem

    names = list(meta["actuator_names"])
    peak_lim = np.asarray(sim.tau_peak, float)
    cont = np.asarray(meta["thermal"]["tau_cont"], float)
    ticks = int(round(args.seconds / sim.control_dt))
    warm_ticks = int(round(args.warm / sim.control_dt))
    fracs = [float(s) for s in args.sticks.split(",")]

    print(f"\n[torque] {bundle.name}  run {meta.get('run')}  v_max {sim.v_max:.2f} m/s  "
          f"{1 / sim.control_dt:.0f} Hz control / {sim.substeps} substeps, "
          f"{sim.delay_ms:.0f} ms actuation delay  lock={sim.lock}")
    print(f"[torque] {args.seconds:.0f} s per rung, first {args.warm:.0f} s discarded; "
          f"torque sampled every substep ({1000 * sim.model.opt.timestep:.0f} ms)\n")

    tap = SubstepTap(sim)
    rungs = []
    try:
        for f in fracs:
            r = run_rung(sim, tap, f, ticks, warm_ticks)
            cmd = sim.v_cmd
            if not r["alive"] or len(r["tau"]) < 100:
                print(f"  stick {f*100:3.0f}%  cmd {cmd:.2f}  FELL at {r['live_s']:.1f} s")
                rungs.append(dict(stick=f, cmd=cmd, v=float("nan"), tau=None, alive=False,
                                  live_s=r["live_s"]))
                continue
            tq = {nm: stats(r["tau"][:, i], peak_lim[i], cont[i]) for i, nm in enumerate(names)}
            print(f"  stick {f*100:3.0f}%  cmd {cmd:.2f}  achieved {r['v']:.2f} m/s  "
                  f"{len(r['tau'])} samples  |y| max {r['y_absmax']:.2f} m  "
                  f"yaw end {r['yaw_end']:+.0f} deg")
            rungs.append(dict(stick=f, cmd=cmd, v=r["v"], tau=r["tau"], torque=tq, alive=True,
                              live_s=r["live_s"], y_absmax=r["y_absmax"], yaw_end=r["yaw_end"],
                              peak_mech_w=float(np.abs(r["tau"] * r["w"]).sum(axis=1).max()),
                              samples=int(len(r["tau"]))))
    finally:
        tap.restore()

    if args.reference and args.reference.lower() != "none":
        ref = Path(args.reference)
        if not ref.is_absolute():
            ref = ROOT / ref
        r = load_reference(ref, names, peak_lim, cont)
        print(f"  reference  {r['ref_run']}  {r['v']:.2f} m/s over {r['live_s']:.1f} s "
              f"(from {ref.name})")
        rungs.append(r)

    print()
    table(names, rungs, peak_lim, cont)

    live = [r for r in rungs if r.get("torque")]
    own = [r for r in live if not r.get("reference")]
    if not own:
        print("\n[torque] nothing stayed upright -- no figure written")
        return 1
    ref_note = ""
    if len(live) > len(own):
        ref_note = (f"; shaded at right, {next(r['ref_run'] for r in live if r.get('reference'))}"
                    f"'s 100 m dash for scale")
    fig_bars(names, rungs, peak_lim, cont, tag,
             f"Per-motor torque demand vs speed - {meta.get('run')} "
             f"({meta.get('step', 0) / 1e6:.0f} M)\n"
             f"{len(own)} speeds held {args.seconds:.0f} s each, settled window only, "
             f"1 kHz sampling through the shipping control law{ref_note}")

    OUT.mkdir(exist_ok=True)
    js = dict(bundle=str(bundle), run=meta.get("run"), step=meta.get("step"),
              v_max=sim.v_max, seconds=args.seconds, warm_s=args.warm, lock=sim.lock,
              delay_ms=sim.delay_ms, sample_hz=1.0 / sim.model.opt.timestep,
              actuators=names, peak_limit_nm=peak_lim.tolist(), cont_rating_nm=cont.tolist(),
              rungs=[{k: v for k, v in r.items() if k != "tau"} for r in rungs])
    (OUT / f"torque_vs_speed_{tag}.json").write_text(json.dumps(js, indent=1))
    print(f"  wrote RLframework/results/torque_vs_speed_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
