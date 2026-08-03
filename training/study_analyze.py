"""Aggregate the ankle-spring study: does the spring help, must it be actuated, is there an optimum?

    python training/study_analyze.py --runs training/runs [--phase m3] [--at 400000000]

Reads every study_<phase>_<arm>_s<seed>/progress*.csv, aggregates the seed replicates, and writes:

  * a RANKING table (mean +- std across seeds, and the seed spread, because an arm that wins on one
    seed and falls over on the others has not won)
  * study_<phase>_curve.png -- the stiffness curve with seed error bars, and the rigid / free /
    active arms drawn as horizontal reference bands, which is the plot that actually answers all
    three questions at once: whether the passive curve ever beats the `rigid`/`free` nulls (is the
    spring useful), whether it beats the `active` band (must it be actuated), and whether it has an
    interior peak (is there an optimum) rather than a plateau or a monotone ramp
  * the exact phase-B command for the top arms

Compares arms at a COMMON step count (--at, default the smallest common maximum) -- comparing a run
at 400 M against one that only reached 250 M is the single easiest way to get a wrong answer here.
"""
import argparse
import glob
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_ab import STEPS, at_step, read_progress
from config import STUDY_ARMS

NAME_RE = re.compile(r"^study_(m\d)_(.+)_s(\d+)$")
# score = the survival metric the whole lineage is graded on; ep_rew_mean breaks ties
SCORE = "rollout/ep_len_mean"
RETURN = "rollout/ep_rew_mean"


def arm_k(arm):
    """The passive arms plot against stiffness; the nulls and the active arms have no k."""
    m = re.match(r"^k(\d+)(?:_(\d+))?$", arm)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}") if m.group(2) else float(m.group(1))


def collect(runs_dir, phase):
    out = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, "study_*"))):
        m = NAME_RE.match(os.path.basename(d))
        if not m or m.group(1) != phase:
            continue
        prog = read_progress(d)
        if prog is None or STEPS not in prog:
            print(f"  (skip {os.path.basename(d)}: no progress data)")
            continue
        out.setdefault(m.group(2), {})[int(m.group(3))] = prog
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="training/runs")
    ap.add_argument("--phase", default="m3")
    ap.add_argument("--at", type=float, default=None,
                    help="compare at this step count (default: the largest COMMON step count)")
    ap.add_argument("--top", type=int, default=3, help="how many arms to advance to phase B")
    args = ap.parse_args()

    data = collect(args.runs, args.phase)
    if not data:
        raise SystemExit(f"no study_{args.phase}_* runs with progress.csv under {args.runs}")

    # common step count: the smallest per-run maximum, so no arm is credited for extra training
    reached = [float(np.nanmax(p[STEPS])) for seeds in data.values() for p in seeds.values()]
    step = args.at if args.at is not None else min(reached)
    print(f"\ncomparing {len(data)} arms at {step/1e6:.0f} M steps "
          f"(runs reached {min(reached)/1e6:.0f}-{max(reached)/1e6:.0f} M)\n")

    rows = []
    for arm, seeds in data.items():
        sc = np.array([at_step(p, SCORE, step) for p in seeds.values()], float)
        rw = np.array([at_step(p, RETURN, step) for p in seeds.values()], float)
        if np.isnan(sc).all():
            continue
        rows.append(dict(arm=arm, k=arm_k(arm), n=len(sc),
                         mean=np.nanmean(sc), std=np.nanstd(sc),
                         lo=np.nanmin(sc), hi=np.nanmax(sc), ret=np.nanmean(rw)))
    rows.sort(key=lambda r: -r["mean"])

    w = max(len(r["arm"]) for r in rows)
    print(f"{'arm':{w}s} {'n':>2s} {'ep_len':>9s} {'+-std':>8s} {'worst':>8s} {'best':>8s} {'return':>9s}")
    print("-" * (w + 48))
    for r in rows:
        print(f"{r['arm']:{w}s} {r['n']:2d} {r['mean']:9.0f} {r['std']:8.0f} "
              f"{r['lo']:8.0f} {r['hi']:8.0f} {r['ret']:9.0f}")

    # ---- the three questions, answered against the nulls -------------------------------------
    by = {r["arm"]: r for r in rows}
    passive = sorted([r for r in rows if r["k"] is not None], key=lambda r: r["k"])
    print()
    best_p = max(passive, key=lambda r: r["mean"]) if passive else None
    for null in ("rigid", "free"):
        if null in by and best_p:
            d = best_p["mean"] - by[null]["mean"]
            # "clearly" = the gap exceeds the pooled seed spread; with 3 seeds this is a sanity
            # check, not a significance test, and it is labelled as one.
            pooled = np.hypot(best_p["std"], by[null]["std"])
            verdict = "CLEAR" if abs(d) > pooled else "within seed noise"
            print(f"  best passive (k={best_p['k']:g}) vs {null:5s}: {d:+8.0f} ep_len  [{verdict}]")
    for act in ("active", "active_k350"):
        if act in by and best_p:
            d = by[act]["mean"] - best_p["mean"]
            pooled = np.hypot(best_p["std"], by[act]["std"])
            verdict = "CLEAR" if abs(d) > pooled else "within seed noise"
            print(f"  {act:11s} vs best passive:      {d:+8.0f} ep_len  [{verdict}]")
    if len(passive) >= 3:
        i = int(np.argmax([r["mean"] for r in passive]))
        shape = ("INTERIOR PEAK -> an optimum exists" if 0 < i < len(passive) - 1 else
                 "monotone/edge -> no interior optimum in the swept range; extend the grid")
        print(f"  passive curve peaks at k={passive[i]['k']:g} ({shape})")

    # ---- the motor SPEC, for whichever arms actually had a motor ------------------------------
    # Half the point of the study: if an actuated ankle wins, what performance does it need? These
    # come from ankle/* in progress.csv (AnkleTelemetryCallback), peaks over the run.
    spec_arms = [a for a in ("active", "active_k350", "active_freeenergy") if a in data]
    if spec_arms:
        print("\nankle motor demand (peak over training — this is the SPEC if active wins):")
        print(f"  {'arm':13s} {'torque N*m':>11s} {'speed rad/s':>12s} {'power W':>9s} "
              f"{'>cont %':>8s} {'util':>6s}")
        for a in spec_arms:
            g = lambda col: np.nanmax([at_step(p, col, step)          # noqa: E731 - local shorthand
                                       for p in data[a].values()])
            print(f"  {a:13s} {g('ankle/ankle_motor_trq_peak'):11.1f} "
                  f"{g('ankle/ankle_motor_w_peak'):12.1f} "
                  f"{g('ankle/ankle_motor_power_peak'):9.1f} "
                  f"{100*g('ankle/ankle_motor_over_cont'):8.1f} "
                  f"{g('ankle/ankle_motor_util'):6.2f}")
        print("  (util ~1.0 means the torque-speed curve, not the policy, is the binding limit;\n"
              "   >cont % is time above the 55 N*m continuous rating — the thermal question)")

    # ---- and the SPRING demand, which decides whether a winning k is buildable ------------------
    if passive:
        print("\npassive spring demand (peak over training — can the part survive it?):")
        print(f"  {'k':>8s} {'torque N*m':>11s} {'energy J':>9s} {'defl rad':>9s}")
        for r in passive:
            seeds = data[r["arm"]]
            g = lambda col: np.nanmax([at_step(p, col, step)          # noqa: E731 - local shorthand
                                       for p in seeds.values()])
            print(f"  {r['k']:8g} {g('ankle/ankle_spring_trq_peak'):11.1f} "
                  f"{g('ankle/ankle_spring_energy_peak'):9.2f} "
                  f"{g('ankle/ankle_defl_peak'):9.3f}")
        print("  (the real spring today is k=28.65 with a 2.27 N*m preload — compare against that\n"
              "   before treating a high-k winner as a drop-in change)")

    # ---- plot ---------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if passive:
        ks = [r["k"] for r in passive]
        ms = [r["mean"] for r in passive]
        ax.errorbar(ks, ms, yerr=[r["std"] for r in passive], marker="o", capsize=4,
                    lw=2, color="#1f77b4", label="passive spring")
        ax.fill_between(ks, [r["lo"] for r in passive], [r["hi"] for r in passive],
                        color="#1f77b4", alpha=0.15, label="seed min-max")
    for arm, col in (("rigid", "#7f7f7f"), ("free", "#8c564b"),
                     ("active", "#d62728"), ("active_k350", "#2ca02c")):
        if arm in by:
            r = by[arm]
            ax.axhline(r["mean"], color=col, ls="--", lw=1.6, label=f"{arm} (no k)")
            ax.axhspan(r["mean"] - r["std"], r["mean"] + r["std"], color=col, alpha=0.10)
    ax.set_xscale("log")
    ax.set_xlabel("passive ankle stiffness k  (N*m/rad, log scale)")
    ax.set_ylabel(f"{SCORE} at {step/1e6:.0f} M steps")
    ax.set_title(f"Ankle-spring study — {args.phase}, {len(rows)} arms x seeds "
                 f"(damping held at zeta=0.7 throughout)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    out = os.path.join(args.runs, f"study_{args.phase}_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")

    if args.phase == "m3":
        top = " ".join(r["arm"] for r in rows[:args.top])
        print(f"\nphase B (m6 confirmation) for the top {args.top}:\n"
              f"  PHASE=m6 ARMS=\"{top}\" bash training/slurm/study_ankle_launch.sh")
    unknown = set(data) - set(STUDY_ARMS)
    if unknown:
        print(f"\nnote: run dirs not in STUDY_ARMS (ignored in the verdicts): {sorted(unknown)}")


if __name__ == "__main__":
    main()
