"""Turn eval_envelope JSON into the margin table a control engineer expects.

  python tools/summarize_envelope.py results/envelope_*.json

Each swept axis is named in classical terms. The honest caveats are printed with the table, not
buried: a margin is only a BALANCE margin if the base DOFs that let the robot fall are actually
free, so the milestone (base_lock) is reported for every run and mixed-milestone comparisons are
flagged rather than silently tabulated.
"""
import argparse
import json
import sys
from pathlib import Path

# axis -> (classical name, does a bigger value mean a harder plant?)
CLASSICAL = {
    "torque":   "actuator-authority margin (loop gain)",
    "kp":       "gain margin (servo stiffness)",
    "delay":    "delay margin (phase)",
    "drive_bw": "bandwidth margin",
    "mass":     "parametric margin - mass",
    "inertia":  "parametric margin - inertia",
    "com":      "parametric margin - CoM offset",
    "friction": "structured uncertainty - contact",
    "slope":    "disturbance - terrain slope",
    "ankle_k":  "component tolerance - ankle spring k",
    "ankle_c":  "component tolerance - ankle damping",
}
NOMINAL_FALLBACK = {"mass": 1.0, "inertia": 1.0, "ankle_k": 1.0, "ankle_c": 1.0, "kp": 1.0,
                    "torque": 1.0, "friction": 1.0, "slope": 0.0, "com": 0.0, "delay": 4.0}


def band(rows, nom_v):
    """Contiguous run of operating points containing nominal that holds >=50% of nominal survival."""
    i = min(range(len(rows)), key=lambda k: abs(rows[k]["value"] - nom_v))
    s_nom = rows[i]["survival"]
    if s_nom <= 0.0:
        return None, s_nom, i
    thr = 0.5 * s_nom
    lo = hi = i
    while lo > 0 and rows[lo - 1]["survival"] >= thr:
        lo -= 1
    while hi < len(rows) - 1 and rows[hi + 1]["survival"] >= thr:
        hi += 1
    return (rows[lo]["value"], rows[hi]["value"]), s_nom, i


def pct(v, nom):
    if nom in (0.0, None):
        return f"{v:g}"
    return f"{(v / nom - 1.0) * 100:+.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    args = ap.parse_args()

    docs = []
    for p in args.json:
        for f in sorted(Path().glob(p)) if any(c in p for c in "*?") else [Path(p)]:
            docs.append((f, json.loads(f.read_text())))

    for f, d in docs:
        print(f"\n{'=' * 78}")
        print(f"{d['run']}   ({d['pkg']}, {d['episodes']} eps x {d['seconds']:.0f}s, greedy, "
              f"paired seeds)")
        print(f"  objective={'command' if d['command_mode'] else 'sprint'}   "
              f"impedance={'yes' if d.get('impedance') else 'no'}")
        print(f"{'=' * 78}")
        print(f"  {'margin':<38} {'band':<22} {'as % of nominal'}")
        print(f"  {'-' * 38} {'-' * 22} {'-' * 18}")
        for ax, blk in d["axes"].items():
            rows = blk["rows"]
            nom_v = blk.get("nominal", NOMINAL_FALLBACK.get(ax))
            bd, s_nom, i = band(rows, nom_v)
            name = CLASSICAL.get(ax, ax)
            if bd is None:
                print(f"  {name:<38} {'NONE (0% at nominal)':<22}")
                continue
            lo, hi = bd
            rng = f"{lo:g} .. {hi:g}"
            rel = f"{pct(lo, nom_v)} .. {pct(hi, nom_v)}" if nom_v else f"{lo:g} .. {hi:g}"
            print(f"  {name:<38} {rng:<22} {rel}")
        # The nominal point is the SAME plant on every axis, so its rows must agree. They disagree
        # only when an axis's sweep does not actually contain the nominal value (imp_m2_long's
        # delay nominal is 1 step but the sweep starts at 2). Take only axes that really bracket
        # nominal, and report the median -- max() would quietly report the best off-nominal row.
        exact = []
        for a, b in d["axes"].items():
            nom = b.get("nominal", NOMINAL_FALLBACK.get(a))
            if nom is None:
                continue
            r = min(b["rows"], key=lambda k: abs(k["value"] - nom))
            if abs(r["value"] - nom) <= 1e-9 + 0.01 * max(abs(nom), 1.0):
                exact.append(r)
            else:
                print(f"  note: '{a}' sweep does not include nominal {nom:g} "
                      f"(closest {r['value']:g}) - excluded from the nominal summary")
        if exact:
            med = lambda k: sorted(r[k] for r in exact)[len(exact) // 2]
            print(f"\n  NOMINAL PLANT: survival {med('survival'):.0%}   "
                  f"mean vx {med('v_mean'):.2f} m/s   distance {med('dist_mean'):.1f} m"
                  f"   (median over {len(exact)} axes)")

    if len(docs) > 1:
        print(f"\n{'=' * 78}")
        print("CAUTION: these runs are NOT a controlled comparison unless base_lock matches.")
        print("A policy whose roll and pitch are railed cannot fall over, so its survival number")
        print("measures vertical collapse, not balance. Check base_lock before tabulating them")
        print("side by side as a robustness/performance trade.")
        print(f"{'=' * 78}")


if __name__ == "__main__":
    sys.exit(main())
