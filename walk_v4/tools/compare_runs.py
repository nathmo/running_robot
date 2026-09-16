"""Compare runs on the greedy ladder, AT MATCHED STEPS. The only honest way to rank them.

    python walk_v4/tools/compare_runs.py --runs walk_v4/runs/v3_stage1*

Two rules this tool exists to enforce, both of which have produced wrong conclusions in this project
within the last hour:

  1. **Never compare at unequal steps.** A run at 65 M against a run at 29 M is not a comparison of
     recipes. Every row here is one step count, and a run that has not reached it shows "-".
  2. **Never rank by training return.** Return is not comparable across runs whose rewards differ --
     and `shape_scale` makes the reward differ WITHIN a run, because raising the gait-quality
     penalties lowers the return by construction. A run whose return fell 1197 -> -18 was at the
     same time going from 0% to 60% upright. Rank on the greedy ladder or not at all.

The columns are what the contract asks for: command error in m/s over the settled tail, the fraction
upright from a settled start, and the fraction upright from a dirty bring-up.
"""
import argparse
import csv
import os
from pathlib import Path


def load(run):
    f = Path(run) / "eval.csv"
    if not f.exists():
        return {}
    out = {}
    for row in csv.DictReader(open(f)):
        try:
            out[int(float(row["step"]))] = row
        except (KeyError, TypeError, ValueError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--last", type=int, default=0, help="show only the last N step rows")
    args = ap.parse_args()

    runs = [r for r in sorted(args.runs) if Path(r).is_dir()]
    data = {Path(r).name: load(r) for r in runs}
    data = {k: v for k, v in data.items() if v}
    if not data:
        print("no eval.csv in any of those runs")
        return 1
    names = list(data)
    steps = sorted(set().union(*(set(d) for d in data.values())))
    if args.last:
        steps = steps[-args.last:]

    w = 22
    print(f"{'step':>12} " + " ".join(f"{n[-w:]:>{w}}" for n in names))
    print(f"{'':>12} " + " ".join(f"{'err / upright / dirty':>{w}}" for _ in names))
    print("-" * (13 + (w + 1) * len(names)))
    for s in steps:
        cells = []
        for n in names:
            row = data[n].get(s)
            if row is None:
                cells.append(f"{'-':>{w}}")
                continue
            g = lambda k: float(row.get(k) or "nan")
            cells.append(f"{g('track_err'):>8.2f} /{g('alive_frac') * 100:>4.0f}% /{g('bringup_alive') * 100:>4.0f}%")
        print(f"{s:>12,} " + " ".join(cells))

    print("\nerr = m/s of command error over the settled tail, averaged across the full 0-100% ladder;"
          "\nupright = settled start, dirty = every episode dropped or released misaligned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
