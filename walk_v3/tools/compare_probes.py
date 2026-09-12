"""Compare training arms side by side: the numbers that decide, not the training curve.

    python walk_v3/tools/compare_probes.py --runs walk_v3/runs/probe_*

Prints, per run, the LAST ladder eval (command error per stick position, upright fraction, heading)
next to what the curricula actually reached. Sorted by the keeper's own score, so the arm that wins is
the one at the top -- and the columns say whether it won by tracking, by surviving, or by cheating on
one command and ignoring the rest.

Training return is deliberately NOT the ranking key. Arms with different rewards (a bring-up grace, a
different command band) earn different returns for the same behaviour, so comparing them by return
compares the reward functions, not the policies.
"""
import argparse
import csv
import json
from pathlib import Path


def read_evals(run):
    f = run / "eval.csv"
    if not f.exists():
        return []
    with open(f) as fh:
        return list(csv.DictReader(fh))


def curricula(run):
    js = sorted(run.glob("ckpt_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not js:
        return {}
    return json.loads(js[-1].read_text()).get("env_params", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    rows = []
    for r in sorted(Path(p) for p in args.runs):
        if not r.is_dir():
            continue
        ev = read_evals(r)
        if not ev:
            rows.append((r.name, None, None, {}))
            continue
        last = ev[-1]
        best = None
        bj = r / "best_eval.json"
        if bj.exists():
            best = json.loads(bj.read_text())
        rows.append((r.name, last, best, curricula(r)))

    def key(row):
        _, last, _, _ = row
        if last is None:
            return 1e9
        try:
            return float(last.get("track_err", 1e9)) + 2.0 * (1.0 - float(last.get("alive_frac", 0)))
        except (TypeError, ValueError):
            return 1e9

    rows.sort(key=key)
    print(f"{'run':<22} {'step':>11} {'cmd err':>8} {'upright':>8} {'bring-up':>9} "
          f"{'head':>7} {'dr':>6} {'cmd_lo':>7} {'best@':>11}")
    print("-" * 100)
    for name, last, best, cur in rows:
        if last is None:
            print(f"{name:<22} {'(no eval yet)':>11}")
            continue
        g = lambda k, d=float('nan'): float(last.get(k, d) or d)
        print(f"{name:<22} {int(float(last['step'])):>11,} {g('track_err'):>8.2f} "
              f"{g('alive_frac') * 100:>7.0f}% {g('bringup_alive') * 100:>8.0f}% "
              f"{g('heading_err') * 57.2958:>6.1f}d {float(cur.get('dr_scale', 0)):>6.3f} "
              f"{float(cur.get('cmd_lo', 1)):>7.2f} "
              f"{int(best['step']):>11,}" if best else "")
    print("\ncmd err is m/s of command error over the settled tail, averaged across the ladder;"
          "\nupright = the settled-start block, bring-up = the dropped / held-misaligned block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
