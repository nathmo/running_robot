"""Rank every checkpoint in a run on the greedy ladder, at an honest number of envs.

`best.msgpack` is chosen during training by `keeper_score`, off an eval that runs `--eval-envs`
environments across a five-rung command ladder. At the old default of 16 that is about three envs
per rung, so "upright" can only read 0 / 33 / 67 / 100% and the keeper is picking on noise. Raising
the default fixes new runs; this fixes the ones already on disk.

    python walk_v3/tools/rank_checkpoints.py --run walk_v3/runs/v3_s3g_b --n-per 12

It re-scores every `ckpt_*.msgpack` on the same ladder `verify.py` uses, with the same keeper
weighting (tracking error, falls, heading), and prints them worst to best so the winner is the last
line. `--write-best` then copies the winner over `best.msgpack` -- which is what `export.py` and the
next stage's warm start both read.

The env and the jitted rollout are built ONCE and the weights passed in as arguments, so the cost is
one compile for the whole sweep plus one rollout per checkpoint -- not one compile each, which at
2-4 minutes a trace is the difference between three minutes and ninety.
"""
import argparse
import shutil
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np

from evaluate import load_run
from verify import ladder_eval


def score(rows, fall_w, head_w):
    """The keeper's question, asked on a bigger sample: error, then falls, then heading."""
    err = float(np.mean([r["err"] for r in rows]))
    fall = 1.0 - float(np.mean([r["upright"] for r in rows]))
    head = float(np.nanmean([r["yaw_deg"] for r in rows])) if rows else float("nan")
    head = 0.0 if np.isnan(head) else np.radians(head)
    return -(err + fall_w * fall + head_w * head), err, fall, np.degrees(head)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-per", type=int, default=12, help="envs per ladder rung")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--last", type=int, default=0, help="only the last N checkpoints")
    ap.add_argument("--fall-weight", type=float, default=2.0)
    ap.add_argument("--heading-weight", type=float, default=1.0)
    ap.add_argument("--write-best", action="store_true",
                    help="copy the winner over best.msgpack (and its sidecar)")
    args = ap.parse_args()

    run = Path(args.run)
    cks = sorted(run.glob("ckpt_*.msgpack"), key=lambda q: int(q.stem.split("_")[1]))
    if not cks:
        print(f"no ckpt_*.msgpack in {run}")
        return 1
    if args.last:
        cks = cks[-args.last:]

    cfg, _env, agent = load_run(run, cks[0])
    n_envs = int(len(cfg.eval_ladder) * args.n_per)

    # one compile, then one rollout per checkpoint: every call here differs ONLY in the policy
    # parameters, which is exactly the invariant `cache` requires
    cache = {}
    out = []
    for ck in cks:
        agent.load(ck, warm_start=False)
        rows = ladder_eval(cfg, agent, n_envs, args.seconds, dr=False, bringup=False, seed=11,
                           cache=cache)
        s, err, fall, head = score(rows, args.fall_weight, args.heading_weight)
        out.append((s, ck, err, fall, head, rows))

    out.sort(key=lambda t: t[0])
    print(f"\n[rank] {run.name}  {len(out)} checkpoints  {args.n_per} envs/rung  "
          f"{args.seconds:.0f}s  (worst first)")
    print(f"{'checkpoint':>22} {'score':>8} {'err':>7} {'upright':>8} {'heading':>8}   per-rung upright")
    for s, ck, err, fall, head, rows in out:
        per = " ".join(f"{r['upright'] * 100:3.0f}" for r in rows)
        print(f"{ck.stem:>22} {s:>8.3f} {err:>7.2f} {(1 - fall) * 100:>7.0f}% {head:>7.1f}d   {per}")

    best = out[-1][1]
    print(f"\nbest on this ladder: {best.name}")
    cur = run / "best.msgpack"
    if cur.exists():
        agent.load(cur, warm_start=False)
        rows = ladder_eval(cfg, agent, n_envs, args.seconds, dr=False, bringup=False, seed=11,
                           cache=cache)
        s, err, fall, head = score(rows, args.fall_weight, args.heading_weight)
        print(f"training keeper:     best.msgpack   score {s:.3f}  err {err:.2f}  "
              f"upright {(1 - fall) * 100:.0f}%")
    if args.write_best:
        shutil.copy2(best, run / "best.msgpack")
        side = best.with_suffix(".json")
        if side.exists():
            shutil.copy2(side, run / "best.json")
        print(f"wrote {best.name} -> best.msgpack")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
