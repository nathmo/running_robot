"""Run the CLASSICAL controller across the foot-shape plants, paired seeds, one table.

The question is narrow and the answer is a number, not a video: does giving this robot a foot with
a FOOTPRINT buy the hand-built controller anything, and if so, is it the lateral line contact
(blade) or the sagittal centre-of-pressure the plate adds on top?

    python training/foot_ab.py                       # m3 (the wall) + m2 (tracking), 6 seeds
    python training/foot_ab.py --milestones m3
    python training/foot_ab.py --plants dash01.xml dash01_flat_bal.xml --episodes 10

REPORTING STANDARD, and it is not optional here. Outcomes on this robot are BIMODAL -- an unpaired
5-episode sample once invented a 27x effect that does not exist ([[walk-fwd-lineage]]). So every
plant is run on the SAME seed list and the table carries survivor counts and the median, not just
a mean. A difference smaller than the seed-to-seed spread is not a difference.

WHICH STANCE. Both, and they answer different questions:
  dash01.xml / _blade / _flat            the SHIPPED stance, footprint ~65-82 mm ahead of the CoM.
                                         Isolates the foot: everything else is the plant the
                                         ladder trains on.
  dash01_sphere_bal / _blade_bal /       the stance re-solved so the footprint centre sits UNDER
  dash01_flat_bal                        the CoM (model/make_foot_variants.py). This is the pairing
                                         the classical controller wants, since it has no way to
                                         recover the 5.7 deg backward lean the shipped stance hands
                                         it at t=0. All three sit at the same ride height, so a
                                         foot A/B here is not secretly a ride-height A/B.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from scripted_walk import GAINS, make_env, rollout                      # noqa: E402

PLANTS = ["model/dash01.xml", "model/dash01_blade.xml", "model/dash01_flat.xml",
          "model/dash01_sphere_bal.xml", "model/dash01_blade_bal.xml",
          "model/dash01_flat_bal.xml"]
# The gains the shipped controller is actually tuned to on the rebuilt plant. Deliberately NOT
# retuned per foot: the first question is whether a foot helps the controller we HAVE. A per-foot
# retune (scripted_walk --tune) is the follow-up, and only worth paying for on a foot that already
# shows something.
GAINS_FILE = "scripted_gains_m2_v2plant.json"


def run(plant, milestone, seeds, v_des, seconds, schedule, gains):
    env = make_env(milestone, dr=0.0, pushes=False, episode_s=seconds, model_path=plant)
    out = []
    for s in seeds:
        out.append(rollout(env, gains, s, v_des=v_des, est="odom", schedule=schedule))
    return out


# z_off is the controller's STATIC LEG-EXTENSION BIAS, in metres of toe offset from the nominal
# stance. It is referenced to the stance, so re-solving the stance moves its operating point, and
# running every plant at one fixed value is not a fair comparison -- it is a comparison at a bias
# that fits one plant. Measured at m2 (where the robot cannot topple, so this isolates the effect),
# median survival over 3 seeds:
#
#     z_off               -0.070  -0.050  -0.030  -0.010   0.000  +0.020
#     dash01 (shipped)     12.00   12.00   12.00   12.00    3.45    1.36
#     sphere_bal            0.78    0.85    0.96    1.03    1.06   12.00
#     blade_bal             0.89    1.02    1.08    1.13    1.25   12.00
#     flat_bal              1.00    1.10    1.15    1.53   12.00   12.00
#
# The whole usable window SHIFTS by ~+0.03 m, and the deaths are workspace-kill (the balanced
# stance has already spent that much of the leg's down-reach, so the controller's extra extension
# puts the toe outside the measured reachable box). Nothing to do with the foot. So each plant is
# refit on its own, on SEPARATE seeds from the ones it is then scored on.
Z_OFF_GRID = [-0.09, -0.075, -0.06, -0.045, -0.03, -0.015, 0.0, 0.015, 0.03, 0.045, 0.06]
FIT_SEEDS = [100, 101, 102]
# THE FIT TASK MUST MATCH THE SCORING TASK, and getting this wrong twice is what makes it worth
# spelling out. Fitting on forward walking alone picked z_off +0.015 for the balanced plate --
# optimal forwards, one grid step into a workspace-kill BACKWARDS. It then scored 0/6 on the mission
# schedule, dying at 23.05 s on all six seeds (3 s into the backward segment) and reading exactly
# like "the plate breaks backward walking". It does not: at +0.000 the same plant does 6/6 x 35 s
# and 1.82 m, level with the sphere's 1.79 m.
#
# The first repair -- forward, then straight into backward -- was no better, just wrong in the other
# direction: it slams through the reversal with no stand phase, which the mission never does, and it
# re-picked the same +0.015 with a fit survival of 10.26 s where every other plant reached 12.00.
#
# So the fit schedule is now a COMPRESSED REPLICA of the scoring schedule: same five phases
# (stand / forward / stand / backward / stand) at a third of the duration. Chosen for STRUCTURE, not
# because it makes any particular plant pass -- the failure mode being avoided is fitting a gain on
# a task the plant is then graded on differently, and the only honest fix is to make the two agree.
FIT_SCHEDULE = [(0.0, 0.0), (1.7, 1.0), (5.0, 0.0), (6.7, -0.6), (10.0, 0.0)]   # x v_des


def fit_z_off(plant, v_des, gains, seconds=12.0, schedule=None, verbose=True):
    """Pick z_off per plant by 1-D sweep at m2 (locked pitch, so this isolates sinking and reach
    rather than balance). Ties -- and a full-length survival is a common tie -- go to the smallest
    |z_off|, i.e. the least the controller has to be told to deviate from the plant's own stance."""
    env = make_env("m2", dr=0.0, pushes=False, episode_s=seconds, model_path=plant)
    sched = schedule if schedule is not None else [(t, f * v_des) for t, f in FIT_SCHEDULE]
    best = None
    for z in Z_OFF_GRID:
        g = dict(gains)
        g["z_off"] = z
        res = [rollout(env, g, s, v_des=v_des, est="odom", schedule=sched) for s in FIT_SEEDS]
        t = statistics.median(r["t_alive"] for r in res)
        e = np.mean([r["v_err"] for r in res])
        key = (-t, abs(z), e)
        if best is None or key < best[0]:
            best = (key, z, t, e)
    if verbose:
        print(f"    {Path(plant).name:28s} z_off {best[1]:+.3f} m "
              f"(fit seeds {FIT_SEEDS}: {best[2]:.2f}/{seconds:.0f} s, |v err| {best[3]:.3f})")
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", nargs="*", default=PLANTS)
    ap.add_argument("--milestones", nargs="*", default=["m3", "m2"])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--v-des", type=float, default=0.30)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--demo", action="store_true",
                    help="m2 only: the stand/fwd/stand/back/stand 35 s mission schedule")
    ap.add_argument("--gains", default=GAINS_FILE)
    ap.add_argument("--no-fit-z-off", action="store_true",
                    help="run every plant at the shipped z_off (unfair to the *_bal plants; see "
                         "the table above fit_z_off)")
    ap.add_argument("--json", default=None, help="write the raw per-episode results here")
    args = ap.parse_args()

    gains = dict(GAINS)
    gf = PKG_DIR / args.gains
    if gf.exists():
        gains.update(json.loads(gf.read_text()))
        print(f"[foot_ab] gains from {gf.name}")
    else:
        print(f"[foot_ab] {gf.name} not found — using the built-in defaults")
    seeds = list(range(args.episodes))
    print(f"[foot_ab] paired seeds {seeds}, v_des {args.v_des} m/s, dr=0, no pushes")

    z_off = {p: gains["z_off"] for p in args.plants}
    if not args.no_fit_z_off:
        print("[foot_ab] refitting the leg-extension bias per plant (see fit_z_off):")
        for p in args.plants:
            z_off[p] = fit_z_off(p, args.v_des, gains)
    print()

    raw = {}
    for m in args.milestones:
        sched = ([(0.0, 0.0), (5.0, args.v_des), (15.0, 0.0),
                  (20.0, -args.v_des * 0.6), (30.0, 0.0)] if (args.demo and m == "m2") else None)
        secs = args.seconds if args.seconds is not None else (35.0 if sched else 12.0)
        print(f"=== {m} " + ("(mission schedule, 35 s)" if sched else f"({secs:.0f} s hold)")
              + " " + "=" * 40)
        print(f"{'plant':28s} {'z_off':>7s} {'survived':>9s} {'t_alive med':>12s} {'mean':>7s} "
              f"{'dist med':>9s} {'v_mean':>8s} {'|v err|':>8s}")
        base = None
        for plant in args.plants:
            g = dict(gains)
            g["z_off"] = z_off[plant]
            res = run(plant, m, seeds, args.v_des, secs, sched, g)
            raw[f"{m}/{plant}"] = [{k: v for k, v in r.items() if k != "tel"} for r in res]
            t = [r["t_alive"] for r in res]
            d = [r["dist"] for r in res]
            surv = sum(r["survived"] for r in res)
            med = statistics.median(t)
            if base is None:
                base = med
            print(f"{Path(plant).name:28s} {z_off[plant]:+7.3f} {surv:>4d}/{len(res):<4d} {med:11.2f}s "
                  f"{np.mean(t):6.2f}s {statistics.median(d):8.2f}m "
                  f"{np.mean([r['v_mean'] for r in res]):+7.3f} "
                  f"{np.mean([r['v_err'] for r in res]):7.3f}"
                  + (f"   x{med / base:.2f}" if base and plant != args.plants[0] else "   (ref)"))
        # the spread is the yardstick: a plant-to-plant gap smaller than this is not a result
        spread = [statistics.median([r["t_alive"] for r in raw[f"{m}/{p}"]]) for p in args.plants]
        per_seed = np.array([[r["t_alive"] for r in raw[f"{m}/{p}"]] for p in args.plants])
        print(f"\n  seed-to-seed spread within a plant (max-min t_alive): "
              f"{np.max(per_seed, axis=1).max() - np.min(per_seed, axis=1).min():.2f} s; "
              f"plant-to-plant spread of medians: {max(spread) - min(spread):.2f} s")
        print("  per-seed t_alive (rows = plants, columns = seeds):")
        for p, row in zip(args.plants, per_seed):
            print(f"    {Path(p).name:28s} " + " ".join(f"{v:6.2f}" for v in row))
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(raw, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
