"""Stage 4 of the gait-library pipeline (artifact §09): the closed-loop search -- the gait, finally.

    J(theta) = t_100m + lam_res * RMS(residual) + lam_th * max(0, dT/dT_max - 0.85)
             + lam_lane * |y_100| + lam_sym * asym + lam_fall * falls
    averaged over paired seeds with the §08 randomisation ON, CMA-ES on theta_r (17) with the
    stabilizer FROZEN, <= 1000 evaluations, warm-started from the library entry.

The residual term is what writes the plant physics into theta: the search lowers the
stabilizer's bill by moving the gait toward where the leg carries its weight cheaply. Output: a
JSON library with the searched entry (x* from the return map if requested) -- the speed-margin
front; the §08 gate picks the point.

    python library/search.py --run runs/v2_lib_s2_s0 --entry model/gait_library.json:1 \\
        --evals 200 --seeds 8 --out model/gait_library_searched.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import gait_v2  # noqa: E402
from library.cmaes import CMAES  # noqa: E402
from library.library_solve import LO, HI, SYM_IDX  # noqa: E402


def rollout_cost(model, venv, raw, theta, v_ref, seeds, max_s, lam):
    """Paired-seed greedy dashes at library entry theta; returns (J, details)."""
    lay = raw._lay
    raw.set_library_entry(theta, v=v_ref)
    costs, dets = [], []
    for sd in seeds:
        venv.seed(int(sd))
        obs = venv.reset()
        res_sq, n, fell, t_line, y_end, th_max = 0.0, 0, False, None, 0.0, 0.0
        for k in range(int(max_s / raw.control_dt)):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, d, info = venv.step(a)
            r = raw._prev_residual
            res_sq += float(np.sum(r ** 2)); n += 1
            v2 = info[0].get("v2", {})
            th_max = max(th_max, float(v2.get("theta_max", 0.0)))
            if d[0]:
                sp = info[0].get("sprint", {})
                t_line = sp.get("t_line")
                fell = t_line is None and not bool(info[0].get("TimeLimit.truncated", False))
                break
            y_end = float(raw.data.qpos[1])
        rms = float(np.sqrt(res_sq / max(n, 1)))
        t100 = float(t_line) if t_line is not None else max_s
        knobs = gait_v2.knob_vector(theta, lay)
        J = (t100 + lam["res"] * rms + lam["th"] * max(0.0, th_max - 0.85)
             + lam["lane"] * abs(y_end) + lam["sym"] * float(np.sum(knobs ** 2)) + lam["fall"] * float(fell))
        costs.append(J)
        dets.append(dict(seed=int(sd), t_line=t_line, fell=fell, res_rms=rms, y_end=y_end, theta_max=th_max))
    return float(np.mean(costs)), dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a trained v2_lib_* run (frozen stabilizer)")
    ap.add_argument("--entry", required=True, help="library.json:index to warm-start from")
    ap.add_argument("--evals", type=int, default=1000)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--lam", default="res=20,th=50,lane=5,sym=2,fall=60",
                    help="penalty weights: seconds per unit of each term")
    ap.add_argument("--asym", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from evaluate import build
    model, venv, raw = build(Path(args.run), None, None)
    assert raw.latched and raw.spec_source == "library", "search needs a library-variant run"
    lam = {k: float(v) for k, v in (kv.split("=") for kv in args.lam.split(","))}
    path, i = args.entry.split(":")
    lib = json.loads(Path(path).read_text())
    entry = lib["entries"][int(i)]
    theta0 = np.asarray(entry["theta"], dtype=float)
    v_ref = float(entry.get("v", 1.0))
    t0 = gait_v2.reduce_full(theta0, raw.cfg, raw._lay)
    seeds = [args.seed0 + k for k in range(args.seeds)]

    def cost(t):
        t = np.asarray(t, dtype=float).copy()
        if not args.asym:
            t[SYM_IDX] = 0.0
        theta = gait_v2.embed_reduced(t, raw.cfg, raw._lay)
        return rollout_cost(model, venv, raw, theta, v_ref, seeds, args.seconds, lam)[0]

    J0 = cost(t0)
    print(f"warm start J {J0:.2f} s-equivalent at v_ref {v_ref:.2f} m/s")
    es = CMAES(np.clip(t0, LO, HI), 0.15, LO, HI, popsize=args.popsize, seed=0)
    t_start = time.time()
    while not es.done(args.evals):
        X = es.ask()
        es.tell(X, [cost(x) for x in X])
        e, fb, fm = es.history[-1]
        print(f"  evals {e:5d}  best {fb:.2f}  median {fm:.2f}  sigma {es.sigma:.3f}  "
              f"{time.time() - t_start:.0f} s", flush=True)
    t_best = es.best_x.copy()
    if not args.asym:
        t_best[SYM_IDX] = 0.0
    theta_best = gait_v2.embed_reduced(t_best, raw.cfg, raw._lay)
    Jb, dets = rollout_cost(model, venv, raw, theta_best, v_ref, seeds, args.seconds, lam)
    print(f"searched: J {J0:.2f} -> {Jb:.2f}; finishes {sum(d['t_line'] is not None for d in dets)}/{len(dets)}, "
          f"res RMS {np.mean([d['res_rms'] for d in dets]):.3f}")
    out = dict(source_run=args.run, warm_entry=args.entry, lam=lam, J_warm=J0, J_best=Jb,
               entries=[dict(v=v_ref, theta=theta_best.tolist(), theta_r=t_best.tolist(),
                             x_star=entry.get("x_star"), lambda_max=entry.get("lambda_max"),
                             details=dets, source="search")] +
               [e for e in lib["entries"] if e.get("v", 0.0) == 0.0])
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
