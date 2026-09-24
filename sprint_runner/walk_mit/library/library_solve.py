"""Stage 1 of the gait-library pipeline (artifact §09): the library solve.

    search variables theta_r (17)  f, cam a0 a1 b1 a2 b2, thigh a0 a1 b1 a2 b2, Delta, s, o_cam,
                                   kp_lvl, kd_lvl, hip a1   (gait_v2.embed_reduced)
    objective                      max v*  s.t. ||d|| < 1e-3, envelope flags clear, max|lambda| <= lam_bar
    optimizer                      CMA-ES, pop 16, <= 2000 evals; continuation slow -> fast in
                                   0.5 m/s steps (how HZD libraries avoid local minima, and it
                                   gives lambda_max along the whole speed axis)
    entries                        v in {0 (stand), 1, 2, 3, v_max} m/s, mirror-symmetric
                                   (Delta = s = o = 0) by default (--asym allows the knobs)

Each evaluation is one fixed-point solve (a few hundred cycle rollouts); a library is an hour on
the cluster, deterministic, no training. Output JSON: entries with theta (44), x* (the episode
reset state for Stage 3), v*, lambda_max, envelope margins. lam_bar is an open decision (§13):
report lambda_max for every entry regardless.

    python library/library_solve.py --out model/gait_library.json [--evals 300] [--speeds 1,2,3]
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
from library.fixed_point import ReturnMap  # noqa: E402

# reduced-variable box (raw units; f in Hz)
LO = np.array([1.0] + [-1.0] * 10 + [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
HI = np.array([5.0] + [1.0] * 10 + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
SYM_IDX = [11, 12, 13]           # Delta, s, o_cam -- pinned at 0 for the symmetric front


class LibraryObjective:
    def __init__(self, rm, v_target, lam_bar, sym=True, newton_iters=8, verbose=False):
        self.rm, self.v_target, self.lam_bar, self.sym = rm, float(v_target), float(lam_bar), sym
        self.newton_iters = newton_iters
        self.verbose = verbose
        self.x_warm = rm.stance_state()
        self.cache = {}

    def theta_of(self, t):
        t = np.asarray(t, dtype=float).copy()
        if self.sym:
            t[SYM_IDX] = 0.0
        return gait_v2.embed_reduced(t, self.rm.cfg, self.rm.lay)

    def __call__(self, t, want_details=False):
        theta = self.theta_of(t)
        res = self.rm.solve(theta, x0=self.x_warm, iters=self.newton_iters, verbose=False,
                            cma_fallback=False)
        x = res["x_star"]
        v = float(x[self.rm.nq_keep])
        d = float(res["defect"])
        J = (v - self.v_target) ** 2
        J += 50.0 * max(0.0, d - 1e-3)
        envl = self.rm.envelope(x, theta)
        if envl["fell"]:
            J += 10.0
        for k in ("torque_ok", "speed_ok", "thermal_ok", "grf_ok", "workspace_ok"):
            if not envl[k]:
                J += 2.0
        lam_max = None
        if d < 5e-3 and not envl["fell"]:
            lam, _ = self.rm.floquet(x, theta)
            lam_max = float(np.max(np.abs(lam)))
            if self.lam_bar > 0:
                J += 1.0 * max(0.0, lam_max - self.lam_bar)
            self.x_warm = x.copy()               # warm-start the next solve from a good orbit
        if want_details:
            return J, dict(theta=theta, x_star=x, v=v, defect=d, lambda_max=lam_max, envelope=envl)
        return J


def solve_library(rm, speeds, evals, popsize=16, lam_bar=0.0, sym=True, seed=0, verbose=True):
    entries = []
    lay = rm.lay
    # the stand entry: zero amplitudes at a mid clock; x* is the settled stance
    stand = gait_v2.neutral_spec(lay, freq_raw=gait_v2.freq_raw_of(2.0, rm.cfg.gait_freq_hz))
    entries.append(dict(v=0.0, theta=stand.tolist(), x_star=rm.stance_state().tolist(),
                        lambda_max=None, defect=0.0, envelope=None, source="stand"))
    # continuation: start from the placeholder gait, then chain along the speed axis
    t = gait_v2.reduce_full(np.asarray(rm.env.default_library(rm.cfg)[0]["theta"]), rm.cfg, lay)
    sigma = 0.25
    for v_t in speeds:
        t0 = time.time()
        obj = LibraryObjective(rm, v_t, lam_bar, sym=sym)
        es = CMAES(np.clip(t, LO, HI), sigma, LO, HI, popsize=popsize, seed=seed)
        while not es.done(evals):
            X = es.ask()
            es.tell(X, [obj(x) for x in X])
            if verbose:
                e, fb, fm = es.history[-1]
                print(f"  v_t {v_t:.1f}: evals {e:5d}  best J {fb:.4f}  median {fm:.4f}  sigma {es.sigma:.3f}",
                      flush=True)
        J, det = obj(es.best_x, want_details=True)
        t = es.best_x.copy()
        sigma = max(0.1, es.sigma)
        print(f"entry v_t {v_t:.1f}: v* {det['v']:+.3f} m/s, ||d|| {det['defect']:.2e}, lambda_max "
              f"{det['lambda_max']}, J {J:.4f}, {rm.n_cycles} cycles, {time.time() - t0:.0f} s", flush=True)
        entries.append(dict(v=float(det["v"]), v_target=float(v_t), theta=det["theta"].tolist(),
                            theta_r=t.tolist(), x_star=det["x_star"].tolist(),
                            lambda_max=det["lambda_max"], defect=det["defect"],
                            envelope=det["envelope"], J=float(J), source="cmaes"))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v2_returnmap")
    ap.add_argument("--speeds", default="1.0,2.0,3.0")
    ap.add_argument("--evals", type=int, default=2000, help="CMA-ES evaluations per speed")
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--lam-bar", type=float, default=0.0, help="multiplier bound (0 = report only)")
    ap.add_argument("--asym", action="store_true", help="allow the Delta/s/o knobs (second pass)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(PKG / "model" / "gait_library.json"))
    args = ap.parse_args()
    rm = ReturnMap(preset=args.preset)
    speeds = [float(v) for v in args.speeds.split(",")]
    entries = solve_library(rm, speeds, args.evals, args.popsize, args.lam_bar, not args.asym,
                            args.seed)
    out = dict(preset=args.preset, speeds=speeds, evals=args.evals, lam_bar=args.lam_bar,
               symmetric=not args.asym, entries=entries)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}: {len(entries)} entries")


if __name__ == "__main__":
    main()
