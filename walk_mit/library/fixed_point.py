"""Stage 0 of the gait-library pipeline (artifact §09): the return map and its fixed points.

    section Sigma : phi = 0+ (just after the wrap), state x = (qpos, qvel) minus the cyclic
                    coordinates x, y, yaw, incl. the passive 4-bar / pushrod joints
    return map    : P(x; theta) -- set the simulator to x, play ONE open-loop cycle T = 1/f of
                    the latched generator (armature-fit drive, 12 ms delay ring primed with the
                    previous cycle's commands, no residual, no reflexes), read x' at the next wrap
    defect        : d(x) = P(x; theta) - x, solved by damped Newton with a finite-difference
                    Jacobian (h_q 1e-4 rad, h_v 1e-3 rad/s; step clip 0.05; <= 20 iters), CMA-ES
                    fallback on ||d||^2
    outputs       : x*, ||d||, v* (the forward speed is a component of x*), the Floquet
                    multipliers lambda_i = eig(dP/dx at x*), per-cycle envelope flags (torque,
                    speed, thermal, GRF <= 3.5 BW, workspace)

Contact makes P non-smooth, hence the damping and the derivative-free fallback. The multipliers
are the number the whole design turns on: they say how hard the stabilizer's job is before any
training (the RUNNER's orbit was ~2.7 per cycle; <= 1 is self-stable).

    python library/fixed_point.py --theta placeholder [--preset v2_returnmap] [--json out.json]
    python library/fixed_point.py --theta lib.json:0          # entry 0 of a library file
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import gait_v2  # noqa: E402
from config import get_config  # noqa: E402
from env import DashEnv  # noqa: E402


class ReturnMap:
    """P(x; theta) on the clean v2 plant. One env, reused; every call is deterministic."""

    def __init__(self, cfg=None, preset="v2_returnmap"):
        self.cfg = cfg or get_config(preset)
        assert self.cfg.action_mode == "latched" and self.cfg.spec_source == "library"
        self.env = DashEnv(self.cfg)
        self.env.reset(seed=0)
        self.keep, self.nv = self.env.section_dims()
        self.nx = len(self.keep) + self.nv
        self.nq_keep = len(self.keep)
        self.lay = self.env._lay
        self.n_cycles = 0

    # ---- the map --------------------------------------------------------------------------
    def x_from_env(self):
        return self.env.section_state()

    def P(self, x, theta, record=None):
        env = self.env
        env.set_section_state(x, theta)
        env._f_hz = gait_v2.frequency(theta[self.lay.freq], self.cfg.gait_freq_hz)
        env._spec_live[:] = theta
        env._theta_ep = np.asarray(theta, dtype=float).copy()
        env._lib_stand = env._theta_ep.copy()
        env._sprint_crossed = False
        env._lib_v_ref = 0.0
        x1 = env.run_cycle(record=record)
        self.n_cycles += 1
        return x1

    def defect(self, x, theta):
        return self.P(x, theta) - x

    def jacobian(self, x, theta, h_q=1e-4, h_v=1e-3):
        """Finite-difference dP/dx (central) -- one cycle per column."""
        n = self.nx
        J = np.zeros((n, n))
        h = np.concatenate([np.full(self.nq_keep, h_q), np.full(self.nv, h_v)])
        for i in range(n):
            e = np.zeros(n)
            e[i] = h[i]
            J[:, i] = (self.P(x + e, theta) - self.P(x - e, theta)) / (2 * h[i])
        return J

    # ---- the solver -------------------------------------------------------------------------
    def solve(self, theta, x0=None, iters=20, step_clip=0.05, tol=1e-3, verbose=True,
              cma_fallback=True, cma_evals=400):
        theta = np.clip(np.asarray(theta, dtype=float), -1.0, 1.0)
        if x0 is None:
            x0 = self.stance_state()
        x = np.asarray(x0, dtype=float).copy()
        best = (np.inf, x.copy())
        hist = []
        for it in range(iters):
            d = self.defect(x, theta)
            nd = float(np.linalg.norm(d))
            hist.append(nd)
            if nd < best[0]:
                best = (nd, x.copy())
            if verbose:
                print(f"  newton {it:2d}: ||d|| {nd:.3e}  vx* {x[self.nq_keep + 0]:+.3f}")
            if nd < tol:
                break
            if not np.all(np.isfinite(d)):
                break
            J = self.jacobian(x, theta)
            A = J - np.eye(self.nx)
            try:
                dx = -np.linalg.lstsq(A, d, rcond=1e-8)[0]
            except np.linalg.LinAlgError:
                break
            nrm = float(np.linalg.norm(dx))
            if nrm > step_clip:
                dx *= step_clip / nrm
            # damped: halve the step until the defect does not grow
            alpha = 1.0
            for _ in range(6):
                xn = x + alpha * dx
                dn = float(np.linalg.norm(self.defect(xn, theta)))
                if dn < nd or not np.isfinite(dn):
                    break
                alpha *= 0.5
            x = xn
        nd, x = best
        if nd >= tol and cma_fallback:
            from library.cmaes import CMAES
            sc = np.concatenate([np.full(self.nq_keep, 0.02), np.full(self.nv, 0.2)])
            es = CMAES(x / sc, 0.5, popsize=12, seed=0)
            while not es.done(cma_evals):
                X = es.ask()
                es.tell(X, [float(np.sum(self.defect(xi * sc, theta) ** 2)) for xi in X])
            xc = es.best_x * sc
            ndc = float(np.sqrt(es.best_f))
            if verbose:
                print(f"  cma fallback: ||d|| {nd:.3e} -> {ndc:.3e} ({es.evals} evals)")
            if ndc < nd:
                nd, x = ndc, xc
        return dict(x_star=x, defect=nd, converged=bool(nd < tol), history=hist)

    def stance_state(self):
        """The settled stance as a section state (the solver's default start)."""
        self.env.reset(seed=0)
        return self.x_from_env()

    def floquet(self, x_star, theta):
        J = self.jacobian(x_star, theta)
        lam = np.linalg.eigvals(J)
        return lam, J

    # ---- the envelope over one cycle -------------------------------------------------------
    def envelope(self, x_star, theta):
        env = self.env
        c = self.cfg
        rec = dict(tau_peak=np.zeros(env.nu), tau_sq=np.zeros(env.nu), n=0, w_peak=np.zeros(env.nu),
                   grf_peak=0.0, ws_out=0, z_min=np.inf, vx=[])
        lim = env._orig_forcerange[:env.nu, 1]
        vl = np.asarray(env._motor_vel_limit, float)
        grav = float(-env.model.opt.gravity[2])
        bw = float(np.sum(env.model.body_mass)) * grav

        def record(e):
            tau = np.abs(e.data.actuator_force[:e.nu])
            rec["tau_peak"] = np.maximum(rec["tau_peak"], tau)
            rec["tau_sq"] += tau ** 2
            rec["n"] += 1
            rec["w_peak"] = np.maximum(rec["w_peak"], np.abs(e.data.qvel[e.act_dadr]))
            rec["grf_peak"] = max(rec["grf_peak"], float(e._contact_normal_forces().sum()))
            rec["ws_out"] += int(e._workspace_violation()) if c.workspace_kill else 0
            rec["z_min"] = min(rec["z_min"], float(e.data.qpos[2]))
            rec["vx"].append(float(e.data.qvel[0]))
        self.P(x_star, theta, record=record)
        n = max(rec["n"], 1)
        tau_rms = np.sqrt(rec["tau_sq"] / n)
        tau_cont = np.asarray(c.thermal_tau_cont, float)[:env.nu]
        return dict(
            torque_util_peak=(rec["tau_peak"] / lim).tolist(),
            torque_ok=bool(np.all(rec["tau_peak"] <= lim * 0.999)),
            speed_util_peak=(rec["w_peak"] / np.where(np.isfinite(vl), vl, np.inf)).tolist(),
            speed_ok=bool(np.all(rec["w_peak"] <= np.where(np.isfinite(vl), vl, np.inf))),
            thermal_rms_frac=(tau_rms / tau_cont).tolist(),
            thermal_ok=bool(np.all(tau_rms <= tau_cont)),
            grf_peak_bw=float(rec["grf_peak"] / bw),
            grf_ok=bool(rec["grf_peak"] <= 3.5 * bw),
            workspace_ok=bool(rec["ws_out"] == 0),
            z_min=float(rec["z_min"]),
            fell=bool(rec["z_min"] < c.term_height),
            v_mean=float(np.mean(rec["vx"])) if rec["vx"] else 0.0,
        )


def load_theta(spec, cfg):
    """--theta: 'placeholder' | 'neutral' | 'lib.json:i' | 'file.npy' | comma-separated 17 (reduced)."""
    lay = gait_v2.Layout(cfg.n_harmonics)
    if spec == "placeholder":
        return np.asarray(DashEnv.default_library(cfg)[0]["theta"], dtype=float)
    if spec == "neutral":
        return gait_v2.neutral_spec(lay)
    if ":" in spec and spec.split(":")[0].endswith(".json"):
        path, i = spec.split(":")
        d = json.loads(Path(path).read_text())
        entries = d["entries"] if isinstance(d, dict) else d
        return np.asarray(entries[int(i)]["theta"], dtype=float)
    if spec.endswith(".npy"):
        return np.load(spec)
    vals = np.array([float(v) for v in spec.split(",")])
    if vals.size == 17:
        return gait_v2.embed_reduced(vals, cfg, lay)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v2_returnmap")
    ap.add_argument("--theta", default="placeholder")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-cma", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rm = ReturnMap(preset=args.preset)
    theta = load_theta(args.theta, rm.cfg)
    f = gait_v2.frequency(theta[rm.lay.freq], rm.cfg.gait_freq_hz)
    print(f"return map on {args.preset}: nx {rm.nx} (qpos {rm.nq_keep} + qvel {rm.nv}), "
          f"f {f:.2f} Hz = {1 / (f * rm.env.control_dt):.0f} ticks/cycle, delay {rm.env._delay_ms:.0f} ms")
    res = rm.solve(theta, iters=args.iters, cma_fallback=not args.no_cma)
    x = res["x_star"]
    lam, _ = rm.floquet(x, theta)
    lam_max = float(np.max(np.abs(lam)))
    envl = rm.envelope(x, theta)
    print(f"\nfixed point: ||d|| {res['defect']:.3e} ({'converged' if res['converged'] else 'NOT converged'}), "
          f"cycles used {rm.n_cycles}")
    print(f"  v*  = {x[rm.nq_keep]:+.3f} m/s (vx), vz {x[rm.nq_keep + 2]:+.3f}, height {x[0]:.3f} m")
    print(f"  Floquet |lambda| max {lam_max:.3f}  (top 5: {np.round(np.sort(np.abs(lam))[::-1][:5], 3)})")
    print(f"  envelope: torque {envl['torque_ok']} (peak util {np.round(envl['torque_util_peak'], 2)}), "
          f"speed {envl['speed_ok']}, thermal {envl['thermal_ok']} (rms/cont {np.round(envl['thermal_rms_frac'], 2)}), "
          f"GRF {envl['grf_peak_bw']:.2f} BW {'ok' if envl['grf_ok'] else 'OVER'}, workspace {envl['workspace_ok']}, "
          f"fell {envl['fell']}")
    if args.json:
        out = dict(theta=theta.tolist(), x_star=x.tolist(), defect=res["defect"],
                   converged=res["converged"], lambda_max=lam_max, lambda_abs=np.abs(lam).tolist(),
                   envelope=envl, v=float(x[rm.nq_keep]), preset=args.preset)
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
