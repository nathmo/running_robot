"""Stage 0 + Stage 1 of the gait-library pipeline (artifact §09): the open-loop return map on
MJX, its fixed points by damped Newton, the Floquet multipliers, and the CMA-ES library solve.

    python walk_v3/gait_lib/solver.py --preset v2_s2_free --out walk_v3/gait_lib/library.json
    python walk_v3/gait_lib/solver.py --preset v2_s1_planar --speeds 0 1 2 3 --evals 600

Section Sigma: phi = 0+ (just after the wrap). State x = (qpos, qvel) minus the cyclic
coordinates (base x, y, yaw). Return map P(x; theta): set the plant to x, play ONE open-loop
cycle T = 1/f of gait.feedforward(theta, phi) targets through the fitted PD drive (kp(phi),
kd(phi) from theta, 12 ms delay, torque-speed clamp), no residual, no reflexes, and read x' at
the next wrap. Defect d(x) = P(x) - x, solved by damped Newton with a finite-difference Jacobian
(h_q 1e-4, h_v 1e-3, step clip 0.05, <= 20 iterations); CMA-ES on ||d||^2 as the fallback.
Every Jacobian column and every CMA-ES candidate is one MJX rollout, vmapped: a Newton
iteration is one batched call.

Outputs per entry: theta (44), x*, ||d||, v* (forward speed at the section), the multipliers
lambda_i = eig(dP/dx), and the per-cycle envelope flags (torque, no-load speed, GRF <= 3.5 BW,
workspace, thermal-per-cycle). theta_r (17) = [f, cam a0 a1 b1 a2 b2, thigh a0 a1 b1 a2 b2,
Delta, s, o_cam, kp_lvl, kd_lvl, hip a1]; the third harmonic and the rest are 0.
"""
import argparse
import json
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from mujoco import mjx

import gait
import drive
from config import get_config
from plant import Plant, model_with, draw_plant, Override
from gait_lib.cmaes import CMAES

# theta_r -> theta (44) embedding
THETA_R_NAMES = ["f", "cam_a0", "cam_a1", "cam_b1", "cam_a2", "cam_b2", "thigh_a0", "thigh_a1", "thigh_b1",
                 "thigh_a2", "thigh_b2", "delta", "s", "o_cam", "kp_lvl", "kd_lvl", "hip_a1"]
THETA_R_IDX = [gait.I_FREQ, 0, 1, 2, 3, 4, 7, 8, 9, 10, 11, gait.I_DELTA, gait.I_S, 41, 21, 28, 15]


def embed(theta_r):
    th = np.zeros(gait.SPEC_DIM)
    th[THETA_R_IDX] = np.clip(theta_r, -1.0, 1.0)
    return th


class ReturnMap:
    def __init__(self, cfg, delay_ms=None):
        self.cfg = cfg
        self.plant = Plant(cfg)
        self.gp = gait.GaitParams.from_cfg(cfg)
        self.delay_ms = float(cfg.drive_delay_ms if delay_ms is None else delay_ms)
        p = self.plant
        # nominal plant fields (no DR)
        self.draw = draw_plant(jax.random.PRNGKey(0), _nominal_cfg(cfg), p, 0.0, Override())
        self.mx = model_with(p, self.draw.fields)
        cyc = [p.base_q[n] for n in ("x", "y", "yaw") if p.base_q[n] >= 0]
        self.cyc_q = np.array(cyc)
        self.cyc_v = np.array([p.base_d[n] for n in ("x", "y", "yaw") if p.base_d[n] >= 0])
        self.free_q = np.array([i for i in range(p.nq) if i not in self.cyc_q])
        self.free_v = np.array([i for i in range(p.nv) if i not in self.cyc_v])
        self.nx = len(self.free_q) + len(self.free_v)
        self.n_ticks_max = int(round(1.0 / cfg.gait_freq_hz[0] / p.control_dt))   # longest cycle
        self._cycle_v = jax.jit(jax.vmap(self._cycle, in_axes=(0, 0)))

    def x_from_qv(self, qpos, qvel):
        return np.concatenate([np.asarray(qpos)[self.free_q], np.asarray(qvel)[self.free_v]])

    def qv_from_x(self, x):
        p = self.plant
        q = np.asarray(p.key_qpos).copy()
        v = np.zeros(p.nv)
        q[self.free_q] = x[:len(self.free_q)]
        v[self.free_v] = x[len(self.free_q):]
        return q, v

    def x_keyframe(self):
        return self.x_from_qv(self.plant.key_qpos, np.zeros(self.plant.nv))

    # -------------------------------------------------------------- one cycle, JAX
    def _cycle(self, x, theta):
        p, gp, c = self.plant, self.gp, self.cfg
        nq_f = len(self.free_q)
        qpos = jnp.asarray(p.key_qpos).at[jnp.asarray(self.free_q)].set(x[:nq_f])
        qvel = jnp.zeros(p.nv).at[jnp.asarray(self.free_v)].set(x[nq_f:])
        data = p.data0.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(p.nu), qfrc_applied=jnp.zeros(p.nv),
                               xfrc_applied=jnp.zeros_like(p.data0.xfrc_applied))
        data = mjx.forward(self.mx, data)
        f = gait.frequency(theta[gait.I_FREQ], gp)
        n_ticks = jnp.round(1.0 / f / p.control_dt).astype(jnp.int32)
        dphi = 2.0 * jnp.pi / n_ticks.astype(jnp.float32)          # exactly one cycle in n ticks
        nominal = jnp.asarray(p.nominal_ctrl)
        cmd0 = jnp.concatenate([nominal, jnp.asarray(gp.drive_kp), jnp.asarray(gp.drive_kd)])
        peak, kt, r = jnp.asarray(p.tau_peak), jnp.asarray(c.motor_kt_joint), jnp.asarray(c.motor_r_ohm)

        def tick(carry, i):
            data, cmd_buf, phi, acc = carry
            active = i < n_ticks
            q_ref = gait.feedforward(theta, phi, nominal, gp)
            q_ref = jnp.clip(q_ref, jnp.asarray(p.q_lo), jnp.asarray(p.q_hi))
            kp, kd = gait.impedance(theta, phi, gp)
            cmd = jnp.concatenate([q_ref, kp, kd])
            cmd_buf = jnp.stack([cmd, cmd_buf[0], cmd_buf[1]])

            def substep(carry2, k):
                d, a = carry2
                live = drive.live_command(k, self.delay_ms, cmd_buf)
                q = d.qpos[p.act_qadr]
                qd = d.qvel[p.act_dadr]
                lim = drive.torque_limit(qd, peak, 1.0, kt, r, c.motor_bus_volts)
                tau = drive.pd_torque(q, qd, live[:6], live[6:12], live[12:18], lim)
                d = d.replace(ctrl=tau)
                d = mjx.step(self.mx, d)
                con = getattr(d, "_impl", d).contact
                efc = getattr(d, "_impl", d).efc_force
                fn = jnp.where(con.efc_address >= 0, efc[jnp.maximum(con.efc_address, 0)], 0.0)
                floor = (con.geom == p.floor_gid).any(axis=1) & (con.dist < 0.0)
                grf = jnp.sum(jnp.where(floor, jnp.maximum(fn, 0.0), 0.0))
                a = dict(tau_max=jnp.maximum(a["tau_max"], jnp.abs(tau) / peak),
                         w_max=jnp.maximum(a["w_max"], jnp.abs(qd) / jnp.asarray(c.motor_vel_limit)),
                         grf_max=jnp.maximum(a["grf_max"], grf / p.bw),
                         tau_sq=a["tau_sq"] + tau ** 2, n=a["n"] + 1.0)
                return (d, a), None

            (data2, acc2), _ = lax.scan(substep, (data, acc), jnp.arange(p.decimation))
            data = jax.tree_util.tree_map(lambda a_, b_: jnp.where(active, a_, b_), data2, data)
            acc = jax.tree_util.tree_map(lambda a_, b_: jnp.where(active, a_, b_), acc2, acc)
            phi = jnp.where(active, phi + dphi, phi)
            return (data, cmd_buf, phi, acc), None

        acc0 = dict(tau_max=jnp.zeros(6), w_max=jnp.zeros(6), grf_max=jnp.zeros(()), tau_sq=jnp.zeros(6),
                    n=jnp.zeros(()))
        (data, _, phi, acc), _ = lax.scan(tick, (data, jnp.stack([cmd0, cmd0, cmd0]), jnp.zeros(()), acc0),
                                          jnp.arange(self.n_ticks_max))
        x_out = jnp.concatenate([data.qpos[jnp.asarray(self.free_q)], data.qvel[jnp.asarray(self.free_v)]])
        v_fwd = data.qvel[p.base_d["x"]]
        R = data.xmat[p.base_bid].reshape(3, 3)
        base = data.xpos[p.base_bid]
        ws = []
        for i, g in enumerate(p.foot_gids):
            tb = R.T @ (data.geom_xpos[g] - base)
            ws.append((jnp.abs(tb[0] - p.ws_ref[i, 0]) > c.workspace_dx_max)
                      | (tb[2] - p.ws_ref[i, 2] > c.workspace_dz_max) | (tb[2] - p.ws_ref[i, 2] < c.workspace_dz_min))
        flags = dict(torque=acc["tau_max"].max(), speed=acc["w_max"].max(), grf=acc["grf_max"],
                     thermal_cycle=jnp.sqrt(acc["tau_sq"] / jnp.maximum(acc["n"], 1.0)).max()
                     / jnp.asarray(c.thermal_tau_cont).max(), workspace=jnp.stack(ws).any(),
                     fell=(base[2] < c.term_height) | (R[2, 2] < 0.5) | ~jnp.isfinite(x_out).all(),
                     n_ticks=n_ticks)
        return x_out, v_fwd, flags

    def P(self, X, thetas):
        """Batched return map. X [B, nx], thetas [B, 44] -> (X' [B, nx], v [B], flags)."""
        X = jnp.asarray(np.asarray(X, np.float32))
        T = jnp.asarray(np.asarray(thetas, np.float32))
        xo, v, fl = self._cycle_v(X, T)
        return np.asarray(xo, np.float64), np.asarray(v, np.float64), {k: np.asarray(val) for k, val in fl.items()}

    # -------------------------------------------------------------- Newton on the section
    def solve(self, theta, x0=None, iters=20, tol=1e-3, h_q=1e-4, h_v=1e-3, step_clip=0.05, verbose=False):
        nq_f = len(self.free_q)
        x = np.array(self.x_keyframe() if x0 is None else x0, float)
        h = np.concatenate([np.full(nq_f, h_q), np.full(self.nx - nq_f, h_v)])
        hist = []
        best = None
        for it in range(iters):
            X = np.stack([x] + [x + h[i] * np.eye(self.nx)[i] for i in range(self.nx)])
            Xo, V, fl = self.P(X, np.tile(theta, (len(X), 1)))
            d = Xo[0] - x
            nd = float(np.linalg.norm(d))
            hist.append(nd)
            if best is None or nd < best[0]:
                best = (nd, x.copy(), Xo[0].copy(), float(V[0]), {k: v[0] for k, v in fl.items()}, None)
            J = (Xo[1:] - Xo[0][None, :]) / h[:, None]         # dP/dx (rows = perturbed coord)
            J = J.T
            if verbose:
                print(f"    newton {it:2d}: |d| {nd:.2e}  v* {V[0]:.3f}", flush=True)
            if nd < tol or not np.isfinite(nd):
                best = best[:5] + (J,)
                break
            A = J - np.eye(self.nx)
            try:
                dx = -np.linalg.solve(A + 1e-6 * np.eye(self.nx), d)
            except np.linalg.LinAlgError:
                dx = -np.linalg.lstsq(A, d, rcond=None)[0]
            n_dx = np.linalg.norm(dx)
            if n_dx > step_clip:
                dx *= step_clip / n_dx
            x = x + dx
            best = best[:5] + (J,)
        nd, x, xo, v, flags, J = best
        lam = np.linalg.eigvals(J) if J is not None else np.array([np.nan])
        return dict(x=x, defect=nd, v_star=v, lambda_max=float(np.abs(lam).max()), lambdas=lam,
                    flags=flags, hist=hist, converged=bool(nd < tol))


def _nominal_cfg(cfg):
    from dataclasses import replace
    return replace(cfg, dr_enable=False, obs_noise_enable=False, thermal_hot_start_max=0.0)


# ------------------------------------------------------------------ Stage 1: the library
def solve_library(cfg, speeds, evals=600, pop=16, lam_bar=2.0, out=None, seed=0, verbose=True):
    rm = ReturnMap(cfg)
    c = cfg
    lo = np.array([-1.0] * 17)
    hi = np.array([1.0] * 17)
    # theta_r start: neutral knobs, mid frequency, a modest first harmonic on cam (forward stride)
    x_r = np.zeros(17)
    x_r[0] = -0.3           # ~ 1.3 Hz
    x_r[2] = 0.4            # cam a1
    x_r[7] = -0.3           # thigh a1
    entries = []
    x_seed = None
    for v_t in speeds:
        t0 = time.time()
        es = CMAES(x_r, 0.15, popsize=pop, bounds=(lo, hi), max_evals=evals, seed=seed)
        best = None
        while not es.done:
            X = es.ask()
            fs = []
            for xr in X:
                th = embed(xr)
                sol = rm.solve(th, x0=x_seed, iters=12)
                fl = sol["flags"]
                pen = (10.0 * max(0.0, sol["defect"] - 1e-3) + 2.0 * max(0.0, fl["torque"] - 1.0)
                       + 2.0 * max(0.0, fl["speed"] - 1.0) + 2.0 * max(0.0, fl["grf"] - 3.5)
                       + 5.0 * float(fl["workspace"]) + 20.0 * float(fl["fell"])
                       + 1.0 * max(0.0, sol["lambda_max"] - lam_bar))
                J = (sol["v_star"] - v_t) ** 2 + pen
                fs.append(J)
                if best is None or J < best[0]:
                    best = (J, xr.copy(), sol)
            es.tell(X, np.array(fs))
            if verbose:
                print(f"  v_t {v_t:.1f}: gen {es.gen:3d} evals {es.evals:4d} best J {es.best_f:.4f} "
                      f"(v* {best[2]['v_star']:.3f}, |d| {best[2]['defect']:.1e}, "
                      f"lam_max {best[2]['lambda_max']:.2f})  {time.time() - t0:.0f}s", flush=True)
        J, xr, sol = best
        x_r = xr.copy()
        x_seed = sol["x"]
        q, v = rm.qv_from_x(sol["x"])
        entries.append(dict(v_target=float(v_t), v_star=float(sol["v_star"]), theta=embed(xr).tolist(),
                            theta_r=xr.tolist(), qpos=q.tolist(), qvel=v.tolist(), defect=float(sol["defect"]),
                            lambda_max=float(sol["lambda_max"]),
                            lambdas_abs=np.abs(sol["lambdas"]).tolist(),
                            flags={k: float(val) for k, val in sol["flags"].items()},
                            converged=bool(sol["converged"]), J=float(J)))
        if verbose:
            print(f"[library] entry v_t {v_t:.1f}: v* {sol['v_star']:.3f} m/s, |d| {sol['defect']:.1e}, "
                  f"lam_max {sol['lambda_max']:.2f}, flags {entries[-1]['flags']}", flush=True)
    lib = dict(preset=getattr(cfg, "_preset", ""), model_path=cfg.model_path, nq=rm.plant.nq,
               theta_r_names=THETA_R_NAMES, entries=entries, lambda_bar=lam_bar)
    if out:
        Path(out).write_text(json.dumps(lib, indent=1))
        print(f"[library] wrote {out}")
    return lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v2_s2_free")
    ap.add_argument("--speeds", type=float, nargs="*", default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--evals", type=int, default=600)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--lambda-bar", type=float, default=2.0)
    ap.add_argument("--out", default=str(PKG / "gait_lib" / "library.json"))
    ap.add_argument("--probe", action="store_true", help="just solve the neutral gait once and print")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    if args.probe:
        rm = ReturnMap(cfg)
        th = embed(np.array([-0.3, 0, 0.4, 0, 0, 0, 0, -0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        t = time.time()
        sol = rm.solve(th, verbose=True)
        print(f"probe: v* {sol['v_star']:.3f} |d| {sol['defect']:.2e} lam_max {sol['lambda_max']:.2f} "
              f"flags {sol['flags']} ({time.time() - t:.0f}s)")
        return
    solve_library(cfg, args.speeds, evals=args.evals, pop=args.pop, lam_bar=args.lambda_bar, out=args.out)


if __name__ == "__main__":
    main()
