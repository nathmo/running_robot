"""Stage 4 (closed-loop search) and Stage 5 (feedforward absorption) of the gait-library
pipeline (artifact §09), on a FROZEN library-variant stabilizer.

Stage 4:
    python walk_v3/gait_lib/search.py --run walk_v3/runs/v2_lib_s2_free_s0 --entry 3 --evals 400
        J(theta) = t_100m + lam_res RMS(residual) + lam_th max(0, dT/dTmax - 0.85) + lam_lane |y_100|
                   + lam_sym asym + lam_fall falls, averaged over 8 paired seeds, §08 randomization ON
        CMA-ES on theta_r (17) warm-started from the library entry; every candidate is one batch
        of 8 dashes on the GPU (population x seeds envs in one vmap).

Stage 5:
    python walk_v3/gait_lib/search.py --run ... --absorb --entry 3
        Phase-average the greedy residual over >= 50 cycles, project r_bar(phi) onto the weighted
        Fourier basis per family (mirror-consistent: L at phi, R at phi - pi - Delta with the
        structural sign), add to theta, repeat <= 5 times or until RMS(r_bar) < 0.01 rad.
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

import gait
from env import DashEnvV2, EnvParams
from plant import Override
from evaluate import load_run
from gait_lib.cmaes import CMAES
from gait_lib.solver import embed, THETA_R_IDX


def dash_batch(env, agent, thetas, seed, n_max, dr=True):
    """Greedy dashes, one theta per env (thetas [N, 44]). Returns per-env stats."""
    params = EnvParams.final(agent.cfg)._replace(dr_scale=1.0 if dr else 0.0, pitch_assist=0.0)
    ov = Override(theta=jnp.asarray(thetas, jnp.float32))
    state, obs = env.reset(jax.random.PRNGKey(seed), params, ov)
    act = agent._act_greedy

    def body(carry, _):
        state, obs, alive, fell, fin, tline, res_sq, n, thmax, y_line = carry
        a = jnp.clip(act(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        state2, obs2, r, done, info = env.step(state, a, params)
        ending = alive & done
        fell = fell | (ending & info["fallen"])
        fin = fin | (ending & info["finished"])
        tline = jnp.where(ending & info["finished"], info["t_line"], tline)
        res_sq = res_sq + jnp.where(alive, jnp.sum(a[:, :6] ** 2, axis=-1), 0.0)
        n = n + alive.astype(jnp.float32)
        thmax = jnp.maximum(thmax, jnp.where(alive, info["thermal_max"], 0.0))
        newly = state2.crossed & ~state.crossed
        y_line = jnp.where(newly, jnp.abs(env._y(state2.data)), y_line)
        return (state2, obs2, alive & ~done, fell, fin, tline, res_sq, n, thmax, y_line), None

    N = env.n_envs
    init = (state, obs, jnp.ones(N, bool), jnp.zeros(N, bool), jnp.zeros(N, bool), jnp.full(N, -1.0),
            jnp.zeros(N), jnp.zeros(N), jnp.zeros(N), jnp.zeros(N))
    (_, _, alive, fell, fin, tline, res_sq, n, thmax, y_line), _ = jax.lax.scan(body, init, None, length=n_max)
    return dict(fell=np.asarray(fell), fin=np.asarray(fin), t_line=np.asarray(tline),
                res_rms=np.sqrt(np.asarray(res_sq) / np.maximum(np.asarray(n), 1.0) / 6.0),
                thermal_max=np.asarray(thmax), y_line=np.asarray(y_line), alive=np.asarray(alive))


def objective(stats, t_cap, lam_res=5.0, lam_th=50.0, lam_lane=2.0, lam_fall=60.0):
    t = np.where(stats["fin"], stats["t_line"], t_cap)
    return (t + lam_res * stats["res_rms"] + lam_th * np.maximum(0.0, stats["thermal_max"] - 0.85)
            + lam_lane * np.where(stats["y_line"] > 0, stats["y_line"], 0.0) + lam_fall * stats["fell"])


def search(run, entry, evals, n_seeds=8, pop=16, seed=1000, out=None):
    cfg, _, agent = load_run(run, n_envs=1)
    lib = json.loads((PKG / cfg.library_path).read_text()) if not Path(cfg.library_path).exists() \
        else json.loads(Path(cfg.library_path).read_text())
    e = lib["entries"][entry]
    theta_r0 = np.array(e["theta_r"])
    env = DashEnvV2(cfg, n_envs=pop * n_seeds)
    n_max = int(cfg.episode_s / env.control_dt)
    es = CMAES(theta_r0, 0.08, popsize=pop, bounds=(-np.ones(17), np.ones(17)), max_evals=evals, seed=seed)
    sym_idx = [11, 12, 13]                         # Delta, s, o_cam in theta_r
    print(f"[search] entry {entry} v* {e['v_star']:.2f}; {pop} candidates x {n_seeds} seeds per generation")
    while not es.done:
        t0 = time.time()
        X = es.ask()                                        # [pop, 17]
        thetas = np.stack([embed(x) for x in X])            # [pop, 44]
        thetas = np.repeat(thetas, n_seeds, axis=0)         # [pop*n_seeds, 44]
        st = dash_batch(env, agent, thetas, seed, n_max)
        J = objective(st, cfg.episode_s).reshape(pop, n_seeds).mean(1)
        J = J + 0.5 * np.sum(X[:, sym_idx] ** 2, axis=1)   # lam_sym asym
        es.tell(X, J)
        fin = st["fin"].reshape(pop, n_seeds).mean(1)
        print(f"  gen {es.gen:3d} evals {es.evals:4d}: best J {es.best_f:.2f}  finish frac {fin.max():.2f}  "
              f"res_rms {st['res_rms'].reshape(pop, n_seeds).mean(1).min():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    result = dict(entry=entry, theta_r=es.best_x.tolist(), theta=embed(es.best_x).tolist(), J=float(es.best_f))
    if out:
        Path(out).write_text(json.dumps(result, indent=1))
    return result


def absorb(run, entry, n_cycles=50, iters=5, seed=1000, out=None):
    """Stage 5: fold the phase-averaged greedy residual into theta."""
    cfg, _, agent = load_run(run, n_envs=1)
    lib_path = Path(cfg.library_path) if Path(cfg.library_path).exists() else PKG / cfg.library_path
    lib = json.loads(lib_path.read_text())
    e = lib["entries"][entry]
    theta = np.array(e["theta"])
    gp = gait.GaitParams.from_cfg(cfg)
    env = DashEnvV2(cfg, n_envs=8)
    n_max = int(min(cfg.episode_s, n_cycles / max(gait.frequency(theta[gait.I_FREQ], gp, np), 0.5) + 5) / env.control_dt)
    params = EnvParams.final(cfg)._replace(dr_scale=0.0, pitch_assist=0.0)
    act = agent._act_greedy
    for it in range(iters):
        ov = Override(theta=jnp.asarray(theta, jnp.float32))
        state, obs = env.reset(jax.random.PRNGKey(seed), params, ov)

        def body(carry, _):
            state, obs = carry
            a = jnp.clip(act(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
            state2, obs2, r, done, info = env.step(state, a, params)
            return (state2, obs2), (state.phase, a[:, :6], state.spec[:, gait.I_DELTA], state.ep_len > 0)
        _, (phase, res, delta, ok) = jax.lax.scan(body, (state, obs), None, length=n_max)
        phase, res, delta = np.asarray(phase).ravel(), np.asarray(res).reshape(-1, 6), np.asarray(delta).ravel()
        r = res * cfg.residual_scale
        rms = float(np.sqrt(np.mean(r ** 2)))
        print(f"[absorb] iter {it}: residual RMS {rms:.4f} rad over {len(phase)} ticks", flush=True)
        if rms < 0.01:
            break
        # per family: the left leg's residual at phi and the right's (negated) at phi - pi - Delta
        # are samples of the same mirror-consistent correction; least-squares onto the basis
        d = gp.delta_max * np.clip(delta, -1, 1)
        for fam, (iL, iR, amp, sl) in {"hip": (0, 3, gp.roll_amp, gait.I_S_HIP), "cam": (1, 4, gp.cam_amp, gait.I_S_CAM),
                                       "thigh": (2, 5, gp.thigh_amp, gait.I_S_THIGH)}.items():
            ph = np.concatenate([phase, phase - np.pi - d])
            y = np.concatenate([r[:, iL], -r[:, iR]]) / amp
            w = gait.WEIGHTS
            A = np.stack([w[0] * np.ones_like(ph)] + sum(([w[k] * np.cos(k * ph), w[k] * np.sin(k * ph)]
                                                        for k in range(1, 4)), []), 1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            theta[sl] = np.clip(theta[sl] + coef, -1.0, 1.0)
    result = dict(entry=entry, theta=theta.tolist(), residual_rms=rms)
    if out:
        Path(out).write_text(json.dumps(result, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--entry", type=int, default=-1)
    ap.add_argument("--evals", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--absorb", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.absorb:
        print(absorb(args.run, args.entry, out=args.out))
    else:
        print(search(args.run, args.entry, args.evals, n_seeds=args.seeds, out=args.out))


if __name__ == "__main__":
    main()
