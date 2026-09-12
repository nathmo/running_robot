"""Replay the CPU arm's golden fixture (walk_mit/golden_v2.py --write) through the MJX env.

    python walk_v3/tools/replay_golden.py walk_mit/golden/v2_s1_clean_seed0.npz [--preset v2_s1_planar_easy]

The fixture holds, per tick, the action the CPU env received and everything it produced (obs
402, reward, commit flag, phase, live spec, delayed ctrl, thermal state, terms, qpos/qvel). The
two plants are not the same MuJoCo model (their qpos is 20-wide, ours 18), so the replay starts
each side from its OWN settled keyframe with reset noise off and feeds the recorded actions.

What must agree and what is reported:
  commit flags   EXACT (integer latch logic; free-running clock at the recorded frequency)
  spec_live      EXACT (the latch)
  phase          within 1e-5 unless a contact resync fired on one side only
  once-block     spec 44 + task 2 + commit 1 exact; the phase channels (cos/sin here, sin/cos
                 there) are compared as a pair
  frames / priv  max |diff| per block, with the plant difference stated: they diverge at the
                 mm / mrad level from tick 0 (different keyframes, float32 vs float64) and the
                 gap grows as the trajectories separate
  reward         per-term max |diff| over the first 20 ticks
"""
import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np
import jax
import jax.numpy as jnp

from config import get_config
from env import DashEnvV2, EnvParams
from train import make_eval_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--preset", default="v2_s1_planar_easy")
    ap.add_argument("--iterations", type=int, default=0, help="MJX solver cap override (0 = preset/XML)")
    ap.add_argument("--ls-iterations", type=int, default=0)
    args = ap.parse_args()
    z = dict(np.load(args.npz, allow_pickle=False))
    acts = z["action"]
    n = len(acts)
    cfg = get_config(args.preset)
    if args.iterations or args.ls_iterations:
        import dataclasses
        cfg = dataclasses.replace(cfg, mjx_iterations=args.iterations or cfg.mjx_iterations,
                                  mjx_ls_iterations=args.ls_iterations or cfg.mjx_ls_iterations)
        print(f'[replay] solver cap iterations={cfg.mjx_iterations} ls={cfg.mjx_ls_iterations}')
    cfg.reset_joint_noise = 0.0
    cfg.resync_enable = False if int(z["commit"].sum()) <= 1 else cfg.resync_enable
    env = make_eval_env(cfg, n=1, keep_assist=True)
    params = EnvParams.final(cfg)._replace(dr_scale=0.0, pitch_assist=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0)
    state, obs = env.reset(jax.random.PRNGKey(0), params)
    obs_all, rew, commit, phase, spec, terms, done_at = [np.asarray(obs[0])], [], [], [], [], [], None
    for k in range(n):
        a = jnp.asarray(acts[k][None].astype(np.float32))
        commit.append(int(np.asarray(state.commit)[0]))
        state, obs, r, d, info = env.step(state, a, params)
        obs_all.append(np.asarray(obs[0]))
        rew.append(float(r[0]))
        phase.append(float(state.phase[0]))
        spec.append(np.asarray(state.spec[0]))
        terms.append({kk: float(v[0]) for kk, v in info["reward_terms"].items()})
        if bool(d[0]):
            done_at = k + 1
            break
    m = len(rew)
    print(f"fixture {Path(args.npz).name}: CPU {len(z['reward'])} ticks, {int(z['commit'].sum())} commits; "
          f"MJX replay {m} ticks{' (ended)' if done_at else ''}, {sum(commit)} commits")
    nn = min(m, len(z["reward"]))
    c_ok = np.array_equal(np.asarray(commit[:nn]), z["commit"][:nn])
    print(f"  commit flags   {'EXACT' if c_ok else 'DIFFER at tick %d' % int(np.argmax(np.asarray(commit[:nn]) != z['commit'][:nn]))}")
    s_err = np.abs(np.asarray(spec[:nn]) - z["spec_live"][:nn]).max()
    print(f"  spec_live      max|diff| {s_err:.2e} {'EXACT' if s_err == 0 else ''}")
    p_err = np.abs(np.asarray(phase[:nn]) - z["phase"][:nn]).max()
    print(f"  phase          max|diff| {p_err:.2e}")
    O, Z = np.asarray(obs_all[:nn + 1]), z["obs"][:nn + 1]
    blocks = [("history frames 0:330", 0, 330), ("once: spec 330:374", 330, 374), ("once: task 374:376", 374, 376),
              ("once: commit 376", 376, 377), ("priv 377:402", 377, 402)]
    for name, a_, b_ in blocks:
        e = np.abs(O[:, a_:b_] - Z[:, a_:b_])
        print(f"  {name:22s} max|diff| {e.max():.2e}  at tick0 {e[0].max():.2e}  tick1 {e[1].max() if nn else 0:.2e}")
    # phase channels: ours (cos, sin), theirs (sin, cos) -> compare as a set per frame
    ph_o = O[:, 25:27]
    ph_z = Z[:, 25:27][:, ::-1]
    print(f"  newest-frame phase (cos,sin vs their sin,cos swapped) max|diff| {np.abs(ph_o[:, :] - ph_z).max():.2e}"
          if False else f"  frame phase channels compared with the order swapped: max|diff| "
          f"{np.abs(O[:, 297 + 25:297 + 27] - Z[:, 297 + 25:297 + 27][:, ::-1]).max():.2e}")
    # per-channel diff of the NEWEST frame over the first ticks (phase swapped): a scale or lag
    # mismatch in a channel that is zero at reset (velocities, torques, residual) hides under the
    # phase diff in the block maxima above
    chan = ([f"q{i}" for i in range(6)] + [f"qd{i}" for i in range(6)] + [f"tau{i}" for i in range(6)]
            + ["gx", "gy", "gz", "wx", "wy", "wz", "yaw_lp", "ph_a", "ph_b"] + [f"res{i}" for i in range(6)])
    Oz = O[:, 297:330].copy(); Zz = Z[:, 297:330].copy()
    Zz[:, 25:27] = Zz[:, 25:27][:, ::-1]
    kt = min(6, nn)
    print("  newest frame per channel |diff| (ticks 0..%d, phase swapped):" % (kt - 1))
    for t in range(kt):
        e = np.abs(Oz[t] - Zz[t])
        top = np.argsort(-e)[:5]
        print("    tick %d: " % t + ", ".join(f"{chan[i]} {e[i]:.3f}" for i in top if e[i] > 1e-4) or "    tick %d: all < 1e-4" % t)
    r_err = np.abs(np.asarray(rew[:nn]) - z["reward"][:nn])
    print(f"  reward         max|diff| first 20 ticks {r_err[:20].max():.3f}, all {r_err.max():.3f}")
    names = [str(x) for x in z["term_names"]]
    worst = []
    for j, t in enumerate(names):
        if t in terms[0]:
            e = max(abs(terms[k][t] - z["terms"][k, j]) for k in range(min(20, nn)))
            worst.append((e, t))
    worst.sort(reverse=True)
    print("  largest per-term diffs (first 20 ticks): " + ", ".join(f"{t} {e:.3f}" for e, t in worst[:6]))
    missing = [t for t in names if t not in terms[0]]
    extra = [t for t in terms[0] if t not in names]
    print(f"  terms only on CPU: {missing}; only here: {extra}")


if __name__ == "__main__":
    main()
