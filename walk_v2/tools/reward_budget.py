"""Where does the per-step reward actually go? Print the term-by-term budget.

The training line reports one number (`rew`), and a negative one is ambiguous: it can mean the policy
is bad, or it can mean the *objective* is upside down -- if the net per-step reward is negative for a
competent policy, the shortest episode is the best episode and the optimiser will happily learn to fall.
Telling those two apart needs the breakdown, not the sum.

This runs a greedy rollout in a given preset's env (optionally from a checkpoint, otherwise the warm
start being considered) and reports, per reward term: the mean over live ticks, and the share of the
total income and of the total cost. It also reports the net, which is the number that decides whether
staying alive pays.

    python walk_v2/tools/reward_budget.py --run walk_v2/runs/<run> --preset v3_joystick_s2

`--preset` re-homes the checkpoint into a DIFFERENT env than it trained in, which is exactly the
warm-start question: v2 weights evaluated under the v3 reward.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from config import PRESETS
from env import EnvParams
from evaluate import load_run
from train import make_eval_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--preset", default=None, help="evaluate under THIS preset's reward (default: the run's)")
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v-cmd", type=float, default=None, help="hold the joystick here (m/s) instead of drawing")
    args = ap.parse_args()

    cfg, env, agent = load_run(args.run, args.checkpoint, n_envs=args.n_envs, dr=False)
    if args.preset:
        cfg = PRESETS[args.preset]()
        env = make_eval_env(cfg, args.n_envs)
    params = EnvParams.final(cfg)._replace(dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0)
    print(f"[budget] preset={args.preset or 'run'} objective={cfg.objective} "
          f"w_alive={cfg.w_alive} w_track={getattr(cfg, 'w_track', 0)} ticks={args.ticks}")

    state, obs = env.reset(jax.random.PRNGKey(args.seed), params)
    if args.v_cmd is not None:
        state = state.replace(v_cmd=jnp.full_like(state.v_cmd, args.v_cmd),
                              cmd_left=jnp.full_like(state.cmd_left, 1e4))

    def step(carry, _):
        state, obs = carry
        a = jnp.clip(agent._act_greedy(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        state2, obs2, r, done, info = env.step(state, a, params)
        if args.v_cmd is not None:   # auto-reset would redraw it; hold the stick where we put it
            state2 = state2.replace(v_cmd=jnp.full_like(state2.v_cmd, args.v_cmd),
                                    cmd_left=jnp.full_like(state2.cmd_left, 1e4))
        live = 1.0 - done.astype(jnp.float32)
        return (state2, obs2), (info["reward_terms"], r, live, info["fallen"].astype(jnp.float32))

    (_, _), (terms, rew, live, fell) = jax.lax.scan(step, (state, obs), None, length=args.ticks)

    # only count ticks before each env's FIRST termination: after that it is a fresh episode and the
    # average silently becomes a mixture over episode ages.
    live = np.asarray(live)
    alive_mask = np.cumprod(np.vstack([np.ones((1, live.shape[1])), live[:-1]]), axis=0)
    n = max(alive_mask.sum(), 1.0)
    rew = np.asarray(rew)
    print(f"[budget] mean episode length {alive_mask.sum(0).mean():.0f} ticks of {args.ticks}, "
          f"falls {np.asarray(fell).max(0).mean() * 100:.0f}%")

    rows = []
    for k, v in terms.items():
        m = float((np.asarray(v) * alive_mask).sum() / n)
        rows.append((k, m))
    income = sum(m for _, m in rows if m > 0)
    cost = sum(-m for _, m in rows if m < 0)
    rows.sort(key=lambda kv: kv[1])
    print(f"\n{'term':>18}  {'mean/tick':>10}  {'share':>7}")
    for k, m in rows:
        share = (-m / cost * 100) if m < 0 else (m / max(income, 1e-9) * 100)
        print(f"{k:>18}  {m:>10.4f}  {share:>6.1f}% {'cost' if m < 0 else 'income'}")
    net = float((rew * alive_mask).sum() / n)
    print(f"\n{'INCOME':>18}  {income:>10.4f}\n{'COST':>18}  {-cost:>10.4f}\n{'NET':>18}  {net:>10.4f}")
    if net < 0:
        print(f"\n[budget] NET IS NEGATIVE: every extra tick alive costs {-net:.3f}. The shortest episode is "
              f"the best episode, so falling early is optimal play -- this is an objective bug, not a "
              f"policy failure. Raise w_alive above {cost - income:.2f} or cut the dominant cost above.")
    else:
        print(f"\n[budget] net positive: staying alive pays {net:.3f}/tick, so termination is a real loss.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
