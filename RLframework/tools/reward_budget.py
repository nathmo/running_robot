"""Where does the per-step reward actually go? Print the term-by-term budget.

The training line reports one number (`rew`), and a negative one is ambiguous: it can mean the policy
is bad, or it can mean the *objective* is upside down -- if the net per-step reward is negative for a
competent policy, the shortest episode is the best episode and the optimiser will happily learn to fall.
Telling those two apart needs the breakdown, not the sum.

This runs a greedy rollout in a given preset's env (optionally from a checkpoint, otherwise the warm
start being considered) and reports, per reward term: the mean over live ticks, and the share of the
total income and of the total cost. It also reports the net, which is the number that decides whether
staying alive pays.

    python RLframework/tools/reward_budget.py --run RLframework/runs/<run> --preset v3_joystick_s2
    python RLframework/tools/reward_budget.py --cold --preset v4_runstop_stage1 --sample

`--preset` re-homes the checkpoint into a DIFFERENT env than it trained in, which is exactly the
warm-start question: v2 weights evaluated under the v3 reward. `--cold` evaluates a freshly
initialised policy instead -- step 0 of a run from scratch -- with the base spring ON, because
training has it on at curriculum start; `--sample` draws actions at the policy's own std, as a
training rollout does. `--set KEY=VALUE` overrides a config field after the preset.

The budget also prints the alive bonus needed for LIVING > 0. Alive is a constant per live tick,
so for a FIXED policy LIVING(b) = LIVING(0) + b * reward_dt_scale exactly (above the step floor) --
one rollout answers every value of w_alive.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

import networks as nets
from config import PRESETS
from env import EnvParams
from evaluate import load_run
from ppo import PPO, initial_params
from train import make_eval_env


def _override(cfg, sets):
    """--set KEY=VALUE, typed from the field's current value (JSON for tuples and bools)."""
    kw = {}
    for s in sets:
        k, _, v = s.partition("=")
        if not hasattr(cfg, k):
            raise SystemExit(f"--set: Config has no field {k!r}")
        cur = getattr(cfg, k)
        val = json.loads(v) if isinstance(cur, (bool, tuple, list)) or v in ("true", "false") else type(cur)(v)
        kw[k] = tuple(val) if isinstance(cur, tuple) else val
    return replace(cfg, **kw) if kw else cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--cold", action="store_true",
                    help="a freshly initialised policy, no checkpoint (needs --preset): step 0 of a cold run")
    ap.add_argument("--sample", action="store_true",
                    help="sample actions at the policy's std (what a training rollout does), not the greedy mean")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config field after the preset, e.g. --set w_alive=0.5")
    ap.add_argument("--profile", default=None, metavar="TERM[,TERM]",
                    help="also print these terms' mean per QUARTER of the rollout -- a term that grows with "
                         "time (lane grows with distance travelled) hides behind its whole-run mean")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--preset", default=None, help="evaluate under THIS preset's reward (default: the run's)")
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v-cmd", type=float, default=None, help="hold the joystick here (m/s) instead of drawing")
    ap.add_argument("--raw-stats", action="store_true",
                    help="load obs stats verbatim; default applies the warm-start floor training applies")
    ap.add_argument("--curriculum", choices=("start", "final"), default="start",
                    help="'start' = what step 0 of training sees (default); 'final' = the end of every ramp")
    args = ap.parse_args()

    if args.cold:
        if not args.preset:
            ap.error("--cold needs --preset")
        cfg = _override(PRESETS[args.preset](), args.set)
        # so the step-0 budget has to be measured with it -- make_eval_env strips it by default
        env = make_eval_env(cfg, args.n_envs)
        agent = PPO(cfg, env, Path("."), cfg.total_steps, seed=args.seed, eval_env=None)
        print(f"[budget] COLD policy (fresh init, seed {args.seed}), {'sampled' if args.sample else 'greedy'}")
    else:
        if not args.run:
            ap.error("give --run, or --cold with --preset")
        cfg, env, agent = load_run(args.run, args.checkpoint, n_envs=args.n_envs, dr=False,
                                   warm_start=not args.raw_stats)
        if args.preset or args.set:
            cfg = _override(PRESETS[args.preset]() if args.preset else cfg, args.set)
            env = make_eval_env(cfg, args.n_envs)
    base = initial_params(cfg) if args.curriculum == "start" else EnvParams.final(cfg)
    params = base._replace(dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0)
    print(f"[budget] preset={args.preset or 'run'} objective={cfg.objective} curriculum={args.curriculum} "
          f"w_alive={cfg.w_alive} w_track={getattr(cfg, 'w_track', 0)} ticks={args.ticks}")
    import numpy as _np
    _m, _v = _np.asarray(agent.stats.mean), _np.asarray(agent.stats.var)
    print(f"[budget] task-channel normalisation: task[0] mean {_m[374]:.4f} std {_np.sqrt(_v[374]):.4f} "
          f"| task[1] mean {_m[375]:.4f} std {_np.sqrt(_v[375]):.4f}"
          f"{'  (RAW)' if args.raw_stats else '  (warm-start floored)'}")
    print(f"[budget] env params: bringup_scale={params.bringup_scale:.2f} "
          f"cmd=[{params.cmd_lo:.2f},{params.cmd_hi:.2f}] zero_p={params.cmd_zero_p:.2f}")

    state, obs = env.reset(jax.random.PRNGKey(args.seed), params)
    if args.v_cmd is not None:
        state = state.replace(v_cmd=jnp.full_like(state.v_cmd, args.v_cmd),
                              cmd_left=jnp.full_like(state.cmd_left, 1e4))

    def step(carry, _):
        state, obs, key = carry
        nobs = agent.stats.normalize(obs)
        if args.sample:
            key, k = jax.random.split(key)
            mu, log_std, _, _ = agent.net.apply(agent.params, nobs)
            a = nets.sample(k, mu, log_std)
        else:
            a = agent._act_greedy(agent.params, nobs)
        state2, obs2, r, done, info = env.step(state, jnp.clip(a, -1.0, 1.0), params)
        if args.v_cmd is not None:   # auto-reset would redraw it; hold the stick where we put it
            state2 = state2.replace(v_cmd=jnp.full_like(state2.v_cmd, args.v_cmd),
                                    cmd_left=jnp.full_like(state2.cmd_left, 1e4))
        live = 1.0 - done.astype(jnp.float32)
        return (state2, obs2, key), (info["reward_terms"], r, live, info["fallen"].astype(jnp.float32),
                                     state.v_cmd)

    (_, _, _), (terms, rew, live, fell, vcmd) = jax.lax.scan(
        step, (state, obs, jax.random.PRNGKey(args.seed + 7)), None, length=args.ticks)
    vals, counts = np.unique(np.round(np.asarray(vcmd), 3), return_counts=True)
    if len(vals) <= 6:
        print("[budget] commands seen (m/s: share of ticks): "
              + ", ".join(f"{v:.2f}: {c / counts.sum() * 100:.0f}%" for v, c in zip(vals, counts)))
    else:
        print(f"[budget] commands seen: {len(vals)} distinct values in [{vals.min():.2f}, {vals.max():.2f}] m/s")

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
    if args.profile:
        quarters = np.array_split(np.arange(args.ticks), 4)
        print(f"\n[budget] per quarter of the {args.ticks}-tick rollout (mean over live ticks):")
        for k in args.profile.split(","):
            v = np.asarray(terms[k]) * alive_mask
            cells = [v[i].sum() / max(alive_mask[i].sum(), 1.0) for i in quarters]
            print(f"{k:>18}  " + "  ".join(f"{x:+8.3f}" for x in cells)
                  + (f"   (term cap -{cfg.penalty_term_cap:g})" if min(cells) < 0 else ""))
    income = sum(m for _, m in rows if m > 0)
    cost = sum(-m for _, m in rows if m < 0)
    rows.sort(key=lambda kv: kv[1])
    print(f"\n{'term':>18}  {'mean/tick':>10}  {'share':>7}")
    for k, m in rows:
        share = (-m / cost * 100) if m < 0 else (m / max(income, 1e-9) * 100)
        print(f"{k:>18}  {m:>10.4f}  {share:>6.1f}% {'cost' if m < 0 else 'income'}")
    net = float((rew * alive_mask).sum() / n)

    # The reward the optimiser sees is NOT the sum of the terms: it is scaled by reward_dt_scale,
    # floored at -step_reward_floor, and then the terminal fall_penalty lands on the one tick that
    # ends the episode. Reporting the raw sum overstates the living cost (the floor clips it) and
    # folds a one-time -100 into the per-tick mean, which reads as if every tick were catastrophic.
    scale = float(env.reward_dt_scale)
    terms_net = (income - cost) * scale
    floor = -float(cfg.step_reward_floor) * scale
    step_net = max(terms_net, floor)
    print(f"\n{'INCOME':>20}  {income:>10.4f}")
    print(f"{'COST':>20}  {-cost:>10.4f}")
    print(f"{'terms net':>20}  {income - cost:>10.4f}  x reward_dt_scale {scale:.3f} = {terms_net:>8.4f}")
    if terms_net < floor:
        print(f"{'step floor':>20}  {step_net:>10.4f}  (clamped -- cost below the floor is invisible to "
              f"the optimiser, and so is income that only climbs back toward it)")
    print(f"{'LIVING':>20}  {step_net:>10.4f} /tick")
    print(f"{'per-tick incl falls':>20}  {net:>10.4f}  (this is the training line's `rew`)")
    # alive is one constant per live tick, so every other w_alive is arithmetic on this rollout
    alive = dict(rows).get("alive", 0.0)
    bare = income - alive - cost                  # terms net with w_alive = 0, before scaling
    print(f"{'LIVING at w_alive=0':>20}  {max(bare * scale, floor):>10.4f} /tick   "
          f"(w_alive {cfg.w_alive:g} here; each unit of w_alive adds {scale:.3f})")
    need = max(0.0, -bare)
    print(f"{'w_alive for LIVING>0':>20}  {need:>10.4f}" + ("   (none needed)" if need == 0 else ""))
    # What the optimiser actually compares is DISCOUNTED: dying now costs fall_penalty once; living
    # at L per tick is worth L * (1 - gamma^H) / (1 - gamma), which at gamma 0.995 is ~200 L however
    # long the episode is. So living beats dying iff L > -fall_penalty / D -- not iff L > 0.
    g = float(cfg.gamma)
    ep_ticks = float(cfg.episode_s) / float(env.control_dt)
    D = (1.0 - g ** ep_ticks) / (1.0 - g)
    tie = -float(cfg.fall_penalty) / D
    if floor > tie:
        # the floor alone keeps every live tick above the tie: living always wins, whatever it costs --
        # which also means the costs below the floor are teaching nothing
        need_d, why = 0.0, f"   (the step floor {floor:.3f} already sits above the tie)"
    else:
        need_d = max(0.0, (tie / scale) - bare)
        why = "   (none needed)" if need_d == 0 else ""
    print(f"{'w_alive: live>die':>20}  {need_d:>10.4f}   (discounted: living beats dying iff LIVING > "
          f"{tie:.3f}/tick at gamma {g}){why}")

    if abs(step_net) < 1e-9:
        print(f"\n[budget] LIVING is ZERO: every live tick is clamped to the step floor, so no cost and no "
              f"income below it reaches the optimiser -- the only signal left is the {cfg.fall_penalty:.0f} "
              f"for falling.")
    elif step_net < 0:
        v_live = step_net * D
        print(f"\n[budget] LIVING COSTS {-step_net:.3f}/tick. Discounted at gamma {g} (effective horizon "
              f"~{1.0 / (1.0 - g):.0f} ticks, not the {ep_ticks:.0f}-tick episode), living on is worth "
              f"{v_live:.1f} against {-cfg.fall_penalty:.0f} for dying now.")
        if v_live < -float(cfg.fall_penalty):
            print(f"[budget] => dying immediately is worth more than living. That is an OBJECTIVE bug, "
                  f"not a policy failure: the optimiser will hunt for the shortest episode, which on this "
                  f"plant means railing the gait clock to its floor and falling early.")
        else:
            print("[budget] => surviving still wins at the optimiser's horizon, so the objective is not upside down.")
    else:
        print(f"\n[budget] net positive: living pays {step_net:.3f}/tick over {ep_ticks:.0f} ticks "
              f"({step_net * ep_ticks:.0f} an episode) against a {cfg.fall_penalty:.0f} fall penalty, so "
              f"termination is a real loss.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
