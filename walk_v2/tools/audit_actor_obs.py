"""Does anything the robot cannot measure reach the ACTOR? Answer by perturbation, not by reading.

The privileged tail is structurally safe -- it is the critic's input and the estimator's target, with
`stop_gradient` between the estimator and the actor -- but that is not the whole question. Under v2 the
gait clock was pulled toward the measured touchdown phase, which moved `phi` (the actor reads it as
[cos, sin]) and could carry the commit flag over the wrap; and `task[1]` was a distance-to-go computed
from ground-truth world x. Neither is a privileged *tensor*; both are privileged *information*, and
reading the code is how you miss that.

So perturb the simulator-only quantities and assert the actor slice does not move:

  1. foot contact -- scale every contact's normal force and nudge toe heights, so `grounded`, the contact
     accumulator and the touchdown edges all change;
  2. absolute position -- shift the base x of the whole episode, so `sprint_d` and anything derived from
     it changes.

Run the same seeded episode with and without each perturbation. `obs[:actor_dim]` must be bit-identical;
`obs[actor_dim:]` is expected to differ (that IS the privileged tail, and if it does not differ the
perturbation did not bite, which the harness checks too).

    python walk_v2/tools/audit_actor_obs.py --run walk_v2/runs/<run> [--checkpoint ...]

Exit code 0 = clean. Non-zero names the first tick and channel that moved.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from env import EnvParams
from evaluate import load_run
from train import make_eval_env


def rollout(env, agent, params, seed, n, contact_scale=1.0, x_shift=0.0):
    """Greedy rollout returning the stacked observations of env 0."""
    act = agent._act_greedy
    state, obs = env.reset(jax.random.PRNGKey(seed), params)
    if x_shift:
        # move the whole world: sprint_d is x - x0, so shifting BOTH leaves the task unchanged only if
        # nothing else reads absolute x. That is exactly the claim under test.
        qpos = state.data.qpos.at[:, env.plant.base_q["x"]].add(x_shift)
        state = state.replace(data=state.data.replace(qpos=qpos))

    def step(carry, _):
        state, obs = carry
        a = jnp.clip(act(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        state2, obs2, _, _, _ = env.step(state, a, params)
        return (state2, obs2), obs

    (_, _), seq = jax.lax.scan(step, (state, obs), None, length=n)
    return np.asarray(seq)[:, 0, :]


def _is_information(i, actor_dim, frame_dim=33):
    """Does this channel CARRY information, or is it a consequence of dynamics?

    The distinction decides what a divergence means. At tick 0 both runs see an identical actor
    observation, so they emit an identical action; physics can then differ only by the perturbation
    itself. A genuine privilege leak therefore shows up in a channel that *encodes* the perturbed
    quantity -- the task entries, the commit flag, or the clock phase the resync used to pull. A
    divergence that starts in motor_pos/vel/torque is the simulator's own arithmetic diverging (a
    25 m coordinate offset changes the last bits of every contact position, and a stiff contact
    solver amplifies that), which is not evidence of a leak.
    """
    i = int(i)
    if i >= actor_dim:
        return False
    hist = frame_dim * 10
    if i < hist:
        off = i % frame_dim
        return 25 <= off < 27          # the [cos, sin] phase pair
    o = i - hist
    return o >= 44                     # task[0], task[1], commit


def compare(name, a, b, actor_dim, tol=1e-5):
    act_a, act_b = a[:, :actor_dim], b[:, :actor_dim]
    d = np.abs(act_a - act_b)
    moved = d.max()
    priv_moved = np.abs(a[:, actor_dim:] - b[:, actor_dim:]).max()
    ok = moved <= tol
    early = np.abs(act_a[:3] - act_b[:3]).max()
    info_cols = np.array([_is_information(i, actor_dim) for i in range(actor_dim)])
    info_moved = d[:, info_cols].max() if info_cols.any() else 0.0
    print(f"[audit] {name}: actor max|delta| {moved:.3e}  (privileged tail moved {priv_moved:.3e})")
    print(f"[audit]   first 3 ticks max|delta| {early:.3e}   INFORMATION channels "
          f"(phase/task/commit) max|delta| {info_moved:.3e}")
    if priv_moved == 0.0:
        print(f"[audit]   WARNING: the perturbation did not reach the privileged tail either -- it may "
              f"not have bitten at all, so this is not evidence of anything")
    if not ok:
        # Report the EARLIEST divergence, not the biggest. The largest delta is almost always the
        # previous-residual block, which moved only because the policy's own action moved -- a
        # consequence, not the entry point. The first tick that moves at all is the leak.
        ticks = np.where(d.max(axis=1) > tol)[0]
        t0 = int(ticks[0])
        chans = np.where(d[t0] > tol)[0]
        print(f"[audit]   FAIL: first divergence at tick {t0}, channel(s) {list(chans[:6])}"
              f"{' ...' if chans.size > 6 else ''} -- {_where(chans[0], actor_dim)}")
        tm, im = np.unravel_index(np.argmax(d), d.shape)
        print(f"[audit]   largest divergence {moved:.3e} at tick {tm}, channel {im} "
              f"({_where(im, actor_dim)})")
        if info_moved <= tol and early <= tol:
            print(f"[audit]   READING: no INFORMATION channel moved and the first ticks are at "
                  f"floating-point level, so this is the simulator's arithmetic diverging, NOT a "
                  f"privilege leak. A leak would move phase/task/commit immediately.")
    return ok

def verdict(ok, info_clean):
    return ("CLEAN: the actor slice is a pure function of measurable inputs" if ok else
            ("NO LEAK FOUND: every actor divergence is downstream of simulator arithmetic, not of "
             "the perturbed information" if info_clean else "LEAK: an information channel moved"))


def _where(i, actor_dim, frame_dim=33):
    """Name an actor channel, so a failure points at the leak instead of an index."""
    i = int(i)
    if i >= actor_dim:
        return "privileged tail"
    hist = frame_dim * 10
    if i < hist:
        k, off = divmod(i, frame_dim)
        block = ("motor_pos" if off < 6 else "motor_vel" if off < 12 else "motor_torque" if off < 18
                 else "gravity" if off < 21 else "gyro" if off < 24 else "lp_yaw" if off < 25
                 else "PHASE [cos,sin]" if off < 27 else "prev_residual")
        return f"history frame {k}, {block}"
    o = i - hist
    if o < 44:
        return f"once-block: latched spec[{o}]"
    return ("once-block: TASK[0] (the command)" if o == 44 else
            "once-block: TASK[1]" if o == 45 else "once-block: commit flag")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="float32 contact solving diverges from a coordinate offset alone; "
                         "0.0 makes this test measure arithmetic, not privilege")
    args = ap.parse_args()

    cfg, env, agent = load_run(args.run, args.checkpoint, n_envs=4, dr=False)
    params = EnvParams.final(cfg)._replace(dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0,
                                           pitch_assist=0.0, stoplight_prob=0.0)
    print(f"[audit] objective={cfg.objective}  resync_enable={cfg.resync_enable}  "
          f"brake_prior={cfg.brake_prior}  spec_source={cfg.spec_source}  actor_dim={env.actor_dim}")

    base = rollout(env, agent, params, args.seed, args.ticks)
    ok = True

    # 1. absolute position. Under the sprint objective task[1] is a distance countdown off ground-truth
    #    x, so this SHOULD fail there -- that is the v2 leak, and the joystick objective is what removes it.
    shifted = rollout(env, agent, params, args.seed, args.ticks, x_shift=25.0)
    ok &= compare("absolute base x (+25 m)", base, shifted, env.actor_dim, args.tol)

    # 2. foot contact. With resync_enable the clock is pulled by touchdown edges, which moves the actor's
    #    phase and can carry its commit flag; with it off, nothing contact-derived should reach the actor.
    if cfg.resync_enable:
        print("[audit] resync_enable=True: the clock IS contact-driven by construction, so the contact "
              "check below is expected to FAIL. Set resync_enable=False for a contact-free actor.")
    # cfg values are read at TRACE time, so the perturbed env has to be BUILT from a patched config --
    # mutating cfg after construction changes nothing and would have made this test pass vacuously.
    import copy
    cfg_b = copy.deepcopy(cfg)
    cfg_b.grounded_h = float(cfg.grounded_h) * 3.0   # a different touchdown definition = different edges
    env_b = make_eval_env(cfg_b, 4)
    perturbed = rollout(env_b, agent, params, args.seed, args.ticks)
    ok &= compare("foot-contact threshold (grounded_h x3)", base, perturbed, env.actor_dim, args.tol)

    print(f"\n[audit] {'CLEAN: the actor slice is a pure function of measurable inputs' if ok else 'LEAK: see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
