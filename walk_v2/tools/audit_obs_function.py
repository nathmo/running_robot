"""Is the actor observation a pure function of measurable state? Test the FUNCTION, not a rollout.

`audit_actor_obs.py` perturbs the world and compares whole trajectories. That test is too blunt in both
directions, and its positive control proves it: a v2 checkpoint whose clock IS contact-driven and whose
task[1] IS ground-truth odometry shows ZERO movement in its information channels, because

  * a +25 m x-shift never moves `task[1] = clip((line - d)/8, 0, 1)` off its saturated 1.0 -- the ramp
    only bites within 8 m of the line; and
  * raising `grounded_h` cannot change `grounded = contact_acc | contact_array | (h < grounded_h)` when
    the contact array is already firing.

Meanwhile float32 contact solving diverges from a coordinate offset alone, so the test reports a "leak"
made entirely of arithmetic. It answers neither question.

This tests the observation function directly. Take ONE real state from a rollout. Recompute the actor
observation from it. Then perturb only quantities the robot cannot measure -- the odometry origin, the
gait clock's touchdown estimate, the contact flags, the privileged tail -- WITHOUT stepping physics, and
recompute. Identical physics, so any difference in obs[:actor_dim] is a genuine information leak, and
there is no chaotic amplification to hide behind.

    python walk_v2/tools/audit_obs_function.py --run walk_v2/runs/<run> --checkpoint ...

Run it on a v2 checkpoint too: that MUST report a leak, or the test is insensitive and proves nothing.
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
from ppo import initial_params

FRAME = 33


def channel_name(i, actor_dim):
    if i >= actor_dim:
        return "privileged tail"
    hist = FRAME * 10
    if i < hist:
        k, off = divmod(i, FRAME)
        block = ("motor_pos" if off < 6 else "motor_vel" if off < 12 else "motor_torque" if off < 18
                 else "gravity" if off < 21 else "gyro" if off < 24 else "lp_yaw" if off < 25
                 else "PHASE[cos,sin]" if off < 27 else "prev_residual")
        return f"frame{k}.{block}"
    o = i - hist
    if o < 44:
        return f"spec[{o}]"
    return ("TASK[0] (command)" if o == 44 else "TASK[1] (reserved/odometry)" if o == 45
            else "commit flag")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=120, help="ticks to reach a representative state")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    cfg, env, agent = load_run(args.run, args.checkpoint, n_envs=args.n_envs, dr=False)
    params = initial_params(cfg)._replace(dr_scale=0.0, ctrl_jitter_ms=0.0, ctrl_drop_prob=0.0,
                                          pitch_assist=0.0)
    ad = env.actor_dim
    print(f"[fn-audit] {args.run}")
    print(f"[fn-audit] objective={cfg.objective} resync_enable={cfg.resync_enable} "
          f"brake_prior={cfg.brake_prior} actor_dim={ad}")

    # --- reach a representative mid-episode state
    state, obs = env.reset(jax.random.PRNGKey(args.seed), params)

    def step(carry, _):
        state, obs = carry
        a = jnp.clip(agent._act_greedy(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        state2, obs2, _, _, _ = env.step(state, a, params)
        return (state2, obs2), None

    (state, obs), _ = jax.lax.scan(step, (state, obs), None, length=args.warmup)
    base = np.asarray(obs)

    # --- perturbations of UNMEASURABLE state only. Physics is untouched: we re-run the same step from
    #     the same data, changing only bookkeeping the robot has no sensor for.
    def rerun(mod, label):
        s2 = mod(state)
        a = jnp.clip(agent._act_greedy(agent.params, agent.stats.normalize(obs)), -1.0, 1.0)
        _, o2, _, _, _ = env.step(s2, a, params)
        d = np.abs(np.asarray(o2)[:, :ad] - np.asarray(env.step(state, a, params)[1])[:, :ad])
        moved = d.max()
        chans = np.where(d.max(0) > 1e-9)[0]
        verdict = "clean" if moved <= 1e-9 else "LEAK"
        print(f"\n[fn-audit] {label}")
        print(f"           actor max|delta| {moved:.3e}  -> {verdict}")
        if chans.size:
            names = sorted({channel_name(int(c), ad) for c in chans})
            print(f"           channels touched: {', '.join(names[:8])}"
                  f"{' ...' if len(names) > 8 else ''}")
        return moved <= 1e-9

    ok = True
    # 1. ODOMETRY: move the episode's origin. sprint_d = x - x0, so shifting x0 changes the distance
    #    travelled by 60 m without touching a single physical quantity. Under v2 this MUST move task[1].
    ok &= rerun(lambda s: s.replace(x0=s.x0 - 60.0) if hasattr(s, "x0") else s,
                "odometry origin shifted 60 m (sprint_d +60, physics identical)")

    # 2. THE TOUCHDOWN ESTIMATE: the resync pulls the clock toward phi_td_hat. Under v2 this MUST move
    #    the phase the actor reads; under v3 the field is inert.
    ok &= rerun(lambda s: s.replace(phi_td_hat=jnp.full_like(s.phi_td_hat, 3.0),
                                    resynced=jnp.ones_like(s.resynced))
                if hasattr(s, "phi_td_hat") else s,
                "touchdown phase estimate forced to 3.0 rad (contact-derived)")

    # 3. THE CONTACT ACCUMULATOR: what the clock's touchdown edge is built from.
    ok &= rerun(lambda s: s.replace(contact_acc=~s.contact_acc) if hasattr(s, "contact_acc") else s,
                "contact accumulator inverted")

    print(f"\n[fn-audit] {'CLEAN: nothing unmeasurable reaches the actor' if ok else 'LEAK: see above'}")
    print("[fn-audit] NOTE: a clean result is only meaningful if this same test reports a LEAK on a v2 "
          "checkpoint (resync on, odometry task). Run it there as the positive control.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
