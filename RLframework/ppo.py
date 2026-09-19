"""PPO for the v2 walker, in JAX on one GPU: rollout = one lax.scan over N envs, update =
minibatch epochs over the flattened rollout. The pieces the artifact adds to plain PPO (§04, §06):

  * masked log-prob / entropy: the latched dims are scored only on commit ticks (info["commit"]),
    so a spec emission that never reached the plant neither moves the ratio nor gets a gradient
  * symmetry loss  w_sym ||k(M_o s) + k(s)||^2 on the actor mean's five knobs (mirror applied in
    RAW observation space, then normalized with the same running stats)
  * supervised velocity estimator: its own Adam over the estimator subtree only, MSE against the
    privileged tail's true velocity, 2 epochs after each rollout, gradients touch nothing else
  * entropy anneal keyed on competence (worse-foot swing fraction) with a hard deadline, and the
    forced log_std clamp anneal down to std 0.25 (the determinism-gap fix, walk_mit 2026-08-28)
  * observation normalization = running mean/var like VecNormalize (clip 10), updated after
    each rollout from the rollout's raw observations, optionally mirror-symmetrized
  * gated, retreating curricula (dr_scale, stance ratio, efficiency, timing jitter/drop) and the
    clock ramps (sprint line 25 -> 100 m, pitch assist 1 -> 0), persisted for --resume

Truncation at the episode cap BOOTSTRAPS from the value of the state it cut (Pardo et al. 2018);
only a fall or a finish is terminal. (Before 2026-09-16 the cap was terminal, which taught the
critic that the world ends there -- on a healthy policy, every episode.) A 60 s cap is rarely hit
in a dash that ends by finish or fall.
"""
import json
import math
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax import struct

import gait
import networks as nets
import env as env_mod
from env import DashEnvV2, EnvParams
from plant import Override


# ------------------------------------------------------------------ running obs stats
@struct.dataclass
class ObsStats:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

    @classmethod
    def init(cls, dim):
        return cls(mean=jnp.zeros(dim), var=jnp.ones(dim), count=jnp.asarray(1e-4))

    def normalize(self, obs, clip=10.0, eps=1e-8):
        return jnp.clip((obs - self.mean) / jnp.sqrt(self.var + eps), -clip, clip)

    def denormalize(self, nobs, eps=1e-8):
        return nobs * jnp.sqrt(self.var + eps) + self.mean

    def update(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        return ObsStats(mean=new_mean, var=m2 / tot, count=tot)


class Transition(NamedTuple):
    obs: jnp.ndarray        # normalized, [T, N, obs]
    action: jnp.ndarray     # unclipped sample
    log_prob: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    mask: jnp.ndarray       # [T, N, A]
    trunc: jnp.ndarray      # done at the TIME LIMIT (and not fallen): bootstrap, do not treat as terminal
    v_final: jnp.ndarray    # V(the state the episode ended in), before the auto-reset


def gae_fn(tr, last_value, gamma, lam):
    """Generalised advantage, with time limits handled as time limits.

    A TIME LIMIT IS NOT A TERMINAL STATE (Pardo et al. 2018, *Time Limits in Reinforcement
    Learning*). Treating it as one teaches the critic that the world ends at the episode cap -- and a
    healthy policy reaches the cap on nearly every episode, so the bias lands precisely on the
    behaviour we are trying to produce. A truncated step bootstraps from the value of the state it
    was cut in (`v_final`, saved by the env before its auto-reset); a fall or a finish bootstraps
    from 0, as it should. Either way the GAE recursion stops there."""
    def scan_fn(carry, x):
        adv_next, v_next = carry
        r, d, v, tc, vf = x
        nonterm = 1.0 - d.astype(jnp.float32)
        v_boot = jnp.where(tc, vf, v_next * nonterm)
        delta = r + gamma * v_boot - v
        adv = delta + gamma * lam * nonterm * adv_next
        return (adv, v), adv
    _, adv = jax.lax.scan(scan_fn, (jnp.zeros_like(last_value), last_value),
                          (tr.reward, tr.done, tr.value, tr.trunc, tr.v_final), reverse=True)
    return adv, adv + tr.value


# ------------------------------------------------------------------ the trainer
def initial_params(c):
    """EnvParams at the START of every curriculum -- what step 0 of training actually sees.

    Module-level so diagnostics can reproduce the training condition exactly: `EnvParams.final(cfg)` is
    the END of the curricula (full bring-up, full command range, assist off), which is a different and
    much harder env than the one a run begins in. Measuring the wrong one reads as a policy failure.
    """
    return EnvParams(dr_scale=(float(getattr(c, "dr_scale_start", 0.0))
                              if (c.dr_enable and c.dr_curriculum_steps > 0) else 1.0),
    alive_scale=1.0 if getattr(c, "alive_decay_steps", 0) > 0
               else float(getattr(c, "alive_scale_final", 1.0)),
    sprint_dist_m=float(c.sprint_dist_start_m if c.sprint_curriculum_steps > 0
                        else c.sprint_dist_m),
    stance_ratio=float(c.stance_ratio_start if c.gait_curriculum_steps > 0
                       else c.stance_ratio_final),
    eff_scale=0.0 if c.efficiency_ramp_steps > 0 else float(c.efficiency_target),
    ctrl_jitter_ms=0.0 if c.jitter_curriculum_steps > 0 else float(c.ctrl_jitter_ms_final),
    ctrl_drop_prob=0.0 if c.jitter_curriculum_steps > 0 else float(c.ctrl_drop_prob_final),
    bringup_scale=0.0 if (getattr(c, 'bringup_enable', False)
                         and c.bringup_curriculum_steps > 0) else 1.0,
    cmd_zero_p=0.0 if c.cmd_curriculum_steps > 0 else float(c.cmd_zero_frac),
    cmd_stop_p=((0.0 if c.cmd_curriculum_steps > 0 else float(getattr(c, "cmd_stop_frac", 0.0)))
                if getattr(c, "stop_flag", False) else 0.0),
    cmd_lo=float(c.cmd_range_start[0] if c.cmd_curriculum_steps > 0
                 else c.cmd_range[0]),
    cmd_hi=float(c.cmd_range_start[1] if c.cmd_curriculum_steps > 0
                 else c.cmd_range[1]),
    gait_freq_lo=float(c.gait_freq_lo_start if c.gait_freq_floor_steps > 0
                       else c.gait_freq_hz[0]),
    track_sigma=float(c.track_sigma_start if getattr(c, 'track_sigma_steps', 0) > 0
                      else c.track_sigma),
    shape_scale=float(c.shape_scale_start if getattr(c, 'shape_curriculum_steps', 0) > 0 else 1.0))


class PPO:
    def __init__(self, cfg, env: DashEnvV2, run_dir, total_steps, seed=0, eval_env=None, n_devices=1):
        self.cfg, self.env, self.run = cfg, env, Path(run_dir)
        self.total_steps = int(total_steps)
        # data parallelism: `env` is built PER DEVICE (n_envs // n_devices envs); each device runs its own
        # rollout, the minibatch gradients are averaged with lax.pmean, params stay identical everywhere
        self.n_dev = int(n_devices)
        self.devices = jax.local_devices()[:self.n_dev]
        assert len(self.devices) == self.n_dev, f"asked for {self.n_dev} devices, have {jax.local_devices()}"
        self.n_envs_dev, self.n_steps = env.n_envs, int(cfg.n_steps)
        self.n_envs = self.n_envs_dev * self.n_dev
        self.batch = self.n_envs * self.n_steps
        self.batch_dev = self.n_envs_dev * self.n_steps
        self.n_minibatches = max(1, self.batch // int(cfg.batch_size))
        self.mb_size = self.batch // self.n_minibatches
        assert self.mb_size % self.n_dev == 0 and int(cfg.est_batch) % self.n_dev == 0, "minibatch not divisible by devices"
        self.mb_dev = self.mb_size // self.n_dev
        self.est_batch_dev = max(1, min(int(cfg.est_batch) // self.n_dev, self.batch_dev))
        self.n_rollouts_total = max(1, math.ceil(self.total_steps / self.batch))
        self.eval_env = eval_env
        self.key = jax.random.PRNGKey(int(seed))
        self.net = nets.ActorCritic(n_actor=env.actor_dim, n_priv=env.priv_dim,
                                    action_dim=env.action_dim,
                                    policy_hidden=tuple(cfg.policy_hidden),
                                    est_hidden=tuple(cfg.est_hidden),
                                    init_log_std=float(cfg.max_log_std))
        self.key, k = jax.random.split(self.key)
        self.params = self.net.init(k, jnp.zeros((1, env.obs_dim)))
        n_updates = self.n_rollouts_total * cfg.n_epochs * self.n_minibatches
        # LR WARMUP, for the first minibatches only. Adam starts with zero moments, so its very
        # first step is ~lr on EVERY parameter at once (m-hat / sqrt(v-hat) ~ +-1 after bias
        # correction) -- a coordinated move of the whole network. A random policy does not care; a
        # converged one that a warm start just loaded is destroyed by it. Measured 2026-09-13, the
        # first update of every stage-3 run reported approx_kl 0.93 against a target_kl of 0.03,
        # and target_kl cannot help: the early stop is checked AFTER a minibatch, so the damage is
        # already in the weights. Warming the step size up over a few hundred minibatches lets the
        # second moment fill in before the step is allowed to be full size.
        warm_up = int(getattr(cfg, "lr_warmup_updates", 0))
        if warm_up > 0:
            self.lr = optax.join_schedules(
                [optax.linear_schedule(cfg.learning_rate * 0.02, cfg.learning_rate, warm_up),
                 optax.linear_schedule(cfg.learning_rate, cfg.lr_final,
                                       max(n_updates - warm_up, 1))],
                [warm_up])
        else:
            self.lr = optax.linear_schedule(cfg.learning_rate, cfg.lr_final, n_updates)
        self.lr_kl_adaptive = bool(getattr(cfg, "lr_kl_adaptive", False))
        if self.lr_kl_adaptive:
            # rl_games' adaptive schedule: the step shrinks when the KL overshoots the target and grows
            # when it undershoots, instead of the early stop throttling learning to one minibatch
            self.lr_now = float(cfg.learning_rate)
            self.tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                                  optax.inject_hyperparams(optax.adam)(learning_rate=self.lr_now))
        else:
            self.tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(self.lr))
        if bool(getattr(cfg, "grad_guard", False)):
            # skip an update whose gradients are not finite instead of corrupting the params
            self.tx = optax.apply_if_finite(self.tx, max_consecutive_errors=20)
        self.opt_state = self.tx.init(self.params)
        est_mask = jax.tree_util.tree_map_with_path(
            lambda path, _: any(getattr(p, "key", None) == "estimator" for p in path), self.params)
        self.est_tx = optax.masked(optax.adam(cfg.est_lr), est_mask)
        self.est_opt_state = self.est_tx.init(self.params)
        self.stats = ObsStats.init(env.obs_dim)
        self.latched = jnp.asarray(env.latched_dims)
        self.mirror = nets.ObsMirror(env)
        # schedule state
        self.step = 0                      # env steps so far
        self.rollout_n = 0
        self.ent_coef = float(cfg.ent_coef)
        self.log_std_clamp = float(cfg.max_log_std)
        self.anneal_from = None
        self.anneal_base = None
        self.streak = 0
        self.deadline, self.anneal_steps = self._size_entropy(cfg, self.total_steps)
        self.cur = {}                      # curriculum state (persisted)
        self.env_params = self._initial_params()
        self.env_state = None
        self.obs = None
        self._build()

    # ---------------------------------------------------------------- jitted pieces
    def _build(self):
        env, cfg, net = self.env, self.cfg, self.net
        latched = self.latched
        mirror = self.mirror
        library = env.library_mode

        def policy_fwd(params, nobs):
            return net.apply(params, nobs)

        def rollout(params, stats, env_state, obs, key, env_params):
            def one(carry, _):
                env_state, obs, key = carry
                key, k = jax.random.split(key)
                nobs = stats.normalize(obs)
                mu, log_std, value, _ = policy_fwd(params, nobs)
                a = nets.sample(k, mu, log_std)
                env_state, obs2, r, done, info = env.step(env_state, jnp.clip(a, -1.0, 1.0), env_params)
                mask = nets.dim_mask(info["commit"], latched)
                lp = nets.log_prob(mu, log_std, a, mask)
                metrics = dict(
                    reward_terms=info["reward_terms"], foot_air=info["foot_air"],
                    fallen=info["fallen"], finished=info["finished"], done=done,
                    ep_return=info["ep_return"], ep_len=info["ep_len"], sprint_d=info["sprint_d"],
                    t_line=info["t_line"], thermal_max=info["thermal_max"],
                    torque_util=info["torque_util"], freq_hz=info["freq_hz"], commit=info["commit"],
                    residual_sat=info["residual_sat"], resync=info["resync"],
                    track_err=info["track_err"], run_tick=info["run_tick"],
                    action=a, raw_obs=obs,
                )
                # the value of the state the episode ended in, for a time-limit truncation: the
                # auto-reset has already overwritten obs2, so this reads the env's saved final obs
                _, _, v_final, _ = policy_fwd(params, stats.normalize(info["obs_final"]))
                trunc = info["truncated"] & ~info["fallen"] & ~info["finished"]
                tr = Transition(obs=nobs, action=a, log_prob=lp, reward=r, done=done,
                                value=value, mask=mask, trunc=trunc, v_final=v_final)
                return (env_state, obs2, key), (tr, metrics)
            (env_state, obs, key), (tr, metrics) = jax.lax.scan(one, (env_state, obs, key), None,
                                                               length=self.n_steps)
            nobs_last = stats.normalize(obs)
            _, _, last_value, _ = policy_fwd(params, nobs_last)
            # raw-obs moments for the stats update
            raw = metrics.pop("raw_obs")
            flat = raw.reshape(-1, raw.shape[-1])
            b_mean, b_var = flat.mean(0), flat.var(0)
            return env_state, obs, key, tr, last_value, metrics, (b_mean, b_var, flat.shape[0])

        self._rollout = jax.jit(rollout)

        self._gae = jax.jit(gae_fn, static_argnums=(2, 3))

        def loss_fn(params, stats, obs, act, old_lp, adv, ret, mask, clip_range, ent_coef):
            mu, log_std, value, _ = policy_fwd(params, obs)
            lp = nets.log_prob(mu, log_std, act, mask)
            ratio = jnp.exp(lp - old_lp)
            adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = -jnp.mean(jnp.minimum(ratio * adv_n, jnp.clip(ratio, 1 - clip_range, 1 + clip_range) * adv_n))
            vf = jnp.mean((ret - value) ** 2)
            ent = jnp.mean(nets.entropy(log_std, mask))
            # symmetry loss on the knobs of the actor mean, mirror in raw space
            if cfg.w_sym > 0.0 and not library:
                raw = stats.denormalize(obs)
                raw_m = raw.at[..., :env.actor_dim].set(mirror(raw[..., :env.actor_dim]))
                mu_m = net.apply(params, stats.normalize(raw_m), method=net.actor_mean)
                k, km = nets.knob_slice(mu, library), nets.knob_slice(mu_m, library)
                sym = jnp.mean(jnp.sum((k + km) ** 2, axis=-1))
                # residual: r(M_o s) must equal M_r r(s) = -r[perm]  (mirrored joints, negated)
                r, rm = mu[..., gait.SPEC_DIM:], mu_m[..., gait.SPEC_DIM:]
                sym = sym + (cfg.w_sym_res / max(cfg.w_sym, 1e-9)) * jnp.mean(
                    jnp.sum((rm + r[..., jnp.asarray(gait.MIRROR_PERM)]) ** 2, axis=-1))
            else:
                sym = jnp.zeros(())
            knob = jnp.mean(jnp.sum(nets.knob_slice(mu, library) ** 2, axis=-1)) if not library else 0.0
            # action-mean bounds loss (v2c, the CPU arm's SymPPO w_bound = rl_games "bounds_loss"): the
            # Gaussian is sampled unbounded and clipped by the env, so means that drift past the box
            # all earn the same clipped sample (bang-bang spec, parked clock, saturated residual) and
            # greedy clip(mu) stops matching the trained E[clip(mu + eps)]. w * mean(relu(|mu| - soft)^2)
            over = jnp.maximum(jnp.abs(mu) - float(cfg.bound_soft), 0.0)
            bound = jnp.mean(over ** 2)
            mu_out = jnp.mean((jnp.abs(mu) > 1.0).astype(jnp.float32))
            total = (pg + cfg.vf_coef * vf - ent_coef * ent + cfg.w_sym * sym + cfg.w_knob_loss * knob
                     + cfg.w_bound * bound)
            approx_kl = jnp.mean((ratio - 1.0) - jnp.log(ratio))
            clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32))
            return total, dict(pg=pg, vf=vf, ent=ent, sym=sym, kl=approx_kl, clipfrac=clipfrac,
                               bound=bound, mu_out=mu_out)

        def update_mb(params, opt_state, stats, batch, clip_range, ent_coef, log_std_clamp):
            obs, act, old_lp, adv, ret, mask = batch
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, stats, obs, act, old_lp, adv, ret, mask, clip_range, ent_coef)
            updates, opt_state = self.tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = _clamp_log_std(params, log_std_clamp)
            return params, opt_state, loss, aux

        self._update_mb = jax.jit(update_mb)

        def est_loss(params, obs):
            est = net.apply(params, obs, method=net.estimate)
            target = obs[..., env.actor_dim:env.actor_dim + 3]
            return jnp.mean((est - target) ** 2)

        def est_update(params, est_opt_state, obs):
            loss, grads = jax.value_and_grad(est_loss)(params, obs)
            updates, est_opt_state = self.est_tx.update(grads, est_opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, est_opt_state, loss

        self._est_update = jax.jit(est_update)
        self._act_greedy = jax.jit(lambda params, nobs: net.apply(params, nobs, method=net.actor_mean))
        if self.n_dev > 1:
            devs = self.devices
            self._rollout_p = jax.pmap(rollout, axis_name="dev", in_axes=(None, None, 0, 0, 0, None), devices=devs)
            self._gae_p = jax.pmap(gae_fn, axis_name="dev", static_broadcasted_argnums=(2, 3), devices=devs)

            def update_mb_p(params, opt_state, stats, data, idx, clip_range, ent_coef, log_std_clamp):
                mb = tuple(x[idx] for x in data)
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    params, stats, *mb, clip_range, ent_coef)
                grads = jax.lax.pmean(grads, "dev")
                aux = jax.lax.pmean(aux, "dev")
                loss = jax.lax.pmean(loss, "dev")
                updates, opt_state = self.tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                params = _clamp_log_std(params, log_std_clamp)
                return params, opt_state, loss, aux

            self._update_mb_p = jax.pmap(update_mb_p, axis_name="dev",
                                         in_axes=(0, 0, None, 0, 0, None, None, None), devices=devs)

            def est_update_p(params, est_opt_state, obs, idx):
                loss, grads = jax.value_and_grad(est_loss)(params, obs[idx])
                grads = jax.lax.pmean(grads, "dev")
                loss = jax.lax.pmean(loss, "dev")
                updates, est_opt_state = self.est_tx.update(grads, est_opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, est_opt_state, loss

            self._est_update_p = jax.pmap(est_update_p, axis_name="dev", in_axes=(0, 0, 0, 0), devices=devs)

            # one launch per EPOCH: lax.scan over the minibatches inside the pmap (the per-minibatch
            # pmap dispatch + all-reduce was ~18 ms a call, 0.3-1.2 s per iteration at 4 GPUs)
            n_mb_epoch, mb_dev = self.n_minibatches, self.mb_dev
            target_kl = float(cfg.target_kl)

            def update_epoch_p(params, opt_state, stats, data, perm, clip_range, ent_coef, log_std_clamp):
                def body(carry, i):
                    params, opt_state, stopped = carry
                    idx = jax.lax.dynamic_slice(perm, (i * mb_dev,), (mb_dev,))
                    mb = tuple(x[idx] for x in data)
                    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                        params, stats, *mb, clip_range, ent_coef)
                    grads = jax.lax.pmean(grads, "dev")
                    aux = jax.lax.pmean(aux, "dev")
                    updates, opt_new = self.tx.update(grads, opt_state, params)
                    p_new = _clamp_log_std(optax.apply_updates(params, updates), log_std_clamp)
                    # KL early stop (the CPU arm's target_kl rule): once tripped, later minibatches are no-ops
                    keep = jnp.logical_not(stopped)
                    params = jax.tree_util.tree_map(lambda a, b: jnp.where(keep, a, b), p_new, params)
                    opt_state = jax.tree_util.tree_map(lambda a, b: jnp.where(keep, a, b), opt_new, opt_state)
                    tripped = (aux["kl"] > 1.5 * target_kl) if target_kl > 0 else jnp.zeros((), bool)
                    stopped = stopped | tripped
                    aux = {**aux, "applied": keep.astype(jnp.float32)}
                    return (params, opt_state, stopped), aux

                (params, opt_state, stopped), auxs = jax.lax.scan(
                    body, (params, opt_state, jnp.zeros((), bool)), jnp.arange(n_mb_epoch))
                return params, opt_state, stopped, auxs

            self._update_epoch_p = jax.pmap(update_epoch_p, axis_name="dev",
                                            in_axes=(0, 0, None, 0, 0, None, None, None), devices=devs)
            n_est_epoch, est_dev = max(1, self.batch_dev // self.est_batch_dev), self.est_batch_dev

            def est_epoch_p(params, est_opt_state, obs, perm):
                def body(carry, i):
                    params, est_opt_state = carry
                    idx = jax.lax.dynamic_slice(perm, (i * est_dev,), (est_dev,))
                    loss, grads = jax.value_and_grad(est_loss)(params, obs[idx])
                    grads = jax.lax.pmean(grads, "dev")
                    updates, est_opt_state = self.est_tx.update(grads, est_opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return (params, est_opt_state), jax.lax.pmean(loss, "dev")

                (params, est_opt_state), losses = jax.lax.scan(body, (params, est_opt_state), jnp.arange(n_est_epoch))
                return params, est_opt_state, losses[-1]

            self._est_epoch_p = jax.pmap(est_epoch_p, axis_name="dev", in_axes=(0, 0, 0, 0), devices=devs)
            self._reset_p = jax.pmap(lambda k, prm: env.reset(k, prm), in_axes=(0, None), devices=devs)

    def _replicate(self, tree):
        """A leading device axis for pmap, placed with the sharding pmap expects (a committed single-
        device array is refused by pmap in jax 0.11; jax.device_put_replicated is gone)."""
        if not hasattr(self, "_rep_sharding"):
            from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
            self._rep_sharding = NamedSharding(Mesh(np.array(self.devices), ("dev",)), P("dev"))
        sh = self._rep_sharding
        return jax.tree_util.tree_map(
            lambda x: jax.device_put(np.broadcast_to(np.asarray(x), (self.n_dev,) + np.shape(x)), sh), tree)

    def reset_envs(self, key):
        """Reset every env (all devices); env_state/obs carry a leading device axis when n_dev > 1."""
        if self.n_dev == 1:
            self.env_state, self.obs = self.env.reset(key, self.env_params)
        else:
            self.env_state, self.obs = self._reset_p(jax.random.split(key, self.n_dev), self.env_params)
        return self.env_state, self.obs

    # ---------------------------------------------------------------- schedules
    @staticmethod
    def _size_entropy(cfg, budget):
        deadline, anneal = int(cfg.ent_anneal_deadline_steps), int(cfg.ent_anneal_steps)
        if not cfg.ent_schedule_autoscale or not budget:
            return deadline, anneal
        new_deadline = deadline
        if deadline > 0 and deadline >= budget:
            new_deadline = max(1, int(0.40 * budget))
        new_anneal = min(anneal, max(1, int(0.50 * budget))) if anneal > 0 else anneal
        if (new_deadline, new_anneal) != (deadline, anneal):
            print(f"[ppo] entropy schedule rescaled to the {budget:,}-step budget: deadline "
                  f"{deadline:,} -> {new_deadline:,}, anneal {anneal:,} -> {new_anneal:,}")
        return new_deadline, new_anneal

    def _initial_params(self):
        return initial_params(self.cfg)

    def _gated(self, key, ep_len, start, target, warmup, gate, retreat, d_steps):
        """Competence-gated MONOTONE ramp with hysteresis: progress in [0, 1], never decreasing.

        This used to retreat (progress -= step below retreat * gate), and retreating ramps thrash: a
        group that falls back below 0.99 becomes "unfinished", so the queue hands the turn BACK to it
        and takes it away from whatever had started. Measured on v4_one_s0, the last 250 M steps were
        the command group and the DR group swapping the turn every ~30 M, each undoing the other --
        with no convergence guarantee anywhere in it, because nothing in the loop is monotone.

        A difficulty the policy has already absorbed is not un-absorbed by a bad rollout, so progress
        only ever rises. What competence controls is the RATE: advancing while the policy is at its
        own recent best, and holding still (not reversing) when it is not. The two bars are a Schmitt
        trigger -- advance until ep_len falls below `retreat * gate`, then require the full `gate`
        again to resume -- so noise around one threshold cannot chatter the schedule."""
        st = self.cur.setdefault(key, {"streak": 0, "open": False, "progress": 0.0, "adv": False})
        if gate <= 0:
            st["progress"] = min(1.0, st["progress"] + d_steps / max(warmup, 1))
        else:
            if not st["open"]:
                st["streak"] = st["streak"] + 1 if ep_len > gate else 0
                if st["streak"] >= 5:
                    st["open"], st["adv"] = True, True
                    print(f"[ppo] curriculum gate '{key}' opened at {self.step:,} steps (ep_len {ep_len:.0f})")
            if st["open"]:
                lo_bar = retreat * gate if retreat > 0 else gate
                st["adv"] = ep_len >= (lo_bar if st.get("adv", True) else gate)
                if st["adv"]:
                    st["progress"] = min(1.0, st["progress"] + d_steps / max(warmup, 1))
        return start + st["progress"] * (target - start)

    def _clock(self, start, target, warmup):
        frac = 1.0 if warmup <= 0 else min(1.0, self.step / warmup)
        return start + frac * (target - start)

    def _clock_q(self, key, start, target, warmup, d_steps):
        """A clock ramp that only ticks while it is this curriculum's turn.

        `_clock` reads self.step, so a queued curriculum on a clock advances anyway -- which is
        exactly what happened the first time the queue ran: `shape_scale` climbed to 0.45 while the
        queue reported it was still advancing `cmd_lo`, because shape_curriculum_gated defaults to
        False and the clock branch never saw the queue. Accumulate queued steps instead."""
        st = self.cur.setdefault(key, {"streak": 0, "open": True, "progress": 0.0})
        if warmup > 0:
            st["progress"] = float(min(1.0, st["progress"] + d_steps / warmup))
        else:
            st["progress"] = 1.0
        return start + st["progress"] * (target - start)

    def _gate_ref(self, ep_len):
        """The yardstick a RELATIVE gate measures against: the best episode length this run has
        reached lately, decayed so an old peak is eventually forgotten."""
        st = self.cur.setdefault("_ep_len_ref", {"v": 0.0})
        st["v"] = float(max(ep_len, st["v"] * self.cfg.curriculum_gate_ref_decay))
        return st["v"]

    def _eff_gate(self, gate, ref):
        """Turn a configured gate into the one actually applied.

        An ABSOLUTE gate is a guess about what the finished policy will manage, made before it
        exists, and if the guess is high every curriculum silently never starts -- measured on this
        lineage, `dr_scale` finished at 0.000 in three separate runs (including the one the project
        called 'robust-trained') because a 1200-tick gate was set on a task that lives at 300-900.
        Nothing in the logs said so; the run simply trained on the nominal plant for 215 M steps.

        RELATIVE mode asks instead for a fraction of what this policy has actually achieved, so the
        curriculum advances whenever the policy is near its own best and retreats when it falls off
        -- it cannot deadlock, and it cannot run away either. The configured gates keep their
        relative ORDER (a curriculum gated later stays gated later), scaled onto the reference."""
        if self.cfg.curriculum_gate_mode != "relative" or gate <= 0:
            return gate
        c = self.cfg
        rel = gate / max(c.curriculum_gate_ep_len, 1e-9)
        return max(c.curriculum_gate_floor, rel * c.curriculum_gate_frac * ref)

    def _queued(self, key):
        """In SEQUENTIAL mode, has this curriculum's turn arrived yet?

        Every curriculum in v3 advances off the same competence gate, which means the task hardens
        in six directions at once: the command band widens, the gait-quality penalties come on, the
        starts get dirty, the plant is randomised, the controller gets jittery, and the assist
        fades. Measured 2026-09-13 across a dozen runs, every one of them follows the same arc --
        climb to a peak, then decline from the point where the curricula start biting together.

        v2 had the opposite failure: absolute gates set so high that nothing ever advanced. The
        answer is neither, it is ONE AT A TIME. `curriculum_order` names the sequence; a curriculum
        may advance only once every curriculum before it has reached 1.0, so the policy is asked to
        absorb one new difficulty at a time and keeps whatever it has already learned."""
        order = self.cfg.curriculum_order
        if not order:
            return True
        # An entry may be a NAME or a TUPLE of names that advance together. cmd_lo, cmd_hi and
        # cmd_zero_p are one curriculum wearing three names -- queued separately they cost three
        # ramps (75 M of a 100 M budget) and the assist fade never got its turn.
        groups = [(g,) if isinstance(g, str) else tuple(g) for g in order]
        pos = next((i for i, g in enumerate(groups) if key in g), None)
        if pos is None:
            return True
        # EXACTLY ONE GROUP ADVANCES. Not "every earlier group is cleared" -- that let a capped
        # group keep ramping underneath its successor, which is the overlap this queue exists to
        # prevent (see below).
        return pos == self._live_group(groups)

    def _live_group(self, groups):
        """Index of the group whose turn it is: the first one not yet CLEARED.

        A group is cleared when it has finished (progress >= 0.99) or when it has held the queue
        for `curriculum_group_max_steps` without finishing.

        THE QUEUE DOES NOT WAIT FOREVER. Every ramp here is competence-gated and retreats, so a
        group can hover below 0.99 indefinitely and starve everything behind it. That is not
        hypothetical: all five 200 M stage-2 seeds finished with `dr_scale` at 0.000 because the
        three groups ahead of it never all completed. The cap bounds how long one group may hold
        the queue -- it keeps whatever progress it has and the next group starts anyway.

        ...AND THE CAPPED GROUP THEN STOPS. That is the 2026-09-17 fix. Previously "cleared" only
        decided who could START; the capped group itself stayed eligible and went on ramping, so
        from the moment the cap fired TWO curricula advanced at once. Measured on dash_s0/s1/s2:
        the shape/eff/stance group held the queue for its full 80 M with `eff_scale` at 0.68, the
        cap let `dr_scale` in, and the policy then had to absorb a rising effort penalty AND a
        randomising plant together -- the exact "task hardens in several directions at once"
        failure `_queued` was written to prevent, reintroduced by its own starvation valve. All
        three seeds peaked within 2 M steps of that moment and collapsed from ep_len 2376 to 45.

        Freezing the capped group keeps both properties: nothing starves, and the policy is still
        asked for one new difficulty at a time. A partly-open curriculum that the policy has
        absorbed beats a fully-open one it never reached.
        """
        cap = float(getattr(self.cfg, "curriculum_group_max_steps", 0.0))

        def cleared(g):
            for k in g:
                st = self.cur.get(k)
                if st is None:                       # never started: holds the queue
                    return False
                if st.get("progress", 0.0) >= 0.99:
                    continue
                if cap > 0.0 and st.get("turn", 0.0) >= cap:
                    continue
                return False
            return True

        live = next((i for i, g in enumerate(groups) if not cleared(g)), None)
        if live is not None:
            return live
        # Nothing is waiting: give the turn BACK to the first group the cap cut short. A group
        # frozen at 0.68 is a curriculum that never reached its target, and once no one else needs
        # the queue there is no reason to leave the rest of the budget unspent -- with the ramps
        # done by ~250 M of a 450 M run, that tail is 200 M steps. Still exactly one at a time.
        unfinished = next((i for i, g in enumerate(groups)
                           if any(self.cur.get(k, {}).get("progress", 0.0) < 0.99 for k in g)), None)
        return unfinished if unfinished is not None else len(groups)

    def update_curricula(self, ep_len, d_steps):
        c = self.cfg
        p = self.env_params
        ref = self._gate_ref(ep_len)
        _g = lambda g: self._eff_gate(g, ref)
        gate, rf = _g(c.curriculum_gate_ep_len), c.curriculum_retreat_frac
        # a queued curriculum is frozen at its current value: gate 0 with a zero step. The steps a
        # curriculum is ALLOWED to advance by are also the steps it has held the queue, so the turn
        # clock that `_queued` reads is accumulated here and nowhere else.
        def _q(key, steps):
            if not self._queued(key):
                return 0
            st = self.cur.setdefault(key, {"streak": 0, "open": False, "progress": 0.0})
            st["turn"] = st.get("turn", 0.0) + steps
            return steps
        kw = dict(p._asdict())
        if c.dr_enable and c.dr_curriculum_steps > 0:
            kw["dr_scale"] = self._gated("dr_scale", ep_len, float(getattr(c, "dr_scale_start", 0.0)),
                                         float(getattr(c, "dr_scale_final", 1.0)),
                                         c.dr_curriculum_steps, gate, rf, _q("dr_scale", d_steps))
        if getattr(c, "alive_decay_steps", 0) > 0:
            # offset clock: hold full weight until the policy can balance, then wean over a window.
            f = (self.step - c.alive_decay_start_steps) / max(c.alive_decay_steps, 1)
            f = min(1.0, max(0.0, f))
            kw["alive_scale"] = 1.0 + f * (float(c.alive_scale_final) - 1.0)
        if c.objective == "sprint" and c.sprint_curriculum_steps > 0:
            kw["sprint_dist_m"] = self._clock(c.sprint_dist_start_m, c.sprint_dist_m, c.sprint_curriculum_steps)
        if c.gait_curriculum_steps > 0 and c.w_phase_contact > 0:
            # _q, not d_steps: stance_ratio is named in curriculum_order, so it must take its turn
            # and bank a turn clock like its group-mates. Passing d_steps straight through let it
            # ramp while another group held the queue AND left its `turn` at 0, so the cap could
            # never excuse it -- it was the member of its group that decided when DR was let in.
            kw["stance_ratio"] = self._gated("stance_ratio", ep_len, c.stance_ratio_start, c.stance_ratio_final,
                                             c.gait_curriculum_steps, _g(c.gait_curriculum_gate_ep_len), rf,
                                             _q("stance_ratio", d_steps))
        if c.efficiency_ramp_steps > 0:
            kw["eff_scale"] = self._gated("eff_scale", ep_len, 0.0, c.efficiency_target,
                                          c.efficiency_ramp_steps, _g(c.efficiency_gate_ep_len), rf, _q("eff_scale", d_steps))
        if c.jitter_curriculum_steps > 0:
            jg = _g(c.jitter_curriculum_gate_ep_len)
            kw["ctrl_jitter_ms"] = self._gated("ctrl_jitter_ms", ep_len, 0.0, c.ctrl_jitter_ms_final,
                                               c.jitter_curriculum_steps, jg, rf, _q("ctrl_jitter_ms", d_steps))
            kw["ctrl_drop_prob"] = self._gated("ctrl_drop_prob", ep_len, 0.0, c.ctrl_drop_prob_final,
                                               c.jitter_curriculum_steps, jg, rf, _q("ctrl_drop_prob", d_steps))
        if getattr(c, "shape_curriculum_steps", 0) > 0:
            # CLOCK by default, gate optional. The gated variant was tried first, on the theory
            # that a clock ramp kills runs on arrival -- training return did fall from 1197 to -18
            # as the clock reached 1.0. That reading was WRONG, and wrong in a way this project has
            # a rule about: return is not comparable across shape_scale values, because raising the
            # penalties lowers it by construction. Measured on the greedy ladder over the same
            # window, the run whose return "collapsed" went from 0% to 60% upright and from 1.93 to
            # 0.59 m/s of command error. It was improving the whole time.
            #
            # Judged on the greedy ladder instead: at 42-65 M the clock runs reached 32-60% upright
            # and 0.59-0.78 m/s error, the gated runs 0-2% and 1.68-1.92, with the gate holding
            # shape_scale at 0.50 and not advancing. So the clock ships. The gate stays available
            # because it is the right instrument if a plant ever cannot pay the full bill -- it
            # settles at the highest weight the policy can afford rather than insisting on 1.0.
            kw["shape_scale"] = (
                self._gated("shape_scale", ep_len, c.shape_scale_start, 1.0,
                            c.shape_curriculum_steps, gate, rf, _q("shape_scale", d_steps))
                if c.shape_curriculum_gated else
                self._clock_q("shape_scale", c.shape_scale_start, 1.0, c.shape_curriculum_steps,
                              _q("shape_scale", d_steps)))
        if c.objective == "joystick" and getattr(c, "track_sigma_steps", 0) > 0:
            kw["track_sigma"] = self._clock(c.track_sigma_start, c.track_sigma, c.track_sigma_steps)
        if getattr(c, "gait_freq_floor_steps", 0) > 0:
            kw["gait_freq_lo"] = self._clock(c.gait_freq_lo_start, float(c.gait_freq_hz[0]),
                                             c.gait_freq_floor_steps)
        if getattr(c, "bringup_enable", False) and c.bringup_curriculum_steps > 0:
            # open the drop height and the release tilt as competence is earned: the measured envelope
            # today is +-5 deg, and the target is +-20, so starting wide would begin most episodes lost
            kw["bringup_scale"] = self._gated("bringup_scale", ep_len, 0.0,
                                              float(getattr(c, "bringup_target", 1.0)),
                                              c.bringup_curriculum_steps, _g(c.bringup_gate_ep_len), rf, _q("bringup_scale", d_steps))
        if c.objective == "joystick" and c.cmd_curriculum_steps > 0:
            # widen the command band DOWNWARD from what the warm start already does. Opening it to
            # [0, 1] at step 0 would spend most episodes asking a runner for speeds it has never
            # produced, which is how the v2 stop runs burned their budget.
            # PACED by tracking (cfg.cmd_gate_err): full rate while the policy tracks the band it has,
            # a fraction of it otherwise. The turn clock still counts real steps; only progress slows.
            _cm = 1.0
            _ge = float(getattr(c, "cmd_gate_err", 0.0))
            if _ge > 0.0:
                _te = getattr(self, "_trk_err_ema", None)
                _cm = 1.0 if (_te is not None and _te < _ge) else float(c.cmd_slow_rate)
            kw["cmd_lo"] = self._gated("cmd_lo", ep_len, float(c.cmd_range_start[0]), float(c.cmd_range[0]),
                                       c.cmd_curriculum_steps, _g(c.cmd_gate_ep_len), rf, _cm * _q("cmd_lo", d_steps))
            kw["cmd_hi"] = self._gated("cmd_hi", ep_len, float(c.cmd_range_start[1]), float(c.cmd_range[1]),
                                       c.cmd_curriculum_steps, _g(c.cmd_gate_ep_len), rf, _cm * _q("cmd_hi", d_steps))
            if getattr(c, "stop_flag", False):
                kw["cmd_stop_p"] = self._gated("cmd_stop_p", ep_len, 0.0, float(c.cmd_stop_frac),
                                               c.cmd_curriculum_steps, _g(c.cmd_gate_ep_len), rf,
                                               _cm * _q("cmd_stop_p", d_steps))
            # and the zero share with it -- "stop" is the hardest command this lineage has, so it
            # arrives last, not alongside the first rollout
            kw["cmd_zero_p"] = self._gated("cmd_zero_p", ep_len, 0.0, float(c.cmd_zero_frac),
                                           c.cmd_curriculum_steps, _g(c.cmd_gate_ep_len), rf, _q("cmd_zero_p", d_steps))
        if getattr(c, "stoplight_prob_final", 0.0) > 0 and c.stoplight_curriculum_steps > 0:
            kw["stoplight_prob"] = self._gated("stoplight_prob", ep_len, 0.0, c.stoplight_prob_final,
                                               c.stoplight_curriculum_steps, _g(c.stoplight_gate_ep_len), rf, d_steps)
        if self.cfg.curriculum_order:
            groups = [(g,) if isinstance(g, str) else tuple(g) for g in self.cfg.curriculum_order]
            # the SAME predicate _queued uses, cap included. Reporting "done" by progress alone
            # announced dr_scale at 152 M in every 2026-09-17 seed when it had been ramping since
            # 119 M, which hid the overlap that ended those runs.
            idx = self._live_group(groups)
            live = groups[idx] if idx < len(groups) else None
            if live != getattr(self, "_live_curriculum", "<none>"):
                self._live_curriculum = live
                n_done = idx
                name = "+".join(live) if live else "nothing left"
                # say WHY the previous group handed over: finished, or ran out of turn. A group
                # that timed out leaves a partly-open curriculum behind and that is worth seeing in
                # the log rather than inferring from a sidecar afterwards.
                timed = [k for g in groups for k in g
                         if self.cur.get(k, {}).get("progress", 0.0) < 0.99
                         and self.cur.get(k, {}).get("turn", 0.0) > 0
                         and (live is None or k not in live)]
                why = (f" -- {'+'.join(timed)} handed over at "
                       f"{self.cur[timed[0]].get('progress', 0.0):.2f}, out of turn" if timed else "")
                print(f"[ppo] curriculum queue at {self.step:,}: now advancing {name}"
                      f" ({n_done}/{len(groups)} complete){why}")
        self.env_params = EnvParams(**{k: float(v) for k, v in kw.items()})

    def update_entropy(self, swing_min, ep_len=None):
        c = self.cfg
        if self.anneal_from is None:
            eg = float(getattr(c, "ent_gate_ep_len", 0.0))
            competent = swing_min > c.ent_gate_swing_frac and (eg <= 0 or (ep_len is not None and ep_len > eg))
            self.streak = self.streak + 1 if competent else 0
            gate_open = self.streak >= 5
            deadline_hit = self.deadline > 0 and self.step >= self.deadline
            if gate_open or deadline_hit:
                self.anneal_from = self.step
                self.anneal_base = float(self.ent_coef)
                print(f"[ppo] entropy anneal opened via {'competence gate' if gate_open else 'hard deadline'} "
                      f"(swing_min={swing_min:.3f}) at {self.step:,} steps")
        if self.anneal_from is not None:
            frac = min(1.0, (self.step - self.anneal_from) / max(self.anneal_steps, 1))
            self.ent_coef = self.anneal_base + frac * (c.ent_final - self.anneal_base)
            if c.std_anneal_target > 0:
                self.log_std_clamp = c.max_log_std + frac * (math.log(c.std_anneal_target) - c.max_log_std)

    # ---------------------------------------------------------------- one iteration
    def iterate(self):
        cfg = self.cfg
        t0 = time.time()
        if self.n_dev == 1:
            self.key, k = jax.random.split(self.key)
            (self.env_state, self.obs, _, tr, last_value, metrics,
             (b_mean, b_var, b_n)) = self._rollout(self.params, self.stats, self.env_state, self.obs, k,
                                                  self.env_params)
            jax.block_until_ready(tr.reward)          # async dispatch: stamp the phases honestly
            t_roll = time.time() - t0
            t1 = time.time()
            adv, ret = self._gae(tr, last_value, float(cfg.gamma), float(cfg.gae_lambda))
            jax.block_until_ready(ret)
            t_gae = time.time() - t1
            t1 = time.time()
            # ---- PPO update
            stats_used = self.stats
            flat = lambda x: x.reshape((self.batch,) + x.shape[2:])
            data = (flat(tr.obs), flat(tr.action), flat(tr.log_prob), flat(adv), flat(ret), flat(tr.mask))
            aux_acc, n_mb, stop = {}, 0, False
            for epoch in range(cfg.n_epochs):
                self.key, k = jax.random.split(self.key)
                perm = jax.random.permutation(k, self.batch)
                for i in range(self.n_minibatches):
                    idx = perm[i * self.mb_size:(i + 1) * self.mb_size]
                    mb = tuple(x[idx] for x in data)
                    self.params, self.opt_state, loss, aux = self._update_mb(
                        self.params, self.opt_state, stats_used, mb, float(cfg.clip_range),
                        float(self.ent_coef), float(self.log_std_clamp))
                    n_mb += 1
                    for kk, v in aux.items():
                        aux_acc[kk] = aux_acc.get(kk, 0.0) + float(v)
                    if cfg.target_kl > 0 and float(aux["kl"]) > 1.5 * cfg.target_kl:
                        stop = True
                        break
                if stop:
                    break
            jax.block_until_ready(self.params)
            t_update = time.time() - t1
            t1 = time.time()
            # ---- estimator (supervised, own optimizer, estimator subtree only)
            est_last = None
            for _ in range(cfg.est_epochs):
                self.key, k = jax.random.split(self.key)
                perm = jax.random.permutation(k, self.batch)
                for i in range(max(1, self.batch // cfg.est_batch)):
                    idx = perm[i * cfg.est_batch:(i + 1) * cfg.est_batch]
                    self.params, self.est_opt_state, est_last = self._est_update(self.params, self.est_opt_state,
                                                                                 data[0][idx])
            jax.block_until_ready(self.params)
            t_est = time.time() - t1
            t1 = time.time()
        else:
            self.key, k = jax.random.split(self.key)
            keys = jax.random.split(k, self.n_dev)
            (self.env_state, self.obs, _, tr, last_value, metrics,
             (b_mean_d, b_var_d, b_n_d)) = self._rollout_p(self.params, self.stats, self.env_state, self.obs, keys,
                                                        self.env_params)
            jax.block_until_ready(tr.reward)
            t_roll = time.time() - t0
            t1 = time.time()
            adv, ret = self._gae_p(tr, last_value, float(cfg.gamma), float(cfg.gae_lambda))
            jax.block_until_ready(ret)
            t_gae = time.time() - t1
            t1 = time.time()
            stats_used = self.stats
            flat = lambda x: x.reshape((self.n_dev, self.batch_dev) + x.shape[3:])     # (dev, T, N, ..) -> (dev, T*N, ..)
            data = (flat(tr.obs), flat(tr.action), flat(tr.log_prob), flat(adv), flat(ret), flat(tr.mask))
            p_rep = self._replicate(self.params)
            opt_rep = self._replicate(self.opt_state)
            aux_acc, n_mb, stop = {}, 0, False
            for epoch in range(cfg.n_epochs):
                self.key, k = jax.random.split(self.key)
                perm = jnp.stack([jax.random.permutation(kd, self.batch_dev) for kd in jax.random.split(k, self.n_dev)])
                p_rep, opt_rep, stopped, auxs = self._update_epoch_p(
                    p_rep, opt_rep, stats_used, data, perm, float(cfg.clip_range),
                    float(self.ent_coef), float(self.log_std_clamp))
                auxs = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), auxs)     # (n_mb,) per key
                applied = auxs.pop("applied").astype(bool)
                n_app = int(applied.sum())
                n_mb += n_app
                for kk, v in auxs.items():
                    aux_acc[kk] = aux_acc.get(kk, 0.0) + float(v[applied].sum()) if n_app else aux_acc.get(kk, 0.0)
                if bool(np.asarray(stopped)[0]):
                    stop = True
                    break
            jax.block_until_ready(p_rep)
            t_update = time.time() - t1
            t1 = time.time()
            est_rep = self._replicate(self.est_opt_state)
            est_last = None
            for _ in range(cfg.est_epochs):
                self.key, k = jax.random.split(self.key)
                perm = jnp.stack([jax.random.permutation(kd, self.batch_dev) for kd in jax.random.split(k, self.n_dev)])
                p_rep, est_rep, est_l = self._est_epoch_p(p_rep, est_rep, data[0], perm)
                est_last = est_l[0]
            unrep = lambda t: jax.tree_util.tree_map(lambda x: x[0], t)
            self.params, self.opt_state, self.est_opt_state = unrep(p_rep), unrep(opt_rep), unrep(est_rep)
            jax.block_until_ready(self.params)
            t_est = time.time() - t1
            t1 = time.time()
            # pooled moments of the raw observations across devices
            n_d = np.asarray(b_n_d, np.float64)
            mu_d, var_d = np.asarray(b_mean_d, np.float64), np.asarray(b_var_d, np.float64)
            b_n = float(n_d.sum())
            b_mean = (n_d[:, None] * mu_d).sum(0) / b_n
            b_var = (n_d[:, None] * (var_d + mu_d ** 2)).sum(0) / b_n - b_mean ** 2
            b_mean, b_var = jnp.asarray(b_mean, jnp.float32), jnp.asarray(np.maximum(b_var, 0.0), jnp.float32)
            # metrics: (dev, T, N, ...) -> (T, dev*N, ...)
            metrics = jax.tree_util.tree_map(
                lambda x: np.moveaxis(np.asarray(x), 0, 1).reshape((x.shape[1], x.shape[0] * x.shape[2]) + x.shape[3:]),
                metrics)
        if self.lr_kl_adaptive and n_mb > 0 and cfg.target_kl > 0:
            kl_mean = aux_acc.get("kl", 0.0) / n_mb
            if kl_mean > 2.0 * cfg.target_kl:
                self.lr_now = max(self.lr_now / 1.5, float(cfg.lr_kl_min))
            elif kl_mean < 0.5 * cfg.target_kl:
                self.lr_now = min(self.lr_now * 1.5, float(cfg.lr_kl_max))
            self.opt_state = _set_lr(self.opt_state, self.lr_now)
        # ---- obs stats (after the update, from the rollout's raw observations)
        self.stats = self.stats.update(b_mean, b_var, float(b_n))
        if getattr(cfg, "sym_obs_stats", True) and not self.env.library_mode:
            self.stats = self._symmetrize_stats(self.stats)
        d_steps = self.batch
        self.step += d_steps
        self.rollout_n += 1
        # ---- metrics
        m = jax.tree_util.tree_map(lambda x: np.asarray(x), metrics)
        done = m["done"].astype(bool)
        n_done = int(done.sum())
        ep_len = float(m["ep_len"][done].mean()) if n_done else float(m["ep_len"].mean())
        ep_ret = float(m["ep_return"][done].mean()) if n_done else float("nan")
        swing_min = float(m["foot_air"].reshape(-1, 2).mean(0).min())
        # mean |v - v_target| over RUN ticks: what paces the command band (cfg.cmd_gate_err)
        _run = np.asarray(m["run_tick"], np.float64)
        # NaN-safe and bounded: a few envs per hundred rollouts blow up numerically (track_err 1e7 or NaN),
        # and one NaN made this EMA NaN for the rest of every dash_joy2 run -> the band never left the
        # slow rate. A blown-up tick counts as a 5 m/s miss, not as infinity.
        _te = np.clip(np.nan_to_num(np.asarray(m["track_err"], np.float64), nan=5.0, posinf=5.0, neginf=5.0), 0.0, 5.0)
        _run = np.nan_to_num(_run, nan=0.0)
        trk_err = float((_te * _run).sum() / max(_run.sum(), 1.0))
        _prev = getattr(self, "_trk_err_ema", None)
        self._trk_err_ema = trk_err if _prev is None else 0.95 * _prev + 0.05 * trk_err
        finishes = int((m["finished"] & done).sum())
        falls = int((m["fallen"] & done).sum())
        t_lines = m["t_line"][m["finished"] & done]
        commit = m["commit"].astype(bool)
        acts = m["action"]
        freq_raw = acts[..., gait.I_FREQ] if not self.env.library_mode else np.zeros(1)
        f_at_commit = freq_raw[commit] if commit.any() else freq_raw.reshape(-1)
        res = acts[..., gait.SPEC_DIM:] if not self.env.library_mode else acts[..., :6]
        log = {
            "time/env_steps": self.step, "time/rollout_s": t_roll, "time/gae_s": t_gae,
            "time/update_s": t_update, "time/est_s": t_est, "time/iter_s": time.time() - t0,
            "time/sps": d_steps / max(time.time() - t0, 1e-9),
            "rollout/ep_len_mean": ep_len, "rollout/ep_ret_mean": ep_ret, "rollout/episodes": n_done,
            "rollout/track_err": trk_err, "rollout/track_err_ema": self._trk_err_ema,
            "rollout/finishes": finishes, "rollout/falls": falls,
            "rollout/t_line_mean": float(t_lines.mean()) if t_lines.size else float("nan"),
            "rollout/sprint_d_mean": float(m["sprint_d"].mean()),
            "rollout/reward_mean": float(np.asarray(tr.reward).mean()),
            "rollout/swing_frac_min": swing_min,
            "rollout/thermal_max": float(m["thermal_max"].max()),
            "rollout/torque_util": float(m["torque_util"].mean()),
            "rollout/commit_frac": float(commit.mean()),
            "rollout/resync_frac": float(m["resync"].mean()),
            "diag/freq_hz_median": float(np.median(m["freq_hz"])),
            "diag/freq_lo_rail": float(np.mean(np.clip(f_at_commit, -1, 1) <= -0.98)),
            "diag/freq_hi_rail": float(np.mean(np.clip(f_at_commit, -1, 1) >= 0.98)),
            "diag/res_sat_sampled": float(np.mean(np.abs(np.clip(res, -1, 1)) >= 0.95)),
            "train/loss_pg": aux_acc.get("pg", 0.0) / max(n_mb, 1),
            "train/loss_vf": aux_acc.get("vf", 0.0) / max(n_mb, 1),
            "train/entropy": aux_acc.get("ent", 0.0) / max(n_mb, 1),
            "train/loss_sym": aux_acc.get("sym", 0.0) / max(n_mb, 1),
            "train/bound_loss": aux_acc.get("bound", 0.0) / max(n_mb, 1),
            "train/mu_out_frac": aux_acc.get("mu_out", 0.0) / max(n_mb, 1),
            "train/approx_kl": aux_acc.get("kl", 0.0) / max(n_mb, 1),
            "train/clipfrac": aux_acc.get("clipfrac", 0.0) / max(n_mb, 1),
            "train/n_minibatch_updates": n_mb, "train/early_stop": float(stop),
            "train/ent_coef": self.ent_coef, "train/log_std_clamp": self.log_std_clamp,
            "train/std_mean": float(np.exp(np.asarray(self.params["params"]["log_std"])).mean()),
            "train/lr": float(self.lr_now) if self.lr_kl_adaptive else float(self.lr(_adam_count(self.opt_state))),
            "est/vel_rmse": float(np.sqrt(float(est_last))) if est_last is not None else float("nan"),
        }
        for kk, v in m["reward_terms"].items():
            log[f"reward_terms/{kk}"] = float(np.asarray(v).mean())
        for kk, v in self.env_params._asdict().items():
            log[f"curriculum/{kk}"] = float(v)
        # ---- schedules for the NEXT rollout
        self.update_entropy(swing_min, ep_len)
        self.update_curricula(ep_len, d_steps)
        return log

    def _symmetrize_stats(self, stats):
        """Mirror the running stats: mean -> 0.5 (m + M m), var -> 0.5 (v + |M| v), so early
        L/R drift in the normalizer cannot bias the symmetric gait (§13)."""
        m = np.asarray(stats.mean)
        v = np.asarray(stats.var)
        a = self.env.actor_dim
        F = env_mod.FRAME_DIM
        mm = np.array(self.mirror(jnp.asarray(m[None, :a]))[0])
        # the phase channels of the mirror depend on Delta (a mean of ~0 anyway): leave them
        H = self.env.cfg.history_len
        for i in range(H):
            mm[i * F + 25:i * F + 27] = m[i * F + 25:i * F + 27]
        vm = np.abs(np.array(self.mirror(jnp.asarray(v[None, :a]))[0]))
        for i in range(H):
            vm[i * F + 25:i * F + 27] = v[i * F + 25:i * F + 27]
        m2, v2 = m.copy(), v.copy()
        m2[:a] = 0.5 * (m[:a] + mm)
        v2[:a] = 0.5 * (v[:a] + vm)
        return ObsStats(mean=jnp.asarray(m2), var=jnp.asarray(v2), count=stats.count)

    # ---------------------------------------------------------------- greedy eval
    def evaluate(self, n_max_steps=None, override=None, seed=1000, params=None, ladder=None,
                 warm_ticks=None):
        """Greedy rollout on the eval env (nominal plant): survival, command tracking, heading.

        Under the joystick objective this is scored on a fixed COMMAND LADDER (one stick position
        per env, held for the whole episode) rather than on whatever the env happened to draw, so
        the keeper compares checkpoints on the same question. `params` lets the caller choose the
        difficulty -- the tracking eval runs from a settled start, the bring-up eval from the drop
        and held-misaligned starts. The loop is jitted once per (env, n_max, ladder-shape) and
        exits as soon as every env has ended."""
        env = self.eval_env or self.env
        n_max = int(n_max_steps or min(env.max_steps, 6000))
        if params is None:
            params = EnvParams.final(self.cfg)._replace(dr_scale=0.0, ctrl_jitter_ms=0.0,
                                                        ctrl_drop_prob=0.0)
        warm = int(self.cfg.eval_warm_ticks if warm_ticks is None else warm_ticks)
        ladder = None if ladder is None else jnp.asarray(ladder, jnp.float32)
        key_ = (id(env), n_max, warm, None if ladder is None else ladder.shape)
        if not hasattr(self, "_eval_fns"):
            self._eval_fns = {}
        if key_ not in self._eval_fns:
            net = self.net

            def run(p, mean, var, count, key, ov):
                stats = ObsStats(mean=mean, var=var, count=count)
                state, obs = env.reset(key, params, ov)

                # a while_loop that stops when every env has ended (or at n_max): a fixed
                # n_max-step scan cost the full 60 s episode cap (~200 s on a V100) even when all
                # 16 greedy envs fell within a second, 63% of the wall clock early in training
                if ladder is not None:
                    # THE COMMAND LADDER. Letting the env draw its own commands makes the keeper's
                    # tracking number a lottery over which stick positions happened to come up, and
                    # the draw is re-rolled mid-episode, so two checkpoints are never scored on the
                    # same question. Pin one stick position per env for the whole episode instead:
                    # the same ladder, every eval, every checkpoint, so the numbers are comparable
                    # across a run and across runs.
                    state = state.replace(v_cmd=jnp.asarray(ladder),
                                          cmd_left=jnp.full_like(state.cmd_left, 1e9))
                    # with a RUN/STOP switch the zero rung of the ladder IS the stop command: the
                    # contract's "0%" is "stand still", and that is the flag's job, not the stick's
                    if getattr(self.cfg, "stop_flag", False):
                        state = state.replace(stop_cmd=(jnp.asarray(ladder) <= 1e-6).astype(jnp.float32))
                    else:
                        state = state.replace(stop_cmd=jnp.zeros_like(state.cmd_left))

                def body(carry):
                    state, obs, alive, first_end, dist, tline, fin, fell, trk, yaw, settled, t = carry
                    a = jnp.clip(net.apply(p, stats.normalize(obs), method=net.actor_mean), -1.0, 1.0)
                    state2, obs2, r, done, info = env.step(state, a, params)
                    ending = alive & done
                    first_end = jnp.where(ending, state.step_n + 1, first_end)
                    dist = jnp.where(alive, info["sprint_d"], dist)
                    tline = jnp.where(ending & info["finished"], info["t_line"], tline)
                    fin = fin | (ending & info["finished"])
                    fell = fell | (ending & info["fallen"])
                    # how far off the commanded speed this policy runs. Under the joystick objective
                    # distance is not the goal -- a policy that ignores the stick and sprints covers
                    # the most ground, so keeping "best by distance" would keep the worst policy.
                    # BODY-frame forward speed, which is what the command means and what the reward
                    # bills; world x would score an obedient robot that has turned as disobedient.
                    # Only the settled tail counts: the first `warm` ticks are the bring-up
                    # transient, where every policy is off its command for reasons that are not
                    # tracking.
                    on = alive & (t >= warm)
                    trk = trk + jnp.where(on, jnp.abs(info["v_body_x"] - info["v_cmd"]), 0.0)
                    yaw = yaw + jnp.where(on, jnp.abs(info["yaw_true"]), 0.0)
                    settled = settled + on.astype(jnp.float32)
                    return (state2, obs2, alive & ~done, first_end, dist, tline, fin, fell, trk,
                            yaw, settled, t + 1)

                def cond(carry):
                    return carry[2].any() & (carry[11] < n_max)

                n = env.n_envs
                init = (state, obs, jnp.ones(n, bool), jnp.zeros(n, jnp.int32), jnp.zeros(n),
                        jnp.full(n, -1.0), jnp.zeros(n, bool), jnp.zeros(n, bool), jnp.zeros(n),
                        jnp.zeros(n), jnp.zeros(n), jnp.zeros((), jnp.int32))
                carry = jax.lax.while_loop(cond, body, init)
                _, _, alive, first_end, dist, tline, fin, fell, trk, yaw, settled, _ = carry
                return alive, first_end, dist, tline, fin, fell, trk, yaw, settled

            self._eval_fns[key_] = jax.jit(run)
        ov = override if override is not None else Override()
        ov = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(jnp.asarray(x, jnp.float32), (env.n_envs,)), ov)
        alive, first_end, dist, tline, fin, fell, trk, yaw, settled = self._eval_fns[key_](
            self.params, self.stats.mean, self.stats.var, self.stats.count, jax.random.PRNGKey(seed), ov)
        alive, first_end = np.asarray(alive), np.asarray(first_end)
        first_end = np.where(alive, n_max, first_end)
        fin, fell = np.asarray(fin), np.asarray(fell)
        out = dict(finishes=int(fin.sum()), falls=int(fell.sum()), n=env.n_envs, t_line=np.asarray(tline),
                   dist=np.asarray(dist), ep_len_s=first_end * env.control_dt)
        out["t_line_mean"] = float(out["t_line"][fin].mean()) if fin.any() else float("nan")
        out["dist_mean"] = float(out["dist"].mean())
        out["speed_mean"] = float((out["dist"] / np.maximum(out["ep_len_s"], 1e-6)).mean())
        settled = np.asarray(settled)
        # A policy that dies before the settled window produced no speed at all, so its command
        # error is the whole command -- NOT zero. Dividing by max(settled, 1) would report 0.00 m/s
        # for a robot that fell on tick 3, which reads as perfect tracking in the log and puts a
        # corpse at the top of the keeper's ladder.
        cmd_v = (np.asarray(ladder) if ladder is not None
                 else np.zeros_like(settled))
        out["track_err"] = np.where(settled > 0, np.asarray(trk) / np.maximum(settled, 1.0),
                                    np.abs(cmd_v))
        out["settled"] = settled
        out["track_err_mean"] = float(out["track_err"].mean())
        out["heading_err_mean"] = float((np.asarray(yaw) / np.maximum(settled, 1.0)).mean())
        out["alive_frac"] = float((~np.asarray(fell)).mean())
        out["settled_ticks"] = float(settled.mean())
        if ladder is not None:
            lad = np.asarray(ladder)
            out["ladder"] = lad
            # per-command breakdown: a mean hides a policy that tracks 50% perfectly and ignores 100%
            out["per_cmd"] = [(float(v), float(out["track_err"][lad == v].mean()),
                               float((~np.asarray(fell))[lad == v].mean()))
                              for v in sorted(set(lad.tolist()))]
        return out

    # ---------------------------------------------------------------- persistence
    def state_dict(self):
        return {
            "params": self.params, "opt_state": self.opt_state, "est_opt_state": self.est_opt_state,
            "stats": {"mean": self.stats.mean, "var": self.stats.var, "count": self.stats.count},
            "step": self.step, "rollout_n": self.rollout_n, "ent_coef": self.ent_coef,
            "log_std_clamp": self.log_std_clamp,
            "anneal_from": -1 if self.anneal_from is None else int(self.anneal_from),
            "anneal_base": -1.0 if self.anneal_base is None else float(self.anneal_base),
            "streak": self.streak, "key": self.key,
        }

    def save(self, path):
        path = Path(path)
        path.write_bytes(serialization.to_bytes(self.state_dict()))
        (path.with_suffix(".json")).write_text(json.dumps({
            "step": self.step, "rollout_n": self.rollout_n, "curriculum": self.cur,
            "env_params": self.env_params._asdict()}, indent=1))

    def load(self, path, warm_start=False):
        path = Path(path)
        d = serialization.from_bytes(self.state_dict(), path.read_bytes())
        self.params = d["params"]
        self.stats = ObsStats(mean=jnp.asarray(d["stats"]["mean"]), var=jnp.asarray(d["stats"]["var"]),
                              count=jnp.asarray(d["stats"]["count"]))
        if warm_start:
            c = self.cfg
            if c.warmstart_obs_count_cap > 0:
                self.stats = self.stats.replace(count=jnp.minimum(self.stats.count, c.warmstart_obs_count_cap))
            if c.warmstart_var_floor > 0:
                self.stats = self.stats.replace(var=jnp.maximum(self.stats.var, c.warmstart_var_floor))
            if c.warmstart_reset_log_std:
                self.params = _clamp_log_std(self.params, float(c.max_log_std), fill=True)
            print(f"[ppo] warm-started weights + obs stats <- {path.name} (fresh optimizer, schedules, step 0)")
            return
        self.opt_state = d["opt_state"]
        self.est_opt_state = d["est_opt_state"]
        self.step, self.rollout_n = int(d["step"]), int(d["rollout_n"])
        self.ent_coef, self.log_std_clamp = float(d["ent_coef"]), float(d["log_std_clamp"])
        self.anneal_from = None if int(d["anneal_from"]) < 0 else int(d["anneal_from"])
        self.anneal_base = None if float(d["anneal_base"]) < 0 else float(d["anneal_base"])
        self.streak = int(d["streak"])
        self.key = jnp.asarray(d["key"])
        js = path.with_suffix(".json")
        if js.exists():
            meta = json.loads(js.read_text())
            self.cur = meta.get("curriculum", {})
            if "env_params" in meta:
                self.env_params = EnvParams(**meta["env_params"])
            else:       # a sidecar without run state (e.g. an eval summary): keep the defaults
                print(f"[ppo] {js.name} carries no env_params; keeping the config's curriculum start")
        print(f"[ppo] resumed {path.name} at {self.step:,} steps")


def _adam_count(opt_state) -> int:
    """Adam's step counter wherever it sits in the optimizer state (chain / apply_if_finite wrappers)."""
    for leaf in jax.tree_util.tree_leaves(opt_state, is_leaf=lambda x: isinstance(x, optax.ScaleByAdamState)):
        if isinstance(leaf, optax.ScaleByAdamState):
            return int(np.asarray(leaf.count))
    return 0


def _set_lr(opt_state, lr):
    """Set the learning rate inside an optax.inject_hyperparams state (wherever it sits in the chain)."""
    def fix(leaf):
        if isinstance(leaf, optax.InjectHyperparamsState):
            hp = dict(leaf.hyperparams)
            hp["learning_rate"] = jnp.asarray(lr, jnp.float32)
            return leaf._replace(hyperparams=hp)
        return leaf
    return jax.tree_util.tree_map(fix, opt_state, is_leaf=lambda x: isinstance(x, optax.InjectHyperparamsState))


def _clamp_log_std(params, clamp, fill=False):
    ls = params["params"]["log_std"]
    new = jnp.full_like(ls, clamp) if fill else jnp.minimum(ls, clamp)
    return {**params, "params": {**params["params"], "log_std": new}}
