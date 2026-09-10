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

Truncation at the episode cap is treated as terminal (no bootstrap); a 60 s cap is rarely hit
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


# ------------------------------------------------------------------ the trainer
class PPO:
    def __init__(self, cfg, env: DashEnvV2, run_dir, total_steps, seed=0, eval_env=None):
        self.cfg, self.env, self.run = cfg, env, Path(run_dir)
        self.total_steps = int(total_steps)
        self.n_envs, self.n_steps = env.n_envs, int(cfg.n_steps)
        self.batch = self.n_envs * self.n_steps
        self.n_minibatches = max(1, self.batch // int(cfg.batch_size))
        self.mb_size = self.batch // self.n_minibatches
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
        self.lr = optax.linear_schedule(cfg.learning_rate, cfg.lr_final, n_updates)
        self.tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(self.lr))
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
                    action=a, raw_obs=obs,
                )
                tr = Transition(obs=nobs, action=a, log_prob=lp, reward=r, done=done,
                                value=value, mask=mask)
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

        def gae(tr, last_value, gamma, lam):
            def scan_fn(carry, x):
                adv_next, v_next = carry
                r, d, v = x
                nonterm = 1.0 - d.astype(jnp.float32)
                delta = r + gamma * v_next * nonterm - v
                adv = delta + gamma * lam * nonterm * adv_next
                return (adv, v), adv
            _, adv = jax.lax.scan(scan_fn, (jnp.zeros_like(last_value), last_value),
                                  (tr.reward, tr.done, tr.value), reverse=True)
            return adv, adv + tr.value

        self._gae = jax.jit(gae, static_argnums=(2, 3))

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
            total = pg + cfg.vf_coef * vf - ent_coef * ent + cfg.w_sym * sym + cfg.w_knob_loss * knob
            approx_kl = jnp.mean((ratio - 1.0) - jnp.log(ratio))
            clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32))
            return total, dict(pg=pg, vf=vf, ent=ent, sym=sym, kl=approx_kl, clipfrac=clipfrac)

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
        c = self.cfg
        return EnvParams(dr_scale=0.0 if (c.dr_enable and c.dr_curriculum_steps > 0) else 1.0,
                         sprint_dist_m=float(c.sprint_dist_start_m if c.sprint_curriculum_steps > 0
                                             else c.sprint_dist_m),
                         stance_ratio=float(c.stance_ratio_start if c.gait_curriculum_steps > 0
                                            else c.stance_ratio_final),
                         eff_scale=0.0 if c.efficiency_ramp_steps > 0 else float(c.efficiency_target),
                         ctrl_jitter_ms=0.0 if c.jitter_curriculum_steps > 0 else float(c.ctrl_jitter_ms_final),
                         ctrl_drop_prob=0.0 if c.jitter_curriculum_steps > 0 else float(c.ctrl_drop_prob_final),
                         pitch_assist=1.0 if (c.pitch_assist_kp > 0 and c.pitch_assist_ramp_steps > 0) else 0.0)

    def _gated(self, key, ep_len, start, target, warmup, gate, retreat, d_steps):
        """Competence-gated, retreating ramp (walk_mit GatedRampCallback): progress in [0,1]."""
        st = self.cur.setdefault(key, {"streak": 0, "open": False, "progress": 0.0})
        if gate <= 0:
            st["progress"] = min(1.0, st["progress"] + d_steps / max(warmup, 1))
        else:
            if not st["open"]:
                st["streak"] = st["streak"] + 1 if ep_len > gate else 0
                if st["streak"] >= 5:
                    st["open"] = True
                    print(f"[ppo] curriculum gate '{key}' opened at {self.step:,} steps (ep_len {ep_len:.0f})")
            if st["open"]:
                stp = d_steps / max(warmup, 1)
                if ep_len >= gate:
                    st["progress"] += stp
                elif retreat > 0 and ep_len < retreat * gate:
                    st["progress"] -= stp
                st["progress"] = float(np.clip(st["progress"], 0.0, 1.0))
        return start + st["progress"] * (target - start)

    def _clock(self, start, target, warmup):
        frac = 1.0 if warmup <= 0 else min(1.0, self.step / warmup)
        return start + frac * (target - start)

    def update_curricula(self, ep_len, d_steps):
        c = self.cfg
        p = self.env_params
        gate, rf = c.curriculum_gate_ep_len, c.curriculum_retreat_frac
        kw = dict(p._asdict())
        if c.dr_enable and c.dr_curriculum_steps > 0:
            kw["dr_scale"] = self._gated("dr_scale", ep_len, 0.0, 1.0, c.dr_curriculum_steps, gate, rf, d_steps)
        if c.objective == "sprint" and c.sprint_curriculum_steps > 0:
            kw["sprint_dist_m"] = self._clock(c.sprint_dist_start_m, c.sprint_dist_m, c.sprint_curriculum_steps)
        if c.gait_curriculum_steps > 0 and c.w_phase_contact > 0:
            kw["stance_ratio"] = self._gated("stance_ratio", ep_len, c.stance_ratio_start, c.stance_ratio_final,
                                             c.gait_curriculum_steps, gate, rf, d_steps)
        if c.efficiency_ramp_steps > 0:
            kw["eff_scale"] = self._gated("eff_scale", ep_len, 0.0, c.efficiency_target,
                                          c.efficiency_ramp_steps, gate, rf, d_steps)
        if c.jitter_curriculum_steps > 0:
            jg = c.jitter_curriculum_gate_ep_len
            kw["ctrl_jitter_ms"] = self._gated("ctrl_jitter_ms", ep_len, 0.0, c.ctrl_jitter_ms_final,
                                               c.jitter_curriculum_steps, jg, rf, d_steps)
            kw["ctrl_drop_prob"] = self._gated("ctrl_drop_prob", ep_len, 0.0, c.ctrl_drop_prob_final,
                                               c.jitter_curriculum_steps, jg, rf, d_steps)
        if c.pitch_assist_kp > 0 and c.pitch_assist_ramp_steps > 0:
            kw["pitch_assist"] = self._clock(1.0, 0.0, c.pitch_assist_ramp_steps)
        self.env_params = EnvParams(**{k: float(v) for k, v in kw.items()})

    def update_entropy(self, swing_min):
        c = self.cfg
        if self.anneal_from is None:
            competent = swing_min > c.ent_gate_swing_frac
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
            "train/approx_kl": aux_acc.get("kl", 0.0) / max(n_mb, 1),
            "train/clipfrac": aux_acc.get("clipfrac", 0.0) / max(n_mb, 1),
            "train/n_minibatch_updates": n_mb, "train/early_stop": float(stop),
            "train/ent_coef": self.ent_coef, "train/log_std_clamp": self.log_std_clamp,
            "train/std_mean": float(np.exp(np.asarray(self.params["params"]["log_std"])).mean()),
            "train/lr": float(self.lr(int(np.asarray(self.opt_state[1][0].count)))),
            "est/vel_rmse": float(np.sqrt(float(est_last))) if est_last is not None else float("nan"),
        }
        for kk, v in m["reward_terms"].items():
            log[f"reward_terms/{kk}"] = float(np.asarray(v).mean())
        for kk, v in self.env_params._asdict().items():
            log[f"curriculum/{kk}"] = float(v)
        # ---- schedules for the NEXT rollout
        self.update_entropy(swing_min)
        self.update_curricula(ep_len, d_steps)
        return log

    def _symmetrize_stats(self, stats):
        """Mirror the running stats: mean -> 0.5 (m + M m), var -> 0.5 (v + |M| v), so early
        L/R drift in the normalizer cannot bias the symmetric gait (§13)."""
        m = np.asarray(stats.mean)
        v = np.asarray(stats.var)
        a = self.env.actor_dim
        mm = np.array(self.mirror(jnp.asarray(m[None, :a]))[0])
        # the phase channels of the mirror depend on Delta (a mean of ~0 anyway): leave them
        H = self.env.cfg.history_len
        for i in range(H):
            mm[i * 33 + 25:i * 33 + 27] = m[i * 33 + 25:i * 33 + 27]
        vm = np.abs(np.array(self.mirror(jnp.asarray(v[None, :a]))[0]))
        for i in range(H):
            vm[i * 33 + 25:i * 33 + 27] = v[i * 33 + 25:i * 33 + 27]
        m2, v2 = m.copy(), v.copy()
        m2[:a] = 0.5 * (m[:a] + mm)
        v2[:a] = 0.5 * (v[:a] + vm)
        return ObsStats(mean=jnp.asarray(m2), var=jnp.asarray(v2), count=stats.count)

    # ---------------------------------------------------------------- greedy eval
    def evaluate(self, n_max_steps=None, override=None, seed=1000):
        """Greedy dash on the eval env (nominal plant): finishes, t_line, mean speed, falls.
        The loop is jitted once per (env, n_max), exits when every env has ended, and takes
        params/stats as arguments."""
        env = self.eval_env or self.env
        n_max = int(n_max_steps or min(env.max_steps, 6000))
        params = EnvParams.final(self.cfg)._replace(dr_scale=0.0, ctrl_jitter_ms=0.0,
                                                    ctrl_drop_prob=0.0, pitch_assist=0.0)
        key_ = (id(env), n_max)
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
                def body(carry):
                    state, obs, alive, first_end, dist, tline, fin, fell, t = carry
                    a = jnp.clip(net.apply(p, stats.normalize(obs), method=net.actor_mean), -1.0, 1.0)
                    state2, obs2, r, done, info = env.step(state, a, params)
                    ending = alive & done
                    first_end = jnp.where(ending, state.step_n + 1, first_end)
                    dist = jnp.where(alive, info["sprint_d"], dist)
                    tline = jnp.where(ending & info["finished"], info["t_line"], tline)
                    fin = fin | (ending & info["finished"])
                    fell = fell | (ending & info["fallen"])
                    return (state2, obs2, alive & ~done, first_end, dist, tline, fin, fell, t + 1)

                def cond(carry):
                    return carry[2].any() & (carry[8] < n_max)

                n = env.n_envs
                init = (state, obs, jnp.ones(n, bool), jnp.zeros(n, jnp.int32), jnp.zeros(n),
                        jnp.full(n, -1.0), jnp.zeros(n, bool), jnp.zeros(n, bool), jnp.zeros((), jnp.int32))
                carry = jax.lax.while_loop(cond, body, init)
                _, _, alive, first_end, dist, tline, fin, fell, _ = carry
                return alive, first_end, dist, tline, fin, fell

            self._eval_fns[key_] = jax.jit(run)
        ov = override if override is not None else Override()
        ov = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(jnp.asarray(x, jnp.float32), (env.n_envs,)), ov)
        alive, first_end, dist, tline, fin, fell = self._eval_fns[key_](
            self.params, self.stats.mean, self.stats.var, self.stats.count, jax.random.PRNGKey(seed), ov)
        alive, first_end = np.asarray(alive), np.asarray(first_end)
        first_end = np.where(alive, n_max, first_end)
        fin, fell = np.asarray(fin), np.asarray(fell)
        out = dict(finishes=int(fin.sum()), falls=int(fell.sum()), n=env.n_envs, t_line=np.asarray(tline),
                   dist=np.asarray(dist), ep_len_s=first_end * env.control_dt)
        out["t_line_mean"] = float(out["t_line"][fin].mean()) if fin.any() else float("nan")
        out["dist_mean"] = float(out["dist"].mean())
        out["speed_mean"] = float((out["dist"] / np.maximum(out["ep_len_s"], 1e-6)).mean())
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
            self.env_params = EnvParams(**meta["env_params"])
        print(f"[ppo] resumed {path.name} at {self.step:,} steps")


def _clamp_log_std(params, clamp, fill=False):
    ls = params["params"]["log_std"]
    new = jnp.full_like(ls, clamp) if fill else jnp.minimum(ls, clamp)
    return {**params, "params": {**params["params"], "log_std": new}}
