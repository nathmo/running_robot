"""PPO for the balance policy: JAX, data-parallel over GPUs with pmap (one device is the same code).

Per iteration: every device rolls out n_steps x (n_envs / n_dev) envs in one lax.scan and computes
its own GAE; the update runs n_epochs over shuffled minibatches with gradients averaged by pmean;
then the supervised velocity estimator gets est_epochs of its own Adam; then the host reads the
rollout's push verdicts and moves the curricula.

Choices that matter (each is a lesson from RLframework/ or the memory notes):
  * truncation at the episode cap BOOTSTRAPS from V(final state); only a fall is terminal
  * adaptive lr from the KL (rl_games): / 1.5 above 2x target, x 1.5 below 0.5x, per epoch
  * clamped exploration: std starts at cfg.init_std (positions) / init_std_gain (kp, kd) and lives
    in [min_std, max_std]. Big enough that a recovery STEP is reachable -- 0.05 was not, and bought
    171 M steps of a stander that could not take one -- and small enough that the early episodes are
    still lessons rather than a robot shaking itself over from the first tick
  * bounds loss on |mu| > 1: the env clips the action, so a mean past the box earns nothing
  * symmetry loss ||mu(M s) - M mu(s)||^2 in RAW obs space (mirror, then normalize)
  * the plant width (CoM shift + DR) is a CLOCK; only the push level is gated, on push verdicts,
    never on episode length (a gate in front of the plant width starved the first campaign)
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

import networks as nets
from networks import init_log_std, std_clamp
from env import BalanceEnv, EnvParams, mirror_actor, mirror_action, FRAME_DIM
from plant import Override
from config import config_to_dict

N_BINS = 12
BIN_W = 0.25          # m/s per survival bin in the logs
REWARD_SCALE = 0.02   # returns of O(10): the critic's MSE stays well conditioned


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

    def update(self, b_mean, b_var, b_n):
        delta = b_mean - self.mean
        tot = self.count + b_n
        mean = self.mean + delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot
        return ObsStats(mean=mean, var=m2 / tot, count=tot)


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    trunc: jnp.ndarray
    v_final: jnp.ndarray


def gae_fn(tr, last_value, gamma, lam):
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


class Trainer:
    def __init__(self, cfg, run_dir, seed=0, n_devices=1, n_eval=240):
        self.cfg = cfg
        self.run = Path(run_dir)
        self.run.mkdir(parents=True, exist_ok=True)
        self.n_dev = int(n_devices)
        self.devices = jax.local_devices()[:self.n_dev]
        assert len(self.devices) == self.n_dev, f"asked for {self.n_dev} devices, have {jax.local_devices()}"
        assert cfg.n_envs % self.n_dev == 0
        self.env = BalanceEnv(cfg, cfg.n_envs // self.n_dev)
        self.eval_env = BalanceEnv(cfg, n_eval) if n_eval else None
        env = self.env
        self.n_steps = int(cfg.n_steps)
        self.batch_dev = env.n_envs * self.n_steps
        self.batch = self.batch_dev * self.n_dev
        self.n_mb = max(1, self.batch // int(cfg.batch_size))
        self.mb_dev = self.batch_dev // self.n_mb
        self.n_est_mb = max(1, self.batch // int(cfg.est_batch))
        self.est_mb_dev = self.batch_dev // self.n_est_mb
        self.net = nets.ActorCritic(n_actor=env.actor_dim, n_priv=env.priv_dim, action_dim=env.action_dim,
                                    policy_hidden=tuple(cfg.policy_hidden), est_hidden=tuple(cfg.est_hidden),
                                    init_log_std=init_log_std(cfg))
        self.key = jax.random.PRNGKey(int(seed))
        self.key, k = jax.random.split(self.key)
        self.params = self.net.init(k, jnp.zeros((1, env.obs_dim)))
        self.tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.scale_by_adam())
        self.opt_state = self.tx.init(self.params)
        est_mask = jax.tree_util.tree_map_with_path(
            lambda path, _: any(getattr(p, "key", None) == "estimator" for p in path), self.params)
        self.est_tx = optax.masked(optax.adam(cfg.est_lr), est_mask)
        self.est_opt_state = self.est_tx.init(self.params)
        self.stats = ObsStats.init(env.obs_dim)
        self.lr = float(cfg.learning_rate)
        self.n_upd = 0                     # minibatch updates, for the warmup
        self.step = 0
        self.rollout_n = 0
        self.cur = dict(plant_scale=float(cfg.plant_scale_start), push_level=float(cfg.push_level_start),
                        n_hard=0, ok_hard=0, n_ep=0, quiet=0, best_score=-1.0,
                        push_ema=float("nan"), n_promote=0)
        self.env_state = self.obs = None
        self._build()

    # ---------------------------------------------------------------- curriculum
    def env_params(self):
        return EnvParams(plant_scale=float(self.cur["plant_scale"]), push_level=float(self.cur["push_level"]),
                         push_on=1.0)

    def update_curricula(self, ev):
        """Two gated ladders, in order: widen the plant (CoM shift + DR) while the robot keeps from
        falling on its own, then grow the pushes while it survives them."""
        c, cur = self.cfg, self.cur
        cur["n_ep"] += ev["episodes"]
        cur["quiet"] += ev["quiet_falls"]
        cur["n_hard"] += ev["n_hard"]
        cur["ok_hard"] += ev["ok_hard"]
        moved = None
        # the plant width is a clock: CoM shift and DR reach full by plant_ramp_steps whatever the
        # policy is doing (see config -- a gate here starved the whole first campaign)
        was = cur["plant_scale"]
        s0 = float(c.plant_scale_start)
        cur["plant_scale"] = min(1.0, s0 + (1.0 - s0) * self.step / max(int(c.plant_ramp_steps), 1))
        if was < 1.0 <= cur["plant_scale"]:
            moved = "plant_scale -> 1.00 (full CoM shift and DR)"
        if cur["n_ep"] >= 2000:
            cur["n_ep"] = cur["quiet"] = 0
        if cur["n_hard"] >= c.push_gate_min_n:
            rate = cur["ok_hard"] / cur["n_hard"]
            cur["push_ema"] = rate
            if rate >= c.push_gate_up and cur["push_level"] < c.push_level_max:
                cur["push_level"] = min(c.push_level_max, cur["push_level"] * c.push_level_up)
                cur["n_promote"] += 1
                moved = (moved or "") + f" push_level -> {cur['push_level']:.3f} (hard survival {rate:.3f})"
            elif rate < c.push_gate_down and cur["push_level"] > c.push_level_start:
                cur["push_level"] = max(c.push_level_start, cur["push_level"] * c.push_level_down)
                moved = (moved or "") + f" push_level RETREAT -> {cur['push_level']:.3f} ({rate:.3f})"
            cur["n_hard"] = cur["ok_hard"] = 0
        return moved

    # ---------------------------------------------------------------- jitted pieces
    def _build(self):
        env, cfg, net = self.env, self.cfg, self.net
        n_hist = cfg.history_len
        A = env.actor_dim
        gamma, lam = float(cfg.gamma), float(cfg.gae_lambda)
        lo_std, hi_std = (jnp.asarray(x) for x in std_clamp(cfg))

        def rollout(params, stats, env_state, obs, key, env_params):
            def one(carry, _):
                env_state, obs, key = carry
                key, k = jax.random.split(key)
                nobs = stats.normalize(obs)
                mu, log_std, value, _ = net.apply(params, nobs)
                a = nets.sample(k, mu, log_std)
                env_state, obs2, r, done, info = env.step(env_state, jnp.clip(a, -1.0, 1.0), env_params)
                lp = nets.log_prob(mu, log_std, a)
                _, _, v_final, _ = net.apply(params, stats.normalize(info["obs_final"]))
                trunc = info["truncated"] & ~info["fallen"]
                tr = Transition(obs=nobs, action=a, log_prob=lp, reward=r * REWARD_SCALE, done=done,
                                value=value, trunc=trunc, v_final=v_final)
                b = jnp.clip(jnp.floor(info["push_dv"] / BIN_W), 0, N_BINS - 1).astype(jnp.int32)
                oh = jax.nn.one_hot(b, N_BINS)
                m = dict(
                    episodes=done.sum(), falls=info["fallen"].sum(), quiet_falls=info["quiet_fall"].sum(),
                    ep_len=jnp.sum(jnp.where(done, info["ep_len"], 0)),
                    ep_ret=jnp.sum(jnp.where(done, info["ep_return"], 0.0)),
                    pushes=info["push_start"].sum(),
                    ok_hard=jnp.sum(info["push_ok"] & info["push_hard"]),
                    n_hard=jnp.sum((info["push_ok"] | info["push_fail"]) & info["push_hard"]),
                    ok_bin=jnp.sum(oh * info["push_ok"][:, None], 0),
                    fail_bin=jnp.sum(oh * info["push_fail"][:, None], 0),
                    kp_mean=info["kp_mean"].mean(), kd_mean=info["kd_mean"].mean(),
                    torque_util=info["torque_util"].mean(), tilt_deg=info["tilt_deg"].mean(),
                    mu_abs=jnp.abs(mu).mean(), reward=r.mean(),
                    terms=jax.tree_util.tree_map(lambda x: x.mean(), info["reward_terms"]),
                )
                return (env_state, obs2, key), (tr, m, obs)
            (env_state, obs, key), (tr, m, raw) = jax.lax.scan(one, (env_state, obs, key), None,
                                                               length=self.n_steps)
            _, _, last_v, _ = net.apply(params, stats.normalize(obs))
            adv, ret = gae_fn(tr, last_v, gamma, lam)
            flat = raw.reshape(-1, raw.shape[-1])
            moments = (flat.mean(0), flat.var(0))
            return env_state, obs, tr, adv, ret, m, moments

        def loss_fn(params, stats, obs, act, old_lp, adv, ret):
            mu, log_std, value, _ = net.apply(params, obs)
            lp = nets.log_prob(mu, log_std, act)
            ratio = jnp.exp(lp - old_lp)
            adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = -jnp.mean(jnp.minimum(ratio * adv_n, jnp.clip(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * adv_n))
            vf = jnp.mean((ret - value) ** 2)
            ent = jnp.mean(nets.entropy(log_std))
            bound = jnp.mean(jnp.maximum(jnp.abs(mu) - 1.0, 0.0) ** 2)
            if cfg.w_sym > 0.0:
                raw = stats.denormalize(obs)
                raw_m = raw.at[..., :A].set(mirror_actor(raw[..., :A], n_hist))
                mu_m = net.apply(params, stats.normalize(raw_m), method=net.actor_mean)
                sym = jnp.mean(jnp.sum((mu_m - mirror_action(mu)) ** 2, axis=-1))
            else:
                sym = jnp.zeros(())
            total = pg + cfg.vf_coef * vf - cfg.ent_coef * ent + cfg.w_bound * bound + cfg.w_sym * sym
            kl = jnp.mean((ratio - 1.0) - jnp.log(ratio))
            clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > cfg.clip_range).astype(jnp.float32))
            return total, dict(pg=pg, vf=vf, ent=ent, bound=bound, sym=sym, kl=kl, clipfrac=clipfrac)

        def clamp_std(params):
            p = params["params"]
            p = {**p, "log_std": jnp.clip(p["log_std"], lo_std, hi_std)}
            return {**params, "params": p}

        n_mb, mb = self.n_mb, self.mb_dev

        def update_epoch(params, opt_state, stats, data, perm, lr):
            def body(carry, i):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice(perm, (i * mb,), (mb,))
                batch = tuple(x[idx] for x in data)
                (_, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, stats, *batch)
                g = jax.lax.pmean(g, "dev")
                aux = jax.lax.pmean(aux, "dev")
                upd, opt_state = self.tx.update(g, opt_state, params)
                params = jax.tree_util.tree_map(lambda p, u: p - lr * u, params, upd)
                return (clamp_std(params), opt_state), aux
            (params, opt_state), auxs = jax.lax.scan(body, (params, opt_state), jnp.arange(n_mb))
            return params, opt_state, jax.tree_util.tree_map(jnp.mean, auxs)

        n_est, est_mb = self.n_est_mb, self.est_mb_dev

        def est_epoch(params, est_state, obs, perm):
            def est_loss(params, o):
                est = net.apply(params, o, method=net.estimate)
                return jnp.mean((est - o[..., A:A + 3]) ** 2)

            def body(carry, i):
                params, est_state = carry
                idx = jax.lax.dynamic_slice(perm, (i * est_mb,), (est_mb,))
                loss, g = jax.value_and_grad(est_loss)(params, obs[idx])
                g = jax.lax.pmean(g, "dev")
                upd, est_state = self.est_tx.update(g, est_state, params)
                return (optax.apply_updates(params, upd), est_state), jax.lax.pmean(loss, "dev")
            (params, est_state), losses = jax.lax.scan(body, (params, est_state), jnp.arange(n_est))
            return params, est_state, losses[-1]

        devs = self.devices
        self._rollout_p = jax.pmap(rollout, axis_name="dev", in_axes=(0, None, 0, 0, 0, None), devices=devs)
        self._update_p = jax.pmap(update_epoch, axis_name="dev", in_axes=(0, 0, None, 0, 0, None), devices=devs)
        self._est_p = jax.pmap(est_epoch, axis_name="dev", in_axes=(0, 0, 0, 0), devices=devs)
        self._reset_p = jax.pmap(lambda k, prm: env.reset(k, prm), in_axes=(0, None), devices=devs)
        self._greedy = jax.jit(lambda params, nobs: net.apply(params, nobs, method=net.actor_mean))

    def _rep(self, tree):
        return jax.device_put_replicated(tree, self.devices) if hasattr(jax, "device_put_replicated") \
            else self._rep_sharded(tree)

    def _rep_sharded(self, tree):
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        sh = NamedSharding(Mesh(np.array(self.devices), ("dev",)), P("dev"))
        return jax.tree_util.tree_map(
            lambda x: jax.device_put(np.broadcast_to(np.asarray(x), (self.n_dev,) + np.shape(x)), sh), tree)

    def reset_envs(self):
        self.key, k = jax.random.split(self.key)
        self.env_state, self.obs = self._reset_p(jax.random.split(k, self.n_dev), self.env_params())

    # ---------------------------------------------------------------- one iteration
    def iterate(self):
        cfg = self.cfg
        t0 = time.time()
        if self.env_state is None:
            self.reset_envs()
        p_rep = self._rep(self.params)
        self.key, k = jax.random.split(self.key)
        (self.env_state, self.obs, tr, adv, ret, m, (b_mean, b_var)) = self._rollout_p(
            p_rep, self.stats, self.env_state, self.obs, jax.random.split(k, self.n_dev), self.env_params())
        jax.block_until_ready(ret)
        t_roll = time.time() - t0
        t1 = time.time()
        flat = lambda x: x.reshape((self.n_dev, self.batch_dev) + x.shape[3:])
        data = (flat(tr.obs), flat(tr.action), flat(tr.log_prob), flat(adv), flat(ret))
        opt_rep = self._rep(self.opt_state)
        auxs = []
        for _ in range(cfg.n_epochs):
            self.key, k = jax.random.split(self.key)
            perm = jnp.stack([jax.random.permutation(kd, self.batch_dev) for kd in jax.random.split(k, self.n_dev)])
            # LR WARMUP: Adam's first step is ~lr on every weight at once, and at this sigma that is a
            # KL of 20 (measured). Ramp it over the first cfg.lr_warmup_updates minibatches.
            warm = min(1.0, (self.n_upd + 1) / max(int(cfg.lr_warmup_updates), 1))
            p_rep, opt_rep, aux = self._update_p(p_rep, opt_rep, self.stats, data, perm, self.lr * warm)
            self.n_upd += self.n_mb
            aux = {kk: float(np.asarray(v)[0]) for kk, v in aux.items()}
            auxs.append(aux)
            # adaptive step size, per epoch
            if cfg.target_kl > 0 and warm >= 1.0:
                if aux["kl"] > 2.0 * cfg.target_kl:
                    self.lr = max(cfg.lr_min, self.lr / 1.5)
                elif aux["kl"] < 0.5 * cfg.target_kl:
                    self.lr = min(cfg.lr_max, self.lr * 1.5)
        est_rep = self._rep(self.est_opt_state)
        est_loss = float("nan")
        for _ in range(cfg.est_epochs):
            self.key, k = jax.random.split(self.key)
            perm = jnp.stack([jax.random.permutation(kd, self.batch_dev) for kd in jax.random.split(k, self.n_dev)])
            p_rep, est_rep, el = self._est_p(p_rep, est_rep, data[0], perm)
            est_loss = float(np.asarray(el)[0])
        un = lambda t: jax.tree_util.tree_map(lambda x: x[0], t)
        self.params, self.opt_state, self.est_opt_state = un(p_rep), un(opt_rep), un(est_rep)
        jax.block_until_ready(self.params)
        t_upd = time.time() - t1
        # pooled raw-obs moments across devices -> the normalizer
        mu_d, var_d = np.asarray(b_mean, np.float64), np.asarray(b_var, np.float64)
        bm = mu_d.mean(0)
        bv = np.maximum((var_d + mu_d ** 2).mean(0) - bm ** 2, 0.0)
        self.stats = self.stats.update(jnp.asarray(bm, jnp.float32), jnp.asarray(bv, jnp.float32), float(self.batch))
        self.step += self.batch
        self.rollout_n += 1
        # ---- events and metrics
        M = jax.tree_util.tree_map(lambda x: np.asarray(x), m)        # (dev, T, ...)
        tot = lambda k: float(M[k].sum())
        ev = dict(episodes=int(tot("episodes")), quiet_falls=int(tot("quiet_falls")),
                  n_hard=int(tot("n_hard")), ok_hard=int(tot("ok_hard")))
        ok_bin = M["ok_bin"].reshape(-1, N_BINS).sum(0)
        fail_bin = M["fail_bin"].reshape(-1, N_BINS).sum(0)
        self._bins = getattr(self, "_bins", np.zeros((2, N_BINS)))
        self._bins = 0.98 * self._bins + np.stack([ok_bin, fail_bin])
        n_ep = max(ev["episodes"], 1)
        log = {
            "time/env_steps": self.step, "time/iter_s": time.time() - t0, "time/rollout_s": t_roll,
            "time/update_s": t_upd, "time/sps": self.batch / max(time.time() - t0, 1e-9),
            "rollout/episodes": ev["episodes"], "rollout/falls": int(tot("falls")),
            "rollout/quiet_falls": ev["quiet_falls"],
            "rollout/ep_len_mean": tot("ep_len") / n_ep, "rollout/ep_ret_mean": tot("ep_ret") / n_ep,
            "rollout/pushes": int(tot("pushes")), "rollout/hard_verdicts": ev["n_hard"],
            "rollout/hard_survival": ev["ok_hard"] / max(ev["n_hard"], 1),
            "rollout/reward_mean": float(M["reward"].mean()),
            "diag/kp_mean": float(M["kp_mean"].mean()), "diag/kd_mean": float(M["kd_mean"].mean()),
            "diag/torque_util": float(M["torque_util"].mean()), "diag/tilt_deg": float(M["tilt_deg"].mean()),
            "diag/mu_abs": float(M["mu_abs"].mean()),
            "train/lr": self.lr, "train/est_loss": est_loss,
            "train/std_mean": float(np.exp(np.asarray(self.params["params"]["log_std"])).mean()),
        }
        for kk in auxs[0]:
            log[f"train/{kk}"] = float(np.mean([a[kk] for a in auxs]))
        for kk, v in M["terms"].items():
            log[f"reward_terms/{kk}"] = float(v.mean())
        moved = self.update_curricula(ev)
        log["curriculum/plant_scale"] = self.cur["plant_scale"]
        log["curriculum/push_level"] = self.cur["push_level"]
        log["curriculum/push_gate_rate"] = self.cur["push_ema"]
        return log, moved

    def survival_bins(self):
        ok, fail = self._bins
        n = ok + fail
        return {f"{i * BIN_W:.2f}": (round(float(ok[i] / n[i]), 3) if n[i] >= 5 else None) for i in range(N_BINS)}

    # ---------------------------------------------------------------- greedy evaluation
    LADDER = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)   # the low end is where this robot lives

    def evaluate(self, params=None, stats=None, n_ticks=800, seed=1234, plant_scale=1.0, noise=True):
        """Greedy push ladder: every env gets pushes of ONE fixed size from random directions, on the
        full-width plant (CoM +-3 cm included) with the sensor noise on. Returns survival per rung
        (a push counts once its verdict is in) and the quiet-fall share."""
        env = self.eval_env
        params = self.params if params is None else params
        stats = self.stats if stats is None else stats
        n = env.n_envs
        rungs = np.array(self.LADDER)
        dv = np.repeat(rungs, int(math.ceil(n / len(rungs))))[:n]
        ov = Override(push_dv=jnp.asarray(dv, jnp.float32))
        prm = EnvParams(plant_scale=plant_scale, push_level=1.0, push_on=1.0)
        st, obs = env.reset(jax.random.PRNGKey(seed), prm, ov)
        # only the FIRST episode of each env is scored: an auto-reset would redraw the push size
        alive = np.ones(n, bool)
        ok = np.zeros(n)
        fail = np.zeros(n)
        quiet = np.zeros(n)
        step = jax.jit(env.step)
        for _ in range(n_ticks):
            a = self._greedy(params, stats.normalize(obs))
            st, obs, r, done, info = step(st, jnp.clip(a, -1.0, 1.0), prm)
            ok += np.asarray(info["push_ok"]) * alive
            fail += np.asarray(info["push_fail"]) * alive
            quiet += np.asarray(info["quiet_fall"]) * alive
            alive &= ~np.asarray(done)
            if not alive.any():
                break
        out = {}
        for r_ in rungs:
            sel = dv == r_
            tot = ok[sel].sum() + fail[sel].sum()
            out[f"{r_:.1f}"] = float(ok[sel].sum() / tot) if tot else float("nan")
        vals = [v for v in out.values() if v == v]
        score = float(np.mean(vals)) if vals else 0.0
        return dict(survival=out, score=score, quiet_falls=int(quiet.sum()))

    # ---------------------------------------------------------------- persistence
    def save(self, tag, extra=None):
        base = self.run / tag
        base.with_suffix(".msgpack").write_bytes(serialization.to_bytes(self.params))
        side = dict(step=self.step, rollout_n=self.rollout_n, lr=self.lr, n_upd=self.n_upd, cur=self.cur,
                    stats=dict(mean=np.asarray(self.stats.mean).tolist(), var=np.asarray(self.stats.var).tolist(),
                               count=float(self.stats.count)),
                    config=config_to_dict(self.cfg), **(extra or {}))
        base.with_suffix(".json").write_text(json.dumps(side))
        (self.run / "opt_state.msgpack").write_bytes(serialization.to_bytes(
            dict(opt=self.opt_state, est=self.est_opt_state)))

    def load(self, path, with_opt=True):
        path = Path(path)
        self.params = serialization.from_bytes(self.params, path.with_suffix(".msgpack").read_bytes())
        side = json.loads(path.with_suffix(".json").read_text())
        s = side["stats"]
        self.stats = ObsStats(mean=jnp.asarray(s["mean"], jnp.float32), var=jnp.asarray(s["var"], jnp.float32),
                              count=jnp.asarray(s["count"]))
        self.step, self.rollout_n, self.lr = int(side["step"]), int(side["rollout_n"]), float(side["lr"])
        self.n_upd = int(side.get("n_upd", 10 ** 9))
        self.cur.update(side["cur"])
        op = self.run / "opt_state.msgpack"
        if with_opt and op.exists():
            st = serialization.from_bytes(dict(opt=self.opt_state, est=self.est_opt_state), op.read_bytes())
            self.opt_state, self.est_opt_state = st["opt"], st["est"]
        return side
