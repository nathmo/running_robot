"""Train a SpiderBot locomotion policy with PPO on GPU via MuJoCo MJX + Brax's low-level PPO
primitives (NOT brax.training.agents.ppo.train.train() -- see below for why).

Examples:
  .venv/bin/python -m rl.mjx_train --preset m1_stand --steps 8000000 --n-envs 2048
  .venv/bin/python -m rl.mjx_train --preset m2_walk --steps 20000000 --n-envs 4096

Outputs go to rl/runs/<name>/ : progress.csv (plot_training.py-compatible), training_plots.png,
and checkpoint_<steps>.pkl / final.pkl (raw Flax params -- see rl/mjx_export.py to convert to an
SB3-loadable checkpoint that rl/evaluate.py, rl/gait_probe.py, rl/joystick.py can use unchanged).

Why not brax.training.agents.ppo.train.train(): that ~700-line function bakes entropy_cost (and
everything except its own adaptive-KL learning rate) into a python-closure loss function traced
ONCE for the whole run -- confirmed by reading the installed brax/training/agents/ppo/train.py.
This project needs THREE genuinely dynamic per-iteration values SB3's callbacks provided
(entropy_cost anneal, log_std clamp, curriculum cmd_vx_frac), none of which train() exposes a
hook for. So this hand-rolls the training loop on top of the same composable pieces train() is
itself built from: brax.training.agents.ppo.{networks,losses}, brax.training.gradients,
brax.training.acme.running_statistics. rl/mjx_env.py is the env; nothing here reimplements physics
or reward math.

Known simplifications vs rl/train.py (SB3), both deliberate scope cuts, not oversights:
  - Minibatches are subsets of PARALLEL ENVS (each contributing a full unroll_length trajectory),
    not a flat shuffle of (env,time) pairs. This is brax's/compute_ppo_loss's native batching (GAE
    is recomputed fresh inside every loss call, using whichever minibatch of trajectories was
    selected), not SB3's precompute-advantages-once-then-SGD recipe. Both are standard PPO
    variants; this one is what compute_ppo_loss is designed for.
  - No target_kl early-stopping within an epoch (SB3 aborts the remaining minibatches of an epoch
    if KL exceeds cfg.target_kl). compute_ppo_loss has no hook for it, and mid-scan conditional
    early-exit would need real complexity (lax.cond-gated minibatch skipping) for a safety net,
    not a correctness requirement -- if clip_fraction/approx_kl run away in practice, this needs
    revisiting before trusting long runs.
"""
import argparse
import csv
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import linen

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import losses as ppo_losses
from brax.training import gradients
from brax.training import types as brax_types
from brax.training.acme import running_statistics

from .config import get_config
from .mjx_env import SpiderBotMjxEnv, TERM_NAMES


def make_networks(env, cfg):
    return ppo_networks.make_ppo_networks(
        observation_size=env.obs_size, action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(cfg.policy_hidden),
        value_hidden_layer_sizes=tuple(cfg.policy_hidden),
        activation=linen.tanh,
        # 'normal' + state_dependent_std=False + noise_std_type='log' gives a plain MLP trunk ->
        # mean head (Dense) + a SEPARATE state-independent log_std parameter -- structurally
        # identical to SB3's MlpPolicy (net_arch=[256,256], Tanh, DiagGaussianDistribution), which
        # is what makes rl/mjx_export.py's weight copy exact rather than approximate.
        distribution_type="normal", noise_std_type="log",
        init_noise_std=1.0, state_dependent_std=False,
    )


def act(nets, normalizer_params, policy_params, obs, rng, deterministic=False):
    """Mirrors brax.training.agents.ppo.networks.make_inference_fn's internal policy() -- written
    directly (not via make_inference_fn) to avoid its (normalizer, policy, value)-tuple params
    convention, which differs from ppo_losses.PPONetworkParams(policy=, value=) used everywhere
    else here."""
    logits = nets.policy_network.apply(normalizer_params, policy_params, obs)
    dist = nets.parametric_action_distribution
    if deterministic:
        return dist.mode(logits), {}
    raw_action = dist.sample_no_postprocessing(logits, rng)
    log_prob = dist.log_prob(logits, raw_action)
    action = dist.postprocess(raw_action)
    extras = {"log_prob": log_prob, "raw_action": raw_action, "distribution_params": logits}
    return action, extras


def make_rollout_fn(env, nets, n_envs, unroll_length):
    def rollout(env_state, policy_params, normalizer_params, rng):
        def step_fn(carry, _):
            state, rng = carry
            rng, k_act, k_step = jax.random.split(rng, 3)
            act_keys = jax.random.split(k_act, n_envs)
            action, pextras = jax.vmap(
                lambda o, k: act(nets, normalizer_params, policy_params, o, k)
            )(state.obs, act_keys)
            next_state = jax.vmap(env.step)(state, action)
            transition = dict(
                observation=state.obs, action=action, reward=next_state.reward,
                discount=1.0 - next_state.terminated.astype(jnp.float32),
                next_observation=next_state.obs,
                truncation=next_state.truncated.astype(jnp.float32),
                policy_extras=pextras,
                reward_terms=next_state.reward_terms,
                done=next_state.done, ep_return=next_state.final_ep_return,
                ep_length=next_state.final_ep_length,
            )
            return (next_state, rng), transition

        (final_state, rng), transitions = jax.lax.scan(
            step_fn, (env_state, rng), None, length=unroll_length)
        return final_state, transitions, rng

    return rollout


def make_train_step(env, nets, cfg, optimizer, n_envs, unroll_length, num_minibatches):
    def loss_fn(params, normalizer_params, data, rng, entropy_cost):
        return ppo_losses.compute_ppo_loss(
            params, normalizer_params, data, rng, nets,
            entropy_cost=entropy_cost, discounting=cfg.gamma, reward_scaling=1.0,
            gae_lambda=cfg.gae_lambda, clipping_epsilon=cfg.clip_range,
            normalize_advantage=True, vf_coefficient=0.5)

    grad_update = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=None, has_aux=True)

    rollout_fn = make_rollout_fn(env, nets, n_envs, unroll_length)
    envs_per_minibatch = n_envs // num_minibatches

    def mb_step(carry, mb):
        params, opt_state, normalizer_params, entropy_cost, rng = carry
        rng, k_loss = jax.random.split(rng)
        (_, metrics), params, opt_state = grad_update(
            params, normalizer_params, mb, k_loss, entropy_cost, optimizer_state=opt_state)
        return (params, opt_state, normalizer_params, entropy_cost, rng), metrics

    def epoch(carry, _):
        params, opt_state, normalizer_params, ppo_data, entropy_cost, rng = carry
        rng, k_perm = jax.random.split(rng)
        perm = jax.random.permutation(k_perm, n_envs)
        shuffled = jax.tree_util.tree_map(lambda x: x[perm], ppo_data)  # shuffle the ENV axis (0)

        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_minibatches, envs_per_minibatch) + x.shape[1:]), shuffled)
        (params, opt_state, normalizer_params, entropy_cost, rng), epoch_metrics = jax.lax.scan(
            mb_step, (params, opt_state, normalizer_params, entropy_cost, rng), minibatches,
            length=num_minibatches)
        return (params, opt_state, normalizer_params, ppo_data, entropy_cost, rng), epoch_metrics

    def train_step(env_state, params, opt_state, normalizer_params, rng, entropy_cost):
        rng, k_rollout = jax.random.split(rng)
        env_state, transitions, _ = rollout_fn(env_state, params.policy, normalizer_params, k_rollout)

        obs_flat = transitions["observation"].reshape(-1, env.obs_size)
        normalizer_params = running_statistics.update(
            normalizer_params, obs_flat, pmap_axis_name=None)

        ppo_data = brax_types.Transition(
            observation=transitions["observation"], action=transitions["action"],
            reward=transitions["reward"], discount=transitions["discount"],
            next_observation=transitions["next_observation"],
            extras={"state_extras": {"truncation": transitions["truncation"]},
                    "policy_extras": transitions["policy_extras"]},
        )
        # [T,B,...] -> [B,T,...]: compute_ppo_loss's own internal swap expects batch-first input.
        ppo_data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), ppo_data)

        (params, opt_state, normalizer_params, _, _, rng), epoch_metrics = jax.lax.scan(
            epoch, (params, opt_state, normalizer_params, ppo_data, entropy_cost, rng), None,
            length=cfg.n_epochs)

        # log_std clamp (EntropyCallback's other job): std <= exp(max_log_std). Uses
        # tree_map_with_path (not manual dict-rebuilding) so the original pytree container type
        # (FrozenDict or plain dict, whichever flax.init() produced) is preserved exactly --
        # rebuilding it by hand risks a pytree-structure mismatch inside the lax.scan carry.
        params = params.replace(policy=jax.tree_util.tree_map_with_path(
            lambda path, x: jnp.minimum(x, cfg.max_log_std)
            if any("std_logparam" in str(p) for p in path) else x,
            params.policy))

        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), epoch_metrics)
        return env_state, params, opt_state, normalizer_params, rng, transitions, metrics

    return jax.jit(train_step)


def curriculum_value(cfg, num_timesteps):
    if cfg.curriculum_steps <= 0 or cfg.cmd_vx_frac <= cfg.cmd_vx_frac_start:
        return cfg.cmd_vx_frac
    frac = min(1.0, num_timesteps / cfg.curriculum_steps)
    return cfg.cmd_vx_frac_start + frac * (cfg.cmd_vx_frac - cfg.cmd_vx_frac_start)


class EntropySchedule:
    """Host-side (plain python) port of train.py's EntropyCallback: hold ent_coef until stepping
    has emerged (rollout-mean reward_terms/air_time clears a gate for `patience` consecutive
    rollouts, or -- for stand-only presets where air_time is structurally zero -- mean completed-
    episode length clears 90% of the episode cap), then anneal linearly to ent_final. Runs once
    per training iteration (this project's unit of "rollout"), same cadence as the SB3 callback's
    _on_rollout_end."""

    def __init__(self, cfg, patience=5):
        self.cfg, self.patience = cfg, patience
        self.streak = 0
        self.anneal_from = None
        self.anneal_base = None
        self.stand_only = cfg.cmd_vx_frac * cfg.vx_max < cfg.gait_cmd_gate

    def _competent(self, air_mean, ep_len_mean, max_steps):
        if self.stand_only:
            return ep_len_mean > 0.9 * max_steps
        return air_mean > self.cfg.ent_gate_air_time

    def step(self, num_timesteps, air_mean, ep_len_mean, max_steps):
        cfg = self.cfg
        if self.anneal_from is None:
            self.streak = self.streak + 1 if self._competent(air_mean, ep_len_mean, max_steps) else 0
            if self.streak >= self.patience:
                self.anneal_from = num_timesteps
                self.anneal_base = cfg.ent_coef
                print(f"[train] competence gate open (air={air_mean:.3f} ep_len={ep_len_mean:.0f}) "
                      f"at {num_timesteps} steps -> annealing ent_coef {self.anneal_base} -> "
                      f"{cfg.ent_final}")
        if self.anneal_from is None:
            return cfg.ent_coef
        frac = min(1.0, (num_timesteps - self.anneal_from) / max(cfg.ent_anneal_steps, 1))
        return self.anneal_base + frac * (cfg.ent_final - self.anneal_base)


class CsvLogger:
    """Writes rl/plot_training.py's expected schema (same column names rl/train.py's SB3
    `configure(..., ["csv"])` logger produces) so runs from this trainer are plottable with the
    existing, unmodified plot_training.py."""

    FIELDS = ["time/total_timesteps", "rollout/ep_rew_mean", "rollout/ep_len_mean",
              "train/loss", "train/value_loss", "train/policy_gradient_loss",
              "train/entropy_loss", "train/approx_kl", "train/std",
              "curriculum/cmd_vx_frac", "curriculum/ent_coef"]

    def __init__(self, run_dir, term_names):
        self.path = Path(run_dir) / "progress.csv"
        self.fields = self.FIELDS + [f"reward_terms/{k}" for k in term_names]
        self._fh = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=self.fields)
        self._w.writeheader()

    def log(self, row):
        self._w.writerow({k: row.get(k, "") for k in self.fields})
        self._fh.flush()

    def close(self):
        self._fh.close()


def save_checkpoint(path, params, normalizer_params, preset):
    with open(path, "wb") as f:
        pickle.dump({
            "policy_params": jax.device_get(params.policy),
            "value_params": jax.device_get(params.value),
            "normalizer_params": jax.device_get(normalizer_params),
            "preset": preset,
        }, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="m1_stand")
    ap.add_argument("--name", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=2048)
    ap.add_argument("--num-minibatches", type=int, default=32)
    ap.add_argument("--checkpoint-every", type=int, default=2_000_000)
    args = ap.parse_args()

    cfg = get_config(args.preset)
    total_steps = args.steps or cfg.total_steps
    n_envs = args.n_envs
    unroll_length = cfg.n_steps
    name = args.name or args.preset
    run = Path("rl/runs") / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "preset.json").write_text(json.dumps({"preset": args.preset, "backend": "mjx"}))

    env = SpiderBotMjxEnv(cfg)
    nets = make_networks(env, cfg)

    key = jax.random.PRNGKey(cfg.seed)
    key, k_policy, k_value, k_env = jax.random.split(key, 4)
    params = ppo_losses.PPONetworkParams(
        policy=nets.policy_network.init(k_policy), value=nets.value_network.init(k_value))
    normalizer_params = running_statistics.init_state(jnp.zeros(env.obs_size))

    steps_per_iter = n_envs * unroll_length
    total_iters = max(1, total_steps // steps_per_iter)
    total_grad_steps = total_iters * cfg.n_epochs * args.num_minibatches
    lr_schedule = optax.linear_schedule(cfg.learning_rate, cfg.lr_final, total_grad_steps)
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(learning_rate=lr_schedule))
    opt_state = optimizer.init(params)

    train_step = make_train_step(env, nets, cfg, optimizer, n_envs, unroll_length,
                                 args.num_minibatches)

    env_keys = jax.random.split(k_env, n_envs)
    env_state = jax.jit(jax.vmap(env.reset))(env_keys, jnp.zeros(n_envs))

    entropy_sched = EntropySchedule(cfg)
    logger = CsvLogger(run, TERM_NAMES)

    print(f"[train] preset={args.preset} n_envs={n_envs} unroll_length={unroll_length} "
          f"total_steps={total_steps} -> {total_iters} iterations -> {run}")

    # entropy gate/anneal reads the PREVIOUS iteration's rollout-averaged stats (a one-iteration
    # lag) -- matches SB3's EntropyCallback._on_rollout_end firing AFTER a rollout completes, so
    # its decision takes effect starting the NEXT rollout, not the one just finished.
    prev_air_mean, prev_ep_len_mean = 0.0, 0.0
    num_timesteps = 0
    for it in range(total_iters):
        cmd_vx_frac = curriculum_value(cfg, num_timesteps)
        env_state = env_state.replace(cmd_vx_frac=jnp.full((n_envs,), cmd_vx_frac))
        ent_coef = entropy_sched.step(
            num_timesteps, air_mean=prev_air_mean, ep_len_mean=prev_ep_len_mean,
            max_steps=env.max_steps,
        )
        key, k_step = jax.random.split(key)
        (env_state, params, opt_state, normalizer_params, key, transitions, metrics
         ) = train_step(env_state, params, opt_state, normalizer_params, k_step,
                        jnp.asarray(ent_coef))
        num_timesteps += steps_per_iter

        done_mask = transitions["done"]
        n_done = float(jnp.sum(done_mask))
        ep_rew_mean = (float(jnp.sum(jnp.where(done_mask, transitions["ep_return"], 0.0)) / n_done)
                       if n_done > 0 else float("nan"))
        ep_len_mean = (float(jnp.sum(jnp.where(done_mask, transitions["ep_length"], 0)) / n_done)
                       if n_done > 0 else float("nan"))
        term_means = {k: float(jnp.mean(v)) for k, v in transitions["reward_terms"].items()}
        prev_air_mean = term_means.get("air_time", 0.0)
        prev_ep_len_mean = ep_len_mean if ep_len_mean == ep_len_mean else prev_ep_len_mean  # skip NaN
        log_std_mean = float(jnp.mean(jnp.exp(
            params.policy["params"]["std_logparam"]["log_value"])))

        row = {
            "time/total_timesteps": num_timesteps,
            "rollout/ep_rew_mean": ep_rew_mean, "rollout/ep_len_mean": ep_len_mean,
            "train/loss": float(metrics["total_loss"]), "train/value_loss": float(metrics["v_loss"]),
            "train/policy_gradient_loss": float(metrics["policy_loss"]),
            "train/entropy_loss": float(metrics["entropy_loss"]),
            "train/approx_kl": float(metrics["kl_mean"]), "train/std": log_std_mean,
            "curriculum/cmd_vx_frac": cmd_vx_frac, "curriculum/ent_coef": ent_coef,
        }
        row.update({f"reward_terms/{k}": v for k, v in term_means.items()})
        logger.log(row)

        print(f"[train] iter {it+1}/{total_iters} steps={num_timesteps} "
              f"ep_rew_mean={ep_rew_mean:.1f} ep_len_mean={ep_len_mean:.0f} "
              f"loss={row['train/loss']:.3f} std={log_std_mean:.3f} "
              f"ent_coef={ent_coef:.4f} cmd_vx_frac={cmd_vx_frac:.3f}")

        if (it + 1) * steps_per_iter // args.checkpoint_every > it * steps_per_iter // args.checkpoint_every:
            save_checkpoint(run / f"checkpoint_{num_timesteps}.pkl", params, normalizer_params,
                            args.preset)

    save_checkpoint(run / "final.pkl", params, normalizer_params, args.preset)
    logger.close()
    try:
        from .plot_training import plot_run
        plots = plot_run(run)
    except Exception as e:
        plots = f"(skipped: {e})"
    print(f"[train] done -> {run/'final.pkl'}  plots -> {plots}")


if __name__ == "__main__":
    main()
