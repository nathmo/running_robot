"""Load a trained balance checkpoint for evaluation / export, without the trainer.

    cfg, env, pol = load_policy("BalanceRL/runs/bal_s0", "best", n_envs=64)
    a = pol.act(obs)          # greedy, jitted, clipped to [-1, 1]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np                                    # noqa: E402
import jax                                            # noqa: E402
import jax.numpy as jnp                               # noqa: E402
from flax import serialization                        # noqa: E402

import networks as nets
from networks import init_log_std                               # noqa: E402
from config import config_from_dict                   # noqa: E402
from env import BalanceEnv                            # noqa: E402


def resolve_ckpt(run, ckpt=None):
    run = Path(run)
    if ckpt is None:
        for n in ("best", "final"):
            if (run / f"{n}.json").exists():
                return run / n
        c = sorted(run.glob("ckpt_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        if not c:
            raise FileNotFoundError(f"no checkpoint in {run}")
        return c[-1].with_suffix("")
    p = Path(ckpt)
    if not p.suffix and not p.parent.name:
        p = run / p
    return p.with_suffix("") if p.suffix in (".json", ".msgpack") else p


class Policy:
    def __init__(self, net, params, mean, var, eps=1e-8, clip=10.0):
        self.net, self.params = net, params
        self.mean, self.std = jnp.asarray(mean, jnp.float32), jnp.sqrt(jnp.asarray(var, jnp.float32) + eps)
        self.clip = clip
        self._act = jax.jit(lambda p, o: jnp.clip(net.apply(p, jnp.clip((o - self.mean) / self.std, -clip, clip),
                                                            method=net.actor_mean), -1.0, 1.0))

    def act(self, obs):
        return self._act(self.params, obs)


def load_policy(run, ckpt=None, n_envs=1, **cfg_over):
    path = resolve_ckpt(run, ckpt)
    side = json.loads(path.with_suffix(".json").read_text())
    cfg = config_from_dict(side["config"])
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    env = BalanceEnv(cfg, n_envs)
    net = nets.ActorCritic(n_actor=env.actor_dim, n_priv=env.priv_dim, action_dim=env.action_dim,
                           policy_hidden=tuple(cfg.policy_hidden), est_hidden=tuple(cfg.est_hidden),
                           init_log_std=init_log_std(cfg))
    tmpl = net.init(jax.random.PRNGKey(0), jnp.zeros((1, env.obs_dim)))
    params = serialization.from_bytes(tmpl, path.with_suffix(".msgpack").read_bytes())
    s = side["stats"]
    pol = Policy(net, params, s["mean"], s["var"], cfg.obs_eps, cfg.clip_obs)
    pol.side, pol.path = side, path
    return cfg, env, pol
