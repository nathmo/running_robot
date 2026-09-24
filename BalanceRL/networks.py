"""Actor-critic for the balance policy, in flax. Wired EXACTLY like RLframework/networks.py so the
Pi's numpy runtime (controller/deploy/policy_net.py) runs it unchanged:

    estimator : actor_obs -> 128 -> 64 -> 3            supervised on the true body velocity
    actor     : [actor_obs, v_hat] -> 256 -> 256 -> mu (18), tanh on BOTH hidden layers
    critic    : actor_obs + privileged tail -> 256 -> 256 -> V
    log_std   : state-independent (18)
"""
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

LOG_2PI = float(np.log(2.0 * np.pi))


class MLP(nn.Module):
    sizes: Sequence[int]
    out: int
    out_act: bool = False

    @nn.compact
    def __call__(self, x):
        for h in self.sizes:
            x = jnp.tanh(nn.Dense(h)(x))
        x = nn.Dense(self.out)(x)
        return jnp.tanh(x) if self.out_act else x


class ActorCritic(nn.Module):
    n_actor: int
    n_priv: int
    action_dim: int
    policy_hidden: Sequence[int] = (256, 256)
    est_hidden: Sequence[int] = (128, 64)
    init_log_std: Sequence[float] = (-2.3,)

    def setup(self):
        self.estimator = MLP(self.est_hidden, 3)
        self.policy_net = MLP(self.policy_hidden[:-1], self.policy_hidden[-1], out_act=True)
        # small head init: the initial mean is ~0 = the standing command
        self.action_net = nn.Dense(self.action_dim, kernel_init=nn.initializers.normal(0.01),
                                   bias_init=nn.initializers.zeros)
        self.value_net = MLP(self.policy_hidden[:-1], self.policy_hidden[-1], out_act=True)
        self.value_head = nn.Dense(1)
        init = jnp.broadcast_to(jnp.asarray(self.init_log_std, jnp.float32), (self.action_dim,))
        self.log_std = self.param("log_std", lambda k: init)

    def estimate(self, obs):
        return self.estimator(obs[..., :self.n_actor])

    def actor_mean(self, obs, est=None):
        a = obs[..., :self.n_actor]
        est = self.estimator(a) if est is None else est
        h = self.policy_net(jnp.concatenate([a, jax.lax.stop_gradient(est)], axis=-1))
        return self.action_net(h)

    def value(self, obs):
        return self.value_head(self.value_net(obs[..., :self.n_actor + self.n_priv]))[..., 0]

    def __call__(self, obs):
        est = self.estimate(obs)
        return self.actor_mean(obs, est), self.log_std, self.value(obs), est


def log_prob(mu, log_std, a):
    var = jnp.exp(2.0 * log_std)
    return jnp.sum(-0.5 * ((a - mu) ** 2 / var + 2.0 * log_std + LOG_2PI), axis=-1)


def entropy(log_std):
    return jnp.sum(0.5 + 0.5 * LOG_2PI + log_std, axis=-1)


def sample(key, mu, log_std):
    return mu + jnp.exp(log_std) * jax.random.normal(key, mu.shape)


def init_log_std(cfg):
    """Per-dim initial log std: position dims cfg.init_std, gain dims cfg.init_std_gain."""
    return tuple([float(np.log(cfg.init_std))] * 6 + [float(np.log(cfg.init_std_gain))] * 12)


def std_clamp(cfg):
    """(lo, hi) log-std per dim. The two families live on different scales: 0.02 of a position dim is
    0.01 rad (half the standing basin), while 0.02 of a gain dim is nothing at all."""
    lo = [float(np.log(cfg.min_std))] * 6 + [float(np.log(cfg.min_std_gain))] * 12
    hi = [float(np.log(cfg.max_std))] * 6 + [float(np.log(cfg.max_std_gain))] * 12
    return np.array(lo, np.float32), np.array(hi, np.float32)
