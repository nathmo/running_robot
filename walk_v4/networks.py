"""The three networks of §04, in flax, plus the masked Gaussian and the mirror (§06).

    estimator : actor_obs -> 128 -> 64 -> 3          supervised on priv[0:3], detached into the actor
    actor     : [actor_obs, v_hat] -> 256 -> 256 -> mu (A)   + state-independent log_std (A)
    critic    : full obs (actor + priv) -> 256 -> 256 -> V

tanh on every hidden layer INCLUDING the actor's last hidden (walk_mit's `out_act=True`),
linear heads — the same wiring the Pi's numpy runtime expects (robot/deploy/policy_net.py).

Masked log-probability (§04): on a non-commit tick the latched dims of the action never reach
the plant, so they are excluded from the log-prob and the entropy:
    log pi(a|s) = sum_{d free} log N(a_d) + commit(s) * sum_{d latched} log N(a_d)
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
    init_log_std: float = 0.0

    def setup(self):
        self.estimator = MLP(self.est_hidden, 3)
        # policy_net = hidden stack with tanh on the last hidden; action_net = linear head
        self.policy_net = MLP(self.policy_hidden[:-1], self.policy_hidden[-1], out_act=True)
        self.action_net = nn.Dense(self.action_dim)
        self.value_net = MLP(self.policy_hidden[:-1], self.policy_hidden[-1], out_act=True)
        self.value_head = nn.Dense(1)
        self.log_std = self.param("log_std", lambda k: jnp.full((self.action_dim,), self.init_log_std))

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
        mu = self.actor_mean(obs, est)
        return mu, self.log_std, self.value(obs), est


# ------------------------------------------------------------------ masked diagonal Gaussian
def log_prob(mu, log_std, a, mask):
    """Sum over dims of the Gaussian log-density, dims weighted by mask in {0,1}."""
    var = jnp.exp(2.0 * log_std)
    lp = -0.5 * ((a - mu) ** 2 / var + 2.0 * log_std + LOG_2PI)
    return jnp.sum(lp * mask, axis=-1)


def entropy(log_std, mask):
    ent = 0.5 + 0.5 * LOG_2PI + log_std
    return jnp.sum(ent * mask, axis=-1)


def sample(key, mu, log_std):
    return mu + jnp.exp(log_std) * jax.random.normal(key, mu.shape)


def dim_mask(commit, latched_dims):
    """[B, A] mask: free dims 1, latched dims = commit flag of the row."""
    latched = jnp.asarray(latched_dims, jnp.float32)
    c = commit.astype(jnp.float32)[..., None]
    return (1.0 - latched) + latched * c


# ------------------------------------------------------------------ the observation mirror (§06)
class ObsMirror:
    """M_o on the actor observation (policy variant): every history frame mirrored (joints
    L<->R negated, gravity y and gyro x/z negated, LP yaw and the heading estimate negated, the
    phase shifted to phi - pi - Delta with Delta read from the once-block), the once-block's five
    knobs and reflex bias negated. Built once from the env's layout."""

    def __init__(self, env):
        import gait
        from env import FRAME_DIM
        self.n_hist = env.cfg.history_len
        self.frame = FRAME_DIM
        self.actor_dim = env.actor_dim
        self.once0 = self.frame * self.n_hist
        self.library = env.library_mode
        self.delta_max = float(env.cfg.delta_max)
        self.gait = gait

    def __call__(self, obs):
        g = self.gait
        H = obs[..., :self.once0].reshape(obs.shape[:-1] + (self.n_hist, self.frame))
        once = obs[..., self.once0:self.actor_dim]
        Hm = g.mirror_frame(H)
        if self.library:
            # once = [q_ref 6, qd_ref 6, q_ahead 6, phi_td 2, task 2, commit 1]
            sw = lambda x: -x[..., g.MIRROR_PERM]
            once_m = jnp.concatenate([sw(once[..., 0:6]), sw(once[..., 6:12]), sw(once[..., 12:18]),
                                      once[..., 18:20][..., ::-1], once[..., 20:23]], axis=-1)
            delta = jnp.zeros(obs.shape[:-1])
        else:
            spec = once[..., :g.SPEC_DIM]
            delta = self.delta_max * jnp.clip(spec[..., g.I_DELTA], -1.0, 1.0)
            spec_m = spec.at[..., 38].multiply(-1.0)
            for i in g.KNOB_IDX:
                spec_m = spec_m.at[..., i].multiply(-1.0)
            once_m = jnp.concatenate([spec_m, once[..., g.SPEC_DIM:]], axis=-1)
        ph = g.mirror_phase(Hm[..., 25:27], delta[..., None])
        Hm = jnp.concatenate([Hm[..., :25], ph, Hm[..., 27:]], axis=-1)
        return jnp.concatenate([Hm.reshape(obs.shape[:-1] + (self.once0,)), once_m], axis=-1)


def knob_slice(mu, library):
    """The five relationship knobs in the actor MEAN (policy variant); none in library mode."""
    if library:
        return jnp.zeros(mu.shape[:-1] + (0,))
    return mu[..., 39:44]
