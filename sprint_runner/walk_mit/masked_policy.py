"""Masked-log-prob policy for the LATCHED gait spec (artifact §04, "credit assignment under the latch").

Under the latch, the spec dims of the action reach the plant only on the tick where the gait phase
wraps; on every other tick they are no-ops. Plain PPO would still sample them, still score their
log-probability into the importance ratio, and still hand them that tick's advantage -- gradient
noise on ~90% of the action, ~97% of the time (44 of 50 dims, 32 of 33 ticks at 3 Hz / 100 Hz).

This policy makes the latched dims part of the action ONLY on commit ticks:

    log pi(a|s) = sum_{d in per-tick} log N(a_d)  +  commit(s) * sum_{d in latched} log N(a_d)

and masks the entropy the same way. `commit(s)` is read from the observation itself (index
`wrap_index`, the last entry of the env's once-block): the env sets the flag when the clock wrapped
while advancing to this tick and latches the spec from the action of exactly that tick. Reading
it from obs means rollout (forward) and update (evaluate_actions) use the same mask with no
rollout-buffer plumbing, and the policy also SEES when its spec emission counts.

The flag survives VecNormalize: a {0,1} variable z-scores to (0-p)/sigma < 0 and (1-p)/sigma > 0
for any running mean p in (0,1), and at the untouched start (mean 0, var 1) it is 0 and 1 -- so
`> 0` recovers it in both raw and normalized space. Deployment is untouched: greedy actions use
the mean, the mask only shapes training statistics.

Two layouts, one mechanism: the policy variant latches action[0:44] (spec_start 0), the library
variant latches the 3 trailing mods action[6:9] (spec_start 6, spec_dims 3).
"""
import torch as th
from stable_baselines3.common.distributions import DiagGaussianDistribution

from asym_policy import AsymmetricACPolicy


class MaskedAsymmetricACPolicy(AsymmetricACPolicy):
    """AsymmetricACPolicy whose log-prob / entropy exclude the latched dims on non-commit ticks.

    Extra kwargs (persisted via SB3's policy_kwargs round-trip):
        spec_dims   width of the latched block of the action
        wrap_index  index in the observation vector of the commit flag ({0,1}, once-block)
        spec_start  first index of the latched block (0 = the front of the action)
    """

    def __init__(self, *args, spec_dims: int = None, wrap_index: int = None, spec_start: int = 0,
                 **kwargs):
        if spec_dims is None or wrap_index is None:
            raise TypeError("MaskedAsymmetricACPolicy needs spec_dims and wrap_index")
        self._spec_dims = int(spec_dims)
        self._spec_start = int(spec_start)
        self._wrap_index = int(wrap_index)
        super().__init__(*args, **kwargs)

    # ---- the mechanism ------------------------------------------------------------------------
    def _dim_mask(self, obs: th.Tensor, n_act: int) -> th.Tensor:
        """[B, n_act] in {0,1}: per-tick dims always 1, latched dims = commit flag of that row."""
        commit = (obs[..., self._wrap_index] > 0.0).to(obs.dtype).unsqueeze(-1)
        m = th.ones(obs.shape[0], n_act, dtype=obs.dtype, device=obs.device)
        m[:, self._spec_start:self._spec_start + self._spec_dims] = commit
        return m

    def _masked_stats(self, dist, actions: th.Tensor, obs: th.Tensor):
        assert isinstance(dist, DiagGaussianDistribution), "mask is written for the diagonal Gaussian"
        m = self._dim_mask(obs, actions.shape[-1])
        log_prob = (dist.distribution.log_prob(actions) * m).sum(-1)
        entropy = (dist.distribution.entropy() * m).sum(-1)
        return log_prob, entropy

    # ---- SB3 surfaces, verbatim except for the two masked lines --------------------------------
    def _latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        pi_f, vf_f = features
        return self.mlp_extractor.forward_actor(pi_f), self.mlp_extractor.forward_critic(vf_f)

    def forward(self, obs, deterministic: bool = False):
        latent_pi, latent_vf = self._latents(obs)
        values = self.value_net(latent_vf)
        dist = self._get_action_dist_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob, _ = self._masked_stats(dist, actions, obs)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    def evaluate_actions(self, obs, actions):
        latent_pi, latent_vf = self._latents(obs)
        dist = self._get_action_dist_from_latent(latent_pi)
        log_prob, entropy = self._masked_stats(dist, actions, obs)
        values = self.value_net(latent_vf)
        return values, log_prob, entropy


# ---- self-test: the mechanism, numerically -----------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import gymnasium as gym

    n_actor, n_priv, n_act, spec = 12, 3, 8, 5
    wrap = n_actor - 1                                  # last entry of the actor slice
    pol = MaskedAsymmetricACPolicy(
        gym.spaces.Box(-np.inf, np.inf, (n_actor + n_priv,), np.float32),
        gym.spaces.Box(-1.0, 1.0, (n_act,), np.float32),
        lambda _: 3e-4, net_arch=[16],
        n_actor_obs=n_actor, n_priv=n_priv, spec_dims=spec, wrap_index=wrap)
    th.manual_seed(0)
    obs = th.randn(4, n_actor + n_priv)
    obs[:, wrap] = th.tensor([0.0, 1.0, 0.0, 1.0])       # rows 1,3 are commit ticks
    act = th.randn(4, n_act)
    NO, YES = [0, 2], [1, 3]

    # 1. on non-commit ticks the spec-dim actions are invisible to the log-prob
    _, lp, ent = pol.evaluate_actions(obs, act)
    act2 = act.clone(); act2[:, :spec] += 3.0
    _, lp2, _ = pol.evaluate_actions(obs, act2)
    d = (lp2 - lp).detach()
    assert th.allclose(d[NO], th.zeros(2)), d
    assert (d[YES].abs() > 1e-3).all(), d

    # 2. no gradient reaches the spec parameters from non-commit ticks; it does from commit ticks
    pol.zero_grad(); lp[NO].sum().backward()
    g = pol.log_std.grad.clone()
    assert th.allclose(g[:spec], th.zeros(spec)) and (g[spec:].abs() > 0).any(), g
    pol.zero_grad(); _, lp_, _ = pol.evaluate_actions(obs, act); lp_[YES].sum().backward()
    assert (pol.log_std.grad[:spec].abs() > 0).all(), pol.log_std.grad

    # 3. entropy is masked the same way
    per_dim = pol.log_std.detach() + 0.5 + 0.5 * np.log(2 * np.pi)
    assert th.allclose(ent[0], per_dim[spec:].sum()) and th.allclose(ent[1], per_dim.sum())

    # 4. rollout and update agree (the ratio is exactly 1 for unchanged parameters)
    a, _, lp_roll = pol.forward(obs)
    _, lp_upd, _ = pol.evaluate_actions(obs, a)
    assert th.allclose(th.exp(lp_upd - lp_roll), th.ones(4), atol=1e-5)

    # 5. the flag survives VecNormalize-style z-scoring
    p = 1.0 / 33.0; sig = float(np.sqrt(p * (1 - p)))
    z = th.tensor([(0 - p) / sig, (1 - p) / sig])
    assert bool(z[0] <= 0.0) and bool(z[1] > 0.0)

    # 6. a trailing latched block (the library variant's 3 mods) masks the tail, not the head
    pol2 = MaskedAsymmetricACPolicy(
        gym.spaces.Box(-np.inf, np.inf, (n_actor + n_priv,), np.float32),
        gym.spaces.Box(-1.0, 1.0, (9,), np.float32), lambda _: 3e-4, net_arch=[16],
        n_actor_obs=n_actor, n_priv=n_priv, spec_dims=3, wrap_index=wrap, spec_start=6)
    m = pol2._dim_mask(obs, 9)
    assert th.equal(m[:, :6], th.ones(4, 6)) and th.equal(m[:, 6:], obs[:, wrap:wrap + 1].gt(0).float().expand(4, 3))

    ticks, spec_d, res_d = 33, 44, 6
    print("masked_policy self-test OK")
    print("  at %d ticks/cycle, %d spec + %d residual dims:" % (ticks, spec_d, res_d))
    print("    unmasked: %.0f%% of sampled action dims are no-ops that still get an advantage"
          % (100 * spec_d * (ticks - 1) / (ticks * (spec_d + res_d))))
    print("    masked:   spec dims scored on 1/%d ticks, residual on all -- no-op dims: 0%%" % ticks)
