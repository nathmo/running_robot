"""Asymmetric actor-critic policy with a concurrent velocity-estimator head (2026-08-10).

Three networks over ONE observation vector (SB3 shares a single obs between actor and critic, so
the split happens here, by slicing):

    obs = [ actor block: history frames, frame_dim x history_len ][ privileged tail: PRIV_DIM ]

  ESTIMATOR  actor block (550) -> 128 -> 64 -> 3      predicts the (VecNormalize-scaled) true base
             velocity. Trained SUPERVISED against priv tail [0:3] by EstimatorCallback in train.py
             -- NOT by the PPO loss. Its output feeds the actor DETACHED, DreamWaQ-style, so PPO
             gradients cannot bend the estimator toward whatever the actor wishes were true.
  ACTOR      actor block + estimate (553) -> policy_hidden -> action dist    deployable: nothing in
             its input requires simulation. On the robot: obs550 -> estimator -> concat -> actor.
  CRITIC     FULL obs (556) -> policy_hidden -> V(s)    train-time only, so it gets ground truth:
             true velocity, foot contacts, height error. A value function estimating the return of
             a velocity-TRACKING task should not have to infer velocity through a sensor model.

Why: obs_base_vel=False is correct for the actor (the hardware has no velocimeter), but it also
blinded the critic. Velocity is only R^2 ~ 0.807 recoverable from the history, so ~20% of the
tracking return's driving state was pure noise in every advantage estimate. Standard fix across the
SOTA velocity-command stacks (Rudin et al. 2022, RMA, DreamWaQ): privileged critic + an explicit
estimator for the actor.

Checkpoint note: SB3 pickles the policy by class reference, so loading these checkpoints needs
training/ on sys.path (evaluate.py and every probe already do this).
"""
from typing import List

import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy

EST_HIDDEN = [128, 64]      # estimator MLP; small on purpose -- it must run on the Pi at 200 Hz
EST_OUT = 3                 # body-frame vx, vy, vz (VecNormalize-scaled, like everything else)


def _mlp(sizes, activation, out_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            layers.append(activation())
    return nn.Sequential(*layers)


class AsymExtractor(nn.Module):
    """Drop-in for SB3's MlpExtractor: same forward()/forward_actor()/forward_critic() surface,
    different wiring. `features` is the flattened full observation."""

    def __init__(self, n_actor: int, n_priv: int, net_arch: List[int], activation):
        super().__init__()
        self.n_actor, self.n_priv = int(n_actor), int(n_priv)
        self.latent_dim_pi = net_arch[-1]
        self.latent_dim_vf = net_arch[-1]
        self.estimator = _mlp([self.n_actor] + EST_HIDDEN + [EST_OUT], activation)
        self.policy_net = _mlp([self.n_actor + EST_OUT] + list(net_arch), activation, out_act=True)
        self.value_net = _mlp([self.n_actor + self.n_priv] + list(net_arch), activation,
                              out_act=True)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        a = features[..., :self.n_actor]
        est = self.estimator(a)
        # detach: the actor may USE the estimate but PPO must not TRAIN it -- the estimator answers
        # to the supervised target alone, otherwise it drifts into an actor-flattering feature
        return self.policy_net(th.cat([a, est.detach()], dim=-1))

    def forward_critic(self, features):
        return self.value_net(features)

    def estimator_loss(self, features):
        """Supervised MSE against priv tail [0:3] (the scaled true base velocity). Called by
        EstimatorCallback on rollout-buffer observations; gradients touch ONLY self.estimator."""
        a = features[..., :self.n_actor]
        target = features[..., self.n_actor:self.n_actor + EST_OUT]
        return nn.functional.mse_loss(self.estimator(a), target)


class AsymmetricACPolicy(ActorCriticPolicy):
    """ActorCriticPolicy whose mlp_extractor is the asymmetric one above.

    Extra kwargs (persisted in the checkpoint via SB3's policy_kwargs round-trip):
        n_actor_obs  width of the actor's slice (frame_dim * history_len)
        n_priv       width of the privileged tail (DashEnv.PRIV_DIM)
    """

    def __init__(self, *args, n_actor_obs: int = None, n_priv: int = None, **kwargs):
        if n_actor_obs is None or n_priv is None:
            raise TypeError("AsymmetricACPolicy needs n_actor_obs and n_priv")
        self._n_actor_obs = int(n_actor_obs)
        self._n_priv = int(n_priv)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        expected = self._n_actor_obs + self._n_priv
        if self.features_dim != expected:
            raise ValueError(
                f"obs is {self.features_dim} wide but n_actor_obs+n_priv = {expected} — the env "
                f"and the policy disagree about where the privileged tail starts")
        net_arch = self.net_arch if isinstance(self.net_arch, list) else self.net_arch["pi"]
        self.mlp_extractor = AsymExtractor(self._n_actor_obs, self._n_priv,
                                           net_arch, self.activation_fn)

    def estimated_velocity(self, obs):
        """The estimator's readout for obs (numpy in, numpy out) — telemetry/deployment hook."""
        with th.no_grad():
            t = th.as_tensor(np.asarray(obs, np.float32)).to(self.device)
            return self.mlp_extractor.estimator(t[..., :self._n_actor_obs]).cpu().numpy()
