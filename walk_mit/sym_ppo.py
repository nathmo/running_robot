"""PPO + the mirror-symmetry (equivariance) loss of the DASH-01 Walker v2 design (artifact §06).

    L = L_PPO + w_sym * || k(M_o s) + k(s) ||^2  [+ w_res * || r(M_o s) - M_r r(s) ||^2 ]

k(s) is the actor's MEAN on the five relationship knobs (Delta, s, o), M_o the observation mirror
(env.mirror_perm_sign: swap+negate every joint block, negate the lateral quantities, shift the
phase by pi), M_r the residual mirror (swap L/R, negate -- the verified FK sign rule). Mirror the
state and the knobs must negate: a robot drifting left trims right, its mirror image trims left,
both satisfy it exactly. Asymmetry justified by the state is free; asymmetry adopted for no
reason is not. The first (knob) term is the artifact's; the residual term is the same
equivariance applied to the per-tick channel, on by w_sym_res.

The mirror is applied in RAW observation space: the rollout buffer holds VecNormalize'd obs, so
each minibatch is un-normalised with the live running stats, mirrored, re-normalised and clipped
exactly as VecNormalize would. (If the stats drift asymmetric early, mirror the stats rather than
raise the weight -- §13.) Everything else is stable_baselines3's PPO.train() verbatim.
"""
import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


class SymPPO(PPO):
    """PPO with the observation-mirror equivariance loss. Extra kwargs persist in the checkpoint
    (they live in __dict__, which SB3 saves), so `SymPPO.load` restores them."""

    def __init__(self, *args, sym_weight=0.0, sym_res_weight=0.0, obs_perm=None, obs_sign=None,
                 act_perm=None, act_sign=None, knob_idx=(), res_idx=(), bound_weight=0.0,
                 bound_soft=1.0, **kwargs):
        self.sym_weight = float(sym_weight)
        self.sym_res_weight = float(sym_res_weight)
        # ACTION-MEAN BOUNDS LOSS (2026-09-10, v2c). The DiagGaussian is sampled unbounded and the
        # env clips to [-1, 1]; once a rail pays, every mean beyond it earns the same clipped
        # sample, so nothing stops the means from drifting out of the box. Measured on v2b at 23 M:
        # 68-79 % of the spec means and 53-60 % of the residual means were outside [-1, 1],
        # median |mu| ~ 2, p90 ~ 3.9 -- the "bang-bang spec", the clock parked on its ceiling and
        # the saturated residual were all this one artifact, and the greedy action clip(mu) no
        # longer matches the effective action E[clip(mu + eps)] the policy was trained on (the
        # determinism gap). w * mean(relu(|mu| - soft)^2), the rl_games "bounds_loss".
        self.bound_weight = float(bound_weight)
        self.bound_soft = float(bound_soft)
        self.obs_perm = None if obs_perm is None else np.asarray(obs_perm, dtype=np.int64)
        self.obs_sign = None if obs_sign is None else np.asarray(obs_sign, dtype=np.float32)
        self.act_perm = None if act_perm is None else np.asarray(act_perm, dtype=np.int64)
        self.act_sign = None if act_sign is None else np.asarray(act_sign, dtype=np.float32)
        self.knob_idx = np.asarray(list(knob_idx), dtype=np.int64)
        self.res_idx = np.asarray(list(res_idx), dtype=np.int64)
        super().__init__(*args, **kwargs)

    # ---- the mirror --------------------------------------------------------------------------
    @property
    def _sym_on(self):
        return (self.obs_perm is not None and
                ((self.sym_weight > 0.0 and self.knob_idx.size > 0)
                 or (self.sym_res_weight > 0.0 and self.res_idx.size > 0)))

    def mirror_obs(self, obs: th.Tensor) -> th.Tensor:
        perm = th.as_tensor(self.obs_perm, device=obs.device)
        sign = th.as_tensor(self.obs_sign, device=obs.device, dtype=obs.dtype)
        vn = self.get_vec_normalize_env()
        if vn is not None and getattr(vn, "norm_obs", False):
            mean = th.as_tensor(vn.obs_rms.mean, device=obs.device, dtype=obs.dtype)
            std = th.sqrt(th.as_tensor(vn.obs_rms.var, device=obs.device, dtype=obs.dtype) + vn.epsilon)
            raw = obs * std + mean
            raw_m = sign * raw[:, perm]
            return th.clamp((raw_m - mean) / std, -vn.clip_obs, vn.clip_obs)
        return sign * obs[:, perm]

    def sym_loss(self, obs: th.Tensor):
        mu = self.policy.get_distribution(obs).distribution.mean
        mu_m = self.policy.get_distribution(self.mirror_obs(obs)).distribution.mean
        loss = th.zeros((), device=obs.device)
        if self.sym_weight > 0.0 and self.knob_idx.size:
            k = th.as_tensor(self.knob_idx, device=obs.device)
            loss = loss + self.sym_weight * ((mu_m[:, k] + mu[:, k]) ** 2).sum(-1).mean()
        if self.sym_res_weight > 0.0 and self.res_idx.size and self.act_perm is not None:
            ap = th.as_tensor(self.act_perm, device=obs.device)
            asg = th.as_tensor(self.act_sign, device=obs.device, dtype=mu.dtype)
            a_mir = asg * mu[:, ap]
            r = th.as_tensor(self.res_idx, device=obs.device)
            loss = loss + self.sym_res_weight * ((mu_m[:, r] - a_mir[:, r]) ** 2).sum(-1).mean()
        return loss

    # ---- SB3 PPO.train(), verbatim + the symmetry term ---------------------------------------
    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses, sym_losses = [], [], []
        bound_losses, out_fracs = [], []
        clip_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if self._sym_on:
                    s_loss = self.sym_loss(rollout_data.observations)
                    sym_losses.append(s_loss.item())
                    loss = loss + s_loss
                if getattr(self, "bound_weight", 0.0) > 0.0:
                    mu = self.policy.get_distribution(rollout_data.observations).distribution.mean
                    excess = th.relu(mu.abs() - self.bound_soft)
                    b_loss = self.bound_weight * (excess ** 2).mean()
                    bound_losses.append(b_loss.item())
                    out_fracs.append((mu.abs() > 1.0).float().mean().item())
                    loss = loss + b_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        if sym_losses:
            self.logger.record("train/sym_loss", np.mean(sym_losses))
        if bound_losses:
            self.logger.record("train/bound_loss", np.mean(bound_losses))
            self.logger.record("train/mu_out_frac", np.mean(out_fracs))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
