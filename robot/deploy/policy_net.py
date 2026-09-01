"""The deployed actor, in numpy. No torch, no pickle, no code execution on load.

Three matrix stacks, exactly as `walk_mit/asym_policy.py` wires them:

    estimator : actor_obs (590) -> 128 -> 64 -> 3        body-frame velocity estimate
    policy_net: [actor_obs, est] (593) -> 256 -> 256     tanh on EVERY layer including the last
    action_net: 256 -> 30                                the Gaussian MEAN; the robot never samples

Two things a naive port gets wrong, both caught by verify_export.py:

  * `_mlp(..., out_act=True)` means policy_net's LAST hidden layer is tanh'd before action_net.
    Leaving it linear changes every action.
  * the estimator is fed the RAW (VecNormalize-scaled) actor observation and its output is
    concatenated UNSCALED -- it is not itself normalized, because in training the estimator's
    target was already the scaled velocity.

Cost on the robot: 590x128 + 128x64 + 64x3 + 593x256 + 256x256 + 256x30 = 305k MACs.
Measured on this Pi 3B for a comparable net: 117 us mean / 333 us max, i.e. ~2% of a 5 ms budget.
"""
import numpy as np


class PolicyNet:
    def __init__(self, b):
        a = b.a
        # float32 throughout: it is what was trained, it halves the memory traffic on the Pi's
        # tiny cache, and float64 promotion in a hot loop is a real cost at 200 Hz.
        self.est = [(np.ascontiguousarray(a["est_w0"].T, np.float32), a["est_b0"].astype(np.float32)),
                    (np.ascontiguousarray(a["est_w1"].T, np.float32), a["est_b1"].astype(np.float32)),
                    (np.ascontiguousarray(a["est_w2"].T, np.float32), a["est_b2"].astype(np.float32))]
        self.pi = [(np.ascontiguousarray(a["pi_w0"].T, np.float32), a["pi_b0"].astype(np.float32)),
                   (np.ascontiguousarray(a["pi_w1"].T, np.float32), a["pi_b1"].astype(np.float32))]
        self.act = (np.ascontiguousarray(a["act_w"].T, np.float32), a["act_b"].astype(np.float32))
        self.obs_mean = a["obs_mean"].astype(np.float32)
        self.obs_std = np.sqrt(a["obs_var"].astype(np.float32) + np.float32(b.vn_epsilon))
        self.clip_obs = np.float32(b.clip_obs)
        self.n_actor = b.n_actor
        self.action_dim = int(b.action_dim)
        self._buf = np.empty(self.n_actor + 3, np.float32)   # [normalized obs | velocity estimate]

    def normalize(self, raw_obs):
        """VecNormalize.normalize_obs for the actor slice: (x - mean)/sqrt(var + eps), clipped."""
        o = (np.asarray(raw_obs, np.float32) - self.obs_mean) / self.obs_std
        return np.clip(o, -self.clip_obs, self.clip_obs, out=o)

    def estimate_velocity(self, norm_obs):
        h = norm_obs
        for w, bb in self.est[:-1]:
            h = np.tanh(h @ w + bb)
        w, bb = self.est[-1]
        return h @ w + bb                       # linear head, no output activation

    def act_from_normalized(self, norm_obs):
        """norm_obs -> (action, velocity_estimate). Deterministic: the distribution MEAN."""
        buf = self._buf
        buf[:self.n_actor] = norm_obs
        est = self.estimate_velocity(norm_obs)
        buf[self.n_actor:] = est
        h = buf
        for w, bb in self.pi:
            h = np.tanh(h @ w + bb)
        w, bb = self.act
        a = h @ w + bb
        return a, est

    def __call__(self, raw_obs):
        """raw (unnormalized) actor observation -> (action clipped to [-1,1], velocity estimate).

        The clip is the env's own (`DashEnv.step` clips before doing anything with the action), so
        applying it here makes the deployed action space identical to the trained one rather than
        relying on a downstream clamp to be equivalent."""
        a, est = self.act_from_normalized(self.normalize(raw_obs))
        return np.clip(a, -1.0, 1.0), est
