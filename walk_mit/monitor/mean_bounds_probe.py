"""Is the determinism gap the clipped-Gaussian artifact? Measures the policy's PRE-CLIP action means
on its own rollout observations: with a DiagGaussian sampled unbounded and clipped to [-1, 1] by
the env, means drift OUTSIDE the box whenever the rail is rewarded (the clipped sample keeps the
same expected reward for any mean beyond it); the greedy action clip(mu) = the rail then differs
from the effective action E[clip(mu + eps)] the policy was trained on, so the stochastic policy
runs and the greedy one dies.

    python walk_mit/monitor/mean_bounds_probe.py RUN CKPT_STEM [--episodes 2] [--dr 0] [--assist 0.85]

Prints, per action group, the fraction of |mu| > 1, the median |mu|, and the greedy-vs-effective
gap |clip(mu) - E[clip(mu + eps)]| under the checkpoint's own std.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from evaluate import build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("ckpt")
ap.add_argument("--episodes", type=int, default=2)
ap.add_argument("--dr", type=float, default=None)
ap.add_argument("--assist", type=float, default=None)
ap.add_argument("--seed0", type=int, default=1000)
args = ap.parse_args()
run = Path(args.run)
model, venv, raw = build(run, None, str(run / args.ckpt))
if args.dr is not None:
    raw.set_dr_scale(args.dr)
if args.assist is not None:
    raw.set_pitch_assist(args.assist)
std = float(torch.exp(model.policy.log_std).mean())
mus, lens = [], []
for e in range(args.episodes):
    venv.seed(args.seed0 + e)
    obs = venv.reset()
    n = 0
    while True:
        with torch.no_grad():
            dist = model.policy.get_distribution(torch.as_tensor(obs, dtype=torch.float32, device=model.device))
            mu = dist.distribution.mean.cpu().numpy()[0]
        mus.append(mu)
        a, _ = model.predict(obs, deterministic=False)
        obs, r, d, info = venv.step(a)
        n += 1
        if d[0]:
            break
    lens.append(n)
M = np.asarray(mus)
groups = dict(fourier=slice(0, 21), kp_kd=slice(21, 35), clock=slice(35, 36), knobs=slice(36, 44), residual=slice(44, 50))
sig = std
eps = np.random.default_rng(0).normal(size=(4000, 1))
print(f"[{run.name} {args.ckpt}] stochastic episodes {lens} ticks, policy std {std:.3f}, {len(M)} rows")
print(f"   {'group':10s} {'|mu|>1':>7s} {'|mu|>2':>7s} {'med|mu|':>8s} {'p90|mu|':>8s} {'greedy-vs-effective':>20s}")
for g, sl in groups.items():
    x = M[:, sl].ravel()
    eff = np.clip(x[None, :] + sig * eps, -1, 1).mean(axis=0)
    gap = np.abs(np.clip(x, -1, 1) - eff)
    print(f"   {g:10s} {np.mean(np.abs(x) > 1):7.2f} {np.mean(np.abs(x) > 2):7.2f} {np.median(np.abs(x)):8.2f} "
          f"{np.percentile(np.abs(x), 90):8.2f} {gap.mean():20.3f}")
