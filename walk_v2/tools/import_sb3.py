"""Cross-load a walk_mit (SB3 torch, AsymmetricACPolicy) checkpoint into a walk_v2 bundle.

The contract's decisive parity test: a policy trained on one arm must run on the other. The two
policies are the same graph (estimator 377->128->64->3, actor [377+3]->256->256 tanh -> 50, critic
402->256->256 -> 1, state-independent log_std; VecNormalize clip 10 / eps 1e-8 = ObsStats), so the
import is a weight transpose plus the normalizer statistics. One layout difference is folded into the
first layers: walk_v2 frames carry (cos phi, sin phi) at columns 25:27 of each 33-dim frame where
walk_mit carries (sin phi, cos phi) -- the two input columns are swapped for every history frame.

    python walk_v2/tools/import_sb3.py --zip walk_mit/runs/v2c_s1_s0/ppo_42000000_steps.zip \
        --vecnorm walk_mit/runs/v2c_s1_s0/ppo_vecnormalize_42000000_steps.pkl \
        --preset v2c_s1_planar --out walk_v2/runs/v2c_cpu_s0_42M
    python walk_v2/evaluate.py --run walk_v2/runs/v2c_cpu_s0_42M --episodes 4 --video ...
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

FRAME_DIM, HIST_FRAMES = 33, 10
PHASE_COLS = (25, 26)          # (cos, sin) here; (sin, cos) on the CPU arm


def torch_state_dict(path):
    """The SB3 zip (needs torch) or an .npz dumped by `--dump` from a venv that has torch."""
    if str(path).endswith(".npz"):
        z = np.load(path)
        return {k: z[k].astype(np.float32) for k in z.files if not k.startswith("vecnorm_")}
    import torch
    z = zipfile.ZipFile(path)
    sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
    return {k: np.asarray(v.detach().cpu().numpy(), dtype=np.float32) for k, v in sd.items()}


def dump(zip_path, vecnorm_path, npz_path):
    """torch venv step: policy weights + VecNormalize stats -> one npz for the JAX venv."""
    sd = torch_state_dict(zip_path)
    vn = pickle.load(open(vecnorm_path, "rb"))
    np.savez(npz_path, **sd, vecnorm_mean=np.asarray(vn.obs_rms.mean, np.float32),
             vecnorm_var=np.asarray(vn.obs_rms.var, np.float32), vecnorm_count=np.asarray(float(vn.obs_rms.count)))
    print(f"[import] dumped {len(sd)} tensors + vecnorm -> {npz_path}")


def swap_phase_inputs(w_in: np.ndarray, actor_width: int):
    """w_in: (in, out) kernel whose first actor_width rows are the actor obs. Swap the cos/sin rows
    of every history frame (the CPU arm's (sin, cos) -> this side's (cos, sin))."""
    w = w_in.copy()
    for f in range(HIST_FRAMES):
        i, j = f * FRAME_DIM + PHASE_COLS[0], f * FRAME_DIM + PHASE_COLS[1]
        assert j < actor_width
        w[[i, j]] = w[[j, i]]
    return w


def swap_phase_vector(v: np.ndarray):
    v = v.copy()
    for f in range(HIST_FRAMES):
        i, j = f * FRAME_DIM + PHASE_COLS[0], f * FRAME_DIM + PHASE_COLS[1]
        v[[i, j]] = v[[j, i]]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--preset", default="v2c_s1_planar")
    ap.add_argument("--out", default=None, help="walk_v2 run dir to write (import step)")
    ap.add_argument("--no-phase-swap", action="store_true", help="assume identical frame layouts")
    ap.add_argument("--model-path", default=None, help="override the preset's plant XML (e.g. the stiff leg)")
    ap.add_argument("--dump", default=None, help="torch-venv step: write this .npz and exit")
    args = ap.parse_args()
    if args.dump:
        dump(args.zip, args.vecnorm, args.dump)
        return
    import jax.numpy as jnp                       # the JAX venv step (the --dump step has torch, no jax)
    from config import config_to_dict, get_config
    from env import DashEnvV2
    from ppo import PPO, ObsStats
    if not args.out:
        ap.error("--out is required for the import step")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = get_config(args.preset)
    if args.model_path:
        import dataclasses
        cfg = dataclasses.replace(cfg, model_path=args.model_path)
    env = DashEnvV2(cfg, n_envs=1)
    agent = PPO(cfg, env, out, cfg.total_steps, seed=0, eval_env=None)
    sd = torch_state_dict(args.zip)
    p = agent.params["params"]
    n_actor = env.actor_dim
    swap = (lambda w: w) if args.no_phase_swap else (lambda w: swap_phase_inputs(w, n_actor))

    def lin(name):
        return sd[name + ".weight"].T, sd[name + ".bias"]      # torch (out,in) -> flax (in,out)

    def put(mod, layer, w, b, first=False):
        cur = p[mod][layer]
        assert cur["kernel"].shape == w.shape and cur["bias"].shape == b.shape, (mod, layer, cur["kernel"].shape, w.shape)
        p[mod][layer] = {"kernel": jnp.asarray(swap(w) if first else w), "bias": jnp.asarray(b)}

    put("estimator", "Dense_0", *lin("mlp_extractor.estimator.0"), first=True)
    put("estimator", "Dense_1", *lin("mlp_extractor.estimator.2"))
    put("estimator", "Dense_2", *lin("mlp_extractor.estimator.4"))
    put("policy_net", "Dense_0", *lin("mlp_extractor.policy_net.0"), first=True)
    put("policy_net", "Dense_1", *lin("mlp_extractor.policy_net.2"))
    w, b = lin("action_net")
    assert p["action_net"]["kernel"].shape == w.shape
    p["action_net"] = {"kernel": jnp.asarray(w), "bias": jnp.asarray(b)}
    put("value_net", "Dense_0", *lin("mlp_extractor.value_net.0"), first=True)
    put("value_net", "Dense_1", *lin("mlp_extractor.value_net.2"))
    w, b = lin("value_net")
    assert p["value_head"]["kernel"].shape == w.shape
    p["value_head"] = {"kernel": jnp.asarray(w), "bias": jnp.asarray(b)}
    assert p["log_std"].shape == sd["log_std"].shape
    p["log_std"] = jnp.asarray(sd["log_std"])
    agent.params = {**agent.params, "params": p}
    if str(args.zip).endswith(".npz"):
        z = np.load(args.zip)
        mean, var, count = z["vecnorm_mean"].astype(np.float32), z["vecnorm_var"].astype(np.float32), float(z["vecnorm_count"])
    else:
        vn = pickle.load(open(args.vecnorm, "rb"))
        mean, var, count = np.asarray(vn.obs_rms.mean, np.float32), np.asarray(vn.obs_rms.var, np.float32), float(vn.obs_rms.count)
    if not args.no_phase_swap:
        mean, var = swap_phase_vector(mean), swap_phase_vector(var)
    agent.stats = ObsStats(mean=jnp.asarray(mean), var=jnp.asarray(var), count=jnp.asarray(count))
    stem = Path(args.zip).stem
    steps = int(stem.split("_")[1]) if stem.startswith("ppo_") and stem.split("_")[1].isdigit() else 0
    agent.step = steps
    (out / "resolved_config.json").write_text(json.dumps({"config": config_to_dict(cfg), "n_envs": 1,
                                                          "total_steps": cfg.total_steps, "preset": args.preset,
                                                          "imported_from": str(args.zip)}, indent=1))
    agent.save(out / "final.msgpack")
    print(f"[import] {Path(args.zip).name} -> {out / 'final.msgpack'} ({steps:,} steps, phase swap "
          f"{'off' if args.no_phase_swap else 'on'}, std mean {float(np.exp(sd['log_std']).mean()):.3f})")


if __name__ == "__main__":
    main()
