"""Quick sanity checks for Dash01Env before training.

Run:  .venv/Scripts/python.exe -m rl.smoke_test [--preset m1]
"""
import argparse
import numpy as np
from .config import get_config
from .env import Dash01Env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="m1_stand")
    args = ap.parse_args()

    cfg = get_config(args.preset)
    env = Dash01Env(cfg)
    zero_action = np.zeros(env.action_dim, np.float32)
    print(f"preset={args.preset}  action_mode={cfg.action_mode}  base_lock={cfg.base_lock}  "
          f"speed_mode={cfg.speed_mode}  z_rail_randomize={cfg.z_rail_randomize}")
    print(f"sim_dt={env.sim_dt}  control_dt={env.control_dt:.3f}s ({1/env.control_dt:.0f} Hz)  "
          f"max_steps={env.max_steps}")
    print(f"obs space  = {env.observation_space.shape}  (frame_dim {env.frame_dim} x "
          f"history {cfg.history_len})")
    print(f"action dim = {env.action_dim}  (pd=6 motor targets; fourier=cam+thigh coeffs+freq+reflex)")

    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs)), "non-finite obs at reset"
    print(f"reset obs finite, shape {obs.shape}; obs range [{obs.min():.2f}, {obs.max():.2f}]")

    # 1) random actions never crash / NaN
    r_sum = 0.0
    for _ in range(200):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        r_sum += r
        if term or trunc:
            obs, _ = env.reset()
    print(f"200 random steps OK; finite throughout")

    # 2) base-DOF locks hold: over several episodes, locked base dofs stay pinned (Z at its ride
    # height, the rest at 0) and the ride height VARIES per episode when z_rail_randomize is on.
    locked = np.asarray(cfg.base_lock, bool)
    ride_heights = []
    for ep in range(5):
        obs, _ = env.reset(seed=100 + ep)
        h0 = float(env.data.qpos[2])
        ride_heights.append(h0)
        for _ in range(60):                          # steps (macro-steps in fourier mode)
            _, _, term, trunc, _ = env.step(zero_action)
            if term or trunc:
                break
        qb, vb = env.data.qpos[:6], env.data.qvel[:6]
        for i in range(6):
            if locked[i]:
                tgt = h0 if i == 2 else 0.0
                assert abs(qb[i] - tgt) < 5e-3, f"locked dof {i} drifted: {qb[i]:.4f} != {tgt:.4f}"
                assert abs(vb[i]) < 5e-2, f"locked dof {i} vel nonzero: {vb[i]:.4f}"
    n_free = int((~locked).sum())
    print(f"locks OK: {6-n_free} dof(s) pinned across 5 episodes; ride heights="
          f"{[round(h,3) for h in ride_heights]}")
    if cfg.z_rail_randomize:
        assert np.ptp(ride_heights) > 1e-3, "z_rail_randomize on but ride height did not vary"
        lo, hi = cfg.z_rail_range
        assert all(lo - 1e-3 <= h <= hi + 1e-3 for h in ride_heights), "ride height out of range"
        print(f"  ride height varies within z_rail_range {cfg.z_rail_range} (OK)")

    # 3) zero-action hold -> balance/survival baseline
    obs, _ = env.reset(seed=1)
    steps, rt = 0, None
    for _ in range(env.max_steps):
        obs, r, term, trunc, info = env.step(zero_action)
        rt = info["reward_terms"]
        steps += 1
        if term or trunc:
            break
    unit = "cycles" if cfg.action_mode == "fourier" else "control steps"
    print(f"zero-action hold: survived {steps} {unit} before fall/timeout")
    print(f"reward terms (last step): " + ", ".join(f"{k}={v:+.3f}" for k, v in rt.items()))
    print(f"command sampled: {info['command']}")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
