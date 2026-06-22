"""Quick sanity checks for SpiderBotEnv before training.

Run:  .venv/Scripts/python.exe -m rl.smoke_test
"""
import numpy as np
from .config import get_config
from .env import SpiderBotEnv, FRAME_DIM


def main():
    cfg = get_config("m1_stand")
    env = SpiderBotEnv(cfg)
    print(f"sim_dt={env.sim_dt}  control_dt={env.control_dt:.3f}s ({1/env.control_dt:.0f} Hz)  "
          f"max_steps={env.max_steps}")
    print(f"obs space  = {env.observation_space.shape}  (expect {FRAME_DIM}*{cfg.history_len}="
          f"{FRAME_DIM*cfg.history_len})")
    print(f"action     = {env.action_space.shape}  (expect {env.nu})")

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

    # 2) hold the standing pose (zero action) -> how long before it falls? (balance baseline)
    obs, _ = env.reset(seed=1)
    steps = 0
    rt = None
    for _ in range(env.max_steps):
        obs, r, term, trunc, info = env.step(np.zeros(env.nu, np.float32))
        rt = info["reward_terms"]
        steps += 1
        if term or trunc:
            break
    print(f"zero-action hold: survived {steps}/{env.max_steps} control steps "
          f"({steps*env.control_dt:.2f}s) before fall/timeout")
    print(f"reward terms (last step): " + ", ".join(f"{k}={v:+.3f}" for k, v in rt.items()))

    # 3) command is (0,0) for m1_stand
    print(f"command sampled (should be 0,0 for m1): {info['command']}")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
