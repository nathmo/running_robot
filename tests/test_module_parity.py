"""Module-injection guard: running the env with the stock LIBRARY modules
(experiments/_lib/rewards/gait_speed_v3.py, experiments/_lib/obs/standard.py) must be
numerically identical to the built-in code paths — that is what makes a per-experiment
./reward.py a safe escape hatch rather than a fork.
"""
import numpy as np
import pytest

from rl.config import get_config
from rl.env import Dash01Env

REWARD_LIB = "experiments/_lib/rewards/gait_speed_v3.py"
OBS_LIB = "experiments/_lib/obs/standard.py"
PRESETS = ["m2", "m2_walk", "m1_sprint_fourier", "m1_stand",
           "m2_fourier_step"]   # speed / tracking / sprint+fourier / stand / per-step fourier
N_STEPS = 80
SEED = 4321


def rollout(cfg):
    env = Dash01Env(cfg)
    obs, _ = env.reset(seed=SEED)
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(N_STEPS):
        a = rng.uniform(-1.0, 1.0, env.action_space.shape).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        out.append((obs.copy(), float(r), bool(term), bool(trunc), dict(info["reward_terms"])))
        if term or trunc:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    return out


@pytest.mark.parametrize("preset", PRESETS)
def test_library_modules_match_builtin(preset):
    builtin = rollout(get_config(preset))
    cfg = get_config(preset)
    cfg.reward_module = REWARD_LIB
    cfg.obs_module = OBS_LIB
    injected = rollout(cfg)
    for i, ((oa, ra, ta, ca, terms_a), (ob, rb, tb, cb, terms_b)) in enumerate(zip(builtin, injected)):
        assert (ta, ca) == (tb, cb), f"{preset} step {i}: termination differs"
        np.testing.assert_array_equal(oa, ob, err_msg=f"{preset} step {i}: obs differs")
        assert ra == pytest.approx(rb, rel=1e-10, abs=1e-10), \
            f"{preset} step {i}: reward {ra} != {rb}"
        assert set(terms_a) == set(terms_b), f"{preset} step {i}: term keys differ"
        for k in terms_a:
            assert terms_a[k] == pytest.approx(terms_b[k], rel=1e-10, abs=1e-10), \
                f"{preset} step {i}: term {k}: {terms_a[k]} != {terms_b[k]}"
