"""action_mode="fourier_step" — the per-step Fourier override.

Three guarantees:
  1. EQUIVALENCE: with a CONSTANT action (and a degenerate frequency band so K is fixed),
     stepping the per-step env K times reproduces the per-cycle macro-step trajectory
     exactly — fourier_step is a strict superset of fourier, not a different gait.
  2. PHASE GATE: the coef_rate penalty for changing the coefficients is ~0 exactly at the
     cycle boundary (sin(phase/2)^2 gate) and strictly negative mid-cycle.
  3. SHAPES: 18-dim action, +2 obs dims per frame ([sin, cos] of the gait phase).
"""
import numpy as np
import pytest

from rl.config import get_config
from rl.env import Dash01Env

FREQ = 2.5                # degenerate band -> f fixed, K = 1/(2.5*0.02) = 20 steps/cycle
K = 20


def _cfg(name):
    cfg = get_config(name)
    cfg.gait_freq_hz = (FREQ, FREQ)
    cfg.action_delay_steps = 0
    return cfg


def test_constant_action_reproduces_per_cycle_trajectory():
    envA = Dash01Env(_cfg("m2_fourier_step"))     # per-step
    envB = Dash01Env(_cfg("m2_fourier"))          # per-cycle macro-step
    envA.reset(seed=11)
    envB.reset(seed=11)
    rng = np.random.default_rng(7)
    a = rng.uniform(-1.0, 1.0, envA.action_space.shape).astype(np.float32)
    _, _, termB, _, _ = envB.step(a)              # one macro-step = K control steps
    assert not termB, "per-cycle env fell during the test cycle (pick a different action)"
    for _ in range(K):
        _, _, termA, _, _ = envA.step(a)
    assert not termA
    np.testing.assert_allclose(envA.data.qpos, envB.data.qpos, atol=1e-10)


def test_coef_change_free_at_cycle_boundary_and_billed_mid_cycle():
    rng = np.random.default_rng(3)
    A = rng.uniform(-0.4, 0.4, 18).astype(np.float32)
    B = np.clip(A + 0.3, -1.0, 1.0).astype(np.float32)
    assert not np.allclose(A, B)

    # --- boundary: step A until the phase wraps, then switch to B ---
    env = Dash01Env(_cfg("m2_fourier_step"))
    env.reset(seed=11)
    prev_phase = env._phase
    for _ in range(3 * K):
        _, _, term, _, _ = env.step(A)
        assert not term
        if env._phase < prev_phase:               # wrapped: the next phase_used is ~0
            break
        prev_phase = env._phase
    else:
        pytest.fail("phase never wrapped")
    assert env._phase < 0.05                      # within 0.05 rad of the boundary
    _, _, _, _, info = env.step(B)
    boundary_pen = info["reward_terms"]["coef_rate"]
    assert abs(boundary_pen) < 1e-6               # changing the spec AT the boundary is free

    # --- mid-cycle: fresh env, switch A -> B near phase = pi ---
    env = Dash01Env(_cfg("m2_fourier_step"))
    env.reset(seed=11)
    for _ in range(K // 2):                       # 10 steps of 2*pi/20 -> phase ~ pi
        _, _, term, _, _ = env.step(A)
        assert not term
    assert abs(env._phase - np.pi) < 0.05
    _, _, _, _, info = env.step(B)
    mid_pen = info["reward_terms"]["coef_rate"]
    assert mid_pen < 0                            # mid-cycle rewrites pay
    assert abs(mid_pen) > 10 * abs(boundary_pen)


def test_shapes_and_phase_obs_tail():
    cfg = _cfg("m2_fourier_step")
    env = Dash01Env(cfg)
    assert env.action_space.shape == (18,)
    assert env.frame_dim == 46                    # 6+6+6+3+3+18+2 (+2 sin/cos phase)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (230,)                    # 46 * history_len 5
    rng = np.random.default_rng(1)
    a = rng.uniform(-0.5, 0.5, 18).astype(np.float32)
    obs, _, _, _, _ = env.step(a)
    tail = obs[-2:]                               # last frame's [sin, cos] of the NEXT phase
    want = np.array([np.sin(env._phase), np.cos(env._phase)], np.float32)
    np.testing.assert_allclose(tail, want, atol=1e-7)
