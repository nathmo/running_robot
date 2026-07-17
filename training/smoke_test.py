"""Fast sanity suite for the sprint training stack — run before every long training launch:

    python training/smoke_test.py

Checks (in ~30 s, no GPU needed): the self-contained model loads; every preset builds; obs/action
dims; a random-policy rollout stays finite; the sprint line latch + task-channel flip + stop-hold
termination + finish bonus; the phase-gated stance indicator's shape (continuity at the wrap,
antiphase overlap = flight window when stance_ratio < 0.5); the residual channel moves the
targets; the coef_rate phase gate is free at the cycle boundary; curriculum setters reach the env.
Exits non-zero on the first failure (safe to gate an sbatch on it).
"""
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import numpy as np

import fourier_gait
from config import Config, PRESETS, get_config, config_from_dict, config_to_dict
from env import DashEnv

FAIL = 0


def check(name, ok, detail=""):
    global FAIL
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAIL += 1


def test_fourier_gait():
    print("fourier_gait:")
    N = 3
    check("dims", fourier_gait.action_dim(N) == 24 and fourier_gait.spec_dim(N) == 18)
    # stance indicator: bounded, ~1 mid-stance, ~0 mid-swing, continuous at the phase wrap
    sr = 0.6
    I = lambda p: fourier_gait.stance_indicator(p, sr)
    mid_st = I(np.pi * sr)                    # middle of the stance window [0, 2*pi*sr)
    mid_sw = I(np.pi * (1 + sr))              # middle of the swing window [2*pi*sr, 2*pi)
    check("mid-stance ~1", mid_st > 0.99, f"{mid_st:.3f}")
    check("mid-swing ~0", mid_sw < 0.01, f"{mid_sw:.3f}")
    for b in (0.0, 2 * np.pi * sr):           # transitions centered on both boundaries
        check(f"boundary {b:.2f} = 0.5", abs(I(b) - 0.5) < 1e-6, f"{I(b):.3f}")
    wrap_gap = abs(I(2 * np.pi - 1e-6) - I(1e-6))
    check("continuous at wrap", wrap_gap < 1e-3, f"gap={wrap_gap:.4f}")
    vals = [fourier_gait.stance_indicator(p, 0.42) for p in np.linspace(0, 2 * np.pi, 500)]
    check("bounded [0,1]", min(vals) >= 0.0 and max(vals) <= 1.0)
    # flight window: with sr < 0.5 there must be phases where BOTH feet are expected in swing
    both_swing = [(1 - fourier_gait.stance_indicator(p, 0.42))
                  * (1 - fourier_gait.stance_indicator(p + np.pi, 0.42))
                  for p in np.linspace(0, 2 * np.pi, 500)]
    check("flight window exists at sr=0.42", max(both_swing) > 0.9, f"{max(both_swing):.3f}")
    both_walk = [(1 - fourier_gait.stance_indicator(p, 0.65))
                 * (1 - fourier_gait.stance_indicator(p + np.pi, 0.65))
                 for p in np.linspace(0, 2 * np.pi, 500)]
    check("no flight demanded at sr=0.65", max(both_walk) < 0.1, f"{max(both_walk):.3f}")
    # mirror/antiphase convention: with zero nominal/reflex, the right leg's deviation at phi
    # must equal MINUS the left leg's at phi+pi — the reward's stance_indicator(phi+pi) windows
    # assume exactly this; a sign regression here would train the right leg against the schedule
    cfg = Config()
    rng = np.random.default_rng(7)
    cam_c = rng.uniform(-1, 1, fourier_gait.per_joint(N))
    thigh_c = rng.uniform(-1, 1, fourier_gait.per_joint(N))
    zero6 = np.zeros(6)
    ok_mirror = True
    for phi in np.linspace(0, 2 * np.pi, 40):
        a = fourier_gait.assemble(cam_c, thigh_c, np.zeros(3), phi, 0.0, 0.0, zero6, cfg)
        b = fourier_gait.assemble(cam_c, thigh_c, np.zeros(3), phi + np.pi, 0.0, 0.0, zero6, cfg)
        if abs(a[fourier_gait.THIGH_R] + b[fourier_gait.THIGH_L]) > 1e-9 \
                or abs(a[fourier_gait.CAM_R] + b[fourier_gait.CAM_L]) > 1e-9:
            ok_mirror = False
            break
    check("right leg = mirrored antiphase left", ok_mirror)


def test_presets():
    print("presets:")
    for name in sorted(PRESETS):
        try:
            cfg = get_config(name)
            check(f"{name} builds", isinstance(cfg, Config))
        except Exception as e:
            check(f"{name} builds", False, repr(e))
    # config JSON round-trip preserves tuples
    cfg = get_config("m1_sprint")
    rt = config_from_dict(config_to_dict(cfg))
    check("config round-trip", rt.base_lock == cfg.base_lock and rt.z_rail_range == cfg.z_rail_range)


def test_env_basic():
    print("env (m2_sprint):")
    env = DashEnv(get_config("m2_sprint"))
    obs, _ = env.reset(seed=0)
    check("action dim 24", env.action_space.shape == (24,))
    check("obs dim 275", obs.shape == (55 * 5,), str(obs.shape))
    rng = np.random.default_rng(0)
    finite, floored = True, True
    for _ in range(150):
        a = rng.uniform(-1, 1, env.action_dim).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        if not (np.all(np.isfinite(obs)) and np.isfinite(r)):
            finite = False
            break
        # suicide-proofing: pre-terminal per-step total is floored (fall subtracts 100 after)
        if not term and r < -env.cfg.step_reward_floor - 1e-6:
            floored = False
        if term or trunc:
            obs, _ = env.reset()
    check("150 random steps finite", finite)
    check("per-step reward floored (non-terminal)", floored)
    # backward motion must pay NEGATIVE income (anti-shuttle): force vx < 0 and read the term
    env.reset(seed=3)
    env.data.qvel[0] = -1.0
    _, terms_bwd = env._reward(np.zeros(6, np.float32), np.array([True, True]))
    check("backward vx pays negative fwd_speed", terms_bwd["fwd_speed"] < -1.0,
          str(terms_bwd["fwd_speed"]))
    # anti-farm invariant: stop-phase income must be below the clock cost
    check("stop phase net-negative (w_stop_vel < w_time)", env.cfg.w_stop_vel < env.cfg.w_time)
    check("reward terms present", all(k in info["reward_terms"] for k in
          ("fwd_speed", "time", "phase_contact", "residual", "energy", "coef_rate")))
    # task channel: run phase = [1, dist/100]
    frame = env._proprio()
    check("task run flag", env._task[0] == 1.0 and 0.0 < env._task[1] <= 1.0,
          str(env._task))


def test_sprint_finish():
    print("sprint line latch + stop:")
    env = DashEnv(get_config("m1_sprint"))
    env.reset(seed=1)
    env.cfg.sprint_dist_m = 1.0        # takes effect on the next reset
    env.reset(seed=1)
    # teleport the base past the line, then hold still: latch -> task flip -> hold -> bonus
    env.data.qpos[0] = env._x0 + 2.0
    env.data.qvel[:] = 0.0
    a = np.zeros(env.action_dim, np.float32)
    fin_reward, finished = 0.0, False
    for _ in range(int(env.cfg.stop_hold_s / env.control_dt) + 10):
        obs, r, term, trunc, info = env.step(a)
        if env._sprint_crossed and not finished:
            check("task flips to stop", env._task[0] == 0.0 and env._task[1] == 0.0)
            finished = True
        if term:
            fin_reward = r
            break
    check("crossed latched", env._sprint_crossed)
    check("finish terminates", term and info["sprint"]["finished"])
    check("finish bonus paid", fin_reward > env.cfg.finish_bonus * 0.5, f"r={fin_reward:.1f}")


def test_residual_and_gate():
    print("residual channel + coef_rate gate:")
    cfg = get_config("m2_sprint")
    env = DashEnv(cfg)
    env.reset(seed=2)
    spec = np.zeros(env.action_dim, np.float32)
    a_res = spec.copy()
    a_res[env.spec_dim:] = 1.0
    env.step(spec)                     # flush the 1-step delay buffer
    env.step(spec)
    cmd_plain = env._prev_motor_cmd.copy()
    env.reset(seed=2)
    env.step(a_res)
    env.step(a_res)
    cmd_res = env._prev_motor_cmd.copy()
    d = float(np.max(np.abs(cmd_res - cmd_plain)))
    expect = cfg.residual_scale / cfg.action_scale
    check("residuals move targets by residual_scale", abs(d - expect) < 1e-5,
          f"d={d:.4f} expect={expect:.4f}")
    # coef_rate phase gate: a spec change applied MID-cycle must be billed; the same change
    # applied at the cycle boundary (phase 0) must be free. The 1-step delay buffer means the
    # spec sent at step k is applied at step k+1 — drive the phase by hand to test both gates.
    big = np.zeros(env.action_dim, np.float32)
    big[:env.spec_dim] = 1.0
    env.reset(seed=3)
    env.step(big)                      # buffers the big spec (applied action this step: zeros)
    env._phase = np.pi                 # next application lands mid-cycle -> sin^2(pi/2) = 1
    _, _, _, _, info = env.step(big)   # big spec applied now, prev applied was zeros
    check("coef_rate bills mid-cycle spec change", info["reward_terms"]["coef_rate"] < -0.5,
          str(info["reward_terms"]["coef_rate"]))
    env.reset(seed=3)
    env.step(big)
    env._phase = 0.0                   # next application lands exactly on the boundary -> free
    _, _, _, _, info = env.step(big)
    check("coef_rate free at cycle boundary", info["reward_terms"]["coef_rate"] == 0.0,
          str(info["reward_terms"]["coef_rate"]))
    # residual-only change is NEVER billed by coef_rate, at any phase
    env.reset(seed=3)
    res_only = np.zeros(env.action_dim, np.float32)
    res_only[env.spec_dim:] = 1.0
    env.step(res_only)
    env._phase = np.pi
    _, _, _, _, info = env.step(res_only)
    check("residual change not billed by coef_rate", info["reward_terms"]["coef_rate"] == 0.0,
          str(info["reward_terms"]["coef_rate"]))


def test_curriculum_setters():
    print("curriculum setters:")
    env = DashEnv(get_config("m2_sprint"))
    env.reset(seed=0)
    env.set_stance_ratio(0.5)
    env.set_efficiency_scale(0.7)
    env.set_sprint_dist(42.0)
    env.reset(seed=0)
    check("stance ratio", env._stance_ratio == 0.5)
    check("eff scale", env._eff_scale == 0.7)
    check("sprint dist applies at reset", env._sprint_D == 42.0)


def test_m1_rail():
    print("m1 rail + ride-height LUT:")
    env = DashEnv(get_config("m1_sprint"))
    heights = set()
    for s in range(3):
        env.reset(seed=s)
        heights.add(round(float(env.data.qpos[2]), 4))
        check(f"reset {s} on-floor stance finite", np.all(np.isfinite(env.data.qpos)))
    check("ride height randomized", len(heights) > 1, str(heights))


if __name__ == "__main__":
    test_fourier_gait()
    test_presets()
    test_env_basic()
    test_sprint_finish()
    test_residual_and_gate()
    test_curriculum_setters()
    test_m1_rail()
    print(f"\n{'ALL OK' if FAIL == 0 else f'{FAIL} FAILURES'}")
    sys.exit(1 if FAIL else 0)
