"""Offline tests for the v3 (BalanceRL) deployment path. No hardware, no jax, no mujoco.

    python -m pytest controller/deploy/tests/test_balance_deploy.py -q

The bit-exact check against the trained policy is BalanceRL/verify_export.py (it needs the training
stack); these pin the runtime's contract on a synthetic bundle: the bundle validates, a = 0 IS the
standing command, the gain map's endpoints are the wire limits, the observation layout and its
ordering, the slew cap, and that a balance bundle has no command an operator could press.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

DEPLOY = Path(__file__).resolve().parents[1]
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from bundle import Bundle                                            # noqa: E402
from controller_balance import PolicyControllerBalance, gain_map      # noqa: E402

NU, HL, STRIDE = 6, 10, 2
AD = 3 * NU
FD = 3 * NU + 6 + AD     # 42
OD = 18                  # the slow block
AC = FD * HL + OD        # 436
NOMINAL = np.array([-0.0007, 0.2298, -0.1161, 0.0016, -0.2181, 0.1238])
STANCE = np.array([0.0, 0.1265, -0.2025, 0.0, -0.1265, 0.2025])
KP0 = np.array([120.0, 200.0, 200.0, 120.0, 200.0, 200.0])
KD0 = np.array([4.0, 5.0, 5.0, 4.0, 5.0, 5.0])


def arrays_and_meta(zero_net=True, seed=0):
    rng = np.random.default_rng(seed)
    w = (lambda *s: np.zeros(s, np.float32)) if zero_net else (lambda *s: rng.normal(0, 0.05, s).astype(np.float32))
    raw = HL * STRIDE - STRIDE + 1
    arrays = {
        "est_w0": w(128, AC), "est_b0": w(128), "est_w1": w(64, 128), "est_b1": w(64),
        "est_w2": w(3, 64), "est_b2": w(3),
        "pi_w0": w(256, AC + 3), "pi_b0": w(256), "pi_w1": w(256, 256), "pi_b1": w(256),
        "act_w": w(AD, 256), "act_b": w(AD),
        "obs_mean": np.zeros(AC), "obs_var": np.ones(AC),
        "nominal_ctrl": NOMINAL, "default_motor_pos": STANCE,
        "q_lo": np.array([-0.785, -1.5, -1.047] * 2), "q_hi": np.array([0.785, 1.5, 1.047] * 2),
        "q_scale": np.array([0.25, 0.5, 0.5] * 2),
        "motor_vel_limit": np.array([10.3, 22.01, 22.01] * 2),
        "forcerange": np.array([61.2, 144.5, 144.5] * 2),
        "drive_kp": KP0, "drive_kd": KD0,
        "hist_idx": (raw - 1 - (np.arange(HL) * STRIDE)[::-1]).astype(np.int32),
    }
    meta = {
        "bundle_version": 3, "kind": "balance", "run": "fixture", "checkpoint": "none", "step": 0,
        "nu": NU, "control_dt": 0.01, "frame_dim": FD, "history_len": HL, "history_stride": STRIDE,
        "actor_dim": AC, "action_dim": AD, "once_dim": OD,
        "slow": {"slow_s": 2.0, "mid_s": 0.5, "fast_s": 0.3}, "action_filter_tau_s": 0.0,
        "obs_scales": dict(motor_pos=1.0, motor_vel=0.1, motor_torque=0.01, gravity=1.0, ang_vel=0.25),
        "clip_obs": 10.0, "obs_eps": 1e-8,
        "gains": {"kp_lo": 20.0, "kp_hi": 500.0, "kd_lo": 0.2, "kd_hi": 5.0},
        "motor_accel_limit": 0.0, "term_gravity_z": -0.5, "est_hidden": [128, 64],
        "policy_hidden": [256, 256], "command": {"kind": "none", "v_max": 0.0},
        "base_lock": [0] * 6, "objective": "balance",
        "trained": {"push_level_mps": 1.0, "plant_scale": 1.0, "com_shift_m": [0.03, 0.03, 0.03]},
        "reflex": {"enable": True, "kp_pitch": 1.0, "kd_pitch": 0.05, "kp_roll": 0.0,
                   "kd_roll": 0.0, "clip_rad": 0.25},
    }
    return arrays, meta


def bundle(**kw):
    return Bundle(*arrays_and_meta(**kw))


UP = np.array([0.0, 0.0, -1.0])


class TestBundle:
    def test_loads_as_v3(self):
        b = bundle()
        assert b.version == 3 and b.n_actor == AC and b.command_kind == "none"
        # the shared vocabulary the governor reads
        np.testing.assert_array_equal(b["ctrl_lo"], b["q_lo"])
        np.testing.assert_array_equal(b["imp_kp_base"], KP0)

    def test_roundtrip_on_disk(self, tmp_path):
        a, m = arrays_and_meta()
        p = tmp_path / "b.npz"
        np.savez(p, meta=np.array(json.dumps(m)), **a)
        assert Bundle.load(p).version == 3

    def test_missing_array_refused(self):
        a, m = arrays_and_meta()
        del a["q_scale"]
        with pytest.raises(ValueError, match="q_scale"):
            Bundle(a, m)

    def test_wrong_kind_refused(self):
        a, m = arrays_and_meta()
        m["kind"] = "walker"
        with pytest.raises(ValueError, match="kind"):
            Bundle(a, m)

    def test_shape_mismatch_refused(self):
        a, m = arrays_and_meta()
        a["act_w"] = np.zeros((AD + 1, 256), np.float32)
        with pytest.raises(ValueError):
            Bundle(a, m)


class TestControlLaw:
    def test_zero_action_is_the_stance_command(self):
        c = PolicyControllerBalance(bundle(zero_net=True))
        c.start(STANCE, np.zeros(6), np.zeros(6), UP, np.zeros(3))
        for _ in range(5):
            cmd = c.step(STANCE, np.zeros(6), np.zeros(6), UP, np.zeros(3))
            np.testing.assert_allclose(cmd.action, 0.0)
            np.testing.assert_allclose(cmd.target, NOMINAL, atol=1e-12)
            np.testing.assert_allclose(cmd.kp, KP0)
            np.testing.assert_allclose(cmd.kd, KD0)

    def test_the_reflex_prior_is_applied_from_the_newest_frame(self):
        c = PolicyControllerBalance(bundle())
        # upright and still: the prior contributes nothing, so a = 0 is still the stance command
        c.start(STANCE, np.zeros(6), np.zeros(6), UP, np.zeros(3))
        cmd = c.step(STANCE, np.zeros(6), np.zeros(6), UP, np.zeros(3))
        np.testing.assert_allclose(cmd.reflex, 0.0)
        np.testing.assert_allclose(cmd.target, NOMINAL, atol=1e-12)
        # leaning forward (measured gravity x = 0.1) and pitching at 0.4 rad/s:
        # thigh_L -= 1.0*0.1 + 0.05*0.4 = 0.12 rad, thigh_R += the same
        g = np.array([0.1, 0.0, -1.0])
        g = g / np.linalg.norm(g)
        w = np.array([0.0, 0.4, 0.0])
        c.step(STANCE, np.zeros(6), np.zeros(6), g, w)          # this frame carries the attitude
        cmd = c.step(STANCE, np.zeros(6), np.zeros(6), g, w)
        want = -(1.0 * g[0] + 0.05 * w[1])
        np.testing.assert_allclose(cmd.reflex, [0.0, 0.0, want, 0.0, 0.0, -want], atol=1e-9)
        np.testing.assert_allclose(cmd.target_prefilter, NOMINAL + [0, 0, want, 0, 0, -want], atol=1e-9)

    def test_the_reflex_is_clipped(self):
        a, m = arrays_and_meta()
        m["reflex"] = dict(m["reflex"], kp_pitch=50.0)
        c = PolicyControllerBalance(Bundle(a, m))
        g = np.array([0.2, 0.0, -1.0]); g = g / np.linalg.norm(g)
        c.start(STANCE, np.zeros(6), np.zeros(6), g, np.zeros(3))
        cmd = c.step(STANCE, np.zeros(6), np.zeros(6), g, np.zeros(3))
        assert abs(cmd.reflex[2]) == pytest.approx(0.25)

    def test_a_bundle_without_the_reflex_block_is_refused(self):
        a, m = arrays_and_meta()
        del m["reflex"]
        with pytest.raises(ValueError, match="reflex"):
            Bundle(a, m)

    def test_gain_map_endpoints(self):
        np.testing.assert_allclose(gain_map(np.ones(6), KP0, 20.0, 500.0), 500.0)
        np.testing.assert_allclose(gain_map(-np.ones(6), KP0, 20.0, 500.0), 20.0)
        np.testing.assert_allclose(gain_map(np.zeros(6), KP0, 20.0, 500.0), KP0)
        # monotone through 0
        a = np.linspace(-1, 1, 41)
        g = gain_map(a, 200.0, 20.0, 500.0)
        assert np.all(np.diff(g) > 0)

    def test_override_action_maps_and_slews(self):
        c = PolicyControllerBalance(bundle())
        c.start(STANCE, np.zeros(6), np.zeros(6), UP, np.zeros(3))
        a = np.array([1.0] * 6 + [1.0] * 6 + [-1.0] * 6)
        cmd = c.step(STANCE, np.zeros(6), np.zeros(6), UP, np.zeros(3), override_action=a)
        want = np.minimum(NOMINAL + np.array([0.25, 0.5, 0.5] * 2), np.array([0.785, 1.5, 1.047] * 2))
        np.testing.assert_allclose(cmd.target_prefilter, want)
        # one tick of the no-load slew cap from the stance command
        step = np.array([10.3, 22.01, 22.01] * 2) * 0.01
        np.testing.assert_allclose(cmd.target, NOMINAL + np.minimum(want - NOMINAL, step))
        np.testing.assert_allclose(cmd.kp, 500.0)
        np.testing.assert_allclose(cmd.kd, 0.2)

    def test_observation_layout_and_order(self):
        c = PolicyControllerBalance(bundle())
        pos0 = STANCE + 0.01
        c.start(pos0, np.ones(6), np.ones(6), UP * 2.0, np.array([0.1, 0.2, 0.3]))
        o = c.obs()[:HL * FD].reshape(HL, FD)
        # start: every row is the same frame, zero velocity / torque / previous action, unit gravity
        assert np.allclose(o, o[0])
        np.testing.assert_allclose(o[0, 0:6], 0.01, rtol=1e-5)
        np.testing.assert_allclose(o[0, 6:18], 0.0)
        np.testing.assert_allclose(o[0, 18:21], UP)
        np.testing.assert_allclose(o[0, 21:24], np.array([0.1, 0.2, 0.3]) * 0.25, rtol=1e-6)
        np.testing.assert_allclose(o[0, 24:42], 0.0)
        # first step does NOT push a frame (the sim's first action sees the reset history)
        a1 = np.full(18, 0.3)
        c.step(pos0, np.zeros(6), np.zeros(6), UP, np.zeros(3), override_action=a1)
        assert np.allclose(c.obs()[:HL * FD].reshape(HL, FD), o)
        # the second measurement lands in the newest raw row, carrying the previous action
        c.step(pos0, np.full(6, 2.0), np.full(6, 30.0), UP, np.zeros(3), override_action=a1)
        newest = c._history[-1]
        np.testing.assert_allclose(newest[6:12], 0.2, rtol=1e-6)
        np.testing.assert_allclose(newest[12:18], 0.3, rtol=1e-6)
        np.testing.assert_allclose(newest[24:42], 0.3, rtol=1e-6)
        # the stacked obs only shows it once the stride reaches it: hist_idx ends at the newest row
        assert np.allclose(c.obs()[:HL * FD].reshape(HL, FD)[-1], newest)

    def test_the_slow_block_is_seeded_and_integrates(self):
        a, m = arrays_and_meta()
        c = PolicyControllerBalance(Bundle(a, m))
        g = np.array([0.05, -0.02, -1.0]); g = g / np.linalg.norm(g)
        c.start(STANCE + 0.01, np.zeros(6), np.zeros(6), g, np.zeros(3))
        once = c.obs()[HL * FD:]
        assert once.size == OD
        np.testing.assert_allclose(once[0:2], g[:2], rtol=1e-5)      # seeded AT the measurement
        np.testing.assert_allclose(once[2:4], g[:2], rtol=1e-5)
        np.testing.assert_allclose(once[4:6], 0.0)
        np.testing.assert_allclose(once[6:12], 0.01, rtol=1e-4)
        np.testing.assert_allclose(once[12:18], 0.0)
        # a step in the measured gravity moves the slow channel by (1 - exp(-dt/tau)) of the gap
        al = np.exp(-0.01 / 2.0)
        g2 = np.array([0.2, 0.0, -1.0]); g2 = g2 / np.linalg.norm(g2)
        c.step(STANCE + 0.01, np.zeros(6), np.zeros(6), g, np.zeros(3))     # primed: no push
        c.step(STANCE + 0.01, np.zeros(6), np.zeros(6), g2, np.zeros(3))
        want = al * g[0] + (1 - al) * g2[0]
        np.testing.assert_allclose(c.obs()[HL * FD], want, rtol=1e-5)

    def test_the_action_filter_slows_the_target(self):
        a, m = arrays_and_meta()
        m["action_filter_tau_s"] = 0.08
        c = PolicyControllerBalance(Bundle(a, m))
        c.start(STANCE, np.zeros(6), np.zeros(6), UP, np.zeros(3))
        act = np.zeros(18); act[2] = 1.0                       # full thigh step
        cmd = c.step(STANCE, np.zeros(6), np.zeros(6), UP, np.zeros(3), override_action=act)
        al = np.exp(-0.01 / 0.08)
        want_raw = NOMINAL[2] + 0.5
        np.testing.assert_allclose(cmd.target[2], al * NOMINAL[2] + (1 - al) * want_raw, rtol=1e-6)

    def test_no_command_channel(self):
        c = PolicyControllerBalance(bundle())
        assert c.command_kind == "none"
        assert c.set_run(True) is False
        assert c.set_speed(1.0) == (False, 0.0)
        ok, why = c.start_brake()
        assert not ok and why

    def test_refuses_other_generations(self):
        from v2_fixture import v2_bundle
        with pytest.raises(ValueError):
            PolicyControllerBalance(v2_bundle())

    def test_real_net_is_finite_and_clipped(self):
        c = PolicyControllerBalance(bundle(zero_net=False))
        c.start(STANCE, np.zeros(6), np.zeros(6), UP, np.zeros(3))
        rng = np.random.default_rng(1)
        for _ in range(50):
            cmd = c.step(STANCE + rng.normal(0, 0.02, 6), rng.normal(0, 0.5, 6), rng.normal(0, 5, 6),
                         UP + rng.normal(0, 0.02, 3), rng.normal(0, 0.1, 3))
            assert np.all(np.isfinite(cmd.target)) and np.all(np.abs(cmd.action) <= 1.0)
            assert np.all((cmd.kp >= 20.0) & (cmd.kp <= 500.0)) and np.all((cmd.kd >= 0.2) & (cmd.kd <= 5.0))
