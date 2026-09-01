"""Offline tests for the deployment stack. No hardware, no CAN, no torch, no mujoco.

    python -m pytest robot/deploy/tests -q

What is deliberately NOT here: any assertion that the exported policy matches the trained one.
That needs torch and MuJoCo and lives in verify_export.py, which is a desktop tool rather than a
unit test because it takes minutes and needs the training package on the path. Run it before
every deployment; these tests are the things that must hold on every commit.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

DEPLOY = Path(__file__).resolve().parents[1]
REPO = DEPLOY.parents[1]
for p in (str(DEPLOY), str(REPO / "robot" / "fixed_gait" / "webui")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mit                                                            # noqa: E402
import thermal as TH                                                  # noqa: E402
import jointmap as JM                                                 # noqa: E402
import thermal_excite as TE                                            # noqa: E402
from safety import Limits, SafetyGovernor, STOP_NONE, STOP_SOFT, STOP_HARD   # noqa: E402

BUNDLE = DEPLOY / "bundles" / "imp_m2_long_204M.npz"


# ===================================================================== wire format
class TestMitFrames:
    def test_arbitration_id_is_extended_command_8(self):
        # node 106 -> 0x86A, the id the 2026-08-26 characterisation proved force control answers on
        assert mit.arbitration_id(106) == 0x86A
        assert mit.arbitration_id(104) == 0x868

    def test_payload_is_always_eight_bytes(self):
        # a DLC-3 frame put 62.5 A into a stalled rotor; there must be no code path to one
        for kp in (0.0, 250.0, 500.0):
            buf, _ = mit.pack(0.0, kp, 1.0)
            assert len(buf) == 8

    def test_roundtrip_within_quantisation(self):
        for p, kp, kd in ((0.0, 0.0, 0.0), (1.234, 200.0, 5.0), (-0.75, 40.0, 1.0)):
            buf, clamped = mit.pack(p, kp, kd)
            assert not clamped
            got = mit.unpack(buf)
            assert abs(got["p_des"] - p) < 1e-3
            assert abs(got["kp"] - kp) < 0.2
            assert abs(got["kd"] - kd) < 0.005

    def test_zero_velocity_and_torque_are_range_independent(self):
        """The whole reason v_des and tau_ff are pinned to zero: their spans are unidentified for
        these motors, and zero is the one value that decodes correctly under ANY span."""
        buf, _ = mit.pack(0.5, 100.0, 2.0)
        for span in ((-20.0, 20.0), (-65.0, 65.0), (-1.0, 1.0), (-500.0, 500.0)):
            got = mit.unpack(buf, v_range=span, t_range=span)
            assert got["v_des"] == pytest.approx(0.0, abs=1e-12)
            assert got["tau_ff"] == pytest.approx(0.0, abs=1e-12)

    def test_nonzero_velocity_or_torque_requires_a_measured_range(self):
        with pytest.raises(mit.RangeNotIdentified):
            mit.pack(0.0, 0.0, 0.0, v_des=1.0)
        with pytest.raises(mit.RangeNotIdentified):
            mit.pack(0.0, 0.0, 0.0, tau_ff=0.3)
        mit.pack(0.0, 0.0, 0.0, v_des=1.0, v_range=(-20.0, 20.0))     # explicit range is fine

    def test_out_of_range_is_clamped_and_reported(self):
        buf, clamped = mit.pack(99.0, 9999.0, 99.0)
        assert set(clamped) == {"position", "kp", "kd"}
        got = mit.unpack(buf)
        assert got["kp"] <= mit.KP_MAX and got["kd"] <= mit.KD_MAX
        assert got["p_des"] <= mit.P_MAX

    def test_limp_payload_commands_nothing(self):
        got = mit.unpack(mit.limp_payload())
        assert got["kp"] == 0.0 and got["kd"] == 0.0
        assert got["p_des"] == pytest.approx(0.0, abs=1e-3)


# ===================================================================== thermal
class TestThermal:
    def make(self, **kw):
        p = TH.ThermalParams("T", k_cu=0.18, c_w=95.0, c_c=1050.0, r_wc=0.72, r_ca=1.45,
                             p_idle=0.5, **kw)
        return p

    def test_steady_state_matches_the_analytic_solution(self):
        p = self.make()
        t = np.arange(0.0, 20000.0, 2.0)
        i = np.full(t.shape, 5.0)
        tw, tc = TH.simulate(p, t, i, t_amb=25.0, t0=25.0)
        # with the copper tempco the steady state is a fixed point, so solve it the same way
        # i_continuous does -- inverting it must land back on the same current
        assert p.i_continuous(25.0, t_limit=tw[-1]) == pytest.approx(5.0, rel=0.02)
        assert tc[-1] < tw[-1]                                  # the case is always cooler
        assert (tw[-1] - tc[-1]) == pytest.approx(
            p.k_cu * 25.0 * (1 + TH.ALPHA_CU * (tw[-1] - TH.T_REF)) * p.r_wc + p.p_idle * p.r_wc,
            rel=0.02)

    def test_discretisation_is_step_size_independent(self):
        """The exact (matrix-exponential) discretisation is what lets the 200 Hz observer, the
        0.5 s fitter and a multi-second projection all be the same model."""
        p = self.make()
        ends = []
        for dt in (0.005, 0.05, 0.5, 2.0):
            t = np.arange(0.0, 600.0 + dt, dt)
            tw, _ = TH.simulate(p, t, np.full(t.shape, 8.0), t_amb=25.0, t0=25.0)
            ends.append(tw[-1])
        assert max(ends) - min(ends) < 0.05          # 50 mK across a 400x change in step

    def test_winding_leads_the_case_on_a_burst(self):
        """The entire point of two nodes: a short burst is invisible in the case reading."""
        p = self.make()
        dt = 0.005
        t = np.arange(0.0, 30.0, dt)
        tw, tc = TH.simulate(p, t, np.full(t.shape, 30.0), t_amb=25.0, t0=25.0)
        assert (tw[-1] - 25.0) > 6 * (tc[-1] - 25.0)

    def test_observer_refuses_uncalibrated_parameters(self):
        p = self.make()
        assert not p.calibrated
        with pytest.raises(ValueError, match="UNCALIBRATED"):
            TH.MotorThermalModel([p] * 6, dt=0.005)
        TH.MotorThermalModel([p] * 6, dt=0.005, allow_uncalibrated=True)     # explicit opt-in

    def test_headroom_and_budget_walk_from_peak_to_continuous(self):
        p = self.make(t_derate=90.0, t_warn=100.0, t_trip=120.0)
        m = TH.MotorThermalModel([p], dt=0.005, allow_uncalibrated=True)
        peak, cont = np.array([144.5]), np.array([50.0])
        m.reset(25.0)
        assert m.torque_budget(peak, cont)[0] == pytest.approx(144.5)
        m.x[0, 0] = 105.0 - m.t_amb                      # halfway through the derate band
        assert m.torque_budget(peak, cont)[0] == pytest.approx(0.5 * (144.5 + 50.0), rel=1e-6)
        m.x[0, 0] = 130.0 - m.t_amb                      # past the trip
        assert m.torque_budget(peak, cont)[0] == pytest.approx(50.0)
        assert m.headroom()[0] == 0.0

    def test_reported_temperature_can_only_push_the_estimate_up_when_implausible(self):
        """A garbled int8 (0, -1, 200) must never drag the winding estimate DOWN."""
        p = self.make()
        m = TH.MotorThermalModel([p], dt=0.005, t_amb=25.0, allow_uncalibrated=True)
        m.reset(80.0)
        before = float(m.t_winding[0])
        for _ in range(400):
            m.step(np.zeros(1), t_reported=np.array([0.0]))       # implausible: rejected
        assert m.t_winding[0] > before - 1.0
        m.reset(80.0)
        for _ in range(400):
            m.step(np.zeros(1), t_reported=np.array([np.nan]))    # missing: rejected
        assert m.t_winding[0] > before - 1.0

    def test_seconds_to_trip_is_finite_only_above_the_continuous_rating(self):
        p = self.make()
        m = TH.MotorThermalModel([p], dt=0.005, t_amb=25.0, allow_uncalibrated=True)
        m.reset(25.0)
        i_c = p.i_continuous(25.0)
        assert np.isinf(m.seconds_to_trip(np.array([0.7 * i_c]))[0])
        assert np.isfinite(m.seconds_to_trip(np.array([3.0 * i_c]), horizon_s=3000.0)[0])


# ===================================================================== joint map
class TestJointMap:
    def test_conversions_are_exact_inverses(self):
        jm = JM.JointMap({n: {"sign": s, "offset_deg": o} for n, s, o in zip(
            JM.MODEL_TO_MOTOR, (1, -1, 1, -1, 1, -1), (0.0, 135.0, -3.0, 0.0, -135.0, 3.0))})
        model = np.array([0.1, -0.4, 0.25, -0.1, 0.4, -0.25])
        assert np.allclose(jm.to_model_rad(jm.to_norm_deg(model)), model, atol=1e-12)

    def test_refuses_to_run_until_every_joint_is_verified(self):
        jm = JM.JointMap()
        ok, why = jm.check_ready()
        assert not ok and "not verified" in why
        for n in JM.MODEL_TO_MOTOR:
            jm.mark_verified(n)
        assert jm.check_ready()[0]

    def test_a_rezero_invalidates_the_map(self):
        """The drives re-randomise their raw origin on every power cycle, so a verified map that
        outlives its calibration is describing a frame that no longer exists."""
        jm = JM.JointMap()
        for n in JM.MODEL_TO_MOTOR:
            jm.mark_verified(n)
        jm.invalidate("re-zeroed")
        assert not jm.check_ready()[0]

    def test_velocity_without_a_measured_scale_raises_rather_than_guessing(self):
        jm = JM.JointMap()
        with pytest.raises(ValueError, match="never been measured"):
            jm.vel_to_model(np.zeros(6))
        assert np.allclose(jm.vel_to_model(np.zeros(6), fallback_rad_s=np.ones(6)), 1.0)

    def test_erpm_scale_recovers_a_known_ratio(self):
        dt, scale = 0.005, 21.0 * 60.0 / (2 * np.pi)
        t = np.arange(0, 6, dt)
        q = np.degrees(2.0 * np.sin(2 * np.pi * 0.7 * t))         # normalized degrees
        w = np.gradient(np.radians(q), dt)
        got, r2, n = JM.fit_erpm_scale(w * scale, q, dt)
        assert got == pytest.approx(scale, rel=0.02)
        assert r2 > 0.99 and n > 100

    def test_torque_uses_the_measured_gearbox_efficiency(self):
        jm = JM.JointMap()
        tau = jm.torque_to_model(np.ones(6))
        # datasheet 2.176 N*m/A for the AKE90-8, derated by the measured 0.85
        assert tau[1] == pytest.approx(2.176 * 0.85, rel=1e-9)
        assert np.allclose(jm.amps_from_torque(tau), 1.0)


# ===================================================================== safety governor
def _lim(**kw):
    d = dict(pos_lo=np.full(6, -1.0), pos_hi=np.full(6, 1.0), vel_max=np.full(6, 10.0),
             tau_peak=np.full(6, 100.0), tau_cont=np.full(6, 35.0))
    d.update(kw)
    return Limits(**d)


def _ok_args(**kw):
    d = dict(motor_pos=np.zeros(6), motor_vel=np.zeros(6), grav=np.array([0.0, 0.0, -1.0]),
             gyro=np.zeros(3), telemetry_age=0.0, deadman_age=0.0)
    d.update(kw)
    return d


class TestGovernorOwnsTheThermalObserver:
    """`current` used to be accepted by step() and silently ignored, with the caller expected to
    step the observer itself. That is a parameter that lies AND a guard that can go missing: a
    governor holding an observer nobody remembered to step reports infinite thermal headroom and
    applies no thermal derate at all, with nothing anywhere saying so."""

    def _gov(self):
        import thermal as TH
        th = TH.MotorThermalModel([TH.DEFAULT_PARAMS["AKE90-8"]] * 6, dt=0.005, t_amb=25.0,
                                  allow_uncalibrated=True)
        return SafetyGovernor(_lim(), dt=0.005, thermal=th), th

    def test_passing_current_advances_the_observer(self):
        g, th = self._gov()
        before = th.t_winding.copy()
        for _ in range(400):
            g.step(np.zeros(6), np.full(6, 200.0), np.full(6, 5.0),
                   current=np.full(6, 25.0), **_ok_args())
        assert np.all(th.t_winding > before + 0.5), (
            "25 A through six windings for 2 s must move the estimate: {} -> {}".format(
                before, th.t_winding))
        assert th.n_steps == 400, "one step per tick, not zero and not two"

    def test_omitting_current_leaves_the_observer_alone(self):
        """Callers that have no current to give (a bench test, a replay) must not get a silently
        advancing observer fed zeros -- that would COOL the estimate, the one direction a safety
        observer must never drift without evidence."""
        g, th = self._gov()
        for _ in range(50):
            g.step(np.zeros(6), np.full(6, 200.0), np.full(6, 5.0), **_ok_args())
        assert th.n_steps == 0

    def test_observe_is_usable_before_there_is_a_verdict_to_ask_for(self):
        """The approach phase wants the case-temperature correction tracking before the policy
        starts, and has no command to filter yet."""
        g, th = self._gov()
        g.observe(np.full(6, 10.0), omega=np.zeros(6), drive_temp=np.full(6, 40.0))
        assert th.n_steps == 1

    def test_the_derate_the_observer_produces_actually_reaches_the_torque_cap(self):
        """The whole point of stepping it here: the budget in step 3 must be the DERATED one, so a
        hot machine gets a smaller torque envelope on the same tick."""
        g, th = self._gov()
        cold = g._torque_budget().copy()
        th.x[:, 0] = th.t_derate + 20.0 - th.t_amb          # drive the winding node up
        hot = g._torque_budget()
        assert np.all(hot < cold), "{} not below {}".format(hot, cold)


class TestSafetyGovernor:
    def test_a_clean_command_passes_through_untouched(self):
        g = SafetyGovernor(_lim(), dt=0.005)
        v = g.step(np.full(6, 0.1), np.full(6, 200.0), np.full(6, 5.0), **_ok_args())
        assert v.ok and not v.clamped
        assert np.allclose(v.target, 0.1)

    def test_momentary_overshoot_is_capped_not_killed(self):
        g = SafetyGovernor(_lim(), dt=0.005)
        v = g.step(np.full(6, 5.0), np.full(6, 200.0), np.full(6, 5.0), **_ok_args())
        assert v.stop == STOP_NONE
        assert "position" in v.clamped
        assert np.all(v.target <= 1.0 + 1e-12)

    def test_persistent_demand_kills(self):
        """The distinction the whole module exists for: a blip is clamped, a policy leaning on the
        limit tick after tick means the control law is no longer the one that was verified."""
        g = SafetyGovernor(_lim(persist_ticks=10), dt=0.005)
        for k in range(9):
            v = g.step(np.full(6, 5.0), np.full(6, 200.0), np.full(6, 5.0), **_ok_args())
            assert v.stop == STOP_NONE, "killed on tick {} of 10".format(k)
        v = g.step(np.full(6, 5.0), np.full(6, 200.0), np.full(6, 5.0), **_ok_args())
        assert v.stop == STOP_SOFT
        assert any("position clamped" in r for r in v.reasons)

    def test_torque_cap_bounds_the_position_error_not_the_gains(self):
        g = SafetyGovernor(_lim(tau_peak=np.full(6, 20.0)), dt=0.005)
        kp = np.full(6, 200.0)
        v = g.step(np.full(6, 0.9), kp, np.full(6, 5.0), **_ok_args(motor_pos=np.zeros(6)))
        assert "torque" in v.clamped
        assert np.allclose(v.kp, kp)                       # gains untouched: dynamics preserved
        assert np.all(np.abs(v.target - 0.0) * kp <= 20.0 + 1e-9)

    def test_thermal_derating_tightens_the_torque_cap(self):
        p = TH.ThermalParams("T", k_cu=0.18, c_w=95.0, c_c=1050.0, r_wc=0.72, r_ca=1.45,
                             t_derate=90.0, t_warn=100.0, t_trip=120.0)
        m = TH.MotorThermalModel([p] * 6, dt=0.005, t_amb=25.0, allow_uncalibrated=True)
        m.reset(25.0)
        g = SafetyGovernor(_lim(), dt=0.005, thermal=m)
        cold = g._torque_budget().copy()
        m.x[:, 0] = 130.0 - m.t_amb
        hot = g._torque_budget()
        assert np.all(cold == 100.0) and np.all(hot == 35.0)

    def test_thermal_trip_is_a_soft_stop(self):
        """Overheating is not a reason to drop the robot on the floor -- put it down."""
        p = TH.ThermalParams("T", k_cu=0.18, c_w=95.0, c_c=1050.0, r_wc=0.72, r_ca=1.45)
        m = TH.MotorThermalModel([p] * 6, dt=0.005, t_amb=25.0, allow_uncalibrated=True)
        m.reset(25.0)
        g = SafetyGovernor(_lim(), dt=0.005, thermal=m)
        m.x[:, 0] = 130.0 - m.t_amb
        v = g.step(np.zeros(6), np.full(6, 200.0), np.full(6, 5.0), **_ok_args())
        assert v.stop == STOP_SOFT
        assert any("winding temperature" in r for r in v.reasons)

    @pytest.mark.parametrize("kw,needle", [
        (dict(grav=np.array([0.9, 0.0, -0.2])), "fallen"),
        (dict(gyro=np.array([0.0, 20.0, 0.0])), "body rate"),
        (dict(telemetry_age=0.2), "telemetry stale"),
        (dict(drive_err=np.array([0, 0, 3, 0, 0, 0])), "drive error"),
        (dict(drive_temp=np.array([30, 30, 85, 30, 30, 30.0])), "case temperature"),
    ])
    def test_hard_kills(self, kw, needle):
        g = SafetyGovernor(_lim(), dt=0.005)
        v = g.step(np.zeros(6), np.full(6, 200.0), np.full(6, 5.0), **_ok_args(**kw))
        assert v.stop == STOP_HARD
        assert any(needle in r for r in v.reasons), v.reasons
        assert np.all(v.kp == 0.0) and np.all(v.kd == 0.0) and v.limp

    def test_non_finite_anything_is_a_hard_kill(self):
        for bad in ("target", "kp", "motor_pos"):
            g = SafetyGovernor(_lim(), dt=0.005)
            args = _ok_args()
            t, kp = np.zeros(6), np.full(6, 200.0)
            if bad == "target":
                t = t.copy(); t[2] = np.nan
            elif bad == "kp":
                kp = kp.copy(); kp[0] = np.inf
            else:
                args["motor_pos"] = args["motor_pos"].copy(); args["motor_pos"][4] = np.nan
            v = g.step(t, kp, np.full(6, 5.0), **args)
            assert v.stop == STOP_HARD and v.limp

    def test_deadman_release_is_a_soft_ramp_to_limp(self):
        g = SafetyGovernor(_lim(soft_stop_s=0.1), dt=0.005)
        g.step(np.full(6, 0.2), np.full(6, 200.0), np.full(6, 5.0), **_ok_args())
        ramps = []
        for _ in range(30):
            v = g.step(np.full(6, 0.2), np.full(6, 200.0), np.full(6, 5.0),
                       **_ok_args(deadman_age=1.0))
            ramps.append(v.ramp)
        assert v.stop == STOP_SOFT
        assert ramps == sorted(ramps, reverse=True)          # monotone bleed-down
        assert v.limp and np.all(v.kp == 0.0)
        # and it FROZE the target rather than tracking the policy
        assert np.allclose(v.target, 0.2)

    def test_gain_clamp_does_not_kill(self):
        """The deployed impedance channel spans kp 40-500 / kd 1.0-5.0, exactly the wire ranges.
        A clamp here means the bundle changed -- surface it, do not drop the robot for it."""
        g = SafetyGovernor(_lim(persist_ticks=3), dt=0.005)
        for _ in range(20):
            v = g.step(np.zeros(6), np.full(6, 900.0), np.full(6, 9.0), **_ok_args())
        assert v.stop == STOP_NONE
        assert "gains" in v.clamped
        assert np.all(v.kp <= 500.0) and np.all(v.kd <= 5.0)

    def test_a_latched_stop_never_clears_itself(self):
        g = SafetyGovernor(_lim(), dt=0.005)
        g.step(np.zeros(6), np.full(6, 200.0), np.full(6, 5.0),
               **_ok_args(grav=np.array([0.9, 0.0, -0.2])))
        for _ in range(50):
            v = g.step(np.zeros(6), np.full(6, 200.0), np.full(6, 5.0), **_ok_args())
        assert v.stop == STOP_HARD
        g.reset()
        assert g.step(np.zeros(6), np.full(6, 200.0), np.full(6, 5.0), **_ok_args()).ok


# ===================================================================== the bundle + controller
@pytest.fixture(scope="module")
def b():
    from bundle import Bundle
    return Bundle.load(BUNDLE)


@pytest.mark.skipif(not BUNDLE.exists(), reason="no exported bundle (run export_policy.py)")
class TestBundleAndController:
    def test_impedance_span_fits_the_drive_wire_ranges(self, b):
        """If this ever fails, the policy is asking for gains the CAN frame cannot carry and the
        deployed control law silently stops being the trained one."""
        kp_lo = b["imp_kp_base"] / b.imp_kp_dn
        kp_hi = b["imp_kp_base"] * b.imp_kp_up
        kd_lo = b["imp_kd_base"] / b.imp_kd_dn
        kd_hi = b["imp_kd_base"] * b.imp_kd_up
        assert kp_lo.min() >= mit.KP_MIN and kp_hi.max() <= mit.KP_MAX
        assert kd_lo.min() >= mit.KD_MIN and kd_hi.max() <= mit.KD_MAX

    def test_joint_targets_stay_inside_the_wire_position_range(self, b):
        assert np.all(b["ctrl_lo"] >= mit.P_MIN) and np.all(b["ctrl_hi"] <= mit.P_MAX)

    def test_controller_is_deterministic_and_bounded(self, b):
        from controller import PolicyController
        outs = []
        for _ in range(2):
            c = PolicyController(b)
            q = np.asarray(b["default_motor_pos"])
            c.start(q, np.zeros(6), np.zeros(6), np.array([0.0, 0.0, -1.0]), np.zeros(3),
                    v_cmd=0.5)
            for _ in range(50):
                cmd = c.step(q, np.zeros(6), np.zeros(6), np.array([0.0, 0.0, -1.0]), np.zeros(3))
            outs.append((cmd.target.copy(), cmd.kp.copy(), cmd.action.copy()))
        for a, bb in zip(outs[0], outs[1]):
            assert np.array_equal(a, bb)                    # bit-identical across instances
        assert np.all(np.abs(outs[0][2]) <= 1.0)            # actions clipped
        assert np.all(outs[0][0] >= b["ctrl_lo"]) and np.all(outs[0][0] <= b["ctrl_hi"])

    def test_first_command_is_the_trained_stance(self, b):
        """start() must leave the robot commanded to the stance, not to wherever the filter
        happened to initialise -- the whole approach phase depends on it."""
        from controller import PolicyController
        c = PolicyController(b)
        q = np.asarray(b["default_motor_pos"])
        c.start(q, np.zeros(6), np.zeros(6), np.array([0.0, 0.0, -1.0]), np.zeros(3))
        cmd = c.step(q, np.zeros(6), np.zeros(6), np.array([0.0, 0.0, -1.0]), np.zeros(3))
        assert np.allclose(cmd.target, b["nominal_ctrl"], atol=1e-9)

    def test_policy_refuses_a_privileged_observation_it_cannot_have(self, b):
        from controller import PolicyController
        c = PolicyController(b)
        c.obs_base_vel = True                               # pretend the bundle wanted it
        with pytest.raises(ValueError, match="PRIVILEGED"):
            c.start(np.zeros(6), np.zeros(6), np.zeros(6), np.array([0.0, 0.0, -1.0]), np.zeros(3))

    def test_command_deadband_matches_training(self, b):
        from controller import PolicyController
        c = PolicyController(b)
        c.set_command(0.05)                                 # below cmd_deadband 0.12
        assert c._v_cmd == 0.0 and c._standing
        c.set_command(0.5)
        assert c._v_cmd == 0.5 and not c._standing


# ===================================================================== vendored gait drift
def test_vendored_gait_is_identical_to_the_training_module():
    """robot/deploy/fourier_gait.py is a byte-for-byte copy of walk_mit/fourier_gait.py. If they drift, the
    robot reconstructs a different gait from the same action and nothing reports it."""
    import hashlib
    train = REPO / "walk_mit" / "fourier_gait.py"
    if not train.exists():
        pytest.skip("training package not present")
    h = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (train, DEPLOY / "fourier_gait.py")]
    assert h[0] == h[1], ("robot/deploy/fourier_gait.py has drifted from walk_mit/fourier_gait.py -- "
                          "re-copy it and re-run verify_export.py")


# ===================================================================== thermal burst excitation
def _env(**kw):
    d = dict(amps=25.0, duration_s=60.0, centre_deg=0.0, window_deg=8.0, speed_erpm=600.0)
    d.update(kw)
    return TE.Envelope(**d)


class TestBurstExciter:
    def test_current_is_always_saturated(self):
        """The deposited energy has to depend on the command, not on the mechanics. If the current
        can sag, the burst measures the plant instead of heating it."""
        ex = TE.BurstExciter(_env(), ramp_s=0.0)
        pos, spd = 0.0, 0.0
        for k in range(2000):
            a, done, _ = ex.step(k * 0.005, pos, spd, 30, 0, 0.0)
            assert done or abs(abs(a) - 25.0) < 1e-9
            spd += float(np.sign(a)) * 40.0          # crude rotor: current accelerates it
            pos += spd * 0.005 / 60.0

    def test_a_blocked_rotor_that_keeps_turning_aborts_instead_of_reversing(self):
        """The blocked law's whole point: sustained motion means the clamp is failing. Reversing
        current at that moment is what self-excited on right.cam (2026-08-28). Debounced, because
        a single twitch is backlash — see TestSlipDebounce."""
        ex = TE.BurstExciter(_env(), ramp_s=0.0)
        done = False
        for k in range(TE.BLOCKED_SLIP_TICKS + 5):
            a, done, ab = ex.step(k * 0.005, 0.0, TE.BLOCKED_SLIP_ERPM + 50.0, 30, 0, 0.0)
            if done:
                break
        assert done and a == 0.0 and "slipping" in ab
        assert ex.n_reversals == 0

    def test_a_blocked_rotor_gets_steady_unidirectional_current(self):
        ex = TE.BurstExciter(_env(), ramp_s=0.0)
        for k in range(400):
            a, done, ab = ex.step(k * 0.005, 0.0, 0.0, 30, 0, 0.0)
            assert not done, ab
            assert a == pytest.approx(25.0)           # never reverses, never sags
        assert ex.n_reversals == 0

    def test_the_retracted_free_joint_dither_is_still_reproducible(self):
        """Kept ONLY so the failure stays testable: blocked=False is the law that ran away, and
        nothing in the daemon constructs it any more."""
        ex = TE.BurstExciter(_env(window_deg=20.0, speed_erpm=300.0, blocked=False), ramp_s=0.0)
        pos, spd = 0.0, 0.0
        for k in range(4000):
            a, done, ab = ex.step(k * 0.005, pos, spd, 30, 0, 0.0)
            if done:
                break
            spd += float(np.sign(a)) * 30.0
            pos += spd * 0.005 / 60.0
        assert ab is None
        assert ex.n_reversals > 5                     # it really is oscillating
        assert abs(ex.pos_max - ex.pos_min) < 20.0    # ...inside the window, not against it

    @pytest.mark.parametrize("kw,needle", [
        (dict(temp_c=95), "temperature"),
        (dict(err=7), "error code"),
        (dict(telemetry_age=1.0), "status frame"),
        (dict(pos_deg=500.0), "left its window"),
    ])
    def test_aborts_command_zero_current(self, kw, needle):
        ex = TE.BurstExciter(_env())
        args = dict(t=0.5, pos_deg=0.0, spd_erpm=0.0, temp_c=30, err=0, telemetry_age=0.0)
        args.update(kw)
        a, done, ab = ex.step(**args)
        assert a == 0.0 and done and needle in ab

    def test_energy_integral_matches_a_constant_current(self):
        ex = TE.BurstExciter(_env(amps=10.0, duration_s=10.0), ramp_s=0.0)
        for k in range(1, 2001):
            ex.step(k * 0.005, 0.0, 0.0, 30, 0, 0.0)
        # 10 A for 10 s = 1000 A^2 s, up to the final partial step
        assert ex.summary()["i_sq_dt"] == pytest.approx(1000.0, rel=0.01)
        assert ex.summary()["i_rms"] == pytest.approx(10.0, rel=0.01)

    def test_envelope_refuses_a_joint_pinned_at_its_limit(self):
        with pytest.raises(ValueError, match="safe window"):
            TE.Envelope(amps=10, duration_s=5, centre_deg=0.0, window_deg=8.0,
                        pos_lo=-0.2, pos_hi=0.2)

    def test_envelope_clamps_beyond_the_hard_ceilings(self):
        e = TE.Envelope(amps=1e6, duration_s=1e6, centre_deg=0.0)
        assert e.amps == TE.MAX_AMPS and e.duration_s == TE.MAX_DURATION_S


class TestBurstSizing:
    """The check that stops an afternoon of unmeasurable runs. MEASURED against the placeholder
    AKE90-8 parameters: 12 A x 10 s deposits 216 J and moves the case 0.2 degC."""

    def test_the_intuitive_burst_is_rejected_as_unmeasurable(self):
        p = TH.DEFAULT_PARAMS["AKE90-8"]
        ok, why, pred = TE.check_burst(p, 12.0, 10.0)
        assert not ok and "resolve" in why
        assert pred["case_c"] < 0.5

    def test_an_over_energetic_burst_is_rejected_on_the_WINDING(self):
        p = TH.DEFAULT_PARAMS["AKE90-8"]
        ok, why, pred = TE.check_burst(p, 30.0, 180.0)
        assert not ok and "winding" in why
        # and the trap it exists for: the case reading would have looked entirely reasonable
        assert pred["case_c"] < 30.0 and pred["winding_c"] > 100.0

    def test_suggest_lands_inside_the_accepted_band(self):
        p = TH.DEFAULT_PARAMS["AKE90-8"]
        amps, dur = TE.suggest(p, target_rise_c=6.0, amps=25.0)
        ok, why, _ = TE.check_burst(p, amps, dur)
        assert ok, why

    def test_winding_rise_is_ten_times_the_case_rise(self):
        p = TH.DEFAULT_PARAMS["AKE90-8"]
        pr = TE.predict(p, 25.0, 60.0)
        assert pr["winding_c"] / pr["case_c"] == pytest.approx(
            (p.c_w + p.c_c) / p.c_w, rel=1e-9)


class TestThermalStore:
    def store(self, tmp_path):
        sys.path.insert(0, str(REPO / "robot" / "fixed_gait" / "webui"))
        import thermalstore
        return thermalstore, str(tmp_path / "runs.json")

    def test_a_burst_is_not_usable_until_it_is_annotated(self, tmp_path):
        ts, path = self.store(tmp_path)
        env = _env(duration_s=60.0).as_dict()
        r = ts.add_burst("left.thigh", env, {"i_rms": 25.0}, 30, 34, path=path)
        assert r["t_peak_c"] is None
        s = ts.summary(path)["left.thigh"]
        assert s["bursts"] == 1 and s["usable"] == 0
        ts.annotate(r["id"], path=path, t_start_c=30.0, t_peak_c=36.0)
        assert ts.summary(path)["left.thigh"]["usable"] == 1

    def test_only_operator_fields_are_writable(self, tmp_path):
        ts, path = self.store(tmp_path)
        r = ts.add_burst("left.thigh", _env().as_dict(), {"i_rms": 25.0}, 30, 34, path=path)
        with pytest.raises(ValueError, match="operator-supplied"):
            ts.annotate(r["id"], path=path, summary={"i_rms": 999.0})

    def test_cooldown_points_stay_sorted(self, tmp_path):
        ts, path = self.store(tmp_path)
        c = ts.start_cooldown("left.thigh", path=path)
        for t, v in ((600, 30.0), (0, 45.0), (300, 36.0)):
            ts.add_point(c["id"], t, v, path=path)
        pts = ts.load(path)["cooldowns"][0]["points"]
        assert [p[0] for p in pts] == [0.0, 300.0, 600.0]

    def test_summary_says_what_is_still_missing(self, tmp_path):
        ts, path = self.store(tmp_path)
        r = ts.add_burst("left.thigh", _env().as_dict(), {"i_rms": 25.0}, 30, 34, path=path)
        ts.annotate(r["id"], path=path, t_start_c=30.0, t_peak_c=36.0)
        s = ts.summary(path)["left.thigh"]
        assert not s["ready"]
        assert any("burst" in n for n in s["needs"])
        assert any("cooldown" in n for n in s["needs"])


class TestSlipDebounce:
    """MEASURED 2026-08-29: a genuinely static joint aborted 26 ms in on a 0.3 deg / 750 ERPM
    twitch while the soft-start was still at 0.04 A rms. Backlash and fixture compliance always
    produce that step; a single-sample threshold cannot tell it from a clamp letting go."""

    def test_a_brief_twitch_does_not_abort_a_blocked_run(self):
        ex = TE.BurstExciter(_env(), ramp_s=0.0)
        for k in range(6):                                  # 30 ms of motion, then still
            a, done, ab = ex.step(k * 0.005, 0.3, 750.0, 30, 0, 0.0)
            assert not done, ab
        for k in range(6, 200):
            a, done, ab = ex.step(k * 0.005, 0.3, 0.0, 30, 0, 0.0)
            assert not done, ab
        assert a == pytest.approx(25.0)

    def test_sustained_motion_still_aborts(self):
        ex = TE.BurstExciter(_env(), ramp_s=0.0)
        done = False
        for k in range(TE.BLOCKED_SLIP_TICKS + 5):
            _a, done, ab = ex.step(k * 0.005, 0.0, TE.BLOCKED_SLIP_ERPM + 500.0, 30, 0, 0.0)
            if done:
                break
        assert done and "clamp is slipping" in ab

    def test_a_free_envelope_refuses_the_current_mode_law(self):
        """Free-rotor runs are position-mode sines now; routing one through the current-mode
        step() would resurrect exactly the class of law the retraction is about."""
        ex = TE.BurstExciter(_env(mode="free"), ramp_s=0.0)
        with pytest.raises(ValueError, match="step_sine"):
            ex.step(0.1, 0.0, 0.0, 30, 0, 0.0)


class TestFreeRotorSine:
    """The free-rotor law: track a position sine, heat by fighting your own rotor inertia.

    The current is set by physics (I = J*alpha/Kt), not by the amps knob, and position mode has
    no current cap this module controls -- so the guards are all on MEASURED quantities: the
    average current (too little heat to be worth the time), the I^2*dt integral (never deposit
    more than the winding gate approved), speed, and net drift out of the sine's band."""

    def free(self, **kw):
        d = dict(mode="free", freq_hz=6.0, sine_amp_deg=20.0)
        d.update(kw)
        return TE.BurstExciter(_env(**d), ramp_s=0.5)

    def track(self, ex, t):
        """Feed the exciter its own target back, i.e. a drive that tracks perfectly."""
        want = getattr(self, "_want", ex.env.centre)
        out = ex.step_sine(t, want, 0.0, 30, 0, 0.0, i_meas=ex.env.amps)
        self._want = out[0]
        return out

    def test_the_sine_stays_inside_its_band_and_finishes_on_time(self):
        ex = self.free()
        e = ex.env
        for k in range(int(e.duration_s / 0.005)):
            want, done, ab = self.track(ex, k * 0.005)
            assert not done, ab
            assert abs(want - e.centre) <= e.sine_amp + 1e-9
        _w, done, ab = self.track(ex, e.duration_s)
        assert done and ab is None

    def test_the_ramp_removes_the_velocity_step_at_entry(self):
        ex = self.free()
        want, _done, _ab = ex.step_sine(0.005, 0.0, 0.0, 30, 0, 0.0)
        # one tick in, the raised cosine has barely opened: the command is still at the centre
        assert abs(want - ex.env.centre) < 0.1

    def test_a_sine_that_cannot_draw_the_target_current_stops_early(self):
        ex = self.free()
        done = False
        for k in range(400):
            _w, done, ab = ex.step_sine(TE.FREE_CURRENT_GRACE_S + k * 0.005, 0.0, 0.0, 30, 0, 0.0,
                                        i_meas=1.0)
            if done:
                break
        assert done and "rotor inertia" in ab

    def test_the_run_can_never_outspend_the_approved_energy(self):
        """Position mode cannot cap the current, so the integral is the wall: a drive drawing far
        more than the amps knob must abort long before the duration is up."""
        ex = self.free()
        e = ex.env
        budget = TE.FREE_ENERGY_HEADROOM * e.amps ** 2 * e.duration_s
        done, t = False, 0.0
        for k in range(int(e.duration_s / 0.005)):
            t = k * 0.005
            _w, done, ab = ex.step_sine(t, 0.0, 0.0, 30, 0, 0.0, i_meas=4.0 * e.amps)
            if done:
                break
        assert done and "sized for" in ab
        assert t < 0.5 * e.duration_s
        assert ex.i_sq_dt <= budget * 1.1

    def test_brief_overshoot_is_tolerated_but_net_drift_aborts(self):
        ex = self.free()
        e = ex.env
        far = e.centre + e.sine_amp + TE.FREE_DRIFT_MARGIN_DEG + 2.0
        for k in range(TE.FREE_DRIFT_TICKS - 2):            # brief excursion: loop overshoot
            _w, done, ab = ex.step_sine(0.1 + k * 0.005, far, 0.0, 30, 0, 0.0, i_meas=e.amps)
            assert not done, ab
        _w, done, ab = ex.step_sine(0.5, e.centre, 0.0, 30, 0, 0.0, i_meas=e.amps)
        assert not done                                     # back inside: counter resets
        done = False
        for k in range(TE.FREE_DRIFT_TICKS + 2):            # sustained: something is on the shaft
            _w, done, ab = ex.step_sine(0.6 + k * 0.005, far, 0.0, 30, 0, 0.0, i_meas=e.amps)
            if done:
                break
        assert done and "net rotation" in ab

    def test_overspeed_still_aborts(self):
        ex = self.free()
        _w, done, ab = ex.step_sine(0.1, 0.0, TE.FREE_SPEED_ERPM + 500.0, 30, 0, 0.0)
        assert done and "ERPM" in ab


class TestDeployMap:
    """deploy_map.json, built from fklut's fitted cam/thigh map plus the measured abduction
    convention. Both halves were established on 2026-08-29: the fit was decisive at 100% coverage
    on each leg, and the operator confirmed +normalized = outward on BOTH legs while the model
    moves hip_roll_L outward and hip_roll_R inward at +0.25 rad."""

    MM = {"left": {"cam": 1, "thigh": -1, "cam_off_deg": -24.5, "thigh_off_deg": -1.2},
          "right": {"cam": 1, "thigh": -1, "cam_off_deg": -18.7, "thigh_off_deg": -8.4},
          "verified": {"left": True, "right": True}}

    def _build(self):
        import make_deploy_map
        return make_deploy_map.build(self.MM, when="test")[0]

    def test_cam_and_thigh_transcribe_from_the_fit(self):
        jm = self._build()
        assert jm.e["left.cam"]["offset_deg"] == pytest.approx(-24.5)
        assert jm.e["right.thigh"]["sign"] == pytest.approx(-1.0)

    def test_the_two_abduction_signs_are_opposite(self):
        """Both hip_roll joints share the axis (+1,0,0) while the robot reads + outward on both,
        so exactly one side must be inverted. Equal signs here would mean one leg abducts the
        wrong way -- a legal pose, and the wrong one."""
        jm = self._build()
        assert jm.e["left.abd"]["sign"] * jm.e["right.abd"]["sign"] < 0

    def test_abduction_offsets_are_zero(self):
        """Zeroed aligned with the base, which is the model's qpos-0, and hip_roll's range is
        symmetric -- no 4-bar dead centre to hide an offset in."""
        jm = self._build()
        for n in ("left.abd", "right.abd"):
            assert jm.e[n]["offset_deg"] == pytest.approx(0.0)

    def test_the_map_is_ready_and_round_trips(self):
        jm = self._build()
        ok, why = jm.check_ready()
        assert ok, why
        model = np.array([0.0, 0.0, 0.12, 0.0, 0.0, -0.12])
        back = jm.to_model_rad(jm.to_norm_deg(model))
        assert np.allclose(back, model, atol=1e-12)

    def test_an_unverified_fit_is_refused(self):
        import make_deploy_map
        mm = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.MM.items()}
        mm["verified"] = {"left": False, "right": True}
        jm, notes = make_deploy_map.build(mm, when="test")
        assert any("left" in n for n in notes)
        ok, _why = jm.check_ready()
        assert not ok, "an unfitted side must not produce a ready map"
