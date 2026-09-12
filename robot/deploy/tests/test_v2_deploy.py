"""Offline tests for the v2 (walk_v2) deployment path. No hardware, no CAN, no jax, no mujoco.

    python -m pytest robot/deploy/tests/test_v2_deploy.py -q

THE ONE THAT MATTERS is `TestAgainstTheTrainedLoop`. `walk_v2/tools/trace.py` records, from the
real MJX training environment, every per-tick number of a deterministic 85-tick rollout: the spec,
the phase, the commanded target, the gains, the measurements and the 33-dim observation frame. It
is the fixture the two training arms (MJX/JAX on GPU, classic MuJoCo on CPU) cross-check each
other on. Here it is used a third way: the deployed numpy runtime is driven through the SAME
recorded states, and its targets, gains and observation frames are diffed against what the trainer
actually produced.

That closes the gap `verify_export.py` closes for v1, and it closes it without torch or MuJoCo, so
it runs on every commit rather than as a desktop ritual. What it does NOT cover: the trained
weights (the fixture drives a fixed spec, not a policy) and the plant. Those need the training
package; this proves the control law and the observation are the ones the policy was trained
against, which is where the silent failures live.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

DEPLOY = Path(__file__).resolve().parents[1]
REPO = DEPLOY.parents[1]
for p in (str(DEPLOY),):
    if p not in sys.path:
        sys.path.insert(0, p)

import gait_v2 as G                                                   # noqa: E402
from bundle import Bundle                                            # noqa: E402
from controller_v2 import PolicyControllerV2                          # noqa: E402
from v2_fixture import (ACTION_DIM, ACTOR_DIM, ACTOR_DIM_V3, FRAME_DIM,  # noqa: E402
                        FRAME_DIM_V3, HIST_LEN,
                        HIST_STRIDE, N_RESIDUAL, ONCE_DIM, Q_LO, SPEC_DIM,
                        VEL_LIMIT, v2_bundle)

TRACE = REPO / "walk_v2" / "results" / "trace_mjx.json"


def _trace():
    if not TRACE.exists():
        pytest.skip("no walk_v2/results/trace_mjx.json (the walk_v2 package is not checked out)")
    return json.loads(TRACE.read_text())


# ===================================================================== the vendored generator
class TestGaitPortIsTheTrainedLaw:
    """`gait_v2.py` is a hand copy of `walk_v2/gait.py` with jax removed. This is the net under it."""

    def test_targets_and_gains_match_the_recorded_mjx_rollout(self):
        tr = _trace()
        gp = G.GaitParams.from_meta(tr["gait_params"])
        nominal = np.array(tr["nominal_ctrl"])
        dt = tr["control_dt"]
        prev, prev_t, prev_v = None, nominal.copy(), np.zeros(6)
        worst = np.zeros(3)
        for row in tr["rows"]:
            if row.get("done"):
                break               # the done row holds the auto-reset state, not this tick's cmd
            grav = np.array(prev["grav"]) if prev else np.array([0.0, 0.0, -1.0])
            gyro = np.array(prev["gyro"]) if prev else np.zeros(3)
            phi = prev["phase"] if prev else 0.0
            res = 0.05 * np.sin(2 * np.pi * 3.0 * (row["t"] - dt) + np.arange(6))
            # this fixture was recorded with pitch_reflex_rate_lp = 0, i.e. the raw pitch rate
            tgt, kp, kd, _ = G.assemble(np.array(row["spec"]), res, phi, grav[1], gyro[0],
                                        grav[0], gyro[1], nominal, gp)
            tgt = np.clip(tgt, Q_LO, -Q_LO)
            tgt, prev_v = G.slew_limit(tgt, prev_t, prev_v, VEL_LIMIT, 0.0, dt)
            worst = np.maximum(worst, [np.abs(np.array(row["target"]) - tgt).max(),
                                       np.abs(np.array(row["kp"]) - kp).max(),
                                       np.abs(np.array(row["kd"]) - kd).max()])
            prev, prev_t = row, np.array(row["target"])
        # float32 (MJX) against float64 (here) on the same states: this is the rounding floor, and
        # anything above 1e-4 is an arithmetic difference, not precision
        assert worst.max() < 1e-4, (
            "the vendored generator disagrees with the trained one: target {:.2e}, kp {:.2e}, "
            "kd {:.2e}".format(*worst))

    def test_the_mirror_symmetric_spec_produces_a_mirror_symmetric_stance(self):
        # spec 0 is the neutral gait: the series are flat, the knobs are zero, so the target is
        # the nominal stance plus the reflexes, and with the robot upright and still that is the
        # stance exactly. A sign slip in the L/R structure shows up here and nowhere else.
        gp = G.GaitParams.from_meta(v2_bundle().meta["gait"])
        nominal = np.array([0.0, 0.0, 0.12, 0.0, 0.0, -0.12])
        for phi in (0.0, 1.0, 3.3, 6.0):
            tgt, kp, kd, _ = G.assemble(np.zeros(SPEC_DIM), np.zeros(6), phi,
                                        0.0, 0.0, 0.0, 0.0, nominal, gp)
            assert np.allclose(tgt, nominal, atol=1e-12)
            assert np.allclose(kp, gp.drive_kp) and np.allclose(kd, gp.drive_kd)

    def test_the_impedance_channel_stays_inside_the_drives_wire_ranges(self):
        # the force-control frame encodes kp over 0-500 and kd over 0-5; a spec that asks for more
        # would be silently quantised at the wire, so the exp map has to land inside them
        gp = G.GaitParams.from_meta(v2_bundle().meta["gait"])
        for sign in (-1.0, 1.0):
            spec = np.zeros(SPEC_DIM)
            spec[G.I_KP] = sign
            spec[G.I_KD] = sign
            for phi in np.linspace(0, 2 * np.pi, 17):
                _, kp, kd, _ = G.assemble(spec, np.zeros(6), phi, 0, 0, 0, 0,
                                          np.zeros(6), gp)
                assert kp.min() > 0.0 and kp.max() <= 500.0, kp
                assert kd.min() > 0.0 and kd.max() <= 5.0, kd


# ============================================================ the v3 heading channel
class TestTheHeadingChannel:
    """v3 gives the actor its own dead-reckoned heading. Three things must hold on the robot, and
    all three fail silently: the channel has to exist, it has to be the integrated gyro clipped the
    way training clipped it, and a v2 bundle must be completely unaffected."""

    @staticmethod
    def _ctrl(**kw):
        b = v2_bundle(objective="joystick", **kw)
        c = PolicyControllerV2(b)
        z6, g = np.zeros(6), np.array([0.0, 0.0, -1.0])
        c.start(z6, z6, z6, g, np.zeros(3))
        return c

    def test_a_v3_bundle_builds_a_34_wide_frame_and_a_v2_bundle_still_builds_33(self):
        assert self._ctrl(heading=True).frame_dim == FRAME_DIM_V3
        assert self._ctrl(heading=True)._obs().size == ACTOR_DIM_V3
        assert self._ctrl().frame_dim == FRAME_DIM          # unchanged by the v3 code path
        assert self._ctrl()._obs().size == ACTOR_DIM

    def test_the_heading_is_the_integrated_gyro_not_the_rate(self):
        """The whole point of v3: a heading has a set point, a rate does not. v2's lp_yaw already
        carried the instantaneous rate, which here would read 1.0 forever.

        The count is 49, not 50, and that is deliberate rather than an off-by-one: `step` skips
        `_proprio` exactly once after `start`, because `start` already latched that measurement
        into every row of the history. So `start` integrates the gyro it was given (a standing
        robot: ~0) and the first `step` integrates nothing. Over a run the difference from the
        trainer is one tick of yaw rate -- ~1e-4 rad at any rate worth reporting -- but it is a real
        difference and this test is where it is written down."""
        c = self._ctrl(heading=True)
        z6, g = np.zeros(6), np.array([0.0, 0.0, -1.0])
        for _ in range(50):
            c.step(z6, z6, z6, g, np.array([0.0, 0.0, 1.0]))
        assert c.heading_deg() == pytest.approx(np.degrees(0.49), abs=1e-6)
        frame = c._obs()[(HIST_LEN - 1) * FRAME_DIM_V3:HIST_LEN * FRAME_DIM_V3]
        assert frame[-1] == pytest.approx(0.49, abs=1e-5)

    def test_the_heading_saturates_where_training_saturated_it(self):
        """Clipped at +-pi/2, not wrapped. A wrapped angle puts a discontinuity in the observation
        exactly when the robot is most sideways, which is the worst moment for one."""
        c = self._ctrl(heading=True)
        z6, g = np.zeros(6), np.array([0.0, 0.0, -1.0])
        for _ in range(300):                                # 3 rad of yaw, well past pi/2
            c.step(z6, z6, z6, g, np.array([0.0, 0.0, 1.0]))
        frame = c._obs()[(HIST_LEN - 1) * FRAME_DIM_V3:HIST_LEN * FRAME_DIM_V3]
        assert frame[-1] == pytest.approx(np.pi / 2, abs=1e-5)

    def test_zero_heading_moves_the_origin_and_nothing_else_does(self):
        """The operator declares which way is straight. Nothing may re-zero it implicitly: a
        heading that silently reset itself would make the robot veer."""
        c = self._ctrl(heading=True)
        z6, g = np.zeros(6), np.array([0.0, 0.0, -1.0])
        for _ in range(50):
            c.step(z6, z6, z6, g, np.array([0.0, 0.0, 1.0]))
        assert c.heading_deg() > 25.0
        c.set_speed(1.0)                                    # an ordinary command must not reset it
        c.step(z6, z6, z6, g, np.zeros(3))
        assert c.heading_deg() > 25.0
        c.zero_heading()
        assert c.heading_deg() == pytest.approx(0.0)


# ===================================================================== the bundle
class TestBundleGenerations:
    def test_a_v2_bundle_loads_and_derives_the_shared_vocabulary(self):
        b = v2_bundle()
        assert b.version == 2
        assert b.n_actor == ACTOR_DIM                      # NOT frame_dim * history_len
        assert b.control_hz == pytest.approx(100.0)
        # the aliases the governor and the panel are written against
        assert np.array_equal(b["ctrl_lo"], b["q_lo"])
        assert np.array_equal(b["ctrl_hi"], b["q_hi"])
        assert np.array_equal(b["imp_kp_base"], np.asarray(b.meta["gait"]["drive_kp"]))
        assert b.vn_epsilon == b.meta["obs_eps"]
        assert b.cmd_v_fwd_trained == 0.0                  # v2 has no velocity channel

    def test_a_v2_bundle_missing_a_meta_field_the_runtime_reads_is_refused(self):
        b = v2_bundle()
        for key in ("pitch_reflex_rate_lp", "motor_accel_limit", "objective", "gait"):
            meta = dict(b.meta)
            meta.pop(key)
            with pytest.raises(ValueError, match=key):
                Bundle(dict(b.a), meta)

    def test_a_v2_bundle_whose_widths_disagree_with_its_meta_is_refused(self):
        b = v2_bundle()
        with pytest.raises(ValueError, match="actor_dim"):
            Bundle(dict(b.a), dict(b.meta, once_dim=ONCE_DIM + 1))

    def test_an_unknown_bundle_version_is_refused(self):
        b = v2_bundle()
        with pytest.raises(ValueError, match="not one of"):
            Bundle(dict(b.a), dict(b.meta, bundle_version=3))

    def test_a_v2_bundle_survives_a_round_trip_through_npz(self, tmp_path):
        b = v2_bundle()
        p = tmp_path / "v2.npz"
        np.savez(p, meta=np.array(json.dumps(b.meta, sort_keys=True)),
                 **{k: v for k, v in b.a.items() if k not in b.derived_keys})
        back = Bundle.load(p)
        assert back.version == 2 and back.n_actor == ACTOR_DIM
        assert np.allclose(back["nominal_ctrl"], b["nominal_ctrl"])

    @pytest.mark.skipif(not (DEPLOY / "bundles" / "imp_m2_long_204M.npz").exists(),
                        reason="no v1 bundle on disk")
    def test_the_v1_bundles_still_load_unchanged(self):
        b = Bundle.load(DEPLOY / "bundles" / "imp_m2_long_204M.npz")
        assert b.version == 1
        assert b.n_actor == int(b.meta["frame_dim"]) * int(b.meta["history_len"])
        assert b.derived_keys == ()          # nothing is synthesised for a v1 bundle

    def test_the_library_variant_is_refused_rather_than_approximated(self):
        # its spec comes from a gait library plus a Raibert prior, neither of which is in the file
        with pytest.raises(ValueError, match="library"):
            PolicyControllerV2(v2_bundle(spec_source="library"))


# ===================================================================== the runtime
def _upright():
    return np.array([0.0, 0.0, -1.0]), np.zeros(3)


def _still(ctrl, n=1):
    """n ticks of a motionless, upright robot at the stance."""
    grav, gyro = _upright()
    out = None
    for _ in range(n):
        out = ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
    return out


class TestObservationAndLatch:
    def test_the_observation_is_history_then_spec_then_task_then_commit(self):
        ctrl = PolicyControllerV2(v2_bundle())
        grav, gyro = _upright()
        obs = ctrl.start(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        assert obs.shape == (ACTOR_DIM,)
        once = obs[FRAME_DIM * HIST_LEN:]
        assert once.shape == (ONCE_DIM,)
        assert np.array_equal(once[:SPEC_DIM], np.zeros(SPEC_DIM))   # nothing latched yet
        assert once[SPEC_DIM] == 0.0                                 # run flag: STOPPED
        assert once[SPEC_DIM + 1] == 1.0                             # the line is far away
        assert once[SPEC_DIM + 2] == 1.0                             # reset commits

    def test_start_fills_the_whole_history_with_one_frame_and_does_not_push_again(self):
        ctrl = PolicyControllerV2(v2_bundle())
        grav, gyro = _upright()
        pos = ctrl.nominal + 0.03
        ctrl.start(pos, np.zeros(6), np.zeros(6), grav, gyro)
        h = ctrl._history
        assert np.allclose(h, h[0]), "reset must latch one frame into every row"
        # ... and the reset frame carries a ZERO torque channel and phase 0, as _reset_one does
        assert np.allclose(h[0, 12:18], 0.0)
        assert h[0, 25] == pytest.approx(1.0) and h[0, 26] == pytest.approx(0.0)
        before = h.copy()
        ctrl.step(pos, np.zeros(6), np.zeros(6), grav, gyro)
        assert np.allclose(ctrl._history, before), (
            "the first step must not push: in the sim the first action sees a history of the "
            "reset frame alone")

    def test_the_spec_is_latched_and_only_a_clock_wrap_releases_it(self):
        ctrl = PolicyControllerV2(v2_bundle())
        grav, gyro = _upright()
        ctrl.start(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        # freq_raw 0 -> the middle of [0.5, 5.0] Hz = 2.75 Hz -> a cycle every ~36.4 ticks
        a1 = np.zeros(ACTION_DIM)
        a1[G.I_S_CAM.start + 1] = 0.5
        c = ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro, override_action=a1)
        assert c.commit is True and c.spec[G.I_S_CAM.start + 1] == pytest.approx(0.5)
        # a different spec on a non-commit tick must be discarded outright
        a2 = np.zeros(ACTION_DIM)
        a2[G.I_S_CAM.start + 1] = -1.0
        seen, ticks_to_commit = [], 0
        for _ in range(200):
            c = ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro, override_action=a2)
            seen.append(bool(c.commit))
            if not any(seen[:-1]) and c.commit:
                ticks_to_commit = len(seen)
            if c.commit:
                break
        assert ticks_to_commit == pytest.approx(round(100.0 / 2.75), abs=1), (
            "the commit must land on the clock wrap: {} ticks at 2.75 Hz and 100 Hz control"
            .format(ticks_to_commit))
        # up to (and including) the commit tick the OLD spec was in force
        assert ctrl._spec[G.I_S_CAM.start + 1] == pytest.approx(-1.0)   # latched on that tick

    def test_the_residual_reaches_the_target_on_every_tick_latched_or_not(self):
        ctrl = PolicyControllerV2(v2_bundle())
        grav, gyro = _upright()
        ctrl.start(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        a = np.zeros(ACTION_DIM)
        c0 = ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro, override_action=a)
        a[SPEC_DIM:] = 1.0
        moved = []
        for _ in range(5):
            c = ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro, override_action=a)
            moved.append(float(np.max(c.target_prefilter - c0.target_prefilter)))
        assert min(moved) > 0.0, "the residual is not latched; it must move the target every tick"
        assert max(moved) == pytest.approx(ctrl.gp.residual_scale, abs=1e-6)

    def test_the_previous_residual_appears_in_the_next_frame(self):
        ctrl = PolicyControllerV2(v2_bundle())
        grav, gyro = _upright()
        ctrl.start(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        a = np.zeros(ACTION_DIM)
        a[SPEC_DIM:] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
        ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro, override_action=a)
        ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro, override_action=np.zeros(ACTION_DIM))
        assert np.allclose(ctrl._history[-1, 27:33], a[SPEC_DIM:], atol=1e-6)

    def test_the_newest_frames_phase_is_the_one_the_generator_will_use(self):
        # `CommandV2.phase` is the clock AFTER the tick's advance -- which is exactly the phase the
        # NEXT tick assembles at, and therefore the phase the next tick's frame must carry. Get
        # this offset wrong by one and the policy reads a clock a tick away from the gait it is
        # being played, which looks almost right and falls over.
        ctrl = PolicyControllerV2(v2_bundle())
        grav, gyro = _upright()
        ctrl.start(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        prev = None
        for _ in range(7):
            cmd = ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro,
                            override_action=np.zeros(ACTION_DIM))
            if prev is not None:
                assert (float(ctrl._history[-1, 25]), float(ctrl._history[-1, 26])) == \
                    pytest.approx((np.cos(prev), np.sin(prev)), abs=1e-6)
            prev = cmd.phase
        assert prev > 0.0                       # and the clock actually moved


class TestTheRunStopCommand:
    def test_a_run_starts_stopped(self):
        ctrl = PolicyControllerV2(v2_bundle())
        grav, gyro = _upright()
        obs = ctrl.start(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        assert ctrl.run is False
        assert obs[ACTOR_DIM - 3] == 0.0

    def test_the_button_moves_exactly_one_number_in_the_observation(self):
        ctrl = PolicyControllerV2(v2_bundle())
        grav, gyro = _upright()
        stopped = ctrl.start(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro).copy()
        assert ctrl.set_run(True) is True
        running = ctrl._obs()
        diff = np.flatnonzero(np.abs(running - stopped) > 0)
        assert diff.tolist() == [ACTOR_DIM - 3], (
            "RUN/STOP must move task[0] and nothing else; it moved {}".format(diff.tolist()))
        assert running[ACTOR_DIM - 3] == 1.0

    def test_a_speed_objective_checkpoint_reports_that_the_flag_is_unreachable(self):
        ctrl = PolicyControllerV2(v2_bundle(objective="speed"))
        grav, gyro = _upright()
        obs = ctrl.start(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        assert ctrl.set_run(False) is False           # the caller must be told
        assert obs[ACTOR_DIM - 3] == 1.0              # the channel is the constant [1, 1]
        assert ctrl._obs()[ACTOR_DIM - 3] == 1.0

    def test_the_distance_countdown_is_the_second_task_entry(self):
        ctrl = PolicyControllerV2(v2_bundle())
        grav, gyro = _upright()
        ctrl.start(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        ctrl.set_distance_to_go(0.25)
        assert ctrl._obs()[ACTOR_DIM - 2] == pytest.approx(0.25)
        ctrl.set_distance_to_go(-3.0)
        assert ctrl._obs()[ACTOR_DIM - 2] == 0.0      # clipped, never off-manifold


class TestTheCommandIsSafeToSend:
    def test_targets_never_leave_the_joint_band_however_wild_the_action(self):
        ctrl = PolicyControllerV2(v2_bundle())
        grav, gyro = _upright()
        ctrl.start(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        rng = np.random.default_rng(1)
        for _ in range(300):
            a = rng.uniform(-3.0, 3.0, ACTION_DIM)         # deliberately outside [-1, 1]
            c = ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro, override_action=a)
            assert np.all(c.target >= ctrl.q_lo - 1e-9) and np.all(c.target <= ctrl.q_hi + 1e-9)
            assert np.all(np.isfinite(c.target)) and np.all(np.isfinite(c.kp))

    def test_the_commanded_target_never_moves_faster_than_the_motor_can(self):
        ctrl = PolicyControllerV2(v2_bundle())
        grav, gyro = _upright()
        ctrl.start(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        rng = np.random.default_rng(2)
        prev = ctrl.nominal.copy()
        for _ in range(200):
            a = rng.uniform(-1.0, 1.0, ACTION_DIM)
            c = ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro, override_action=a)
            step = np.abs(c.target - prev) / ctrl.control_dt
            assert np.all(step <= VEL_LIMIT + 1e-6), step
            prev = c.target.copy()

    def test_the_pitch_reflex_pushes_the_thighs_in_opposite_directions(self):
        # (+,-) on the thighs is the mirror-SYMMETRIC pattern: both feet move the same fore-aft
        # way. A sign slip here drives the balance correction backwards at 200 N*m/rad.
        ctrl = PolicyControllerV2(v2_bundle())
        grav, gyro = _upright()
        ctrl.start(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        flat = ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro,
                         override_action=np.zeros(ACTION_DIM)).target_prefilter
        tipped = np.array([0.2, 0.0, -0.98])               # nose down
        ctrl2 = PolicyControllerV2(v2_bundle())
        ctrl2.start(ctrl2.nominal, np.zeros(6), np.zeros(6), tipped, gyro)
        lean = ctrl2.step(ctrl2.nominal, np.zeros(6), np.zeros(6), tipped, gyro,
                          override_action=np.zeros(ACTION_DIM)).target_prefilter
        d = lean - flat
        assert d[2] != 0.0
        assert d[2] == pytest.approx(-d[5], abs=1e-12)
        assert np.allclose(d[[0, 1, 3, 4]], 0.0)


# ===================================================================== against the trained loop
class TestAgainstTheTrainedLoop:
    """Drive the deployed runtime through the MJX trace's own recorded states.

    This is the whole point of the file. `walk_v2/tools/trace.py` played a FIXED spec plus a known
    sinusoidal residual through the real training environment and wrote down, per tick, the state,
    the command it produced and the observation frame it published. Feeding those states to
    `PolicyControllerV2` with the same actions must reproduce both -- and it exercises exactly the
    things a port gets wrong: which measurement the reflexes read, which phase the generator
    assembles at, where the previous residual goes, and what order the clip and the slew limit
    come in."""

    def _replay(self):
        tr = _trace()
        gpd = dict(tr["gait_params"])
        b = v2_bundle(gait_params=gpd, nominal=tr["nominal_ctrl"],
                      default_motor_pos=tr["default_motor_pos"],
                      pitch_lp=0.0,          # this fixture predates the EMA; it used the raw rate
                      control_dt=tr["control_dt"])
        ctrl = PolicyControllerV2(b)
        spec = np.array(tr["fixed_spec"])
        dt = tr["control_dt"]
        rows = [r for r in tr["rows"] if not r.get("done")]
        # the reset state: the keyframe, no joint noise, zero velocity, upright
        ctrl.start(np.array(tr["default_motor_pos"]), np.zeros(6), np.zeros(6),
                   np.array([0.0, 0.0, -1.0]), np.zeros(3))
        prev = None
        out = []
        for row in rows:
            act = np.concatenate([spec, 0.05 * np.sin(2 * np.pi * 3.0 * (row["t"] - dt)
                                                      + np.arange(6))])
            m = prev or {"qpos": tr["default_motor_pos"], "qvel": [0.0] * 6, "tau": [0.0] * 6,
                         "grav": [0.0, 0.0, -1.0], "gyro": [0.0] * 3}
            c = ctrl.step(np.array(m["qpos"]), np.array(m["qvel"]), np.array(m["tau"]),
                          np.array(m["grav"]), np.array(m["gyro"]), override_action=act)
            out.append((row, c, ctrl._history[-1].copy()))
            prev = row
        return tr, out

    def test_the_commanded_targets_and_gains_are_the_trainers(self):
        _, out = self._replay()
        d_t = max(np.abs(np.array(r["target"]) - c.target).max() for r, c, _ in out)
        d_p = max(np.abs(np.array(r["kp"]) - c.kp).max() for r, c, _ in out)
        d_d = max(np.abs(np.array(r["kd"]) - c.kd).max() for r, c, _ in out)
        assert max(d_t, d_p / 100.0, d_d) < 1e-4, (
            "the deployed runtime commands something else: target {:.2e} rad, kp {:.2e}, "
            "kd {:.2e}".format(d_t, d_p, d_d))

    def test_the_gait_clock_tracks_the_trainers_tick_for_tick(self):
        _, out = self._replay()
        d = max(abs(r["phase"] - c.phase) for r, c, _ in out)
        assert d < 1e-5, "the clock drifted {:.2e} rad from the trainer's".format(d)

    def test_the_observation_frame_is_the_one_the_policy_was_trained_on(self):
        # the frame the controller pushes at tick n+1 is the frame the trainer published at the
        # end of tick n: same measurement, same phase, same previous residual
        _, out = self._replay()
        worst, where = 0.0, -1
        for i in range(len(out) - 1):
            want = np.array(out[i][0]["frame"])
            got = out[i + 1][2]
            d = float(np.abs(want - got).max())
            if d > worst:
                worst, where = d, i
        assert worst < 1e-4, (
            "observation frame diverges by {:.2e} at tick {} -- the policy would be reading an "
            "observation off its training manifold".format(worst, where))


# ===================================================================== the fitted brake
class TestTheBrakeIsNotTheFlag:
    """STOP on this lineage is a fitted open-loop schedule with the task flag HELD AT 1.

    The cluster measured the alternative (walk_v2/README.md, 2026-09-11 17:00): 512/512 upright
    when the policy is not told it has finished, 3/512 when it is. Ten training configurations
    across two months failed to teach a brake because the command channel IS the disturbance. So
    the invariants below are not stylistic -- every one of them is the difference between a robot
    that stops and a robot that accelerates into the floor."""

    THETA = [1.0003686178569042, 1.4107132966160563, 1.1282642389078434,      # freq_scale knots
             1.017690240948479, 0.9599241223468952, 0.7816998488700864,       # amp_scale
             -0.09893795516675613, 0.0010773722707862338, 0.4078593514852848,  # o_cam
             0.1282132819050873, -0.3751108557299587, 0.08925503370987752]     # o_thigh

    def _braking(self, window_s=12.0):
        b = v2_bundle(brake={"theta": self.THETA, "window_s": window_s, "source": "test"})
        ctrl = PolicyControllerV2(b)
        grav, gyro = _upright()
        ctrl.start(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        ctrl.set_run(True)
        for _ in range(40):                       # let a spec latch and the clock settle
            ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        return ctrl, grav, gyro

    def test_the_brake_holds_the_task_flag_at_one(self):
        ctrl, grav, gyro = self._braking()
        assert ctrl.start_brake() == (True, "")
        for _ in range(50):
            c = ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
            assert c.run_flag == 1.0, (
                "the brake dropped the run flag -- that is the 3/512 case, not the 512/512 one")
        assert ctrl.run is True
        assert ctrl._obs()[ACTOR_DIM - 3] == 1.0

    def test_a_bundle_without_a_schedule_refuses_instead_of_dropping_the_flag(self):
        ctrl = PolicyControllerV2(v2_bundle())
        grav, gyro = _upright()
        ctrl.start(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        ok, why = ctrl.start_brake()
        assert not ok and "3/512" in why
        assert ctrl.braking is False
        assert ctrl.run is False                  # and it did NOT quietly fall back to the flag

    def test_the_schedule_drives_exactly_the_four_channels_the_search_fitted(self):
        # freq (spec units, multiplied), cam+thigh series (multiplied), o_cam and o_thigh (SET).
        # Everything else is the frozen cruise spec: the search overrode nothing else, so neither
        # may this.
        ctrl, grav, gyro = self._braking()
        cruise = ctrl._spec.copy()
        ctrl.start_brake()
        c = ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        touched = np.flatnonzero(np.abs(c.spec - cruise) > 1e-9)
        allowed = set([G.I_FREQ, G.I_O.start, G.I_O.start + 1])
        allowed |= set(range(G.I_S_CAM.start, G.I_S_CAM.stop))
        allowed |= set(range(G.I_S_THIGH.start, G.I_S_THIGH.stop))
        assert set(touched.tolist()) <= allowed, (
            "the brake moved spec dims the search never fitted: "
            f"{sorted(set(touched.tolist()) - allowed)}")
        th = np.asarray(self.THETA).reshape(4, 3)
        assert c.spec[G.I_FREQ] == pytest.approx(np.clip(cruise[G.I_FREQ] * th[0, 0], -1, 1), abs=1e-6)
        assert c.spec[G.I_O.start] == pytest.approx(th[2, 0], abs=1e-6)
        assert c.spec[G.I_O.start + 1] == pytest.approx(th[3, 0], abs=1e-6)

    def test_the_policys_residual_still_reaches_the_target_while_braking(self):
        # 45-95% of joint motion on this lineage is the residual; the search kept it deliberately,
        # because zeroing it removes the stabiliser and everything falls whatever the schedule says.
        # Two controllers in lockstep, so the only difference at any tick is the residual -- the
        # schedule itself is moving the feedforward underneath.
        zero, one = np.zeros(ACTION_DIM), np.zeros(ACTION_DIM)
        one[SPEC_DIM:] = 1.0
        ctrls = []
        for act in (zero, one):
            ctrl, grav, gyro = self._braking()
            ctrl.start_brake()
            ctrls.append((ctrl, act))
        for _ in range(30):
            outs = [c.step(c.nominal, np.zeros(6), np.zeros(6), grav, gyro, override_action=a)
                    for c, a in ctrls]
            assert np.array_equal(outs[0].spec, outs[1].spec), "the schedules diverged"
            d = outs[1].target_prefilter - outs[0].target_prefilter
            assert np.max(d) == pytest.approx(ctrls[0][0].gp.residual_scale, abs=1e-6)

    def test_the_window_is_walked_once_and_then_held(self):
        ctrl, grav, gyro = self._braking(window_s=1.0)     # 100 ticks
        assert ctrl.brake_ticks == 100
        ctrl.start_brake()
        fracs = [ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro).brake_frac
                 for _ in range(160)]
        assert fracs[0] == 0.0
        assert fracs[99] == pytest.approx(0.99, abs=0.02)
        assert all(f == 1.0 for f in fracs[105:]), "the window must saturate, not wrap or reset"
        # ... and the held spec is the final knot, so the robot stays where the schedule left it
        th = np.asarray(self.THETA).reshape(4, 3)
        c = ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        assert c.spec[G.I_O.start] == pytest.approx(th[2, -1], abs=1e-6)

    def test_run_hands_the_gait_back_to_the_policy(self):
        ctrl, grav, gyro = self._braking()
        ctrl.start_brake()
        for _ in range(20):
            ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        assert ctrl.braking is True
        assert ctrl.cancel_brake() is True
        c = ctrl.step(ctrl.nominal, np.zeros(6), np.zeros(6), grav, gyro)
        assert c.brake_frac == 0.0 and ctrl.braking is False
        # the latch kept running underneath, so what comes back is a live spec, not a stale one
        assert np.array_equal(c.spec, ctrl._spec)

    def test_a_schedule_without_its_window_is_refused(self):
        # replaying a 12 s fit over 16 s stopped 0/512: the window is part of the schedule, so a
        # bundle that cannot say what it was fitted over must not load as a brake
        with pytest.raises(ValueError, match="window"):
            PolicyControllerV2(v2_bundle(brake={"theta": self.THETA, "window_s": 0.0}))
        with pytest.raises(ValueError, match="12-number"):
            PolicyControllerV2(v2_bundle(brake={"theta": self.THETA[:6], "window_s": 8.0}))


# ===================================================================== the joystick
class TestTheJoystick:
    """objective='joystick': task[0] IS the commanded speed, and there is no flag anywhere.

    The retrain exists because of the pair of numbers in `TestTheBrakeIsNotTheFlag` -- a binary
    command the policy meets once per episode is a step disturbance, and this lineage falls over
    it. A speed the policy has been trained at every value of is the same channel without the
    cliff, so these tests are mostly about the channel staying a channel: one number in the
    observation, clamped to what was trained, and never delivered as a step."""

    GRAV = np.array([0.0, 0.0, -1.0])

    def _ctrl(self, **kw):
        kw.setdefault("objective", "joystick")
        c = PolicyControllerV2(v2_bundle(**kw))
        c.start(c.nominal, np.zeros(6), np.zeros(6), self.GRAV, np.zeros(3))
        return c

    def _tick(self, c):
        return c.step(c.nominal, np.zeros(6), np.zeros(6), self.GRAV, np.zeros(3))

    def test_a_run_comes_up_asking_for_zero(self):
        """Not stopped -- asking for 0 m/s, which is walking in place, which is a speed this
        lineage was trained at. Same promise as the run/stop lineage's 'always comes up STOPPED',
        made in the units the policy actually reads."""
        c = self._ctrl()
        assert c.command_kind == "speed"
        assert c.speed_target == 0.0 and c.speed_cmd == 0.0
        assert c._task() == (0.0, 1.0)      # task[1] reserved at ONE -- see the test below
        assert c._obs()[c.actor_dim - 3] == 0.0

    def test_the_slider_moves_exactly_one_number_in_the_observation(self):
        c = self._ctrl()
        before = c._obs().copy()
        c.set_speed(1.5, immediate=True)
        after = c._obs()
        moved = np.flatnonzero(before != after)
        assert moved.tolist() == [c.spec_dim + (c.actor_dim - c.once_dim)]
        assert after[c.actor_dim - 3] == pytest.approx(1.5 / c.v_max)

    def test_the_second_task_entry_is_reserved_and_pinned_at_one(self):
        """v2's task[1] was clip((line - d)/8, 0, 1), computed in the sim from ground-truth world
        x. The robot cannot produce it, so under the joystick it carries no odometry -- but it is
        pinned at ONE, not zero, and the difference is the whole behaviour.

        Under v2 semantics this channel is the distance-to-go ramp: 1.0 means "the line is far
        away", 0 means "brake now". A bundle shipping 0 holds the policy in a permanent stop
        request. Measured 2026-09-11: a 3.27 m/s runner made 0.1 m/s under a 3.2 m/s command,
        earned 0.05 of a possible 9.0 of tracking income, and fell in 100% of episodes -- while the
        same checkpoint ran 600/600 upright under the objective it was trained on. The trainer pins
        it at 1.0 and this must match bit for bit."""
        c = self._ctrl()
        for v in (0.0, 0.7, 3.0):
            c.set_speed(v, immediate=True)
            self._tick(c)
            assert c._task()[1] == 1.0
            assert c._obs()[c.actor_dim - 2] == 1.0

    def test_the_command_is_clamped_to_the_range_that_was_trained(self):
        """The sim clips task[0] at the rails, so asking for more does not ask for more -- it asks
        for the rail while the panel says something else."""
        c = self._ctrl()
        assert c.set_speed(99.0) == (True, c.v_max)
        assert c.set_speed(-5.0) == (True, c.v_min)

    def test_a_backward_range_reaches_the_negative_rail(self):
        """v_min is 0 while the trainer clips there. When walking backwards is trained it arrives
        as a negative v_min, and the same normalisation has to carry it."""
        c = self._ctrl(v_min=-1.0)
        c.set_speed(-1.0, immediate=True)
        assert c._task()[0] == pytest.approx(-1.0 / c.v_max)
        assert c.set_speed(-2.0) == (True, -1.0)

    def test_a_dragged_slider_arrives_as_a_ramp_and_not_a_step(self):
        """A browser can move the slider from 0 to top speed in one event. The one thing measured
        to drop this robot is a step change on the command channel, so the control law slews it."""
        c = self._ctrl()
        c.set_speed(c.v_max)
        assert c.speed_cmd == 0.0, "the target moved; the applied command has not, yet"
        prev, seen = 0.0, []
        for _ in range(int(round(4.0 / c.control_dt))):
            cmd = self._tick(c)
            assert cmd.speed_cmd - prev <= c.cmd_slew_mps2 * c.control_dt + 1e-9
            prev = cmd.speed_cmd
            seen.append(cmd.speed_cmd)
        assert seen[-1] == pytest.approx(c.v_max)
        reached = next(i for i, v in enumerate(seen) if v >= c.v_max - 1e-9) * c.control_dt
        assert reached == pytest.approx(c.v_max / c.cmd_slew_mps2, abs=2 * c.control_dt)

    def test_the_slew_rate_is_the_trainers_when_the_bundle_records_one(self):
        c = self._ctrl(v_cmd_rate=0.25)
        assert c.cmd_slew_mps2 == 0.25
        c.set_speed(c.v_max)
        self._tick(c)
        assert c.speed_cmd == pytest.approx(0.25 * c.control_dt)

    def test_asking_for_zero_is_the_stop_and_the_flag_is_not_reachable(self):
        """There is no green light in this observation, so `set_run` must refuse rather than move
        a number that means something else -- and the fitted brake, which exists only because the
        run/stop lineage could not be asked to slow down, has nothing to do here either."""
        c = self._ctrl(brake={"theta": [1.0] * 12, "window_s": 8.0})
        assert c.set_run(False) is False
        assert c.set_run(True) is False
        ok, why = c.start_brake()
        assert ok is False and "0 m/s" in why
        c.set_speed(2.0, immediate=True)
        self._tick(c)
        c.set_speed(0.0, immediate=True)
        assert self._tick(c).run_flag == 0.0

    def test_the_trained_band_comes_from_the_checkpoint_and_not_the_config(self):
        """The command is drawn from a fraction band a curriculum widens DOWNWARD from the
        warm-start parent's one speed, so a mid-ramp checkpoint has never been asked to go slowly.
        Its slider still runs to 0 — but 0 is then off-distribution, and only the checkpoint's own
        sidecar knows. Same trap as freq_lo, which was worth 0.4 rad of thigh target."""
        c = self._ctrl()
        assert c.v_trained == (c.v_min, c.v_max)            # no sidecar: the whole channel
        mid = self._ctrl(env_params_at_checkpoint={"cmd_lo": 0.8, "cmd_hi": 1.0})
        assert mid.v_trained == pytest.approx((0.8 * mid.v_max, mid.v_max))
        assert mid.set_speed(0.0) == (True, 0.0), "the slider still reaches 0; the panel warns"

    def test_a_joystick_bundle_with_no_scale_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="v_max"):
            PolicyControllerV2(v2_bundle(objective="joystick", v_max=0.0))

    def test_the_run_stop_lineage_is_untouched_by_any_of_this(self):
        c = PolicyControllerV2(v2_bundle())
        c.start(c.nominal, np.zeros(6), np.zeros(6), self.GRAV, np.zeros(3))
        assert c.command_kind == "run_stop"
        assert c.set_speed(1.0) == (False, 0.0)
        assert c.set_run(True) is True and c._task() == (1.0, 1.0)


# ===================================================================== the deployed hot path
class TestTheFastGeneratorIsTheReference:
    """`gait_v2.GaitEval` is what the robot runs; `gait_v2.assemble` is the line-by-line mirror of
    `walk_v2/gait.py` it has to agree with. EXACTLY -- not to a tolerance.

    A tolerance would be the wrong test here. The point of the substitution is that it changes how
    many numpy calls the same arithmetic takes (5.2 ms -> 1.1 ms of an 8.2 ms tick on the Pi 3B,
    which is the difference between this policy fitting its 100 Hz budget and not), so any
    numerical difference at all means the rewrite changed the control law rather than its cost.
    Two silent float32 promotions were found this way and would not have failed a 1e-6 test:

      * `delta` (and with it the RIGHT leg's whole phase) is float32, because the spec is float32
        and numpy's NEP 50 rules let it decide. Recomputing that phase in float64 moved the
        right-leg gains by 5e-6 relative.
      * the series' constant term a0 is float64 only because `WEIGHTS[0]` is a numpy scalar, which
        outranks a float32 array. Unwrapping it to a Python float -- which is weak -- silently
        computed a0 in float32, worth 5e-5 on the target.

    Both are faithful to the trainer, which is float32 throughout, so the fast path reproduces the
    promotions rather than tidying them away."""

    N = 2000

    def _cases(self, dtype, seed=0):
        rng = np.random.default_rng(seed)
        for _ in range(self.N):
            # PAST the rails on purpose: every clip in the generator has to be exercised, and the
            # clipped branch is where a Python-float rewrite silently changes dtype
            yield (rng.uniform(-1.6, 1.6, SPEC_DIM).astype(dtype),
                   rng.uniform(-1.6, 1.6, N_RESIDUAL).astype(dtype),
                   float(rng.uniform(-8.0, 8.0)),
                   tuple(float(x) for x in rng.normal(size=4)))

    def _both(self, dtype):
        b = v2_bundle()
        p, nominal = b.gait_params(), np.asarray(b["nominal_ctrl"], float)
        ev = G.GaitEval(p, nominal)
        worst = 0.0
        for spec, resid, phi, (roll, rr, pitch, pr) in self._cases(dtype):
            ref = G.assemble(spec, resid, phi, roll, rr, pitch, pr, nominal, p)
            fast = ev(spec, resid, phi, roll, rr, pitch, pr)
            for r, f in zip(ref, fast):
                worst = max(worst, float(np.max(np.abs(np.asarray(r) - np.asarray(f)))))
        return worst

    def test_a_float32_spec_gives_bit_identical_targets_and_gains(self):
        assert self._both(np.float32) == 0.0

    def test_a_float64_spec_gives_bit_identical_targets_and_gains(self):
        assert self._both(np.float64) == 0.0

    def test_the_right_legs_phase_is_still_rounded_the_way_the_trainer_rounds_it(self):
        """The regression that hides behind a tolerance: `delta` carries the spec's float32, so
        the right leg reads a float32 phase. Recomputing it in float64 is 'more accurate' and
        wrong -- it is not the law the policy was trained against."""
        b = v2_bundle()
        p, nominal = b.gait_params(), np.asarray(b["nominal_ctrl"], float)
        spec = np.zeros(SPEC_DIM, np.float32)
        spec[G.I_DELTA] = 0.3               # a delta whose float32 and float64 products differ
        spec[G.I_KP] = 0.7
        ev = G.GaitEval(p, nominal)
        _t, kp, _kd, _q = ev(spec, np.zeros(N_RESIDUAL, np.float32), 1.0, 0.0, 0.0, 0.0, 0.0)
        _t2, kp_ref, _kd2, _q2 = G.assemble(spec, np.zeros(N_RESIDUAL, np.float32), 1.0,
                                            0.0, 0.0, 0.0, 0.0, nominal, p)
        assert np.array_equal(kp, kp_ref)
        # and the right leg genuinely reads a different phase from the left, so this is not vacuous
        assert kp[3] != kp[0]

    def test_the_controller_runs_the_fast_path(self):
        """A test that passes because the two agree is worthless if the runtime calls neither."""
        b = v2_bundle()
        c = PolicyControllerV2(b)
        assert isinstance(c.gait_eval, G.GaitEval)
        pos = np.asarray(b["nominal_ctrl"], float)
        z, grav = np.zeros(6), np.array([0.0, 0.0, -1.0])
        c.start(pos, z, z, grav, np.zeros(3))
        calls = []
        real = c.gait_eval

        class Counting:
            def __call__(self, *a):
                calls.append(1)
                return real(*a)

        c.gait_eval = Counting()
        c.step(pos, z, z, grav, np.zeros(3))
        assert calls == [1]
