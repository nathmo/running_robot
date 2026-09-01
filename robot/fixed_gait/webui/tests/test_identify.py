"""The joint-identification wiggle, and the panel bug that made the thermal Start button dead.

MockBus only — no hardware, no CAN, no robot.

    python -m pytest robot/fixed_gait/webui/tests/test_identify.py -v
"""
import time

import pytest

import calibration
import daemon as daemon_mod
import paths
from test_blackbox import capture_zero, robot, wait_mode      # noqa: F401  (pytest fixtures)


def arm(d, cal):
    """Zero + confirm, and vouch for continuity, so the pre-move guard lets a wiggle through."""
    capture_zero(d, cal)
    d._zero_epoch_at_start = -1


def run_wiggle(d, timeout=8.0, **spec):
    spec.setdefault("motor", "left.thigh")
    ok, why = d.identify_start(spec)
    assert ok, why
    t_end = time.time() + timeout
    while time.time() < t_end:
        time.sleep(0.05)
        res = d.get_identify()
        if res is not None:
            return res
    pytest.fail("wiggle never finished")


# ===================================================================== validation
@pytest.mark.parametrize("spec,fragment", [
    ({"motor": "nope"}, "unknown motor"),
    ({"motor": "left.thigh", "amp_deg": 99.0}, "amplitude must be"),
    ({"motor": "left.thigh", "amp_deg": 0.0}, "amplitude must be"),
    ({"motor": "left.thigh", "amp_deg": float("nan")}, "must be numbers"),
    ({"motor": "left.thigh", "amp_deg": "wide"}, "must be numbers"),
    ({"motor": "left.thigh", "duration_s": 60.0}, "duration must be"),
    ({"motor": "left.thigh", "duration_s": 0.1}, "duration must be"),
])
def test_bad_specs_are_refused(robot, spec, fragment):
    d, cal, _b, _dir = robot
    arm(d, cal)
    ok, why = d.identify_start(spec)
    assert not ok
    assert fragment in why


def test_refused_before_calibration(robot):
    """The wiggle is a motion endpoint and inherits the same gate as every other one."""
    d, _cal, _b, _dir = robot
    ok, why = d.identify_start({"motor": "left.thigh"})
    # either the calibration gate or the pre-move guard, but it must not move
    if ok:
        time.sleep(0.4)
        assert d.get_snapshot()["mode"] != "IDENTIFY", "wiggled without a calibration"


# ===================================================================== the run itself
def test_wiggle_moves_the_selected_joint_and_nothing_else(robot):
    d, cal, _b, _dir = robot
    arm(d, cal)
    res = run_wiggle(d, motor="left.thigh", amp_deg=5.0, duration_s=2.0)

    assert res["abort"] is None, res["abort"]
    assert res["verdict"] == "confirmed", res["detail"]
    assert res["moved"] == ["left.thigh"]
    assert res["excursions"]["left.thigh"] >= 2 * res["threshold_deg"]
    for n in paths.MOTOR_NAMES:
        if n != "left.thigh":
            assert res["excursions"][n] < res["threshold_deg"], n


def test_every_joint_can_be_identified(robot):
    """Including abduction, which nothing else in this repo has ever exercised."""
    d, cal, _b, _dir = robot
    arm(d, cal)
    for n in paths.MOTOR_NAMES:
        res = run_wiggle(d, motor=n, amp_deg=5.0, duration_s=1.0)
        assert res["verdict"] == "confirmed", "{}: {}".format(n, res["detail"])
        assert res["moved"] == [n]
        d.request_mode("LIMP")
        assert wait_mode(d, "LIMP")


def test_the_wiggle_returns_the_joint_to_where_it_started(robot):
    """Whole sine cycles plus a raised-cosine envelope: it must not leave the joint displaced."""
    d, cal, _b, _dir = robot
    arm(d, cal)
    before = d.by_name["left.cam"].pos
    run_wiggle(d, motor="left.cam", amp_deg=5.0, duration_s=2.0)
    time.sleep(0.3)
    assert abs(d.by_name["left.cam"].pos - before) < 1.5


def test_amplitude_is_shrunk_to_the_room_the_joint_has(robot, monkeypatch):
    d, cal, _b, _dir = robot
    arm(d, cal)
    monkeypatch.setattr(daemon_mod.RobotDaemon, "_safe_room",
                        lambda self, pose, name, direction, reach: min(2.5, reach))
    ok, why = d.identify_start({"motor": "left.thigh", "amp_deg": 5.0})
    assert ok, why
    assert d._wiggle_req["amp_deg"] == pytest.approx(2.5)


def test_no_room_is_refused_rather_than_wiggled_invisibly(robot, monkeypatch):
    """A 0.4 deg twitch would answer the mapping question WRONGLY, not decline to answer it."""
    d, cal, _b, _dir = robot
    arm(d, cal)
    monkeypatch.setattr(daemon_mod.RobotDaemon, "_safe_room",
                        lambda self, pose, name, direction, reach: 0.4)
    ok, why = d.identify_start({"motor": "left.thigh", "amp_deg": 5.0})
    assert not ok
    assert "of room here" in why


def test_stop_aborts_the_run(robot):
    d, cal, _b, _dir = robot
    arm(d, cal)
    ok, why = d.identify_start({"motor": "left.thigh", "amp_deg": 5.0, "duration_s": 8.0})
    assert ok, why
    assert wait_mode(d, "IDENTIFY")
    time.sleep(0.6)
    d.identify_stop()
    t_end = time.time() + 3.0
    while time.time() < t_end and d.get_identify() is None:
        time.sleep(0.05)
    res = d.get_identify()
    assert res is not None and res["verdict"] == "aborted"
    assert "stopped by operator" in res["abort"]


def test_limp_clears_the_result(robot):
    d, cal, _b, _dir = robot
    arm(d, cal)
    run_wiggle(d, motor="left.thigh", duration_s=1.0)
    d.request_mode("LIMP")
    assert wait_mode(d, "LIMP")
    assert d._wiggle is None
    assert d.get_snapshot()["identify"] is None


# ===================================================================== mutual exclusion
def test_a_burst_cannot_start_on_top_of_a_wiggle(robot):
    """Otherwise the mode switches, _tick_identify stops being called, and the wiggle reads
    'running' forever."""
    d, cal, _b, _dir = robot
    arm(d, cal)
    ok, why = d.identify_start({"motor": "left.thigh", "duration_s": 6.0})
    assert ok, why
    assert wait_mode(d, "IDENTIFY")
    ok, why = d.thermal_start({"motor": "left.thigh", "amps": 20.0, "duration_s": 60.0,
                                "rotor_mode": "blocked"})
    assert not ok and "wiggle is still running" in why


def test_a_wiggle_cannot_start_on_top_of_a_burst(robot, short_bursts):
    d, cal, _b, _dir = robot
    arm(d, cal)
    ok, why = d.thermal_start({"motor": "left.thigh", "amps": 25.0, "duration_s": 45.0,
                                "rotor_mode": "blocked"})
    assert ok, why
    assert wait_mode(d, "THERMAL")
    ok, why = d.identify_start({"motor": "left.cam"})
    assert not ok and "burst is still running" in why
    d.thermal_stop()


# ===================================================================== regressions
def test_wiggle_state_does_not_shadow_thread_ident(robot):
    """RobotDaemon is a threading.Thread, and Thread._set_ident() assigns self._ident = get_ident().

    Naming the wiggle state `_ident` silently handed the daemon's own thread id to the publisher,
    which crashed the control loop on the first tick after a wiggle started (2026-08-28)."""
    d, cal, _b, _dir = robot
    arm(d, cal)
    ok, why = d.identify_start({"motor": "left.thigh", "duration_s": 1.0})
    assert ok, why
    assert wait_mode(d, "IDENTIFY")
    assert isinstance(d._ident, int), "Thread's own attribute was overwritten"
    assert isinstance(d._wiggle, dict)
    assert d.get_snapshot()["identify"]["motor"] == "left.thigh"


def test_daemon_survives_a_whole_wiggle_without_tripping(robot):
    d, cal, _b, _dir = robot
    arm(d, cal)
    run_wiggle(d, motor="right.thigh", amp_deg=5.0, duration_s=2.0)
    snap = d.get_snapshot()
    assert snap["daemon_alive"] and not snap["estop"]["latched"], snap["estop"]


# ===================================================================== verdict classification
def _fake(sel, exc, abort=None, amp=5.0):
    """_identify_result is pure: feed it the state a run would have left behind."""
    d = daemon_mod.RobotDaemon.__new__(daemon_mod.RobotDaemon)
    state = {"motor": sel, "amp_deg": amp, "duration_s": 2.0, "abort": abort, "track_err": 0.2,
             "raw_lo": {n: 0.0 for n in exc}, "raw_hi": dict(exc)}
    return daemon_mod.RobotDaemon._identify_result(d, state)


ZERO = {n: 0.0 for n in paths.MOTOR_NAMES}


def test_verdict_confirmed():
    r = _fake("left.thigh", dict(ZERO, **{"left.thigh": 9.0}))
    assert r["verdict"] == "confirmed"


def test_verdict_mismatch_names_the_joint_that_actually_moved():
    r = _fake("left.thigh", dict(ZERO, **{"right.cam": 9.0}))
    assert r["verdict"] == "mismatch"
    assert "right.cam" in r["detail"] and "map is wrong" in r["detail"]


def test_verdict_no_motion():
    r = _fake("left.abd", ZERO)
    assert r["verdict"] == "no-motion"


def test_verdict_coupled_when_a_second_joint_follows():
    r = _fake("left.thigh", dict(ZERO, **{"left.thigh": 9.0, "left.cam": 4.0}))
    assert r["verdict"] == "coupled"
    assert "left.cam" in r["detail"]


def test_abort_beats_every_other_verdict():
    r = _fake("left.thigh", dict(ZERO, **{"left.thigh": 9.0}), abort="telemetry lost")
    assert r["verdict"] == "aborted" and r["detail"] == "telemetry lost"


def test_excursion_is_raw_and_so_survives_a_wrong_calibration():
    """The verdict must not depend on the calibration -- checking the calibration is its job."""
    r = _fake("left.thigh", dict(ZERO, **{"left.thigh": 9.0}))
    assert r["excursions"]["left.thigh"] == pytest.approx(9.0)


# ===================================================================== the plan (read-only)
def test_plan_queues_nothing_and_moves_nothing(robot):
    d, cal, _b, _dir = robot
    arm(d, cal)
    ok, why, plan = d.identify_plan({"motor": "left.thigh"})
    assert ok, why
    assert plan["amp_deg"] == pytest.approx(5.0)
    assert d._wiggle_req is None and d._wiggle is None
    time.sleep(0.3)
    assert d.get_snapshot()["mode"] == "LIMP"


def test_a_pose_outside_the_workspace_falls_back_to_the_hard_limits(robot, monkeypatch):
    """The gait polygon can only advise from inside itself: _safe_room scans outward from the
    current pose and returns 0.0 in every direction once that pose is outside the recorded set.
    A robot hanging limp on a stand is routinely outside it, and taking the 0.0 literally refused
    every wiggle on the real robot (2026-08-28)."""
    d, cal, _b, _dir = robot
    arm(d, cal)
    monkeypatch.setattr(daemon_mod.RobotDaemon, "_validate_pose",
                        lambda self, targets, override: (False, "left leg outside the hull"))
    monkeypatch.setattr(daemon_mod.RobotDaemon, "_safe_room",
                        lambda self, pose, name, direction, reach: 0.0)
    ok, why, plan = d.identify_plan({"motor": "left.thigh", "amp_deg": 5.0})
    assert ok, why
    assert plan["amp_deg"] == pytest.approx(5.0)
    assert plan["bound"] == "hard limits"
    assert plan["pose_in_workspace"] is False
    assert "outside the recorded safe workspace" in plan["note"]


def test_the_hard_band_still_binds_when_the_workspace_is_skipped(robot, monkeypatch):
    """Falling back to the hard limits must mean the hard limits, not no limits."""
    d, cal, _b, _dir = robot
    arm(d, cal)
    monkeypatch.setattr(daemon_mod.RobotDaemon, "_validate_pose",
                        lambda self, targets, override: (False, "outside"))
    centre = cal.norm("left.thigh", d.by_name["left.thigh"].pos)
    monkeypatch.setattr(daemon_mod.RobotDaemon, "_hard_bounds",
                        lambda self, side, role: (centre - 2.0, centre + 2.0))
    ok, why, plan = d.identify_plan({"motor": "left.thigh", "amp_deg": 5.0})
    assert ok, why
    assert plan["amp_deg"] == pytest.approx(2.0)
    assert plan["bound"] == "hard limits"


def test_a_workspace_that_can_speak_still_shrinks_the_amplitude(robot, monkeypatch):
    d, cal, _b, _dir = robot
    arm(d, cal)
    monkeypatch.setattr(daemon_mod.RobotDaemon, "_validate_pose",
                        lambda self, targets, override: (True, ""))
    monkeypatch.setattr(daemon_mod.RobotDaemon, "_safe_room",
                        lambda self, pose, name, direction, reach: min(3.0, reach))
    ok, why, plan = d.identify_plan({"motor": "left.thigh", "amp_deg": 5.0})
    assert ok, why
    assert plan["amp_deg"] == pytest.approx(3.0)
    assert plan["bound"] == "safe workspace"


def test_start_uses_exactly_the_amplitude_the_plan_reported(robot):
    d, cal, _b, _dir = robot
    arm(d, cal)
    _ok, _why, plan = d.identify_plan({"motor": "right.cam", "amp_deg": 5.0, "duration_s": 1.0})
    res = run_wiggle(d, motor="right.cam", amp_deg=5.0, duration_s=1.0)
    assert res["amp_deg"] == pytest.approx(plan["amp_deg"])


def test_the_premove_guard_refuses_at_the_http_layer_not_silently(robot):
    """A guard refusal used to land AFTER the endpoint returned 200: the CAN thread declined and
    the operator saw a button that did nothing. It must refuse where the message is visible."""
    d, cal, _b, _dir = robot
    capture_zero(d, cal)
    d._zero_epoch_at_start = cal.zero_epoch          # "restored from disk, not re-zeroed here"
    for n, m in d.by_name.items():                   # drives re-randomise their origin on power-up
        m.pos = (m.pos or 0.0) + 40.0
    ok, why, plan = d.identify_plan({"motor": "left.thigh"})
    assert not ok
    assert "Re-zero before moving" in why
    assert plan is not None and plan["guard_ok"] is False
    ok, why = d.identify_start({"motor": "left.thigh"})
    assert not ok and "Re-zero" in why


def test_a_healthy_zero_passes_the_guard_in_the_plan(robot):
    d, cal, _b, _dir = robot
    arm(d, cal)
    ok, why, plan = d.identify_plan({"motor": "left.thigh"})
    assert ok, why
    assert plan["guard_ok"] is True and plan["guard_why"] == ""


# ===================================================================== the loop must survive
@pytest.fixture
def short_bursts(monkeypatch):
    """Let a 2 s burst through. The viability gate correctly refuses one -- it deposits 0.18 degC
    and measures nothing -- but these tests are about the crash path, not about the gate, and a
    viable burst is 60 s of test time."""
    monkeypatch.setattr(daemon_mod.thermal_excite, "check_burst",
                        lambda params, amps, duration: (True, "", {"energy_j": 0.0,
                                                                   "case_c": 0.0,
                                                                   "winding_c": 0.0}))
    # MockBus motors drift under their own physics, and a blocked-rotor burst correctly aborts on
    # any motion. Widen the slip tripwire so a burst can actually run against simulated hardware
    # that is not, in fact, clamped.
    monkeypatch.setattr(daemon_mod.thermal_excite, "BLOCKED_SLIP_DEG", 1e6)
    monkeypatch.setattr(daemon_mod.thermal_excite, "BLOCKED_SLIP_ERPM", 1e6)


def _burst_to_end(d, timeout=25.0, **spec):
    spec.setdefault("motor", "left.thigh")
    spec.setdefault("rotor_mode", "blocked")
    ok, why = d.thermal_start(spec)
    assert ok, why
    assert wait_mode(d, "THERMAL")
    t_end = time.time() + timeout
    while time.time() < t_end:
        time.sleep(0.1)
        if d.get_thermal() is not None:
            return d.get_thermal()
    pytest.fail("burst never finished")


def test_a_completed_burst_does_not_kill_the_control_loop(robot, short_bursts):
    """`_bb_event("thermal.done", abort=abort, **ex.summary())` passed `abort` twice -- summary()
    already carries it -- and TypeError'd at argument binding, inside the branch that handles an
    ABORT. On the robot it killed the daemon thread while shutting a runaway burst down; the UI
    then showed telemetry frozen at the instant of the crash (2026-08-28)."""
    d, cal, _b, _dir = robot
    arm(d, cal)
    res = _burst_to_end(d, amps=25.0, duration_s=2.0)
    assert res is not None
    assert d.is_alive(), "daemon thread died completing a burst"
    assert d.get_snapshot()["daemon_alive"] is True
    assert d.loop_error in (None, ""), d.loop_error
    d.request_mode("LIMP")
    assert wait_mode(d, "LIMP"), "mode stuck -- the loop is not running"


def test_an_exciter_abort_does_not_kill_the_control_loop(robot, short_bursts, monkeypatch):
    """The abort path is the one that must never fail: it is what stops a burst going wrong, and
    it is the path that actually crashed on the robot -- the joint ran out of its window, the
    exciter correctly returned (0 A, done, reason), and the logging call in that branch killed the
    loop. An operator `stop` does NOT reach this branch (it returns early on running=False), so it
    has to be the exciter's own verdict that is exercised here."""
    d, cal, _b, _dir = robot
    arm(d, cal)
    real_step = daemon_mod.thermal_excite.BurstExciter.step

    def abort_soon(self, t, *a, **k):
        if t > 0.4:
            self.abort = "synthetic: the clamp is slipping"
            return 0.0, True, self.abort
        return real_step(self, t, *a, **k)

    monkeypatch.setattr(daemon_mod.thermal_excite.BurstExciter, "step", abort_soon)
    ok, why = d.thermal_start({"motor": "left.thigh", "amps": 25.0, "duration_s": 30.0,
                                "rotor_mode": "blocked"})
    assert ok, why
    t_end = time.time() + 10.0
    while time.time() < t_end and d.get_thermal() is None:
        time.sleep(0.1)
    res = d.get_thermal()
    assert res is not None, "burst never ended -- the loop is probably dead"
    # any exciter-generated abort exercises the branch; what matters is that it came from the
    # exciter's own verdict and NOT from thermal_stop(), which returns early and never gets here
    assert res["abort"], "expected an exciter abort"
    assert d.is_alive(), "daemon thread died aborting a burst"
    assert d.get_snapshot()["daemon_alive"] is True
    assert d.loop_error in (None, ""), d.loop_error
    d.request_mode("LIMP")
    assert wait_mode(d, "LIMP"), "mode stuck -- the loop is not running"


def test_the_recorder_actually_receives_the_burst_event(robot, short_bursts):
    """bb=None would hide a bad call site only if the failure were inside _bb_event. It is not --
    it is at argument binding -- but run it with a REAL recorder attached anyway."""
    import blackbox
    d, cal, b, _dir = robot
    arm(d, cal)
    _burst_to_end(d, amps=25.0, duration_s=2.0)
    time.sleep(0.6)
    kinds = [e.get("kind") for e in blackbox.read_events(
        __import__("os").path.join(_dir, blackbox.EVENTS_NAME))]
    assert "thermal.done" in kinds, kinds[-8:]


# ===================================================================== rotor mode
def test_a_burst_needs_the_rotor_declared(robot, short_bursts):
    d, cal, _b, _dir = robot
    arm(d, cal)
    ok, why = d.thermal_start({"motor": "left.thigh", "amps": 25.0, "duration_s": 10.0})
    assert not ok and "rotor declared" in why
    ok, why = d.thermal_start({"motor": "left.thigh", "amps": 25.0, "duration_s": 10.0,
                               "rotor_mode": "spinning"})
    assert not ok and "rotor declared" in why


@pytest.mark.parametrize("mode", ["blocked", "free"])
def test_both_declared_modes_are_accepted(robot, short_bursts, mode):
    d, cal, _b, _dir = robot
    arm(d, cal)
    ok, why = d.thermal_start({"motor": "left.thigh", "amps": 25.0, "duration_s": 10.0,
                               "rotor_mode": mode})
    assert ok, why
    assert wait_mode(d, "THERMAL")
    assert d._therm["env"].free_rotor is (mode == "free")
    d.thermal_stop()


# ===================================================================== free-rotor sine
@pytest.mark.parametrize("spec,fragment", [
    ({"freq_hz": 100.0}, "frequency"),
    ({"freq_hz": 0.01}, "frequency"),
    ({"amp_deg": 500.0}, "amplitude"),
    ({"amp_deg": 0.0}, "amplitude"),
    ({"freq_hz": float("nan")}, "must be numbers"),
    ({"amp_deg": "wide"}, "must be numbers"),
])
def test_a_free_burst_validates_its_sine(robot, short_bursts, spec, fragment):
    d, cal, _b, _dir = robot
    arm(d, cal)
    s = {"motor": "left.thigh", "amps": 25.0, "duration_s": 10.0, "rotor_mode": "free"}
    s.update(spec)
    ok, why = d.thermal_start(s)
    assert not ok and fragment in why


def test_a_free_burst_streams_position_never_current(robot, short_bursts):
    """Free mode is the sine tracker (2026-09-01): the daemon must command POSITION on the test
    motor -- pos_cmd populated, amps_cmd absent -- because a current-mode law on an unloaded
    shaft is exactly what the 2026-08-29 retraction is about. The deposited-energy integral then
    comes from the MEASURED current, which the exciter owns."""
    d, cal, _b, _dir = robot
    arm(d, cal)
    ok, why = d.thermal_start({"motor": "left.thigh", "amps": 25.0, "duration_s": 4.0,
                               "rotor_mode": "free", "freq_hz": 2.0, "amp_deg": 10.0})
    assert ok, why
    assert wait_mode(d, "THERMAL")
    assert d._therm["env"].freq_hz == 2.0 and d._therm["env"].sine_amp == 10.0
    time.sleep(1.5)                       # hold-before-move (0.35 s) + a few sine ticks
    buf = d._therm["buf"]
    assert any(v is not None for v in buf["pos_cmd"]), "no position commands logged"
    assert all(v is None for v in buf["amps_cmd"]), "a free run must never command current"
    d.thermal_stop()
