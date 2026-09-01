"""POLICY mode: running a trained bundle from the web UI, on the drives.

MockBus only -- no hardware, no CAN, no robot.

WHY THIS EXISTS
---------------
Until 2026-09-01 a real policy run meant stopping the web UI and running robot/deploy/run_policy.py
over ssh, because the CAN bus has exactly one owner. Moving it into the daemon means the 200 Hz
loop that owns the buses now also runs a learned control law, and three specific things had to be
true for that to be safe rather than merely convenient:

  * the daemon must not die doing it. Two of the checks below are regressions for bugs found while
    building it, and both killed the CAN thread: a lock re-entry (`_publish` holds self.lock and
    called into a helper that took it again -- threading.Lock is not reentrant) and, historically,
    a duplicate kwarg at an event call site (see the thermal.done comment in daemon.py).
  * every refusal must be a refusal, not a silent no-op. A policy that quietly does not start
    looks exactly like a policy that is not working.
  * letting go must stop the robot. The dead-man is the one guard whose absence is invisible.

    python -m pytest robot/fixed_gait/webui/tests/test_policy_run.py -v
"""
import json
import os
import time
import types

import numpy as np
import pytest

import paths
import daemon as daemon_mod
import server
from test_blackbox import capture_zero, robot, wait_mode          # noqa: F401 (pytest fixtures)

BUNDLE_DIRS = (paths.POLICY_DIR, os.path.join(paths.DEPLOY, "bundles"))


def _find_bundle():
    """Any exported bundle, wherever it lives. Bundles are ~5 MB of trained weights and are not in
    git, so the tests that need one skip rather than fail when the robot has not been fed yet."""
    for d in BUNDLE_DIRS:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".npz"):
                return os.path.join(d, f)
    return None


BUNDLE_SRC = _find_bundle()
needs_bundle = pytest.mark.skipif(BUNDLE_SRC is None,
                                  reason="no exported policy bundle (run export_policy.py)")


# ===================================================================== harness
class FakeSense:
    """Just enough Sense HAT for the daemon: a fast() tuple and a calibrated mount.

    Gravity is world DOWN in body axes, so upright is [0, 0, -1] and the HAT publishes its
    negation. Tilting is one attribute away, which is how the fall kill gets tested."""

    def __init__(self):
        self.up = np.array([0.0, 0.0, 1.0])
        self.gyro = np.zeros(3)
        self.mount = types.SimpleNamespace(calibrated=True)
        self.stale_by = 0.0

    def fast(self):
        return (time.time() - self.stale_by, self.up.copy(), self.gyro.copy())


@pytest.fixture
def armed(robot):                                                  # noqa: F811
    """A zeroed daemon on MockBus with an IMU and a bundle in data/policies/."""
    d, cal, _b, _dir = robot
    capture_zero(d, cal)
    d._zero_epoch_at_start = -1
    d.sense = FakeSense()
    yield d, cal


def spec(**kw):
    """A run request that would be accepted, with the acknowledgements a bench robot needs."""
    s = {"file": os.path.basename(BUNDLE_SRC or "none.npz"), "supported": True,
         "v_cmd": 0.0, "max_seconds": 2.0,
         "allow_uncalibrated_thermal": True, "skip_jointmap_check": True}
    s.update(kw)
    return s


@pytest.fixture
def staged(armed):
    """The bundle, staged into data/policies/ and removed again afterwards."""
    d, cal = armed
    dest = os.path.join(paths.POLICY_DIR, os.path.basename(BUNDLE_SRC))
    made = not os.path.exists(dest)
    if made:
        with open(BUNDLE_SRC, "rb") as src, open(dest, "wb") as out:
            out.write(src.read())
    yield d, cal, os.path.basename(dest)
    if made:
        os.remove(dest)


def wait_phase(d, want, timeout=20.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        p = d.get_snapshot().get("policy")
        if p and p["phase"] == want:
            return p
        assert d.is_alive(), "the CAN thread died: {}".format(d.loop_error)
        time.sleep(0.02)
    return d.get_snapshot().get("policy")


def keep_alive_until(d, pred, timeout=20.0):
    """Poll the dead-man like the panel does, until `pred(policy_snapshot)`."""
    t_end = time.time() + timeout
    last = None
    while time.time() < t_end:
        d.policy_keepalive()
        last = d.get_snapshot().get("policy")
        if last and pred(last):
            return last
        assert d.is_alive(), "the CAN thread died: {}".format(d.loop_error)
        time.sleep(0.02)
    return last


# ===================================================================== refusals (no bundle needed)
def test_a_run_without_the_support_acknowledgement_is_refused(armed):
    """The one hazard no check in the daemon can see. Every deployable bundle is railed in roll and
    yaw in training, so on a free-standing robot it is open-loop in exactly the axes that fall."""
    d, _cal = armed
    ok, why, _info = d.policy_arm({"file": "whatever.npz"})
    assert not ok
    assert "supported" in why and "railed" in why.lower()


def test_an_unknown_bundle_is_refused_by_name(armed):
    d, _cal = armed
    ok, why, _ = d.policy_arm({"file": "not_a_real_bundle.npz", "supported": True})
    assert not ok and "not_a_real_bundle.npz" in why


def test_a_path_is_not_a_bundle_name(armed):
    """The filename is basename()d and must end in .npz -- data/policies/ is a jail, not a hint."""
    d, _cal = armed
    ok, why, _ = d.policy_arm({"file": "../../../etc/passwd", "supported": True})
    assert not ok and ".npz" in why


def test_arming_is_refused_while_another_activity_owns_the_bus(armed):
    """One thread, one bus, one thing at a time. A burst and a policy would interleave frames."""
    d, _cal = armed
    d._therm = {"running": True, "motor": "left.thigh"}
    ok, why, _ = d.policy_arm(spec())
    assert not ok and "thermal burst" in why


def test_arming_is_refused_while_estopped(armed):
    d, _cal = armed
    d.estop("test")
    assert wait_mode(d, "ESTOPPED")
    ok, why, _ = d.policy_arm(spec())
    assert not ok and "e-stop" in why.lower()


# ===================================================================== bundle-level refusals
@needs_bundle
def test_a_privileged_observation_bundle_can_never_be_deployed(armed, tmp_path):
    """obs_base_vel=True puts the simulator's true base velocity in the ACTOR observation. No robot
    can produce that number, so the policy is undeployable -- and it must be refused at arm time,
    where it is one message, not at tick 1 where it is an exception inside the CAN thread."""
    d, _cal = armed
    from bundle import Bundle
    b = Bundle.load(BUNDLE_SRC)
    dest = os.path.join(paths.POLICY_DIR, "privileged_test.npz")
    try:
        Bundle.save(dest, b.a, dict(b.meta, obs_base_vel=True))
        ok, why, _ = d.policy_arm(spec(file="privileged_test.npz"))
        assert not ok
        assert "obs_base_vel" in why and "PRIVILEGED" in why
    finally:
        if os.path.exists(dest):
            os.remove(dest)


@needs_bundle
def test_a_bundle_for_another_control_rate_is_refused(armed):
    """The action filter, the actuation delay and the slew limit are all PER-STEP constants. Run
    at the wrong rate they are a different control law, not a slightly-off one."""
    d, _cal = armed
    from bundle import Bundle
    b = Bundle.load(BUNDLE_SRC)
    dest = os.path.join(paths.POLICY_DIR, "slow_test.npz")
    try:
        Bundle.save(dest, b.a, dict(b.meta, control_dt=0.02))
        ok, why, _ = d.policy_arm(spec(file="slow_test.npz"))
        assert not ok and "50 Hz" in why and "200" in why
    finally:
        if os.path.exists(dest):
            os.remove(dest)


@needs_bundle
def test_an_unverified_joint_map_is_refused_unless_acknowledged(staged):
    """A sign error here drives balance corrections the WRONG WAY at up to 500 N*m/rad and looks
    exactly like a bad policy. It is overridable -- but only on purpose."""
    d, _cal, fname = staged
    import jointmap as JM
    real = JM.JointMap.load
    JM.JointMap.load = staticmethod(lambda p: JM.JointMap())      # nothing verified
    try:
        ok, why, _ = d.policy_arm(spec(file=fname, skip_jointmap_check=False))
        assert not ok and "not verified" in why
        ok, _why, _ = d.policy_arm(spec(file=fname, skip_jointmap_check=True))
        assert ok, "the acknowledgement must actually let it through"
    finally:
        JM.JointMap.load = real
        d.policy_stop(hard=True)


@needs_bundle
def test_a_missing_deploy_map_names_the_tool_that_builds_it(staged, tmp_path):
    d, _cal, fname = staged
    old = daemon_mod.paths.DEPLOY
    daemon_mod.paths.DEPLOY = str(tmp_path)
    try:
        ok, why, _ = d.policy_arm(spec(file=fname))
        assert not ok
        assert "deploy_map.json" in why and "make_deploy_map.py" in why
    finally:
        daemon_mod.paths.DEPLOY = old


@needs_bundle
def test_no_imu_is_refused_unless_acknowledged(staged):
    """Gravity is the only fall detector there is."""
    d, _cal, fname = staged
    d.sense = None
    ok, why, _ = d.policy_arm(spec(file=fname))
    assert not ok and "fall detector" in why
    ok, _why, _ = d.policy_arm(spec(file=fname, no_imu=True))
    assert ok
    d.policy_stop(hard=True)


@needs_bundle
def test_a_command_outside_the_trained_box_is_clamped_and_says_so(staged):
    """A command the policy was never trained to resolve puts the task channel off-manifold. It is
    clamped rather than refused, and the clamp is reported so the panel can show what was run."""
    d, _cal, fname = staged
    ok, why, info = d.policy_arm(spec(file=fname, v_cmd=99.0))
    assert ok, why
    assert info["v_cmd_clamped"] is True
    assert info["v_cmd"] < 99.0
    d.policy_stop(hard=True)


# ===================================================================== the run itself
@needs_bundle
def test_a_run_goes_hold_approach_run_and_stops_on_its_own_deadline(staged):
    """The whole phase machine, end to end, on MockBus."""
    d, _cal, fname = staged
    ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=1.5))
    assert ok, why
    assert wait_mode(d, "POLICY"), d.get_snapshot()
    p = keep_alive_until(d, lambda p: p["phase"] == "done", timeout=30.0)
    assert p["phase"] == "done", p
    assert p["reached_run"], "never got past the approach: {}".format(p["exit_reason"])
    assert "max run time" in p["exit_reason"]
    assert p["ticks"] > 100, "a 1.5 s run at 200 Hz should log hundreds of ticks, got {}".format(
        p["ticks"])
    assert d.is_alive() and d.loop_error is None


@needs_bundle
def test_the_daemon_keeps_publishing_while_a_policy_runs(staged):
    """REGRESSION. _publish() builds the snapshot while holding self.lock and calls _policy_pub()
    inside it; the dead-man helper took self.lock again. threading.Lock is not reentrant, so the
    CAN thread deadlocked mid-publish -- with a policy armed and the motors live. The symptom was
    a UI frozen on its last frame, which is the worst possible way for this to fail."""
    d, _cal, fname = staged
    ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=2.0))
    assert ok, why
    assert wait_mode(d, "POLICY")
    first = d.get_snapshot()["loop"]["ticks"]
    t_end = time.time() + 1.5
    while time.time() < t_end:
        d.policy_keepalive()
        time.sleep(0.05)
    later = d.get_snapshot()["loop"]["ticks"]
    assert later > first + 100, "the control loop stopped ticking ({} -> {})".format(first, later)
    d.policy_stop(hard=True)


@needs_bundle
def test_letting_go_of_the_deadman_stops_the_run(staged):
    """The dead-man is the guard whose absence is invisible: everything looks fine right up to the
    moment nobody is watching. Stop refreshing it and the governor must soft-stop -- target frozen,
    gains bled out -- within POLICY_DEADMAN_S plus the ramp."""
    d, _cal, fname = staged
    ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=30.0))
    assert ok, why
    assert wait_mode(d, "POLICY")
    keep_alive_until(d, lambda p: p["phase"] == "run", timeout=30.0)
    t0 = time.time()
    p = None
    while time.time() - t0 < daemon_mod.POLICY_DEADMAN_S + 4.0:
        p = d.get_snapshot().get("policy")                 # deliberately NOT refreshing it
        if p and p["phase"] == "done":
            break
        time.sleep(0.02)
    assert p and p["phase"] == "done", "the run outlived the dead-man: {}".format(p)
    assert "dead-man" in p["exit_reason"]
    assert time.time() - t0 < daemon_mod.POLICY_DEADMAN_S + 3.0


@needs_bundle
def test_a_soft_stop_bleeds_the_gains_out_and_a_hard_stop_does_not(staged):
    """A soft stop puts the robot DOWN; a hard stop drops it. Both are correct answers to different
    questions, and the difference has to be real rather than a label."""
    d, _cal, fname = staged
    ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=30.0))
    assert ok, why
    keep_alive_until(d, lambda p: p["phase"] == "run", timeout=30.0)
    d.policy_stop(hard=False)
    ramps = []
    t_end = time.time() + 2.0
    while time.time() < t_end:
        d.policy_keepalive()
        p = d.get_snapshot().get("policy")
        if p:
            ramps.append(p["ramp"])
            if p["phase"] == "done":
                break
        time.sleep(0.01)
    assert ramps and min(ramps) <= 0.0, "the gains never reached zero: {}".format(ramps[:20])
    assert any(0.0 < r < 1.0 for r in ramps), "a soft stop must RAMP, not step: {}".format(ramps)


@needs_bundle
def test_a_hard_kill_limps_immediately(staged):
    d, _cal, fname = staged
    ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=30.0))
    assert ok, why
    keep_alive_until(d, lambda p: p["phase"] == "run", timeout=30.0)
    d.policy_stop(hard=True)
    p = keep_alive_until(d, lambda p: p["phase"] == "done", timeout=3.0)
    assert p["phase"] == "done"
    assert p["ramp"] == 0.0
    assert not d._last_cmd_raw, "a limped robot must not still be publishing a position target"


@needs_bundle
def test_a_fall_hard_stops_the_run(staged):
    """grav_z above term_gravity_z is the termination the policy was TRAINED to: past it the robot
    is outside every state it has ever seen, quite apart from being on its way to the floor."""
    d, _cal, fname = staged
    ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=30.0))
    assert ok, why
    keep_alive_until(d, lambda p: p["phase"] == "run", timeout=30.0)
    d.sense.up = np.array([1.0, 0.0, 0.0])                 # flat on its side
    p = keep_alive_until(d, lambda p: p["phase"] == "done", timeout=3.0)
    assert p["phase"] == "done"
    assert "fallen" in p["exit_reason"]


@needs_bundle
def test_a_stale_imu_stops_the_run(staged):
    """A frozen IMU reads as a perfectly upright robot forever, so the fall kill can never fire.
    Staleness has to be its own kill or the guard silently stops existing."""
    d, _cal, fname = staged
    ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=30.0))
    assert ok, why
    keep_alive_until(d, lambda p: p["phase"] == "run", timeout=30.0)
    d.sense.stale_by = 5.0
    p = keep_alive_until(d, lambda p: p["phase"] == "done", timeout=3.0)
    assert "IMU sample" in p["exit_reason"]


@needs_bundle
def test_going_limp_ends_the_run_without_throwing_the_log_away(staged):
    """The 200 Hz log is the entire product of a run. Dropping the state on the way to LIMP -- the
    way every other mode here drops its scratch state -- would delete it at exactly the moment the
    operator reached for the safest-looking button."""
    d, _cal, fname = staged
    ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=30.0))
    assert ok, why
    keep_alive_until(d, lambda p: p["phase"] == "run", timeout=30.0)
    d.request_mode("LIMP")
    assert wait_mode(d, "LIMP")
    got = d.get_policy()
    assert got is not None, "the run log did not survive the mode change"
    assert got["ticks"] > 0 and got["log"].shape[0] == got["ticks"]
    assert "LIMP" in got["exit_reason"]


@needs_bundle
def test_an_estop_during_a_run_also_keeps_the_log(staged):
    """An e-stop mid-run is precisely when the 200 Hz window is worth having."""
    d, _cal, fname = staged
    ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=30.0))
    assert ok, why
    keep_alive_until(d, lambda p: p["phase"] == "run", timeout=30.0)
    d.estop("test e-stop")
    assert wait_mode(d, "ESTOPPED")
    got = d.get_policy()
    assert got is not None and got["ticks"] > 0
    assert d.is_alive() and d.loop_error is None


@needs_bundle
def test_the_log_columns_match_what_is_written(staged):
    """A log whose header lies is worse than no log. 64 columns, and the docstring says which."""
    d, _cal, fname = staged
    ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=1.0))
    assert ok, why
    got_p = keep_alive_until(d, lambda p: p["phase"] == "done", timeout=30.0)
    assert got_p["phase"] == "done"
    got = d.get_policy()
    assert got["log"].shape[1] == daemon_mod.POLICY_LOG_COLS == 64
    named = got["columns"].split("(")[0]
    # t + eleven six-vectors + two threes + three scalars
    assert named.count("6 |") + named.count("6 |") >= 1
    assert "MODEL actuator order" in got["columns"]
    assert np.all(np.isfinite(got["log"]))


@needs_bundle
def test_a_runaway_joint_is_killed_before_the_drives_have_to_do_it(staged):
    """REGRESSION, 2026-09-01. First policy run on the real drives: four of six faulted at once
    with error code 3, across both buses, at 46-52 degC -- right.thigh at 15570 ERPM being braked
    with 19.8 A. Over-voltage from regenerated energy the bus cannot sink.

    POLICY had NO measured-speed kill. The governor's rate clamp bounds how fast the TARGET may
    move, which is a different quantity from how fast the joint is turning, so the only thing that
    stopped it was the drives' own hardware fault trip -- which takes the whole bus down and leaves
    the robot limp at speed."""
    d, _cal, fname = staged
    # Poking one motor's .spd cannot work: the drain overwrites it from the bus every tick. Lower
    # the ceiling instead, so the mock's own honest motion crosses it -- which is also the thing
    # being tested, that the ceiling is READ each tick rather than baked in at arm time.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(daemon_mod, "POLICY_MAX_ERPM", 1.0)
    try:
        ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=30.0))
        assert ok, why
        p = keep_alive_until(d, lambda p: p["phase"] == "done", timeout=30.0)
        assert p["phase"] == "done", p
        assert "runaway" in p["exit_reason"] and "ERPM" in p["exit_reason"]
        assert wait_mode(d, "ESTOPPED"), "a runaway must latch the e-stop, not just end the run"
        assert d.get_policy() is not None, "the log must survive -- this is the run worth reading"
    finally:
        monkeypatch.undo()


def test_the_runaway_ceiling_is_well_under_what_playback_allows(armed):
    """A walking policy has no business near no-load speed. In the 2026-09-01 recording the leg
    that was NOT running away sat at ~1500 ERPM, while the one that was reached 15570 -- and
    PLAYBACK's own hard cut is 16000, which is far too loose to protect a policy."""
    assert daemon_mod.POLICY_MAX_ERPM < daemon_mod.PLAYBACK_DEFAULTS["max_speed"] / 1.5
    assert daemon_mod.POLICY_MAX_ERPM > 4000.0, "and not so tight that a real gait trips it"


def test_the_speed_bypass_disarms_the_runaway_kill(armed):
    """Consistent with PLAYBACK, whose speed governor the same bypass switches off. The operator
    who took the legs off is allowed to spin a bare motor."""
    d, _cal = armed
    ok, _why = d.set_bypass("speed", True, note="test")
    assert ok and d.bypass["speed"] is True


# ===================================================================== can this machine run it?
@needs_bundle
def test_a_machine_too_slow_for_the_bundle_refuses_and_says_by_how_much(staged, monkeypatch):
    """MEASURED on the robot's Pi 3B (2026-09-01): 10.8 ms per control tick against a 5 ms budget,
    because that numpy is linked against the reference netlib BLAS rather than OpenBLAS. The loop
    free-runs when it slips, and control_dt is a CONSTANT inside the control law -- so the failure
    is not "a bit jittery", it is a gait clock at 0.45x and joint velocities inflated 2.2x in the
    observation. That has to be a refusal with the number in it, not a silent slow run."""
    d, _cal, fname = staged
    monkeypatch.setattr(daemon_mod, "POLICY_MAX_STEP_MS", 1e-6)
    ok, why, _info = d.policy_arm(spec(file=fname))
    assert not ok
    assert "ms per control tick" in why and "allow_slow_loop" in why
    ok, _why, info = d.policy_arm(spec(file=fname, allow_slow_loop=True))
    assert ok, "the acknowledgement must let it through -- watching the drives move is a real use"
    assert info["slow_loop"] is True and info["step_ms"] > 0
    d.policy_stop(hard=True)


@needs_bundle
def test_a_loop_that_falls_behind_mid_run_is_stopped(staged, monkeypatch):
    """The arm-time probe says the machine CAN; this says it IS. They are different questions --
    the IMU thread, Flask, the recorder and the GC all compete for the same four cores, and a
    bundle that probed fine can still lose the race once everything else is running."""
    d, _cal, fname = staged
    ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=30.0))
    assert ok, why
    keep_alive_until(d, lambda p: p["phase"] == "run", timeout=30.0)
    monkeypatch.setattr(daemon_mod, "POLICY_MIN_RATE_FRAC", 5.0)   # nothing can satisfy this
    p = keep_alive_until(d, lambda p: p["phase"] == "done", timeout=10.0)
    assert p["phase"] == "done", p
    assert "control loop is running at" in p["exit_reason"]
    assert p["rate_hz"] > 0


@needs_bundle
def test_an_acknowledged_slow_loop_is_not_killed_by_the_rate_guard(staged, monkeypatch):
    """Having said 'yes, slower than trained', the operator must not then be stopped for it every
    second. The acknowledgement has to disarm BOTH gates or it disarms neither usefully."""
    d, _cal, fname = staged
    monkeypatch.setattr(daemon_mod, "POLICY_MAX_STEP_MS", 1e-6)
    monkeypatch.setattr(daemon_mod, "POLICY_MIN_RATE_FRAC", 5.0)
    ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=3.0, allow_slow_loop=True))
    assert ok, why
    p = keep_alive_until(d, lambda p: p["phase"] == "done", timeout=30.0)
    assert "max run time" in p["exit_reason"], (
        "an acknowledged slow loop was killed by the rate guard anyway: {}".format(
            p["exit_reason"]))


# ===================================================================== the workspace polygon
class _Limits:
    """A workspace that accepts the first `n_ok` poses and then refuses everything.

    That shape is the point: the STANCE has to pass (or the run is refused at arm time and the
    run-time path never executes), and then the polygon has to start refusing while the policy is
    live. A workspace that refuses from the start tests a different, easier thing."""

    def __init__(self, n_ok):
        self.n = 0
        self.n_ok = n_ok

    def has_leg(self, side):
        return side == "right"

    def validate(self, side, abd, cam, thigh):
        self.n += 1
        if self.n <= self.n_ok:
            return True, ""
        return False, "test polygon says no"


@needs_bundle
def test_the_policy_is_frozen_then_killed_at_the_workspace_edge(staged):
    """The governor bounds each joint independently and therefore cannot see self-collision, which
    is a property of the COMBINATION. So the polygon check is applied to the governor's OUTPUT --
    and it is applied the governor's way: freeze at the last pose that passed, kill only if the
    policy is still asking for the same forbidden place 100 ms later. One transient must not drop a
    standing robot."""
    d, _cal, fname = staged
    lim = _Limits(n_ok=10_000)
    d.wstore = types.SimpleNamespace(limits=lim, legs={})
    ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=30.0))
    assert ok, why
    keep_alive_until(d, lambda p: p["phase"] == "run", timeout=30.0)
    lim.n_ok = lim.n                                        # from here on, refuse everything
    p = keep_alive_until(d, lambda p: p["phase"] == "done", timeout=5.0)
    assert p["phase"] == "done", p
    assert "safe workspace" in p["exit_reason"]
    assert p["ws_blocked_ticks"] >= daemon_mod.POLICY_WS_PERSIST_TICKS, (
        "it must FREEZE for the dwell window before killing, not kill on the first refusal: "
        "{} blocked ticks".format(p["ws_blocked_ticks"]))


@needs_bundle
def test_a_stance_outside_the_workspace_is_refused_at_arm_time(staged):
    """If the pose the gait is centred on is not one this robot may hold, the approach would refuse
    on its first tick. Saying so before anything is energised is the whole difference between a
    message and a mystery."""
    d, _cal, fname = staged
    d.wstore = types.SimpleNamespace(limits=_Limits(n_ok=0), legs={})
    ok, why, _ = d.policy_arm(spec(file=fname))
    assert not ok
    assert "stance" in why and "workspace" in why


# ===================================================================== HTTP contract
@pytest.fixture
def client(armed):
    d, cal = armed
    keys = ("daemon", "calib", "mock", "fk", "sense", "wstore", "dyn")
    saved = {k: server.STATE[k] for k in keys}
    server.STATE.update(daemon=d, calib=cal, mock=True, sense=d.sense,
                        fk=types.SimpleNamespace(available=False, try_reload=lambda: None,
                                                 side_verified=lambda side: False),
                        wstore=types.SimpleNamespace(legs={"left": {}, "right": {}},
                                                     source="workspace_test.npz",
                                                     list_files=lambda: []),
                        dyn=types.SimpleNamespace(snapshot=lambda: {}))
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c, d
    server.STATE.update(saved)


def test_arming_without_a_calibration_is_a_403(robot):              # noqa: F811
    """Every joint angle the policy reads is derived from the zero. There is no run without one."""
    d, cal, _b, _dir = robot
    saved = {k: server.STATE[k] for k in ("daemon", "calib")}
    server.STATE.update(daemon=d, calib=cal)
    server.app.config["TESTING"] = True
    try:
        with server.app.test_client() as c:
            r = c.post("/api/policy/arm", json={"file": "x.npz", "supported": True})
            assert r.status_code == 403
            assert "calibration" in r.get_json()["error"]
    finally:
        server.STATE.update(saved)


def test_a_refused_arm_is_an_error_envelope_not_a_silent_ok(client):
    """The shared api() helper in app.js throws on ok:false and banners d.error. A refusal that
    came back as ok:true with a flag would be a Start button that looks like it worked."""
    c, _d = client
    r = c.post("/api/policy/arm", json={"file": "nope.npz", "supported": True})
    j = r.get_json()
    assert r.status_code == 400
    assert j["ok"] is False and "nope.npz" in j["error"]


def test_the_keepalive_carries_the_daemon_state(client):
    """The panel's dead-man post IS its status poll -- one request that both says 'someone is
    watching' and refreshes the readout. If the envelope ever stopped carrying `state`, the run
    display would freeze while the run kept going."""
    c, _d = client
    j = c.post("/api/policy/keepalive", json={}).get_json()
    assert j["ok"] is True
    assert j["state"]["mode"] in daemon_mod.MODES


def test_saving_with_nothing_to_save_is_a_404_not_a_crash(client):
    c, _d = client
    assert c.post("/api/policy/run/save", json={}).status_code == 404


def test_the_preflight_reports_the_imu_and_whether_the_robot_is_free(client):
    """These two are new gates on a real run, and the panel decides what to enable from them."""
    c, d = client
    names = [x["name"] for x in server._policy_preflight()]
    assert "IMU live" in names and "robot free" in names
    d._therm = {"running": True, "motor": "left.thigh"}
    busy = [x for x in server._policy_preflight() if x["name"] == "robot free"][0]
    assert not busy["ok"] and "thermal" in busy["why"]


@needs_bundle
def test_a_finished_run_is_saved_with_a_readable_meta_block(client, staged):
    """The npz is the artefact. It has to be readable by numpy alone, with allow_pickle off."""
    c, d = client
    _d, _cal, fname = staged
    ok, why, _ = d.policy_arm(spec(file=fname, max_seconds=1.0))
    assert ok, why
    keep_alive_until(d, lambda p: p["phase"] == "done", timeout=30.0)
    j = c.post("/api/policy/run/save", json={}).get_json()
    assert j["ok"] is True and j["rows"] > 0
    path = os.path.join(paths.POLICYRUN_DIR, j["file"])
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            data = z["data"]
        assert data.shape == (j["rows"], daemon_mod.POLICY_LOG_COLS)
        assert meta["run"] and meta["exit_reason"] and meta["columns"]
        assert c.get("/api/policy/runs").get_json()["ok"] is True
    finally:
        os.remove(path)


# ===================================================================== the force-control release
def test_limping_after_force_control_also_streams_the_zero_gain_frame(armed):
    """REGRESSION-shaped. A drive holds its last force-control frame, and SET_CURRENT 0 is not the
    release for that mode -- so LIMP after a policy run has to stream the zero-gain force frame
    too, for as long as it takes the drive to have heard it."""
    d, _cal = armed
    sent = []
    import canio
    real = canio.force_control
    canio.force_control = lambda bus, cid, payload: sent.append(payload)
    try:
        d._force_until = d._tick_mono + 5.0
        d._stream_limp()
        assert sent, "no force frame streamed inside the release window"
        import mit
        assert sent[0] == mit.limp_payload()
        sent.clear()
        d._force_until = 0.0
        d._stream_limp()
        assert not sent, "the release window must expire -- this is 1200 extra frames a second"
    finally:
        canio.force_control = real


def test_the_telemetry_age_is_the_oldest_motor_not_the_newest_frame(armed):
    """One drive going quiet while its five neighbours keep talking is the failure that matters,
    and a bus-wide 'did anything arrive' cannot see it."""
    d, _cal = armed
    assert len(d._rx_at) == paths.N_MOTORS, "the drain is not stamping arrivals"
    d._rx_at["left.cam"] = d._tick_mono - 10.0
    assert d._telemetry_age() >= 10.0
