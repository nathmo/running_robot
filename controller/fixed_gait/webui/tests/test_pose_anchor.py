"""The pre-move guard across a daemon restart, and the growable workspace canvas.

Two operator-reported faults, 2026-09-23:

  1. Restarting the web UI on a robot nobody had touched was answered with "the encoder origin
     moved, re-zero before moving". It had not moved. The guard's raw-at-rest check compared the
     live pose against the ZERO CAPTURE, so what it really asked was "is the robot standing in its
     zero pose?" -- false for any parked robot, and false by 45 deg after a Home. Measured live on
     DASH-01 the same day: right.cam +45.0, left.cam -41.5, both thighs ~21 deg off the capture,
     with a calibration in perfect health.

  2. The workspace editor would not let a safe region be drawn past ~36 deg of thigh. There was no
     such constant: the grid is built to hug the backdriven sweep, and the hit test dropped every
     stroke outside the array without a word.

These tests pin the fix from both ends: the guard must still refuse a genuinely moved origin, and
it must stop refusing one that did not move.
"""
import json
import os
import time

import numpy as np
import pytest

import paths
import calibration
import daemon as daemon_mod
import blackbox
import workspace


# ===================================================================== helpers
def _fresh_daemon(tmp_path, cal, name="pose_anchor.json", wstore=None):
    d = daemon_mod.RobotDaemon(mock=True, calib=cal, wstore=wstore, fklut=None, bb=None,
                               anchor_file=str(tmp_path / name))
    d.start()
    assert d._started_ok.wait(5.0)
    for _ in range(300):
        if all(m.pos is not None for m in d.motors):
            break
        time.sleep(0.02)
    assert all(m.pos is not None for m in d.motors), "mock drives never reported"
    return d


def _stop(d):
    d.stop_event.set()
    d.join(2.0)


def _freeze(d):
    """Stop the control loop but keep the object.

    The mock bus keeps answering while the loop runs, so a pose written onto a live daemon is
    overwritten by the next _drain -- these tests would then assert against the mock's idea of the
    robot instead of the parked pose they set. _premove_guard and _tick_anchor are both pure reads
    of motor state, so a frozen daemon exercises exactly the code a running one would."""
    _stop(d)
    return d


def _pose(d, raw):
    for n, m in d.by_name.items():
        m.pos = raw[n]
        m.spd = 0.0


def _zeroed(d, cal):
    ok, why = cal.set_zero(d.latest_raw_positions())
    assert ok, why
    for n in paths.MOTOR_NAMES:
        cal.confirm(n)
    assert cal.complete


@pytest.fixture
def cal():
    c = calibration.Calibration()
    c.save = lambda *a, **k: None          # never touch the operator's real calibration
    return c


# ===================================================================== the anchor itself
def test_an_anchor_is_only_evidence_for_the_calibration_it_was_written_under(tmp_path):
    """A re-zero makes every older anchor irrelevant: it describes a frame nobody uses any more."""
    p = str(tmp_path / "a.json")
    a = calibration.PoseAnchor(p)
    a.write({n: 10.0 for n in paths.MOTOR_NAMES}, zero_epoch=4, mode="LIMP", mono_s=1.0)

    back = calibration.PoseAnchor.load(p)
    assert back.valid_for(4)
    assert not back.valid_for(5), "an anchor older than the calibration must not be believed"


def test_a_partial_anchor_is_not_evidence(tmp_path):
    """A joint the writing process never heard from is a joint the reader cannot compare. Half a
    pose must not silently narrow the check to whoever happened to be awake."""
    p = str(tmp_path / "a.json")
    a = calibration.PoseAnchor(p)
    a.write({n: 1.0 for n in paths.MOTOR_NAMES[:3]}, zero_epoch=1, mode="LIMP", mono_s=1.0)
    assert not calibration.PoseAnchor.load(p).valid_for(1)


def test_a_missing_or_corrupt_anchor_never_raises(tmp_path):
    missing = calibration.PoseAnchor.load(str(tmp_path / "nope.json"))
    assert missing.raw == {} and not missing.valid_for(0)

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert calibration.PoseAnchor.load(str(bad)).raw == {}


def test_the_writer_rate_limits_a_still_robot_but_always_catches_a_move(tmp_path):
    """One small write per refresh window while nothing moves, immediately once something does."""
    a = calibration.PoseAnchor(str(tmp_path / "a.json"))
    pose = {n: 0.0 for n in paths.MOTOR_NAMES}
    assert a.should_write(pose, 0.0, 1.0, 30.0, 0.2), "the first sample is always worth writing"
    a.write(pose, 1, "LIMP", 0.0)

    assert not a.should_write(pose, 0.5, 1.0, 30.0, 0.2), "inside the rate limit"
    assert not a.should_write(pose, 5.0, 1.0, 30.0, 0.2), "still, and inside the refresh window"
    assert a.should_write(pose, 40.0, 1.0, 30.0, 0.2), "the refresh heartbeat must still land"

    moved = dict(pose, **{"left.cam": 9.0})
    assert a.should_write(moved, 2.0, 1.0, 30.0, 0.2), "a moved joint is written at once"


# ===================================================================== the guard across a restart
def test_a_restart_on_a_parked_robot_no_longer_demands_a_re_zero(tmp_path, cal):
    """THE REPORTED BUG. Zero, move the robot well away from the zero pose (a Home puts the cams at
    46 deg), restart the daemon without touching anything: the calibration must still be good."""
    d = _freeze(_fresh_daemon(tmp_path, cal))
    _zeroed(d, cal)
    # Park it far from the zero pose, exactly as Home does: STAND_POSE_DEG puts the cams at 46 deg.
    parked = {n: m.pos + (46.0 if n.endswith(".cam") else 21.0) for n, m in d.by_name.items()}
    _pose(d, parked)
    d._tick_anchor(time.monotonic())                     # the heartbeat the real loop would write
    assert os.path.exists(str(tmp_path / "pose_anchor.json")), "no heartbeat was ever written"

    # --- the restart: same calibration off disk, same untouched robot ---
    again = _freeze(_fresh_daemon(tmp_path, cal))
    again._zero_epoch_at_start = cal.zero_epoch          # restored from disk, not re-zeroed here
    _pose(again, parked)                                 # nobody moved the leg
    ok, why, detail = again._premove_guard()
    assert ok, f"a healthy calibration was refused after a restart: {why}"
    assert detail["anchor_usable"], "the guard did not use the anchor it was given"
    # The pose really is far from the zero capture, so the OLD check would have fired here: this
    # is the regression, not a robot that happened to be sitting at zero.
    assert abs(detail["compare"]["left.cam"]["delta"]) > daemon_mod.RAW_AT_REST_TOL_DEG
    assert max(abs(c["delta"]) for c in detail["anchor_compare"].values()) < 1e-6


def test_a_restart_still_refuses_when_an_origin_actually_moved(tmp_path, cal):
    """The safety property the anchor must not cost: a renumbered origin in the blind window is
    still caught, and the message says so."""
    d = _freeze(_fresh_daemon(tmp_path, cal))
    _zeroed(d, cal)
    parked = {n: m.pos + 30.0 for n, m in d.by_name.items()}
    _pose(d, parked)
    d._tick_anchor(time.monotonic())

    again = _freeze(_fresh_daemon(tmp_path, cal))
    again._zero_epoch_at_start = cal.zero_epoch
    _pose(again, parked)
    again.by_name["left.cam"].pos = 0.0                  # the documented renumber signature
    ok, why, _detail = again._premove_guard()
    assert not ok, "an origin that moved while the daemon was down must still be refused"
    assert "left.cam" in why
    assert "Re-zero before moving" in why
    assert "exactly 0.0" in why, "the renumber signature is worth naming: " + why


def test_with_no_anchor_at_all_the_guard_falls_back_to_the_zero_capture(tmp_path, cal):
    """No continuity record means no continuity evidence. The daemon must NOT adopt whatever the
    encoders happen to read at startup -- that would hand a clean bill of health to the power
    cycle this guard exists to catch."""
    d = _freeze(_fresh_daemon(tmp_path, cal, name="never_written.json"))
    _zeroed(d, cal)
    d._zero_epoch_at_start = cal.zero_epoch
    assert not d._anchor_at_start.valid_for(cal.zero_epoch)
    _pose(d, {n: m.pos + 40.0 for n, m in d.by_name.items()})      # park it away from zero
    ok, why, detail = d._premove_guard()
    assert not ok and "Re-zero before moving" in why
    assert not detail["anchor_usable"]


def test_continuity_once_established_is_not_re_examined(tmp_path, cal):
    """Latched on purpose. Once the startup comparison passes, the tick-by-tick jump watchdog is a
    better witness than any stored pose, and re-running this check against a startup snapshot
    would refuse the robot the moment it legitimately moved."""
    d = _freeze(_fresh_daemon(tmp_path, cal))
    _zeroed(d, cal)
    parked = {n: m.pos for n, m in d.by_name.items()}
    _pose(d, parked)
    d._tick_anchor(time.monotonic())

    again = _freeze(_fresh_daemon(tmp_path, cal))
    again._zero_epoch_at_start = cal.zero_epoch
    _pose(again, parked)
    assert again._premove_guard()[0]
    assert again._continuity_ok

    _pose(again, {n: v + 55.0 for n, v in parked.items()})   # now drive it somewhere legitimately
    ok, why, _ = again._premove_guard()
    assert ok, f"a robot that moved AFTER continuity was established was refused: {why}"


def test_the_heartbeat_is_not_written_while_a_joint_is_turning(tmp_path, cal):
    """A daemon killed mid-move must not leave an anchor from before the travel -- the next process
    would read that travel as an origin jump, which is the same false alarm one level down."""
    d = _freeze(_fresh_daemon(tmp_path, cal, name="spinning.json"))
    _zeroed(d, cal)
    path = str(tmp_path / "spinning.json")
    if os.path.exists(path):
        os.remove(path)

    for m in d.motors:
        m.spd = daemon_mod.ANCHOR_STILL_ERPM + 1.0       # every joint turning
    for _ in range(10):
        d._tick_anchor(time.monotonic())
    assert not os.path.exists(path), "a moving pose was anchored"

    # one joint is enough to disqualify the sample -- the pose is written whole or not at all
    for m in d.motors:
        m.spd = 0.0
    d.by_name["right.thigh"].spd = daemon_mod.ANCHOR_STILL_ERPM + 1.0
    d._tick_anchor(time.monotonic())
    assert not os.path.exists(path), "a pose with one joint still turning was anchored"

    d.by_name["right.thigh"].spd = 0.0                   # and once it settles, it lands
    d._tick_anchor(time.monotonic())
    assert os.path.exists(path), "a stationary pose was never anchored"


# ===================================================================== the clamps
def test_the_knee_pair_clamp_is_180_and_abduction_is_untouched():
    assert daemon_mod.HARD_CLAMP["cam"] == 180.0
    assert daemon_mod.HARD_CLAMP["thigh"] == 180.0
    assert daemon_mod.HARD_CLAMP["abd"] == 48.0, "abduction is a plain DOF with a real limit"
    assert daemon_mod.NOMINAL_RANGE == {"abd": 48.0, "cam": 88.0, "thigh": 62.0}


def test_an_empty_grid_padding_never_widens_the_never_exceed_clamp(tmp_path, cal):
    """The editor can grow the canvas past the sweep now. A canvas that was merely made bigger is
    not a demonstration of anything, so the clamp must size off the OCCUPIED cells -- otherwise
    drawing room would quietly buy clamp room."""
    ws = workspace.WorkspaceStore()
    ws._active_path = str(tmp_path / "active.npz")
    ws._persist_active = lambda: None

    res = 1.0
    occupied = np.zeros((400, 400), bool)
    occupied[195:205, 195:205] = True                     # a small blob in a very large canvas
    ws.apply_grid("left", occupied, cam_origin=-200.0, thigh_origin=-200.0, grid_deg=res)

    d = daemon_mod.RobotDaemon(mock=True, calib=cal, wstore=ws, fklut=None, bb=None,
                               anchor_file=str(tmp_path / "a.json"))
    lo, hi = d._hard_bounds("left", "thigh")
    assert (lo, hi) == (-180.0, 180.0), (
        f"empty padding widened the clamp to [{lo}, {hi}] — it must follow the occupied cells")

    occupied[:, 0] = True                                 # now DEMONSTRATE the far edge
    ws.apply_grid("left", occupied, cam_origin=-200.0, thigh_origin=-200.0, grid_deg=res)
    lo2, _ = d._hard_bounds("left", "thigh")
    assert lo2 == pytest.approx(-210.0), "a demonstrated cell still widens the clamp as before"


def test_the_travel_budget_stays_under_one_output_turn(tmp_path, cal):
    """The 2026-08-10 backstop. It is sized off the nominal range, NOT the +-180 clamp: budgeting
    off the clamp would raise it from ~0.6 of an output turn to 1.3 of one, past the excursion it
    was built to cut."""
    d = daemon_mod.RobotDaemon(mock=True, calib=cal, wstore=None, fklut=None, bb=None,
                               anchor_file=str(tmp_path / "a.json"))
    for role in ("cam", "thigh", "abd"):
        lo, hi = d._nominal_bounds("left", role)
        assert daemon_mod.TRAVEL_BUDGET_FACTOR * (hi - lo) < 360.0, role


def test_the_manual_span_does_not_follow_the_never_exceed_clamp(tmp_path, cal):
    """A slider spanning +-180 deg on the knee pair is a hard-stop collision one careless drag
    away. The UI spans the nominal range; only the refusal uses the clamp."""
    d = daemon_mod.RobotDaemon(mock=True, calib=cal, wstore=None, fklut=None, bb=None,
                               anchor_file=str(tmp_path / "a.json"))
    for role in ("cam", "thigh"):
        assert d._nominal_bounds("left", role) == (-daemon_mod.NOMINAL_RANGE[role],
                                                   daemon_mod.NOMINAL_RANGE[role])
        assert d._hard_bounds("left", role) == (-180.0, 180.0)
