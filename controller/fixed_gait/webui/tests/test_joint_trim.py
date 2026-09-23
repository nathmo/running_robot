"""Per-joint trim on the standing pose.

STAND_POSE_DEG is solved from CAD and is right for a robot whose zero is right. When a joint is
zeroed a degree or two out, or one hip simply sits differently, the whole pose is off and there was
nothing to turn but the balance loop -- which is the wrong tool, and on 2026-09-23 was not working
anyway. This is the knob.

What it must reach: Home, the standing hold, the pose the balance loop regulates around, and the
tolerance check that decides whether the robot is close enough to START balancing. What it must NOT
reach: a policy run, which is far more sensitive to joint-zero error than anything here (98%
upright at +-0.3 deg, 54% at +-2 deg -- RLframework/tools/homing_tolerance.py, 2026-09-17), and the
manual sliders, which show an absolute angle the operator is looking at.
"""
import time

import numpy as np
import pytest

import paths
import balance
import calibration
import daemon as daemon_mod


@pytest.fixture
def robot(tmp_path):
    cal = calibration.Calibration()
    cal.save = lambda *a, **k: None
    d = daemon_mod.RobotDaemon(mock=True, calib=cal, wstore=None, fklut=None, bb=None,
                               anchor_file=str(tmp_path / "pose_anchor.json"),
                               balance_file=str(tmp_path / "balance.json"))
    d.start()
    assert d._started_ok.wait(5.0)
    for _ in range(300):
        if all(m.pos is not None for m in d.motors):
            break
        time.sleep(0.02)
    yield d, cal
    d.stop_event.set()
    d.join(2.0)


# ===================================================================== what it moves
def test_the_trim_moves_the_standing_pose(robot):
    d, _cal = robot
    base = dict(d._stand_pose())
    d.joint_trim({"left.cam": 4.5, "right.abd": -3.0})
    got = d._stand_pose()
    assert got["left.cam"] == pytest.approx(base["left.cam"] + 4.5)
    assert got["right.abd"] == pytest.approx(base["right.abd"] - 3.0)
    for n in paths.MOTOR_NAMES:
        if n not in ("left.cam", "right.abd"):
            assert got[n] == pytest.approx(base[n]), f"{n} moved and should not have"


def test_home_drives_to_the_trimmed_pose(robot):
    d, _cal = robot
    d.joint_trim({"left.thigh": 6.0})
    want = daemon_mod.STAND_POSE_DEG["left.thigh"] + 6.0
    assert d._stand_targets()["left.thigh"] == pytest.approx(want)


def test_every_motor_can_be_trimmed(robot):
    """'cam + thigh + left and right abduction' -- all six, independently."""
    d, _cal = robot
    want = {n: (i + 1) * 1.5 for i, n in enumerate(paths.MOTOR_NAMES)}
    got, note = d.joint_trim(want)
    assert not note, note
    for n, v in want.items():
        assert got[n] == pytest.approx(v)
        assert d._stand_pose()[n] == pytest.approx(daemon_mod.STAND_POSE_DEG[n] + v)


def test_the_trim_is_clipped_to_its_limit(robot):
    d, _cal = robot
    lim = daemon_mod.JOINT_TRIM_LIMIT_DEG
    assert lim == 20.0
    got, _ = d.joint_trim({"left.cam": 500.0, "right.cam": -500.0})
    assert got["left.cam"] == lim and got["right.cam"] == -lim


def test_an_unknown_motor_is_refused_rather_than_ignored(robot):
    d, _cal = robot
    got, why = d.joint_trim({"left.knee": 3.0})
    assert got is None and "unknown motor" in why


# ===================================================================== what it must NOT move
def test_the_trim_never_reaches_a_policy_run():
    """The one hard rule. A policy is commanded from its own controller output and the trim is not
    in that path at all; this pins it so nobody wires it in later by helpfully 'applying the trim
    everywhere'."""
    import inspect
    src = inspect.getsource(daemon_mod.RobotDaemon._tick_policy)
    assert "_joint_trim" not in src and "_stand_pose" not in src, (
        "the policy tick now reads the standing-pose trim — see JOINT_TRIM_LIMIT_DEG")
    src = inspect.getsource(daemon_mod.RobotDaemon._tick_playback)
    assert "_joint_trim" not in src and "_stand_pose" not in src, (
        "PLAYBACK now reads the standing-pose trim, which would silently shift a recorded gait")


def test_the_trim_does_not_move_the_manual_sliders(robot):
    """A manual target is an absolute angle the operator is looking at."""
    d, cal = robot
    ok, why = cal.set_zero(d.latest_raw_positions())
    assert ok, why
    for n in paths.MOTOR_NAMES:
        cal.confirm(n)
    with d.lock:
        d._manual_targets = {n: 3.0 for n in paths.MOTOR_NAMES}
    d.joint_trim({"left.cam": 9.0})
    with d.lock:
        assert d._manual_targets["left.cam"] == 3.0, "a manual target was shifted by the trim"


# ===================================================================== the balance loop
def test_a_running_loop_is_rebased_not_restarted(robot):
    """Changing a trim while balancing must move the setpoint, not throw away the integrator: a
    loop that reset its state on every click of ◀ would kick the robot each time."""
    d, _cal = robot
    bal = balance.Balancer(d._stand_pose())
    bal.i_pitch = 0.42
    with d.lock:
        d._bal = bal
    d.joint_trim({"left.cam": 5.0})
    assert bal.stand["left.cam"] == pytest.approx(daemon_mod.STAND_POSE_DEG["left.cam"] + 5.0)
    assert bal.i_pitch == pytest.approx(0.42), "the loop's integrator was reset by a trim"


def test_the_balance_start_check_compares_against_the_trimmed_pose(robot):
    """Home to the trimmed pose, then start Balance: the tolerance check has to be looking at the
    same pose Home just drove to, or it refuses a robot standing exactly where it was told to."""
    d, _cal = robot
    d.joint_trim({"left.cam": 3.0, "right.thigh": -3.0})
    stand = d._stand_targets()
    # every joint AT the trimmed pose -> zero error, comfortably inside the tolerance
    off = {n: stand[n] - d._stand_pose()[n] for n in paths.MOTOR_NAMES}
    assert max(abs(v) for v in off.values()) < daemon_mod.BALANCE_START_TOL_DEG


# ===================================================================== persistence
def test_the_trim_survives_a_restart(robot, tmp_path):
    d, cal = robot
    d.joint_trim({"right.cam": -7.5, "left.abd": 2.0})
    d.stop_event.set()
    d.join(2.0)

    again = daemon_mod.RobotDaemon(mock=True, calib=cal, wstore=None, fklut=None, bb=None,
                                   anchor_file=str(tmp_path / "pose_anchor.json"),
                                   balance_file=str(tmp_path / "balance.json"))
    again._load_balance_settings()
    assert again._joint_trim["right.cam"] == pytest.approx(-7.5)
    assert again._joint_trim["left.abd"] == pytest.approx(2.0)
    assert again._joint_trim["left.cam"] == 0.0


def test_a_saved_trim_is_re_clipped_on_load(robot, tmp_path):
    """A hand-edited or older settings file must not be able to smuggle in a 90 deg trim."""
    import json
    d, cal = robot
    p = tmp_path / "balance.json"
    p.write_text(json.dumps({"joint_trim": {"left.cam": 90.0}}), encoding="utf-8")
    d._bal_file = str(p)
    d._load_balance_settings()
    assert d._joint_trim["left.cam"] == daemon_mod.JOINT_TRIM_LIMIT_DEG
