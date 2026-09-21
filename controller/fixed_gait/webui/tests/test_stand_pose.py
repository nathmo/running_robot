"""🏠 Home drives to the standing pose (daemon.STAND_POSE_DEG), and that constant is still what
tools/solve_stand_pose.py derives from the CAD + masses (skipped where mujoco/scipy are missing,
i.e. on the Pi).

    python -m pytest controller/fixed_gait/webui/tests/test_stand_pose.py -v
"""
import importlib.util
import os

import numpy as np
import pytest

import daemon as daemon_mod
import paths
from test_blackbox import capture_zero, robot, wait_mode      # noqa: F401  (pytest fixtures)

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools",
                    "solve_stand_pose.py")


def test_home_targets_the_standing_pose(robot):
    d, cal, _b, _dir = robot
    capture_zero(d, cal)
    ok, why = d.home()
    assert ok, why
    with d.lock:
        assert d._manual_targets == {n: daemon_mod.STAND_POSE_DEG[n] for n in paths.MOTOR_NAMES}
        assert d._home_kind == "stand"
    assert wait_mode(d, "MANUAL")


def test_stand_pose_is_symmetric_and_inside_the_hard_clamps():
    for side in paths.SIDES:
        for role in paths.ROLES:
            assert abs(daemon_mod.STAND_POSE_DEG[f"{side}.{role}"]) < daemon_mod.HARD_CLAMP[role]
    for role in paths.ROLES:      # normalized frame: both legs read the same in a mirrored pose
        assert daemon_mod.STAND_POSE_DEG[f"left.{role}"] == daemon_mod.STAND_POSE_DEG[f"right.{role}"]


def test_stand_pose_matches_the_solver():
    pytest.importorskip("mujoco")
    pytest.importorskip("scipy")
    spec = importlib.util.spec_from_file_location("solve_stand_pose", TOOL)
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    rb = sp.Robot()
    cam, thigh = rb.solve()[0]                    # the solution nearer the homing pose
    s = rb.pose(cam, thigh)
    assert abs(s["com"][0] - s["sole"][0]) < 1e-4, "CoM must sit over the sole centre"
    got = daemon_mod.STAND_POSE_DEG
    assert got["left.cam"] == pytest.approx(sp.SIGNS["cam"] * np.degrees(cam), abs=0.1)
    assert got["left.thigh"] == pytest.approx(sp.SIGNS["thigh"] * np.degrees(thigh), abs=0.1)
    assert got["left.abd"] == 0.0
