"""⚖ Balance: the controller (balance.py), its daemon integration, and its derivation.

MockBus + a fake IMU for the daemon; the kinematic maps and the MuJoCo scenarios only where mujoco
and scipy are installed (the dev machine, not the Pi).

    python -m pytest controller/fixed_gait/webui/tests/test_balance.py -v
"""
import importlib.util
import os
import time

import numpy as np
import pytest

import balance
import daemon as daemon_mod
import paths
from test_blackbox import capture_zero, robot, wait_mode      # noqa: F401  (pytest fixtures)

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
STAND = daemon_mod.STAND_POSE_DEG


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(TOOLS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the controller alone
def test_attitude_conventions_are_right_handed():
    # leaning forward by 10 deg (rotation about +y): world up tips toward body -x
    a = np.radians(10)
    assert balance.attitude_from_up([-np.sin(a), 0, np.cos(a)])[0] == pytest.approx(10)
    # leaning right by 10 deg (rotation about +x, left side up): world up tips toward body +y
    assert balance.attitude_from_up([0, np.sin(a), np.cos(a)])[1] == pytest.approx(10)


def test_level_and_still_returns_the_standing_pose():
    b = balance.Balancer(STAND)
    t = b.step(0.01, 0.0, 0.0, 0.0, 0.0)
    assert t == pytest.approx(STAND)


def test_a_torso_leaning_forward_is_levelled_by_the_integrator_and_the_cam_thigh_move_together():
    b = balance.Balancer(STAND)
    for _ in range(200):                        # 2 s of a steady 1 deg forward lean
        t = b.step(0.01, 1.0, 0.0, 0.0, 0.0)
    assert b.out["pitch"] < 0                   # pitching back
    assert b.out["com_x"] < 0                   # and the CoM pulled back along the soles
    for side in ("left", "right"):              # both legs the same in normalized degrees
        assert t[f"{side}.cam"] == pytest.approx(t["left.cam"])
        assert t[f"{side}.thigh"] == pytest.approx(t["left.thigh"])


def test_the_integrator_cannot_wind_past_its_clip():
    b = balance.Balancer(STAND)
    for _ in range(100000):
        b.step(0.01, 10.0, 0.0, 0.0, 0.0)
    assert abs(b.out["pitch"]) == pytest.approx(b.p["pitch_clip"])
    b.step(0.01, -10.0, 0.0, 0.0, 0.0)          # and it comes straight back off the clip
    assert abs(b.out["pitch"]) < b.p["pitch_clip"]


def test_roll_moves_both_hips_together_and_the_outputs_stay_clipped():
    b = balance.Balancer(STAND)
    t = b.step(0.01, 0.0, 30.0, 0.0, 500.0)
    assert abs(b.out["com_y"]) == pytest.approx(b.p["com_y_clip"])
    # a parallelogram: equal and opposite in normalized degrees (+ = outward on each side)
    assert t["left.abd"] == pytest.approx(-t["right.abd"])
    assert t["left.cam"] == pytest.approx(STAND["left.cam"])


def test_trims_are_clipped():
    b = balance.Balancer(STAND)
    b.set_trim(com_x_mm=999, com_y_mm=-999, pitch_deg=99)
    assert b.trim == {"com_x_mm": b.p["com_x_clip"], "com_y_mm": -b.p["com_y_clip"], "pitch_deg": 5.0}


# ---------------------------------------------------------------- daemon integration (MockBus)
class FakeIMU:
    """sensehat.SenseHat.fast() stand-in: (t, up_body, gyro_dps)."""

    def __init__(self):
        self.pitch = self.roll = 0.0
        self.dead = False
        self.age = 0.0

    def fast(self):
        if self.dead:
            return None
        p, r = np.radians(self.pitch), np.radians(self.roll)
        up = np.array([-np.sin(p) * np.cos(r), np.sin(r), np.cos(p) * np.cos(r)])
        return time.time() - self.age, up / np.linalg.norm(up), (0.0, 0.0, 0.0)


def _homed(d, cal):
    capture_zero(d, cal)
    ok, why = d.home()                  # the operator's default slew; the mock drives lag faster ones
    assert ok, why
    assert wait_mode(d, "MANUAL")
    t_end = time.time() + 15
    while time.time() < t_end:
        pose = {n: cal.norm(n, m.pos) for n, m in d.by_name.items()}
        if all(abs(pose[n] - STAND[n]) < 1.0 for n in paths.MOTOR_NAMES):
            return
        time.sleep(0.05)
    raise AssertionError(f"home never arrived: {pose}")


def test_balance_is_refused_unless_manual_at_the_stand_with_a_live_imu(robot):
    d, cal, _b, _dir = robot
    capture_zero(d, cal)
    d.sense = FakeIMU()
    ok, why = d.balance_start()
    assert not ok and "Home" in why                      # LIMP
    _homed(d, cal)
    d.sense = None
    ok, why = d.balance_start()
    assert not ok and "IMU" in why
    d.sense = FakeIMU()
    d.sense.mount = type("M", (), {"calibrated": False})()
    ok, why = d.balance_start()
    assert not ok and "mount" in why                     # chip axes are not robot axes
    d.sense.mount.calibrated = True
    ok, why = d.balance_start()
    assert ok, why
    time.sleep(0.2)                     # the snapshot is republished at 20 Hz
    assert d.get_snapshot()["manual"]["balance"]["active"]


def test_balance_follows_the_imu_then_stop_holds(robot):
    d, cal, _b, _dir = robot
    imu = FakeIMU()
    d.sense = imu
    _homed(d, cal)
    assert d.balance_start()[0]
    imu.pitch = 1.5
    time.sleep(1.0)
    out = d.get_snapshot()["manual"]["balance"]["out"]
    assert out["pitch"] == pytest.approx(1.5, abs=0.05)
    assert out["pitch_corr"] < 0 and out["com_x"] < 0
    assert d.get_snapshot()["mode"] == "MANUAL"
    held = dict(d._manual_targets)
    d.balance_stop()
    time.sleep(0.3)
    assert not d.get_snapshot()["manual"]["balance"]["active"]
    assert d.get_snapshot()["mode"] == "MANUAL", "stop HOLDS, it does not go limp"
    for n in paths.MOTOR_NAMES:
        assert d._manual_targets[n] == pytest.approx(held[n], abs=1.0)


def test_a_fall_trips_and_a_dead_imu_stops_balancing(robot):
    d, cal, _b, _dir = robot
    imu = FakeIMU()
    d.sense = imu
    _homed(d, cal)
    assert d.balance_start()[0]
    imu.dead = True
    time.sleep(0.3)
    assert not d.get_snapshot()["manual"]["balance"]["active"]
    assert d.get_snapshot()["mode"] == "MANUAL"          # held, not dropped
    imu.dead = False
    assert d.balance_start()[0]
    imu.pitch = 25.0
    assert wait_mode(d, "ESTOPPED"), d.get_snapshot()
    assert "falling" in d.get_snapshot()["estop"]["reason"]


def test_a_late_imu_sample_holds_and_a_stale_one_stops(robot):
    d, cal, _b, _dir = robot
    imu = FakeIMU()
    d.sense = imu
    _homed(d, cal)
    assert d.balance_start()[0]
    imu.pitch = 1.0
    time.sleep(0.5)
    before = dict(d._manual_targets)
    imu.age, imu.pitch = 0.1, -3.0                # late: the new reading must NOT be steered on
    time.sleep(0.4)
    assert d.get_snapshot()["manual"]["balance"]["active"]
    for n in paths.MOTOR_NAMES:
        assert d._manual_targets[n] == pytest.approx(before[n], abs=1e-9)
    imu.age = 0.5                                  # stale: stop, and hold
    time.sleep(0.3)
    assert not d.get_snapshot()["manual"]["balance"]["active"]
    assert d.get_snapshot()["mode"] == "MANUAL"


def test_a_long_balance_does_not_trip_the_homing_travel_budget(robot):
    d, cal, _b, _dir = robot
    imu = FakeIMU()
    d.sense = imu
    _homed(d, cal)
    assert d.balance_start()[0]
    t_end = time.time() + 3.0
    k = 0
    while time.time() < t_end:              # a jittery IMU: small corrections every tick
        imu.pitch = 0.8 * np.sin(k)
        k += 1
        time.sleep(0.02)
    assert d.get_snapshot()["mode"] == "MANUAL", d.get_snapshot().get("estop")


def test_jogging_cancels_balance(robot):
    d, cal, _b, _dir = robot
    d.sense = FakeIMU()
    _homed(d, cal)
    assert d.balance_start()[0]
    ok, why = d.manual_update({"left.abd": 1.0}, override=True)
    assert ok, why
    assert not d.get_snapshot()["manual"]["balance"]["active"]


# ---------------------------------------------------------------- derivation (dev machine only)
def test_the_maps_match_the_kinematics():
    pytest.importorskip("mujoco")
    pytest.importorskip("scipy")
    sp = _tool("solve_stand_pose")
    rb = sp.Robot()
    cam, thigh = rb.solve()[0]
    m = sp.balance_maps(rb, cam, thigh)
    for role in ("cam", "thigh"):
        assert balance.D_PITCH[role] == pytest.approx(m["D_PITCH"][role], rel=1e-3)
        assert balance.D_COM_X[role] == pytest.approx(m["D_COM_X"][role], rel=1e-3)
    assert balance.D_COM_Y_ABD == pytest.approx(m["D_COM_Y_ABD"], rel=1e-2)


def test_in_simulation_it_saves_a_zero_error_that_topples_the_uncontrolled_robot():
    pytest.importorskip("mujoco")
    pytest.importorskip("scipy")
    bs = _tool("balance_sim")
    z = {"left.cam": .65, "right.cam": .65, "left.thigh": -.65, "right.thigh": -.65}
    res = {}
    for on in (False, True):
        s = bs.Sim(zero_err=z)
        s.reset(STAND)
        res[on] = s.run(4.0, ctrl=balance.Balancer(STAND) if on else None, pose=STAND)
    assert res[False]["fell"], "the scenario must be one the uncontrolled robot fails"
    assert not res[True]["fell"], res[True]
    assert abs(res[True]["pitch"]) < 1.0
