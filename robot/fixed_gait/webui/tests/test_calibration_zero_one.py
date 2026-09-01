"""Per-joint re-zero during the direction check (added 2026-09-01).

Posing all six joints at once takes three hands, and one joint whose zero came out wrong must
not cost re-posing the other five. The rules under test: only that joint's offset moves, its
sign survives (wiring property, not a pose property), its confirmation is cleared, the
raw-at-rest fingerprint and zero_epoch update (the pre-move guard compares against the new
pose), and a complete calibration drops back to the direction step until re-confirmed.

MockBus only -- no hardware.

    python -m pytest robot/fixed_gait/webui/tests/test_calibration_zero_one.py -v
"""
import paths
from test_blackbox import capture_zero, robot           # noqa: F401  (pytest fixtures)
from test_thermal_api import client                     # noqa: F401  (the wired test client)


def test_zero_one_moves_one_offset_and_nothing_else(robot):
    d, cal, _b, _dir = robot
    ok, why = cal.set_zero(d.latest_raw_positions())
    assert ok, why
    before = dict(cal.offsets)
    before_signs = dict(cal.signs)
    epoch = cal.zero_epoch
    raw = d.latest_raw_positions()["left.thigh"]

    ok, why = cal.set_zero_one("left.thigh", raw + 7.5)
    assert ok, why
    assert cal.offsets["left.thigh"] == raw + 7.5
    assert cal.zero_raw["left.thigh"] == raw + 7.5
    for n in paths.MOTOR_NAMES:
        if n != "left.thigh":
            assert cal.offsets[n] == before[n], "another joint's zero moved"
    assert cal.signs == before_signs, "a re-zero must never touch a sign"
    assert cal.confirmed["left.thigh"] is False
    assert cal.zero_epoch == epoch + 1, "the fingerprint changed; the guard must know"


def test_zero_one_drops_a_complete_calibration_back_to_the_direction_step(robot):
    d, cal, _b, _dir = robot
    capture_zero(d, cal)
    assert cal.complete
    ok, why = cal.set_zero_one("right.cam", 12.0)
    assert ok, why
    assert cal.stage == "zero_set", "an unconfirmed joint cannot leave the calibration complete"
    assert cal.confirmed["right.cam"] is False
    # confirming just that joint completes it again -- the other five kept their ticks
    ok, why = cal.confirm("right.cam")
    assert ok, why
    assert cal.complete


def test_zero_one_refuses_before_a_full_capture(robot):
    _d, cal, _b, _dir = robot
    ok, why = cal.set_zero_one("left.thigh", 5.0)
    assert not ok and "full zero pose first" in why


def test_zero_one_refuses_an_unknown_or_silent_motor(robot):
    d, cal, _b, _dir = robot
    ok, why = cal.set_zero(d.latest_raw_positions())
    assert ok, why
    ok, why = cal.set_zero_one("middle.knee", 5.0)
    assert not ok and "unknown motor" in why
    ok, why = cal.set_zero_one("left.thigh", None)
    assert not ok and "no position" in why


# ===================================================================== the HTTP layer
def test_the_endpoint_zeroes_from_the_live_position(client):
    """The client sends only the motor name; the raw angle must come from telemetry the daemon
    owns, never from the browser."""
    c, d = client
    import server
    st = server.STATE["calib"]
    assert st.complete
    live = d.latest_raw_positions()["left.thigh"]
    r = c.post("/api/calibration/zero_one", json={"motor": "left.thigh"})
    assert r.status_code == 200, r.get_json()
    assert abs(st.offsets["left.thigh"] - live) < 2.0   # mock drifts a little between reads
    assert st.stage == "zero_set"


def test_the_endpoint_refuses_when_not_limp(client):
    c, d = client
    d.request_mode("MANUAL")
    from test_blackbox import wait_mode
    assert wait_mode(d, "MANUAL")
    r = c.post("/api/calibration/zero_one", json={"motor": "left.thigh"})
    assert r.status_code == 400 and "LIMP" in r.get_json()["error"]
    d.request_mode("LIMP")
