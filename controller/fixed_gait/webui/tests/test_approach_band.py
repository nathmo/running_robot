"""The approach's tracking test: what counts as "the leg got there".

The 2026-09-23 flight failed twice with `reached_run: false` and zero policy ticks while the legs
were in fact standing on the policy's stance, because arrival was measured against `nominal_ctrl`
-- a PD command that sits a gravity deflection past any pose a loaded leg can hold. These pin both
ends of the band so neither case can regress.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import daemon as daemon_mod                                        # noqa: E402

miss = daemon_mod.approach_miss

# the numbers this actually ships against: dash_sprint_s2_191M, cam and thigh, radians
REST = np.array([0.0, 0.1265, -0.2025, 0.0, -0.1265, 0.2025])
CMD = np.array([-0.0007, 0.2298, -0.1161, 0.0016, -0.2181, 0.1238])
ARRIVE = np.radians(daemon_mod.POLICY_APPROACH_ARRIVE_DEG)
ABORT = np.radians(daemon_mod.POLICY_APPROACH_TRACK_ERR_DEG)


def test_the_command_and_the_rest_pose_really_are_a_deflection_apart():
    """If these ever coincide the whole band collapses and these tests stop proving anything."""
    assert np.degrees(np.max(np.abs(CMD - REST))) > 5.0


def test_a_leg_resting_under_full_weight_has_arrived():
    """The real robot: it sags to default_motor_pos. This is the case that failed on the robot."""
    assert float(np.max(miss(REST, REST, CMD))) == pytest.approx(0.0)


def test_a_leg_carrying_nothing_has_arrived():
    """Every mock, and the robot hanging on a hoist: it tracks the command exactly."""
    assert float(np.max(miss(CMD, REST, CMD))) == pytest.approx(0.0)


def test_a_leg_half_loaded_has_arrived():
    """On a tether, taking part of the weight -- anywhere between the two is a real equilibrium."""
    assert float(np.max(miss(0.5 * (REST + CMD), REST, CMD))) == pytest.approx(0.0)


def test_neither_endpoint_alone_would_accept_both():
    """Why the band exists: either endpoint on its own rejects one of the two legitimate cases."""
    assert float(np.max(np.abs(REST - CMD))) > ARRIVE       # command alone rejects the loaded leg
    assert float(np.max(np.abs(CMD - REST))) > ARRIVE       # rest alone rejects the unloaded one


def test_a_leg_stuck_short_of_the_band_is_caught():
    """A joint map that is wrong, a stale zero, or an obstruction -- all land outside."""
    bad = REST.copy()
    bad[1] -= np.radians(15.0)
    m = miss(bad, REST, CMD)
    assert float(np.max(m)) > ABORT
    assert int(np.argmax(m)) == 1


def test_a_leg_past_the_band_on_the_far_side_is_caught_too():
    """Overshoot is not arrival: the band is bounded at both ends, not a half space."""
    bad = CMD.copy()
    bad[2] += np.radians(20.0)
    m = miss(bad, REST, CMD)
    assert float(np.max(m)) > ABORT
    assert int(np.argmax(m)) == 2


def test_the_error_is_measured_from_the_near_edge():
    """A joint 4 deg outside is 4 deg of error, not 4 plus the width of the band."""
    off = np.radians(4.0)
    bad = np.minimum(REST, CMD) - off
    assert np.degrees(miss(bad, REST, CMD)) == pytest.approx(4.0, abs=1e-6)


def test_it_survives_the_endpoints_arriving_in_either_order():
    """rest is above the command on one leg and below it on the other -- min/max, not subtraction."""
    assert np.all((CMD - REST)[1:3] * (CMD - REST)[4:6] < 0)    # genuinely opposite signs
    for pos in (REST, CMD, 0.5 * (REST + CMD)):
        assert float(np.max(miss(pos, CMD, REST))) == pytest.approx(0.0)
        assert float(np.max(miss(pos, REST, CMD))) == pytest.approx(0.0)
