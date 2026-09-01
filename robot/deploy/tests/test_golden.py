"""The control law must not change when it is only supposed to get faster.

verify_export.py is the authority that ties this numpy control law to the torch policy inside
MuJoCo, and it needs torch, mujoco and a training run on disk. This is the complementary net, and
it exists for exactly one job: when safety.step / controller.step / fourier_gait are rewritten for
speed, prove that every number they produce is BIT IDENTICAL to what they produced before.

Not "close". Identical. A rewrite that changes the last bit of a joint target is a different
control law from the one verify_export signed off, and the whole argument for optimising rather
than retraining is that the law is untouched.

The recorded trace covers the ordinary path for all 400 steps -- both governors stay `running`,
because a latched stop would spend the rest of the trace in the ramp-down branch -- plus the
position, rate, torque and gains clamps via a second governor fed deliberately bad commands.

    python robot/deploy/tests/make_golden.py     # regenerate, ONLY with a stated reason
"""
import os

import numpy as np
import pytest

import make_golden

GOLDEN = make_golden.GOLDEN


@pytest.mark.skipif(not os.path.exists(GOLDEN), reason="no golden trace recorded")
@pytest.mark.skipif(not os.path.isdir(os.path.join(make_golden.DEPLOY, "bundles")),
                    reason="no bundle to replay against")
def test_the_control_law_is_bit_identical_to_the_recorded_trace():
    want = np.load(GOLDEN, allow_pickle=True)["trace"]
    got, _status = make_golden.trace(make_golden._bundle_path())
    assert got.shape == want.shape, (
        "the trace changed SHAPE ({} vs {}) -- if that was deliberate, regenerate the golden and "
        "say why in the commit".format(got.shape, want.shape))
    if np.array_equal(got, want):
        return
    bad = np.flatnonzero(~(got == want).all(axis=1))
    col = np.flatnonzero(~(got == want).all(axis=0))
    worst = float(np.max(np.abs(got - want)))
    pytest.fail(
        "the control law changed. {} of {} steps differ, first at step {}, columns {}, worst "
        "|delta| {:.3e}.\n"
        "Columns are: target6 kp6 kd6 action30 [phase freq] | v.target6 v.kp6 v.kd6 winding6 "
        "case6 [stop ramp] | v2.target6 v2.kp6 v2.kd6 [stop2 ramp2].\n"
        "A speed optimisation must not move ANY of them. If the change is intended, regenerate "
        "the golden and re-run verify_export.py against the torch policy.".format(
            bad.size, want.shape[0], int(bad[0]) if bad.size else -1, col.tolist()[:12], worst))


@pytest.mark.skipif(not os.path.exists(GOLDEN), reason="no golden trace recorded")
def test_the_golden_actually_exercises_the_clamp_ladder():
    """A golden that only ever walks the happy path would sign off a rewrite that broke every
    limit in the governor."""
    _got, status = make_golden.trace(make_golden._bundle_path())
    counts = status["clamp_counts"]
    for name in ("position", "rate", "torque", "gains"):
        assert counts.get(name, 0) > 0, "the trace never exercises the {} clamp: {}".format(
            name, counts)
    assert status["stop"] == "running" and status["stop2"] == "running", (
        "a governor latched a stop mid-trace, so the rest of it exercises the ramp-down path "
        "instead of the ladder: {}".format(status))
