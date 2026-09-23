"""Building a safe region from a hand sweep.

Reported 2026-09-23: "process + build workspace only returns a few smears of green".

The operator sweeps the leg around the EDGE of where it may go and comes back to the start, so the
yellow trail is a closed curve and the region they mean is the area inside it. The pipeline only
ever rasterized the cells the path itself crossed, so the region being outlined was never in the
grid at all -- and then, with the panel's own defaults, the morphology destroyed most of what was:

    dilate 2 deg (radius 2)  then  erode 3 deg (radius 3)   ==  a NET erosion of one cell

A traced boundary is about one cell wide. Dilated it is five; eroded by three it is gone. Measured
on a synthetic trace of the shape people actually sweep, an outline enclosing ~2356 cells of real
workspace produced 100 -- a broken ring, which is what "a few smears" looks like.
"""
import sys

import numpy as np
import pytest

import paths                                   # noqa: F401  (puts fixed_gait/ on sys.path)
import calibrate_workspace as cw
import workspace


def loop(turns=1.0, n=1200, cx=100.0, cy=5.0, rx=30.0, ry=25.0):
    """The trace of a hand sweep round the edge of a region. turns=1 closes it."""
    a = np.linspace(0.0, 2 * np.pi * turns, n)
    return rx * np.cos(a) + cx, ry * np.sin(a) + cy


def enclosed_area(rx=30.0, ry=25.0):
    return np.pi * rx * ry


# ===================================================================== the region itself
def test_a_traced_outline_becomes_the_region_it_encloses():
    """THE REPORTED BUG. The whole point of tracing a boundary is that the inside is the answer."""
    cam, thigh = loop()
    k = cw._knee_grid(cam, thigh, 1.0, 2.0, 3.0, paths=[(cam, thigh)], close_region=True)
    kept = k["safe_cells"]
    assert kept > 0.85 * enclosed_area(), (
        f"only {kept} cells kept of ~{enclosed_area():.0f} enclosed — the outline was not filled")

    # and without closing it is the old, useless answer
    open_k = cw._knee_grid(cam, thigh, 1.0, 2.0, 3.0, paths=[(cam, thigh)], close_region=False)
    assert open_k["safe_cells"] < 0.1 * kept, (
        "closing must be what makes the difference, or this test proves nothing")


def test_the_margin_is_taken_off_the_region_not_off_the_trace():
    """The kept region must sit strictly inside the swept outline, by about the margin."""
    cam, thigh = loop()
    for margin in (3.0, 6.0):
        k = cw._knee_grid(cam, thigh, 1.0, 2.0, margin, paths=[(cam, thigh)])
        grid = k["safe_grid"]
        js = np.flatnonzero(grid.any(axis=0))
        height = (js[-1] - js[0] + 1) * k["grid_deg"]
        # swept half-height is 25 deg -> 50 deg tall, less ~2*(margin - dilate)
        want = 50.0 - 2.0 * (margin - 2.0)
        assert abs(height - want) <= 4.0, f"margin {margin}: kept height {height}, wanted ~{want}"


def test_an_open_sweep_encloses_nothing_and_says_so():
    """A trace that does not close has no interior. Failing is correct; failing SILENTLY is not."""
    cam, thigh = loop(turns=0.5)
    k = cw._knee_grid(cam, thigh, 1.0, 2.0, 3.0, paths=[(cam, thigh)], close_region=True)
    assert k["enclosed_cells"] == 0
    why = workspace._region_warning(k, 3.0, 2.0, True)
    assert "does not CLOSE" in why and "come back to where you started" in why


def test_a_sparse_trace_still_closes_because_the_path_is_joined():
    """Two consecutive samples can be many cells apart when the leg is swung briskly. Marking only
    the sample cells leaves a dotted outline, and one gap wider than the dilation lets the interior
    fill leak straight out."""
    cam, thigh = loop(n=26)                      # ~7 cells between neighbouring samples
    joined = cw._knee_grid(cam, thigh, 1.0, 1.0, 1.0, paths=[(cam, thigh)], close_region=True)
    dotted = cw._knee_grid(cam, thigh, 1.0, 1.0, 1.0, paths=None, close_region=True)
    assert joined["safe_cells"] > 0.8 * enclosed_area(), "the joined path did not close"
    assert dotted["safe_cells"] < 0.2 * joined["safe_cells"], (
        "this test only means something if the dotted version actually leaks")


def test_separate_takes_are_not_joined_to_each_other():
    """Lifting the leg between passes must not draw a line across the middle of the region."""
    a = (np.array([90.0, 95.0]), np.array([0.0, 0.0]))
    b = (np.array([110.0, 115.0]), np.array([40.0, 40.0]))
    k = cw._knee_grid(np.r_[a[0], b[0]], np.r_[a[1], b[1]], 1.0, 0.0, 0.0,
                      paths=[a, b], close_region=False)
    assert k["raw_cells"] <= 14, f"{k['raw_cells']} cells — the two takes were bridged"


def test_the_margin_is_applied_at_the_array_edge_too():
    """_binary_erode shifts zeros in from beyond the array, so it does not eat inward from the
    border. Without the internal pad, a region touching the edge of its own grid silently kept no
    margin there -- and the grid is built to hug the sweep, so it touches by construction."""
    cam, thigh = loop()
    k = cw._knee_grid(cam, thigh, 1.0, 2.0, 5.0, paths=[(cam, thigh)])
    g = k["safe_grid"]
    assert not g[0, :].any() and not g[-1, :].any(), "kept cells sit on the cam edge of the grid"
    assert not g[:, 0].any() and not g[:, -1].any(), "kept cells sit on the thigh edge of the grid"


# ===================================================================== through the store
@pytest.fixture
def store(tmp_path):
    ws = workspace.WorkspaceStore()
    ws.legs = {}
    ws._active_path = str(tmp_path / "active.npz")
    return ws


def test_process_segments_fills_what_was_traced(store):
    cam, thigh = loop()
    seg = np.stack([np.linspace(-20, 20, len(cam)), cam, thigh], axis=1)
    warn = store.process_segments("left", [seg])
    assert not warn, warn
    kept = int(store.legs["left"]["knee_grid"].sum())
    assert kept > 0.85 * enclosed_area(), f"{kept} cells kept of ~{enclosed_area():.0f}"


def test_an_empty_result_explains_which_knob_to_turn(store):
    """"region is EMPTY" on its own sent people back to re-sweep when the fix was a number in a
    box. Each cause gets its own sentence."""
    cam, thigh = loop(ry=6.0)                     # a thin loop the margin will eat
    seg = np.stack([np.linspace(-20, 20, len(cam)), cam, thigh], axis=1)

    warn = store.process_segments("left", [seg], margin_deg=12.0)
    assert "12 deg margin eats more than" in warn, warn

    # a WIDE loop, so the trace itself stays a thin line the margin can eat: without closing,
    # the swept line is all there is and 3 deg of margin is thicker than it
    wc, wt = loop()
    wide = np.stack([np.linspace(-20, 20, len(wc)), wc, wt], axis=1)
    warn = store.process_segments("left", [wide], margin_deg=3.0, close_region=False)
    assert "region-closing is off" in warn, warn

    half = np.stack([np.linspace(-20, 20, 600)], axis=1)
    c2, t2 = loop(turns=0.5, n=600)
    warn = store.process_segments("left", [np.stack([half[:, 0], c2, t2], axis=1)])
    assert "does not CLOSE" in warn, warn
