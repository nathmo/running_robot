"""The hard-bounds lookup is on the hot path, and how far _safe_room is asked to LOOK.

Reported from the robot 2026-09-23: "control process is unreachable". It was not. The control
process was alive and answering; `POST /api/measure/defaults` had simply become slow enough that
uiproc's 4 s proxy timeout fired, and the 503 the proxy synthesises says "control process
unreachable: <timeout>". The control process logged its own 200 for the same request one second
later.

Two causes, both introduced with the +-180 deg clamp on 2026-09-23:

  1. _widen_to_workspace scanned the grid for its occupied bounding box on EVERY call. That is
     np.flatnonzero over ~25 000 cells, _validate_pose asks for it once per joint, and
     measure_defaults validates MEASURE_ENVELOPE_SAMPLES = 480 poses per request -- 2880 full-grid
     reductions per request. _hard_bounds went 0.2 -> 10.2 us, 45x. The box is cached on the leg
     dict now and recomputed only when the grid is edited.

  2. _safe_room was handed the +-180 clamp as its search reach. It walks that in 2 deg steps with
     a full _validate_pose each step. The clamp is a refusal threshold, not a statement that a
     joint can swing 180 deg, so the callers ask for _nominal_bounds instead -- the same
     correction already made to the travel budget when the clamp was raised.
"""
import time

import numpy as np
import pytest

import paths
import calibration
import daemon as daemon_mod
import workspace


def _grid_store(tmp_path, nc=260, nt=95):
    """A store holding a grid the size of the robot's real one, with a realistic blob in it."""
    ws = workspace.WorkspaceStore()
    ws.legs = {}
    ws._active_path = str(tmp_path / "active.npz")
    ws._persist_active = lambda: None
    g = np.zeros((nc, nt), bool)
    g[40:200, 20:70] = True
    for side in paths.SIDES:
        ws.apply_grid(side, g, cam_origin=-14.0, thigh_origin=-38.0, grid_deg=1.0)
        ws.apply_abduction(side, -51.0, 51.0)
    return ws, g


@pytest.fixture
def dm(tmp_path):
    """A RUNNING daemon: measure_defaults and identify_plan both bail out early unless every drive
    is reporting, and an early bail scans nothing — which would make these tests vacuous."""
    ws, _g = _grid_store(tmp_path)
    cal = calibration.Calibration()
    cal.save = lambda *a, **k: None
    d = daemon_mod.RobotDaemon(mock=True, calib=cal, wstore=ws, fklut=None, bb=None,
                               anchor_file=str(tmp_path / "a.json"))
    d.start()
    assert d._started_ok.wait(5.0)
    for _ in range(300):
        if all(m.pos is not None for m in d.motors):
            break
        time.sleep(0.02)
    assert all(m.pos is not None for m in d.motors), "mock drives never reported"
    yield d
    d.stop_event.set()
    d.join(2.0)


# ===================================================================== the cached bounding box
def test_the_occupied_box_is_computed_when_the_grid_is_set(tmp_path):
    ws, g = _grid_store(tmp_path)
    for side in paths.SIDES:
        assert ws.legs[side]["knee_occupied"] == (40, 199, 20, 69)

    g2 = np.zeros_like(g)
    g2[5:9, 7:11] = True
    ws.apply_grid("left", g2, cam_origin=-14.0, thigh_origin=-38.0, grid_deg=1.0)
    assert ws.legs["left"]["knee_occupied"] == (5, 8, 7, 10), "the cache went stale on an edit"
    assert ws.legs["right"]["knee_occupied"] == (40, 199, 20, 69), "the other leg was disturbed"


def test_an_empty_grid_has_no_box(tmp_path):
    ws, g = _grid_store(tmp_path)
    ws.apply_grid("left", np.zeros_like(g), cam_origin=0.0, thigh_origin=0.0, grid_deg=1.0)
    assert ws.legs["left"]["knee_occupied"] is None
    assert workspace.occupied_box(None) is None
    assert workspace.occupied_box(np.zeros((4, 4), bool)) is None


def test_hard_bounds_does_not_scan_the_grid_on_the_hot_path(dm, monkeypatch):
    """The property that actually matters. A timing assertion would be flaky; this is exact."""
    calls = {"n": 0}
    real = workspace.occupied_box
    monkeypatch.setattr(workspace, "occupied_box",
                        lambda g: (calls.__setitem__("n", calls["n"] + 1), real(g))[1])
    monkeypatch.setattr(daemon_mod.wsmod, "occupied_box", workspace.occupied_box)
    for _ in range(50):
        for side in paths.SIDES:
            for role in paths.ROLES:
                dm._hard_bounds(side, role)
    assert calls["n"] == 0, f"the grid was scanned {calls['n']} times by plain bounds lookups"


def test_the_cache_and_a_fresh_scan_agree(dm, tmp_path):
    """The cache must not be a different answer, only a cheaper one."""
    for side in paths.SIDES:
        cached = {r: dm._hard_bounds(side, r) for r in paths.ROLES}
        box = dm.wstore.legs[side].pop("knee_occupied")          # force the fallback path
        fresh = {r: dm._hard_bounds(side, r) for r in paths.ROLES}
        dm.wstore.legs[side]["knee_occupied"] = box
        assert cached == fresh, side


def test_bounds_stay_cheap_on_a_large_grid(dm, tmp_path):
    """A coarse backstop for the 45x regression, generous enough not to flake on a busy machine."""
    ws, _ = _grid_store(tmp_path, nc=400, nt=400)
    dm.wstore = ws
    n = 4000
    t0 = time.perf_counter()
    for _ in range(n):
        dm._hard_bounds("right", "thigh")
    per_us = (time.perf_counter() - t0) / n * 1e6
    assert per_us < 20.0, (
        f"{per_us:.1f} us per _hard_bounds on a 160k-cell grid — it is scanning the grid again")


# ===================================================================== the search reach
def test_safe_room_is_never_asked_to_search_the_never_exceed_clamp(dm, monkeypatch):
    """_safe_room walks its reach in 2 deg steps with a full _validate_pose each step, so the reach
    is a cost. It must be the plausible travel range, not the +-180 deg refusal threshold."""
    seen = []
    real = daemon_mod.RobotDaemon._safe_room
    monkeypatch.setattr(daemon_mod.RobotDaemon, "_safe_room",
                        lambda self, pose, name, d, reach: (seen.append((name, reach)),
                                                            real(self, pose, name, d, reach))[1])
    cal = dm.calib
    ok, why = cal.set_zero(dm.latest_raw_positions())
    assert ok, why
    for n in paths.MOTOR_NAMES:
        cal.confirm(n)

    dm.measure_defaults(leg="right")
    dm.sine_defaults()                      # the manual-slider presets scan the same way
    dm.identify_plan({"motor": "right.thigh"})
    assert seen, "no search happened at all — this test proves nothing"

    for name, reach in seen:
        side, role = paths.split_name(name)
        n_lo, n_hi = dm._nominal_bounds(side, role)
        assert reach <= (n_hi - n_lo) + 1e-6, (
            f"{name}: asked to search {reach:.0f} deg, wider than its whole nominal range "
            f"[{n_lo:.0f}, {n_hi:.0f}] — the never-exceed clamp leaked into a search reach")
