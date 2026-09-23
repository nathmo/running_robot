"""Run the workspace-grid JS harness (tests/js/ws_grid.js) as part of the Python suite.

The grid is the safety object the daemon consults on every guided move, and half of it lives in
the browser: the editor grows, crops and renumbers the array before posting it back. Growing the
canvas shifts every index, so a mistake there does not throw -- it silently moves the safe region
onto a different part of the joint. The harness pulls the real functions out of static/app.js and
checks that a painted cell keeps its ANGLE across a grow, a crop and a mid-stroke resize.

Skipped where node is not installed; it is not a build dependency of the robot.
"""
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "js", "ws_grid.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_workspace_grid_editor_keeps_every_cell_at_its_own_angle():
    r = subprocess.run([shutil.which("node"), HARNESS], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"grid harness failed:\n{r.stdout}\n{r.stderr}"
    assert "all grid checks passed" in r.stdout, r.stdout
