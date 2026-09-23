"""Run the twin-source JS harness (tests/js/twin_source.js) as part of the Python suite.

Three things can pose the digital twin -- live telemetry, the gait preview, and a policy's DRY RUN
-- and the panel picks between them in the browser. Both ways of getting that order wrong are
silent: live telemetry during a rehearsal is a still picture of a limp robot, and a leftover
rehearsal pose is a robot that appears to be walking with nothing energised. The harness pulls the
real twinAngles() out of static/app.js and checks the precedence and the fallback.

Skipped where node is not installed; it is not a build dependency of the robot.
"""
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "js", "twin_source.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_twin_draws_the_dry_run_over_live_telemetry_but_never_over_the_preview():
    r = subprocess.run([shutil.which("node"), HARNESS], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"twin source harness failed:\n{r.stdout}\n{r.stderr}"
    assert "all twin source checks passed" in r.stdout, r.stdout
