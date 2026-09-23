"""The policy-inference panel's HTTP contract.

MockBus only -- no hardware, no CAN, no robot. The bundles used here are synthetic but REAL:
they round-trip through controller/deploy/bundle.py's own save() (which validates before writing),
so the endpoints are exercised with exactly the artifact export_policy.py produces.

    python -m pytest controller/fixed_gait/webui/tests/test_policy_api.py -v
"""
import io
import os
import sys
import time

import numpy as np
import pytest

import paths
from test_blackbox import capture_zero, robot          # noqa: F401  (pytest fixtures)
from test_thermal_api import client                    # noqa: F401  (the wired test client)


def make_bundle(path, **meta_over):
    from bundle import Bundle
    nu, ad, fd, hl = 6, 8, 10, 4
    n = fd * hl
    arrays = dict(
        est_w0=np.zeros((16, n)), est_b0=np.zeros(16),
        est_w1=np.zeros((16, 16)), est_b1=np.zeros(16),
        est_w2=np.zeros((3, 16)), est_b2=np.zeros(3),
        pi_w0=np.zeros((32, n + 3)), pi_b0=np.zeros(32),
        pi_w1=np.zeros((32, 32)), pi_b1=np.zeros(32),
        act_w=np.zeros((ad, 32)), act_b=np.zeros(ad),
        obs_mean=np.zeros(n), obs_var=np.ones(n),
        nominal_ctrl=np.zeros(nu), default_motor_pos=np.zeros(nu),
        ctrl_lo=-np.ones(nu), ctrl_hi=np.ones(nu),
        motor_vel_limit=np.full(nu, 30.0), forcerange=np.full(nu, 60.0),
        imp_kp_base=np.full(nu, 120.0), imp_kd_base=np.full(nu, 2.0),
        imp_leg_ix=np.arange(nu), hist_idx=np.arange(hl),
    )
    meta = dict(nu=nu, action_dim=ad, frame_dim=fd, history_len=hl,
                est_hidden=[16, 16], policy_hidden=[32, 32],
                run="test_run", checkpoint="42M", control_dt=0.005,
                base_lock=[0, 1, 0, 1, 1, 1],
                cmd_v_fwd_trained=1.0, cmd_v_back_trained=0.3, cmd_yaw_trained=0.5,
                gait_cfg={})
    meta.update(meta_over)
    Bundle.save(path, arrays, meta)


@pytest.fixture
def poldir(tmp_path, monkeypatch):
    """An isolated bundle search path so the tests never see (or leave) real bundles.

    BOTH directories are redirected: the panel offers data/policies/ and deploy/bundles/ together,
    so isolating only one leaves the operator's real 5 MB checkpoints in every listing assertion."""
    d = tmp_path / "policies"
    d.mkdir()
    empty = tmp_path / "bundles"
    empty.mkdir()
    monkeypatch.setattr(paths, "POLICY_DIR", str(d))
    monkeypatch.setattr(paths, "BUNDLE_DIR", str(empty))
    return str(d)


# ===================================================================== listing
def test_no_bundles_is_an_empty_list_not_an_error(client, poldir):
    c, _d = client
    j = c.get("/api/policy/list").get_json()
    assert j["ok"] is True and j["bundles"] == []


def test_a_valid_bundle_lists_with_its_identity(client, poldir):
    make_bundle(os.path.join(poldir, "test_run_42M.npz"))
    c, _d = client
    j = c.get("/api/policy/list").get_json()
    assert [b["file"] for b in j["bundles"]] == ["test_run_42M.npz"]
    b = j["bundles"][0]
    assert b["valid"] is True and b["run"] == "test_run" and b["hz"] == 200


def test_both_bundle_directories_are_offered(client, poldir, monkeypatch, tmp_path):
    """export_policy.py writes to controller/deploy/bundles/ and the panel's upload writes to
    data/policies/. For a while the panel read only the second, so a freshly exported bundle sitting
    on the robot answered "no bundles in data/policies/" -- with no way to tell from the UI that it
    was looking somewhere else. Both are listed, and each row says where it came from."""
    other = tmp_path / "bundles"
    monkeypatch.setattr(paths, "BUNDLE_DIR", str(other))
    make_bundle(os.path.join(poldir, "uploaded_1M.npz"))
    make_bundle(os.path.join(str(other), "exported_2M.npz"), run="exported")
    c, _d = client
    rows = {b["file"]: b for b in c.get("/api/policy/list").get_json()["bundles"]}
    assert set(rows) == {"uploaded_1M.npz", "exported_2M.npz"}
    assert rows["uploaded_1M.npz"]["where"] == "data/policies"
    assert rows["exported_2M.npz"]["where"] == "deploy/bundles"
    # and a NAME from either directory resolves for the endpoints that take one
    j = c.post("/api/policy/info", json={"file": "exported_2M.npz"}).get_json()
    assert j["ok"] is True and j["info"]["run"] == "exported"


def test_a_foreign_npz_is_listed_as_invalid_with_its_error(client, poldir):
    """'The bundle I scp'd is not offered' must diagnose itself from the panel, so an unloadable
    file is listed with its error rather than silently skipped."""
    np.savez(os.path.join(poldir, "not_a_policy.npz"), x=np.zeros(3))
    c, _d = client
    j = c.get("/api/policy/list").get_json()
    b = j["bundles"][0]
    assert b["valid"] is False and b["error"]


# ===================================================================== info / preflight
def test_info_reports_the_architecture_the_bundle_carries(client, poldir):
    make_bundle(os.path.join(poldir, "b.npz"))
    c, _d = client
    j = c.post("/api/policy/info", json={"file": "b.npz"}).get_json()
    assert j["ok"] is True
    i = j["info"]
    assert i["obs_dim"] == 40
    assert i["estimator"] == [40, 16, 16, 3]
    assert i["policy"] == [43, 32, 32, 8]
    assert i["control_hz"] == 200
    # base_lock rails Y/roll/pitch/yaw in the synthetic bundle -- the one warning that matters
    assert any("RAILED" in w for w in j["warnings"])
    assert "run_policy.py" in j["command"] and "b.npz" in j["command"]


def test_info_preflight_names_what_blocks_a_real_run(client, poldir):
    make_bundle(os.path.join(poldir, "b.npz"))
    c, _d = client
    j = c.post("/api/policy/info", json={"file": "b.npz"}).get_json()
    pf = {chk["name"]: chk for chk in j["preflight"]}
    for name in ("zeroing", "joint map", "thermal model", "IMU mount"):
        assert name in pf, "preflight lost the {} gate".format(name)
    assert pf["zeroing"]["ok"] is True                 # the client fixture captured a zero
    assert pf["IMU mount"]["ok"] is False              # no Sense HAT in a test process
    assert pf["IMU mount"]["why"]


def test_info_offers_a_slider_for_a_joystick_bundle_and_says_what_it_spans(client, poldir):
    """The panel shows ONE command control, chosen here. A joystick bundle must come back as a
    speed in m/s with the range the policy was trained over -- offering a RUN / STOP pair for it
    would be a control that cannot reach the policy, and offering a slider without the scale is a
    control that means something different from what it says."""
    import sys
    sys.path.insert(0, os.path.join(paths.DEPLOY, "tests"))
    import v2_fixture
    v2_fixture.write_v2_bundle(os.path.join(poldir, "joy.npz"),
                               objective="joystick", v_max=2.5, v_min=0.0)
    c, _d = client
    j = c.post("/api/policy/info", json={"file": "joy.npz"}).get_json()
    assert j["ok"] is True
    i = j["info"]
    assert i["command_kind"] == "speed" and i["v_max"] == 2.5 and i["v_min"] == 0.0
    assert i["has_run_flag"] is False
    assert any("SPEED" in w and "walking in place" in w for w in j["warnings"])
    assert any("SLEWED" in w for w in j["warnings"]), "the slider's rate limit must be stated"
    assert not any("RUN / STOP" in w for w in j["warnings"])
    assert "/tmp/dash_command" in j["command"] and "echo 1.0" in j["command"]


def test_info_says_when_the_bottom_of_the_slider_was_never_commanded(client, poldir):
    """The command curriculum widens the draw band DOWNWARD from the warm-start parent's one
    speed. A checkpoint taken mid-ramp still gets a slider that reaches 0 — and 0 is then a speed
    it has never been asked for, which is exactly the kind of thing that has to be on screen
    before the robot is on the floor."""
    import sys
    sys.path.insert(0, os.path.join(paths.DEPLOY, "tests"))
    import v2_fixture
    v2_fixture.write_v2_bundle(os.path.join(poldir, "mid.npz"), objective="joystick", v_max=2.5,
                               env_params_at_checkpoint={"cmd_lo": 0.8, "cmd_hi": 1.0})
    c, _d = client
    j = c.post("/api/policy/info", json={"file": "mid.npz"}).get_json()
    assert j["info"]["v_trained"] == [2.0, 2.5]
    assert any("MID-CURRICULUM" in w and "never seen a command for" in w for w in j["warnings"])


def test_a_speed_command_is_refused_when_nothing_is_running(client, poldir):
    c, _d = client
    r = c.post("/api/policy/command", json={"speed": 1.0})
    assert r.status_code == 409 and "no policy run" in r.get_json()["error"]


def test_a_command_with_neither_shape_says_which_two_there_are(client, poldir):
    c, _d = client
    r = c.post("/api/policy/command", json={})
    assert r.status_code == 400
    err = r.get_json()["error"]
    assert "speed" in err and "run" in err


def test_info_is_jailed_to_the_policy_dir(client, poldir, tmp_path):
    outside = tmp_path / "outside.npz"
    make_bundle(str(outside))
    c, _d = client
    r = c.post("/api/policy/info", json={"file": "../outside.npz"})
    assert r.status_code == 404


# ===================================================================== upload
def test_upload_validates_before_anything_lands_on_disk(client, poldir):
    c, _d = client
    r = c.post("/api/policy/upload", data={"file": (io.BytesIO(b"junk"), "evil.npz")})
    assert r.status_code == 400
    assert os.listdir(poldir) == []


def test_upload_accepts_a_real_bundle(client, poldir, tmp_path):
    src = tmp_path / "up.npz"
    make_bundle(str(src))
    c, _d = client
    with open(src, "rb") as f:
        r = c.post("/api/policy/upload", data={"file": (io.BytesIO(f.read()), "up.npz")})
    assert r.status_code == 200 and r.get_json()["file"] == "up.npz"
    assert os.path.exists(os.path.join(poldir, "up.npz"))


# ===================================================================== rehearsal
def test_rehearse_refuses_a_missing_bundle(client, poldir):
    c, _d = client
    r = c.post("/api/policy/rehearse", json={"file": "ghost.npz"})
    assert r.status_code == 404


def test_rehearse_status_is_quiet_when_nothing_ran(client, poldir):
    c, _d = client
    j = c.get("/api/policy/rehearse/status").get_json()
    assert j["ok"] is True and j["rehearsal"] is None


_JOY3 = os.path.join(paths.BUNDLE_DIR, "dash_joy3_lr_s2_180M.npz")


@pytest.mark.skipif(not os.path.exists(_JOY3), reason="the dash_joy3 bundle is not in deploy/bundles/")
def test_a_rehearsal_lifts_the_hardware_guards_and_says_so(tmp_path):
    """The panel's Rehearse button runs run_policy.py --mock with NO acknowledgement flags. Until
    2026-09-22 that died in setup on the placeholder thermal fit, for every bundle. A dry run
    energises nothing, so it lifts those guards itself, runs the control law, and ends with a
    summary that names what it bypassed -- a rehearsal that passed is not a robot that is ready."""
    import subprocess
    out = subprocess.run([sys.executable, os.path.join(paths.DEPLOY, "run_policy.py"), "--bundle", _JOY3,
                          "--mock", "--max-seconds", "2", "--no-log", "--speed", "1.0"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "DRY RUN PASSED" in out.stdout
    assert "BYPASSED for the dry run" in out.stdout and "thermal model: placeholder" in out.stdout
    assert "ticks of the control law" in out.stdout and "never reached the policy" not in out.stdout


# ===================================================================== the dry run's live pose
def test_the_pose_endpoint_is_quiet_when_no_rehearsal_is_running(client, poldir):
    """No process, no pose. The twin falls back to live telemetry on exactly this answer."""
    c, _d = client
    j = c.get("/api/policy/rehearse/pose").get_json()
    assert j["ok"] is True and j["pose"] is None


class _FakeProc:
    """A rehearsal that is 'running' without one: poll() is the only thing the endpoint asks."""

    def __init__(self, rc=None):
        self.rc = rc

    def poll(self):
        return self.rc


def test_a_running_rehearsal_serves_the_pose_the_runner_published(client, poldir, tmp_path):
    """run_policy.py --pose-file writes normalized degrees per motor; the endpoint hands them over
    with an age, and the panel feeds them to the digital twin. The age is what stops a frozen twin
    claiming a dry run is still moving."""
    import json as _json
    import server

    pose = tmp_path / "rehearsal_pose.json"
    rec = {"t": 1.25, "phase": "RUN", "ticks": 250, "mock": True,
           "norm_deg": {n: 3.5 for n in paths.MOTOR_NAMES}}
    pose.write_text(_json.dumps(rec), encoding="utf-8")
    server._REHEARSAL.update(proc=_FakeProc(), file="b.npz", pose=str(pose))
    try:
        c, _d = client
        p = c.get("/api/policy/rehearse/pose").get_json()["pose"]
        assert p["phase"] == "RUN" and p["file"] == "b.npz"
        assert sorted(p["norm_deg"]) == sorted(paths.MOTOR_NAMES)
        assert p["age_s"] < 5.0

        # the same file, once the process is gone: nothing to draw
        server._REHEARSAL.update(proc=_FakeProc(rc=0))
        assert c.get("/api/policy/rehearse/pose").get_json()["pose"] is None
    finally:
        server._REHEARSAL.update(proc=None, file=None, pose=None)


def test_a_half_written_pose_is_no_pose_rather_than_a_500(client, poldir, tmp_path):
    """The runner writes to a temp name and renames, so a reader sees one record or the other --
    but a truncated file from a killed run must not take the endpoint down with it."""
    import server

    pose = tmp_path / "rehearsal_pose.json"
    pose.write_text('{"phase": "RU', encoding="utf-8")
    server._REHEARSAL.update(proc=_FakeProc(), file="b.npz", pose=str(pose))
    try:
        c, _d = client
        j = c.get("/api/policy/rehearse/pose").get_json()
        assert j["ok"] is True and j["pose"] is None
    finally:
        server._REHEARSAL.update(proc=None, file=None, pose=None)


@pytest.mark.skipif(not os.path.exists(_JOY3), reason="the dash_joy3 bundle is not in deploy/bundles/")
def test_a_rehearsal_publishes_a_pose_that_actually_moves(tmp_path):
    """End to end: --pose-file makes a dry run watchable. A mock bus swallows every frame, so
    without this the only evidence a rehearsal produces is a log tail -- and the digital twin, the
    one thing that CAN show a policy moving with nothing energised, sits still.

    The claim is that the pose CHANGES while the run is up: a file written once would draw a frozen
    robot, which looks exactly like a working one."""
    import json as _json
    import subprocess

    pose = tmp_path / "pose.json"
    proc = subprocess.Popen([sys.executable, os.path.join(paths.DEPLOY, "run_policy.py"),
                             "--bundle", _JOY3, "--mock", "--max-seconds", "4", "--no-log",
                             "--speed", "1.0", "--pose-file", str(pose)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    seen = []
    t_end = time.time() + 300.0
    while proc.poll() is None and time.time() < t_end:
        try:
            rec = _json.loads(pose.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            time.sleep(0.02)
            continue
        if not seen or rec["norm_deg"] != seen[-1]["norm_deg"]:
            seen.append(rec)
        time.sleep(0.02)
    out = proc.communicate()[0]
    assert proc.returncode == 0, out[-2000:]
    assert len(seen) >= 2, "the pose never changed: {}".format(seen)
    assert sorted(seen[-1]["norm_deg"]) == sorted(paths.MOTOR_NAMES)
    assert {r["phase"] for r in seen} <= {"APPROACH", "RUN"}
    assert any(r["phase"] == "RUN" for r in seen), "never published a pose from the policy itself"


@pytest.mark.skipif(not os.path.exists(_JOY3), reason="the dash_joy3 bundle is not in deploy/bundles/")
def test_a_finished_rehearsal_takes_its_pose_away(tmp_path):
    """A pose left behind is a viewer claiming a run that has ended -- the twin would keep drawing
    the last commanded stance as if the policy were still in it.

    Nothing reads the file while the run is up here, deliberately: on Windows a reader holding the
    file open makes the runner's own remove() fail, which is a property of the test's polling and
    not of the product (on the Pi it is not a race at all). The server guards on the process too."""
    import subprocess

    pose = tmp_path / "pose.json"
    out = subprocess.run([sys.executable, os.path.join(paths.DEPLOY, "run_policy.py"),
                          "--bundle", _JOY3, "--mock", "--max-seconds", "2", "--no-log",
                          "--pose-file", str(pose)],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert not pose.exists(), "the runner left its pose behind after the run ended"
    assert not (tmp_path / "pose.json.tmp").exists(), "left its scratch file behind"
