"""The policy-inference panel's HTTP contract.

MockBus only -- no hardware, no CAN, no robot. The bundles used here are synthetic but REAL:
they round-trip through robot/deploy/bundle.py's own save() (which validates before writing),
so the endpoints are exercised with exactly the artifact export_policy.py produces.

    python -m pytest robot/fixed_gait/webui/tests/test_policy_api.py -v
"""
import io
import os

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
    """export_policy.py writes to robot/deploy/bundles/ and the panel's upload writes to
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
