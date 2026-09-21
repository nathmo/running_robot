"""IMU mount frame from the upright capture + four-tilt sequence (mountcal.py), the sequence's
auto-capture state machine and the noise recorder (sensehat.py, mock IMU).

    python -m pytest controller/fixed_gait/webui/tests/test_mountcal.py -v
"""
import math
import os
import time

import numpy as np
import pytest

import imunoise
import mountcal
import paths
import sensehat

R_TRUE = sensehat.MOCK_R                 # the mock's chip -> body rotation, the answer to find


@pytest.fixture(autouse=True)
def _tmp_data(tmp_path, monkeypatch):
    """Never touch the real data/sensehat_mount.json or data/imu_noise/."""
    monkeypatch.setattr(mountcal, "MOUNT_FILE", str(tmp_path / "sensehat_mount.json"))
    monkeypatch.setattr(paths, "DATA", str(tmp_path))
    return tmp_path


def _acc(pitch_deg, roll_deg, R=R_TRUE):
    """What a still chip reads with the body pitched forward / rolled right by these angles."""
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    up = np.array([-math.sin(p) * math.cos(r), math.sin(r), math.cos(p) * math.cos(r)])
    return R.T @ up


TILT_POSE = {"fwd": (12, 0), "back": (-12, 0), "left": (0, -12), "right": (0, 12)}


def _calibrated(tilts=("fwd", "left", "right", "back"), pose=TILT_POSE):
    m = mountcal.MountCal()
    m.set_level(_acc(0, 0), {})
    for k in tilts:
        ok, why = m.set_tilt(k, _acc(*pose[k]), {})
        assert ok, why
    return m


def test_the_four_tilts_recover_the_mount():
    m = _calibrated()
    assert m.calibrated and not m.conflict
    assert np.allclose(m.R, R_TRUE, atol=1e-9)
    s = m.snapshot()
    assert s["axes"]["fwd"][0] == "+y" and s["axes"]["left"][0] == "+x" and s["axes"]["up"][0] == "-z"
    assert s["check"]["pairs"]["fore_aft"]["disagree_deg"] == pytest.approx(0, abs=1e-6)
    assert s["check"]["right_angle_deg"] == pytest.approx(0, abs=1e-6)
    assert max(s["check"]["residual_deg"].values()) < 1e-6


@pytest.mark.parametrize("only", [("fwd",), ("back",), ("left",), ("right",)])
def test_any_single_tilt_already_fixes_the_heading(only):
    assert np.allclose(_calibrated(only).R, R_TRUE, atol=1e-9)


def test_a_sloppy_tilt_is_averaged_not_trusted_alone():
    pose = dict(TILT_POSE, fwd=(12, 3))              # 3 deg of unintended lean on the forward tilt
    m = _calibrated(pose=pose)
    err = mountcal.angle_between(m.R[0], R_TRUE[0])
    alone = mountcal.angle_between(_calibrated(("fwd",), pose).R[0], R_TRUE[0])
    assert err < alone / 2
    assert m.snapshot()["check"]["pairs"]["fore_aft"]["disagree_deg"] > 5


def test_flipping_one_pair_is_a_mirror_and_is_reported_not_averaged():
    m = _calibrated()
    m.set_flip("fore_aft", True)
    assert m.conflict
    assert m.snapshot()["check"]["used"] == ["fore_aft"]
    yaw180 = np.diag([-1.0, -1.0, 1.0])
    assert np.allclose(m.R, yaw180 @ R_TRUE, atol=1e-9)    # still a rotation, fore-aft wins
    m.set_flip("lateral", True)                           # the robot's front was the other end
    assert not m.conflict
    assert np.allclose(m.R, yaw180 @ R_TRUE, atol=1e-9)
    m.set_flip("fore_aft", False)
    m.set_flip("lateral", False)
    assert np.allclose(m.R, R_TRUE, atol=1e-9)


def test_a_pair_tilted_the_same_way_twice_is_left_out():
    pose = dict(TILT_POSE, left=TILT_POSE["right"])       # tilted right when asked for left
    m = _calibrated(pose=pose)
    pair = m.snapshot()["check"]["pairs"]["lateral"]
    assert pair["excluded"] and pair["disagree_deg"] > 170
    assert np.allclose(m.R, R_TRUE, atol=1e-9)            # the fore-aft pair carries the frame


def test_every_rotation_stays_proper():
    rng = np.random.default_rng(3)
    for _ in range(20):
        m = _calibrated(pose={k: (p + rng.normal(0, 2), r + rng.normal(0, 2))
                              for k, (p, r) in TILT_POSE.items()})
        for fa in (False, True):
            for lat in (False, True):
                m.set_flip("fore_aft", fa)
                m.set_flip("lateral", lat)
                assert np.linalg.det(m.R) == pytest.approx(1.0, abs=1e-9)


def test_the_old_single_tilt_file_still_loads(tmp_path):
    # the Pi's file as it was: one 4 deg nose-down capture, chip -X forward
    old = {"up_chip": [-0.0033, -0.0199, -0.9998], "fwd_chip": [-0.9979, -0.0434, -0.0484],
           "fwd_declared": None, "lever_cad": None, "lever_use": "cad",
           "captures": {"level": {"n": 300}, "forward": {"tilt_deg": 3.99, "weak": True}}}
    p = tmp_path / "old.json"
    p.write_text(__import__("json").dumps(old))
    m = mountcal.MountCal.load_or_new(str(p))
    assert m.calibrated
    assert mountcal.nearest_chip_axis(m.R[0])[0] == "-x"
    assert m.tilts["fwd"]["legacy"]
    R_old = mountcal.rotation_from(old["up_chip"], old["fwd_chip"])
    assert np.allclose(m.R, R_old, atol=1e-9)
    m.save()                                               # round trip in the new format
    again = mountcal.MountCal.load_or_new()
    assert np.allclose(again.R, R_old, atol=1e-9)


def test_too_shallow_a_tilt_is_refused():
    m = mountcal.MountCal()
    m.set_level(_acc(0, 0), {})
    ok, why = m.set_tilt("fwd", _acc(2, 0), {})
    assert not ok and "tilt" in why
    ok, why = m.set_tilt("fwd", _acc(5, 0), {})
    assert ok and "Redo" in why                            # accepted, flagged weak


# ---------------------------------------------------------------- the live sequence (mock IMU)
@pytest.fixture
def hat():
    h = sensehat.SenseHat(mock=True, mount=mountcal.MountCal(), imu_hz=200)
    h.start()
    t_end = time.time() + 5
    while not h.snapshot().get("available") and time.time() < t_end:
        time.sleep(0.02)
    yield h
    h.stop()
    h.join(2)


def _wait(pred, timeout=10.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_the_sequence_captures_all_four_by_itself(hat):
    assert not hat.seq_start()["ok"]                        # needs the upright reference first
    assert hat.start_capture("level")["ok"]
    assert _wait(lambda: hat.capture_status["state"] == "ok")
    assert hat.seq_start()["ok"]
    for k in mountcal.TILTS:
        assert _wait(lambda: hat.seq_status.get("phase") == "tilt" and hat.seq_status["which"] == k), \
            hat.seq_status
        hat.mock_pose(k)
        assert _wait(lambda: k in hat.mount.tilts), hat.seq_status
        hat.mock_pose("still")                              # back upright arms the next step
    assert _wait(lambda: hat.seq_status["state"] == "done"), hat.seq_status
    assert hat.mount.calibrated and not hat.mount.conflict
    assert mountcal.angle_between(hat.mount.R[0], R_TRUE[0]) < 2.0
    assert mountcal.angle_between(hat.mount.R[1], R_TRUE[1]) < 2.0


def test_the_sequence_waits_for_upright_between_tilts(hat):
    hat.start_capture("level")
    assert _wait(lambda: hat.capture_status["state"] == "ok")
    hat.seq_start()
    assert _wait(lambda: hat.seq_status.get("phase") == "tilt")
    hat.mock_pose("fwd")
    assert _wait(lambda: "fwd" in hat.mount.tilts)
    time.sleep(1.0)                                         # still tilted forward: nothing else
    assert set(hat.mount.tilts) == {"fwd"}
    assert hat.seq_status["which"] == "left" and hat.seq_status["phase"] == "upright"
    assert hat.seq_cancel()["ok"] and not hat.seq_status["active"]


def test_the_noise_recorder_analyses_and_saves(hat, _tmp_data):
    assert hat.noise_start()["ok"]
    time.sleep(1.5)
    r = hat.noise_stop()
    assert r["ok"], r
    res = r["result"]
    assert res["n"] > 150 and res["frame"] == "chip"         # mount not calibrated in this test
    assert len(res["acc_rms_mg"]) == 3 and "pitch" in res
    assert os.path.exists(os.path.join(str(_tmp_data), "imu_noise", res["file"]))
    assert _wait(lambda: (hat.snapshot().get("noise_result") or {}).get("n") == res["n"])


def test_analyze_flags_a_moving_record():
    rng = np.random.default_rng(0)
    t = np.arange(2000) / 200.0
    acc = rng.normal(0, 0.001, (2000, 3)) + [0, 0, 1]
    gyr = rng.normal(0, 0.05, (2000, 3))
    assert imunoise.analyze(acc, gyr, t)["still"]
    gyr[:, 0] += 5 * np.sin(2 * np.pi * 0.5 * t)             # a slow sway
    assert not imunoise.analyze(acc, gyr, t)["still"]
