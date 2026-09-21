"""The digital twin's sign map (twinmap.py, /api/twin/map) and the assets it is drawn from.

The default signs are DERIVED from the published MJCF's joint axes plus the calibration wizard's
physical convention (thigh + swings forward, cam + moves the crank down, abduction + moves the foot
outward, both legs alike). A CAD re-export that flips an axis would silently mirror a joint in the
twin, so the derivation is re-done here from static/twin/*.xml with a plain serial FK.

MockBus only -- no hardware.

    python -m pytest controller/fixed_gait/webui/tests/test_twinmap.py -v
"""
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import twinmap
from test_blackbox import capture_zero, robot           # noqa: F401  (pytest fixtures)
from test_thermal_api import client                     # noqa: F401  (the wired test client)

TWIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "twin")
TWIN_XML = os.path.join(TWIN_DIR, "SpiderBotInitPos.xml")


# ---------------------------------------------------------------- map file
def test_missing_file_gives_the_defaults(tmp_path):
    m = twinmap.load(str(tmp_path / "none.json"))
    assert m["signs"] == twinmap.DEFAULT_SIGNS
    assert m["defaults"] == twinmap.DEFAULT_SIGNS


def test_save_merges_and_persists(tmp_path):
    p = str(tmp_path / "twin_map.json")
    twinmap.save({"left.cam": -1}, p)
    m = twinmap.save({"right.abd": 1}, p)
    assert m["signs"]["left.cam"] == -1 and m["signs"]["right.abd"] == 1
    assert m["signs"]["left.thigh"] == twinmap.DEFAULT_SIGNS["left.thigh"]
    assert json.load(open(p))["signs"]["left.cam"] == -1


@pytest.mark.parametrize("bad", [{"left.knee": 1}, {"left.cam": 2}, {"left.cam": 0}, {"left.cam": "x"}])
def test_save_refuses_anything_but_a_known_motor_and_pm1(tmp_path, bad):
    p = str(tmp_path / "twin_map.json")
    with pytest.raises(ValueError):
        twinmap.save(bad, p)
    assert not os.path.exists(p), "a refused write must not leave a file behind"


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    p = tmp_path / "twin_map.json"
    p.write_text("{not json")
    assert twinmap.load(str(p))["signs"] == twinmap.DEFAULT_SIGNS


def test_endpoint_round_trip(client, tmp_path, monkeypatch):   # noqa: F811
    c, _d = client
    monkeypatch.setattr(twinmap, "TWIN_MAP_FILE", str(tmp_path / "twin_map.json"))
    assert c.get("/api/twin/map").get_json()["signs"] == twinmap.DEFAULT_SIGNS
    r = c.post("/api/twin/map", json={"signs": {"right.thigh": -1}})
    assert r.status_code == 200 and r.get_json()["signs"]["right.thigh"] == -1
    r = c.post("/api/twin/map", json={"signs": {"right.thigh": 3}})
    assert r.status_code == 400 and r.get_json()["ok"] is False
    assert c.get("/api/twin/map").get_json()["signs"]["right.thigh"] == -1


# ---------------------------------------------------------------- published assets
def _mjcf():
    if not os.path.exists(TWIN_XML):
        pytest.skip("static/twin not built (tools/build_twin.py)")
    return ET.parse(TWIN_XML).getroot()


def test_every_mesh_the_mjcf_names_is_published():
    root = _mjcf()
    for m in root.iter("mesh"):
        assert os.path.exists(os.path.join(TWIN_DIR, m.get("file"))), m.get("file")


def _rot(axis, q):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def _euler(e):          # MuJoCo default eulerseq "xyz", intrinsic
    return _rot([1, 0, 0], e[0]) @ _rot([0, 1, 0], e[1]) @ _rot([0, 0, 1], e[2])


def _chain_point(root, names, q, point):
    """World position of `point` (in the last body's frame) down the named serial chain, each
    body's hinge at q[name] (default 0). Plain serial FK: the loop is not closed, so this is only
    used for the joint's OWN child point, which the loop does not constrain at first order."""
    R, p, el = np.eye(3), np.zeros(3), root.find("worldbody")
    for n in names:
        el = next(b for b in el.findall("body") if b.get("name") == n)
        p = p + R @ np.array([float(v) for v in el.get("pos", "0 0 0").split()])
        R = R @ _euler([float(v) for v in el.get("euler", "0 0 0").split()])
        j = el.find("joint")
        if j is not None and q.get(n):
            jp = np.array([float(v) for v in j.get("pos", "0 0 0").split()])
            Rj = _rot([float(v) for v in j.get("axis").split()], q[n])
            p = p + R @ (jp - Rj @ jp)
            R = R @ Rj
    return p + R @ np.asarray(point, float)


@pytest.mark.parametrize("side", ["left", "right"])
def test_default_signs_match_the_wizard_convention(side):
    root = _mjcf()
    S, s, out = side.capitalize(), twinmap.DEFAULT_SIGNS, (1 if side == "left" else -1)
    hip, cam, thigh = f"Hip{S}NCS-v1", f"Cam{S}NCS-v1", f"Thigh{S}NCS-v1"
    knee = [0, 0, -0.35]                                   # thigh -> foot body, thigh frame
    crank = [0.12, 0, 0]                                   # cam -> pushrod pin, cam frame
    dq = np.radians(5)

    def moved(names, motor, body, pt):
        a = _chain_point(root, names, {}, pt)
        b = _chain_point(root, names, {body: s[f"{side}.{motor}"] * dq}, pt)
        return b - a

    d = moved(["bodyNCS-v1", hip, thigh], "thigh", thigh, knee)
    assert d[0] > 0.02, f"{side}.thigh +: knee must swing FORWARD (+x), moved {d}"
    d = moved(["bodyNCS-v1", hip, cam], "cam", cam, crank)
    assert d[2] < -0.005, f"{side}.cam +: crank pin must move DOWN, moved {d}"
    d = moved(["bodyNCS-v1", hip, thigh], "abd", hip, knee)
    assert out * d[1] > 0.02, f"{side}.abd +: foot must move OUTWARD, moved {d}"
