"""Read link inertials out of the DASH-01 MuJoCo model — PURE numpy + ElementTree (no mujoco).

The web server compares these CAD "given" values against the identified ones, and it runs on the Pi
without mujoco, so this parser cannot depend on it. Each body's <inertial> is read in that body's
own frame (mass, CoM pos, and the full symmetric tensor), supporting both MuJoCo `fullinertia`
(ixx iyy izz ixy ixz iyz) and `diaginertia` + optional principal-axis `quat`.
"""
import xml.etree.ElementTree as ET

import numpy as np

from . import frames


def _quat_to_R(q):
    """MuJoCo quat (w x y z) -> rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _parse_inertial(el):
    """<inertial> element -> {mass, com[3], inertia{ixx..iyz}} in the body frame."""
    mass = float(el.get("mass"))
    com = [float(v) for v in (el.get("pos") or "0 0 0").split()]
    if el.get("fullinertia"):
        ixx, iyy, izz, ixy, ixz, iyz = (float(v) for v in el.get("fullinertia").split())
        M = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    else:
        di = [float(v) for v in el.get("diaginertia").split()]
        R = _quat_to_R(el.get("quat", "1 0 0 0").split())
        M = R @ np.diag(di) @ R.T
    return {"mass": mass, "com": com, "inertia": frames.to_dict(M)}


def read_bodies(xml_path):
    """{body_name: {mass, com, inertia}} for every body that carries an explicit <inertial>."""
    root = ET.parse(xml_path).getroot()
    out = {}
    for body in root.iter("body"):
        name = body.get("name")
        inertial = body.find("inertial")            # direct child only
        if name and inertial is not None:
            out[name] = _parse_inertial(inertial)
    return out
