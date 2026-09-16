"""Derive geometry that the sim-ready model needs, straight from the CAD OPEN_B export.

Everything here is computed at the rest pose (qpos = 0), where the user says the cam/pushrod
linkage is already aligned. Re-run this if the CAD is regenerated; `build_model.py` imports it.

Provides:
  loop_anchors() -> per side, the pushrod distal tip (pushrod-local) and the coincident
                    anchor on the shin (leg-local) -> the two <site>s of the closed-loop <connect>.
  foot_boxes()   -> per side, an axis-aligned box (pos+half-size, foot-body-local) approximating
                    the foot for ground collision.
"""
import numpy as np
import mujoco

OPEN_B = "robotCADdescription/MJCF_OPEN_MUJOCO_B/dash01/dash01.xml"

_SIDES = {
    "L": dict(pushrod="PushrodLeftNCS-v1",  leg="LegLeftNCS-v1",  foot="FootLeftNCS-v1"),
    "R": dict(pushrod="PushrodRightNCS-v1", leg="LegRightNCS-v1", foot="FootRightNCS-v1"),
}


def _load():
    m = mujoco.MjModel.from_xml_path(OPEN_B)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _bid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def _gid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)


def _mesh_world(m, d, geom_name):
    """World coords of a mesh geom's vertices (self-consistent w/ MuJoCo's recentering)."""
    g = _gid(m, geom_name)
    mid = m.geom_dataid[g]
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    v = m.mesh_vert[a:a + n].reshape(-1, 3)
    return d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T


def _to_body_local(m, d, world_pts, body_name):
    b = _bid(m, body_name)
    return (world_pts - d.xpos[b]) @ d.xmat[b].reshape(3, 3)  # (W-p) @ R == R^T @ (W-p)


def loop_anchors():
    m, d = _load()
    out = {}
    for s, names in _SIDES.items():
        pb = _bid(m, names["pushrod"])
        verts_w = _mesh_world(m, d, names["pushrod"] + "_geom")
        tip_w = verts_w[np.argmax(np.linalg.norm(verts_w - d.xpos[pb], axis=1))]
        out[s] = dict(
            pushrod_site=_to_body_local(m, d, tip_w[None], names["pushrod"])[0],
            leg_site=_to_body_local(m, d, tip_w[None], names["leg"])[0],
        )
    return out


def foot_boxes():
    m, d = _load()
    out = {}
    for s, names in _SIDES.items():
        verts_local = _to_body_local(m, d, _mesh_world(m, d, names["foot"] + "_geom"), names["foot"])
        lo, hi = verts_local.min(0), verts_local.max(0)
        out[s] = dict(pos=(lo + hi) / 2, size=(hi - lo) / 2)
    return out


def foot_tips():
    """The contact 'ball' at the END of each foot: the mesh vertex farthest from the ankle
    (foot body origin), in foot-body-local frame. Only a small sphere here touches the ground."""
    m, d = _load()
    out = {}
    for s, names in _SIDES.items():
        verts_local = _to_body_local(m, d, _mesh_world(m, d, names["foot"] + "_geom"), names["foot"])
        out[s] = verts_local[np.argmax(np.linalg.norm(verts_local, axis=1))]
    return out


def foot_heels():
    """The far (ankle) end of the foot: the mesh vertex FARTHEST from the toe tip, projected onto
    the foot's mid-plane (y=0), in foot-body-local frame. A heel collision sphere here is clear of
    the floor in the nominal toe-down stance (it sits ~28 cm up) but catches the floor if the leg
    folds and the foot flattens — a physical stop so the long foot can't clip through the ground,
    which the toe sphere alone can't prevent."""
    m, d = _load()
    tips = foot_tips()
    out = {}
    for s, names in _SIDES.items():
        verts_local = _to_body_local(m, d, _mesh_world(m, d, names["foot"] + "_geom"), names["foot"])
        heel = verts_local[np.argmax(np.linalg.norm(verts_local - tips[s], axis=1))]
        heel = heel.copy()
        heel[1] = 0.0                                   # onto the foot mid-plane
        out[s] = heel
    return out


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    la, ft = loop_anchors(), foot_tips()
    for s in "LR":
        print(f"[{s}] loop pushrod_site = {la[s]['pushrod_site']}   leg_site = {la[s]['leg_site']}")
    for s in "LR":
        print(f"[{s}] foot tip (toe sphere center) = {ft[s]}")
