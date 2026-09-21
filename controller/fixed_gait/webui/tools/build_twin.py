#!/usr/bin/env python3
"""Copy the homing-pose MJCF into the web UI as the browser-side digital twin (static/twin/).

    python controller/fixed_gait/webui/tools/build_twin.py            # dev machine, then scp static/twin/

The browser (static/twin3d.js) parses the MJCF itself, so the XML is copied VERBATIM: what the
panel draws is this file, not a transcription of it. Only the meshes change. The torso STL is 134k
triangles / 6.7 MB, which a Pi 3B on its own Wi-Fi AP serves in seconds and a phone then has to
parse, so any mesh over --max-faces is vertex-clustered (numpy only, no decimation library): snap
vertices to a grid, merge, drop the triangles that collapsed. The grid is refined until the mesh
fits the budget. Winding is kept, so the recomputed face normals point the same way as the CAD's.

Default source is dash-01CAD/homing/SpiderBotInitPos: the pose in which the drives are zeroed, so
MJCF qpos = 0 IS normalized 0 deg on every motor and the twin needs no offsets, only signs.
"""
import argparse
import os
import shutil
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEBUI = os.path.dirname(HERE)
REPO_ROOT = os.path.abspath(os.path.join(WEBUI, "..", "..", ".."))
DEFAULT_SRC = os.path.join(REPO_ROOT, "dash-01CAD", "homing", "SpiderBotInitPos",
                           "SpiderBotInitPos.xml")
DEFAULT_OUT = os.path.join(WEBUI, "static", "twin")


def read_stl(path):
    with open(path, "rb") as f:
        d = f.read()
    n = struct.unpack("<I", d[80:84])[0]
    if 84 + 50 * n != len(d):
        sys.exit(f"{path}: not a binary STL (ASCII STL is not supported)")
    rec = np.frombuffer(d[84:], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3, 3).astype(np.float64)


def write_stl(path, tris):
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    nrm = np.cross(e1, e2)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(tris), dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    rec["n"] = nrm
    rec["v"] = tris.reshape(-1, 9)
    with open(path, "wb") as f:
        f.write(b"build_twin.py".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        f.write(rec.tobytes())


def cluster(tris, cell):
    """Vertex clustering on a `cell`-sized grid (mesh units). Returns the surviving triangles."""
    v = tris.reshape(-1, 3)
    key = np.floor(v / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    # representative vertex per cell = mean of its members (keeps the surface where it was)
    cnt = np.bincount(inv)
    rep = np.stack([np.bincount(inv, weights=v[:, k]) / cnt for k in range(3)], axis=1)
    f = inv.reshape(-1, 3)
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[keep]
    f = np.unique(np.sort(f, axis=1), axis=0, return_index=True)[1]    # duplicate faces
    g = inv.reshape(-1, 3)[keep][np.sort(f)]
    return rep[g]


def simplify(tris, max_faces):
    if len(tris) <= max_faces:
        return tris, None
    ext = float(np.ptp(tris.reshape(-1, 3), axis=0).max())
    cell = ext / 400.0
    out = cluster(tris, cell)
    while len(out) > max_faces:
        cell *= 1.25
        out = cluster(tris, cell)
    return out, cell


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default=DEFAULT_SRC, help="MJCF to publish (meshes resolved beside it)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-faces", type=int, default=30000, help="per-mesh triangle budget")
    a = ap.parse_args()

    src_dir = os.path.dirname(os.path.abspath(a.src))
    root = ET.parse(a.src).getroot()
    comp = root.find("compiler")
    meshdir = comp.get("meshdir", "") if comp is not None else ""
    os.makedirs(a.out, exist_ok=True)
    shutil.copyfile(a.src, os.path.join(a.out, os.path.basename(a.src)))
    total_in = total_out = 0
    for m in root.iter("mesh"):
        rel = os.path.join(meshdir, m.get("file"))
        tris = read_stl(os.path.join(src_dir, rel))
        small, cell = simplify(tris, a.max_faces)
        dst = os.path.join(a.out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        write_stl(dst, small)
        total_in += len(tris)
        total_out += len(small)
        note = f"clustered at {cell:.2f} (mesh units)" if cell else "copied"
        print(f"{m.get('name'):24s} {len(tris):7d} -> {len(small):6d} faces  {note}")
    print(f"total {total_in} -> {total_out} faces; wrote {a.out}")
    print("copy to the Pi:  scp -r controller/fixed_gait/webui/static/twin "
          "nemo@<pi>:running_robot/controller/fixed_gait/webui/static/")


if __name__ == "__main__":
    main()
