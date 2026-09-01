#!/usr/bin/env python3
"""DASH-01 web control interface — run this ON THE ROBOT (Raspberry Pi), browse from the hotspot.

    python fixed_gait/webui/server.py                 # real robot (socketcan can0/can1)
    python fixed_gait/webui/server.py --mock          # simulated motors (UI dev on any machine)

Serves everything locally (no internet needed): telemetry + strip charts, the zero/direction
calibration wizard, manual control + sine, the safe-workspace viewer/editor, gait record /
hand-draw / playback, the EE-space linkage animation, and a big E-STOP.

Design: one RobotDaemon thread owns the CAN buses (daemon.py); Flask handlers only post requests
to it and read snapshots — they NEVER touch the buses, except the dead-daemon e-stop fallback.
"""
import argparse
import atexit
import io
import json
import os
import secrets
import subprocess
import sys
import time

# MEASURED on the robot's Pi 3B, 2026-09-01, with the imp_m3d bundle:
#
#     OPENBLAS_NUM_THREADS=1   controller.step p50 4.53 ms
#     OPENBLAS_NUM_THREADS=2                       4.75 ms
#     OPENBLAS_NUM_THREADS=4 (the default, = cores) 5.05 ms
#
# The policy's largest matrix is 593x256. That is far too small to pay for thread synchronisation,
# so OpenBLAS's default of one thread per core makes the control law SLOWER while also putting
# three extra runnable threads in front of the 200 Hz CAN loop and the 200 Hz IMU thread on a
# four-core machine. Pinned to one.
#
# This must run before numpy is imported: OpenBLAS reads the variable when the shared library is
# loaded, and by the time `import numpy` returns it is too late.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

import paths
import blackbox
import calibration
import canio
import daemon as daemon_mod
import dynstore
import fklut
import gaitstore
import measurestore
import thermalstore
import thermal_excite
import sensehat
import workspace

# pure-numpy identification helpers (safe to import on the Pi; the heavy estimator is imported
# lazily inside /api/identify/run only when its mujoco/scipy deps are actually present)
from identification import frames, model_inertials, paramio

IDENT_PARAMS_FILE = os.path.join(paths.IDENT_DIR, "identified_params.json")
MESH_DIR = paths.MESH_DIR

app = Flask(__name__, static_folder="static", static_url_path="/static")

STATE = {
    "daemon": None,          # RobotDaemon
    "calib": None,           # calibration.Calibration
    "wstore": None,          # workspace.WorkspaceStore
    "fk": None,              # fklut.FkLut
    "dyn": None,             # dynstore.DynConfig (weighed masses, drive PID, Kt)
    "sense": None,           # sensehat.SenseHat (I2C poll thread; None if --no-sensors)
    "bb": None,              # blackbox.BlackBox (flight recorder; writer thread)
    "interface": "socketcan",
    "mock": False,
    "ctl": {"token": None, "ts": 0.0},
}
CTL_TIMEOUT_S = 15.0


# ===================================================================== helpers
def _dm():
    return STATE["daemon"]


def _since_arg():
    """The ring cursor from ?since=, tolerating the value a browser sends before it has one.

    A fresh page sends `since=undefined` on its first poll of each stream, which int() raised on --
    a 500 and a traceback in the journal on every single page load, for a request whose correct
    answer is obviously "start from the beginning".
    """
    try:
        return int(request.args.get("since", 0))
    except (TypeError, ValueError):
        return 0


def _ok(**extra):
    return jsonify({"ok": True, "state": _dm().get_snapshot(), **extra})


def _err(msg, code=400):
    return jsonify({"ok": False, "error": str(msg), "state": _dm().get_snapshot()}), code


def _require_calibrated():
    if not STATE["calib"].complete:
        return "calibration incomplete — finish the zero/direction wizard first"
    return None


def _acquire_control(payload):
    """Single-controller guard for motion endpoints: holder refreshes on use; a stale holder
    (no motion request for CTL_TIMEOUT_S) or an explicit takeover releases the claim."""
    ctl = STATE["ctl"]
    token = (payload or {}).get("token") or request.headers.get("X-Ctl-Token")
    now = time.time()
    expired = (now - ctl["ts"]) > CTL_TIMEOUT_S
    if ctl["token"] is None or expired or token == ctl["token"] or (payload or {}).get("takeover"):
        if token != ctl["token"]:
            ctl["token"] = token or secrets.token_hex(8)
        ctl["ts"] = now
        return ctl["token"], None
    return None, "another client is controlling the robot (send takeover:true to take control)"


def _estop_fallback():
    """Daemon thread dead: open throwaway buses and stream zero current ourselves."""
    try:
        buses = canio.open_buses(STATE["interface"], sorted(set(paths.SIDE_CHANNEL.values())),
                                 mock=STATE["mock"])
        t_end = time.time() + 0.3
        while time.time() < t_end:
            for bus in buses.values():
                for cid in paths.ROLE_ID.values():
                    canio.set_current(bus, cid, 0.0)
            time.sleep(0.01)
        for bus in buses.values():
            bus.shutdown()
        return True
    except Exception as e:
        print(f"!! e-stop fallback failed: {e}")
        return False


# ===================================================================== static
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ===================================================================== state / telemetry
@app.get("/api/state")
def api_state():
    d = _dm()
    fk = STATE["fk"]
    if not fk.available:
        fk.try_reload()                 # hot-load a LUT scp'd in after server start
    snap = d.get_snapshot()
    snap["calibration"] = STATE["calib"].snapshot()
    snap["workspace"] = {"legs": sorted(STATE["wstore"].legs.keys()),
                         "source": STATE["wstore"].source,
                         "files": STATE["wstore"].list_files()}
    snap["trajectories"] = gaitstore.list_files()
    snap["measurements"] = measurestore.list_summaries()
    snap["dynamics"] = STATE["dyn"].snapshot()
    snap["identified"] = os.path.exists(IDENT_PARAMS_FILE)
    snap["fk"] = {"available": fk.available,
                  "verified": dict(fk.model_map["verified"]) if fk.available else {},
                  "model_map": {s: fk.model_map[s] for s in paths.SIDES} if fk.available else {}}
    snap["daemon_thread_alive"] = d.is_alive()
    return jsonify(snap)


@app.get("/api/state_hot")
def api_state_hot():
    """The part of /api/state that is FREE: live daemon state, already in memory.

    Split out because /api/state was the last thing stalling the CAN loop after the process split.
    It is polled ~1/s and every call ran measurestore.list_summaries(), which opens and reads the
    metadata of EVERY saved run (18 files) off the SD card, plus two more directory listings —
    disk I/O in the process with the 200 Hz deadline. None of that changes unless a human does
    something. So: hot here, cold below, and uiproc.py caches the cold half.
    /api/state is left exactly as it was; running server.py alone is still the rollback."""
    d = _dm()
    snap = d.get_snapshot()
    snap["daemon_thread_alive"] = d.is_alive()
    return jsonify(snap)


@app.get("/api/state_cold")
def api_state_cold():
    """The disk-backed half of /api/state. Only changes when a human saves, deletes or calibrates —
    every one of which is a non-GET request, so uiproc.py can drop its cache on any proxied write
    and never serve a stale list."""
    fk = STATE["fk"]
    if not fk.available:
        fk.try_reload()                 # hot-load a LUT scp'd in after server start
    return jsonify({
        "calibration": STATE["calib"].snapshot(),
        "workspace": {"legs": sorted(STATE["wstore"].legs.keys()),
                      "source": STATE["wstore"].source,
                      "files": STATE["wstore"].list_files()},
        "trajectories": gaitstore.list_files(),
        "measurements": measurestore.list_summaries(),
        "dynamics": STATE["dyn"].snapshot(),
        "identified": os.path.exists(IDENT_PARAMS_FILE),
        "fk": {"available": fk.available,
               "verified": dict(fk.model_map["verified"]) if fk.available else {},
               "model_map": {s: fk.model_map[s] for s in paths.SIDES} if fk.available else {}},
    })


@app.get("/api/telemetry")
def api_telemetry():
    d = _dm()
    since = _since_arg()
    seq, t, data = d.ring.read_since(since)
    out = {"seq": seq, "t": np.round(t, 3).tolist(),
           "motors": {}}
    for i, n in enumerate(paths.MOTOR_NAMES):
        out["motors"][n] = {
            "pos_norm": _nan_list(data["pos_norm"][:, i]),
            "pos_raw": _nan_list(data["pos_raw"][:, i]),
            "cmd_norm": _nan_list(data["cmd_norm"][:, i]),
            "cur": _nan_list(data["cur"][:, i]),
            "temp": _nan_list(data["temp"][:, i]),
            "spd": _nan_list(data["spd"][:, i]),
        }
    # live linkage pose per side (FK LUT, verified sides only)
    fk = STATE["fk"]
    snap = d.get_snapshot()
    linkage = {}
    for side in paths.SIDES:
        mm = snap.get("motors", {})
        cam = (mm.get(f"{side}.cam") or {}).get("pos_norm")
        thigh = (mm.get(f"{side}.thigh") or {}).get("pos_norm")
        if fk.available and fk.side_verified(side) and cam is not None and thigh is not None:
            nodes, valid = fk.interp_nodes(side, cam, thigh)
            linkage[side] = {"nodes": nodes, "valid": valid, "cam": cam, "thigh": thigh}
        else:
            linkage[side] = None
    out["linkage"] = linkage
    out["mode"] = snap.get("mode")
    return jsonify(out)


def _nan_list(a, nd=2):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


# ===================================================================== the streaming transport
# WHY. Measured on the robot 2026-09-01, idle and LIMP: the 200 Hz loop holds 193 Hz with nothing
# polling and 107 Hz with a browser open, and the cost tracks REQUEST COUNT rather than payload --
# roughly 1 Hz of control rate per request per second, whichever endpoint it is. The 2026-08-19
# process split moved the JSON work out (uiproc.py) but left the shape: three polling endpoints,
# ~22 upstream requests a second, each paying for a socket accept, an HTTP parse, a Werkzeug
# request thread and a response assembly INSIDE the interpreter that owes a CAN frame every 5 ms.
#
# So collapse them. One connection, opened once by uiproc.py, over which this process pushes a
# frame every 50 ms. The per-request overhead stops scaling with the browser's poll rate because
# there are no more per-poll requests: the browser polls uiproc, which answers from memory.
#
# FRAMING is deliberately not JSON for the bulk: a length-prefixed header (JSON, small) followed by
# raw .tobytes() of the arrays. np.save would add a header per array and json would put every float
# through the interpreter, which is the per-element cost the split exists to avoid.
#
#     4 bytes big-endian header length | header JSON | payload
#
# ROLLBACK: /api/telemetry_raw, /api/sensors_raw and /api/state_hot are untouched and still work,
# and uiproc falls back to them the moment the stream drops. Running server.py alone on :8080
# remains exactly today's robot.
STREAM_HZ = 20.0
STREAM_MAX_S = 3600.0                    # a client that stops reading must not pin a thread forever


def _telemetry_frame(since):
    """(header_dict, [arrays]) for the telemetry half. Shared with /api/telemetry_raw so there is
    one definition of what a telemetry sample IS."""
    d = _dm()
    seq, t, data = d.ring.read_since(since)
    snap = d.get_snapshot()
    fk = STATE["fk"]
    linkage = {}
    for side in paths.SIDES:
        mm = snap.get("motors", {})
        cam = (mm.get(f"{side}.cam") or {}).get("pos_norm")
        thigh = (mm.get(f"{side}.thigh") or {}).get("pos_norm")
        if fk.available and fk.side_verified(side) and cam is not None and thigh is not None:
            nodes, valid = fk.interp_nodes(side, cam, thigh)
            linkage[side] = {"nodes": nodes, "valid": valid, "cam": cam, "thigh": thigh}
        else:
            linkage[side] = None
    arr = np.stack([data[f] for f in d.ring.FIELDS])
    return ({"seq": seq, "fields": list(d.ring.FIELDS), "mode": str(snap.get("mode")),
             "linkage": linkage, "n": int(t.size), "shape": list(arr.shape)},
            [np.ascontiguousarray(t, np.float64), np.ascontiguousarray(arr, np.float32)])


def _sensors_frame(since):
    sh = STATE["sense"]
    if sh is None:
        return ({"seq": 0, "fields": [], "n": 0, "shape": [0],
                 "meta": {"available": False, "error": "sensors disabled (--no-sensors)"}},
                [np.zeros(0), np.zeros(0, np.float32)])
    meta = sh.snapshot() or {"available": False, "error": sh.error}
    seq, t, data = sh.ring.read_since(since)
    arr = (np.stack([data[f] for f in sh.ring.fields]) if sh.ring.fields
           else np.zeros((0, 0), np.float32))
    return ({"seq": seq, "fields": list(sh.ring.fields), "meta": meta,
             "n": int(np.asarray(t).size), "shape": list(arr.shape)},
            [np.ascontiguousarray(t, np.float64), np.ascontiguousarray(arr, np.float32)])


def _pack_frame(header, blobs):
    head = json.dumps(header).encode("utf-8")
    return b"".join([len(head).to_bytes(4, "big"), head] + [b.tobytes() for b in blobs])


@app.get("/api/stream")
def api_stream():
    """One frame every 50 ms until the client goes away. Replaces the browser's whole poll loop."""
    try:
        tel_since = int(request.args.get("since", 0))
        sen_since = int(request.args.get("sensors_since", 0))
    except (TypeError, ValueError):
        tel_since = sen_since = 0

    def gen(tel_since=tel_since, sen_since=sen_since):
        t_end = time.time() + STREAM_MAX_S
        period = 1.0 / STREAM_HZ
        next_t = time.time()
        while time.time() < t_end:
            th, tb = _telemetry_frame(tel_since)
            sh_, sb = _sensors_frame(sen_since)
            tel_since, sen_since = th["seq"], sh_["seq"]
            snap = _dm().get_snapshot()
            snap["daemon_thread_alive"] = _dm().is_alive()
            yield _pack_frame({"tel": th, "sen": sh_, "state_hot": snap}, tb + sb)
            next_t += period
            # sleep releases the GIL, which is the entire point of doing this on a timer rather
            # than as fast as the socket will take it
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()

    return Response(gen(), mimetype="application/octet-stream",
                    headers={"X-Stream-Hz": str(STREAM_HZ),
                             "Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/telemetry_raw")
def api_telemetry_raw():
    """The SAME data as /api/telemetry, as an .npz of raw arrays instead of JSON.

    WHY THIS EXISTS. /api/telemetry costs this process 36 per-element Python loops (6 fields x 6
    motors, via _nan_list) plus jsonify, and it is served from the process that also runs the
    200 Hz CAN thread. Measured on the robot 2026-08-19 with candump: the motors' own status frames
    arrive at 200.1 Hz with 0.04 ms of jitter -- the SPI/CAN path is clean -- while the Pi's own
    SET_CURRENT stream managed only 156.6 Hz, dropping 21.6% of its slots, with 6.4 ms sd and 72 ms
    worst case. Gaps over 10 ms occurred 9.16 times a second against a browser poll rate of 9.1/s:
    one stall per HTTP request. The UI was eating a fifth of the control loop through the GIL.
    Serialising here is what costs; reading the ring does not. So this endpoint does the one thing
    that must happen in this process (copy the ring under its lock) and hands the bytes to uiproc.py
    to turn into JSON in a DIFFERENT interpreter, with a different GIL.
    np.savez writes at C level, so the per-element work leaves this process entirely.
    /api/telemetry stays exactly as it was: running server.py alone on :8080 still works, which is
    the rollback."""
    d = _dm()
    since = _since_arg()
    seq, t, data = d.ring.read_since(since)
    snap = d.get_snapshot()
    # linkage stays HERE: it is two LUT interpolations, not a per-element loop, and it needs the
    # calibration + FK state that only this process owns.
    fk = STATE["fk"]
    linkage = {}
    for side in paths.SIDES:
        mm = snap.get("motors", {})
        cam = (mm.get(f"{side}.cam") or {}).get("pos_norm")
        thigh = (mm.get(f"{side}.thigh") or {}).get("pos_norm")
        if fk.available and fk.side_verified(side) and cam is not None and thigh is not None:
            nodes, valid = fk.interp_nodes(side, cam, thigh)
            linkage[side] = {"nodes": nodes, "valid": valid, "cam": cam, "thigh": thigh}
        else:
            linkage[side] = None
    # FLAT, not np.savez. savez builds a ZIP container -- 9 entries, each with its own header, on
    # both write and read -- and measured at ~30 ms round trip it was a large part of why the proxy
    # hop cost 300 ms. Two bare np.save calls into one buffer are a header and a memcpy each, and
    # np.load reads them back sequentially from the same stream. Field order travels in X-Fields so
    # the two sides cannot silently disagree.
    buf = io.BytesIO()
    np.save(buf, t)
    np.save(buf, np.stack([data[f] for f in d.ring.FIELDS]))
    return Response(buf.getvalue(), mimetype="application/octet-stream",
                    headers={"X-Seq": str(seq), "X-Mode": str(snap.get("mode")),
                             "X-Fields": json.dumps(list(d.ring.FIELDS)),
                             "X-Linkage": json.dumps(linkage)})


@app.get("/api/sensors_raw")
def api_sensors_raw():
    """Sense HAT counterpart of /api/telemetry_raw — same reasoning, same contract."""
    sh = STATE["sense"]
    if sh is None:
        return Response(b"", mimetype="application/octet-stream",
                        headers={"X-Seq": "0", "X-Meta": json.dumps(
                            {"available": False, "error": "sensors disabled (--no-sensors)"})})
    meta = sh.snapshot() or {"available": False, "error": sh.error}
    since = _since_arg()
    seq, t, data = sh.ring.read_since(since)
    buf = io.BytesIO()
    np.save(buf, t)
    np.save(buf, np.stack([data[f] for f in sh.ring.fields]) if sh.ring.fields else np.zeros(0))
    return Response(buf.getvalue(), mimetype="application/octet-stream",
                    headers={"X-Seq": str(seq), "X-Fields": json.dumps(list(sh.ring.fields)),
                             "X-Meta": json.dumps(meta)})


# ===================================================================== Sense HAT (B) sensors
@app.get("/api/sensors")
def api_sensors():
    """Latest Sense HAT values + new ring samples since `since` (same seq contract as telemetry).

    Always 200, even with no HAT: the panel renders the reason from `available`/`error` rather than
    the client having to treat a missing board as a failed request."""
    sh = STATE["sense"]
    if sh is None:
        return jsonify({"available": False, "error": "sensors disabled (--no-sensors)", "seq": 0,
                        "t": [], "series": {}})
    out = sh.snapshot() or {"available": False, "error": sh.error}
    since = _since_arg()
    seq, t, data = sh.ring.read_since(since)
    out["seq"] = seq
    out["t"] = np.round(t, 3).tolist()
    out["series"] = {f: _nan_list(data[f], 3) for f in sh.ring.fields}
    return jsonify(out)


def _sense():
    sh = STATE["sense"]
    return (sh, None) if sh is not None else (None, "sensors disabled (--no-sensors)")


@app.post("/api/sensors/capture")
def api_sensors_capture():
    """Start a still-robot average: `gyro` (zero-rate bias), `level` (the upright reference on the
    rig) or `forward` (the nose-down tilt that pins the fore-aft axis). The robot must hold still."""
    sh, why = _sense()
    if why:
        return _err(why)
    b = request.get_json(force=True, silent=True) or {}
    r = sh.start_capture(b.get("kind", "gyro"))
    return _ok() if r.get("ok") else _err(r.get("error", "capture failed"))


@app.post("/api/sensors/mount")
def api_sensors_mount():
    """Edit the parts of the mount calibration that are typed rather than measured: the declared
    forward axis, the CAD lever arm, and which lever source feeds the live compensation."""
    sh, why = _sense()
    if why:
        return _err(why)
    b = request.get_json(force=True, silent=True) or {}
    m = sh.mount
    if "forward_axis" in b:
        ok, why = m.set_declared(b["forward_axis"])
        if not ok:
            return _err(why)
    if "lever_cad" in b:
        ok, why = m.set_lever_cad(b["lever_cad"])
        if not ok:
            return _err(why)
    if "lever_use" in b:
        ok, why = m.set_lever_use(b["lever_use"])
        if not ok:
            return _err(why)
    return _ok(mount=m.snapshot())


@app.post("/api/sensors/mount/reset")
def api_sensors_mount_reset():
    """Forget the measured mount rotation (and the fitted lever) — values go back to chip axes."""
    sh, why = _sense()
    if why:
        return _err(why)
    sh.mount.reset()
    return _ok(mount=sh.mount.snapshot())


@app.post("/api/sensors/lever")
def api_sensors_lever():
    """`start` begins recording a rocking excitation, `stop` fits the lever arm from it."""
    sh, why = _sense()
    if why:
        return _err(why)
    b = request.get_json(force=True, silent=True) or {}
    action = b.get("action", "start")
    if action == "start":
        r = sh.lever_start()
    elif action == "stop":
        r = sh.lever_stop()
    else:
        return _err(f"unknown lever action '{action}'")
    return _ok(mount=sh.mount.snapshot(), fit=r.get("fit")) if r.get("ok") \
        else _err(r.get("error", "lever-arm fit failed"))


# ===================================================================== e-stop / mode
@app.post("/api/estop")
def api_estop():
    d = _dm()
    d.estop("user e-stop (web)")
    if not d.is_alive():
        ok = _estop_fallback()
        return jsonify({"ok": ok, "fallback": True,
                        "error": None if ok else "daemon dead and fallback failed",
                        "state": d.get_snapshot()})
    t_end = time.time() + 0.05
    while time.time() < t_end and d.get_snapshot().get("mode") != "ESTOPPED":
        time.sleep(0.005)
    return _ok()


@app.post("/api/estop/clear")
def api_estop_clear():
    _dm().clear_estop()
    time.sleep(0.02)
    return _ok()


@app.post("/api/mode")
def api_mode():
    body = request.get_json(force=True, silent=True) or {}
    mode = str(body.get("mode", "limp")).upper()
    if mode not in ("LIMP", "MANUAL"):
        return _err(f"cannot request mode {mode} directly")
    if mode == "MANUAL":
        why = _require_calibrated()
        if why:
            return _err(why, 403)
    _dm().request_mode(mode)
    time.sleep(0.02)
    return _ok()


# ===================================================================== calibration wizard
@app.get("/api/calibration")
def api_calibration_get():
    return jsonify(STATE["calib"].snapshot())


def _invalidate_fk_map():
    """A re-zero moves the frame every downstream map is expressed in. fklut's cam/thigh
    offsets were fitted against the OLD zero, so leaving them marked verified leaves a map
    describing a frame that no longer exists — which is exactly what was found on
    2026-08-29: a map dated 4 August, still flagged verified, with offsets never fitted.
    JointMap.invalidate() already states this rule; fklut had no hook to enforce it."""
    fk = STATE["fk"]
    mm = getattr(fk, "model_map", None)        # test stubs / degraded boots have no map at all
    if mm and any(mm.get("verified", {}).values()):
        for side in paths.SIDES:
            mm["verified"][side] = False
        try:
            fk.save_map()
        except Exception:                      # a stale flag in memory is what matters
            pass


@app.post("/api/calibration/zero")
def api_calibration_zero():
    d = _dm()
    if d.get_snapshot().get("mode") not in ("LIMP", "ESTOPPED"):
        return _err("go LIMP before setting zero")
    ok, why = STATE["calib"].set_zero(d.latest_raw_positions())
    if ok:
        _invalidate_fk_map()
    return _ok() if ok else _err(why)


@app.post("/api/calibration/zero_one")
def api_calibration_zero_one():
    """Re-capture ONE joint's zero from wherever it is right now (direction-check step).

    Same gates as the full capture: motors limp, position reported. It shares the FK-map
    invalidation too — moving one joint's zero moves the frame the fitted offsets live in just
    as surely as moving all six."""
    d = _dm()
    if d.get_snapshot().get("mode") not in ("LIMP", "ESTOPPED"):
        return _err("go LIMP before setting zero")
    b = request.get_json(force=True, silent=True) or {}
    name = b.get("motor", "")
    ok, why = STATE["calib"].set_zero_one(name, d.latest_raw_positions().get(name))
    if ok:
        _invalidate_fk_map()
    return _ok() if ok else _err(why)


@app.post("/api/calibration/sign")
def api_calibration_sign():
    b = request.get_json(force=True, silent=True) or {}
    ok, why = STATE["calib"].set_sign(b.get("motor", ""), int(b.get("sign", 1)))
    return _ok() if ok else _err(why)


@app.post("/api/calibration/confirm")
def api_calibration_confirm():
    b = request.get_json(force=True, silent=True) or {}
    ok, why = STATE["calib"].confirm(b.get("motor", ""))
    return _ok() if ok else _err(why)


@app.post("/api/calibration/complete")
def api_calibration_complete():
    ok, why = STATE["calib"].complete_now()
    return _ok() if ok else _err(why)


@app.post("/api/calibration/reset")
def api_calibration_reset():
    STATE["calib"].reset()
    _dm().request_mode("LIMP")
    return _ok()


# ===================================================================== manual + sine
@app.post("/api/manual")
def api_manual():
    why = _require_calibrated()
    if why:
        return _err(why, 403)
    b = request.get_json(force=True, silent=True) or {}
    token, err = _acquire_control(b)
    if err:
        return _err(err, 409)
    ok, why = _dm().manual_update(targets=b.get("targets"), override=b.get("override"),
                                  slew_dps=b.get("slew_dps"))
    return _ok(token=token) if ok else _err(why)


@app.post("/api/manual/release")
def api_manual_release():
    _dm().request_mode("LIMP")
    time.sleep(0.02)
    return _ok()


@app.post("/api/manual/home")
def api_manual_home():
    why = _require_calibrated()
    if why:
        return _err(why, 403)
    b = request.get_json(force=True, silent=True) or {}
    token, err = _acquire_control(b)
    if err:
        return _err(err, 409)
    ok, why = _dm().home(slew_dps=b.get("slew_dps"))
    return _ok(token=token) if ok else _err(why)


@app.post("/api/manual/center")
def api_manual_center():
    """Slew both legs to the pose with the most room around it (the inscribed-square centre of the
    safe workspace). Returns the room each joint has there, which is the excitation amplitude the
    system-ID panel can use without being refused."""
    why = _require_calibrated()
    if why:
        return _err(why, 403)
    b = request.get_json(force=True, silent=True) or {}
    token, err = _acquire_control(b)
    if err:
        return _err(err, 409)
    targets, info, why = _dm().center(slew_dps=b.get("slew_dps"))
    if why:
        return _err(why)
    return _ok(token=token, targets=targets, legs=info)


@app.post("/api/manual/sine_defaults")
def api_sine_defaults():
    why = _require_calibrated()
    if why:
        return _err(why, 403)
    b = request.get_json(force=True, silent=True) or {}
    out, err = _dm().sine_defaults(frac=float(b.get("frac", 0.7)))
    if out is None:
        return _err(err)
    return _ok(defaults=out)


@app.post("/api/sine")
def api_sine():
    why = _require_calibrated()
    if why:
        return _err(why, 403)
    b = request.get_json(force=True, silent=True) or {}
    token, err = _acquire_control(b)
    if err:
        return _err(err, 409)
    ok, why = _dm().sine_update(b.get("actuator", ""), enabled=b.get("enabled"),
                                a=b.get("a_deg"), b=b.get("b_deg"), freq=b.get("freq_hz"))
    return _ok(token=token) if ok else _err(why)


# ===================================================================== workspace
@app.get("/api/workspace")
def api_workspace():
    ws = STATE["wstore"]
    fk = STATE["fk"]
    out = {"source": ws.source, "files": ws.list_files(), "legs": {}}
    for leg in paths.SIDES:
        lj = ws.leg_json(leg)
        if lj is not None and fk.available and fk.side_verified(leg):
            lj["ee_region"] = fk.ee_region(leg, ws.legs[leg])
            lj["ee_zero"] = fk.ee_zero(leg)                  # robot's calibrated zero pose
            lj["ee_model_zero"] = fk.ee_model_zero()         # MJCF/URDF qpos-0 pose (differs!)
        out["legs"][leg] = lj
    return jsonify(out)


@app.post("/api/workspace/grid")
def api_workspace_grid():
    import base64
    b = request.get_json(force=True, silent=True) or {}
    leg = b.get("leg")
    if leg not in paths.SIDES:
        return _err("leg must be right|left")
    try:
        shape = tuple(int(v) for v in b["shape"])
        bits = np.unpackbits(np.frombuffer(base64.b64decode(b["grid_b64"]), np.uint8))
        grid = bits[: shape[0] * shape[1]].reshape(shape).astype(bool)
        STATE["wstore"].apply_grid(leg, grid, float(b["cam_origin"]),
                                   float(b["thigh_origin"]), float(b["res_deg"]))
    except (KeyError, ValueError) as e:
        return _err(f"bad grid payload: {e}")
    if not grid.any():
        return _ok(warning="grid is EMPTY — nothing will pass the workspace check")
    return _ok()


@app.post("/api/workspace/abduction")
def api_workspace_abduction():
    b = request.get_json(force=True, silent=True) or {}
    ok, why = STATE["wstore"].apply_abduction(b.get("leg"), b.get("safe_min"), b.get("safe_max"))
    return _ok() if ok else _err(why)


@app.post("/api/workspace/mirror")
def api_workspace_mirror():
    b = request.get_json(force=True, silent=True) or {}
    src, dst = b.get("from"), b.get("to")
    if src not in paths.SIDES or dst not in paths.SIDES:
        return _err("from/to must be right|left")
    ok, why = STATE["wstore"].mirror(src, dst, flips=b.get("flips"))
    return _ok() if ok else _err(why)


@app.post("/api/workspace/save")
def api_workspace_save():
    b = request.get_json(force=True, silent=True) or {}
    path = STATE["wstore"].save(b.get("name", "workspace"))
    return _ok(saved=path)


@app.post("/api/workspace/delete")
def api_workspace_delete():
    b = request.get_json(force=True, silent=True) or {}
    ok, why = STATE["wstore"].delete_file(b.get("name", ""))
    return _ok() if ok else _err(why, 404)


@app.get("/api/workspace/export")
def api_workspace_export():
    blob, fname = STATE["wstore"].export_bytes(request.args.get("name"))
    return send_file(io.BytesIO(blob), download_name=fname, as_attachment=True,
                     mimetype="application/octet-stream")


@app.post("/api/workspace/import")
def api_workspace_import():
    f = request.files.get("file")
    if f is None:
        return _err("multipart 'file' missing")
    signs = None
    if request.form.get("legacy_signs"):
        signs = json.loads(request.form["legacy_signs"])
    try:
        msg = STATE["wstore"].import_bytes(f.read(), f.filename, legacy_signs=signs)
    except ValueError as e:
        return _err(e)
    return _ok(message=msg)


@app.post("/api/workspace/process")
def api_workspace_process():
    b = request.get_json(force=True, silent=True) or {}
    leg = b.get("leg")
    rec = _dm().get_recording()
    segs = rec["segments"].get(leg or "", [])
    if not segs:
        return _err(f"no recorded workspace segments for leg={leg}")
    warn = STATE["wstore"].process_segments(
        leg, segs, margin_deg=float(b.get("margin_deg", 3.0)),
        grid_deg=float(b.get("grid_deg", 1.0)), dilate_deg=float(b.get("dilate_deg", 2.0)))
    return _ok(warning=warn or None)


# ===================================================================== recording (gait + workspace)
@app.post("/api/record/mode")
def api_record_mode():
    why = _require_calibrated()
    if why:
        return _err(why, 403)
    b = request.get_json(force=True, silent=True) or {}
    kind = b.get("kind")
    if kind not in ("gait", "workspace"):
        return _err("kind must be gait|workspace")
    token, err = _acquire_control(b)
    if err:
        return _err(err, 409)
    _dm().record_command("start_mode", None, kind)
    time.sleep(0.02)
    return _ok(token=token)


@app.post("/api/record/take")
def api_record_take():
    b = request.get_json(force=True, silent=True) or {}
    leg = b.get("leg")
    if leg not in paths.SIDES:
        return _err("leg must be right|left")
    action = b.get("action")
    if action not in ("start", "stop"):
        return _err("action must be start|stop")
    _dm().record_command("take_start" if action == "start" else "take_stop", leg)
    time.sleep(0.02)
    return _ok()


@app.post("/api/record/undo")
def api_record_undo():
    b = request.get_json(force=True, silent=True) or {}
    _dm().record_command("undo", b.get("leg"))
    time.sleep(0.02)
    return _ok()


@app.post("/api/record/center")
def api_record_center():
    b = request.get_json(force=True, silent=True) or {}
    _dm().record_command("center", b.get("leg"))
    time.sleep(0.02)
    return _ok()


@app.post("/api/record/reset")
def api_record_reset():
    _dm().record_command("reset", None)
    time.sleep(0.02)
    return _ok()


@app.post("/api/record/finish")
def api_record_finish():
    b = request.get_json(force=True, silent=True) or {}
    rec = _dm().get_recording()
    try:
        data = gaitstore.finish_recording(
            rec, b.get("name", "gait_web"),
            harmonics=int(b.get("harmonics", 8)), split=float(b.get("split", 0.5)),
            left_phase=float(b.get("left_phase", 0.5)))
    except ValueError as e:
        return _err(e)
    return _ok(trajectory=gaitstore.data_to_json(data, STATE["fk"]))


# ===================================================================== trajectories
@app.get("/api/trajectory")
def api_trajectory():
    name = request.args.get("name")
    if not name:
        return jsonify({"files": gaitstore.list_files()})
    try:
        data = gaitstore.load(name)
    except (ValueError, FileNotFoundError) as e:
        return _err(e, 404 if isinstance(e, FileNotFoundError) else 400)
    return jsonify(gaitstore.data_to_json(data, STATE["fk"]))


@app.post("/api/trajectory/draw")
def api_trajectory_draw():
    b = request.get_json(force=True, silent=True) or {}
    try:
        data = gaitstore.draw_to_trajectory(
            b.get("name", "gait_drawn"), b.get("leg", "right"), b.get("points", []),
            abd_hold=float(b.get("abd_hold", 0.0)), center=b.get("center"),
            harmonics=int(b.get("harmonics", 8)), split=float(b.get("split", 0.5)),
            left_phase=float(b.get("left_phase", 0.5)), reverse=bool(b.get("reverse")))
    except ValueError as e:
        return _err(e)
    return _ok(trajectory=gaitstore.data_to_json(data, STATE["fk"]))


@app.post("/api/trajectory/mirror")
def api_trajectory_mirror():
    b = request.get_json(force=True, silent=True) or {}
    src, dst = b.get("from"), b.get("to")
    if src not in paths.SIDES or dst not in paths.SIDES:
        return _err("from/to must be right|left")
    try:
        data = gaitstore.mirror(b.get("name", ""), src, dst,
                                left_phase=float(b.get("left_phase", 0.5)))
    except (ValueError, FileNotFoundError) as e:
        return _err(e)
    return _ok(trajectory=gaitstore.data_to_json(data, STATE["fk"]))


@app.get("/api/trajectory/export")
def api_trajectory_export():
    try:
        blob, fname = gaitstore.export_bytes(request.args.get("name", ""))
    except (FileNotFoundError, ValueError) as e:
        return _err(e, 404)
    return send_file(io.BytesIO(blob), download_name=fname, as_attachment=True,
                     mimetype="application/octet-stream")


@app.post("/api/trajectory/import")
def api_trajectory_import():
    f = request.files.get("file")
    if f is None:
        return _err("multipart 'file' missing")
    try:
        name = gaitstore.import_bytes(f.read(), f.filename)
    except ValueError as e:
        return _err(e)
    return _ok(imported=name)


@app.post("/api/trajectory/delete")
def api_trajectory_delete():
    b = request.get_json(force=True, silent=True) or {}
    try:
        removed = gaitstore.delete(b.get("name", ""))
    except FileNotFoundError as e:
        return _err(e, 404)
    except ValueError as e:
        return _err(e)
    return _ok(deleted=removed)


# ===================================================================== playback
@app.post("/api/playback/start")
def api_playback_start():
    why = _require_calibrated()
    if why:
        return _err(why, 403)
    b = request.get_json(force=True, silent=True) or {}
    token, err = _acquire_control(b)
    if err:
        return _err(err, 409)
    name = b.get("name")
    if not name:
        return _err("trajectory 'name' missing")
    try:
        data = gaitstore.load(name)
    except (ValueError, FileNotFoundError) as e:
        return _err(e)
    params = {k: b[k] for k in daemon_mod.PLAYBACK_DEFAULTS if k in b}
    if b.get("mode") not in (None, "position", "current"):
        return _err("mode must be position|current")
    _dm().playback_start(data, params)
    time.sleep(0.05)
    return _ok(token=token)


@app.patch("/api/playback")
def api_playback_patch():
    b = request.get_json(force=True, silent=True) or {}
    _dm().playback_patch(b)
    time.sleep(0.02)
    return _ok()


@app.post("/api/playback/stop")
def api_playback_stop():
    _dm().request_mode("LIMP")
    time.sleep(0.02)
    return _ok()


# ===================================================================== system-ID: MEASURE capture
@app.post("/api/measure/defaults")
def api_measure_defaults():
    """Excitation preset for the CURRENT pose: 80% of the safe travel each joint has here, and the
    chirp frequency that puts the resulting sine at 80% of the motor's no-load speed."""
    why = _require_calibrated()
    if why:
        return _err(why, 403)
    b = request.get_json(force=True, silent=True) or {}
    out, err = _dm().measure_defaults(leg=b.get("leg", "right"),
                                      profile=b.get("profile", "dynamic"),
                                      frac=float(b.get("frac", daemon_mod.MEASURE_FRAC)))
    return _ok(defaults=out) if out else _err(err)


@app.post("/api/measure/start")
def api_measure_start():
    why = _require_calibrated()
    if why:
        return _err(why, 403)
    b = request.get_json(force=True, silent=True) or {}
    token, err = _acquire_control(b)
    if err:
        return _err(err, 409)
    ok, why = _dm().measure_start(b.get("spec") or b)
    time.sleep(0.05)
    return _ok(token=token) if ok else _err(why)


@app.post("/api/measure/stop")
def api_measure_stop():
    _dm().measure_stop()
    time.sleep(0.02)
    return _ok()


@app.post("/api/measure/finish")
def api_measure_finish():
    """Save the accumulated high-rate log as a run (npz + json), embedding the calibration + the
    weighed masses / drive PID / Kt in effect, so the run is self-describing for the estimator."""
    b = request.get_json(force=True, silent=True) or {}
    got = _dm().get_measurement()
    if got is None:
        return _err("no measurement in progress — start one first", 404)
    run, meta = got
    if len(run["t"]) < 5:
        _dm().request_mode("LIMP")
        return _err("measurement captured too few samples to save")
    meta = dict(meta,
                calibration=STATE["calib"].snapshot(),
                dynamics=STATE["dyn"].as_dict(),
                model_map={s: STATE["fk"].model_map[s] for s in paths.SIDES}
                          if STATE["fk"].available else None)
    try:
        saved = measurestore.save(b.get("name", "measure"), run, meta)
    except ValueError as e:
        return _err(e)
    _dm().request_mode("LIMP")                       # clears the run buffer in the daemon
    time.sleep(0.02)
    return _ok(saved=saved, measurements=measurestore.list_summaries())


@app.get("/api/measure/export")
def api_measure_export():
    try:
        blob, fname = measurestore.export_bytes(request.args.get("name", ""))
    except (FileNotFoundError, OSError) as e:
        return _err(e, 404)
    return send_file(io.BytesIO(blob), download_name=fname, as_attachment=True,
                     mimetype="application/octet-stream")


# ===================================================================== thermal calibration
# The experiment: saturate ONE motor's torque for 1-30 s, then read how far its temperature rises.
# The operator supplies the temperatures (an external probe is the only instrument that sees the
# right node); the daemon supplies what it actually did -- the measured current integral, which is
# the deposited energy the fit needs. Those two halves arrive at different TIMES, which is why
# starting a burst and recording its peak are separate calls: when the burst ends, the peak has
# not happened yet.
@app.get("/api/thermal/runs")
def api_thermal_runs():
    runs, cools = thermalstore.all_runs()
    return _ok(runs=runs, cooldowns=cools, summary=thermalstore.summary(),
               limits={"max_amps": daemon_mod.THERMAL_MAX_AMPS,
                       "max_duration_s": daemon_mod.THERMAL_MAX_DURATION_S,
                       "min_duration_s": thermal_excite.MIN_DURATION_S,
                       "abort_temp_c": daemon_mod.THERMAL_ABORT_TEMP_C,
                       "free_freq_min_hz": thermal_excite.FREE_SINE_FREQ_MIN_HZ,
                       "free_freq_max_hz": thermal_excite.FREE_SINE_FREQ_MAX_HZ,
                       "free_amp_max_deg": thermal_excite.FREE_SINE_AMP_MAX_DEG},
               motors=list(paths.MOTOR_NAMES))


@app.post("/api/thermal/predict")
def api_thermal_predict():
    """What a proposed burst would do, at both nodes, before it is run.

    This exists because the obvious burst does not work. A handheld probe resolves ~0.5 degC, and
    the case rise is E / (C_w + C_c) -- for a ~1 kg servo, 12 A for 10 s moves it about 0.1 degC.
    Three of those runs and an afternoon later you would have three unusable rows. The panel shows
    the number up front instead, together with the WINDING rise, which is the one that decides
    whether the burst is safe and which no instrument on this robot can see."""
    b = request.get_json(force=True, silent=True) or {}
    motor = b.get("motor", "")
    if motor not in paths.MOTOR_NAMES:
        return _err("unknown motor {!r}".format(motor))
    try:
        amps = float(b.get("amps", 0.0))
        dur = float(b.get("duration_s", 0.0))
    except (TypeError, ValueError):
        return _err("amps and duration_s must be numbers")
    params = _dm()._thermal_params(motor)
    viable, why, pred = thermal_excite.check_burst(params, amps, dur)
    # The verdict is `viable`, NOT `ok`. `ok` belongs to the _ok() envelope, and the shared api()
    # helper in app.js treats a response with ok:false as a FAILED REQUEST -- it banners d.error
    # and throws. Returning the burst verdict in that field made every non-viable burst look like
    # a server error: the prediction readout stopped updating, the banner said "undefined", and
    # Start stayed disabled forever, which is exactly how this shipped broken on 2026-08-28.
    return _ok(prediction=pred, viable=viable, why=why, calibrated=bool(params.calibrated),
               suggestion=dict(zip(("amps", "duration_s"),
                                   thermal_excite.suggest(params, 6.0, amps or None))))


@app.post("/api/thermal/start")
def api_thermal_start():
    why = _require_calibrated()
    if why:
        return _err(why, 403)
    b = request.get_json(force=True, silent=True) or {}
    # An explicit, typed acknowledgement rather than a checkbox that defaults to true: this call
    # energises a motor to saturation, and the one failure mode the software cannot detect is
    # something bolted to the output shaft that should not be there.
    if b.get("rotor_mode") not in ("blocked", "free"):
        return _err("a burst needs the rotor declared: rotor_mode 'blocked' (joint clamped, "
                    "saturated current) or 'free' (motor off the robot, nothing on the shaft — "
                    "tracks a position sine and heats by fighting its own rotor inertia). The "
                    "free-JOINT dither — a leg on a turning shaft — was retracted on 2026-08-29.",
                    400)
    token, err = _acquire_control(b)
    if err:
        return _err(err, 409)
    ok, why = _dm().thermal_start(b)
    time.sleep(0.05)
    return _ok(token=token) if ok else _err(why)


# ===================================================================== joint identification
@app.post("/api/thermal/identify")
def api_thermal_identify():
    """Wiggle ONE joint a few degrees so the operator can see which joint it actually is.

    This is a motion endpoint and carries the same guards as a burst -- calibration, the control
    token, and an explicit confirm_free -- because a joint with something bolted to it does not
    care that the commanded amplitude is small."""
    why = _require_calibrated()
    if why:
        return _err(why, 403)
    b = request.get_json(force=True, silent=True) or {}
    if b.get("confirm_free") is not True:
        return _err("confirm the joint is free to move before wiggling it", 400)
    token, err = _acquire_control(b)
    if err:
        return _err(err, 409)
    ok, why = _dm().identify_start(b)
    time.sleep(0.05)
    return _ok(token=token) if ok else _err(why)


@app.post("/api/bypass")
def api_bypass():
    """Switch one software safety limit off, or back on.

    Turning a limit OFF requires `acknowledged: true` — not as ceremony, but because the UI's
    confirmation is the only place the consequence gets stated, and a bypass set by a stray click
    is the failure mode this feature would otherwise introduce. Turning one back ON never needs it.
    """
    b = request.get_json(force=True, silent=True) or {}
    name, on = b.get("name", ""), bool(b.get("on"))
    if on and b.get("acknowledged") is not True:
        return _err("disabling a safety limit needs an explicit acknowledgement", 400)
    ok, why = _dm().set_bypass(name, on, note=b.get("note", ""))
    return _ok(bypass=dict(_dm().bypass)) if ok else _err(why)


@app.post("/api/thermal/identify/plan")
def api_thermal_identify_plan():
    """Would this wiggle run, and at what amplitude? Read-only: queues nothing, moves nothing.

    Same shape as /api/thermal/predict -- the verdict is `viable`, never the envelope's `ok`."""
    b = request.get_json(force=True, silent=True) or {}
    viable, why, plan = _dm().identify_plan(b)
    return _ok(viable=viable, why=why, plan=plan)


@app.post("/api/thermal/identify/stop")
def api_thermal_identify_stop():
    _dm().identify_stop()
    time.sleep(0.05)
    return _ok()


@app.post("/api/thermal/stop")
def api_thermal_stop():
    _dm().thermal_stop()
    time.sleep(0.02)
    return _ok()


@app.post("/api/thermal/save")
def api_thermal_save():
    """Persist the finished burst. Operator temperatures are optional here and can be added later
    via /api/thermal/annotate -- the case peak lags the burst by a winding time constant, so the
    run has to be saveable before it has been read."""
    got = _dm().get_thermal()
    if got is None:
        return _err("no finished burst to save -- start one, and wait for it to end", 404)
    b = request.get_json(force=True, silent=True) or {}
    run = thermalstore.add_burst(
        motor=got["motor"], envelope=got["envelope"], summary=got["summary"],
        drive_t_start=got["drive_t_start_c"], drive_t_peak=got["drive_t_peak_c"],
        ambient_c=b.get("ambient_c", got.get("ambient_c")), aborted=got["abort"])
    fields = {k: b[k] for k in ("t_start_c", "t_peak_c", "t_peak_at_s", "probe", "notes")
              if b.get(k) is not None}
    if fields:
        run = thermalstore.annotate(run["id"], **fields)
    return _ok(run=run, summary=thermalstore.summary())


@app.post("/api/thermal/annotate")
def api_thermal_annotate():
    b = request.get_json(force=True, silent=True) or {}
    rid = b.pop("id", "")
    try:
        run = thermalstore.annotate(rid, **{k: v for k, v in b.items() if k != "token"})
    except (KeyError, ValueError) as e:
        return _err(e, 404 if isinstance(e, KeyError) else 400)
    return _ok(run=run, summary=thermalstore.summary())


@app.post("/api/thermal/delete")
def api_thermal_delete():
    b = request.get_json(force=True, silent=True) or {}
    try:
        thermalstore.delete(b.get("id", ""))
    except KeyError as e:
        return _err(e, 404)
    return _ok(summary=thermalstore.summary())


@app.post("/api/thermal/cooldown/start")
def api_thermal_cooldown_start():
    b = request.get_json(force=True, silent=True) or {}
    motor = b.get("motor", "")
    if motor not in paths.MOTOR_NAMES:
        return _err("unknown motor {!r}".format(motor))
    c = thermalstore.start_cooldown(motor, ambient_c=b.get("ambient_c"),
                                    probe=b.get("probe", "case"), after_run=b.get("after_run"))
    return _ok(cooldown=c)


@app.post("/api/thermal/cooldown/point")
def api_thermal_cooldown_point():
    b = request.get_json(force=True, silent=True) or {}
    try:
        c = thermalstore.add_point(b.get("id", ""), b.get("t_s"), b.get("temp_c"))
    except (KeyError, TypeError, ValueError) as e:
        return _err(e, 404 if isinstance(e, KeyError) else 400)
    return _ok(cooldown=c, summary=thermalstore.summary())


@app.post("/api/thermal/cooldown/drop")
def api_thermal_cooldown_drop():
    b = request.get_json(force=True, silent=True) or {}
    try:
        c = thermalstore.drop_point(b.get("id", ""), b.get("index", -1))
    except (KeyError, IndexError) as e:
        return _err(e, 404 if isinstance(e, KeyError) else 400)
    return _ok(cooldown=c, summary=thermalstore.summary())


# ===================================================================== policy inference
# The panel: pick an exported bundle, read the controller architecture it carries, prove the
# export + runtime end to end (a --mock dress rehearsal), and RUN IT ON THE DRIVES.
#
# The real run happens inside the daemon, as mode POLICY — not in a subprocess. The CAN bus has
# exactly one owner, so robot/deploy/run_policy.py (which does the same job from a terminal)
# refuses to start while this daemon is up, and the two must never share a bus. Rather than ask the
# operator to stop the web UI and ssh in, the daemon runs the deploy package's own control law and
# safety governor in the loop that already owns the buses. run_policy.py remains the headless
# path and is unchanged.
#
# So the only subprocess this section ever launches is still the dress rehearsal, whose bus and IMU
# are mocks and which therefore cannot collide with anything.
_REHEARSAL = {"proc": None, "file": None, "log": None, "t0": 0.0}
_AXES6 = ("X", "Y", "Z", "roll", "pitch", "yaw")


def _policy_path(fname):
    """POLICY_DIR-jailed path for a client-supplied name. This is the WRITE path (uploads); reads
    go through _policy_read_path, which also sees robot/deploy/bundles/."""
    fname = os.path.basename(str(fname))
    if not fname.endswith(".npz") or not fname[:-4]:
        raise ValueError("a policy bundle is a .npz file")
    return os.path.join(paths.POLICY_DIR, fname)


def _policy_read_path(fname):
    """Where a named bundle actually is, across both bundle directories, or None."""
    return paths.find_policy_bundle(fname)


def _load_bundle(path):
    from bundle import Bundle          # robot/deploy — numpy+json only, importable on the Pi
    return Bundle.load(path)


@app.get("/api/policy/list")
def api_policy_list():
    """Every .npz in either bundle directory (data/policies/ and deploy/bundles/), each fully
    validated by the bundle loader. An unloadable file is listed with its error rather than
    hidden: 'the bundle I scp'd is not offered' must diagnose itself from the panel."""
    out = []
    for f, p, where in paths.list_policy_bundles():
        try:
            m = _load_bundle(p).meta
            out.append({"file": f, "valid": True, "where": where,
                        "run": m.get("run"), "checkpoint": m.get("checkpoint"),
                        "hz": (round(1.0 / float(m["control_dt"]))
                               if m.get("control_dt") else None),
                        "size_kb": os.path.getsize(p) // 1024})
        except Exception as e:                                     # noqa: BLE001 — shown as data
            out.append({"file": f, "valid": False, "where": where, "error": str(e)})
    return _ok(bundles=out, limits={
        "max_seconds": daemon_mod.POLICY_MAX_SECONDS,
        "default_seconds": daemon_mod.POLICY_DEFAULT_SECONDS,
        "deadman_s": daemon_mod.POLICY_DEADMAN_S,
        "approach_dps": daemon_mod.POLICY_APPROACH_DPS,
        "approach_kp": daemon_mod.POLICY_APPROACH_KP,
        "approach_kd": daemon_mod.POLICY_APPROACH_KD,
        "control_hz": daemon_mod.TICK_HZ})


# Per-tick cost is a property of (this bundle, this machine's numpy), so it is measured once and
# remembered until the file changes. The measurement itself costs ~0.5 s on a Pi 3B, which is fine
# once and not fine on every dropdown change.
_PROBE_CACHE = {}


def _policy_step_ms(path):
    """Measured milliseconds per control tick for this bundle here, or None if it will not build.

    This is the number that decides whether the loop can hold 200 Hz, and it is not guessable: the
    same code on the same Pi runs ~90x slower against the reference BLAS than against OpenBLAS, and
    nothing in the bundle or the config says which one numpy found."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    try:
        from controller import PolicyController
        bd = _load_bundle(path)
        ms = _dm()._policy_probe(PolicyController(bd), bd)
    except Exception:                                          # noqa: BLE001 — absence is the answer
        ms = None
    _PROBE_CACHE[key] = ms
    return ms


def _policy_preflight(bundle_path=None):
    """What stands between this robot and a REAL run of a bundle, as data. run_policy.py
    re-checks all of it itself before energising anything — this is the same list surfaced
    where the operator plans, instead of as a SystemExit at the terminal."""
    checks = []
    cal = STATE["calib"]
    ok = bool(cal and cal.complete)
    checks.append({"name": "zeroing", "ok": ok, "why": "" if ok else
                   "run the zeroing wizard — every joint angle the policy reads derives from it"})
    if ok and getattr(cal, "restored_from_disk", False):
        checks.append({"name": "zeroing freshness", "ok": False, "why":
                       "calibration was restored from disk — valid only if the drives were not "
                       "power-cycled since (they re-randomise their origin every power cycle)"})
    jm_path = os.path.join(paths.DEPLOY, "deploy_map.json")
    if os.path.exists(jm_path):
        try:
            import jointmap
            jm_ok, jm_why = jointmap.JointMap.load(jm_path).check_ready()
        except Exception as e:                                     # noqa: BLE001 — shown as data
            jm_ok, jm_why = False, "deploy_map.json is unreadable: {}".format(e)
    else:
        jm_ok, jm_why = False, ("robot/deploy/deploy_map.json does not exist — the model→motor "
                                "joint map has never been verified on this robot")
    checks.append({"name": "joint map", "ok": jm_ok, "why": jm_why})
    th_ok = os.path.exists(os.path.join(paths.DEPLOY, "thermal_params.json"))
    checks.append({"name": "thermal model", "ok": th_ok, "why": "" if th_ok else
                   "no fitted thermal_params.json — continuous-torque limits fall back to "
                   "placeholders (the runner then needs --allow-uncalibrated-thermal)"})
    sh = STATE.get("sense")
    mount_ok = bool(sh and getattr(getattr(sh, "mount", None), "calibrated", False))
    checks.append({"name": "IMU mount", "ok": mount_ok, "why": "" if mount_ok else
                   ("the Sense HAT is not running — gravity is the only fall detector there is"
                    if sh is None else
                    "the IMU mount rotation is not calibrated — 'up' would be in chip axes, not "
                    "body axes (see the gyro calibration panel)")})
    live = bool(sh is not None and sh.fast() is not None)
    checks.append({"name": "IMU live", "ok": live, "why": "" if live else
                   "no IMU sample has arrived — a balance-relevant controller will not be run "
                   "blind"})
    # The run happens in the daemon, so the daemon has to be in a state that can start one.
    d = _dm()
    busy = d._policy_busy_with() if d is not None else "the daemon is not running"
    checks.append({"name": "robot free", "ok": not busy, "why": busy})
    # ...and it has to be able to run it AT RATE. The control law's dt is a constant, so a loop
    # that cannot keep up does not degrade, it changes: the gait plays slow and the observed joint
    # velocities are inflated by exactly the ratio.
    if bundle_path:
        ms = _policy_step_ms(bundle_path)
        budget = 1000.0 / daemon_mod.TICK_HZ
        rate_ok = ms is not None and ms <= daemon_mod.POLICY_MAX_STEP_MS
        checks.append({"name": "loop rate", "ok": rate_ok,
                       "step_ms": None if ms is None else round(ms, 2),
                       "budget_ms": round(budget, 2),
                       "why": "" if rate_ok else (
                           "the bundle would not build here"
                           if ms is None else
                           "this machine needs {:.1f} ms per control tick and the loop period is "
                           "{:.1f} ms — the gait would play at {:.2f}x and every observed joint "
                           "velocity would be inflated {:.1f}x".format(
                               ms, budget, min(1.0, budget / ms), max(1.0, ms / budget)))})
    return checks


@app.post("/api/policy/info")
def api_policy_info():
    b = request.get_json(force=True, silent=True) or {}
    p = _policy_read_path(b.get("file", ""))
    if p is None:
        return _err("no bundle {!r} in data/policies/ or deploy/bundles/".format(
            os.path.basename(str(b.get("file", "")))), 404)
    try:
        bd = _load_bundle(p)
    except Exception as e:                                         # noqa: BLE001 — shown as data
        return _err("not a loadable policy bundle: {}".format(e))
    m = bd.meta
    lock = list(m.get("base_lock") or [])
    railed = [n for n, l in zip(_AXES6, lock) if l]
    warnings = []
    if railed:
        warnings.append("RAILED IN TRAINING: {}. The policy has never experienced those axes "
                        "free and nothing in it stabilises them — on a free-standing robot it "
                        "is open-loop there. Run it on a gantry/boom or with the torso "
                        "supported.".format(", ".join(railed)))
    kp, kd = bd["imp_kp_base"], bd["imp_kd_base"]
    info = {
        "file": os.path.basename(p),
        "run": m.get("run"), "checkpoint": m.get("checkpoint"),
        "control_hz": (round(1.0 / float(m["control_dt"])) if m.get("control_dt") else None),
        "action_dim": m.get("action_dim"),
        "frame_dim": m.get("frame_dim"), "history_len": m.get("history_len"),
        "obs_dim": bd.n_actor,
        # layer sizes, input → output: what "its needed controller architecture" concretely is
        "estimator": [bd.n_actor] + [int(v) for v in (m.get("est_hidden") or [])] + [3],
        "policy": ([bd.n_actor + 3] + [int(v) for v in (m.get("policy_hidden") or [])]
                   + [int(m.get("action_dim") or 0)]),
        "imp_kp": [round(float(np.min(kp)), 1), round(float(np.max(kp)), 1)],
        "imp_kd": [round(float(np.min(kd)), 2), round(float(np.max(kd)), 2)],
        "cmd_box": {"fwd_ms": m.get("cmd_v_fwd_trained"), "back_ms": m.get("cmd_v_back_trained"),
                    "yaw_rads": m.get("cmd_yaw_trained")},
        "base_lock": lock,
        "bundle_version": m.get("bundle_version"),
    }
    # The headless equivalent, for the record and for running without a browser. It is NOT the
    # path the panel uses: run_policy.py refuses to start while this daemon is up, because the CAN
    # bus has one owner.
    cmd = ("sudo systemctl stop runningrobot-webui.service\n"
           "python robot/deploy/run_policy.py \\\n"
           "    --bundle robot/fixed_gait/webui/data/policies/{} \\\n"
           "    --jointmap robot/deploy/deploy_map.json \\\n"
           "    --thermal robot/deploy/thermal_params.json \\\n"
           "    --v-cmd 0.0 --max-seconds 20 --deadman-file /tmp/dash_deadman"
           .format(os.path.basename(p)))
    return _ok(info=info, warnings=warnings, preflight=_policy_preflight(p), command=cmd)


# --------------------------------------------------------------------- running one, for real
@app.post("/api/policy/arm")
def api_policy_arm():
    """Start a real run on the drives. The daemon validates everything again itself; this handler
    only adds the two guards every motion endpoint here carries — the calibration and the
    single-controller token — and then hands the spec over.

    The acknowledgements (`supported`, and the two that stand in for run_policy's
    --skip-jointmap-check / --allow-uncalibrated-thermal flags) are checked in the daemon, next to
    the checks they switch off, rather than here."""
    why = _require_calibrated()
    if why:
        return _err(why, 403)
    b = request.get_json(force=True, silent=True) or {}
    token, err = _acquire_control(b)
    if err:
        return _err(err, 409)
    ok, why, info = _dm().policy_arm(b)
    if not ok:
        return _err(why)
    time.sleep(0.08)                      # let the CAN thread pick it up so the reply shows POLICY
    return _ok(token=token, armed=info)


@app.post("/api/policy/keepalive")
def api_policy_keepalive():
    """The dead-man. The panel calls this ~5 Hz while the run is on screen AND the page is
    visible; if it stops arriving the governor soft-stops within POLICY_DEADMAN_S — gains bled out
    over 0.3 s with the target frozen, which puts the robot down under control.

    Deliberately its own endpoint rather than a side effect of the status poll: a poll proves a
    browser is alive, not that a person is watching."""
    _dm().policy_keepalive()
    return _ok()


@app.post("/api/policy/stop")
def api_policy_stop():
    """Soft by default (freeze the target, bleed the gains out); `hard: true` zeroes them now."""
    b = request.get_json(force=True, silent=True) or {}
    _dm().policy_stop(hard=bool(b.get("hard")))
    time.sleep(0.05)
    return _ok()


@app.post("/api/policy/run/save")
def api_policy_run_save():
    """Persist the finished run's 200 Hz log. Written from THIS thread — the CAN loop never
    touches the filesystem."""
    got = _dm().get_policy()
    if got is None:
        return _err("no finished policy run to save", 404)
    data = np.asarray(got.pop("log"))
    got["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    name = "policyrun_{}_{}".format(time.strftime("%Y%m%d_%H%M%S"),
                                    os.path.splitext(got["file"])[0])
    path = os.path.join(paths.POLICYRUN_DIR, name + ".npz")
    np.savez_compressed(path, data=data, meta_json=np.array(json.dumps(got, default=str)))
    return _ok(file=os.path.basename(path), rows=int(data.shape[0]), run=got)


@app.get("/api/policy/runs")
def api_policy_runs():
    """Saved runs, newest first — the panel links them and the blackbox keeps the rest."""
    out = []
    for f in sorted(os.listdir(paths.POLICYRUN_DIR), reverse=True):
        if not f.endswith(".npz"):
            continue
        p = os.path.join(paths.POLICYRUN_DIR, f)
        row = {"file": f, "size_kb": os.path.getsize(p) // 1024,
               "when": time.strftime("%Y-%m-%d %H:%M:%S",
                                     time.localtime(os.path.getmtime(p)))}
        try:
            with np.load(p, allow_pickle=False) as z:
                m = json.loads(str(z["meta_json"]))
            row.update(run=m.get("run"), checkpoint=m.get("checkpoint"),
                       exit_reason=m.get("exit_reason"), seconds=m.get("run_seconds"),
                       reached_run=m.get("reached_run"))
        except Exception as e:                                     # noqa: BLE001 — shown as data
            row["error"] = str(e)
        out.append(row)
    return _ok(runs=out)


@app.get("/api/policy/run/download")
def api_policy_run_download():
    fname = os.path.basename(request.args.get("file", ""))
    path = os.path.join(paths.POLICYRUN_DIR, fname)
    if not fname.endswith(".npz") or not os.path.exists(path):
        return _err("no such run log", 404)
    with open(path, "rb") as f:
        blob = f.read()
    return send_file(io.BytesIO(blob), download_name=fname, as_attachment=True,
                     mimetype="application/octet-stream")


@app.post("/api/policy/upload")
def api_policy_upload():
    """Accept a bundle over the hotspot (the Pi has no internet; the alternative is scp). The
    whole file is validated by the bundle loader BEFORE anything lands on disk — allow_pickle
    stays False all the way down, so an upload cannot be a code-execution vector."""
    f = request.files.get("file")
    if f is None:
        return _err("no file in the upload")
    raw = f.read()
    if len(raw) > 64 * 1024 * 1024:
        return _err("bundle is {} MB — a policy bundle is a few MB at most"
                    .format(len(raw) >> 20))
    try:
        from bundle import Bundle
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            arrays = {k: z[k] for k in z.files if k != "meta"}
        Bundle(arrays, meta)
    except Exception as e:                                         # noqa: BLE001 — shown as data
        return _err("not a valid policy bundle: {}".format(e))
    try:
        dest = _policy_path(f.filename or "bundle.npz")
    except ValueError:
        dest = os.path.join(paths.POLICY_DIR, "uploaded.npz")
    with open(dest, "wb") as out:
        out.write(raw)
    return _ok(file=os.path.basename(dest))


@app.post("/api/policy/rehearse")
def api_policy_rehearse():
    b = request.get_json(force=True, silent=True) or {}
    if _REHEARSAL["proc"] is not None and _REHEARSAL["proc"].poll() is None:
        return _err("a rehearsal is already running", 409)
    p = _policy_read_path(b.get("file", ""))
    if p is None:
        return _err("no bundle {!r} in data/policies/ or deploy/bundles/".format(
            os.path.basename(str(b.get("file", "")))), 404)
    try:
        secs = min(30.0, max(1.0, float(b.get("seconds") or 5.0)))
    except (TypeError, ValueError):
        secs = 5.0
    log = os.path.join(paths.POLICYRUN_DIR, "rehearsal.log")
    cmd = [sys.executable, os.path.join(paths.DEPLOY, "run_policy.py"),
           "--bundle", p, "--mock", "--max-seconds", str(secs), "--no-log"]
    with open(log, "w", encoding="utf-8") as fh:       # Popen dups the fd; ours can close
        _REHEARSAL["proc"] = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    _REHEARSAL.update(file=os.path.basename(p), log=log, t0=time.time())
    return _ok(started=True)


@app.get("/api/policy/rehearse/status")
def api_policy_rehearse_status():
    r = _REHEARSAL
    if r["proc"] is None:
        return _ok(rehearsal=None)
    rc = r["proc"].poll()
    tail = ""
    try:
        with open(r["log"], "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            n = f.tell()
            f.seek(max(0, n - 8000))
            tail = f.read()
    except OSError:
        pass
    return _ok(rehearsal={"file": r["file"], "running": rc is None, "returncode": rc,
                          "elapsed_s": round(time.time() - r["t0"], 1), "tail": tail})


@app.post("/api/policy/rehearse/stop")
def api_policy_rehearse_stop():
    if _REHEARSAL["proc"] is not None and _REHEARSAL["proc"].poll() is None:
        _REHEARSAL["proc"].terminate()
    return _ok()


@atexit.register
def _kill_rehearsal():
    if _REHEARSAL["proc"] is not None and _REHEARSAL["proc"].poll() is None:
        _REHEARSAL["proc"].terminate()


# ===================================================================== black box (flight recorder)
def _bb():
    return STATE["bb"]


@app.get("/api/blackbox/list")
def api_blackbox_list():
    """What is on disk right now, plus the recorder's own health. Reads the directory and each
    file's header only — never the bulk data — so polling this can never disturb the 200 Hz loop."""
    bb = _bb()
    if bb is None:
        return jsonify({"ok": False, "error": "black box not running", "files": [], "status": {}})
    return jsonify({"ok": True, "status": bb.status(), "files": bb.list_files(),
                    "events_file": blackbox.EVENTS_NAME})


@app.get("/api/blackbox/download")
def api_blackbox_download():
    """Download one recording. Segments are append-only and never rewritten, so a download taken
    while the daemon is running is simply a consistent prefix — no locking, nothing to block."""
    bb = _bb()
    if bb is None:
        return _err("black box not running", 404)
    try:
        blob, fname = bb.read_bytes(request.args.get("name", ""))
    except (FileNotFoundError, OSError) as e:
        return _err(e, 404)
    return send_file(io.BytesIO(blob), download_name=fname, as_attachment=True,
                     mimetype="application/octet-stream")


@app.post("/api/blackbox/mark")
def api_blackbox_mark():
    """Operator annotation — 'it made that noise again'. Lands on the same timeline as the
    machine's own events and pulls a dump of the 200 Hz window around it."""
    bb = _bb()
    if bb is None:
        return _err("black box not running", 404)
    b = request.get_json(force=True, silent=True) or {}
    text = str(b.get("text", "")).strip()
    if not text:
        return _err("mark needs a 'text'")
    name = bb.mark(text, mode=_dm().get_snapshot().get("mode"))
    return _ok(marked=text, dump=name)


@app.post("/api/blackbox/dump")
def api_blackbox_dump():
    """Dump now. The file appears POST_TRIGGER_S later (the tail is still being recorded), which is
    why the name is returned immediately rather than the bytes."""
    bb = _bb()
    if bb is None:
        return _err("black box not running", 404)
    b = request.get_json(force=True, silent=True) or {}
    name = bb.trigger_dump(str(b.get("reason", "manual")) or "manual", manual=True,
                           mode=_dm().get_snapshot().get("mode"))
    return _ok(dump=name, ready_in_s=bb.post_trigger_s)


@app.post("/api/measure/delete")
def api_measure_delete():
    b = request.get_json(force=True, silent=True) or {}
    try:
        removed = measurestore.delete(b.get("name", ""))
    except FileNotFoundError as e:
        return _err(e, 404)
    return _ok(deleted=removed, measurements=measurestore.list_summaries())


# ===================================================================== dynamics config (mass/PID)
@app.post("/api/dynamics/mass")
def api_dynamics_mass():
    b = request.get_json(force=True, silent=True) or {}
    ok, why = STATE["dyn"].set_mass(b.get("body", ""), b.get("mass"))
    return _ok(dynamics=STATE["dyn"].snapshot()) if ok else _err(why)


@app.post("/api/dynamics/pid")
def api_dynamics_pid():
    b = request.get_json(force=True, silent=True) or {}
    ok, why = STATE["dyn"].set_pid(b.get("motor", ""), kp=b.get("kp"), ki=b.get("ki"),
                                   kd=b.get("kd"))
    return _ok(dynamics=STATE["dyn"].snapshot()) if ok else _err(why)


# ===================================================================== inertia inspect / compare
def _model_inertia_payload():
    """Per-body CAD (model) inertials + identified inertials (if present) + the frames.compare()
    verdict/rotation for each. Drives the Limbs & Inertia panel."""
    try:
        cad = model_inertials.read_bodies(paths.MODEL_XML)
    except (OSError, ValueError) as e:
        return {"error": f"could not read model inertials from {paths.MODEL_XML}: {e}", "bodies": {}}
    params = paramio.load_or_none(IDENT_PARAMS_FILE) or {}
    idb = params.get("bodies", {})
    weighed = STATE["dyn"].as_dict().get("masses", {})
    bodies = {}
    for name, c in cad.items():
        ident = idb.get(name)
        bodies[name] = {"comparison": frames.compare(c, ident),
                        "weighed_mass": weighed.get(name)}
    return {"bodies": bodies, "has_identified": bool(idb),
            "kt": params.get("kt", {}), "friction": params.get("friction", {}),
            "rotor_armature": params.get("rotor_armature", {}),
            "validation": params.get("validation", {}),
            "created": params.get("created"), "sources": params.get("sources", [])}


@app.get("/api/model/inertia")
def api_model_inertia():
    return jsonify(_model_inertia_payload())


@app.post("/api/inertia/compare")
def api_inertia_compare():
    """Ad-hoc: compare two supplied {mass, com, inertia} bodies (e.g. a pasted alternate CAD tensor
    against the identified one) — same maths as the panel, no persistence."""
    b = request.get_json(force=True, silent=True) or {}
    try:
        return jsonify(frames.compare(b.get("cad"), b.get("identified")))
    except (KeyError, ValueError, TypeError) as e:
        return _err(f"bad tensor payload: {e}")


@app.get("/api/mesh/<name>")
def api_mesh(name):
    """Serve a link STL for the 3D viewer (sanitized; must be a known model mesh)."""
    safe = os.path.basename(name)
    if not safe.endswith(".stl") or not os.path.exists(os.path.join(MESH_DIR, safe)):
        return _err(f"mesh '{name}' not found", 404)
    return send_from_directory(MESH_DIR, safe, mimetype="model/stl")


# ===================================================================== identification estimator
@app.get("/api/identify")
def api_identify_get():
    params = paramio.load_or_none(IDENT_PARAMS_FILE)
    return jsonify({"available": params is not None, "params": params})


@app.post("/api/identify/import")
def api_identify_import():
    """Upload an identified_params.json produced offline on the dev machine (the estimator host)."""
    f = request.files.get("file")
    if f is None:
        return _err("multipart 'file' missing")
    try:
        params = json.loads(f.read().decode("utf-8-sig"))
    except (ValueError, UnicodeDecodeError) as e:
        return _err(f"not valid JSON: {e}")
    if "bodies" not in params and "kt" not in params:
        return _err("that JSON has no 'bodies' or 'kt' — is it an identified_params file?")
    paramio.save(params, IDENT_PARAMS_FILE)
    return _ok(imported=True)


@app.post("/api/identify/run")
def api_identify_run():
    """Run the offline estimator in-process host if mujoco+scipy are available here; otherwise 501
    with the exact CLI to run on the dev/training machine. Estimation is CPU-heavy but bounded."""
    b = request.get_json(force=True, silent=True) or {}
    measures = b.get("measurements") or [m["name"] for m in measurestore.list_summaries()]
    if not measures:
        return _err("no measurement runs to identify from — capture some first")
    cmd = [sys.executable, "-m", "identification.run",
           "--model", paths.MODEL_XML, "--out", IDENT_PARAMS_FILE,
           "--config", paths.DYN_CONFIG_FILE, "--measure-dir", paths.MEASURE_DIR,
           "--measures", *measures]
    try:
        import mujoco  # noqa: F401
        import scipy    # noqa: F401
    except ImportError:
        return jsonify({"ok": False, "code": "no_estimator_deps",
                        "error": "this host has no mujoco/scipy — run the estimator on the dev "
                                 "machine, then upload identified_params.json here",
                        "cli": " ".join(f'"{c}"' if " " in c else c for c in cmd),
                        "cwd": paths.REPO, "state": _dm().get_snapshot()}), 501
    try:
        r = subprocess.run(cmd, cwd=paths.REPO, capture_output=True, text=True, timeout=1800)
    except (subprocess.TimeoutExpired, OSError) as e:
        return _err(f"estimator failed to launch/finish: {e}", 500)
    if r.returncode != 0:
        return _err(f"estimator exited {r.returncode}: {r.stderr[-1500:] or r.stdout[-1500:]}", 500)
    params = paramio.load_or_none(IDENT_PARAMS_FILE)
    return _ok(ran=True, log=r.stdout[-4000:], available=params is not None)


# ===================================================================== FK / misc
@app.post("/api/fk/verify")
def api_fk_verify():
    report = STATE["fk"].verify_against_workspace(STATE["wstore"])
    return _ok(report=report)


@app.post("/api/fk/map")
def api_fk_map():
    """Manually set (and optionally force-verify) a side's sign map — the escape hatch when
    auto-verify is not decisive (e.g. recorded cam travel exceeds the model's joint range)."""
    b = request.get_json(force=True, silent=True) or {}
    side = b.get("side")
    if side not in paths.SIDES:
        return _err("side must be right|left")
    fk = STATE["fk"]
    if not fk.available:
        return _err("no FK LUT loaded")
    try:
        cam_s = 1 if int(b.get("cam", 1)) >= 0 else -1
        thigh_s = 1 if int(b.get("thigh", 1)) >= 0 else -1
    except (TypeError, ValueError):
        return _err("cam/thigh must be ±1")
    fk.model_map[side].update({"cam": cam_s, "thigh": thigh_s})
    for key in ("cam_off_deg", "thigh_off_deg"):
        if key in b:
            try:
                v = float(b[key])
            except (TypeError, ValueError):
                return _err(f"{key} must be a number (degrees)")
            if not -360.0 <= v <= 360.0:
                return _err(f"{key} out of range (±360°)")
            fk.model_map[side][key] = v
    if "flip_view" in b:
        fk.model_map[side]["flip_view"] = bool(b["flip_view"])
    if "verified" in b:
        fk.model_map["verified"][side] = bool(b["verified"])
    else:
        fk.model_map["verified"][side] = True
    fk.save_map()
    return _ok(model_map={s: fk.model_map[s] for s in paths.SIDES},
               verified=dict(fk.model_map["verified"]))


@app.post("/api/mock/drag")
def api_mock_drag():
    if not STATE["mock"]:
        return _err("mock mode only", 403)
    b = request.get_json(force=True, silent=True) or {}
    ok = _dm().mock_drag(b.get("motor", ""), b.get("norm_deg"))
    return _ok() if ok else _err("unknown motor or not mock")


@app.post("/api/mock/sensors")
def api_mock_sensors():
    """Pose the simulated IMU (still / tilt / rock) so the mount calibration can be walked through
    end to end without the robot."""
    if not STATE["mock"]:
        return _err("mock mode only", 403)
    sh, why = _sense()
    if why:
        return _err(why)
    b = request.get_json(force=True, silent=True) or {}
    r = sh.mock_pose(b.get("pose", "still"))
    return _ok() if r.get("ok") else _err(r.get("error", "mock pose failed"))


# ===================================================================== lifecycle
def _shutdown():
    sh = STATE.get("sense")
    if sh is not None:
        sh.stop()
    d = STATE.get("daemon")
    if d is not None:
        d.stop_event.set()
        d.join(2.0)
        if d.is_alive():
            print("!! daemon did not stop — firing zero-current fallback")
            _estop_fallback()
    bb = STATE.get("bb")
    if bb is not None:
        bb.log_event("server.stop", clean=True)
        bb.stop()          # LAST: it must outlive whatever it is recording the death of


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--mock", action="store_true", help="simulated motors (no CAN needed)")
    ap.add_argument("--no-sensors", action="store_true",
                    help="do not touch the Sense HAT (B) / I2C bus at all")
    ap.add_argument("--i2c-bus", type=int, default=sensehat.I2C_BUS,
                    help="I2C bus the Sense HAT (B) sits on (default %(default)s)")
    ap.add_argument("--imu-hz", type=float, default=sensehat.IMU_HZ,
                    help="IMU read/AHRS rate in Hz (default %(default)s, matching the control loop)")
    ap.add_argument("--no-blackbox", action="store_true",
                    help="do not record anything to disk (you will regret this)")
    ap.add_argument("--blackbox-budget-mb", type=float, default=blackbox.BUDGET_BYTES / 1e6,
                    help="total disk the flight recorder may use, MB (default %(default).0f)")
    args = ap.parse_args()

    # Python hands the GIL between threads at most every `switchinterval` seconds — 5 ms by
    # default, which is exactly the 200 Hz CAN loop's period, so one thread could hold it across a
    # whole control tick. Measured on the robot: dropping it takes the daemon's late ticks from
    # ~10% back to well under 1% once the IMU thread also runs at 200 Hz.
    sys.setswitchinterval(0.0005)

    STATE["interface"] = args.interface
    STATE["mock"] = args.mock

    # The flight recorder comes up FIRST so the calibration/dynamics load below is already on the
    # timeline — "which calibration was live at boot" is one of the five questions it must answer.
    bb = None
    if not args.no_blackbox:
        bb = blackbox.BlackBox(budget_bytes=int(args.blackbox_budget_mb * 1e6))
        blackbox.install(bb)
        bb.start()
    STATE["bb"] = bb

    STATE["calib"] = calibration.Calibration.load_or_new()
    STATE["wstore"] = workspace.WorkspaceStore()
    STATE["fk"] = fklut.FkLut()
    STATE["dyn"] = dynstore.DynConfig.load_or_new()
    if bb is not None:
        # config provenance: every Tier B dump references the hash of exactly this
        bb.set_config_provider(lambda: {"calibration": STATE["calib"].snapshot(),
                                        "dynamics": STATE["dyn"].as_dict()})
        bb.note_config_change("boot")
    d = daemon_mod.RobotDaemon(interface=args.interface, mock=args.mock,
                               calib=STATE["calib"], wstore=STATE["wstore"], fklut=STATE["fk"],
                               bb=bb)
    STATE["daemon"] = d
    d.start()
    d._started_ok.wait(5.0)

    # Sense HAT (B) on I2C — its own thread, started after the daemon so a wedged I2C bus can never
    # delay motor bring-up. In --mock it synthesises values so the panel works off the robot.
    if not args.no_sensors:
        sh = sensehat.SenseHat(bus_num=args.i2c_bus, mock=args.mock, imu_hz=args.imu_hz)
        STATE["sense"] = sh
        # POLICY mode reads gravity and body rate off this at 200 Hz (sh.fast() is one lock and a
        # tuple read). Attached rather than constructor-injected because the HAT is deliberately
        # started AFTER the daemon, so a wedged I2C bus cannot delay motor bring-up.
        d.sense = sh
        sh.start()
    atexit.register(_shutdown)

    print(f"\nDASH-01 web UI: http://{args.host}:{args.port}/  "
          f"({'MOCK motors' if args.mock else f'{args.interface} can0/can1'})")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
