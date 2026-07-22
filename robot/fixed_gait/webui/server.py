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

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

import paths
import calibration
import canio
import daemon as daemon_mod
import dynstore
import fklut
import gaitstore
import measurestore
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
    "interface": "socketcan",
    "mock": False,
    "ctl": {"token": None, "ts": 0.0},
}
CTL_TIMEOUT_S = 15.0


# ===================================================================== helpers
def _dm():
    return STATE["daemon"]


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


@app.get("/api/telemetry")
def api_telemetry():
    d = _dm()
    since = int(request.args.get("since", 0))
    seq, t, data = d.ring.read_since(since)
    out = {"seq": seq, "t": np.round(t, 3).tolist(),
           "motors": {}}
    for i, n in enumerate(paths.MOTOR_NAMES):
        out["motors"][n] = {
            "pos_norm": _nan_list(data["pos_norm"][:, i]),
            "pos_raw": _nan_list(data["pos_raw"][:, i]),
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


def _nan_list(a):
    return [None if not np.isfinite(v) else round(float(v), 2) for v in a]


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


@app.post("/api/calibration/zero")
def api_calibration_zero():
    d = _dm()
    if d.get_snapshot().get("mode") not in ("LIMP", "ESTOPPED"):
        return _err("go LIMP before setting zero")
    ok, why = STATE["calib"].set_zero(d.latest_raw_positions())
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


# ===================================================================== lifecycle
def _shutdown():
    d = STATE.get("daemon")
    if d is None:
        return
    d.stop_event.set()
    d.join(2.0)
    if d.is_alive():
        print("!! daemon did not stop — firing zero-current fallback")
        _estop_fallback()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--mock", action="store_true", help="simulated motors (no CAN needed)")
    args = ap.parse_args()

    STATE["interface"] = args.interface
    STATE["mock"] = args.mock
    STATE["calib"] = calibration.Calibration.load_or_new()
    STATE["wstore"] = workspace.WorkspaceStore()
    STATE["fk"] = fklut.FkLut()
    STATE["dyn"] = dynstore.DynConfig.load_or_new()
    d = daemon_mod.RobotDaemon(interface=args.interface, mock=args.mock,
                               calib=STATE["calib"], wstore=STATE["wstore"], fklut=STATE["fk"])
    STATE["daemon"] = d
    d.start()
    d._started_ok.wait(5.0)
    atexit.register(_shutdown)

    print(f"\nDASH-01 web UI: http://{args.host}:{args.port}/  "
          f"({'MOCK motors' if args.mock else f'{args.interface} can0/can1'})")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
