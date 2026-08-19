#!/usr/bin/env python3
"""Operator-facing web front end, in its OWN process — so the UI cannot stall the 200 Hz CAN loop.

WHY THIS FILE EXISTS
--------------------
Measured on the robot with candump, 2026-08-19, 8 s capture, motors LIMP:

    direction                       rate       sd       p99      worst
    RX  motor status 0x29xx      200.1 Hz   0.04 ms   5.1 ms    5.5 ms
    TX  Pi's SET_CURRENT 0x01xx  156.6 Hz   6.4  ms  45   ms   72.9 ms

Both directions cross the same MCP251xFD on the same 20 MHz SPI, so the transport is not the
problem: receive is clean to 40 microseconds. The Pi's own transmit loop was dropping 21.6% of its
200 Hz slots. Gaps over 10 ms happened 9.16 times per second; the browser was polling at 9.1
requests per second. One stall per request, 24.3% of wall-clock time lost inside them.

The cause is the GIL, not the scheduler. /api/telemetry runs 36 per-element Python loops (6 fields
x 6 motors through _nan_list) plus jsonify, in the same interpreter as the CAN thread. Note what
does NOT fix it: sys.setswitchinterval is already 0.5 ms (server.py), and SCHED_FIFO cannot help
either — priority does not let one thread preempt another that holds the lock it needs. The only
real fix is a second interpreter, which means a second process.

SO: server.py keeps the daemon, CAN, calibration, the black box and every endpoint, and gains two
additive raw endpoints that copy the ring under its lock and hand back np.savez bytes (C level, no
per-element Python). This process does the JSON. Everything else is proxied through untouched.

FAILURE + ROLLBACK, deliberately boring:
  * server.py is unchanged in behaviour. Run it alone on :8080 and you have exactly today's robot;
    that is the rollback, and it needs no code change.
  * the control process binds 0.0.0.0:8081, NOT localhost. If this process dies, the operator can
    still reach the full UI — including E-STOP — directly on :8081. Losing the front end must never
    mean losing the stop button.
  * anything this process cannot answer is proxied verbatim, so there is no second copy of the API
    to keep in sync. New endpoints in server.py work here the day they are added.

Stdlib only for HTTP (urllib): the Pi has no internet, so `requests` is not installable there.
"""
import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

import paths

app = Flask(__name__, static_folder="static", static_url_path="/static")

UPSTREAM = "http://127.0.0.1:8081"
TIMEOUT_S = 4.0

# Hop-by-hop headers must not be forwarded (RFC 7230 6.1); Content-Length is recomputed by Flask.
_DROP = {"content-length", "connection", "keep-alive", "transfer-encoding", "upgrade",
         "proxy-authenticate", "proxy-authorization", "te", "trailer", "host"}


def _fetch(path, query=b"", method="GET", body=None, headers=None):
    """One upstream call. Returns (status, body_bytes, headers_dict).

    query MUST be decoded: Flask's request.query_string is bytes, and f-string-ing it yields the
    repr -- the upstream then saw `?b'since=94'`, parsed no `since` at all, and fell back to 0, so
    every poll re-sent the WHOLE 512-sample ring instead of the handful of new samples. That turned
    the cheap path into the worst case on every request, which is the opposite of the point."""
    if isinstance(query, bytes):
        query = query.decode("utf-8", "replace")
    url = f"{UPSTREAM}{path}" + (f"?{query}" if query else "")
    req = urllib.request.Request(url, data=body, method=method)
    for k, v in (headers or {}).items():
        if k.lower() not in _DROP:
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:          # a real answer, just not 2xx — pass it through
        return e.code, e.read(), dict(e.headers)


def _nan_list(a, nd=2):
    """Identical to server.py's, on purpose: this is the work that moved off the control process."""
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/telemetry")
def telemetry():
    """Same JSON shape as server.py's /api/telemetry — the browser cannot tell the difference."""
    try:
        st, body, hdrs = _fetch("/api/telemetry_raw", request.query_string)
    except (urllib.error.URLError, OSError) as e:
        return jsonify({"error": f"control process unreachable: {e}"}), 503
    if st != 200:
        return Response(body, status=st, mimetype="application/json")
    z = np.load(io.BytesIO(body))
    out = {"seq": int(hdrs.get("X-Seq", 0)), "t": np.round(z["t"], 3).tolist(), "motors": {}}
    for i, n in enumerate(paths.MOTOR_NAMES):
        out["motors"][n] = {f: _nan_list(z[f][:, i])
                            for f in ("pos_norm", "pos_raw", "cmd_norm", "cur", "temp", "spd")}
    out["linkage"] = json.loads(hdrs.get("X-Linkage", "null"))
    out["mode"] = hdrs.get("X-Mode")
    return jsonify(out)


_cold = {"at": 0.0, "body": None}         # cached /api/state_cold
COLD_TTL_S = 5.0


def _invalidate_cold():
    _cold["body"] = None


@app.get("/api/state")
def state():
    """/api/state rebuilt from a cheap hot half + a CACHED cold half.

    The cold half costs the control process 18 npz metadata reads off the SD card (every saved run)
    plus two directory listings, and it was the last thing left stalling the 200 Hz CAN loop after
    the process split. It only changes when a human saves, deletes or calibrates — all non-GET —
    so proxy() below drops the cache on any write and the TTL is only a backstop for changes that
    do not come through HTTP."""
    try:
        st, body, _ = _fetch("/api/state_hot")
    except (urllib.error.URLError, OSError) as e:
        return jsonify({"ok": False, "error": f"control process unreachable: {e}"}), 503
    if st != 200:
        return Response(body, status=st, mimetype="application/json")
    out = json.loads(body)
    now = time.monotonic()
    if _cold["body"] is None or now - _cold["at"] > COLD_TTL_S:
        try:
            cst, cbody, _ = _fetch("/api/state_cold")
            if cst == 200:
                _cold["body"], _cold["at"] = json.loads(cbody), now
        except (urllib.error.URLError, OSError):
            pass                          # keep whatever we had; hot half is still fresh
    if _cold["body"]:
        out.update(_cold["body"])
    return jsonify(out)


@app.get("/api/sensors")
def sensors():
    try:
        st, body, hdrs = _fetch("/api/sensors_raw", request.query_string)
    except (urllib.error.URLError, OSError) as e:
        return jsonify({"available": False, "error": f"control process unreachable: {e}",
                        "seq": 0, "t": [], "series": {}}), 200
    if st != 200:
        return Response(body, status=st, mimetype="application/json")
    out = json.loads(hdrs.get("X-Meta", "{}"))
    out["seq"] = int(hdrs.get("X-Seq", 0))
    if not body:                                  # sensors disabled upstream
        out.setdefault("t", []), out.setdefault("series", {})
        return jsonify(out)
    z = np.load(io.BytesIO(body))
    out["t"] = np.round(z["t"], 3).tolist()
    out["series"] = {f: _nan_list(z[f], 3) for f in json.loads(hdrs.get("X-Fields", "[]"))}
    return jsonify(out)


@app.route("/<path:path>", methods=["GET", "POST", "PATCH", "PUT", "DELETE"])
def proxy(path):
    """Everything else goes upstream verbatim, so there is only ONE implementation of the API.

    These are all human-rate (a click, a form) — a 20 ms stall when someone presses a button costs
    nothing, which is exactly why only the polling endpoints above were worth moving."""
    if request.method != "GET":
        _invalidate_cold()               # any write can change the cold half — never serve it stale
    try:
        st, body, hdrs = _fetch("/" + path, request.query_string, request.method,
                                request.get_data() or None, dict(request.headers))
    except (urllib.error.URLError, OSError) as e:
        return jsonify({"ok": False, "error": f"control process unreachable: {e}"}), 503
    ct = hdrs.get("Content-Type", "application/json")
    resp = Response(body, status=st, mimetype=ct.split(";")[0])
    for k, v in hdrs.items():
        if k.lower() not in _DROP and k.lower() not in ("content-type",):
            resp.headers[k] = v
    return resp


def main():
    global UPSTREAM                      # declared first: `default=UPSTREAM` below reads it
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080, help="operator-facing port")
    ap.add_argument("--upstream", default=UPSTREAM, help="control process base URL")
    args = ap.parse_args()
    UPSTREAM = args.upstream.rstrip("/")
    # This process has no real-time work at all, so it does NOT need the 0.5 ms switch interval
    # server.py sets — leaving it at the default keeps its own threads cheap.
    print(f"\nDASH-01 UI front: http://{args.host}:{args.port}/  -> control at {UPSTREAM}")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
