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
import struct
import sys
import threading
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
    """Same OUTPUT as server.py's version, but vectorised — this is the hot path of this process.

    The original ran one Python iteration per element with np.isfinite() and round() inside, and a
    telemetry response is 36 of these (6 fields x 6 motors). That per-element loop is what saturated
    this interpreter's GIL once the work moved here: measured, an upstream call that takes 17 ms
    from a standalone script took 191 ms from inside a request handler, purely from contention.
    .tolist() and np.isfinite() are both C level, and the NaN fixups are usually zero elements."""
    lst = np.round(a, nd).tolist()
    bad = np.flatnonzero(~np.isfinite(a))
    for i in bad:
        lst[int(i)] = None
    return lst




# ===================================================================== the upstream feed
# The browser polls this process ~22 times a second (telemetry 10 Hz, sensors 10 Hz, state 2 Hz,
# plus a policy dead-man at 5 Hz while a run is live). Until 2026-09-01 each of those became an
# upstream request into the process that owns the 200 Hz CAN loop. Measured there: roughly 1 Hz of
# control rate lost per upstream request per second, whichever endpoint it was -- the cost is the
# socket accept, the HTTP parse, the Werkzeug request thread and the response assembly, not the
# payload. 193 Hz idle became 107 Hz with one tab open.
#
# So: ONE persistent connection, opened here, over which the control process pushes a frame every
# 50 ms. Browser polls are answered from `FEED` without touching upstream at all. The browser's
# poll rate stops being something the control loop can feel.
#
# The `since` cursor the browser sends is served from a rolling local buffer, because the stream
# delivers each sample exactly once and a client that reconnects or lags must still be able to ask
# for a window rather than being told "nothing new since a sequence I never saw".
#
# FALLBACK IS THE POINT, not an afterthought: if the stream is not connected -- upstream restarted,
# this process started first, an old server.py without /api/stream -- every handler below falls
# back to the request-per-poll path that shipped in 2026-08. Losing the optimisation must never
# mean losing the UI.
STREAM_HZ = 20.0
FEED_KEEP = 1200                 # samples of history kept locally (~60 s of telemetry at 20 Hz)


class _Feed:
    """The latest frame from upstream, plus enough history to answer a `since` cursor."""

    def __init__(self):
        self.lock = threading.Lock()
        self.connected = False
        self.err = ""
        self.frames = 0
        self.state_hot = None
        self.mode = None
        self.linkage = None
        self.tel = None          # (seq, fields, t[n], arr[f, n, 6])
        self.sen = None          # (seq, fields, meta, t[m], arr[f, m])

    # -- writer (the reader thread) ----------------------------------------------------------
    def push(self, header, tel_t, tel_a, sen_t, sen_a):
        with self.lock:
            self.state_hot = header.get("state_hot")
            th, sh_ = header["tel"], header["sen"]
            self.mode, self.linkage = th.get("mode"), th.get("linkage")
            self.tel = _append(self.tel, th["seq"], th["fields"], tel_t, tel_a)
            self.sen = _append(self.sen, sh_["seq"], sh_["fields"], sen_t, sen_a,
                               meta=sh_.get("meta"))
            self.frames += 1
            self.connected = True
            self.err = ""

    def drop(self, why):
        with self.lock:
            self.connected = False
            self.err = str(why)

    # -- readers (Flask threads) --------------------------------------------------------------
    def since(self, which, cursor):
        """(seq, fields, t, arr, meta) for the samples after `cursor`, or None if not streaming.

        Served from local history rather than from the newest frame alone: the stream delivers each
        sample exactly once, so a browser that reconnects, lags, or asks for a window wider than
        one frame would otherwise be told there is nothing new since a sequence it never saw."""
        with self.lock:
            cur = self.tel if which == "tel" else self.sen
            if not self.connected or cur is None:
                return None
            seq, fields, t, arr, meta = cur
            try:
                c = int(cursor or 0)
            except (TypeError, ValueError):
                c = 0
            n = int(t.size)
            take = min(max(seq - c, 0), n)
            sl = slice(n - take, n)
            return seq, fields, t[sl], (arr[:, sl] if arr.ndim >= 2 else arr), meta


def _append(prev, seq, fields, t, arr, meta=None):
    """Concatenate a frame's new samples onto the local history, bounded to FEED_KEEP.

    The sample axis is 1 for both shapes carried here: telemetry is (fields, samples, motors) and
    sensors is (fields, samples). A frame with no new samples keeps the history it had."""
    if prev is not None and prev[1] == fields and prev[3].ndim == arr.ndim:
        if t.size and arr.ndim >= 2:
            t = np.concatenate([prev[2], t])[-FEED_KEEP:]
            arr = np.concatenate([prev[3], arr], axis=1)[:, -FEED_KEEP:]
        else:
            t, arr = prev[2], prev[3]
            meta = prev[4] if meta is None else meta
    return (seq, fields, t, arr, meta)



FEED = _Feed()


def _reader():
    """One connection, reopened whenever it drops. Never raises out of the thread."""
    backoff = 0.5
    while True:
        try:
            with FEED.lock:
                tel_seq = FEED.tel[0] if FEED.tel else 0
                sen_seq = FEED.sen[0] if FEED.sen else 0
            url = f"{UPSTREAM}/api/stream?since={tel_seq}&sensors_since={sen_seq}"
            with urllib.request.urlopen(url, timeout=30) as r:
                backoff = 0.5
                while True:
                    head_len = _read_exactly(r, 4)
                    if head_len is None:
                        break
                    n = struct.unpack(">I", head_len)[0]
                    if not 0 < n < 4_000_000:
                        break
                    raw = _read_exactly(r, n)
                    if raw is None:
                        break
                    header = json.loads(raw.decode("utf-8"))
                    th, sh_ = header["tel"], header["sen"]
                    tel_t = _read_array(r, (th["n"],), np.float64)
                    tel_a = _read_array(r, tuple(th["shape"]), np.float32)
                    sen_t = _read_array(r, (sh_["n"],), np.float64)
                    sen_a = _read_array(r, tuple(sh_["shape"]), np.float32)
                    if tel_t is None or tel_a is None or sen_t is None or sen_a is None:
                        break
                    FEED.push(header, tel_t, tel_a, sen_t, sen_a)
        except Exception as e:                     # noqa: BLE001 -- a dropped feed is not fatal
            FEED.drop(e)
        else:
            FEED.drop("stream closed by upstream")
        time.sleep(backoff)
        backoff = min(backoff * 2.0, 5.0)


def _read_exactly(fh, n):
    out = bytearray()
    while len(out) < n:
        chunk = fh.read(n - len(out))
        if not chunk:
            return None
        out.extend(chunk)
    return bytes(out)


def _read_array(fh, shape, dtype):
    count = 1
    for d in shape:
        count *= int(d)
    nbytes = count * np.dtype(dtype).itemsize
    raw = _read_exactly(fh, nbytes) if nbytes else b""
    if raw is None:
        return None
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def start_feed():
    threading.Thread(target=_reader, name="upstream-feed", daemon=True).start()


@app.get("/api/feed")
def feed_status():
    """Is the stream up, and how far behind? Diagnosing 'the UI looks frozen' should not need ssh."""
    with FEED.lock:
        return jsonify({"connected": FEED.connected, "frames": FEED.frames, "error": FEED.err,
                        "tel_seq": FEED.tel[0] if FEED.tel else 0,
                        "sen_seq": FEED.sen[0] if FEED.sen else 0,
                        "stream_hz": STREAM_HZ})


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/telemetry")
def telemetry():
    """Same JSON shape as server.py's /api/telemetry — the browser cannot tell the difference."""
    got = FEED.since("tel", request.args.get("since", 0))
    if got is not None:
        seq, fields, t, arr, _meta = got
        mode, linkage = FEED.mode, FEED.linkage
    else:
        # stream down: the request-per-poll path, exactly as it was before the feed existed
        try:
            st, body, hdrs = _fetch("/api/telemetry_raw", request.query_string)
        except (urllib.error.URLError, OSError) as e:
            return jsonify({"error": f"control process unreachable: {e}"}), 503
        if st != 200:
            return Response(body, status=st, mimetype="application/json")
        buf = io.BytesIO(body)
        t = np.load(buf)                   # two flat arrays, read back in the order written
        arr = np.load(buf)                 # (n_fields, n_samples, N_MOTORS)
        fields = json.loads(hdrs.get("X-Fields", "[]"))
        seq = int(hdrs.get("X-Seq", 0))
        mode = hdrs.get("X-Mode")
        linkage = json.loads(hdrs.get("X-Linkage", "null"))
    idx = {f: k for k, f in enumerate(fields)}
    out = {"seq": seq, "t": np.round(t, 3).tolist(), "motors": {}}
    for i, n in enumerate(paths.MOTOR_NAMES):
        out["motors"][n] = {f: _nan_list(arr[idx[f]][:, i])
                            for f in ("pos_norm", "pos_raw", "cmd_norm", "cur", "temp", "spd")
                            if f in idx}
    out["linkage"] = linkage
    out["mode"] = mode
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
    with FEED.lock:
        hot = FEED.state_hot if FEED.connected else None
    if hot is not None:
        out = dict(hot)
    else:
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
    got = FEED.since("sen", request.args.get("since", 0))
    if got is not None:
        seq, fields, t, arr, meta = got
        out = dict(meta or {})
        out["seq"] = seq
        out["t"] = np.round(t, 3).tolist()
        out["series"] = {f: _nan_list(arr[k], 3) for k, f in enumerate(fields) if k < len(arr)}
        return jsonify(out)
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
    buf = io.BytesIO(body)
    t = np.load(buf)
    arr = np.load(buf)
    fields = json.loads(hdrs.get("X-Fields", "[]"))
    out["t"] = np.round(t, 3).tolist()
    out["series"] = {f: _nan_list(arr[k], 3) for k, f in enumerate(fields) if k < len(arr)}
    return jsonify(out)


@app.route("/<path:path>", methods=["GET", "POST", "PATCH", "PUT", "DELETE"])
def proxy(path):
    """Everything else goes upstream verbatim, so there is only ONE implementation of the API.

    These are all human-rate (a click, a form) — a 20 ms stall when someone presses a button costs
    nothing, which is exactly why only the polling endpoints above were worth moving."""
    if request.method != "GET":
        _invalidate_cold()               # any write can change the cold half — never serve it stale
    t_in = time.perf_counter()
    data = request.get_data() or None
    t_body = time.perf_counter()
    try:
        st, body, hdrs = _fetch("/" + path, request.query_string, request.method,
                                data, dict(request.headers))
    except (urllib.error.URLError, OSError) as e:
        return jsonify({"ok": False, "error": f"control process unreachable: {e}"}), 503
    t_up = time.perf_counter()
    ct = hdrs.get("Content-Type", "application/json")
    resp = Response(body, status=st, mimetype=ct.split(";")[0])
    for k, v in hdrs.items():
        if k.lower() not in _DROP and k.lower() not in ("content-type",):
            resp.headers[k] = v
    # where the time actually goes, per request — the proxy hop measured 280 ms against a 17 ms
    # upstream and neither CPU nor nice explained it, so the split has to be visible from outside.
    resp.headers["X-Proxy-Body-Ms"] = f"{(t_body - t_in) * 1000:.1f}"
    resp.headers["X-Proxy-Upstream-Ms"] = f"{(t_up - t_body) * 1000:.1f}"
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
    start_feed()          # one upstream connection, opened before the first browser arrives
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
