"""The thermal panel's HTTP contract.

The bug these guard against shipped on 2026-08-28 and made the Start button permanently dead:
/api/thermal/predict returned the burst-viability verdict in the response envelope's own `ok`
field, and the shared api() helper in app.js treats `ok: false` as a FAILED REQUEST -- it banners
d.error (undefined here) and throws. Every non-viable burst therefore aborted the render before it
could update the readout or enable the button, and the panel looked broken rather than refusing.

MockBus only -- no hardware, no CAN, no robot.

    python -m pytest robot/fixed_gait/webui/tests/test_thermal_api.py -v
"""
import io
import time
import os
import re
import types

import pytest

import paths
import server
from test_blackbox import capture_zero, robot          # noqa: F401  (pytest fixtures)

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")


@pytest.fixture
def client(robot):                                     # noqa: F811
    d, cal, _b, _dir = robot
    capture_zero(d, cal)
    d._zero_epoch_at_start = -1
    keys = ("daemon", "calib", "mock", "fk", "sense", "wstore", "dyn")
    saved = {k: server.STATE[k] for k in keys}
    # the raw-stream endpoints reach for the FK LUT and the Sense HAT; neither exists in a test
    # process, and neither is what these tests are about
    server.STATE.update(daemon=d, calib=cal, mock=True,
                        fk=types.SimpleNamespace(available=False, try_reload=lambda: None,
                                                 side_verified=lambda side: False),
                        wstore=types.SimpleNamespace(legs={"left": {}, "right": {}},
                                                     source="workspace_test.npz",
                                                     list_files=lambda: []),
                        dyn=types.SimpleNamespace(snapshot=lambda: {}),
                        sense=None)
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c, d
    server.STATE.update(saved)


def predict(client, amps, duration_s, motor="left.thigh"):
    r = client.post("/api/thermal/predict",
                    json={"motor": motor, "amps": amps, "duration_s": duration_s})
    return r, r.get_json()


# ===================================================================== the shipped bug
def test_a_non_viable_burst_is_not_reported_as_a_failed_request(client):
    """12 A for 10 s deposits 216 J and moves the case 0.2 degC -- correctly refused. The refusal
    must arrive as data, not as an error, or the front end throws before it can show it."""
    c, _d = client
    r, j = predict(c, 12.0, 10.0)
    assert r.status_code == 200
    assert j["ok"] is True, "envelope ok must stay true: api() throws on ok:false"
    assert j["viable"] is False
    assert "CASE rise" in j["why"]
    assert j["prediction"]["case_c"] < 1.0


def test_a_viable_burst_says_so(client):
    c, _d = client
    r, j = predict(c, 25.0, 60.0)
    assert r.status_code == 200 and j["ok"] is True
    assert j["viable"] is True and j["why"] == ""
    assert j["prediction"]["case_c"] >= 3.0


def test_an_unsafe_burst_is_refused_on_the_winding_nobody_can_see(client):
    c, _d = client
    _r, j = predict(c, 30.0, 150.0)
    assert j["ok"] is True and j["viable"] is False
    assert "winding" in j["why"]


def test_the_panel_default_opens_on_a_viable_burst(client):
    """The panel used to open on 6 A x 5 s = 45 J, which is refused. The first thing an operator
    sees should be a runnable burst, not a wall of red."""
    c, _d = client
    html = io.open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    amps = float(re.search(r'id="th-amps"[^>]*value="([\d.]+)"', html).group(1))
    dur = float(re.search(r'id="th-dur"[^>]*value="(\d+)"', html).group(1))
    _r, j = predict(c, amps, dur)
    assert j["viable"] is True, "index.html defaults {} A x {} s: {}".format(amps, dur, j["why"])


def test_no_endpoint_overloads_the_envelope_ok_field(client):
    """The general form of the bug. `ok` belongs to _ok(); any endpoint that also passes ok= is
    telling every front-end caller that the request failed."""
    src = io.open(os.path.join(os.path.dirname(STATIC), "server.py"), encoding="utf-8").read()
    bad = re.findall(r"_ok\([^)]*\bok=", src)
    assert not bad, bad


def test_the_front_end_reads_the_verdict_field_that_is_actually_sent(client):
    js = io.open(os.path.join(STATIC, "thermal.js"), encoding="utf-8").read()
    assert "d.viable" in js
    assert "TH.pred.viable" in js
    assert not re.search(r"\bd\.ok\b", js), "thermal.js still reads the envelope field"


# ===================================================================== motion guards
def test_start_requires_the_rotor_to_be_declared_clamped(client):
    c, _d = client
    r = c.post("/api/thermal/start", json={"motor": "left.thigh", "amps": 25.0,
                                           "duration_s": 60.0})
    assert r.status_code == 400 and "rotor declared" in r.get_json()["error"]


def test_identify_requires_an_explicit_free_confirmation(client):
    c, _d = client
    r = c.post("/api/thermal/identify", json={"motor": "left.thigh"})
    assert r.status_code == 400 and "confirm" in r.get_json()["error"]


def test_identify_refuses_an_unknown_joint(client):
    c, _d = client
    r = c.post("/api/thermal/identify", json={"motor": "middle.knee", "confirm_free": True})
    assert r.status_code == 400 and "unknown motor" in r.get_json()["error"]


def test_identify_starts_and_publishes_live_state(client):
    c, d = client
    r = c.post("/api/thermal/identify",
               json={"motor": "left.thigh", "amp_deg": 5.0, "duration_s": 2.0,
                     "confirm_free": True})
    assert r.status_code == 200, r.get_json()
    st = r.get_json()["state"]
    assert st["identify"] is None or st["identify"]["motor"] == "left.thigh"
    c.post("/api/thermal/identify/stop", json={})


def test_every_motor_name_the_panel_offers_is_one_the_daemon_accepts(client):
    c, _d = client
    j = c.get("/api/thermal/runs").get_json()
    assert set(j["motors"]) == set(paths.MOTOR_NAMES)


def test_runs_endpoint_publishes_the_real_limits_the_slider_must_use(client):
    """The slider shipped capped at 30 s while the daemon allowed 180 -- and 30 s of readable
    burst barely exists, so the cap alone could make every reachable setting non-viable."""
    c, _d = client
    lim = c.get("/api/thermal/runs").get_json()["limits"]
    html = io.open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    dur_max = float(re.search(r'id="th-dur"[^>]*max="(\d+)"', html).group(1))
    amp_max = float(re.search(r'id="th-amps"[^>]*max="([\d.]+)"', html).group(1))
    assert dur_max == lim["max_duration_s"]
    assert amp_max == lim["max_amps"]


# ===================================================================== journal noise
@pytest.mark.parametrize("path", ["/api/telemetry_raw", "/api/sensors_raw"])
@pytest.mark.parametrize("since", ["undefined", "null", "", "NaN", "-3"])
def test_a_fresh_page_cursor_does_not_crash_the_stream_endpoints(client, path, since):
    """A page sends ?since=undefined on its first poll, which used to 500 with a traceback in the
    journal on every load -- noise sitting on top of the log you read when something real breaks."""
    c, _d = client
    assert c.get("{}?since={}".format(path, since)).status_code == 200


# ===================================================================== request-storm regression
def test_the_plan_poll_key_never_reads_a_field_the_hot_state_lacks(client):
    """The panel re-asks /identify/plan when its cache key changes. Keying on a field that only
    exists in the COLD half of the state made the key flip 4x a second — hot said `undefined`,
    cold said the epoch — and each plan response ran applyState, which flipped it back. A
    self-sustaining storm inside the process that runs the 200 Hz CAN loop (2026-08-29)."""
    c, _d = client
    hot = c.get("/api/state_hot").get_json()
    js = io.open(os.path.join(STATIC, "thermal.js"), encoding="utf-8").read()
    key = re.search(r"const key = `([^`]+)`", js).group(1)
    for field in re.findall(r"st\.(\w+)", key):
        assert field in hot, (
            "thermal.js keys the plan poll on st.{} but /api/state_hot does not carry it, so the "
            "key flips on every hot/cold alternation".format(field))


def test_the_plan_request_is_rate_limited_and_reentrancy_guarded(client):
    js = io.open(os.path.join(STATIC, "thermal.js"), encoding="utf-8").read()
    body = js[js.index("async function thIdentPlan"):js.index("async function thIdentify")]
    assert "planInFlight" in body, "no in-flight guard: a response can re-trigger its own request"
    assert re.search(r"TH\.planAt\s*<\s*\d{3,}", body), "no minimum interval between plan requests"


def test_the_plan_endpoint_is_read_only(client):
    """It must never queue motion — it is polled."""
    c, d = client
    before = d.get_snapshot()["mode"]
    for _ in range(5):
        r = c.post("/api/thermal/identify/plan", json={"motor": "left.thigh"})
        assert r.status_code == 200
        assert r.get_json()["ok"] is True          # verdict rides in `viable`, never the envelope
    assert d._wiggle_req is None and d._wiggle is None
    assert d.get_snapshot()["mode"] == before


# ===================================================================== canvas runaway regression
def _static(name):
    return io.open(os.path.join(STATIC, name), encoding="utf-8").read()


def test_canvas_autofit_is_opt_in():
    """_fit() sizes the backing store from the element's measured box. On a canvas with no CSS
    width the width ATTRIBUTE drives layout, so writing it changes the measurement that produced
    it and the canvas doubles every frame. #meas-chart did exactly that on 2026-08-29 and, sitting
    in a 1fr grid, dragged four neighbouring panels out to infinity."""
    js = _static("app.js")
    fit = js[js.index("  _fit() {"):js.index("  _timeAxis(")]
    assert "if (!this.autoFit)" in fit, "_fit must no-op unless the caller opted in"
    assert "this.autoFit = false;" in js, "autoFit must default to off"


def test_every_autofitting_canvas_gets_its_width_from_css():
    """The opt-in is only safe for a canvas whose layout width comes from a stylesheet."""
    js, css = _static("app.js"), _static("style.css")
    assert 'for (const k of ["pos", "cur", "temp"]) charts[n][k].autoFit = true;' in js, (
        "the only autoFit opt-in should be the motor-card charts; a new one needs a CSS width")
    rule = re.search(r"\.motor-card canvas \{([^}]*)\}", css)
    assert rule and "width: 100%" in rule.group(1), (
        "motor-card canvases opt into autoFit, so they must be width:100% in CSS")


def test_no_panel_canvas_can_outgrow_its_column():
    css = _static("style.css")
    assert re.search(r"\.panel canvas \{[^}]*max-width:\s*100%", css), (
        "a canvas with a runaway width must still be clamped by its panel")
    assert re.search(r"#meas-chart \{[^}]*max-width:\s*100%", css)


# ===================================================================== top-bar lamps
def test_the_lamps_read_fields_that_partial_states_may_lack(client):
    """/api/state merges hot and cold, but api() calls applyState with the DAEMON SNAPSHOT alone,
    which carries neither calibration nor workspace. Reading them straight off `st` made the
    calibrated badge flip to NOT CALIBRATED on every API call; the lamps must not inherit that."""
    c, _d = client
    snap = c.get("/api/state_hot").get_json()
    assert "calibration" not in snap and "workspace" not in snap, "premise changed"
    js = _static("app.js")
    assert "if (st.calibration) S.cal = st.calibration;" in js
    assert "if (st.workspace) S.ws = st.workspace;" in js
    lamps = js[js.index("function updateLamps"):js.index("/* ============") ]
    assert "st.calibration" not in lamps and "st.workspace" not in lamps, (
        "updateLamps must read the cached S.cal/S.ws, never the possibly-partial state")


def test_the_state_actually_carries_what_the_lamps_need(client):
    c, _d = client
    cold = c.get("/api/state_cold").get_json()
    assert "stage" in cold["calibration"] and "restored_from_disk" in cold["calibration"]
    assert "zero_epoch" in cold["calibration"]
    assert "source" in cold["workspace"] and "legs" in cold["workspace"]


def test_lamp_markup_and_styles_exist():
    html, css = _static("index.html"), _static("style.css")
    assert 'id="lamp-zero"' in html and 'id="lamp-ws"' in html
    for cls in (r"\.lamp\.ok", r"\.lamp\.warn", r"\.lamp\.off"):
        assert re.search(cls + r"\s*\{", css), cls


# ===================================================================== safety bypasses
def test_disabling_a_limit_needs_an_explicit_acknowledgement(client):
    c, d = client
    r = c.post("/api/bypass", json={"name": "torque", "on": True})
    assert r.status_code == 400 and "acknowledgement" in r.get_json()["error"]
    assert d.bypass["torque"] is False


def test_re_enabling_a_limit_never_needs_one(client):
    c, d = client
    d.bypass["speed"] = True
    r = c.post("/api/bypass", json={"name": "speed", "on": False})
    assert r.status_code == 200 and d.bypass["speed"] is False


def test_an_unknown_bypass_is_refused(client):
    c, _d = client
    r = c.post("/api/bypass", json={"name": "gravity", "on": True, "acknowledged": True})
    assert r.status_code == 400 and "unknown bypass" in r.get_json()["error"]


@pytest.mark.parametrize("name", ["workspace", "speed", "torque", "tracking"])
def test_each_bypass_round_trips_and_is_published(client, name):
    c, d = client
    r = c.post("/api/bypass", json={"name": name, "on": True, "acknowledged": True})
    assert r.status_code == 200
    assert r.get_json()["bypass"][name] is True
    assert d.bypass_active() == [name]
    # the snapshot is rebuilt at 20 Hz, so a fresh daemon can still be serving the placeholder
    for _ in range(100):
        if "bypass" in d.get_snapshot():
            break
        time.sleep(0.02)
    assert d.get_snapshot()["bypass"][name] is True


def test_a_bypass_is_never_persisted_across_a_restart(robot):                    # noqa: F811
    """The state a bypass leaves behind outlives the reason for it."""
    import calibration as _cal, daemon as _dm
    d, _c, _b, _dir = robot
    d.bypass["torque"] = True
    fresh = _dm.RobotDaemon(mock=True, calib=_cal.Calibration(), wstore=None, fklut=None, bb=None)
    assert fresh.bypass == {n: False for n in _dm.BYPASS_NAMES}


def test_the_workspace_bypass_does_not_reach_the_joint_hard_limits(robot):       # noqa: F811
    """_validate_pose checks hard bounds BEFORE the workspace, so the bypass cannot touch them."""
    d, cal, _b, _dir = robot
    capture_zero(d, cal)
    d.set_bypass("workspace", True)
    side, role = paths.split_name("left.thigh")
    lo, hi = d._hard_bounds(side, role)
    pose = {n: 0.0 for n in paths.MOTOR_NAMES}
    pose["left.thigh"] = hi + 25.0
    ok, why = d._validate_pose(pose, override=False)
    assert not ok and "hard limit" in why


def test_a_bypass_is_recorded_in_the_flight_recorder(robot):                     # noqa: F811
    import blackbox
    d, _c, b, tmpdir = robot
    d.set_bypass("tracking", True, note="chirp sweep")
    time.sleep(0.6)
    ev = blackbox.read_events(os.path.join(tmpdir, blackbox.EVENTS_NAME))
    hits = [e for e in ev if e.get("kind") == "bypass"]
    assert hits and hits[-1]["name"] == "tracking" and hits[-1]["on"] is True


def test_the_ui_confirms_before_disabling_and_mirrors_the_daemon():
    js = _static("app.js")
    g = js[js.index("function wireGuards"):js.index("function updateGuards")]
    assert "confirm(" in g, "unticking a guard must ask first"
    assert "el.checked = true;" in g, "a refused confirmation must put the guard back"
    u = js[js.index("function updateGuards"):js.index("function buildMotorCards")]
    assert "if (!by) return;" in u, "a partial state must not be read as 'no bypasses'"
    assert "wireGuards();" in js


# ===================================================================== free-rotor sine (HTTP)
def test_a_bad_sine_is_refused_at_the_start_endpoint(client):
    c, _d = client
    r = c.post("/api/thermal/start", json={"motor": "left.thigh", "amps": 25.0,
                                           "duration_s": 60.0, "rotor_mode": "free",
                                           "freq_hz": 100.0, "amp_deg": 10.0})
    assert r.status_code == 400 and "frequency" in r.get_json()["error"]


def test_the_sine_knobs_match_the_law_bounds(client):
    """Same rule as the duration slider: the HTML must offer exactly the range the daemon
    accepts, or a reachable setting is a guaranteed refusal."""
    c, _d = client
    lim = c.get("/api/thermal/runs").get_json()["limits"]
    html = io.open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    freq_max = float(re.search(r'id="th-freq"[^>]*max="([\d.]+)"', html).group(1))
    amp_max = float(re.search(r'id="th-amp"[^>]*max="([\d.]+)"', html).group(1))
    assert freq_max == lim["free_freq_max_hz"]
    assert amp_max == lim["free_amp_max_deg"]
    js = io.open(os.path.join(STATIC, "thermal.js"), encoding="utf-8").read()
    assert "freq_hz" in js and "amp_deg" in js, "the panel never posts the sine it shows"
