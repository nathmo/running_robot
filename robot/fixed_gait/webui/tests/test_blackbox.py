"""Black box + pre-move guard, on MockBus only — no hardware, no CAN, no robot.

The last test is the ACCEPTANCE case from BLACKBOX_TASK.md: reproduce the 2026-08-10 incident
shape (capture a zero, move the raw origin underneath the daemon, command an absolute move) and
show that the guard refuses it, the evidence lands on disk, and blackbox_read --postmortem says
what happened without anyone reading a raw file.

    python -m pytest robot/fixed_gait/webui/tests -v
"""
import io
import json
import os
import time
from contextlib import redirect_stdout

import numpy as np
import pytest

import blackbox
import blackbox_read
import calibration
import canio
import daemon as daemon_mod
import dynstore
import paths

N = paths.N_MOTORS


# ===================================================================== helpers
def row(t_mono, t_wall=None, mode=0, estop=0, slip=0, pos_raw=0.0, cmd_raw=None, spd=0.0):
    """One push_sample() tuple, laid out exactly as blackbox.ROW_HEAD + ROW_BLOCKS."""
    vals = {"pos_raw": pos_raw, "pos_norm": pos_raw, "cmd_raw": pos_raw if cmd_raw is None
            else cmd_raw, "cmd_norm": 0.0, "spd": spd, "cur": 0.0, "temp": 35.0, "err": 0.0}
    out = [t_mono, t_wall if t_wall is not None else time.time(), 0.005, mode, estop, slip]
    for f in blackbox.ROW_BLOCKS:
        out.extend([vals[f]] * N)
    return tuple(out)


def drain_writer(bb, cycles=6, period=0.12):
    for _ in range(cycles):
        time.sleep(period)


def events_of(d, kind=None):
    ev = blackbox.read_events(os.path.join(d, blackbox.EVENTS_NAME))
    return [e for e in ev if kind is None or e.get("kind") == kind]


def files_of(d, ext):
    return sorted(f for f in os.listdir(d) if f.endswith(ext))


@pytest.fixture
def bb(tmp_path):
    """A recorder writing into a throwaway directory, tuned for fast tests."""
    b = blackbox.BlackBox(directory=str(tmp_path), heartbeat_s=0.25, post_trigger_s=0.3,
                          space_check_s=0.2)
    b.start()
    yield b
    b.stop()


# ===================================================================== Tier A: rotation + budget
def test_rotation_and_budget_enforcement(tmp_path):
    """Segments rotate at the size limit, and the oldest are deleted to stay inside the budget.
    A full SD card must never be able to stop the robot, so the recorder polices itself."""
    b = blackbox.BlackBox(directory=str(tmp_path), segment_max_bytes=4000,
                          budget_bytes=30_000, min_free_bytes=0, heartbeat_s=99,
                          post_trigger_s=0.2, space_check_s=0.1)
    b.start()
    try:
        t0 = time.monotonic()
        for i in range(4000):
            b.push_sample(row(t0 + i * 0.005))
            if i % 400 == 0:
                time.sleep(0.12)
        drain_writer(b, cycles=12)
    finally:
        b.stop()

    segs = files_of(str(tmp_path), blackbox.SEG_EXT)
    assert len(segs) >= 2, f"expected the segment to rotate, got {segs}"
    # the size limit is a rotation THRESHOLD checked after each drained batch, so a segment may
    # overshoot by at most one batch — never unboundedly
    for s in segs[:-1]:
        assert os.path.getsize(tmp_path / s) <= 4000 + 400 * blackbox.RECORD_BYTES

    total = sum(os.path.getsize(tmp_path / f) for f in os.listdir(tmp_path))
    assert total <= 30_000 + blackbox.RECORD_BYTES * 40, f"budget blown: {total} B"

    dropped = events_of(str(tmp_path), "space.dropped")
    assert dropped, "deleting history for space must be logged, not silent"
    assert dropped[-1]["removed"], "the event must name what was lost"


def test_tier_a_is_decimated_not_full_rate(tmp_path):
    """The whole point of the tiering: 200 Hz never touches the disk continuously."""
    b = blackbox.BlackBox(directory=str(tmp_path), heartbeat_s=99, space_check_s=99)
    b.start()
    try:
        t0 = time.monotonic()
        for i in range(1000):
            b.push_sample(row(t0 + i * 0.005))
        drain_writer(b)
    finally:
        b.stop()
    h, rec = blackbox.read_segment(tmp_path / files_of(str(tmp_path), blackbox.SEG_EXT)[0])
    assert 90 <= len(rec) <= 110, f"expected ~1/10 of 1000 samples on disk, got {len(rec)}"
    assert h["rate_hz"] == pytest.approx(20.0)
    assert h["motor_names"] == list(paths.MOTOR_NAMES)          # right leg first, explicitly


# ===================================================================== queue-full accounting
def test_queue_full_drops_are_counted_and_visible_in_the_data(tmp_path):
    """A black box that lies about its own gaps is worse than none: the drop counter has to end up
    IN the record stream, so a reader sees exactly where the hole is."""
    b = blackbox.BlackBox(directory=str(tmp_path), sample_queue_max=50, heartbeat_s=99,
                          space_check_s=99)
    t0 = time.monotonic()
    accepted = sum(b.push_sample(row(t0 + i * 0.005)) for i in range(200))   # writer NOT started
    assert accepted == 50
    assert b._dropped == 150

    b.start()
    try:
        drain_writer(b, cycles=3)
        for i in range(20):                     # these are accepted again, after the flood
            b.push_sample(row(t0 + (200 + i) * 0.005))
        drain_writer(b, cycles=3)
        ring = b._ring_snapshot()
    finally:
        b.stop()
    assert ring["drop"].max() == 150, "the accepted record must carry the drop count so far"
    assert ring["drop"].min() == 0
    assert b.status()["dropped"] == 150


def test_push_sample_is_cheap_enough_for_the_200hz_loop(tmp_path):
    """Hard requirement: the control thread only appends to a bounded deque. If this ever grows a
    lock or an allocation storm, the 200 Hz tick pays for it."""
    b = blackbox.BlackBox(directory=str(tmp_path), sample_queue_max=100_000)
    r = row(time.monotonic())
    for _ in range(1000):                                        # warm up
        b.push_sample(r)
    t0 = time.perf_counter()
    for _ in range(20_000):
        b.push_sample(r)
    per_call_us = (time.perf_counter() - t0) / 20_000 * 1e6
    assert per_call_us < 50.0, f"push_sample costs {per_call_us:.1f} us of a 5000 us tick"


# ===================================================================== Tier B: pre-trigger history
def test_dump_contains_genuine_pre_trigger_history(bb, tmp_path):
    """The tier that would have solved 2026-08-10: the dump must contain the 200 Hz window from
    BEFORE the trigger, not just what happened after someone noticed."""
    now = time.monotonic()
    for i in range(4000):                                   # 20 s of 200 Hz, ending 'now'
        bb.push_sample(row(now - 20.0 + i * 0.005, pos_raw=float(i)))
    drain_writer(bb, cycles=4)

    name = bb.trigger_dump("unit_test_trigger")
    assert name and name.endswith(blackbox.DUMP_EXT)
    drain_writer(bb, cycles=8)

    h, rec = blackbox.read_segment(tmp_path / name)
    assert h["tier"] == "B"
    assert h["rate_hz"] == pytest.approx(200.0)
    assert h["n_pre_trigger"] > 0
    assert h["pre_trigger_s"] >= 10.0, f"only {h['pre_trigger_s']} s of pre-trigger history"
    t_trig = h["trigger"]["t_trig_mono"]
    assert (rec["t_mono"] <= t_trig).sum() >= 2000
    assert rec["pos_raw"][0, 0] < rec["pos_raw"][-1, 0]        # oldest first, chronological
    assert "config_hash" in h and "config" in h                 # self-describing without context


def test_dump_cooldown_is_per_reason(bb, tmp_path):
    """A repeating trigger must not write the same window over and over — but a NEW kind of
    trigger always gets its own file, because that is the one nobody has evidence for."""
    now = time.monotonic()
    for i in range(200):
        bb.push_sample(row(now - 1.0 + i * 0.005))
    bb.trigger_dump("reason_a")
    drain_writer(bb, cycles=8)
    bb.trigger_dump("reason_a")                 # same reason, inside the cooldown -> suppressed
    bb.trigger_dump("reason_b")                 # different reason -> must still be written
    drain_writer(bb, cycles=8)

    reasons = []
    for f in files_of(str(tmp_path), blackbox.DUMP_EXT):
        reasons.append(blackbox.read_header(tmp_path / f)["trigger"]["reason"])
    assert reasons.count("reason_a") == 1
    assert "reason_b" in reasons
    assert events_of(str(tmp_path), "dump.suppressed"), "suppression must be on the record"


# ===================================================================== Tier C: boot / heartbeat
def test_boot_and_heartbeat_records(bb, tmp_path):
    time.sleep(1.0)
    start = events_of(str(tmp_path), "daemon.start")
    hb = events_of(str(tmp_path), "heartbeat")
    assert len(start) == 1
    assert len(hb) >= 2, "the heartbeat is what dates a power cut; it must actually beat"
    for e in start + hb:
        assert e["boot_id"] and e["session_id"]
        assert "t_mono" in e and "t_wall" in e and "wall_trusted" in e
    assert start[0]["motor_names"] == list(paths.MOTOR_NAMES)


def test_power_off_is_inferred_from_the_last_heartbeat(tmp_path):
    """A process that dies with the power cannot log its own death. The reader has to infer the
    power-off time from the last heartbeat, and say so."""
    p = tmp_path / blackbox.EVENTS_NAME
    lines = []

    def ev(sid, kind, t, **f):
        lines.append(json.dumps({"kind": kind, "t_mono": t, "t_wall": 1e9 + t, "uptime_s": t,
                                 "boot_id": "boot-" + sid, "session_id": sid,
                                 "wall_trusted": False, **f}))

    ev("aaaa", "daemon.start", 0.0, boot_t_wall=1e9)            # session A: killed by the power
    for k in range(1, 6):
        ev("aaaa", "heartbeat", 8.0 * k)
    ev("bbbb", "daemon.start", 0.0, boot_t_wall=1e9 + 200)      # session B: clean, with a stall
    ev("bbbb", "heartbeat", 8.0)
    ev("bbbb", "heartbeat", 16.0)
    ev("bbbb", "heartbeat", 90.0)                               # <- a 74 s gap inside the session
    ev("bbbb", "heartbeat", 98.0)
    ev("bbbb", "daemon.stop", 100.0)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    events, files = blackbox_read.load_dir(str(tmp_path))
    ss = {s["session_id"]: s for s in blackbox_read.sessions(events)}
    assert ss["aaaa"]["clean_stop"] is False
    assert ss["aaaa"]["last_hb"]["t_mono"] == 40.0              # the inferred power-off instant
    assert ss["bbbb"]["clean_stop"] is True
    assert len(ss["bbbb"]["gaps"]) == 1
    assert ss["bbbb"]["gaps"][0]["seconds"] == pytest.approx(74.0)

    out = io.StringIO()
    with redirect_stdout(out):
        blackbox_read.postmortem(str(tmp_path), events, files)
    text = out.getvalue()
    assert "POWER LOSS" in text
    assert "RECORDING GAP" in text
    assert "cannot log its own death" in text
    assert "UNTRUSTED CLOCK" in text, "an unsynced clock must be labelled, not printed as fact"


# ===================================================================== reader round-trip
def test_reader_round_trip_and_torn_tail(bb, tmp_path):
    now = time.monotonic()
    pushed = [row(now + i * 0.005, pos_raw=1.5 * i, cmd_raw=1.5 * i + 0.25, mode=1, slip=i)
              for i in range(500)]
    for r in pushed:
        bb.push_sample(r)
    drain_writer(bb, cycles=6)
    bb.stop()

    seg = tmp_path / files_of(str(tmp_path), blackbox.SEG_EXT)[0]
    h, rec = blackbox.read_segment(seg)
    assert len(rec) == 50                                       # 500 / TIER_A_DIV
    assert rec["pos_raw"][1, 0] == pytest.approx(1.5 * 10, abs=1e-3)
    assert rec["cmd_raw"][1, 0] == pytest.approx(1.5 * 10 + 0.25, abs=1e-3)
    assert rec["mode"][0] == 1

    frame = blackbox_read.to_frame(rec)
    assert "right.abd.pos_raw" in frame                          # MOTOR_NAMES order is explicit
    assert len(frame["right.abd.pos_raw"]) == len(rec)

    csv = tmp_path / "out.csv"
    blackbox_read.write_csv(rec, str(csv))
    assert csv.exists() and csv.stat().st_size > 0

    # killed mid-write: the tail record is torn. Append-only means the file is still readable.
    with open(seg, "ab") as f:
        f.write(b"\x01" * (blackbox.RECORD_BYTES // 2))
    h2, rec2 = blackbox.read_segment(seg)
    assert len(rec2) == len(rec)
    assert h2["_torn_bytes"] == blackbox.RECORD_BYTES // 2


def test_writer_death_does_not_break_the_push_api(bb):
    """Non-goal: the robot must never depend on the recorder. If the writer dies, pushing keeps
    working (as a counter) and the failure is surfaced, not thrown."""
    bb.error = "pretend the writer thread exploded"
    assert bb.push_sample(row(time.monotonic())) is False
    assert bb.trigger_dump("nope") is None
    assert bb.status()["alive"] is False and bb.status()["error"]


# ===================================================================== the pre-move guard
@pytest.fixture
def robot(tmp_path):
    """A daemon on MockBus with a recorder, and NO writes to the real data/ directory."""
    cal = calibration.Calibration()
    cal.save = lambda *a, **k: None                    # never touch the operator's real calibration
    b = blackbox.BlackBox(directory=str(tmp_path), heartbeat_s=1.0, post_trigger_s=0.4,
                          space_check_s=1.0)
    blackbox.install(b)
    dyn = dynstore.DynConfig()                    # defaults only; never saved to disk
    b.set_config_provider(lambda: {"calibration": cal.snapshot(),
                                   "dynamics": dyn.as_dict()})
    b.start()
    d = daemon_mod.RobotDaemon(mock=True, calib=cal, wstore=None, fklut=None, bb=b)
    d.start()
    assert d._started_ok.wait(5.0)
    for _ in range(200):
        if all(m.pos is not None for m in d.motors):
            break
        time.sleep(0.02)
    yield d, cal, b, str(tmp_path)
    d.stop_event.set()
    d.join(2.0)
    b.stop()
    blackbox.install(None)


def capture_zero(d, cal):
    ok, why = cal.set_zero(d.latest_raw_positions())
    assert ok, why
    for n in paths.MOTOR_NAMES:
        cal.confirm(n)
    assert cal.complete


def wait_mode(d, want, timeout=3.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if d.get_snapshot().get("mode") == want:
            return True
        time.sleep(0.02)
    return d.get_snapshot().get("mode") == want


def test_guard_does_not_false_fire_on_the_normal_workflow(robot):
    """Homing straight after a zero capture is the CORRECT operator action — the legs are limp and
    sagging and homing is what catches them. The guard must not add a step to that."""
    d, cal, b, _dir = robot
    capture_zero(d, cal)
    ok, why = d.home()
    assert ok, f"the guard refused the normal zero -> home workflow: {why}"
    assert wait_mode(d, "MANUAL"), d.get_snapshot()
    assert d._guard_latched == ""


def test_hold_before_moving_commands_the_measured_raw_not_the_calibration(robot):
    """(a) The first CAN command after enabling is set_pos(where you already are). It never
    consults the offsets, so it is safe even against a completely wrong zero, and it stops the
    sag immediately."""
    d, cal, b, _dir = robot
    capture_zero(d, cal)
    before = dict(d.latest_raw_positions())
    d.request_mode("MANUAL")
    assert wait_mode(d, "MANUAL")
    time.sleep(0.05)
    snap = d.get_snapshot()
    assert snap["premove"]["holding"] is True
    for n, raw in before.items():
        assert d._last_cmd_raw[n] == pytest.approx(raw, abs=1e-6), \
            f"{n} was commanded somewhere other than where it already was"
    time.sleep(daemon_mod.HOLD_BEFORE_MOVE_S + 0.2)
    assert d.get_snapshot()["premove"]["holding"] is False


def test_guard_fires_when_the_origin_moves_under_a_valid_zero(robot):
    """(b) The 2026-08-10 sequence: a good zero, then the board renumbers its origin, then an
    absolute move. The move must be refused, with both raw poses on the record."""
    d, cal, b, bdir = robot
    capture_zero(d, cal)
    m = d.by_name["left.cam"]
    raw_before = m.pos
    m.bus.shift_origin(m.cid, 400.0)              # the shaft does not move; the numbering does
    time.sleep(0.3)

    ok, why = d.home()
    assert not ok, "the guard let a move through against a moved origin"
    assert "left.cam" in why or "Re-zero" in why
    assert not wait_mode(d, "MANUAL", timeout=0.6), "the robot must stay LIMP"
    assert d._guard_latched

    jumps = events_of(bdir, "raw.jump")
    assert jumps, "an origin move while limp must be detected and logged"
    assert jumps[-1]["motor"] == "left.cam"
    assert abs(jumps[-1]["delta_deg"] - 400.0) < 5.0
    assert abs(jumps[-1]["spd"]) < daemon_mod.RAW_JUMP_SPD_MAX, \
        "the whole discriminator is that the drive reports no motion"

    ref = events_of(bdir, "premove.refused")
    assert ref, "the refusal must be an event, not just a UI string"
    r = ref[-1]
    assert r["raw_now"]["left.cam"] == pytest.approx(raw_before + 400.0, abs=2.0)
    assert r["raw_at_last_zero"]["left.cam"] == pytest.approx(raw_before, abs=2.0)
    assert abs(r["compare"]["left.cam"]["delta"] - 400.0) < 5.0


def test_guard_fires_on_a_calibration_restored_after_a_power_cycle(tmp_path):
    """(b, the reboot case) The drives re-randomise their raw origin on every power cycle, so a
    calibration restored from disk cannot be vouched for — and we were not running to watch."""
    cal = calibration.Calibration()
    cal.save = lambda *a, **k: None
    cal.zero_raw = {n: 0.0 for n in paths.MOTOR_NAMES}      # "captured" before the power cycle
    cal.offsets = dict(cal.zero_raw)
    cal.stage = "complete"
    cal.confirmed = {n: True for n in paths.MOTOR_NAMES}
    cal.restored_from_disk = True
    cal.zero_epoch = 7

    d = daemon_mod.RobotDaemon(mock=True, calib=cal, wstore=None, fklut=None, bb=None)
    d.start()
    assert d._started_ok.wait(5.0)
    try:
        for _ in range(200):
            if all(m.pos is not None for m in d.motors):
                break
            time.sleep(0.02)
        ok, why = d.home()
        assert not ok
        assert "Re-zero" in why
        assert not wait_mode(d, "MANUAL", timeout=0.6)
    finally:
        d.stop_event.set()
        d.join(2.0)


def test_homing_uses_a_tighter_tracking_threshold(robot):
    """(c) A slow guided slew should track well; 25 deg is far too loose for it."""
    assert daemon_mod.MAX_TRACK_ERR_HOMING_DEG < daemon_mod.MAX_TRACK_ERR_DEG
    d, cal, b, _dir = robot
    capture_zero(d, cal)
    assert d.home()[0]
    assert wait_mode(d, "MANUAL")
    time.sleep(daemon_mod.HOLD_BEFORE_MOVE_S + 0.3)
    assert d.get_snapshot()["mode"] == "MANUAL", "a healthy slow home must not trip"


def test_travel_budget_is_bounded_by_the_joint_range(robot):
    """(d) The backstop. left.cam reached ~1.9 output turns on a +-88 deg joint on 2026-08-10; the
    budget cuts any guided move at 1.3x the joint's own range whatever the calibration claims."""
    d, cal, b, _dir = robot
    lo, hi = d._hard_bounds("left", "cam")
    budget = daemon_mod.TRAVEL_BUDGET_FACTOR * (hi - lo)
    assert budget < 360.0, f"the budget ({budget:.0f} deg) must cut well under one output turn"

    capture_zero(d, cal)
    assert d.home()[0]
    assert wait_mode(d, "MANUAL")
    time.sleep(daemon_mod.HOLD_BEFORE_MOVE_S + 0.2)        # the guided move arms (and zeroes) it
    d._travel = {n: 1e6 for n in paths.MOTOR_NAMES}        # pretend it has swept forever
    d._travel_prev = {n: m.pos for n, m in d.by_name.items()}
    with d.lock:
        d._manual_targets = {n: (40.0 if n.endswith("cam") else 0.0) for n in paths.MOTOR_NAMES}
    assert wait_mode(d, "ESTOPPED", timeout=3.0), "an endless guided move must trip"
    assert "travel budget" in d.get_snapshot()["estop"]["reason"]


def test_trip_raises_an_event_and_a_dump(robot):
    d, cal, b, bdir = robot
    capture_zero(d, cal)
    d._trip("unit test trip")
    time.sleep(1.2)
    trips = events_of(bdir, "trip")
    assert trips and trips[-1]["reason"] == "unit test trip"
    assert set(trips[-1]["raw"]) == set(paths.MOTOR_NAMES)
    # the trip may be COALESCED into a dump already in flight (the zero capture's) — same 40 s
    # window, and every reason that landed in it is recorded in `also`
    covered = []
    for f in files_of(bdir, blackbox.DUMP_EXT):
        t = blackbox.read_header(os.path.join(bdir, f))["trigger"]
        covered += [t["reason"]] + (t.get("also") or [])
    assert "trip" in covered


# ===================================================================== CAN bus down
class _DeadBus(canio.MockBus):
    """A bus whose sends fail the way socketcan does when nothing on the wire ACKs: the 10-frame
    TX queue fills and send() raises ENOBUFS. This is what the Pi does at boot with the motor
    power off — measured 2026-08-10, first run of the daemon under systemd."""

    def send(self, msg):
        raise OSError(105, "No buffer space available")

    def recv(self, timeout=0.0):
        return None                      # unpowered drives broadcast nothing either


def test_a_bus_that_will_not_take_frames_does_not_kill_the_daemon(tmp_path, monkeypatch):
    """It used to: an unhandled ENOBUFS inside _setup()'s preflight took the whole daemon thread
    down, leaving no control loop and — before the black box — no record of why."""
    monkeypatch.setattr(canio, "MockBus", _DeadBus)
    canio._send_errors.clear()
    cal = calibration.Calibration()
    cal.save = lambda *a, **k: None
    b = blackbox.BlackBox(directory=str(tmp_path), heartbeat_s=1.0, post_trigger_s=0.3,
                          space_check_s=1.0)
    blackbox.install(b)
    b.start()
    d = daemon_mod.RobotDaemon(mock=True, calib=cal, wstore=None, fklut=None, bb=b)
    d.start()
    try:
        assert d._started_ok.wait(6.0), "the daemon must survive setup on a dead bus"
        time.sleep(1.5)
        snap = d.get_snapshot()
        assert snap["daemon_alive"] is True, "the control loop must keep running"
        assert snap["loop_error"] is None
        assert sum(snap["can_errors"].values()) > 0, "the failures must be counted, not swallowed"
        assert d.get_snapshot()["mode"] == "LIMP"
        # ...and motion is still refused: with no frames going out nothing reports back, so no
        # motor is alive and there is nothing to calibrate against either
        assert all(m.pos is None for m in d.motors)
        ok, why = d._motion_allowed()
        assert not ok and why
    finally:
        d.stop_event.set()
        d.join(2.0)
        b.stop()
        blackbox.install(None)
        canio._send_errors.clear()

    evs = events_of(str(tmp_path), "can.error")
    assert evs, "a bus that will not take frames must be on the timeline"
    assert evs[0]["kind"] == "can.error" and "105" in str(evs[0]["error"])
    assert not events_of(str(tmp_path), "daemon.crash")


def test_a_dead_bus_is_backed_off_not_hammered(monkeypatch):
    """MEASURED on the robot 2026-08-10: three failing sends per tick cost ~11 ms against a 5 ms
    budget, and the 200 Hz loop ran at 60 Hz. A failing socketcan send is enormously more expensive
    than a successful one, so a channel that fails every time must be retried slowly."""
    monkeypatch.setattr(canio, "MockBus", _DeadBus)
    canio._send_errors.clear()
    canio._send_state.clear()
    bus = canio.MockBus("can1")
    t0 = time.monotonic()
    attempts_at_backoff = None
    for i in range(2000):                                   # ~10 s of a 200 Hz loop
        canio.set_current(bus, 104, 0.0)
        if attempts_at_backoff is None and canio._send_errors.get("can1", 0) >= \
                canio.CHANNEL_BACKOFF_AFTER:
            attempts_at_backoff = i + 1
    elapsed = time.monotonic() - t0
    st = canio.send_stats()["can1"]
    assert st["backoff"] is True
    assert st["skipped"] > 1500, "most frames must be skipped, not attempted"
    # attempts after the backoff engages are bounded by the retry rate, not by the call rate
    max_expected = canio.CHANNEL_BACKOFF_AFTER + elapsed / canio.CHANNEL_BACKOFF_S + 2
    assert st["errors"] <= max_expected, \
        f"{st['errors']} attempts in {elapsed:.3f}s — the dead bus is still being hammered"

    # ...and one success puts it straight back to full rate, so a healthy bus pays nothing
    canio._send_state["can1"]["next_try"] = 0.0
    monkeypatch.setattr(bus, "send", lambda msg: None)
    assert canio.set_current(bus, 104, 0.0) is True
    assert canio.send_stats()["can1"]["backoff"] is False
    canio._send_errors.clear()
    canio._send_state.clear()


# ===================================================================== ACCEPTANCE
@pytest.mark.slow
def test_acceptance_reproduces_the_2026_08_10_incident(robot):
    """The acceptance case, end to end.

    Capture a zero, mutate the raw origin underneath the daemon, command an absolute move. The
    guard must refuse it, an event must record the mismatch with BOTH raw poses, a Tier B dump must
    contain >= 10 s of pre-trigger 200 Hz data, and --postmortem must state what happened without
    a human reading raw files.
    """
    d, cal, b, bdir = robot
    capture_zero(d, cal)
    time.sleep(11.5)                       # let a real >10 s pre-trigger window accumulate

    m = d.by_name["left.cam"]
    raw_before = m.pos
    m.bus.shift_origin(m.cid, 400.0)
    time.sleep(0.4)
    ok, why = d.home()
    assert not ok, "the incident's own command must be refused"
    time.sleep(2.0)                        # let the post-trigger tail close the dump

    # --- a dump with genuine pre-trigger history at the full rate
    dumps = []
    for f in files_of(bdir, blackbox.DUMP_EXT):
        h = blackbox.read_header(os.path.join(bdir, f))
        reasons = [h["trigger"]["reason"]] + (h["trigger"].get("also") or [])
        if {"raw_origin_jump", "premove_guard_refused"} & set(reasons):
            dumps.append((f, h))
    assert dumps, "the incident produced no Tier B dump"
    f, h = max(dumps, key=lambda x: x[1]["pre_trigger_s"])
    assert h["pre_trigger_s"] >= 10.0, f"only {h['pre_trigger_s']} s of pre-trigger data"
    assert h["rate_hz"] == pytest.approx(200.0)
    assert h["config_hash"] and h["config"]["calibration"]["stage"] == "complete"

    hh, rec = blackbox.read_segment(os.path.join(bdir, f))
    pre = rec["t_mono"] <= h["trigger"]["t_trig_mono"]
    assert pre.sum() > 500
    i = paths.MOTOR_NAMES.index("left.cam")
    assert np.isfinite(rec["pos_raw"][:, i]).all()
    assert abs(rec["pos_raw"][-1, i] - rec["pos_raw"][0, i]) > 300.0, "the jump must be in the data"

    # --- the postmortem tells the story
    events, files = blackbox_read.load_dir(bdir)
    out = io.StringIO()
    with redirect_stdout(out):
        rc = blackbox_read.postmortem(bdir, events, files)
    text = out.getvalue()
    assert rc == 0
    assert "WHAT HAPPENED" in text
    assert "pre-move guard REFUSED" in text
    assert "left.cam" in text
    assert "the joint did not move, the numbering did" in text
    assert f"{raw_before:.1f}" in text or "400" in text
    assert "pos_raw AT THE LAST ZERO CAPTURE" in text
    assert "drive gains" in text                      # question 4: config live at that instant
