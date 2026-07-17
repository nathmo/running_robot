"""CAN bus access for the web UI: real socketcan buses or a mock simulator.

This is the ONLY webui module that constructs CAN messages. The servo-mode protocol constants and
encodings mirror fixed_gait/run_hardware.py:53-82 (which itself copies tools/ak_servo_sweep.py) —
kept here with a python-can-free fallback message class so `--mock` runs on machines without
python-can installed.

MockBus simulates the three CubeMars motors of one leg well enough to exercise every UI flow:
  * status frames (pos/spd/cur/temp/err) broadcast at ~100 Hz per motor
  * SET_POS -> first-order lag toward the commanded position (like the drive's own loop)
  * SET_CURRENT 0 streamed -> motor is limp and can be "backdriven" via drag targets
  * random per-boot encoder offset (and optionally inverted direction) per motor, so the
    zero/direction calibration wizard is exercised for real
"""
import math
import struct
import threading
import time

import numpy as np

import paths  # noqa: F401  (sys.path side effect)

try:
    import can
except ImportError:
    can = None

# servo-mode protocol (run_hardware.py:54-56)
CAN_PACKET_SET_CURRENT = 1
CAN_PACKET_SET_POS = 4
BITRATE = 1_000_000


class _Msg:
    """Minimal stand-in for can.Message (mock mode without python-can)."""
    __slots__ = ("arbitration_id", "data", "is_extended_id")

    def __init__(self, arbitration_id, data, is_extended_id=True):
        self.arbitration_id = arbitration_id
        self.data = bytes(data)
        self.is_extended_id = is_extended_id


def _make_msg(arb_id, data):
    if can is not None:
        return can.Message(arbitration_id=arb_id, data=data, is_extended_id=True)
    return _Msg(arb_id, data)


def set_pos(bus, cid, pos_deg):
    """Position command: deg * 10000, big-endian int32 (run_hardware.py:63-67)."""
    val = int(round(pos_deg * 10_000.0))
    val = max(-2_147_483_648, min(2_147_483_647, val))
    bus.send(_make_msg(cid | (CAN_PACKET_SET_POS << 8), struct.pack(">i", val)))


def set_current(bus, cid, amps):
    """Current command: A * 1000, big-endian int32 (run_hardware.py:70-71)."""
    val = int(round(amps * 1000.0))
    val = max(-2_147_483_648, min(2_147_483_647, val))
    bus.send(_make_msg(cid | (CAN_PACKET_SET_CURRENT << 8), struct.pack(">i", val)))


def parse_status(data):
    """Decode the 8-byte broadcast status frame (run_hardware.py:74-82)."""
    if len(data) < 8:
        return None
    pos = struct.unpack(">h", bytes(data[0:2]))[0] * 0.1
    spd = struct.unpack(">h", bytes(data[2:4]))[0] * 10.0
    cur = struct.unpack(">h", bytes(data[4:6]))[0] * 0.01
    temp = struct.unpack(">b", bytes(data[6:7]))[0]
    err = data[7]
    return {"pos": pos, "spd": spd, "cur": cur, "temp": temp, "err": err}


def open_buses(interface, channels, mock=False, mock_seed=None):
    """Return {channel: bus}. Real socketcan unless mock (or python-can missing -> forced mock)."""
    if not mock and can is None:
        print("!! python-can not installed -> falling back to MOCK buses")
        mock = True
    buses = {}
    for ch in channels:
        if mock:
            buses[ch] = MockBus(ch, seed=mock_seed)
        else:
            buses[ch] = can.Bus(interface=interface, channel=ch, bitrate=BITRATE)
    return buses


# ===================================================================== mock simulator
class _MockMotor:
    """One simulated CubeMars motor in servo mode, state in RAW encoder degrees."""

    def __init__(self, cid, rng):
        self.cid = cid
        # random per-boot encoder zero: raw angle is meaningless until calibrated
        self.raw = float(rng.uniform(-250.0, 250.0))
        # some motors "point the other way": +1 normalized motion decreases raw
        self.direction = -1.0 if rng.random() < 0.4 else 1.0
        self.target = None          # last SET_POS raw target (deg) or None
        self.limp = True            # True while only SET_CURRENT 0 arrives
        self.cur = 0.0              # simulated current draw (A)
        self.temp_base = float(rng.uniform(28.0, 38.0))
        self.t0 = time.time()
        self.drag_target = None     # raw target the "hand" pulls toward when limp
        self.spd = 0.0              # deg/s (converted to fake ERPM in frames)

    def step(self, dt, now):
        prev = self.raw
        if not self.limp and self.target is not None:
            # first-order lag toward the commanded position, rate-limited like a real drive
            err = self.target - self.raw
            max_step = 240.0 * dt                       # ~240 deg/s slew
            self.raw += float(np.clip(err * min(1.0, 12.0 * dt), -max_step, max_step))
            self.cur = float(np.clip(0.25 * err, -6.0, 6.0))
        else:
            self.cur = 0.0
            if self.drag_target is not None:            # hand backdrives the limp joint
                err = self.drag_target - self.raw
                self.raw += float(np.clip(err * min(1.0, 6.0 * dt), -180.0 * dt, 180.0 * dt))
        self.spd = (self.raw - prev) / dt if dt > 0 else 0.0

    def status_frame(self, now):
        temp = self.temp_base + 3.0 * math.sin((now - self.t0) / 30.0) + abs(self.cur)
        pos_i = int(np.clip(round(self.raw * 10.0), -32768, 32767))
        spd_erpm = self.spd / 360.0 * 60.0 * 21         # fake pole-pair scaling, just plausible
        spd_i = int(np.clip(round(spd_erpm / 10.0), -32768, 32767))
        cur_i = int(np.clip(round(self.cur * 100.0), -32768, 32767))
        data = (struct.pack(">h", pos_i) + struct.pack(">h", spd_i) + struct.pack(">h", cur_i)
                + struct.pack(">b", int(temp)) + bytes([0]))
        return _make_msg(self.cid, data)


class MockBus:
    """Duck-types the can.Bus surface the daemon uses: recv(timeout), send(msg), shutdown().

    Physics + status broadcast are advanced lazily inside recv()/send() calls — no extra thread.
    """
    STATUS_HZ = 100.0

    def __init__(self, channel, seed=None):
        self.channel = channel
        rng = np.random.default_rng(seed if seed is None else seed + hash(channel) % 1000)
        self.motors = {cid: _MockMotor(cid, rng) for cid in (104, 105, 106)}
        self._rx = []
        self._lock = threading.Lock()
        self._last_step = time.time()
        self._last_status = time.time()

    def _advance(self):
        now = time.time()
        dt = now - self._last_step
        if dt >= 0.004:
            self._last_step = now
            for m in self.motors.values():
                m.step(dt, now)
        if (now - self._last_status) >= 1.0 / self.STATUS_HZ:
            self._last_status = now
            for m in self.motors.values():
                self._rx.append(m.status_frame(now))
            if len(self._rx) > 600:                     # bound the queue if nobody reads
                del self._rx[:-600]

    def recv(self, timeout=0.0):
        with self._lock:
            self._advance()
            if self._rx:
                return self._rx.pop(0)
        if timeout and timeout > 0:
            time.sleep(min(timeout, 0.005))
            with self._lock:
                self._advance()
                if self._rx:
                    return self._rx.pop(0)
        return None

    def send(self, msg):
        with self._lock:
            self._advance()
            cid = msg.arbitration_id & 0xFF
            cmd = (msg.arbitration_id >> 8) & 0xFF
            m = self.motors.get(cid)
            if m is None:
                return
            if cmd == CAN_PACKET_SET_POS:
                m.target = struct.unpack(">i", bytes(msg.data[:4]))[0] / 10_000.0
                m.limp = False
            elif cmd == CAN_PACKET_SET_CURRENT:
                amps = struct.unpack(">i", bytes(msg.data[:4]))[0] / 1000.0
                if amps == 0.0:
                    m.limp = True
                    m.target = None
                else:
                    # crude torque response: nonzero current accelerates the joint
                    m.limp = False
                    m.target = m.raw + np.sign(amps) * 15.0

    def shutdown(self):
        pass

    # ---- test hooks (used by /api/mock/drag) ----
    def drag(self, cid, raw_target):
        with self._lock:
            if cid in self.motors:
                self.motors[cid].drag_target = raw_target

    def drag_release(self, cid):
        with self._lock:
            if cid in self.motors:
                self.motors[cid].drag_target = None
