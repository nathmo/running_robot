#!/usr/bin/env python3
"""ROBOT TOOL. Excite ONE motor with a known current schedule and log its temperature response.

    # dry run first, on the desktop, with the mock bus -- no hardware, no CAN
    python robot/deploy/thermal_calibrate.py --motor left.thigh --mock --steps 6:20,0:60

    # on the robot, with the webui daemon STOPPED
    sudo systemctl stop runningrobot-webui.service
    python robot/deploy/thermal_calibrate.py --motor left.thigh --steps 6:180,0:900 \\
        --joint-is-blocked

THE EXPERIMENT
--------------
A blocked rotor at constant current is the cleanest thermal test there is: the shaft does no
mechanical work, so every watt delivered becomes heat, and the heat input is exactly k_cu * I^2
with no efficiency term to argue about. Hold a current, watch the reported temperature rise, cut
it, watch it fall. The rise identifies the fast (winding) dynamics and the plateau identifies the
steady-state thermal resistance; the fall identifies the slow (case-to-ambient) dynamics with no
input at all, which is why the cooldown must be LONG -- it is not dead time, it is the half of the
experiment with the best signal-to-noise.

Use at least two current levels. The model says power goes as I^2; a single level cannot tell that
apart from any other monotone law, and the whole safety argument rests on the square.

    default schedule: 6 A for 180 s, off for 900 s, 9 A for 150 s, off for 1500 s

WHY THIS SCRIPT IS PARANOID
---------------------------
On 2026-08-26 a single malformed CAN frame put 62.5 A into a stalled rotor on this robot. It was
caught in about a second only because something was watching speed and current at 200 Hz with a
hard abort. This tool deliberately reproduces that watchfulness:

  * SERVO-mode SET_CURRENT only. No force-control frames, no undocumented command bytes. The
    current commanded is the current requested, in the one protocol this robot has used for
    months.
  * ONE motor. Every other drive on both buses is streamed SET_CURRENT 0 for the whole run, so
    the rest of the robot is actively held limp rather than merely unaddressed.
  * Aborts, checked every tick at 200 Hz: rotor speed (the joint is supposed to be BLOCKED, so any
    real motion means the restraint failed), reported temperature, drive error flag, loss of
    telemetry, wall-clock cap, and Ctrl-C.
  * Soft ramp in, and every exit path -- normal, abort, exception, Ctrl-C -- ends in SET_CURRENT 0
    streamed for half a second before the bus is closed.
  * --joint-is-blocked must be passed explicitly. There is no default that energises a motor.

The log is written INCREMENTALLY, so an aborted run still yields usable data.
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]                # robot/
WEBUI = REPO / "fixed_gait" / "webui"
for _p in (str(WEBUI), str(REPO / "fixed_gait")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                                              # noqa: E402  (sys.path bootstrap)
import canio                                              # noqa: E402

LOG_HZ = 50.0            # temperature is an int8; 50 Hz is far more than enough and keeps files small
TICK_HZ = 200.0          # but WATCH at the full rate -- aborts are the reason this loop is fast
RAMP_S = 2.0
DEFAULT_STEPS = "6:180,0:900,9:150,0:1500"

# --- abort thresholds -------------------------------------------------------------------------
# Speed: the joint is supposed to be mechanically blocked. A few hundred ERPM is encoder noise and
# compliance in the restraint; more than that means the rotor is turning and the premise of the
# experiment (zero mechanical power) is false, quite apart from the safety question.
ABORT_ERPM = 400.0
# Temperature: the daemon trips the whole robot at 80 degC (MAX_TEMP_C). Stop well below it -- the
# fit does not need the last 10 degrees, and the REPORTED temperature lags the winding.
ABORT_TEMP_C = 70
# Current: refuse anything above this from the CLI. Raising it needs a hardware argument, not a
# convenience one.
MAX_CLI_AMPS = 15.0
STALE_TELEMETRY_S = 0.25


def parse_steps(s):
    """'6:180,0:900' -> [(6.0, 180.0), (0.0, 900.0)]"""
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        a, _, d = part.partition(":")
        out.append((float(a), float(d)))
    if not out:
        raise ValueError("empty schedule")
    return out


class Aborted(Exception):
    pass


class Run:
    def __init__(self, args):
        self.args = args
        self.name = args.motor
        side, role = paths.split_name(self.name)
        self.channel = paths.SIDE_CHANNEL[side]
        self.cid = paths.ROLE_ID[role]
        self.steps = parse_steps(args.steps)
        self.total_s = sum(d for _, d in self.steps)
        self.rows = []
        self.abort_reason = None
        self._stop = False

    # ------------------------------------------------------------------ schedule
    def amps_at(self, t):
        """Commanded current at run time t, with a soft ramp at every level CHANGE.

        The ramp matters for the fit as much as for the hardware: a step in current that the log
        records as instantaneous but the drive delivered over 20 ms puts a systematic error into
        exactly the fast transient the winding time constant is read from."""
        acc = 0.0
        prev = 0.0
        for a, d in self.steps:
            if t < acc + d:
                if RAMP_S > 0:
                    f = min(1.0, max(0.0, (t - acc) / RAMP_S))
                    return prev + (a - prev) * f
                return a
            acc += d
            prev = a
        return 0.0

    # ------------------------------------------------------------------ safety
    def check(self, st, t_last_rx, now):
        if now - t_last_rx > STALE_TELEMETRY_S:
            raise Aborted("no status frame from {} for {:.2f} s".format(
                self.name, now - t_last_rx))
        if st is None:
            return
        if st["err"]:
            raise Aborted("drive error code {}".format(st["err"]))
        if abs(st["spd"]) > self.args.abort_erpm:
            raise Aborted("rotor is TURNING ({:.0f} ERPM > {:.0f}) -- the joint is not blocked"
                          .format(st["spd"], self.args.abort_erpm))
        if st["temp"] >= self.args.abort_temp:
            raise Aborted("reported temperature {} C >= {} C".format(
                st["temp"], self.args.abort_temp))
        if abs(st["cur"]) > self.args.amps_max + 3.0:
            raise Aborted("measured current {:.1f} A far above the {:.1f} A commanded ceiling"
                          .format(st["cur"], self.args.amps_max))

    # ------------------------------------------------------------------ main
    def execute(self):
        buses = canio.open_buses(self.args.interface, sorted(set(paths.SIDE_CHANNEL.values())),
                                 mock=self.args.mock)
        bus = buses[self.channel]
        others = [(b, cid) for ch, b in buses.items() for cid in paths.ROLE_ID.values()
                  if not (ch == self.channel and cid == self.cid)]
        t0 = time.monotonic()
        t_last_rx = t0
        t_next_log = 0.0
        last = None
        amb = self.args.ambient
        try:
            print("streaming 0 A to every drive for 2 s (settle + prove the buses answer)")
            while time.monotonic() - t0 < 2.0:
                for b, cid in others:
                    canio.set_current(b, cid, 0.0)
                canio.set_current(bus, self.cid, 0.0)
                for m in self._drain(buses):
                    if m is not None:
                        last, t_last_rx = m, time.monotonic()
                time.sleep(1.0 / TICK_HZ)
            if last is None:
                raise Aborted("{} never reported -- refusing to command a drive we cannot watch"
                              .format(self.name))
            print("start temp {} C, ambient {} C".format(last["temp"], amb))
            print("schedule: " + ", ".join("{:.1f} A for {:.0f} s".format(a, d)
                                           for a, d in self.steps))

            t0 = time.monotonic()
            next_tick = t0
            while not self._stop:
                now = time.monotonic()
                t = now - t0
                if t >= self.total_s:
                    break
                amps = self.amps_at(t)
                canio.set_current(bus, self.cid, amps)
                for b, cid in others:
                    canio.set_current(b, cid, 0.0)
                for m in self._drain(buses):
                    if m is not None:
                        last, t_last_rx = m, now
                self.check(last, t_last_rx, now)
                if t >= t_next_log:
                    t_next_log += 1.0 / LOG_HZ
                    self.rows.append((t, amps, last["cur"], last["spd"], last["pos"],
                                      float(last["temp"]), amb))
                next_tick += 1.0 / TICK_HZ
                s = next_tick - time.monotonic()
                if s > 0:
                    time.sleep(s)
                else:
                    next_tick = time.monotonic()
        except Aborted as e:
            self.abort_reason = str(e)
            print("\n!! ABORT: {}".format(e))
        except KeyboardInterrupt:
            self.abort_reason = "operator interrupt"
            print("\n!! interrupted")
        except Exception as e:                          # noqa: BLE001 -- never skip the release
            self.abort_reason = "exception: {!r}".format(e)
            print("\n!! {!r}".format(e))
        finally:
            print("releasing: streaming 0 A to every drive for 0.5 s")
            t_rel = time.monotonic()
            while time.monotonic() - t_rel < 0.5:
                for b in buses.values():
                    for cid in paths.ROLE_ID.values():
                        canio.set_current(b, cid, 0.0)
                time.sleep(0.005)
            for b in buses.values():
                try:
                    b.shutdown()
                except Exception:
                    pass
        return self.abort_reason is None

    def _drain(self, buses):
        """Every frame waiting on OUR motor's bus this tick. Draining the whole queue each tick
        (rather than one recv per iteration) is what keeps the loop at rate -- the drives never
        stop talking."""
        out = []
        b = buses[self.channel]
        for _ in range(64):
            m = b.recv(timeout=0.0)
            if m is None:
                break
            if (m.arbitration_id & 0xFF) == self.cid and len(m.data) >= 8:
                out.append(canio.parse_status(m.data))
        return out

    def save(self, out_dir):
        if not self.rows:
            print("nothing logged")
            return None
        a = np.asarray(self.rows, float)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = Path(out_dir) / "thermal_{}_{}".format(self.name.replace(".", "_"), stamp)
        base.parent.mkdir(parents=True, exist_ok=True)
        meta = {"motor": self.name, "channel": self.channel, "cid": self.cid,
                "steps": self.steps, "ramp_s": RAMP_S, "log_hz": LOG_HZ,
                "ambient_c": self.args.ambient, "mock": bool(self.args.mock),
                "aborted": self.abort_reason, "created": stamp,
                "columns": ["t_s", "amps_cmd", "amps_meas", "erpm", "pos_deg", "temp_c", "amb_c"],
                "note": "blocked-rotor step response; temp_c is the DRIVE's reported temperature, "
                        "not winding temperature -- that is the whole point of the fit"}
        np.savez(str(base) + ".npz", data=a, meta_json=np.array(json.dumps(meta)))
        with open(str(base) + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        rise = a[:, 5].max() - a[0, 5]
        print("wrote {}.npz  ({} samples, {:.0f} s, temperature rose {:.0f} C)".format(
            base, len(a), a[-1, 0], rise))
        if rise < 8:
            print("!! only {:.0f} C of rise. The fit needs a big excursion to separate the two "
                  "time constants -- raise the current or lengthen the hold and run it again."
                  .format(rise))
        return str(base) + ".npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motor", required=True, choices=paths.MOTOR_NAMES)
    ap.add_argument("--steps", default=DEFAULT_STEPS,
                    help="amps:seconds pairs, e.g. '6:180,0:900,9:150,0:1500'")
    ap.add_argument("--ambient", type=float, default=25.0, help="room air temperature, degC")
    ap.add_argument("--out", default=str(WEBUI / "data" / "thermal"))
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--mock", action="store_true", help="MockBus -- no hardware, no CAN")
    ap.add_argument("--abort-erpm", type=float, default=ABORT_ERPM)
    ap.add_argument("--abort-temp", type=float, default=ABORT_TEMP_C)
    ap.add_argument("--joint-is-blocked", action="store_true",
                    help="REQUIRED for a real run: you have confirmed this joint cannot rotate")
    args = ap.parse_args()

    args.amps_max = max(a for a, _ in parse_steps(args.steps))
    if args.mock and args.abort_erpm == ABORT_ERPM:
        # canio.MockBus models a FREE rotor accelerating under current, so the blocked-rotor speed
        # abort fires immediately and the dry run never reaches the schedule. Relax it for the
        # mock only: the point of --mock is to exercise the command path, the schedule and the
        # logging, not to simulate thermodynamics the mock does not have.
        args.abort_erpm = float("inf")
        print("(mock: the rotor-speed abort is disabled -- MockBus simulates a FREE rotor)")
    if args.amps_max > MAX_CLI_AMPS:
        raise SystemExit("{:.1f} A exceeds the {:.1f} A ceiling this tool will command. Raising it "
                         "needs a hardware argument, and a stalled rotor is the worst place to "
                         "find out you were wrong.".format(args.amps_max, MAX_CLI_AMPS))
    if not args.mock and not args.joint_is_blocked:
        raise SystemExit(
            "refusing to energise {}.\n"
            "This test stalls a motor on purpose. Mechanically block the joint (rest it against a "
            "hard stop, or clamp it), confirm nothing can move or be pinched, then pass "
            "--joint-is-blocked. Run --mock first to see exactly what it will do."
            .format(args.motor))
    if not args.mock and os.system("systemctl is-active --quiet runningrobot-webui.service") == 0:
        raise SystemExit("the webui daemon is running and streaming its own CAN commands. "
                         "sudo systemctl stop runningrobot-webui.service, then retry.")

    run = Run(args)
    print("thermal calibration: {} on {} (id {})  total {:.0f} s"
          .format(args.motor, run.channel, run.cid, run.total_s))

    def _sigterm(_s, _f):
        run._stop = True
    signal.signal(signal.SIGTERM, _sigterm)

    ok = run.execute()
    run.save(args.out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
