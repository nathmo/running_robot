#!/usr/bin/env python3
"""Slow position sweep for a CubeMars / T-Motor AK actuator in SERVO mode.

This is the servo-mode counterpart to ak_position_sweep.py (which is MIT-mode only).
Use it when can_scan_motors.py reports the motor broadcasting servo status frames.

How it works:
  * Reads the motor's CURRENT position from its broadcast status frame, so the sweep
    is centered where the motor already is (no jump when it starts).
  * Streams a sine position setpoint via CAN_PACKET_SET_POS at 100 Hz. The speed of
    the motion is set by the (slow) trajectory, NOT the firmware's configured max
    speed, so a small slow sweep stays small and slow regardless of motor settings.
  * Soft-starts: amplitude ramps 0 -> full over a couple seconds.
  * Ctrl+C sends SET_CURRENT 0 (motor goes limp), then exits.

Servo-mode protocol (confirmed against CubeMars firmware / TMotorCANControl):
  extended id = controller_id | (command << 8)
  CAN_PACKET_SET_POS     = 4 : data = int32(pos_deg   * 1_000_000), big-endian
  CAN_PACKET_SET_CURRENT = 1 : data = int32(amps      * 1_000),     big-endian
  status frame           : pos = int16BE * 0.1 deg, spd = int16BE * 10 ERPM,
                           cur = int16BE * 0.01 A, temp = int8 degC, err = uint8

Example (the motor you found on can1):
    python tools/ak_servo_sweep.py --id 104 --channel can1 --amp 2.5 --period 4
    # --amp 2.5 => 5 deg peak-to-peak (2.5 deg either side of center)

Requires: pip install python-can
"""
import argparse
import math
import struct
import time

import can

# ---- Servo-mode command indices ----
CAN_PACKET_SET_CURRENT = 1
CAN_PACKET_SET_POS = 4

# ---- Defaults (override on the CLI) ----
DEFAULT_INTERFACE = "socketcan"
DEFAULT_CHANNEL = "can1"
DEFAULT_BITRATE = 1_000_000
DEFAULT_ID = 104

AMPLITUDE_DEG = 2.5     # deflection either side of center; 2.5 => 5 deg peak-to-peak
PERIOD_S = 4.0          # seconds per full back-and-forth cycle (bigger = slower)
SOFT_START_S = 2.0      # ramp amplitude 0 -> full over this many seconds
CMD_RATE_HZ = 100.0     # setpoint streaming rate


def send_servo(bus, controller_id, command, payload):
    arb = controller_id | (command << 8)
    bus.send(can.Message(arbitration_id=arb, data=payload, is_extended_id=True))


def set_pos(bus, controller_id, pos_deg):
    # int32, big-endian, degrees * 1e6. Clamp to int32 range for safety.
    val = int(round(pos_deg * 1_000_000.0))
    val = max(-2_147_483_648, min(2_147_483_647, val))
    send_servo(bus, controller_id, CAN_PACKET_SET_POS, struct.pack(">i", val))


def set_current(bus, controller_id, amps):
    val = int(round(amps * 1000.0))
    send_servo(bus, controller_id, CAN_PACKET_SET_CURRENT, struct.pack(">i", val))


def parse_status(data):
    """Decode an AK servo status frame (8 bytes, big-endian)."""
    if len(data) < 8:
        return None
    pos = struct.unpack(">h", bytes(data[0:2]))[0] * 0.1
    spd = struct.unpack(">h", bytes(data[2:4]))[0] * 10.0
    cur = struct.unpack(">h", bytes(data[4:6]))[0] * 0.01
    temp = struct.unpack(">b", bytes(data[6:7]))[0]
    err = data[7]
    return {"pos": pos, "spd": spd, "cur": cur, "temp": temp, "err": err}


def read_status(bus, controller_id, timeout=1.0):
    """Wait for a status frame from `controller_id` and return the decoded dict."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = bus.recv(timeout=deadline - time.time())
        if msg is None:
            break
        if msg.is_extended_id and (msg.arbitration_id & 0xFF) == controller_id:
            st = parse_status(msg.data)
            if st:
                return st
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", type=int, default=DEFAULT_ID)
    ap.add_argument("--interface", default=DEFAULT_INTERFACE)
    ap.add_argument("--channel", default=DEFAULT_CHANNEL)
    ap.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE)
    ap.add_argument("--amp", type=float, default=AMPLITUDE_DEG,
                    help="deg either side of center (2.5 => 5 deg peak-to-peak)")
    ap.add_argument("--period", type=float, default=PERIOD_S,
                    help="seconds per full cycle (bigger = slower)")
    ap.add_argument("--max-deviation", type=float, default=None,
                    help="deg the actual position may stray from start before the motor "
                         "is cut (default: amp + 5 deg margin). Guards a bugged start pos.")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="run time in s (0 = until Ctrl+C)")
    args = ap.parse_args()

    if args.amp > 30:
        print(f"Refusing amp={args.amp} deg (>30) as a safety guard; edit the script "
              f"if you really mean it.")
        return

    print(f"Opening {args.interface}:{args.channel} @ {args.bitrate} bps, motor id={args.id}")
    bus = can.Bus(interface=args.interface, channel=args.channel, bitrate=args.bitrate)
    try:
        st = read_status(bus, args.id, timeout=1.5)
        if st is None:
            print(f"No status frame from motor {args.id}. Is it powered / in servo mode / "
                  f"on this channel? Aborting so we don't command a blind position.")
            return
        # RELATIVE sweep: the center is wherever the motor sits at launch. The sine
        # starts at 0, so the first commanded target equals start_pos (no jump).
        start_pos = st["pos"]
        print(f"Motor {args.id} start position: {start_pos:.2f} deg "
              f"(temp {st['temp']}C, err {st['err']}).")
        print(f"Sweep is RELATIVE to that start: {start_pos - args.amp:.2f}..{start_pos + args.amp:.2f} deg "
              f"({2 * args.amp:.1f} deg peak-to-peak), period {args.period:.1f}s. Ctrl+C to stop.")

        # Safety window: if the ACTUAL position ever strays this far from the start
        # position, the motion is out of control (most likely a bugged/mis-scaled
        # start position or an origin mismatch) -> cut the motor and abort. The
        # window is the sweep amplitude plus a margin for normal tracking overshoot.
        max_dev = args.max_deviation if args.max_deviation is not None else args.amp + 5.0
        print(f"Safety: motor is cut if actual position leaves "
              f"{start_pos - max_dev:.2f}..{start_pos + max_dev:.2f} deg "
              f"(|dev| > {max_dev:.2f} deg).\n")

        dt = 1.0 / CMD_RATE_HZ
        t0 = time.time()
        next_t = t0
        last_print = 0.0
        aborted = False
        while not aborted:
            now = time.time()
            elapsed = now - t0
            if args.duration and elapsed >= args.duration:
                break

            ramp = min(1.0, elapsed / SOFT_START_S) if SOFT_START_S > 0 else 1.0
            offset = args.amp * ramp * math.sin(2.0 * math.pi * elapsed / args.period)
            target = start_pos + offset   # relative to launch position
            set_pos(bus, args.id, target)

            # Drain ALL queued status frames so the safety check sees the latest
            # position promptly (the motor broadcasts faster than this loop runs).
            fb = None
            while True:
                msg = bus.recv(timeout=0.0)
                if msg is None:
                    break
                if msg.is_extended_id and (msg.arbitration_id & 0xFF) == args.id:
                    parsed = parse_status(msg.data)
                    if parsed:
                        fb = parsed
            if fb is not None:
                deviation = abs(fb["pos"] - start_pos)
                if deviation > max_dev:
                    print(f"\n!! SAFETY STOP: position {fb['pos']:.2f} deg is "
                          f"{deviation:.2f} deg from start ({start_pos:.2f}) -- exceeds "
                          f"{max_dev:.2f} deg. Cutting motor (start position likely bad).")
                    aborted = True
                    continue
                if fb["err"]:
                    print(f"\n!! Motor error code {fb['err']}. Cutting motor.")
                    aborted = True
                    continue
                if (now - last_print) > 0.1:
                    last_print = now
                    print(f"  t={elapsed:5.1f}s  target={target:7.2f}  "
                          f"actual={fb['pos']:7.2f}deg  dev={fb['pos'] - start_pos:+6.2f}  "
                          f"I={fb['cur']:5.2f}A  temp={fb['temp']}C  err={fb['err']}   ", end="\r")

            next_t += dt
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()  # fell behind; resync
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            set_current(bus, args.id, 0.0)   # release: motor goes limp
            time.sleep(0.02)
        finally:
            bus.shutdown()
        print("Motor released (0 A).")


if __name__ == "__main__":
    main()
