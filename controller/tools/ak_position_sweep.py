#!/usr/bin/env python3
"""Drive a CubeMars / T-Motor AK-series actuator through a position sweep (MIT mode).

The motor is commanded with an impedance (Kp/Kd) position target that follows a
sine wave. A soft-start ramps the amplitude up over the first couple of seconds so
the motor eases into motion instead of snapping. Ctrl+C stops cleanly: it commands
zero, then sends "exit motor mode" so the motor is left disabled.

  *** SAFETY ***
  - This assumes the motor is in MIT firmware (run can_scan_motors.py to confirm).
  - The sweep is RELATIVE to the pose at startup: the script zeroes the encoder, so
    the motor sweeps +/- AMPLITUDE_RAD around wherever it sits when you launch it.
  - Start with a SMALL amplitude and LOW Kp. Make sure the output can move freely
    through the whole range. Keep a hand near the power switch the first time.

Usage:
    python tools/ak_position_sweep.py --id 1 --interface slcan --channel COM5
    python tools/ak_position_sweep.py --id 1 --amp 0.5 --period 3 --kp 20 --kd 1

Requires: pip install python-can
"""
import argparse
import math
import struct
import time

import can

# ---------------------------------------------------------------------------
# Defaults -- override on the command line.
# ---------------------------------------------------------------------------
DEFAULT_INTERFACE = "slcan"
DEFAULT_CHANNEL = "COM5"
DEFAULT_BITRATE = 1_000_000
DEFAULT_ID = 1

# Sweep shape
AMPLITUDE_RAD = 1.0     # peak deflection from center (rad). 1.0 rad ~= 57 deg.
PERIOD_S = 3.0          # seconds per full back-and-forth cycle
SOFT_START_S = 2.0      # ramp amplitude 0 -> full over this many seconds
CMD_RATE_HZ = 200.0     # command/update rate

# Impedance gains -- keep these modest to start.
KP = 20.0               # Nm/rad  (stiffness toward the target)
KD = 1.0                # Nm*s/rad (damping)

# ---- Motor parameter ranges (MIT mode). AK80-9 defaults shown. ----
# VERIFY these against your exact variant in the CubeMars GUI; wrong ranges mean
# wrong commanded values. Pos/Vel/Torque ranges differ across AK models.
P_MIN, P_MAX = -12.5, 12.5     # rad
V_MIN, V_MAX = -50.0, 50.0     # rad/s
KP_MIN, KP_MAX = 0.0, 500.0    # Nm/rad
KD_MIN, KD_MAX = 0.0, 5.0      # Nm*s/rad
T_MIN, T_MAX = -18.0, 18.0     # torque-current units

ENTER = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC])
EXIT = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD])
ZERO = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE])


def float_to_uint(x, lo, hi, bits):
    x = max(lo, min(hi, x))
    return int((x - lo) * ((1 << bits) - 1) / (hi - lo))


def uint_to_float(x, lo, hi, bits):
    return x * (hi - lo) / ((1 << bits) - 1) + lo


def pack_cmd(pos, vel, kp, kd, tff):
    p = float_to_uint(pos, P_MIN, P_MAX, 16)
    v = float_to_uint(vel, V_MIN, V_MAX, 12)
    kpi = float_to_uint(kp, KP_MIN, KP_MAX, 12)
    kdi = float_to_uint(kd, KD_MIN, KD_MAX, 12)
    ti = float_to_uint(tff, T_MIN, T_MAX, 12)
    return bytes([
        (p >> 8) & 0xFF, p & 0xFF,
        (v >> 4) & 0xFF, ((v & 0xF) << 4) | (kpi >> 8), kpi & 0xFF,
        (kdi >> 4) & 0xFF, ((kdi & 0xF) << 4) | (ti >> 8), ti & 0xFF,
    ])


def parse_reply(data):
    motor_id = data[0]
    p_int = (data[1] << 8) | data[2]
    v_int = (data[3] << 4) | (data[4] >> 4)
    i_int = ((data[4] & 0xF) << 8) | data[5]
    res = {
        "id": motor_id,
        "pos": uint_to_float(p_int, P_MIN, P_MAX, 16),
        "vel": uint_to_float(v_int, V_MIN, V_MAX, 12),
        "cur": uint_to_float(i_int, T_MIN, T_MAX, 12),
        "temp": data[6] if len(data) >= 7 else None,
        "err": data[7] if len(data) >= 8 else None,
    }
    return res


def send_special(bus, motor_id, payload):
    bus.send(can.Message(arbitration_id=motor_id, data=payload, is_extended_id=False))


def send_cmd(bus, motor_id, pos, vel, kp, kd, tff):
    bus.send(can.Message(arbitration_id=motor_id, data=pack_cmd(pos, vel, kp, kd, tff),
                         is_extended_id=False))
    # The motor answers each command with a feedback frame.
    msg = bus.recv(timeout=0.005)
    if msg is not None and not msg.is_extended_id and len(msg.data) >= 6:
        return parse_reply(msg.data)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", type=int, default=DEFAULT_ID)
    ap.add_argument("--interface", default=DEFAULT_INTERFACE)
    ap.add_argument("--channel", default=DEFAULT_CHANNEL)
    ap.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE)
    ap.add_argument("--amp", type=float, default=AMPLITUDE_RAD, help="amplitude (rad)")
    ap.add_argument("--period", type=float, default=PERIOD_S, help="sweep period (s)")
    ap.add_argument("--kp", type=float, default=KP)
    ap.add_argument("--kd", type=float, default=KD)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="run time in s (0 = until Ctrl+C)")
    args = ap.parse_args()

    print(f"Opening {args.interface}:{args.channel} @ {args.bitrate} bps, motor id={args.id}")
    print(f"Sweep: +/-{args.amp:.2f} rad, period {args.period:.1f}s, Kp={args.kp} Kd={args.kd}")
    bus = can.Bus(interface=args.interface, channel=args.channel, bitrate=args.bitrate)

    dt = 1.0 / CMD_RATE_HZ
    try:
        # Enable and zero so the sweep is centered on the startup pose.
        send_special(bus, args.id, ENTER)
        time.sleep(0.05)
        send_special(bus, args.id, ZERO)
        time.sleep(0.05)
        # Drain the zero-reply.
        bus.recv(timeout=0.05)

        print("Running. Press Ctrl+C to stop.\n")
        t0 = time.time()
        next_t = t0
        last_print = 0.0
        while True:
            now = time.time()
            elapsed = now - t0
            if args.duration and elapsed >= args.duration:
                break

            ramp = min(1.0, elapsed / SOFT_START_S) if SOFT_START_S > 0 else 1.0
            amp = args.amp * ramp
            phase = 2.0 * math.pi * elapsed / args.period
            target = amp * math.sin(phase)
            # Feed-forward velocity = derivative of the target (helps tracking).
            vel_ff = amp * (2.0 * math.pi / args.period) * math.cos(phase)

            fb = send_cmd(bus, args.id, target, vel_ff, args.kp, args.kd, 0.0)
            if fb and (now - last_print) > 0.1:
                last_print = now
                temp = f" temp={fb['temp']}C" if fb["temp"] is not None else ""
                err = f" err={fb['err']}" if fb["err"] else ""
                print(f"  t={elapsed:5.1f}s  tgt={target:+.3f}  "
                      f"pos={fb['pos']:+.3f}rad  vel={fb['vel']:+.2f}  "
                      f"I={fb['cur']:+.2f}{temp}{err}", end="\r")

            next_t += dt
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()  # we fell behind; resync
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Command zero torque/stiffness, then disable.
        try:
            send_special(bus, args.id, bytes(pack_cmd(0, 0, 0, 0, 0)))
            time.sleep(0.02)
            send_special(bus, args.id, EXIT)
        finally:
            bus.shutdown()
        print("Motor disabled.")


if __name__ == "__main__":
    main()
