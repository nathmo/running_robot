#!/usr/bin/env python3
"""Gentle TORQUE (current) control for a CubeMars / T-Motor AK actuator in SERVO mode.

Servo mode has no direct "Nm" command -- torque is produced by the current loop
(CAN_PACKET_SET_CURRENT). This script converts a desired torque to a current using
    current_A = torque_Nm / KT_OUTPUT      (KT_OUTPUT = output Nm per amp)
and streams it at a fixed rate.

  *** IMPORTANT: torque accuracy depends on KT_OUTPUT, which is motor-specific and
      NOT known exactly for the AK80-4830. Calibrate it (see bottom of this file).
      Until then, treat the Nm number as approximate. The safety clamps below make
      the script safe to run regardless -- they just cap current, not verify Nm. ***

Safety layers (all active at once):
  1. Torque is clamped to +/- MAX_TORQUE_NM (0.1 Nm).
  2. The commanded current is independently clamped to +/- MAX_CURRENT_A.
  3. Runaway guard: if |speed| exceeds MAX_SPEED_ERPM the current is cut to 0 and the
     script stops. A torque command on a FREE shaft accelerates it -- expected. Hold
     the output, load it against a spring, or keep the guard tight.

Examples:
    python tools/ak_servo_torque.py --id 104 --channel can1 --torque 0.05
    python tools/ak_servo_torque.py --id 104 --channel can1 --current 0.1   # bypass Kt
    python tools/ak_servo_torque.py --id 104 --channel can1 --torque 0.05 --kt 0.8

Requires: pip install python-can
"""
import argparse
import struct
import time

import can

# ---- Servo-mode command indices ----
CAN_PACKET_SET_CURRENT = 1

# ---- Defaults (override on the CLI) ----
DEFAULT_INTERFACE = "socketcan"
DEFAULT_CHANNEL = "can1"
DEFAULT_BITRATE = 1_000_000
DEFAULT_ID = 104

# Output torque constant [Nm per amp of commanded current]. PLACEHOLDER -- calibrate!
# Setting this HIGHER makes the script command LESS current for a given torque (safer,
# under-torques); LOWER makes it command MORE current (be careful). See calibration note.
KT_OUTPUT = 1.0

# Safety clamps
MAX_TORQUE_NM = 0.1      # hard ceiling on the torque you asked for
MAX_CURRENT_A = 0.5      # independent absolute ceiling on commanded current
MAX_SPEED_ERPM = 1000.0  # runaway guard: cut current if the shaft spins faster
MAX_TEMP_C = 80          # cut current if the motor reports this temperature

CMD_RATE_HZ = 100.0      # current-command streaming rate
RAMP_S = 1.0             # ramp the torque 0 -> target over this many seconds


def send_servo(bus, controller_id, command, payload):
    arb = controller_id | (command << 8)
    bus.send(can.Message(arbitration_id=arb, data=payload, is_extended_id=True))


def set_current(bus, controller_id, amps):
    val = int(round(amps * 1000.0))
    val = max(-2_147_483_648, min(2_147_483_647, val))
    send_servo(bus, controller_id, CAN_PACKET_SET_CURRENT, struct.pack(">i", val))


def parse_status(data):
    if len(data) < 8:
        return None
    pos = struct.unpack(">h", bytes(data[0:2]))[0] * 0.1
    spd = struct.unpack(">h", bytes(data[2:4]))[0] * 10.0
    cur = struct.unpack(">h", bytes(data[4:6]))[0] * 0.01
    temp = struct.unpack(">b", bytes(data[6:7]))[0]
    err = data[7]
    return {"pos": pos, "spd": spd, "cur": cur, "temp": temp, "err": err}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", type=int, default=DEFAULT_ID)
    ap.add_argument("--interface", default=DEFAULT_INTERFACE)
    ap.add_argument("--channel", default=DEFAULT_CHANNEL)
    ap.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE)
    ap.add_argument("--torque", type=float, default=0.05, help="target torque (Nm)")
    ap.add_argument("--current", type=float, default=None,
                    help="command current (A) directly, bypassing Kt/torque")
    ap.add_argument("--kt", type=float, default=KT_OUTPUT,
                    help="output torque constant, Nm per amp")
    ap.add_argument("--max-current", type=float, default=MAX_CURRENT_A,
                    help="absolute current ceiling (A)")
    ap.add_argument("--max-speed", type=float, default=MAX_SPEED_ERPM,
                    help="runaway guard speed (ERPM)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="run time in s (0 = until Ctrl+C)")
    args = ap.parse_args()

    # Resolve the target current, with both torque and current clamps applied.
    if args.current is not None:
        target_current = args.current
        req_desc = f"current {args.current:+.3f} A (direct)"
    else:
        torque = max(-MAX_TORQUE_NM, min(MAX_TORQUE_NM, args.torque))
        if torque != args.torque:
            print(f"Torque {args.torque} Nm clamped to {torque} Nm (|max| {MAX_TORQUE_NM}).")
        target_current = torque / args.kt if args.kt else 0.0
        req_desc = f"torque {torque:+.3f} Nm -> {target_current:+.3f} A (Kt={args.kt} Nm/A)"

    clamped = max(-args.max_current, min(args.max_current, target_current))
    if clamped != target_current:
        print(f"Current {target_current:+.3f} A clamped to {clamped:+.3f} A "
              f"(|max| {args.max_current} A).")
    target_current = clamped

    print(f"Opening {args.interface}:{args.channel} @ {args.bitrate} bps, motor id={args.id}")
    print(f"Request: {req_desc}")
    print(f"Commanding {target_current:+.3f} A (~{target_current * args.kt:+.3f} Nm at "
          f"Kt={args.kt}). Runaway guard at {args.max_speed:.0f} ERPM. Ctrl+C to stop.")
    print("NOTE: on a FREE shaft this torque will make the motor spin up.\n")

    bus = can.Bus(interface=args.interface, channel=args.channel, bitrate=args.bitrate)
    dt = 1.0 / CMD_RATE_HZ
    try:
        t0 = time.time()
        next_t = t0
        last_print = 0.0
        while True:
            now = time.time()
            elapsed = now - t0
            if args.duration and elapsed >= args.duration:
                break

            ramp = min(1.0, elapsed / RAMP_S) if RAMP_S > 0 else 1.0
            set_current(bus, args.id, target_current * ramp)

            msg = bus.recv(timeout=0.0)
            if msg is not None and msg.is_extended_id and (msg.arbitration_id & 0xFF) == args.id:
                fb = parse_status(msg.data)
                if fb:
                    if abs(fb["spd"]) > args.max_speed:
                        print(f"\n!! Runaway: |speed| {fb['spd']:.0f} > {args.max_speed:.0f} "
                              f"ERPM. Cutting current.")
                        break
                    if fb["temp"] >= MAX_TEMP_C:
                        print(f"\n!! Overtemp: {fb['temp']}C. Cutting current.")
                        break
                    if fb["err"]:
                        print(f"\n!! Motor error code {fb['err']}. Cutting current.")
                        break
                    if (now - last_print) > 0.1:
                        last_print = now
                        print(f"  t={elapsed:5.1f}s  cmd={target_current * ramp:+.3f}A  "
                              f"actualI={fb['cur']:+.3f}A  pos={fb['pos']:7.2f}deg  "
                              f"spd={fb['spd']:6.0f}erpm  temp={fb['temp']}C   ", end="\r")

            next_t += dt
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            set_current(bus, args.id, 0.0)   # release
            time.sleep(0.02)
        finally:
            bus.shutdown()
        print("Motor released (0 A).")


# ---------------------------------------------------------------------------
# Calibrating KT_OUTPUT (Nm per amp) so the torque number is real:
#   1. Clamp/hold the output arm horizontally at a known radius r (m) resting on a
#      kitchen/luggage scale, so the scale reads the force the arm pushes down.
#   2. Run:  python tools/ak_servo_torque.py --current 0.20 --channel can1 --id 104
#      (start small; increase current in steps, keeping the arm from moving).
#   3. Read the scale mass m (kg). Torque = m*9.81*r  (Nm). Then KT_OUTPUT = torque/current.
#   4. Put that number in --kt (or edit KT_OUTPUT above).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
