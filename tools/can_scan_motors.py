#!/usr/bin/env python3
"""Scan a CAN bus for CubeMars / T-Motor AK-series actuators and print what's found.

Detects motors in BOTH firmware modes:
  * Servo mode  -> motors broadcast extended-frame status automatically (passive listen).
  * MIT mode    -> motors are silent until commanded, so we actively probe each CAN ID
                   with "enter motor mode" and read the reply (active probe).

The bus is only touched read-mostly: the MIT probe enables a motor, reads its reply,
then immediately sends "exit motor mode" so nothing is left energized.

Usage (edit the defaults below or override on the CLI):
    python tools/can_scan_motors.py --interface pcan    --channel PCAN_USBBUS1
    python tools/can_scan_motors.py --interface slcan   --channel COM5      # CANable
    python tools/can_scan_motors.py --interface gs_usb  --channel 0         # candleLight
    python tools/can_scan_motors.py --interface kvaser  --channel 0
    python tools/can_scan_motors.py --interface socketcan --channel can0    # Linux

Requires: pip install python-can
"""
import argparse
import struct
import time

import can

# ---------------------------------------------------------------------------
# Defaults -- override on the command line.
# ---------------------------------------------------------------------------
DEFAULT_INTERFACE = "slcan"   # one of: pcan, slcan, gs_usb, kvaser, vector, socketcan, ...
DEFAULT_CHANNEL = "COM5"      # PCAN_USBBUS1 / COMx / 0 / can0  depending on adapter
DEFAULT_BITRATE = 1_000_000   # AK motors run at 1 Mbps

# ID range to probe in MIT mode (motors usually ship as ID 1..several).
MIT_PROBE_IDS = range(1, 17)

# Servo-mode "status" frame is command index 9 on the extended ID.
CAN_PACKET_STATUS = 9

# Approximate MIT feedback ranges -- ONLY used to pretty-print the probe reply.
# These are AK80-9 defaults; the *identity* of the motor does not depend on them,
# so scanning works regardless, but the decoded pos/vel/current is only right if
# your variant matches. Verify ranges in the CubeMars GUI for your motor.
P_MIN, P_MAX = -12.5, 12.5     # rad
V_MIN, V_MAX = -50.0, 50.0     # rad/s
T_MIN, T_MAX = -18.0, 18.0     # "torque current" units (A-equivalent)


def uint_to_float(x, lo, hi, bits):
    return x * (hi - lo) / ((1 << bits) - 1) + lo


def parse_mit_reply(data):
    """Decode an AK MIT-mode feedback frame (>=6 bytes). Returns a dict."""
    d = data
    motor_id = d[0]
    p_int = (d[1] << 8) | d[2]
    v_int = (d[3] << 4) | (d[4] >> 4)
    i_int = ((d[4] & 0xF) << 8) | d[5]
    out = {
        "motor_id": motor_id,
        "pos_rad": uint_to_float(p_int, P_MIN, P_MAX, 16),
        "vel_rad_s": uint_to_float(v_int, V_MIN, V_MAX, 12),
        "current": uint_to_float(i_int, T_MIN, T_MAX, 12),
    }
    if len(d) >= 7:
        out["temp_C"] = d[6]            # newer firmware only
    if len(d) >= 8:
        out["error"] = d[7]
    return out


def parse_servo_status(data):
    """Decode an AK servo-mode periodic status frame (8 bytes)."""
    if len(data) < 8:
        return None
    pos = struct.unpack(">h", bytes(data[0:2]))[0] * 0.1     # deg
    spd = struct.unpack(">h", bytes(data[2:4]))[0] * 10.0    # ERPM
    cur = struct.unpack(">h", bytes(data[4:6]))[0] * 0.01    # A
    temp = struct.unpack(">b", bytes(data[6:7]))[0]          # deg C
    err = data[7]
    return {"pos_deg": pos, "speed_erpm": spd, "current_A": cur,
            "temp_C": temp, "error": err}


def passive_listen(bus, seconds):
    """Listen quietly and report traffic.

    Motors that share a CAN ID appear as ONE arbitration id carrying SEVERAL
    distinct payload streams. We group frames per id, then per id split them into
    streams by (rounded) reported position -- so three idle motors on one shared id
    show up as three streams. Keep motors roughly still while scanning for this to
    separate cleanly (a moving motor sweeps through many positions).
    """
    print(f"[1/2] Passive listen for {seconds:.1f}s (catches servo-mode motors)...")
    aids = {}   # aid -> {"count", "ext", "streams": {poskey: [st, data, count]}}
    errors = 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        msg = bus.recv(timeout=0.2)
        if msg is None:
            continue
        # Error frames on SocketCAN surface here -- a wiring/bitrate/collision signal.
        if getattr(msg, "is_error_frame", False):
            errors += 1
            continue
        aid = msg.arbitration_id
        data = bytes(msg.data)
        rec = aids.setdefault(aid, {"count": 0, "ext": msg.is_extended_id, "streams": {}})
        rec["count"] += 1
        st = parse_servo_status(data) if (msg.is_extended_id and len(data) >= 8) else None
        # Group by ~5deg position bucket (stable for an idle motor); else by raw bytes.
        poskey = round(st["pos_deg"] / 5.0) if st else data.hex()
        stream = rec["streams"].setdefault(poskey, [st, data, 0])
        stream[2] += 1

    motors, apparent_devices = set(), 0
    for aid in sorted(aids):
        rec = aids[aid]
        nstreams = len(rec["streams"])
        apparent_devices += nstreams if rec["ext"] else 1
        if rec["ext"]:
            mid, cmd = aid & 0xFF, (aid >> 8) & 0xFF
            motors.add(mid)
            shared = ("   <-- SHARED ID: multiple motors on the same id!"
                      if nstreams > 1 else "")
            print(f"      ext id=0x{aid:08X}  motor_id={mid}  cmd={cmd}  "
                  f"frames={rec['count']}  streams={nstreams}{shared}")
            for st, data, c in rec["streams"].values():
                if st:
                    print(f"          - pos={st['pos_deg']:8.1f}deg  I={st['current_A']:5.2f}A  "
                          f"temp={st['temp_C']}C  err={st['error']}  ({c} frames)")
                else:
                    print(f"          - {data.hex(' ')}  ({c} frames)")
        else:
            print(f"      std id=0x{aid:03X}  frames={rec['count']}")

    if errors:
        print(f"      !! {errors} CAN ERROR frames during listen -> bitrate/wiring, "
              f"or shared-id frames starting to overlap.")
    if motors:
        print(f"      -> distinct CAN IDs seen: {sorted(motors)}  |  "
              f"apparent motor count (by payload stream): {apparent_devices}")
        if apparent_devices > len(motors):
            print("      -> MORE motors than CAN IDs: some motors share an ID and CANNOT "
                  "be addressed individually. Assign each a unique ID in the CubeMars GUI "
                  "(one motor connected at a time).")
    elif not aids:
        print("      -> no spontaneous traffic (normal if all motors are in MIT mode).")
    return sorted(motors)


def mit_probe(bus, ids):
    """Ping each ID with 'enter motor mode' and capture the reply -> MIT-mode motors."""
    ENTER = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC])
    EXIT = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD])
    print(f"[2/2] Active MIT probe of IDs {ids.start}..{ids.stop - 1} ...")
    found = []
    for mid in ids:
        bus.send(can.Message(arbitration_id=mid, data=ENTER, is_extended_id=False))
        reply = None
        deadline = time.time() + 0.05
        while time.time() < deadline:
            msg = bus.recv(timeout=0.05)
            if msg is None:
                continue
            if not msg.is_extended_id and len(msg.data) >= 6 and msg.data[0] == mid:
                reply = msg
                break
        # Leave the motor disabled regardless of whether it answered.
        bus.send(can.Message(arbitration_id=mid, data=EXIT, is_extended_id=False))
        if reply is not None:
            info = parse_mit_reply(reply.data)
            found.append(mid)
            extra = ""
            if "temp_C" in info:
                extra = f" temp={info['temp_C']}C"
            if "error" in info:
                extra += f" err={info['error']}"
            print(f"      MIT motor FOUND  id={mid}  (reply on 0x{reply.arbitration_id:03X})  "
                  f"pos={info['pos_rad']:.3f}rad vel={info['vel_rad_s']:.2f}rad/s "
                  f"I={info['current']:.2f}{extra}")
        time.sleep(0.005)
    if not found:
        print("      -> no MIT-mode motors answered.")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interface", default=DEFAULT_INTERFACE)
    ap.add_argument("--channel", default=DEFAULT_CHANNEL)
    ap.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE)
    ap.add_argument("--listen", type=float, default=2.0,
                    help="seconds to passively listen for servo-mode motors")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the active MIT-mode probe")
    args = ap.parse_args()

    print(f"Opening {args.interface}:{args.channel} @ {args.bitrate} bps")
    bus = can.Bus(interface=args.interface, channel=args.channel, bitrate=args.bitrate)
    try:
        servo = passive_listen(bus, args.listen)
        mit = [] if args.no_probe else mit_probe(bus, MIT_PROBE_IDS)
        print("\n=== Summary ===")
        print(f"  Servo-mode motors (broadcasting): {servo or 'none'}")
        print(f"  MIT-mode motors   (probed reply): {mit or 'none'}")
        if not servo and not mit:
            print("  Nothing found. Check bitrate (must be 1 Mbps), wiring, and that the\n"
                  "  motor is powered. If you see raw frames above but no decode, the motor\n"
                  "  may use a non-default master/reply ID.")
    finally:
        bus.shutdown()


if __name__ == "__main__":
    main()
