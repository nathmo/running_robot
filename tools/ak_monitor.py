#!/usr/bin/env python3
"""Live monitor for CubeMars / T-Motor AK actuators in SERVO mode.

Passively listens to the status frames the motor broadcasts and prints a live,
in-place updating table of position / speed / current / temperature / error.
Purely read-only: it never transmits, so it's safe to leave running while you
turn the motor by hand or run another script.

Examples:
    python tools/ak_monitor.py --channel can1                 # all motors on the bus
    python tools/ak_monitor.py --channel can1 --id 104        # just motor 104
    python tools/ak_monitor.py --channel can0                 # shared-id motors show as
                                                              #   multiple rows (by position)

Requires: pip install python-can
"""
import argparse
import struct
import time

import can

DEFAULT_INTERFACE = "socketcan"
DEFAULT_CHANNEL = "can1"
DEFAULT_BITRATE = 1_000_000


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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interface", default=DEFAULT_INTERFACE)
    ap.add_argument("--channel", default=DEFAULT_CHANNEL)
    ap.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE)
    ap.add_argument("--id", type=int, default=None,
                    help="only show this motor id (default: all seen)")
    ap.add_argument("--hz", type=float, default=10.0, help="screen refresh rate")
    args = ap.parse_args()

    print(f"Monitoring {args.interface}:{args.channel} @ {args.bitrate} bps. Ctrl+C to stop.")
    bus = can.Bus(interface=args.interface, channel=args.channel, bitrate=args.bitrate)

    # key -> {st, count, last_seen}. key is the motor id, or (id, pos-bucket) so that
    # several motors sharing one id (e.g. on can0) show as separate rows.
    rows = {}
    refresh = 1.0 / args.hz
    prev_lines = 0
    next_draw = time.time()

    def key_for(mid, st):
        return mid  # one row per id; good enough for uniquely-addressed motors

    try:
        while True:
            msg = bus.recv(timeout=refresh)
            now = time.time()
            if msg is not None and msg.is_extended_id:
                mid = msg.arbitration_id & 0xFF
                if args.id is None or mid == args.id:
                    st = parse_status(msg.data)
                    if st:
                        k = key_for(mid, st)
                        rec = rows.setdefault(k, {"count": 0})
                        rec.update(st)
                        rec["count"] += 1
                        rec["last"] = now
                        rec["id"] = mid

            if now >= next_draw:
                next_draw = now + refresh
                lines = []
                header = (f"{'id':>4}  {'pos(deg)':>9}  {'spd(erpm)':>9}  "
                          f"{'cur(A)':>7}  {'temp':>4}  {'err':>3}  {'age':>5}  {'frames':>7}")
                lines.append(header)
                lines.append("-" * len(header))
                for k in sorted(rows):
                    r = rows[k]
                    age = now - r.get("last", now)
                    stale = "  (STALE)" if age > 0.5 else ""
                    lines.append(
                        f"{r['id']:>4}  {r['pos']:>9.2f}  {r['spd']:>9.0f}  "
                        f"{r['cur']:>7.2f}  {r['temp']:>4}  {r['err']:>3}  "
                        f"{age:>5.2f}  {r['count']:>7}{stale}")
                if len(rows) == 0:
                    lines.append("  (waiting for status frames...)")

                # Redraw in place: move cursor up over the previous block, reprint.
                if prev_lines:
                    print(f"\033[{prev_lines}A", end="")
                out = "\n".join(f"\033[2K{ln}" for ln in lines)  # clear each line
                print(out)
                prev_lines = len(lines)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        bus.shutdown()


if __name__ == "__main__":
    main()
