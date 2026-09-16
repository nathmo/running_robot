#!/usr/bin/env python3
"""Ask an AK drive whether it answers MIT-mode frames, WITHOUT reflashing anything.

The AK3.0 upper computer documents "seamless switching between servo and force control modes
without manual physical switching", i.e. the firmware is supposed to accept MIT frames while it is
still speaking servo mode. This checks that claim on the real drives, and recovers the MIT id.

HOW IT TELLS THEM APART, and why no drain loop is needed:
    servo-mode status  -> EXTENDED (29-bit) ids, free-running at 200 Hz
    MIT-mode reply     -> STANDARD (11-bit) id
so every standard-id frame arriving after we send ENTER is an MIT reply, and the 200 Hz servo
broadcast can simply be filtered out. (An earlier version tried to drain the bus first and hung
forever, because the motors never stop talking.)

SAFETY. This sends ENTER MOTOR MODE and then immediately EXIT. It never sends a setpoint, never a
gain, never a torque -- entering the mode alone commands no motion, and we leave the drive disabled
again within milliseconds. A `finally` sends EXIT to every id touched even if something throws.
"""
import argparse
import sys
import time

try:
    import can
except ImportError:
    can = None

ENTER = bytes([0xFF] * 7 + [0xFC])
EXIT = bytes([0xFF] * 7 + [0xFD])


def u2f(x, lo, hi, bits):
    return x * (hi - lo) / ((1 << bits) - 1) + lo


def parse(d):
    """MIT feedback: id, 16-bit pos, 12-bit vel, 12-bit current. Decoded with the AK80-9 ranges we
    are trying to validate, so treat the numbers as provisional -- what matters here is that a
    reply exists AT ALL and which id it carries."""
    if len(d) < 6:
        return None
    return {"id": d[0],
            "pos_rad": round(u2f((d[1] << 8) | d[2], -12.5, 12.5, 16), 4),
            "vel_rad_s": round(u2f((d[3] << 4) | (d[4] >> 4), -50.0, 50.0, 12), 3),
            "cur": round(u2f(((d[4] & 0xF) << 8) | d[5], -18.0, 18.0, 12), 3)}


def probe(channel, ids, listen_s=0.35, bitrate=1_000_000):
    bus = can.interface.Bus(channel=channel, interface="socketcan", bitrate=bitrate)
    touched = []
    found = {}
    try:
        for mid in ids:
            bus.send(can.Message(arbitration_id=mid, data=ENTER, is_extended_id=False))
            touched.append(mid)
            t0 = time.time()
            rep = None
            while time.time() - t0 < listen_s:
                m = bus.recv(timeout=0.02)
                if m is None:
                    continue
                if not m.is_extended_id:            # servo status is extended; this is MIT
                    rep = m
                    break
            bus.send(can.Message(arbitration_id=mid, data=EXIT, is_extended_id=False))
            if rep is not None:
                found[mid] = (rep.arbitration_id, rep.data.hex(), parse(rep.data))
                print(f"  {channel} id={mid:3d}: MIT REPLY on 0x{rep.arbitration_id:03X} "
                      f"data={rep.data.hex()}")
                print(f"            {found[mid][2]}")
            else:
                print(f"  {channel} id={mid:3d}: no MIT reply")
    finally:
        for mid in touched:
            try:
                bus.send(can.Message(arbitration_id=mid, data=EXIT, is_extended_id=False))
            except Exception:                        # noqa: BLE001 - never mask the exit path
                pass
        bus.shutdown()
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", default="can0,can1")
    ap.add_argument("--ids", default="105,106",
                    help="ids to probe (default: the two AKE90-8s, cam and thigh)")
    ap.add_argument("--listen", type=float, default=0.35)
    args = ap.parse_args()
    if can is None:
        print("python-can not installed", file=sys.stderr)
        return 2
    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    print(f"probing MIT on {args.channels} ids {ids} (ENTER then immediate EXIT, no setpoint)")
    all_found = {}
    for ch in args.channels.split(","):
        all_found[ch] = probe(ch.strip(), ids, args.listen)
    print()
    hit = [(c, i) for c, f in all_found.items() for i in f]
    print("MIT reachable on:", hit or "NONE — the drives are servo-only until reflashed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
