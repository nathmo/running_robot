#!/usr/bin/env python3
"""Play back a recorded trajectory on both legs, dephased 180 deg, with a hard TORQUE (current)
limit and adjustable speed. Start slow + low current, then raise both.

Control: a compliant SOFTWARE PD loop per motor drives current (servo-mode SET_CURRENT, the same
command tools/ak_servo_torque.py uses). The commanded current is hard-clamped to --current-limit,
so torque can never exceed that — if the leg hits something it yields instead of forcing through.
Amps (not Nm) because the motor's Nm-per-amp (Kt) is not known exactly (see ak_servo_torque.py).

    can0 = RIGHT leg, can1 = LEFT leg. Motors 104=abduction (held fixed), 105=cam, 106=hip.

Examples:
    python fixed_gait/play_trajectory.py --period 8 --current-limit 1.0     # very slow, gentle
    python fixed_gait/play_trajectory.py --period 4 --current-limit 2.0
    python fixed_gait/play_trajectory.py --period 2 --current-limit 3.0     # faster, stronger
    python fixed_gait/play_trajectory.py --dry-run                         # no torque, just prints
    python fixed_gait/play_trajectory.py --abduction-right 5 --abduction-left -3   # set abduction
"""
import argparse
import sys
import time

import numpy as np

try:
    import can
except ImportError:
    can = None

import trajectory as traj

BITRATE = 1_000_000
SIDE_CHANNEL = {"right": "can0", "left": "can1"}
MOTOR_IDS = [104, 105, 106]                 # abduction, cam, hip  == cols 0,1,2
CAN_PACKET_SET_CURRENT = 1
CMD_HZ = 200.0

# safety
MAX_SPEED_ERPM = 5000.0     # runaway guard (a torque command on a free shaft spins up)
MAX_TEMP_C = 80


def set_current(bus, cid, amps):
    val = int(round(amps * 1000.0))
    val = max(-2_147_483_648, min(2_147_483_647, val))
    bus.send(can.Message(arbitration_id=cid | (CAN_PACKET_SET_CURRENT << 8),
                         data=val.to_bytes(4, "big", signed=True), is_extended_id=True))


def parse_status(data):
    if len(data) < 8:
        return None
    return dict(pos=int.from_bytes(bytes(data[0:2]), "big", signed=True) * 0.1,
                spd=int.from_bytes(bytes(data[2:4]), "big", signed=True) * 10.0,
                cur=int.from_bytes(bytes(data[4:6]), "big", signed=True) * 0.01,
                temp=int.from_bytes(bytes(data[6:7]), "big", signed=True),
                err=data[7])


class Motor:
    def __init__(self, bus, cid, side, col):
        self.bus, self.cid, self.side, self.col = bus, cid, side, col
        self.pos = None
        self.spd = 0.0
        self.temp = 0
        self.err = 0
        self.vel = 0.0          # EMA of finite-difference velocity (deg/s)
        self._prev_pos = None

    def update_from(self, st, dt):
        if self._prev_pos is not None and dt > 0:
            v = (st["pos"] - self._prev_pos) / dt
            self.vel = 0.3 * v + 0.7 * self.vel
        self._prev_pos = st["pos"]
        self.pos, self.spd, self.temp, self.err = st["pos"], st["spd"], st["temp"], st["err"]


def drain(buses, motors_by_bus, dt):
    for ch, bus in buses.items():
        while True:
            msg = bus.recv(timeout=0.0)
            if msg is None:
                break
            if not msg.is_extended_id:
                continue
            m = motors_by_bus[ch].get(msg.arbitration_id & 0xFF)
            if m is not None:
                st = parse_status(msg.data)
                if st:
                    m.update_from(st, dt)


def preflight(motors, buses, motors_by_bus):
    print("Reading start positions ...")
    t_end = time.time() + 2.0
    while time.time() < t_end and any(m.pos is None for m in motors):
        drain(buses, motors_by_bus, 0.0)
        time.sleep(0.005)
    ok = True
    for m in motors:
        if m.pos is None:
            print(f"  !! {m.side} id{m.cid}: no status frame"); ok = False
        else:
            print(f"  {m.side:5s} id{m.cid}: pos={m.pos:+8.2f} deg temp={m.temp}C err={m.err}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="fixed_gait/trajectories/gait_recorded.npz")
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--period", type=float, default=8.0, help="seconds per cycle (bigger = slower)")
    ap.add_argument("--current-limit", type=float, default=1.0, help="HARD current cap per motor (A)")
    ap.add_argument("--kp", type=float, default=0.15, help="position gain (A per deg of error)")
    ap.add_argument("--kd", type=float, default=0.004, help="damping (A per deg/s)")
    ap.add_argument("--ramp", type=float, default=3.0, help="soft-start seconds (ease in target+current)")
    ap.add_argument("--duration", type=float, default=0.0, help="run time s (0 = until Ctrl+C)")
    ap.add_argument("--leg", choices=["both", "right", "left"], default="both")
    ap.add_argument("--abduction-right", type=float, default=None, help="hold angle for right abduction (deg)")
    ap.add_argument("--abduction-left", type=float, default=None, help="hold angle for left abduction (deg)")
    ap.add_argument("--max-current", type=float, default=None, help="absolute ceiling (default = limit)")
    ap.add_argument("--dry-run", action="store_true", help="compute + print, but command 0 A (no motion)")
    args = ap.parse_args()

    if can is None:
        print("python-can not installed. `pip install python-can`."); sys.exit(1)
    data = traj.load(args.file)
    sides = ["right", "left"] if args.leg == "both" else [args.leg]
    sides = [s for s in sides if data[s] is not None]
    if not sides:
        print(f"trajectory {args.file} has no calibration for {args.leg}. "
              f"Record that leg first."); sys.exit(1)
    abd_override = {"right": args.abduction_right, "left": args.abduction_left}
    ceiling = args.max_current if args.max_current is not None else args.current_limit

    buses, motors, motors_by_bus = {}, [], {}
    for s in sides:
        ch = SIDE_CHANNEL[s]
        buses[ch] = can.Bus(interface=args.interface, channel=ch, bitrate=BITRATE)
        motors_by_bus[ch] = {}
        for col, cid in enumerate(MOTOR_IDS):
            m = Motor(buses[ch], cid, s, col)
            motors.append(m); motors_by_bus[ch][cid] = m
    print(f"Loaded {args.file}. Playing {sides}, period={args.period}s, "
          f"current-limit={args.current_limit}A{' (DRY-RUN)' if args.dry_run else ''}")

    dt = 1.0 / CMD_HZ
    try:
        if not preflight(motors, buses, motors_by_bus):
            print("Aborting — not all motors reported. Nothing commanded."); return
        start_pos = {id(m): m.pos for m in motors}

        t0 = time.time()
        next_t = t0
        last_print = 0.0
        while True:
            now = time.time()
            elapsed = now - t0
            if args.duration and elapsed >= args.duration:
                break
            drain(buses, motors_by_bus, dt)

            # phase advances only after the soft-start; target eases from the start pose
            ramp = min(1.0, elapsed / args.ramp) if args.ramp > 0 else 1.0
            play_t = max(0.0, elapsed - args.ramp)
            phase = (play_t / args.period) % 1.0
            lim = ramp * args.current_limit

            aborted = None
            for m in motors:
                tgt_full = traj.reconstruct(data, m.side, phase,
                                            abduction_override=abd_override[m.side])[m.col]
                # ease from where the motor started to the trajectory target over the ramp
                target = (1 - ramp) * start_pos[id(m)] + ramp * tgt_full
                curr = args.kp * (target - m.pos) - args.kd * m.vel
                curr = float(np.clip(curr, -lim, lim))
                curr = float(np.clip(curr, -ceiling, ceiling))
                set_current(m.bus, m.cid, 0.0 if args.dry_run else curr)
                m._last_cmd = curr
                m._last_tgt = target
                if abs(m.spd) > MAX_SPEED_ERPM:
                    aborted = f"{m.side} id{m.cid} runaway {m.spd:.0f} ERPM"
                elif m.temp >= MAX_TEMP_C:
                    aborted = f"{m.side} id{m.cid} temp {m.temp}C"
                elif m.err:
                    aborted = f"{m.side} id{m.cid} error {m.err}"
            if aborted:
                print(f"\n!! SAFETY STOP: {aborted}. Releasing."); break

            if (now - last_print) > 0.2:
                last_print = now
                mt = max(m.temp for m in motors)
                mc = max(abs(m._last_cmd) for m in motors)
                print(f"  t={elapsed:6.1f}s phase={phase:4.2f} ramp={ramp:3.2f} "
                      f"maxI={mc:4.2f}/{lim:.2f}A maxT={mt}C   ", end="\r")

            next_t += dt
            s = next_t - time.time()
            if s > 0:
                time.sleep(s)
            else:
                next_t = time.time()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        for m in motors:
            try:
                set_current(m.bus, m.cid, 0.0)
            except Exception:
                pass
        time.sleep(0.02)
        for b in buses.values():
            try:
                b.shutdown()
            except Exception:
                pass
        print("Motors released (0 A).")


if __name__ == "__main__":
    main()
