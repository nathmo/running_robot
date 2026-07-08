#!/usr/bin/env python3
"""Play back a recorded trajectory on both legs, dephased 180 deg, at an adjustable speed.

Two control modes (--mode):
  current  (default) — a software PID on the Pi drives SET_CURRENT, hard-clamped to a torque
                       (current) limit. Compliant + torque-capped; kp/ki/kd tune it.
  position           — stream SET_POS position waypoints; the MOTOR's internal loop tracks them
                       (rock-solid tracking, but NO torque cap from our side). kp/ki/kd and the
                       current/speed limits are ignored; a position tracking-error guard applies.

Control (current mode): a software PID loop per motor drives current (servo-mode SET_CURRENT, same command as
tools/ak_servo_torque.py). The commanded current is hard-clamped to --current-limit, so torque can
never exceed that. Amps (not Nm) because the motor's Nm-per-amp (Kt) is not known exactly.

    current = kp*err + ki*integral(err) + kd*(target_vel - actual_vel),  clamped to +-limit
    err = target - actual        (deg)

  - Raise kp for STRICTER tracking (less compliant). At kp=0.15 the leg only pulled ~2 A even with
    a 20 A limit -> way too soft; friction/gravity then make it stick-slip (jagged). Push kp up.
  - ki removes the steady lag from gravity/friction (integral, with anti-windup).
  - kd damps; it uses (target_vel - actual_vel) so it tracks the MOVING trajectory instead of
    braking against it. No gravity feedforward (yet).

    can0 = RIGHT leg, can1 = LEFT leg. Motors 104=abduction (held fixed), 105=cam, 106=hip.

Tuning recipe:
    1. --dry-run first (prints targets, commands 0 A).
    2. Slow + soft:   --period 10 --current-limit 3 --kp 0.4 --ki 0 --kd 0
    3. Raise --kp until tracking is crisp but not buzzing; then add --ki to kill the lag;
       add a little --kd only if it oscillates. --log saves a target-vs-actual plot to tune from.

Examples:
    python fixed_gait/play_trajectory.py --dry-run
    python fixed_gait/play_trajectory.py --period 8 --current-limit 3 --kp 0.8 --ki 0.4 --log
    python fixed_gait/play_trajectory.py --mode position --period 8 --log     # drive runs the loop
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
CAN_PACKET_SET_POS = 4
CMD_HZ = 200.0

# Speed handling: a soft GOVERNOR tapers accelerating current as the motor nears --speed-limit,
# so speed saturates there smoothly (no crash). The hard --max-speed cut is only a last-resort
# runaway net, set well above the governor so normal fast moves never trip it.
DEFAULT_SPEED_LIMIT = 9000.0    # ERPM: governor onset (0 = governor off)
DEFAULT_MAX_SPEED = 16000.0     # ERPM: hard emergency cut
MAX_TEMP_C = 80


def set_current(bus, cid, amps):
    val = int(round(amps * 1000.0))
    val = max(-2_147_483_648, min(2_147_483_647, val))
    bus.send(can.Message(arbitration_id=cid | (CAN_PACKET_SET_CURRENT << 8),
                         data=val.to_bytes(4, "big", signed=True), is_extended_id=True))


def set_pos(bus, cid, pos_deg):
    # servo-mode position waypoint: deg * 10000, big-endian int32 (same as tools/ak_servo_sweep.py).
    # The motor's own internal loop tracks it — no torque cap from our side.
    val = int(round(pos_deg * 10_000.0))
    val = max(-2_147_483_648, min(2_147_483_647, val))
    bus.send(can.Message(arbitration_id=cid | (CAN_PACKET_SET_POS << 8),
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
        self.cur = 0.0             # drive-reported current (A)
        self.temp = 0
        self.err = 0
        self.vel = 0.0              # EMA finite-difference velocity (deg/s)
        self._prev_pos = None
        self.integ = 0.0           # PID integral state (deg*s)
        self.prev_target = None
        self.tvel = 0.0            # EMA target velocity (deg/s)
        self.last_cmd = 0.0
        self.last_tgt = 0.0

    def update_from(self, st, dt):
        if self._prev_pos is not None and dt > 0:
            self.vel = 0.3 * (st["pos"] - self._prev_pos) / dt + 0.7 * self.vel
        self._prev_pos = st["pos"]
        self.pos, self.spd, self.temp, self.err = st["pos"], st["spd"], st["temp"], st["err"]
        self.cur = st["cur"]


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


def save_log(log, motors, path):
    t = np.array(log["t"])
    arr = {k: np.array(v) for k, v in log.items()}
    np.savez(path + ".npz", labels=[f"{m.side}_{n}" for m in motors
                                     for n in [["abd", "cam", "hip"][m.col]]], **arr)
    print(f"saved tracking log -> {path}.npz")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed on this machine -> plot skipped; use the .npz elsewhere)")
        return
    nm = len(motors)
    fig, ax = plt.subplots(nm, 1, figsize=(11, 2.1 * nm), sharex=True)
    ax = np.atleast_1d(ax)
    for i, m in enumerate(motors):
        lbl = f"{m.side} {['abd','cam','hip'][m.col]} (id{m.cid})"
        ax[i].plot(t, arr["tgt"][:, i], "k--", lw=1.2, label="target")
        ax[i].plot(t, arr["act"][:, i], "b", lw=1.0, label="actual")
        axc = ax[i].twinx()
        axc.plot(t, arr["cur"][:, i], "r", lw=0.7, alpha=0.5)
        axc.set_ylabel("A", color="r"); axc.tick_params(axis="y", colors="r")
        ax[i].set_ylabel("deg"); ax[i].set_title(lbl, fontsize=9, loc="left")
        ax[i].grid(alpha=.3); ax[i].legend(loc="upper right", fontsize=7)
    ax[-1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(path + ".png", dpi=110)
    print(f"saved tracking plot -> {path}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="fixed_gait/trajectories/gait_recorded.npz")
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--mode", choices=["current", "position"], default="current",
                    help="'current' = Python PID -> SET_CURRENT (torque-limited); "
                         "'position' = stream SET_POS waypoints, the drive runs the loop (no torque cap). "
                         "kp/ki/kd/current-limit/speed-limit apply to 'current' only.")
    ap.add_argument("--max-track-err", type=float, default=30.0,
                    help="position mode: cut if actual strays this many deg from the commanded waypoint")
    ap.add_argument("--period", type=float, default=8.0, help="seconds per cycle (bigger = slower)")
    ap.add_argument("--current-limit", type=float, default=3.0, help="HARD current cap per motor (A)")
    ap.add_argument("--kp", type=float, default=0.8, help="P gain (A per deg of error)")
    ap.add_argument("--ki", type=float, default=0.4, help="I gain (A per deg*s); kills gravity/friction lag")
    ap.add_argument("--kd", type=float, default=0.02, help="D gain (A per deg/s); damping")
    ap.add_argument("--ramp", type=float, default=3.0, help="soft-start seconds (ease in target+current)")
    ap.add_argument("--duration", type=float, default=0.0, help="run time s (0 = until Ctrl+C)")
    ap.add_argument("--leg", choices=["both", "right", "left"], default="both")
    ap.add_argument("--abduction-right", type=float, default=None, help="hold angle for right abduction (deg)")
    ap.add_argument("--abduction-left", type=float, default=None, help="hold angle for left abduction (deg)")
    ap.add_argument("--max-current", type=float, default=None, help="absolute ceiling (default = limit)")
    ap.add_argument("--speed-limit", type=float, default=DEFAULT_SPEED_LIMIT,
                    help="soft speed governor onset (ERPM); tapers accel current so speed saturates. 0=off")
    ap.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED,
                    help="hard runaway cut (ERPM); a last-resort net above the governor")
    ap.add_argument("--dry-run", action="store_true", help="compute + print, but command 0 A (no motion)")
    ap.add_argument("--log", nargs="?", const="fixed_gait/trajectories/last_run",
                    default=None, help="record target-vs-actual and save a plot (path prefix)")
    args = ap.parse_args()

    if can is None:
        print("python-can not installed. `pip install python-can`."); sys.exit(1)
    data = traj.load(args.file)
    sides = ["right", "left"] if args.leg == "both" else [args.leg]
    sides = [s for s in sides if data[s] is not None]
    if not sides:
        print(f"trajectory {args.file} has no calibration for {args.leg}."); sys.exit(1)
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
    dry = " (DRY-RUN)" if args.dry_run else ""
    print(f"Loaded {args.file}. Playing {sides}, mode={args.mode.upper()}, period={args.period}s{dry}")
    if args.mode == "position":
        print(f"POSITION mode: streaming SET_POS waypoints — the DRIVE runs the loop (no torque cap). "
              f"ramp={args.ramp}s  max-track-err={args.max_track_err:.0f} deg")
    else:
        print(f"CURRENT mode: Python PID -> SET_CURRENT, torque cap={args.current_limit}A. "
              f"kp={args.kp} ki={args.ki} kd={args.kd} ramp={args.ramp}s "
              f"speed-limit={args.speed_limit:.0f} max-speed={args.max_speed:.0f} ERPM")

    dt = 1.0 / CMD_HZ
    log = {"t": [], "tgt": [], "act": [], "cur": []} if args.log else None
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

            ramp = min(1.0, elapsed / args.ramp) if args.ramp > 0 else 1.0
            play_t = max(0.0, elapsed - args.ramp)
            phase = (play_t / args.period) % 1.0
            lim = ramp * args.current_limit

            aborted = None
            row_t, row_a, row_c = [], [], []
            for m in motors:
                tgt_full = traj.reconstruct(data, m.side, phase,
                                            abduction_override=abd_override[m.side])[m.col]
                target = (1 - ramp) * start_pos[id(m)] + ramp * tgt_full

                if args.mode == "position":
                    # stream a position waypoint; the drive's own loop tracks it (no torque cap).
                    if not args.dry_run:
                        set_pos(m.bus, m.cid, target)
                    logged = m.cur                         # drive-reported current
                    m.last_cmd, m.last_tgt = m.cur, target
                    if abs(target - m.pos) > args.max_track_err:
                        aborted = (f"{m.side} id{m.cid} tracking err {target - m.pos:+.0f} deg "
                                   f"(> --max-track-err {args.max_track_err:.0f}) — hitting a stop?")
                else:
                    # current mode: Python PID -> SET_CURRENT, hard-clamped (torque-limited)
                    if m.prev_target is not None:
                        m.tvel = 0.3 * (target - m.prev_target) / dt + 0.7 * m.tvel
                    m.prev_target = target
                    err = target - m.pos
                    if ramp >= 1.0 and args.ki > 0:        # integral, anti-windup after soft-start
                        m.integ += err * dt
                        m.integ = float(np.clip(m.integ, -lim / args.ki, lim / args.ki))
                    d_term = args.kd * (m.tvel - m.vel)
                    curr = args.kp * err + args.ki * m.integ + d_term
                    curr = float(np.clip(np.clip(curr, -lim, lim), -ceiling, ceiling))
                    # speed governor: taper only the ACCELERATING current so speed saturates near
                    # --speed-limit (braking is never limited).
                    if args.speed_limit > 0 and curr * m.spd > 0 and abs(m.spd) > args.speed_limit:
                        band = 0.3 * args.speed_limit
                        curr *= float(np.clip((args.speed_limit + band - abs(m.spd)) / band, 0.0, 1.0))
                    set_current(m.bus, m.cid, 0.0 if args.dry_run else curr)
                    logged = curr
                    m.last_cmd, m.last_tgt = curr, target
                    if abs(m.spd) > args.max_speed:
                        aborted = f"{m.side} id{m.cid} runaway {m.spd:.0f} ERPM (> --max-speed {args.max_speed:.0f})"

                row_t.append(target); row_a.append(m.pos); row_c.append(logged)
                if m.temp >= MAX_TEMP_C:
                    aborted = f"{m.side} id{m.cid} temp {m.temp}C"
                elif m.err:
                    aborted = f"{m.side} id{m.cid} error {m.err}"

            if log is not None:
                log["t"].append(elapsed); log["tgt"].append(row_t)
                log["act"].append(row_a); log["cur"].append(row_c)
            if aborted:
                print(f"\n!! SAFETY STOP: {aborted}. Releasing."); break

            if (now - last_print) > 0.2:
                last_print = now
                mt = max(m.temp for m in motors)
                mc = max(abs(m.last_cmd) for m in motors)
                me = max(abs(m.last_tgt - m.pos) for m in motors)
                ms = max(abs(m.spd) for m in motors)
                print(f"  t={elapsed:6.1f}s phase={phase:4.2f} maxErr={me:5.1f}deg "
                      f"maxI={mc:4.2f}/{lim:.2f}A maxSpd={ms:5.0f}erpm maxT={mt}C   ", end="\r")

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
        if log is not None and log["t"]:
            save_log(log, motors, args.log)


if __name__ == "__main__":
    main()
