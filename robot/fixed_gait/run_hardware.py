#!/usr/bin/env python3
"""Stream the hard-coded walking gait to DASH-01's six CubeMars/AK motors over CAN (servo mode).

Runs on the Raspberry Pi (onnxruntime/numpy/python-can only — NO mujoco). Imports the SAME
gait.py the simulator uses, so what you saw in sim_fixed_base.py is exactly what the motors get.

The robot base is FIXED IN SPACE (in a rig / clamped). This is a demonstrator: it proves all six
motors move together in a walking pattern. There is no balance loop.

WIRING (from you):
    can0 = RIGHT leg motors,  can1 = LEFT leg motors.
    On each bus the CubeMars IDs are:  104 = abduction,  105 = cam,  106 = hip.

CONTROL PATH (identical protocol to tools/ak_servo_sweep.py, which is tested):
    servo-mode CAN_PACKET_SET_POS at 100 Hz, position in degrees, big-endian int32 * 10000.
    Direct 1:1 coupling: joint angle (rad) -> motor output-shaft degrees via * 180/pi.

HOME = CAPTURE AT STARTUP (your choice):
    At launch each motor's CURRENT position is read and taken as that joint's home (= the gait
    center pose). The gait then adds OFFSETS on top, so the first command equals the start
    position (no jump). >>> Pose the legs in the nominal standing pose before you start. <<<

SIGN CALIBRATION:
    The sim's +offset direction may or may not match a motor's +rotation. Each joint has a SIGN
    (+1/-1) in CALIB below, default +1. Confirm every joint once with --test-joint before running
    the full gait (see the bring-up steps in fixed_gait/README.md).

Examples:
    python fixed_gait/run_hardware.py --test-joint thigh_L --deg 8   # jog one joint, watch direction
    python fixed_gait/run_hardware.py --amp-scale 0.3                # first full run at 30% amplitude
    python fixed_gait/run_hardware.py                                # full gait
    python fixed_gait/run_hardware.py --settle-only                 # just hold the center pose
"""
import argparse
import math
import struct
import sys
import time

import numpy as np

try:
    import can
except ImportError:
    can = None

from gait import (GaitParams, GaitGenerator,
                  HIP_ROLL_L, CAM_L, THIGH_L, HIP_ROLL_R, CAM_R, THIGH_R)
import joint_limits

RAD2DEG = 180.0 / math.pi

# ----- servo-mode protocol (copied from tools/ak_servo_sweep.py; keep in sync) -----
CAN_PACKET_SET_CURRENT = 1
CAN_PACKET_SET_POS = 4
BITRATE = 1_000_000


def send_servo(bus, cid, command, payload):
    bus.send(can.Message(arbitration_id=cid | (command << 8), data=payload, is_extended_id=True))


def set_pos(bus, cid, pos_deg):
    # int32 big-endian, degrees * 10000 (CubeMars manual V1.0.14 sec 5.1.5; multi-turn +-36000 deg)
    val = int(round(pos_deg * 10_000.0))
    val = max(-2_147_483_648, min(2_147_483_647, val))
    send_servo(bus, cid, CAN_PACKET_SET_POS, struct.pack(">i", val))


def set_current(bus, cid, amps):
    send_servo(bus, cid, CAN_PACKET_SET_CURRENT, struct.pack(">i", int(round(amps * 1000.0))))


def parse_status(data):
    if len(data) < 8:
        return None
    pos = struct.unpack(">h", bytes(data[0:2]))[0] * 0.1
    spd = struct.unpack(">h", bytes(data[2:4]))[0] * 10.0
    cur = struct.unpack(">h", bytes(data[4:6]))[0] * 0.01
    temp = struct.unpack(">b", bytes(data[6:7]))[0]
    err = data[7]
    return {"pos": pos, "spd": spd, "cur": cur, "temp": temp, "err": err}


# ============================ CALIBRATION TABLE ============================
# One row per joint, in gait actuator order. Edit SIGN (and gear, if ever not 1:1) during bring-up.
#   name, gait_index, channel, motor_id, sign, gear
# sign: +1 if the motor's +rotation matches the sim's +offset for this joint; -1 to flip.
# gear: motor-shaft degrees per joint degree (1.0 for the direct 1:1 coupling you confirmed).
CALIB = [
    ("hip_roll_L", HIP_ROLL_L, "can1", 104, +1.0, 1.0),
    ("cam_L",      CAM_L,      "can1", 105, +1.0, 1.0),
    ("thigh_L",    THIGH_L,    "can1", 106, +1.0, 1.0),
    ("hip_roll_R", HIP_ROLL_R, "can0", 104, +1.0, 1.0),
    ("cam_R",      CAM_R,      "can0", 105, +1.0, 1.0),
    ("thigh_R",    THIGH_R,    "can0", 106, +1.0, 1.0),
]

# ----- safety limits -----
MAX_OFFSET_DEG = 30.0     # refuse to command more than this from home (bounds a bad param/gear)
MAX_TRACK_ERR_DEG = 25.0  # cut if a motor's actual position strays this far from its command
MAX_TEMP_C = 80           # cut above this motor temperature
CMD_RATE_HZ = 100.0


class Motor:
    def __init__(self, bus, name, cid, sign, gear):
        self.bus, self.name, self.cid, self.sign, self.gear = bus, name, cid, sign, gear
        self.home_deg = None
        self.last_cmd_deg = None
        self.last_fb = None

    def read_status(self, timeout=1.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.bus.recv(timeout=max(0.0, deadline - time.time()))
            if msg is None:
                break
            if msg.is_extended_id and (msg.arbitration_id & 0xFF) == self.cid:
                st = parse_status(msg.data)
                if st:
                    self.last_fb = st
                    return st
        return None

    def drain_latest(self):
        """Non-blocking: return the most recent status frame for this motor id (or None)."""
        fb = None
        while True:
            msg = self.bus.recv(timeout=0.0)
            if msg is None:
                break
            if msg.is_extended_id and (msg.arbitration_id & 0xFF) == self.cid:
                st = parse_status(msg.data)
                if st:
                    fb = st
        if fb is not None:
            self.last_fb = fb
        return fb

    def target_deg(self, offset_deg):
        """Absolute raw motor degrees for home + sign*gear*offset, WITHOUT sending anything --
        lets a target be workspace-checked before it's committed."""
        return self.home_deg + self.sign * self.gear * offset_deg

    def command_offset(self, offset_deg):
        """Command home + sign*gear*offset (joint degrees)."""
        cmd = self.target_deg(offset_deg)
        self.last_cmd_deg = cmd
        set_pos(self.bus, self.cid, cmd)

    def release(self):
        set_current(self.bus, self.cid, 0.0)


def open_buses(interface, channels):
    buses = {}
    for ch in channels:
        buses[ch] = can.Bus(interface=interface, channel=ch, bitrate=BITRATE)
    return buses


def build_motors(buses):
    motors = []
    idx_of = {}
    for name, gidx, ch, cid, sign, gear in CALIB:
        m = Motor(buses[ch], name, cid, sign, gear)
        motors.append(m)
        idx_of[name] = (m, gidx)
    return motors, idx_of


def leg_groups(motors):
    """{'left': {'abd':Motor,'cam':Motor,'thigh':Motor}, 'right': {...}} -- for workspace checks."""
    role_of = {"hip_roll": "abd", "cam": "cam", "thigh": "thigh"}
    groups = {"left": {}, "right": {}}
    for m in motors:
        leg = "left" if m.name.endswith("_L") else "right"
        role = next(r for prefix, r in role_of.items() if m.name.startswith(prefix))
        groups[leg][role] = m
    return groups


def check_workspace(groups, abs_pos, limits):
    """abs_pos: dict Motor -> absolute raw target deg (about to be sent, or already held).
    Checks every leg that has calibration; None limits (no calibration file yet) always passes."""
    if limits is None:
        return True, ""
    for leg, roles in groups.items():
        if not limits.has_leg(leg):
            continue
        ok, reason = limits.validate(leg, abs_pos[roles["abd"]], abs_pos[roles["cam"]],
                                     abs_pos[roles["thigh"]])
        if not ok:
            return False, reason
    return True, ""


def preflight(motors):
    """Read every motor's start position = its home. Abort if any is silent (never command blind)."""
    print("Reading start positions (home = current pose). Keep the robot still...")
    ok = True
    for m in motors:
        st = m.read_status(timeout=1.5)
        if st is None:
            print(f"  !! {m.name} (id {m.cid}): NO status frame — powered? on this bus? servo mode?")
            ok = False
            continue
        m.home_deg = st["pos"]
        m.last_cmd_deg = st["pos"]
        flag = "" if st["err"] == 0 else f"  ERR={st['err']}"
        print(f"  {m.name:10s} id{m.cid}: home={st['pos']:+8.2f} deg  temp={st['temp']}C{flag}")
    return ok


def safety_check(motors):
    """Return (aborted, reason). Cuts on tracking error, motor error flag, or over-temp."""
    for m in motors:
        fb = m.drain_latest()
        if fb is None:
            continue
        if fb["err"]:
            return True, f"{m.name} error code {fb['err']}"
        if fb["temp"] >= MAX_TEMP_C:
            return True, f"{m.name} temp {fb['temp']}C >= {MAX_TEMP_C}"
        if m.last_cmd_deg is not None and abs(fb["pos"] - m.last_cmd_deg) > MAX_TRACK_ERR_DEG:
            return True, (f"{m.name} tracking error {fb['pos'] - m.last_cmd_deg:+.1f} deg "
                          f"(> {MAX_TRACK_ERR_DEG})")
    return False, ""


def release_all(motors):
    for m in motors:
        try:
            m.release()
        except Exception:
            pass


# --------------------------------------------------------------------------- modes
def run_test_joint(motors, idx_of, gen, name, deg, hold, groups, limits):
    """Slowly ramp ONE joint to +deg (home-relative) and back, for sign/direction calibration."""
    if name not in idx_of:
        print(f"unknown joint '{name}'. choices: {[c[0] for c in CALIB]}")
        return
    m, _ = idx_of[name]
    deg = min(abs(deg), MAX_OFFSET_DEG) * (1 if deg >= 0 else -1)
    print(f"Jogging {name} to {deg:+.1f} deg (home-relative) over 2 s, holding {hold}s, back.")
    print(f"  sign={m.sign:+.0f}. If it moves the WRONG way, flip this joint's sign in CALIB.")
    dt = 1.0 / CMD_RATE_HZ
    for phase in ("up", "hold", "down"):
        n = int((2.0 if phase != "hold" else hold) / dt)
        for k in range(max(1, n)):
            if phase == "up":
                off = deg * (k + 1) / n
            elif phase == "down":
                off = deg * (1 - (k + 1) / n)
            else:
                off = deg
            abs_pos = {mm: mm.last_cmd_deg for mm in motors}
            abs_pos[m] = m.target_deg(off)
            ok, reason = check_workspace(groups, abs_pos, limits)
            if not ok:
                print(f"\n  !! SAFETY STOP: {reason}. Not commanding, cutting."); return
            m.command_offset(off)
            fb = m.drain_latest()
            if fb and fb["err"]:
                print(f"  !! error {fb['err']}, cutting."); return
            if fb:
                print(f"   off={off:+6.2f} cmd={m.last_cmd_deg:+7.2f} act={fb['pos']:+7.2f} "
                      f"I={fb['cur']:+4.1f}A T={fb['temp']}C   ", end="\r")
            time.sleep(dt)
    print("\n done.")


def run_gait(motors, idx_of, gen, duration, settle_only, groups, limits):
    dt = 1.0 / CMD_RATE_HZ
    center = gen.center_pose()
    t0 = time.time()
    next_t = t0
    last_print = 0.0
    print("\nStreaming gait. Ctrl+C to stop (motors released on exit).")
    while True:
        now = time.time()
        t = now - t0
        if duration and t >= duration:
            break
        targets = center if settle_only else gen.targets(t)      # rad, sim convention
        offsets_deg = (targets - center) * RAD2DEG               # home-relative joint offsets
        # clamp offsets, workspace-check the ABSOLUTE targets, THEN command (reject before sending)
        pending = {}
        abs_pos = {m: m.last_cmd_deg for m in motors}
        for m, gidx in (idx_of[n] for n, *_ in CALIB):
            off = float(np.clip(offsets_deg[gidx], -MAX_OFFSET_DEG, MAX_OFFSET_DEG))
            pending[m] = off
            abs_pos[m] = m.target_deg(off)
        ok, reason = check_workspace(groups, abs_pos, limits)
        if not ok:
            print(f"\n!! SAFETY STOP: {reason}. Not commanding, releasing motors.")
            break
        for m, off in pending.items():
            m.command_offset(off)
        aborted, reason = safety_check(motors)
        if aborted:
            print(f"\n!! SAFETY STOP: {reason}. Releasing motors.")
            break
        if (now - last_print) > 0.2:
            last_print = now
            temps = max((m.last_fb["temp"] for m in motors if m.last_fb), default=0)
            print(f"  t={t:6.1f}s  max|off|={np.max(np.abs(offsets_deg)):5.1f}deg  "
                  f"maxT={temps}C   ", end="\r")
        next_t += dt
        sleep = next_t - time.time()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.time()


def make_params(args):
    p = GaitParams()
    p.thigh_amp *= args.amp_scale
    p.cam_amp *= args.amp_scale
    if args.period is not None:
        p.period_s = args.period
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--amp-scale", type=float, default=1.0,
                    help="scale gait amplitudes (start low, e.g. 0.3, for first runs)")
    ap.add_argument("--period", type=float, default=None, help="stride period seconds")
    ap.add_argument("--duration", type=float, default=0.0, help="run time s (0 = until Ctrl+C)")
    ap.add_argument("--settle-only", action="store_true", help="hold the center pose, do not step")
    ap.add_argument("--test-joint", default=None, help="jog one joint for sign calibration")
    ap.add_argument("--deg", type=float, default=8.0, help="jog amplitude for --test-joint")
    ap.add_argument("--hold", type=float, default=1.0, help="hold seconds for --test-joint")
    ap.add_argument("--countdown", type=float, default=3.0, help="seconds before motion starts")
    args = ap.parse_args()

    if can is None:
        print("python-can not installed. `pip install python-can` (and bring up the CAN buses).")
        sys.exit(1)

    channels = sorted({row[2] for row in CALIB})
    print(f"Opening {args.interface} {channels} @ {BITRATE} bps ...")
    buses = open_buses(args.interface, channels)
    motors, idx_of = build_motors(buses)
    groups = leg_groups(motors)
    limits = joint_limits.load_or_warn()
    gen = GaitGenerator(make_params(args))

    try:
        if not preflight(motors):
            print("Aborting: not all motors reported. Nothing was commanded.")
            return
        print(f"center pose (rad): {np.round(gen.center_pose(), 3)}")
        if args.test_joint:
            run_test_joint(motors, idx_of, gen, args.test_joint, args.deg, args.hold, groups, limits)
            return
        mode = "SETTLE (hold center)" if args.settle_only else f"WALK amp={args.amp_scale:g}"
        print(f"\nMode: {mode}. Motion in {args.countdown:g}s — Ctrl+C to abort.")
        for s in range(int(args.countdown), 0, -1):
            print(f"  {s}...", end=" ", flush=True)
            time.sleep(1.0)
        print()
        run_gait(motors, idx_of, gen, args.duration, args.settle_only, groups, limits)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        release_all(motors)
        time.sleep(0.02)
        for b in buses.values():
            try:
                b.shutdown()
            except Exception:
                pass
        print("Motors released (0 A). Buses closed.")


if __name__ == "__main__":
    main()
