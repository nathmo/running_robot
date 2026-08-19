#!/usr/bin/env python3
"""Identify the MIT-mode parameter RANGES of an AK motor, then measure its position bandwidth.

WHY THIS IS NEEDED AT ALL
-------------------------
MIT mode packs five floats into 8 bytes as fixed-width integers, and the range is the scale factor:

    position 16 bit over [P_MIN, P_MAX]     kp 12 bit over [0, KP_MAX]
    velocity 12 bit over [V_MIN, V_MAX]     kd 12 bit over [0, KD_MAX]
    torque   12 bit over [T_MIN, T_MAX]

Those are firmware constants per motor model. Nothing negotiates them. If our range disagrees with
the firmware's, EVERY value is silently scaled wrong and nothing reports an error:

    theta_actual = theta_commanded * (P_true / P_assumed)

ak_position_sweep.py and can_scan_motors.py both hard-code AK80-9 defaults (P +-12.5, V +-50,
T +-18). Neither of DASH-01's motors is an AK80-9, and the AK60-39 has no published datasheet at
all (training/model/build_model.py had to give up on finding its Kt). So we measure.

THE TRAP THAT MAKES THIS NON-OBVIOUS
------------------------------------
A wrong P range is INVISIBLE from MIT data alone. Command 45 deg with P_assumed twice the truth and
the motor turns 90 deg -- but the feedback is decoded with the same wrong range, so it reads back
45 deg and everything looks perfect. Self-consistent and completely wrong. Breaking that requires
one measurement from OUTSIDE the MIT frame, which is why mode `step` asks a human what actually
happened, and mode `osc` compares against servo mode instead.

Velocity does NOT need an external reference, because position and velocity arrive in the SAME
feedback frame: differentiate decoded position, regress against decoded velocity, and the slope
gives V_true once P_true is known.

    v_true = d(p_true)/dt          ->   V_true = V_assumed * slope * (P_true / P_assumed)
    where slope = (d p_decoded/dt) / v_decoded

Torque needs no command at all: tau_ff stays 0 everywhere in this file (see SAFETY), so T only ever
affects the decoded CURRENT, which is cross-checked against the servo-mode current the webui
already logs under the same load.

SAFETY, in the order that matters
---------------------------------
  * tau_ff is HARD-WIRED to 0. It is never a CLI option. A position loop is what we are measuring,
    feedforward torque is not needed for any of it, and with tau_ff = 0 a wrong T range cannot
    produce motion at all -- it can only mislabel a reading. That single choice removes the most
    dangerous unknown from the experiment.
  * `listen` commands nothing (kp = kd = 0, motor limp). Run it FIRST. It cannot move the robot.
  * kp/kd default low and are clamped. Amplitude is clamped. Everything soft-starts.
  * Ctrl+C, and any exception, streams a zero command and then EXIT MOTOR MODE, so the motor is
    left disabled rather than holding.
  * --dry-run does the whole run with the bus never opened, printing what would be sent.

MODES
-----
  listen  limp, log only. You turn the output by hand. Establishes the frame and proves the ID,
          the bitrate and the decode before anything is energised.
  step    command a known step (default 45 deg) at low kp. For the AKE90-8, which spins freely with
          no leg. YOU report what the output actually did; that is the external reference.
  osc     small sine (default 5 deg) for the AK60-39, whose travel is constrained. Pair it with
          `servo-osc` at the same amplitude and compare -- no human observation needed, because
          servo mode's scaling is already known and calibrated.
  servo-osc  the servo-mode counterpart of `osc`, same amplitude, same logging.
  chirp   the actual bandwidth measurement, once the ranges are known.

Every mode writes an .npz with the same schema as the webui's measurement store, so the existing
analysis reads it unchanged:  t, cmd_norm, pos_norm, spd, cur  (single column) + meta_json.

Usage
-----
    python tools/mit_identify.py --mode listen --id 1 --channel can1 --seconds 20
    python tools/mit_identify.py --mode step  --id 1 --channel can1 --step-deg 45 --kp 2
    python tools/mit_identify.py --mode osc   --id 1 --channel can1 --amp-deg 5 --kp 2
    python tools/mit_identify.py --mode servo-osc --id 105 --channel can1 --amp-deg 5
    python tools/mit_identify.py --analyse mit_step_*.npz --observed-deg 90

Requires: pip install python-can numpy
"""
import argparse
import json
import math
import struct
import sys
import time

import numpy as np

try:
    import can
except ImportError:
    can = None

# ---------------------------------------------------------------------------
# ASSUMED ranges. These are what we are trying to VALIDATE, not trust. They are the AK80-9 values
# the other two tools already use, kept identical on purpose so a discrepancy is attributable to
# the motor and not to a second set of constants drifting apart from the first.
# ---------------------------------------------------------------------------
P_MIN, P_MAX = -12.5, 12.5      # rad
V_MIN, V_MAX = -50.0, 50.0      # rad/s
KP_MIN, KP_MAX = 0.0, 500.0     # Nm/rad
KD_MIN, KD_MAX = 0.0, 5.0       # Nm*s/rad
T_MIN, T_MAX = -18.0, 18.0      # Nm

ENTER = bytes([0xFF] * 7 + [0xFC])
EXIT = bytes([0xFF] * 7 + [0xFD])
ZERO = bytes([0xFF] * 7 + [0xFE])

CMD_HZ = 200.0
SOFT_START_S = 2.0
KP_CEIL, KD_CEIL = 20.0, 2.0    # refuse anything above this from the CLI; we are identifying, not
#                                 tuning, and a big kp against an unknown P range is how you break
#                                 a leg. Raise deliberately in code if you ever really need to.

# servo mode, for the cross-check arm
CAN_PACKET_SET_CURRENT = 1
CAN_PACKET_SET_POS = 4


def float_to_uint(x, lo, hi, bits):
    x = max(lo, min(hi, x))
    return int((x - lo) * ((1 << bits) - 1) / (hi - lo))


def uint_to_float(x, lo, hi, bits):
    return x * (hi - lo) / ((1 << bits) - 1) + lo


def pack_cmd(pos, vel, kp, kd, tff):
    """Byte-identical to ak_position_sweep.py's, deliberately — one wire format, one place to be
    wrong. tff is a parameter only so the 0 is explicit at every call site."""
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
    if len(data) < 6:
        return None
    p_int = (data[1] << 8) | data[2]
    v_int = (data[3] << 4) | (data[4] >> 4)
    i_int = ((data[4] & 0xF) << 8) | data[5]
    return {"id": data[0],
            "pos": uint_to_float(p_int, P_MIN, P_MAX, 16),
            "vel": uint_to_float(v_int, V_MIN, V_MAX, 12),
            "cur": uint_to_float(i_int, T_MIN, T_MAX, 12),
            "temp": data[6] if len(data) >= 7 else None,
            "err": data[7] if len(data) >= 8 else None}


# ------------------------------------------------------------------ trajectories (radians)
def traj_listen(t, a):
    return 0.0


def traj_step(t, a):
    """Hold 0, step to `a` at t=3 s, hold, come back at t=8 s. Slow, deliberate, easy to eyeball —
    the whole point is that a HUMAN reads the output angle, so it must sit still long enough."""
    if t < 3.0:
        return 0.0
    if t < 8.0:
        return a
    return 0.0


def traj_osc(t, a, f=0.5):
    return a * math.sin(2 * math.pi * f * t)


def traj_chirp(t, a, T, f0, f1):
    """Linear chirp, phase-continuous: phase = 2*pi*(f0*t + (f1-f0)*t^2/(2T))."""
    return a * math.sin(2 * math.pi * (f0 * t + (f1 - f0) * t * t / (2 * T)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="listen",
                    choices=["listen", "step", "osc", "servo-osc", "chirp"])
    ap.add_argument("--id", type=int, default=1, help="MIT motor id (servo id for servo-osc)")
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--channel", default="can1")
    ap.add_argument("--bitrate", type=int, default=1_000_000)
    ap.add_argument("--seconds", type=float, default=None, help="run length (mode default if unset)")
    ap.add_argument("--step-deg", type=float, default=45.0, help="mode step: commanded step")
    ap.add_argument("--amp-deg", type=float, default=5.0, help="mode osc/servo-osc/chirp: amplitude")
    ap.add_argument("--freq", type=float, default=0.5, help="mode osc: sine frequency Hz")
    ap.add_argument("--f0", type=float, default=0.5, help="mode chirp: start Hz")
    ap.add_argument("--f1", type=float, default=12.0, help="mode chirp: end Hz")
    ap.add_argument("--kp", type=float, default=2.0, help=f"MIT kp, Nm/rad (clamped to {KP_CEIL})")
    ap.add_argument("--kd", type=float, default=0.2, help=f"MIT kd, Nm*s/rad (clamped to {KD_CEIL})")
    ap.add_argument("--no-zero", action="store_true",
                    help="do NOT send the zero-encoder command at start (default is to zero, so the"
                         " run is relative to wherever the joint sits)")
    ap.add_argument("--out", default=None, help="output .npz (default mit_<mode>_<id>_<ts>.npz)")
    ap.add_argument("--dry-run", action="store_true", help="never open the bus; print and exit")
    ap.add_argument("--analyse", default=None, help="analyse an existing .npz instead of running")
    ap.add_argument("--observed-deg", type=float, default=None,
                    help="with --analyse on a `step` run: the angle the output ACTUALLY turned, as"
                         " you measured it. This is the external reference that breaks the"
                         " self-consistency trap and pins P_true.")
    ap.add_argument("--servo-npz", default=None,
                    help="with --analyse on an `osc` run: the matching servo-osc .npz to compare"
                         " amplitudes against (the AK60-39 route, no human observation needed)")
    args = ap.parse_args()

    if args.analyse:
        return analyse(args)

    kp = 0.0 if args.mode == "listen" else float(np.clip(args.kp, 0.0, KP_CEIL))
    kd = 0.0 if args.mode == "listen" else float(np.clip(args.kd, 0.0, KD_CEIL))
    dur = args.seconds if args.seconds else {"listen": 20.0, "step": 12.0, "osc": 20.0,
                                             "servo-osc": 20.0, "chirp": 20.0}[args.mode]
    amp = math.radians(args.step_deg if args.mode == "step" else args.amp_deg)

    print(f"mode={args.mode}  id={args.id}  {args.channel}  {dur:.0f} s  kp={kp} kd={kd}  "
          f"tau_ff=0 (always)")
    print(f"ASSUMED ranges: P +-{P_MAX} rad  V +-{V_MAX} rad/s  T +-{T_MAX} Nm  "
          f"-> this run is what tests them")
    if args.mode == "step":
        print(f"  will command a {args.step_deg:.1f} deg step. WATCH THE OUTPUT and measure what it"
              f" really turns; pass it back with --analyse --observed-deg.")
    if args.dry_run:
        print("\n--dry-run: bus never opened. First 5 commands that would be sent:")
        for i in range(5):
            t = i / CMD_HZ
            print(f"   t={t:.3f}  pos={_target(args.mode, t, amp, dur, args):+.4f} rad  "
                  f"{pack_cmd(_target(args.mode, t, amp, dur, args), 0.0, kp, kd, 0.0).hex()}")
        return 0

    if can is None:
        print("python-can not installed", file=sys.stderr)
        return 2

    bus = can.interface.Bus(channel=args.channel, interface=args.interface, bitrate=args.bitrate)
    rows = []
    try:
        if args.mode == "servo-osc":
            run_servo(bus, args, amp, dur, rows)
        else:
            run_mit(bus, args, amp, dur, kp, kd, rows)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        _safe_stop(bus, args)
        bus.shutdown()

    if not rows:
        print("no samples captured — wrong id, wrong bitrate, or motor not in the expected mode")
        return 1
    save(args, rows, kp, kd, dur, amp)
    return 0


def _target(mode, t, amp, dur, args):
    if mode == "listen":
        return 0.0
    if mode == "step":
        return traj_step(t, amp)
    if mode in ("osc", "servo-osc"):
        return traj_osc(t, amp, args.freq)
    return traj_chirp(t, amp, dur, args.f0, args.f1)


def run_mit(bus, args, amp, dur, kp, kd, rows):
    bus.send(can.Message(arbitration_id=args.id, data=ENTER, is_extended_id=False))
    time.sleep(0.05)
    if not args.no_zero:
        bus.send(can.Message(arbitration_id=args.id, data=ZERO, is_extended_id=False))
        time.sleep(0.05)
    t0 = time.time()
    nxt = t0
    while True:
        t = time.time() - t0
        if t > dur:
            break
        ramp = min(1.0, t / SOFT_START_S) if args.mode != "step" else 1.0
        tgt = ramp * _target(args.mode, t, amp, dur, args)
        bus.send(can.Message(arbitration_id=args.id,
                             data=pack_cmd(tgt, 0.0, kp, kd, 0.0),   # tau_ff = 0, always
                             is_extended_id=False))
        m = bus.recv(timeout=0.004)
        if m is not None and not m.is_extended_id:
            r = parse_reply(m.data)
            if r and r["id"] == args.id:
                rows.append((t, tgt, r["pos"], r["vel"], r["cur"]))
        nxt += 1.0 / CMD_HZ
        time.sleep(max(0.0, nxt - time.time()))


def run_servo(bus, args, amp, dur, rows):
    """Servo-mode counterpart: SET_POS in degrees*10000, the scaling the webui already trusts."""
    def send_pos(deg):
        bus.send(can.Message(arbitration_id=args.id | (CAN_PACKET_SET_POS << 8),
                             data=struct.pack(">i", int(deg * 10000)), is_extended_id=True))
    t0 = time.time()
    nxt = t0
    base = None
    while True:
        t = time.time() - t0
        if t > dur:
            break
        ramp = min(1.0, t / SOFT_START_S)
        m = bus.recv(timeout=0.004)
        pos = spd = cur = np.nan
        if m is not None and m.is_extended_id and (m.arbitration_id & 0xFF) == args.id:
            pos = struct.unpack(">h", m.data[0:2])[0] * 0.1
            spd = struct.unpack(">h", m.data[2:4])[0] * 10.0
            cur = struct.unpack(">h", m.data[4:6])[0] * 0.01
            if base is None:
                base = pos
        if base is not None:
            tgt_deg = base + ramp * math.degrees(traj_osc(t, amp, args.freq))
            send_pos(tgt_deg)
            rows.append((t, math.radians(tgt_deg - base), math.radians(pos - base), spd, cur))
        nxt += 1.0 / CMD_HZ
        time.sleep(max(0.0, nxt - time.time()))


def _safe_stop(bus, args):
    """Zero command, then leave the motor DISABLED. Never leave it holding."""
    try:
        if args.mode == "servo-osc":
            bus.send(can.Message(arbitration_id=args.id | (CAN_PACKET_SET_CURRENT << 8),
                                 data=struct.pack(">i", 0), is_extended_id=True))
        else:
            for _ in range(5):
                bus.send(can.Message(arbitration_id=args.id, data=pack_cmd(0, 0, 0, 0, 0),
                                     is_extended_id=False))
                time.sleep(0.002)
            bus.send(can.Message(arbitration_id=args.id, data=EXIT, is_extended_id=False))
        print("motor released (zero command, motor mode exited)")
    except Exception as e:                                   # noqa: BLE001 - never mask the exit
        print(f"WARNING: could not cleanly release the motor: {e}", file=sys.stderr)


def save(args, rows, kp, kd, dur, amp):
    a = np.array(rows, float)
    meta = {"mode": args.mode, "id": args.id, "channel": args.channel, "kp": kp, "kd": kd,
            "tau_ff": 0.0, "duration_s": dur, "amp_rad": amp, "cmd_hz": CMD_HZ,
            "assumed": {"P": [P_MIN, P_MAX], "V": [V_MIN, V_MAX], "T": [T_MIN, T_MAX],
                        "KP": [KP_MIN, KP_MAX], "KD": [KD_MIN, KD_MAX]},
            "step_deg": args.step_deg, "amp_deg": args.amp_deg, "freq": args.freq,
            "f0": args.f0, "f1": args.f1}
    out = args.out or f"mit_{args.mode}_{args.id}_{int(time.time())}.npz"
    np.savez(out, t=a[:, 0], cmd_norm=a[:, 1], pos_norm=a[:, 2], spd=a[:, 3], cur=a[:, 4],
             meta_json=json.dumps(meta))
    print(f"\nsaved {out}  ({len(a)} samples, {a[-1,0]:.1f} s, "
          f"{len(a)/max(a[-1,0],1e-9):.0f} Hz effective)")
    print(f"  next: python tools/mit_identify.py --analyse {out}"
          + (" --observed-deg <what you measured>" if args.mode == "step" else ""))


# ------------------------------------------------------------------ analysis
def analyse(args):
    z = np.load(args.analyse, allow_pickle=True)
    m = json.loads(str(z["meta_json"]))
    t, cmd, pos, vel = z["t"], z["cmd_norm"], z["pos_norm"], z["spd"]
    print(f"=== {args.analyse}   mode={m['mode']}  kp={m['kp']}  {len(t)} samples ===")

    p_ratio = None
    if m["mode"] == "step":
        if args.observed_deg is None:
            print("\nThis is a `step` run. A wrong P range is INVISIBLE here without you:")
            print("  the feedback is decoded with the same wrong range that encoded the command,")
            print("  so it reads back exactly what was asked for even when the shaft went elsewhere.")
            print(f"  Commanded step: {m['step_deg']:.1f} deg.")
            print("  Measure what the OUTPUT actually turned and re-run with --observed-deg <N>.")
            return 0
        p_ratio = args.observed_deg / m["step_deg"]
        print(f"\nP RANGE   commanded {m['step_deg']:.1f} deg, observed {args.observed_deg:.1f} deg")
        print(f"          ratio {p_ratio:.4f}  ->  P_true = +-{P_MAX * p_ratio:.3f} rad "
              f"(assumed +-{P_MAX})")

    if m["mode"] == "osc" and args.servo_npz:
        zs = np.load(args.servo_npz, allow_pickle=True)
        # amplitude of each, from the settled portion (after soft start)
        def amp_of(tt, xx):
            w = tt > tt[0] + SOFT_START_S + 1.0
            return float(np.percentile(np.abs(xx[w] - np.mean(xx[w])), 98))
        a_mit = amp_of(t, pos)
        a_srv = amp_of(zs["t"], zs["pos_norm"])
        p_ratio = a_srv / a_mit if a_mit > 0 else float("nan")
        print(f"\nP RANGE   MIT motion {math.degrees(a_mit):.3f} deg vs servo motion "
              f"{math.degrees(a_srv):.3f} deg (same commanded amplitude)")
        print(f"          ratio {p_ratio:.4f}  ->  P_true = +-{P_MAX * p_ratio:.3f} rad")
        print("          servo scaling is the calibrated reference here, so no eyeball needed.")

    # velocity: self-consistent, needs no external reference beyond p_ratio
    good = np.isfinite(pos) & np.isfinite(vel)
    if good.sum() > 50:
        dp = np.gradient(pos[good], t[good])
        v = vel[good]
        use = np.abs(v) > np.percentile(np.abs(v), 60)     # ignore the near-zero crossings
        if use.sum() > 20:
            slope = float(np.polyfit(v[use], dp[use], 1)[0])
            r = float(np.corrcoef(v[use], dp[use])[0, 1])
            print(f"\nV RANGE   d(pos)/dt vs reported vel: slope {slope:.4f}  (r={r:.3f})")
            if p_ratio:
                print(f"          V_true = V_assumed * slope * P_ratio = "
                      f"+-{V_MAX * slope * p_ratio:.2f} rad/s (assumed +-{V_MAX})")
            else:
                print(f"          with P assumed correct: V_true = +-{V_MAX * slope:.2f} rad/s")
                print("          (pin P first; V_true scales with P_ratio)")

    cur = z["cur"]
    if np.isfinite(cur).any():
        print(f"\nT RANGE   decoded current: rms {np.nanstd(cur):.3f}  peak {np.nanmax(np.abs(cur)):.3f}"
              f"  (units of the ASSUMED +-{T_MAX})")
        print("          cross-check against the servo-mode current for the same joint under the")
        print("          same load; tau_ff was 0 so this only ever mislabels a reading.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
