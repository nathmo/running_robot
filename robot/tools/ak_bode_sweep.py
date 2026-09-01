#!/usr/bin/env python3
"""Stepped-sine bandwidth sweep of one AK drive, servo mode or MIT impedance mode.

WHAT THIS MEASURES AND WHY IT SUPERSEDES THE 2026-08 SWEEPS
------------------------------------------------------------
Position command in, reported position out, at --df Hz steps from --f0 to --f1 with >= --dwell
seconds per frequency. BOTH arms are position loops, so servo and MIT are finally an
apples-to-apples comparison at the interface each mode is actually used through:

    servo  SET_POS, the drive's internal position loop (what fixed_gait uses)
    mit    force-control impedance, tau = kp*(p_des - p) - kd*v with THE DEPLOYED GAINS
           (kp=200, kd=5 -- the <position kp=200 kv=5> servo the policy was trained against,
           walk_mit/config.py:497; override with --kp/--kd)

The 2026-08 measurement paired a servo position chirp with an MIT *velocity* loop (a different
loop, free rotor, ERPM-referenced to 1 Hz) because the velocity span was unidentified. Here
v_des and tau_ff stay 0 -- robot/deploy/mit.py raises on anything else -- so no unidentified
span is anywhere in the path, and the MIT arm needs no reference-point normalisation.

AMPLITUDE SCHEDULE
------------------
A(f) = min(--amp-max, --vel-amp / (2 pi f)): constant position amplitude at low f, constant
VELOCITY amplitude above the corner, so neither excursion nor speed grows with frequency.
Frequency steps happen at upward zero crossings, so position (and, above the corner, velocity)
is continuous across every step.

SAFETY, in the order that matters
---------------------------------
  * The robot must be HOMED and the legs UNLOADED (in the air, on the stand). The run REFUSES
    to start if the holding current seen during the listen phase exceeds --hold-cur-gate: a
    loaded joint sags hard the moment the servo hold is released into a ramping MIT gain.
  * tau_ff and v_des are structurally zero (deploy/mit.py raises on anything else).
  * The MIT arm takes over from the servo hold AT the held position: p_des = center while the
    gains ramp 0 -> kp/kd over 1 s, so the proportional term never sees a step. Amplitude then
    ramps over 1 s. Every abort/exit path streams zero-gain force frames + SET_CURRENT 0.
  * A 200 Hz watch aborts on: drive error flag, temperature >= --temp-abort, |current| >=
    --cur-abort (instant) or > --cur-mean-abort averaged over 2 s, position outside the safe
    window, a single-frame position step > 20 deg (the drives renumber their origin -- that is
    NUMBERING, not motion, and the record is invalid either way), > 0.15 s without a status
    frame, or 50 consecutive failed sends (never command a joint we cannot see or reach).
  * A clean run ends by walking the target back to center at full authority and handing the
    drive to servo mode holding there -- the robot stays homed for the next run.

The webui daemon owns the CAN buses: `sudo systemctl stop runningrobot-webui` first.

Usage (on the Pi, one actuator x one mode per invocation):
    python robot/tools/ak_bode_sweep.py --channel can1 --id 106 --mode servo --out /tmp/x.npz
    python robot/tools/ak_bode_sweep.py --channel can1 --id 106 --mode mit   --out /tmp/y.npz
Analysis/plot: tools/plot_actuator_bode.py (desktop).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "fixed_gait", "webui"))
sys.path.insert(0, os.path.join(_HERE, "..", "deploy"))
import canio                     # noqa: E402
import mit                       # noqa: E402

CMD_HZ = 200.0
DT = 1.0 / CMD_HZ
GAIN_RAMP_S = 1.0                # MIT: gains 0 -> full at the held center, before any motion
AMP_RAMP_S = 1.0                 # first-frequency amplitude ramp
SETTLE_EXCLUDE_S = 1.2           # analysis drops this much after every frequency switch
POS_WINDOW_MARGIN_RAD = 0.12     # abort beyond center +- (A + this) while sweeping
RAMP_WINDOW_RAD = 0.25           # flat window during the gain ramp (gravity sag allowance)
POS_STEP_ABORT_DEG = 20.0        # single-frame jump = origin renumbering, not motion
FB_STALE_ABORT_S = 0.15
SEND_FAIL_ABORT = 50             # consecutive failed sends = dead bus
MIT_EXPRESSIBLE_DEG = 600.0      # |raw pos| beyond this cannot be a MIT p_des (+-12.56 rad wire)


class Abort(Exception):
    pass


class Watch:
    """Per-run safety state: last status, rolling current, send-failure streak."""

    def __init__(self, args, center_deg):
        self.a = args
        self.center = center_deg
        self.st = {}
        self.cur_hist = []               # (t, |A|) for the 2 s rolling mean
        self.send_fails = 0
        self.t_off = []                  # monotonic - kernel timestamp samples

    def note_send(self, ok):
        self.send_fails = 0 if ok else self.send_fails + 1
        if self.send_fails >= SEND_FAIL_ABORT:
            raise Abort("{} consecutive sends failed -- bus dead or power lost"
                        .format(self.send_fails))

    def drain(self, bus, cid, rows_fb):
        """Pull every pending frame; keep the target drive's status, kernel-timestamped."""
        for _ in range(64):
            msg = bus.recv(timeout=0.0)
            if msg is None:
                return
            if not getattr(msg, "is_extended_id", True):
                continue
            # cid-only filter, like the daemon's: real drives stamp 0x29 in the command byte,
            # the MockBus stamps 0 -- and nothing else on this bus carries our cid
            if (msg.arbitration_id & 0xFF) != cid:
                continue
            st = canio.parse_status(msg.data)
            if st is None:
                continue
            now = time.monotonic()
            tk = getattr(msg, "timestamp", None)      # socketcan kernel stamp: sub-ms truth
            if tk:
                self.t_off.append(now - tk)
            else:
                tk = now
            prev = self.st.get("pos")
            if prev is not None and abs(st["pos"] - prev) > POS_STEP_ABORT_DEG:
                raise Abort("position stepped {:+.1f} deg in one frame -- the drives renumber "
                            "their origin; the record is invalid".format(st["pos"] - prev))
            self.st.update(st, t_rx=now)
            self.cur_hist.append((now, abs(st["cur"])))
            rows_fb.append((tk, st["pos"], st["spd"], st["cur"], st["temp"]))

    def check(self, now, window_rad):
        st = self.st
        if st.get("t_rx") is None or now - st["t_rx"] > FB_STALE_ABORT_S:
            raise Abort("no status frame for {:.0f} ms -- never command a joint we cannot see"
                        .format(1e3 * (now - st.get("t_rx", now - 9))))
        if st["err"]:
            raise Abort("drive error flag {}".format(st["err"]))
        if st["temp"] >= self.a.temp_abort:
            raise Abort("drive temperature {} C >= {} C".format(st["temp"], self.a.temp_abort))
        if abs(st["cur"]) >= self.a.cur_abort:
            raise Abort("|current| {:.1f} A >= {:.1f} A".format(st["cur"], self.a.cur_abort))
        self.cur_hist = [(t, c) for t, c in self.cur_hist if now - t < 2.0]
        if len(self.cur_hist) > 100:
            mean = sum(c for _, c in self.cur_hist) / len(self.cur_hist)
            if mean > self.a.cur_mean_abort:
                raise Abort("mean |current| {:.1f} A over 2 s > {:.1f} A -- this is not an "
                            "unloaded joint".format(mean, self.a.cur_mean_abort))
        lim = math.degrees(window_rad)
        if abs(st["pos"] - self.center) > lim:
            raise Abort("position {:+.1f} deg is outside center {:+.1f} +- {:.1f} deg"
                        .format(st["pos"], self.center, lim))


def release_limp(bus, cid):
    """The abort discipline: stream zero-gain force frames + 0 A. Commands nothing, holds
    nothing -- never command a position after an abort."""
    try:
        limp = mit.limp_payload()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.5:
            canio.force_control(bus, cid, limp)
            canio.set_current(bus, cid, 0.0)
            time.sleep(DT)
        print("released: LIMP (zero-gain force control + 0 A streamed 0.5 s)")
    except Exception as e:                                    # noqa: BLE001 - never mask the exit
        print("WARNING: limp release failed: {!r}".format(e), file=sys.stderr)


def release_hold(bus, cid, center_deg):
    """Clean-exit discipline: hand the drive to servo mode holding center. The MIT frame just
    before this held the same position at full gain, so the handoff is a no-op in torque."""
    try:
        for _ in range(20):
            canio.set_pos(bus, cid, center_deg)
            time.sleep(DT)
        print("released: servo mode holding {:+.1f} deg (robot stays homed)".format(center_deg))
    except Exception as e:                                    # noqa: BLE001
        print("WARNING: servo handoff failed: {!r}".format(e), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", required=True, choices=["can0", "can1"])
    ap.add_argument("--id", type=int, required=True, help="drive node id (cam 105, thigh 106)")
    ap.add_argument("--mode", required=True, choices=["servo", "mit"])
    ap.add_argument("--f0", type=float, default=1.0)
    ap.add_argument("--f1", type=float, default=30.0)
    ap.add_argument("--df", type=float, default=1.0)
    ap.add_argument("--dwell", type=float, default=5.0, help="minimum seconds per frequency")
    ap.add_argument("--vel-amp", type=float, default=1.0,
                    help="velocity amplitude ceiling, rad/s (sets A above the corner)")
    ap.add_argument("--amp-max", type=float, default=0.06,
                    help="position amplitude ceiling, rad (sets A at low f)")
    ap.add_argument("--kp", type=float, default=200.0, help="MIT impedance kp (deployed: 200)")
    ap.add_argument("--kd", type=float, default=5.0, help="MIT impedance kd (deployed: 5)")
    ap.add_argument("--cur-abort", type=float, default=18.0)
    ap.add_argument("--cur-mean-abort", type=float, default=6.0)
    ap.add_argument("--temp-abort", type=float, default=55.0)
    ap.add_argument("--hold-cur-gate", type=float, default=2.5,
                    help="refuse to start if the servo hold draws more than this (loaded joint)")
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--mock", action="store_true", help="canio MockBus, no hardware")
    ap.add_argument("--dry-run", action="store_true", help="print the schedule and exit")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    freqs = np.arange(args.f0, args.f1 + 1e-9, args.df)
    amps = np.minimum(args.amp_max, args.vel_amp / (2 * np.pi * freqs))
    total = GAIN_RAMP_S + AMP_RAMP_S + len(freqs) * (args.dwell + 0.5) + 3.5
    print("{} {} id {}  {:.0f}-{:.0f} Hz step {:g}, dwell >= {:.0f} s  (~{:.0f} s total)"
          .format(args.mode.upper(), args.channel, args.id, args.f0, args.f1, args.df,
                  args.dwell, total))
    print("amplitude {:.2f}-{:.2f} deg (vel ceiling {:.2f} rad/s)".format(
        math.degrees(amps.min()), math.degrees(amps.max()), args.vel_amp))
    if args.mode == "mit":
        print("MIT gains kp={:.0f} kd={:.1f}; v_des=0 tau_ff=0 (structurally); worst-case "
              "proportional torque kp*2A = {:.0f} N*m".format(
                  args.kp, args.kd, args.kp * 2 * args.amp_max))
    if args.dry_run:
        for f, a in zip(freqs, amps):
            print("  {:5.1f} Hz  A {:5.2f} deg  ({:.0f} cycles)".format(
                f, math.degrees(a), math.ceil(args.dwell * f)))
        return 0

    buses = canio.open_buses(args.interface, [args.channel], mock=args.mock)
    bus = buses[args.channel]
    rows_fb, rows_cmd = [], []
    watch = Watch(args, 0.0)
    abort_reason = None
    center_deg = None
    try:
        # ---- listen: prove the drive is talking, healthy, unloaded, and where it is ------------
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            watch.drain(bus, args.id, rows_fb)
            time.sleep(0.002)
        if len(rows_fb) < 50:
            raise Abort("only {} status frames in 2 s of listening -- wrong id, wrong bus, or "
                        "drive not powered".format(len(rows_fb)))
        if watch.st["err"]:
            raise Abort("drive reports error {} before anything was commanded"
                        .format(watch.st["err"]))
        center_deg = float(np.median([r[1] for r in rows_fb]))
        hold_cur = float(np.mean([abs(r[3]) for r in rows_fb]))
        print("listen: {} frames, pos {:+.1f} deg, hold {:.2f} A, {} C".format(
            len(rows_fb), center_deg, hold_cur, watch.st["temp"]))
        if hold_cur > args.hold_cur_gate:
            raise Abort("servo hold is drawing {:.1f} A (> {:.1f}) -- this joint is LOADED; "
                        "the sweep is unloaded-legs-only".format(hold_cur, args.hold_cur_gate))
        if args.mode == "mit" and abs(center_deg) > MIT_EXPRESSIBLE_DEG:
            raise Abort("raw position {:+.0f} deg is outside the +-12.56 rad the MIT position "
                        "field can express -- re-home first".format(center_deg))
        watch.center = center_deg
        center_rad = math.radians(center_deg)
        rows_fb.clear()

        # ---- the sweep ------------------------------------------------------------------------
        phase = 0.0
        fi = 0
        t_start = time.monotonic()
        gain_ramp_end = t_start + (GAIN_RAMP_S if args.mode == "mit" else 0.5)
        amp_ramp_end = gain_ramp_end + AMP_RAMP_S
        f_started = amp_ramp_end                # f0's dwell clock starts at full amplitude
        pend_switch = False
        next_t = time.monotonic()
        while True:
            now = time.monotonic()
            f = float(freqs[fi])
            a_full = float(amps[fi])
            g = min(1.0, (now - t_start) / GAIN_RAMP_S) if args.mode == "mit" else 1.0
            moving = now >= gain_ramp_end
            a = a_full * min(1.0, max(0.0, (now - gain_ramp_end) / AMP_RAMP_S))
            if moving:
                new_phase = phase + 2 * math.pi * f * DT
                if pend_switch and math.floor(new_phase / (2 * math.pi)) > \
                        math.floor(phase / (2 * math.pi)):
                    fi += 1
                    if fi >= len(freqs):
                        phase = 0.0             # end exactly at center
                        break
                    pend_switch = False
                    f_started = now
                phase = new_phase
                if not pend_switch and now - f_started >= args.dwell:
                    pend_switch = True
            delta = a * math.sin(phase)
            if args.mode == "servo":
                ok = canio.set_pos(bus, args.id, center_deg + math.degrees(delta))
            else:
                payload, clamped = mit.pack(center_rad + delta, g * args.kp, g * args.kd)
                if clamped:
                    raise Abort("MIT field clamped: {} -- command left the verified range"
                                .format(clamped))
                ok = canio.force_control(bus, args.id, payload)
            watch.note_send(ok)
            rows_cmd.append((now, f, a, phase, delta, g))
            watch.drain(bus, args.id, rows_fb)
            window = RAMP_WINDOW_RAD if not moving else a + POS_WINDOW_MARGIN_RAD
            watch.check(now, window)
            next_t += DT
            s = next_t - time.monotonic()
            if s > 0:
                time.sleep(s)
            else:
                next_t = time.monotonic()

        # ---- walk home at FULL authority, then hand off to a servo hold ------------------------
        print("sweep done, walking back to center")
        last_delta = float(rows_cmd[-1][4])
        back0, back_s = time.monotonic(), 1.0
        while time.monotonic() - back0 < back_s:
            now = time.monotonic()
            delta = last_delta * max(0.0, 1.0 - (now - back0) / back_s)
            if args.mode == "servo":
                ok = canio.set_pos(bus, args.id, center_deg + math.degrees(delta))
            else:
                payload, _ = mit.pack(center_rad + delta, args.kp, args.kd)
                ok = canio.force_control(bus, args.id, payload)
            watch.note_send(ok)
            watch.drain(bus, args.id, rows_fb)
            watch.check(now, RAMP_WINDOW_RAD)
            time.sleep(DT)
    except Abort as e:
        abort_reason = str(e)
        print("!! ABORT: {}".format(abort_reason), file=sys.stderr)
    except KeyboardInterrupt:
        abort_reason = "operator interrupt"
        print("\n!! interrupted", file=sys.stderr)
    except Exception as e:                                    # noqa: BLE001 - release either way
        abort_reason = "exception: {!r}".format(e)
        import traceback
        traceback.print_exc()
    finally:
        if abort_reason is None and center_deg is not None:
            release_hold(bus, args.id, center_deg)
        else:
            release_limp(bus, args.id)
        try:
            bus.shutdown()
        except Exception:
            pass

    if not rows_cmd:
        print("nothing recorded ({})".format(abort_reason or "?"))
        return 1
    cmd = np.array(rows_cmd, float)
    fb = np.array(rows_fb, float) if rows_fb else np.zeros((0, 5))
    t_off = float(np.median(watch.t_off)) if watch.t_off else 0.0
    meta = {"mode": args.mode, "channel": args.channel, "id": args.id,
            "f0": args.f0, "f1": args.f1, "df": args.df, "dwell": args.dwell,
            "vel_amp": args.vel_amp, "amp_max": args.amp_max,
            "kp": args.kp if args.mode == "mit" else None,
            "kd": args.kd if args.mode == "mit" else None,
            "center_deg": center_deg, "cmd_hz": CMD_HZ,
            "settle_exclude_s": SETTLE_EXCLUDE_S,
            "t_off_mono_minus_kernel": t_off,
            "abort": abort_reason, "legs": "attached, unloaded, robot homed on stand"}
    out = args.out or "bode_{}_{}_{}.npz".format(args.mode, args.channel, args.id)
    np.savez(out,
             t_cmd=cmd[:, 0], f_cmd=cmd[:, 1], amp=cmd[:, 2], phase=cmd[:, 3],
             cmd_rad=cmd[:, 4], gain_ramp=cmd[:, 5],
             t_fb=fb[:, 0], pos_deg=fb[:, 1], spd_erpm=fb[:, 2], cur_a=fb[:, 3],
             temp_c=fb[:, 4], meta_json=json.dumps(meta))
    print("saved {}  ({} cmd ticks, {} fb frames{})".format(
        out, len(cmd), len(fb), ", ABORTED: " + abort_reason if abort_reason else ""))
    return 1 if abort_reason else 0


if __name__ == "__main__":
    sys.exit(main())
