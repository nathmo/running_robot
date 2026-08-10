#!/usr/bin/env python3
"""Read a DASH-01 black box offline — and, with --postmortem, tell you what happened.

    python blackbox_read.py --postmortem                 # the five questions, answered
    python blackbox_read.py --list                       # what is on disk
    python blackbox_read.py --timeline 80                # the event log, newest last
    python blackbox_read.py --dump B_xxx.bbdump --csv out.csv     # 200 Hz data out
    python blackbox_read.py --dir ./downloaded_blackbox --postmortem

The five questions this must answer from disk ALONE (BLACKBOX_TASK.md):
  1. when did the Pi last boot, and when did it last lose power?
  2. what was pos_raw for all six at the last zero capture — does it match pos_raw now?
  3. at a trip: commanded vs reported position per motor, 200 Hz, >= 10 s BEFORE the trigger
  4. which calibration / drive PID gains / dynamics config were live at that instant
  5. did pos_raw ever move while the robot was LIMP and untouched?

On time: t_mono (time.monotonic) is the only trustworthy clock. t_wall is whatever the Pi believed,
which with no RTC and no internet can be years wrong — every record carries wall_trusted, and this
reader says so rather than quietly printing a confident wrong date.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths                                     # noqa: E402  (sys.path side effect)
import blackbox                                  # noqa: E402

BAR = "=" * 78


# ===================================================================== loading
def load_dir(d):
    """(events, [(name, header, path)]) — headers only, so this is cheap on a big directory."""
    events, files = [], []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        if name.endswith(".jsonl"):
            events.extend(blackbox.read_events(p))
        elif name.endswith((blackbox.SEG_EXT, blackbox.DUMP_EXT)):
            try:
                files.append((name, blackbox.read_header(p), p))
            except (OSError, ValueError):
                files.append((name, {"_bad": True}, p))
    events.sort(key=lambda e: (e.get("session_id", ""), e.get("t_mono", 0.0)))
    return events, files


def to_frame(rec):
    """Structured records -> a flat pandas DataFrame (one column per motor per field), or a dict of
    arrays if pandas is not installed. Column order is paths.MOTOR_NAMES — right leg first."""
    cols = {}
    for f in ("t_mono", "t_wall", "dt", "mode", "estop", "slip", "drop"):
        cols[f] = rec[f]
    for f in list(blackbox.MOTOR_FIELDS) + ["err"]:
        for i, n in enumerate(paths.MOTOR_NAMES):
            cols[f"{n}.{f}"] = rec[f][:, i]
    try:
        import pandas as pd
        return pd.DataFrame(cols)
    except ImportError:
        return cols


def write_csv(rec, path):
    cols = to_frame(rec)
    try:
        cols.to_csv(path, index=False)
        return
    except AttributeError:
        pass
    names = list(cols)
    arr = np.column_stack([np.asarray(cols[k], float) for k in names])
    np.savetxt(path, arr, delimiter=",", header=",".join(names), comments="")


# ===================================================================== time helpers
def _fmt_wall(t, trusted=True):
    if t is None:
        return "?"
    s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(t)))
    return s if trusted else s + " (UNTRUSTED CLOCK)"


def _dur(s):
    if s is None:
        return "?"
    s = float(s)
    if s < 90:
        return f"{s:.1f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"


def sessions(events):
    """One entry per daemon run, in the order they started, with power-loss inference.

    THE INFERENCE, stated explicitly: the process that dies with the power cannot log its own
    death. So a session whose event stream simply STOPS — no daemon.stop, no server.stop — ended
    when its LAST HEARTBEAT was written, to within one heartbeat interval. That is the power-off
    time, and it is the best resolution that exists.
    """
    out = {}
    for e in events:
        sid = e.get("session_id", "?")
        s = out.setdefault(sid, {"session_id": sid, "boot_id": e.get("boot_id"),
                                 "first": e, "last": e, "heartbeats": [], "events": [],
                                 "clean_stop": False, "start": None})
        s["last"] = e
        s["events"].append(e)
        k = e.get("kind")
        if k == "heartbeat":
            s["heartbeats"].append(e)
        elif k == "daemon.start":
            s["start"] = e
        elif k in ("daemon.stop", "server.stop"):
            s["clean_stop"] = True
    for s in out.values():
        hb = s["heartbeats"]
        s["hb_interval"] = (np.median(np.diff([h["t_mono"] for h in hb]))
                            if len(hb) > 2 else blackbox.HEARTBEAT_S)
        s["last_hb"] = hb[-1] if hb else None
        # gaps INSIDE a session: the writer was alive on both sides, so something stalled it
        s["gaps"] = []
        for a, b in zip(hb, hb[1:]):
            d = b["t_mono"] - a["t_mono"]
            if d > 3 * s["hb_interval"]:
                s["gaps"].append({"from": a, "to": b, "seconds": d})
    return list(out.values())


# ===================================================================== reports
def cmd_list(d, files, events):
    print(f"{BAR}\n{len(files)} recording(s) + {len(events)} event(s) in {d}\n{BAR}")
    for name, h, p in sorted(files, key=lambda f: os.path.getmtime(f[2]), reverse=True):
        size = os.path.getsize(p)
        if h.get("_bad"):
            print(f"  {name:<52} {size / 1e6:7.2f} MB   UNREADABLE HEADER")
            continue
        n = h.get("n_samples")
        if n is None:
            n = max(0, (size - h["_data_offset"]) // blackbox.RECORD_BYTES)
        extra = ""
        if h.get("tier") == "B":
            t = h.get("trigger", {})
            extra = f"  trigger={t.get('reason')}  pre={h.get('pre_trigger_s')}s"
        print(f"  {name:<52} {size / 1e6:7.2f} MB  tier {h.get('tier')}  "
              f"{n:>7} samples @ {h.get('rate_hz')} Hz{extra}")


def cmd_timeline(events, n, kinds=None):
    sel = [e for e in events if not kinds or e.get("kind") in kinds]
    print(f"{BAR}\nEVENT TIMELINE — last {min(n, len(sel))} of {len(sel)}\n{BAR}")
    for e in sel[-n:]:
        w = _fmt_wall(e.get("t_wall"), e.get("wall_trusted", False))
        head = f"{w}  +{e.get('uptime_s', 0):9.2f}s  {e.get('kind','?'):<22}"
        rest = {k: v for k, v in e.items()
                if k not in ("kind", "t_mono", "t_wall", "uptime_s", "boot_id", "session_id",
                             "wall_trusted", "note", "config")}
        s = json.dumps(rest, default=str)
        print(head + (s if len(s) < 150 else s[:147] + "..."))
        if e.get("note"):
            print(" " * 22 + f"  -> {e['note']}")


def _latest_samples(files):
    """(name, header, records) of the newest file that actually holds data."""
    for name, h, p in sorted(files, key=lambda f: os.path.getmtime(f[2]), reverse=True):
        if h.get("_bad"):
            continue
        try:
            hh, rec = blackbox.read_segment(p)
        except (OSError, ValueError):
            continue
        if len(rec):
            return name, hh, rec
    return None, None, None


def _pick_incident(events, files):
    """The dump that most deserves attention: a trip, else a refused pre-move, else the newest."""
    order = ("trip", "premove_guard_refused", "raw_origin_jump")
    dumps = [(n, h, p) for n, h, p in files
             if h.get("tier") == "B" and not h.get("_bad")]
    if not dumps:
        return None
    for want in order:
        hits = [f for f in dumps if (f[1].get("trigger") or {}).get("reason") == want
                or want in ((f[1].get("trigger") or {}).get("also") or [])]
        if hits:
            return max(hits, key=lambda f: f[1].get("trigger", {}).get("t_trig_wall", 0))
    return max(dumps, key=lambda f: os.path.getmtime(f[2]))


def postmortem(d, events, files):
    print(f"{BAR}\nDASH-01 BLACK BOX — POSTMORTEM\n{d}\n{BAR}")
    ss = sessions(events)
    if not ss:
        print("\nNo events on disk. Either the recorder never ran, or this is the wrong "
              "directory (expected events.jsonl next to *.bbseg / *.bbdump).")
        return 1

    # ---------------------------------------------------------------- 1. boot / power history
    print("\n1. WHEN DID THE PI BOOT, AND WHEN DID IT LAST LOSE POWER?")
    print("   " + "-" * 72)
    boots = {}
    for s in ss:
        boots.setdefault(s["boot_id"], []).append(s)
    for bid, group in boots.items():
        st = group[0].get("start") or {}
        trusted = st.get("wall_trusted", False)
        boot_wall = st.get("boot_t_wall")
        print(f"   Pi boot {bid}")
        print(f"     booted   {_fmt_wall(boot_wall, trusted)}"
              f"   (= t_wall - /proc/uptime at daemon start)")
        for s in group:
            first, last = s["first"], s["last"]
            print(f"     session {s['session_id'][:8]}  "
                  f"start {_fmt_wall(first.get('t_wall'), trusted)}  "
                  f"ran {_dur(last.get('uptime_s'))}")
            for g in s["gaps"]:
                print(f"       !! RECORDING GAP of {_dur(g['seconds'])} inside the session "
                      f"(heartbeats {g['from'].get('uptime_s'):.0f}s -> "
                      f"{g['to'].get('uptime_s'):.0f}s) — the writer was stalled, not dead")
            if s["clean_stop"]:
                print("       ended: CLEAN SHUTDOWN (a daemon.stop/server.stop was written)")
            elif s["last_hb"]:
                hb = s["last_hb"]
                print(f"       ended: NO CLEAN STOP RECORD -> POWER LOSS / KILL INFERRED at "
                      f"{_fmt_wall(hb.get('t_wall'), hb.get('wall_trusted', False))} "
                      f"+/- {s['hb_interval']:.0f}s")
                print("              (a process that dies with the power cannot log its own "
                      "death; the last heartbeat before the gap IS the power-off time, to "
                      "heartbeat resolution)")
            else:
                print("       ended: no heartbeat was ever written — session too short to date")
    untrusted = [s for s in ss if not (s.get("start") or {}).get("wall_trusted", False)]
    if untrusted:
        print(f"\n   NOTE: {len(untrusted)}/{len(ss)} session(s) ran with an UNSYNCED clock (no "
              "RTC, no NTP). Their wall-clock stamps are not evidence; t_mono is.")
    steps = [e for e in events if e.get("kind") == "clock.step"]
    for e in steps:
        print(f"   CLOCK STEPPED by {e.get('step_s', 0):+.1f} s at +{e.get('uptime_s', 0):.0f}s "
              f"— every earlier t_wall in that session is off by about {-e.get('step_s', 0):+.1f} s")

    # ---------------------------------------------------------------- 2. raw at rest
    print("\n2. pos_raw AT THE LAST ZERO CAPTURE vs pos_raw NOW")
    print("   " + "-" * 72)
    zeros = [e for e in events if e.get("kind") == "calib.set_zero"]
    name, hdr, rec = _latest_samples(files)
    if not zeros:
        print("   No zero capture is on record in this event log.")
    else:
        z = zeros[-1]
        print(f"   last capture: {_fmt_wall(z.get('t_wall'), z.get('wall_trusted', False))}  "
              f"session {z.get('session_id', '?')[:8]}  zero_epoch={z.get('zero_epoch')}")
        captured = z.get("raw_captured") or {}
        latest = {}
        if rec is not None:
            latest = {n_: float(rec["pos_raw"][-1, i]) for i, n_ in enumerate(paths.MOTOR_NAMES)}
            print(f"   latest pos_raw from {name} at "
                  f"{_fmt_wall(rec['t_wall'][-1], hdr.get('wall_trusted', False))}")
        print(f"     {'motor':<14}{'at zero':>12}{'latest':>12}{'delta':>12}")
        worst = 0.0
        for n_ in paths.MOTOR_NAMES:
            a, b = captured.get(n_), latest.get(n_)
            if a is None:
                continue
            if b is None:
                print(f"     {n_:<14}{a:>12.2f}{'?':>12}{'?':>12}")
                continue
            worst = max(worst, abs(b - a))
            print(f"     {n_:<14}{a:>12.2f}{b:>12.2f}{b - a:>+12.2f}")
        same_session = rec is not None and hdr.get("session_id") == z.get("session_id")
        if worst > 10.0:
            print(f"   VERDICT: the largest joint differs by {worst:.1f} deg from the zero "
                  f"capture. If the robot was not moved, this calibration is STALE and any "
                  f"absolute command is a full-authority slew to the wrong place.")
            if not same_session:
                print("            (the samples are from a LATER session than the capture, so a "
                      "power cycle sits in between — the drives re-randomise their raw origin "
                      "there, which is the expected cause)")
        else:
            print(f"   VERDICT: agreement to {worst:.1f} deg — consistent with the robot simply "
                  f"having moved/sagged since, no evidence of an origin shift here (see 5).")

    # ---------------------------------------------------------------- 3. the incident
    print("\n3. AT THE MOMENT OF THE TRIP: COMMANDED vs REPORTED, 200 Hz")
    print("   " + "-" * 72)
    got = _pick_incident(events, files)
    inc_hdr = None
    if got is None:
        print("   No Tier B dump on disk — nothing ever triggered one.")
    else:
        iname, inc_hdr, ipath = got
        try:
            inc_hdr, irec = blackbox.read_segment(ipath)
        except (OSError, ValueError) as e:
            print(f"   {iname}: unreadable ({e})")
            irec = None
        trig = inc_hdr.get("trigger", {}) if inc_hdr else {}
        print(f"   {iname}")
        print(f"     trigger  : {trig.get('reason')}  {trig.get('also') or ''}")
        if trig.get("why") or trig.get("message"):
            print(f"     detail   : {trig.get('why') or trig.get('message')}")
        print(f"     at       : {_fmt_wall(trig.get('t_trig_wall'), inc_hdr.get('wall_trusted'))}")
        if irec is not None and len(irec):
            t = irec["t_mono"]
            tt = trig.get("t_trig_mono", t[-1])
            pre = t <= tt
            print(f"     samples  : {len(irec)} at {inc_hdr.get('rate_hz')} Hz — "
                  f"{int(pre.sum())} BEFORE the trigger "
                  f"({tt - t[0]:.1f} s of pre-trigger history), "
                  f"{int((~pre).sum())} after ({t[-1] - tt:.1f} s)")
            if tt - t[0] < 10.0:
                print("     !! less than the 10 s of pre-trigger history the design requires — "
                      "the recorder had not been running long enough")
            gaps = np.diff(irec["drop"].astype(np.int64))
            if gaps.any():
                print(f"     !! {int(gaps.sum())} sample(s) were DROPPED inside this window "
                      f"(queue full) — the data has holes and says so")
            k = int(np.argmin(np.abs(t - tt)))
            print(f"     {'motor':<14}{'cmd_raw':>11}{'pos_raw':>11}{'err':>9}"
                  f"{'|err| max pre':>15}{'pos_norm':>11}")
            for i, n_ in enumerate(paths.MOTOR_NAMES):
                c, p = irec["cmd_raw"][k, i], irec["pos_raw"][k, i]
                e_pre = np.abs(irec["pos_raw"][pre, i] - irec["cmd_raw"][pre, i])
                e_pre = e_pre[np.isfinite(e_pre)]
                print(f"     {n_:<14}{c:>11.2f}{p:>11.2f}{p - c:>+9.2f}"
                      f"{(e_pre.max() if len(e_pre) else float('nan')):>15.2f}"
                      f"{irec['pos_norm'][k, i]:>11.2f}")
            mode_names = inc_hdr.get("mode_names") or blackbox.mode_names()
            mi = int(irec["mode"][k])
            print(f"     mode at the trigger: "
                  f"{mode_names[mi] if mi < len(mode_names) else mi}, "
                  f"loop slip {int(irec['slip'][k])}, "
                  f"tick period {np.median(irec['dt'][pre]) * 1e3:.2f} ms median")

    # ---------------------------------------------------------------- 4. config provenance
    print("\n4. WHICH CALIBRATION / DRIVE GAINS / DYNAMICS WERE LIVE AT THAT INSTANT")
    print("   " + "-" * 72)
    cfg = (inc_hdr or {}).get("config") or {}
    if not cfg:
        snaps = [e for e in events if e.get("kind") == "config.snapshot"]
        cfg = (snaps[-1].get("config") if snaps else {}) or {}
        if cfg:
            print("   (no dump to read it from — showing the last config.snapshot event instead)")
    if not cfg:
        print("   No config snapshot recorded.")
    else:
        print(f"   config_hash: {(inc_hdr or {}).get('config_hash', '?')}")
        cal = cfg.get("calibration") or {}
        print(f"   calibration: stage={cal.get('stage')} created={cal.get('created')} "
              f"zero_epoch={cal.get('zero_epoch')} "
              f"restored_from_disk={cal.get('restored_from_disk')}")
        print(f"     {'motor':<14}{'offset':>11}{'sign':>6}{'confirmed':>11}")
        for n_, m in (cal.get("motors") or {}).items():
            print(f"     {n_:<14}{m.get('offset_deg', 0):>11.2f}{m.get('sign', 0):>6}"
                  f"{str(m.get('confirmed')):>11}")
        dyn = cfg.get("dynamics") or {}
        print(f"   dynamics: updated={dyn.get('updated')}  "
              f"masses={len(dyn.get('masses') or {})} bodies  kt={dyn.get('kt')}")
        dg = dyn.get("drive_gains") or {}
        if dg:
            print("   drive gains (board config — RECORDED, never pushed):")
            for n_ in paths.MOTOR_NAMES:
                g = dg.get(n_, {})
                pos, cur_, spd_ = (g.get("position", {}), g.get("current", {}), g.get("speed", {}))
                print(f"     {n_:<14} pos kp={pos.get('kp')} ki={pos.get('ki')} kd={pos.get('kd')}"
                      f" | spd kp={spd_.get('kp')} ki={spd_.get('ki')}"
                      f" | cur kp={cur_.get('kp')} ki={cur_.get('ki')}")

    # ---------------------------------------------------------------- 5. origin jumps
    print("\n5. DID pos_raw EVER MOVE WHILE LIMP AND UNTOUCHED?")
    print("   " + "-" * 72)
    jumps = [e for e in events if e.get("kind") == "raw.jump"]
    limp_s = 0.0
    for _n, h, p in files:
        if h.get("_bad") or h.get("tier") != "A":
            continue
        try:
            _hh, r = blackbox.read_segment(p)
        except (OSError, ValueError):
            continue
        names = h.get("mode_names") or blackbox.mode_names()
        passive = {i for i, m in enumerate(names) if m in ("LIMP", "ESTOPPED")}
        if len(r):
            limp_s += float(np.isin(r["mode"], list(passive)).sum()) / max(1.0, h.get("rate_hz", 20))
    if not jumps:
        print(f"   No origin jump was ever observed, across {_dur(limp_s)} of recorded LIMP time.")
        print("   (detector: pos_raw stepping >4 deg / >800 deg/s while the mode commands nothing "
              "AND the drive reports |spd| < 200 ERPM — motion by hand shows up as speed, an "
              "origin rewrite does not.)")
    else:
        print(f"   YES — {len(jumps)} origin jump(s), over {_dur(limp_s)} of recorded LIMP time:")
        for e in jumps[-12:]:
            print(f"     {_fmt_wall(e.get('t_wall'), e.get('wall_trusted', False))}  "
                  f"{e.get('motor')}: {e.get('before'):.1f} -> {e.get('after'):.1f} "
                  f"({e.get('delta_deg'):+.1f} deg) while {e.get('mode')}, "
                  f"spd={e.get('spd')} ERPM")
        print("   This is a driver board rewriting its own multi-turn origin. Every absolute "
              "command issued after it went to the wrong place.")

    # ---------------------------------------------------------------- what happened
    print(f"\n{BAR}\nWHAT HAPPENED\n{BAR}")
    for line in _narrative(events, inc_hdr, ss):
        print(line)
    print()
    return 0


def _narrative(events, inc_hdr, ss):
    """The plain-language conclusion, so nobody has to read raw files to get the story."""
    out = []
    trig = (inc_hdr or {}).get("trigger", {}) or {}
    refused = [e for e in events if e.get("kind") == "premove.refused"]
    trips = [e for e in events if e.get("kind") == "trip"]
    jumps = [e for e in events if e.get("kind") == "raw.jump"]

    if refused:
        r = refused[-1]
        out.append(f"  The pre-move guard REFUSED to leave LIMP: {r.get('reason')}")
        cmp_ = r.get("compare") or {}
        if cmp_:
            worst = max(cmp_, key=lambda n: abs(cmp_[n].get("delta", 0)))
            out.append(f"  Worst joint {worst}: pos_raw was {cmp_[worst]['then']:.1f} at the last "
                       f"zero capture and {cmp_[worst]['now']:.1f} when the move was commanded "
                       f"({cmp_[worst]['delta']:+.1f} deg).")
        out.append("  No absolute command was issued. The robot did not move.")
    if jumps:
        j = jumps[-1]
        out.append(f"  A driver board moved its own encoder origin: {j.get('motor')} stepped "
                   f"{j.get('delta_deg'):+.1f} deg while {j.get('mode')} with the drive reporting "
                   f"spd={j.get('spd')} — i.e. the joint did not move, the numbering did.")
    if trips:
        t = trips[-1]
        out.append(f"  Safety trip: {t.get('reason')} (mode {t.get('mode')}).")
    if trig:
        out.append(f"  The 200 Hz window around it is preserved in the dump above "
                   f"(trigger: {trig.get('reason')}).")
    if not (refused or trips or jumps):
        out.append("  Nothing dangerous is on record: no trip, no refused move, no origin jump.")
        out.append("  The recorder has been running and the timeline is continuous apart from any "
                   "gaps listed in section 1.")
    unclean = [s for s in ss if not s["clean_stop"] and s["last_hb"]]
    if unclean:
        out.append(f"  {len(unclean)} session(s) ended without a clean stop — see section 1 for "
                   f"the inferred power-off times.")
    return out


# ===================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=paths.BLACKBOX_DIR,
                    help="black box directory (default %(default)s)")
    ap.add_argument("--postmortem", action="store_true",
                    help="answer the five questions from disk alone")
    ap.add_argument("--list", action="store_true", help="list recordings")
    ap.add_argument("--timeline", nargs="?", type=int, const=60, default=None,
                    help="print the last N events (default 60)")
    ap.add_argument("--kind", action="append", help="filter --timeline to these event kinds")
    ap.add_argument("--dump", help="read one segment/dump and summarise it")
    ap.add_argument("--csv", help="with --dump: write the 200 Hz records to this CSV")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dir):
        print(f"no such directory: {args.dir}")
        return 2
    events, files = load_dir(args.dir)

    did = False
    if args.list:
        cmd_list(args.dir, files, events)
        did = True
    if args.timeline is not None:
        cmd_timeline(events, args.timeline, args.kind)
        did = True
    if args.dump:
        p = args.dump if os.path.isfile(args.dump) else os.path.join(args.dir, args.dump)
        h, rec = blackbox.read_segment(p)
        print(json.dumps({k: v for k, v in h.items() if k != "config"}, indent=2, default=str))
        print(f"{len(rec)} records, {h.get('_torn_bytes', 0)} torn trailing bytes "
              f"(a file killed mid-write is still readable — that is the point of the format)")
        if args.csv:
            write_csv(rec, args.csv)
            print(f"wrote {args.csv}")
        did = True
    if args.postmortem or not did:
        return postmortem(args.dir, events, files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
