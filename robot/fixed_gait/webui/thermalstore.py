"""Thermal calibration runs: torque-saturating bursts and hand-recorded cooldown curves.

Deliberately JSON, not npz. The measurement here is a handful of numbers per run -- a start
temperature, a peak, when the peak was read -- because the instrument is a person with a
thermometer, not a 200 Hz stream. What the DAEMON captures during the burst (the measured current
trace, and therefore the deposited energy) is reduced to its integral before it is stored: the fit
needs `integral of I^2 dt`, and keeping 6000 samples to represent one number would make a
calibration campaign unreadable.

The drive's OWN reported temperature is recorded alongside the operator's reading at both ends of
every burst. That is not redundancy. Nobody knows what node the AK's internal sensor is bonded to
-- driver board or stator -- and a campaign of bursts with both numbers answers it: if the drive's
reading tracks the external probe with the same amplitude and delay, it is on the case, and the
observer's Luenberger correction is measuring what it thinks it is measuring.

    { "runs":      [ one per burst ],
      "cooldowns": [ one per hand-recorded decay curve ] }
"""
import json
import os
import time
import uuid

import paths
import blackbox

STORE = os.path.join(paths.DATA, "thermal_runs.json")
PROBES = ("case", "winding", "drive")


def _blank():
    return {"runs": [], "cooldowns": []}


def load(path=STORE):
    if not os.path.exists(path):
        return _blank()
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            d = json.load(f)
        d.setdefault("runs", [])
        d.setdefault("cooldowns", [])
        return d
    except (ValueError, OSError) as e:
        print("(thermal store unreadable: {} -- starting empty)".format(e))
        return _blank()


def save(d, path=STORE):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, path)


def add_burst(motor, envelope, summary, drive_t_start, drive_t_peak, ambient_c=None,
              aborted=None, path=STORE):
    """Record a burst the daemon just finished. Operator temperatures are filled in later, by
    `annotate` -- the peak has not happened yet when the burst ends, which is the whole reason
    this is two steps."""
    d = load(path)
    run = {"id": uuid.uuid4().hex[:8],
           "motor": motor,
           "created": time.time(),
           "created_str": time.strftime("%Y-%m-%d %H:%M:%S"),
           "envelope": dict(envelope),
           "summary": dict(summary),
           "drive_t_start_c": drive_t_start,
           "drive_t_peak_c": drive_t_peak,
           "ambient_c": ambient_c,
           "aborted": aborted,
           "probe": "case",
           "t_start_c": None,
           "t_peak_c": None,
           "t_peak_at_s": None,
           "notes": ""}
    d["runs"].append(run)
    save(d, path)
    blackbox.log_event("thermal.burst", **{k: run[k] for k in
                                           ("id", "motor", "aborted", "drive_t_start_c",
                                            "drive_t_peak_c")})
    return run


def annotate(run_id, path=STORE, **fields):
    """Attach the operator's readings to a burst. Only the fields a human supplies are writable --
    the envelope and the measured-current summary are the daemon's record of what happened and
    must not be editable after the fact."""
    allowed = {"t_start_c", "t_peak_c", "t_peak_at_s", "ambient_c", "probe", "notes"}
    bad = set(fields) - allowed
    if bad:
        raise ValueError("not an operator-supplied field: {}".format(", ".join(sorted(bad))))
    if "probe" in fields and fields["probe"] not in PROBES:
        raise ValueError("probe must be one of {}".format(", ".join(PROBES)))
    d = load(path)
    for r in d["runs"]:
        if r["id"] == run_id:
            for k, v in fields.items():
                r[k] = v
            save(d, path)
            blackbox.log_event("thermal.annotate", id=run_id, **fields)
            return r
    raise KeyError("no thermal run {}".format(run_id))


def delete(run_id, path=STORE):
    d = load(path)
    n0 = len(d["runs"]) + len(d["cooldowns"])
    d["runs"] = [r for r in d["runs"] if r["id"] != run_id]
    d["cooldowns"] = [c for c in d["cooldowns"] if c["id"] != run_id]
    if len(d["runs"]) + len(d["cooldowns"]) == n0:
        raise KeyError("no thermal run {}".format(run_id))
    save(d, path)
    return True


# ---------------------------------------------------------------- cooldown curves
def start_cooldown(motor, ambient_c=None, probe="case", after_run=None, path=STORE):
    d = load(path)
    c = {"id": uuid.uuid4().hex[:8], "motor": motor, "created": time.time(),
         "created_str": time.strftime("%Y-%m-%d %H:%M:%S"), "ambient_c": ambient_c,
         "probe": probe, "after_run": after_run, "points": [], "notes": ""}
    d["cooldowns"].append(c)
    save(d, path)
    return c


def add_point(cooldown_id, t_s, temp_c, path=STORE):
    """One (seconds since the curve started, degC) sample.

    Kept sorted, because the operator will inevitably enter one out of order and the fit
    integrates forward in time -- an unsorted series would silently produce a negative dt and a
    frozen sample rather than an error."""
    d = load(path)
    for c in d["cooldowns"]:
        if c["id"] == cooldown_id:
            c["points"].append([float(t_s), float(temp_c)])
            c["points"].sort(key=lambda p: p[0])
            save(d, path)
            return c
    raise KeyError("no cooldown curve {}".format(cooldown_id))


def drop_point(cooldown_id, index, path=STORE):
    d = load(path)
    for c in d["cooldowns"]:
        if c["id"] == cooldown_id:
            if not 0 <= int(index) < len(c["points"]):
                raise IndexError("point {} does not exist".format(index))
            c["points"].pop(int(index))
            save(d, path)
            return c
    raise KeyError("no cooldown curve {}".format(cooldown_id))


# ---------------------------------------------------------------- readiness
def summary(path=STORE):
    """What has been collected, per motor, and whether it is enough to fit.

    The thresholds are the fit's, restated here so the panel can say 'two more bursts' instead of
    making the operator run the fitter to find out. See thermal_fit.adequacy for why a campaign
    that looks fine can still be unfittable."""
    d = load(path)
    by_motor = {}
    for r in d["runs"]:
        m = by_motor.setdefault(r["motor"], {"bursts": 0, "usable": 0, "durations": [],
                                             "cooldowns": 0, "cooldown_points": 0,
                                             "cooldown_span_s": 0.0})
        m["bursts"] += 1
        if r.get("t_start_c") is not None and r.get("t_peak_c") is not None and not r.get("aborted"):
            m["usable"] += 1
            m["durations"].append(round(float(r["envelope"]["duration_s"]), 1))
    for c in d["cooldowns"]:
        m = by_motor.setdefault(c["motor"], {"bursts": 0, "usable": 0, "durations": [],
                                             "cooldowns": 0, "cooldown_points": 0,
                                             "cooldown_span_s": 0.0})
        m["cooldowns"] += 1
        m["cooldown_points"] += len(c["points"])
        if c["points"]:
            m["cooldown_span_s"] = max(m["cooldown_span_s"], float(c["points"][-1][0]))
    for m, v in by_motor.items():
        need = []
        if v["usable"] < 3:
            need.append("{} more annotated burst(s)".format(3 - v["usable"]))
        if len(set(v["durations"])) < 2:
            need.append("bursts at a SECOND duration (one duration cannot test the I^2*t law)")
        if v["cooldown_points"] < 6:
            need.append("{} more cooldown point(s)".format(6 - v["cooldown_points"]))
        if v["cooldown_span_s"] < 900:
            need.append("a cooldown followed for longer (>= 15 min pins the case time constant)")
        v["needs"] = need
        v["ready"] = not need
    return by_motor


def all_runs(path=STORE):
    d = load(path)
    return d["runs"], d["cooldowns"]
