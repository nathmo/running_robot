"""Flight recorder ("black box") for DASH-01 — always-on, on-disk, survives the process.

WHY: on 2026-08-10 a joint destroyed itself and the investigation was impossible. The only
telemetry that existed was ringbuffer.TelemetryRing — 4096 samples of RAM at 20 Hz (3.4 min), of
which the HTTP API can only ever hand out the last 512 (25.6 s), and nothing on disk. By the time
anyone looked the window had rolled past the event.

THREE TIERS, because 200 Hz x 6 motors continuously is 35 kB/s = 3 GB/day and that kills the SD
card:

  A  continuous, 20 Hz, every field, fixed-width binary segments, rotated, budgeted.  ~4 kB/s
  B  full 200 Hz, kept in a RAM ring (RING_SECONDS), dumped to its own file on a trigger —
     PRE-trigger history included. This is the tier that would have solved 2026-08-10.
  C  events.jsonl, append-only, fsync'd, one line per event, never deleted without archiving.

THE CONTROL LOOP IS SACRED. The daemon thread only ever appends a tuple to a bounded deque
(collections.deque append/len are O(1) and atomic under the GIL — no lock is taken, so the 200 Hz
tick can never block on I/O). A single writer thread owns every file, the Tier-B ring, the
decimation to Tier A, rotation and the disk budget. If the queue is full we DROP AND COUNT, and the
count is stamped into the next accepted record so a reader can see exactly where the holes are — a
black box that lies about its own gaps is worse than none.

THE ROBOT DOES NOT DEPEND ON THIS. Every public method is non-blocking and swallows its own errors;
if the writer thread dies, push_sample()/log_event() degrade to counters, status()["alive"] goes
False and the UI shows it. Nothing here can stop the robot from running.

TIME. The Pi has no RTC and serves an AP with no guaranteed internet, so time.time() may be years
wrong until (or unless) NTP syncs. Every record therefore carries t_mono (time.monotonic(), the
primary base), t_wall, and — in Tier C — wall_trusted plus the ids below. When the wall clock steps
(NTP finally syncing) a `clock.step` event records the jump so earlier records can be
retro-corrected offline.

IDENTITY. Two ids, and they answer different questions:
  boot_id     /proc/sys/kernel/random/boot_id — changes only when the PI reboots. Together with
              the uptime captured at start it gives the wall time of the boot itself.
  session_id  random UUID per daemon start. Records from two sessions can never be silently
              interleaved even if the wall clock repeats.

FORMAT. Segments are: a magic line, a one-line JSON header (naming every column, the numpy dtype,
and paths.MOTOR_NAMES order explicitly), then fixed-width records appended forever. Files are only
ever appended to and never rewritten in place, so a file is readable even if the process was killed
mid-write — the reader floors the tail to a whole number of records. See blackbox_read.py.
"""
import collections
import errno
import hashlib
import json
import os
import shutil
import threading
import time
import traceback
import uuid

import numpy as np

import paths

VERSION = 1
MAGIC = b"DASHBB01"
SEG_EXT = ".bbseg"                       # Tier A segment
DUMP_EXT = ".bbdump"                     # Tier B triggered dump
EVENTS_NAME = "events.jsonl"             # Tier C

N = paths.N_MOTORS

# ---------------------------------------------------------------- record layout
# Per-motor fields are float32 (NaN = that motor is silent); err is the drive's raw error byte.
# Scalars carry the loop's own health: `dt` is the ACTUAL period of the tick that produced the
# sample (not 1/TICK_HZ), `slip` the daemon's running late-tick count, `drop` the black box's own
# running dropped-sample count at the moment this record was accepted.
MOTOR_FIELDS = ("pos_raw", "pos_norm", "cmd_raw", "cmd_norm", "spd", "cur", "temp")
SCALAR_FIELDS = ("t_mono", "t_wall", "dt", "mode", "estop", "slip", "drop")
RECORD_DTYPE = np.dtype(
    [("t_mono", "<f8"), ("t_wall", "<f8"), ("dt", "<f4"),
     ("mode", "u1"), ("estop", "u1"), ("_pad", "<u2"),
     ("slip", "<u4"), ("drop", "<u4")]
    + [(f, "<f4", (N,)) for f in MOTOR_FIELDS]
    + [("err", "<u2", (N,))])
RECORD_BYTES = RECORD_DTYPE.itemsize                      # 212 B

# What the daemon hands to push_sample(): one flat tuple, no dicts, no numpy, no allocation beyond
# the tuple itself. Field-major so the writer can slice it straight into RECORD_DTYPE columns.
ROW_HEAD = ("t_mono", "t_wall", "dt", "mode", "estop", "slip")
ROW_BLOCKS = MOTOR_FIELDS + ("err",)                      # each block is N values
ROW_LEN = len(ROW_HEAD) + len(ROW_BLOCKS) * N             # 6 + 8*6 = 54

# ---------------------------------------------------------------- tuning
TIER_A_DIV = 10                  # keep 1 sample in 10 => 20 Hz from a 200 Hz push
RING_SECONDS = 40.0              # Tier B RAM ring (brief demands >= 30 s)
RING_HZ = 200.0
POST_TRIGGER_S = 3.0             # keep recording this long after a trigger, then close the file
DUMP_COOLDOWN_S = 10.0           # a storm of triggers must not write a storm of 1.7 MB files
# ...and routine triggers (a mode toggle) get a much longer one. One dump is ~1.7 MB, Tier A
# history is what gets deleted to make room, and an operator toggling LIMP/MANUAL for a minute
# must not cost the last day of continuous recording.
ROUTINE_COOLDOWN_S = 120.0
HEARTBEAT_S = 8.0                # power-off is inferred from the last heartbeat before a gap
WRITER_PERIOD_S = 0.10
SAMPLE_QUEUE_MAX = 4000          # 20 s of 200 Hz backlog before we start dropping
EVENT_QUEUE_MAX = 4096
SEGMENT_MAX_BYTES = 8 << 20      # rotate Tier A here (~30 min of 20 Hz)
EVENTS_MAX_BYTES = 32 << 20      # archive (never delete) the event log here
BUDGET_BYTES = 1_200_000_000     # everything under blackbox/ must fit in this
MIN_FREE_BYTES = 512 << 20       # ...and leave at least this much free on the filesystem
SPACE_CHECK_S = 5.0

# (The "did the encoder origin move on its own" watchdog lives in daemon._watch_continuity, not
# here: the pre-move guard depends on its answer, and nothing the robot depends on may depend on
# the recorder being alive. It reports its findings back through log_event/trigger_dump.)

# Mode names are the daemon's; it registers them at import so the segment header is self-describing
# and the reader never has to guess what mode code 3 meant.
_MODE_NAMES = ["LIMP", "MANUAL", "RECORD_GAIT", "RECORD_WS", "PLAYBACK", "MEASURE", "ESTOPPED"]


def register_modes(names):
    """Called by daemon.py at import so mode codes in the binary match daemon.MODES exactly."""
    global _MODE_NAMES
    _MODE_NAMES = list(names)


def mode_names():
    return list(_MODE_NAMES)


# ===================================================================== small helpers
def _slug(s, n=40):
    keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(s))
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_")[:n] or "x"


def _kernel_boot_id():
    """The kernel's boot id: constant for the life of a boot, new after every reboot/power cycle.
    Absent off Linux (dev laptops), where we fall back to a per-process UUID."""
    try:
        with open("/proc/sys/kernel/random/boot_id", "r") as f:
            return f.read().strip()
    except OSError:
        return "no-proc-" + uuid.uuid4().hex[:12]


def _uptime_s():
    try:
        with open("/proc/uptime", "r") as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _wall_trusted():
    """Is time.time() worth believing? Only if something authoritative set it.

    systemd-timesyncd touches /run/systemd/timesync/synchronized once NTP has actually landed. With
    no RTC and no internet the Pi restores a plausible-looking but WRONG time from fake-hwclock at
    boot, so "the year looks sane" is not evidence — we deliberately do not accept it. Untrusted
    does not mean unusable: t_mono still orders everything within a session, and a later clock.step
    event lets an offline reader retro-correct the wall stamps.
    """
    try:
        return os.path.exists("/run/systemd/timesync/synchronized")
    except OSError:
        return False


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _jsonable(v):
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if v != v else round(v, 6)          # NaN -> null, JSON has no NaN
    if isinstance(v, (np.integer, int, bool)) or v is None or isinstance(v, str):
        return v.item() if isinstance(v, np.integer) else v
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)


# ===================================================================== the recorder
class BlackBox:
    """One instance per daemon run. Construct, install(), start(); stop() at shutdown."""

    def __init__(self, directory=None, budget_bytes=BUDGET_BYTES, min_free_bytes=MIN_FREE_BYTES,
                 ring_seconds=RING_SECONDS, heartbeat_s=HEARTBEAT_S,
                 post_trigger_s=POST_TRIGGER_S, segment_max_bytes=SEGMENT_MAX_BYTES,
                 sample_queue_max=SAMPLE_QUEUE_MAX, tier_a=True,
                 space_check_s=SPACE_CHECK_S):
        self.dir = directory or paths.BLACKBOX_DIR
        self.budget_bytes = int(budget_bytes)
        self.min_free_bytes = int(min_free_bytes)
        self.post_trigger_s = float(post_trigger_s)
        self.segment_max_bytes = int(segment_max_bytes)
        self.heartbeat_s = float(heartbeat_s)
        self.sample_queue_max = int(sample_queue_max)
        self.tier_a_enabled = bool(tier_a)
        self.space_check_s = float(space_check_s)

        self.boot_id = _kernel_boot_id()
        self.session_id = uuid.uuid4().hex
        self.short = self.session_id[:8]
        self.t0_mono = time.monotonic()
        self.t0_wall = time.time()
        self.uptime_at_start = _uptime_s()

        # queues: the ONLY thing the 200 Hz thread touches
        self._sq = collections.deque()
        self._eq = collections.deque()
        self._tq = collections.deque()
        self._dropped = 0                 # samples dropped (queue full)
        self._dropped_events = 0

        # writer-thread-owned state (no locks needed: exactly one thread touches these)
        self._ring = np.zeros(max(64, int(ring_seconds * RING_HZ)), RECORD_DTYPE)
        self._ring_seq = 0
        self._decim = 0
        self._seg_f = None
        self._seg_name = None
        self._seg_bytes = 0
        self._seg_idx = 0
        self._ev_f = None
        self._ev_bytes = 0
        self._pending = None              # dump being filled with post-trigger samples
        self._dump_idx = 0
        self._last_dump_end = {}          # reason -> monotonic time its dump closed
        self._last_hb = 0.0
        self._last_space = 0.0
        self._wall_offset0 = self.t0_wall - self.t0_mono
        self._degraded = False            # Tier A stopped for space
        self._n_written = 0
        self._n_seen = 0
        self._bytes_used = 0              # refreshed by _enforce_space, not by every status publish

        # published status (rebuilt wholesale by the writer, read by HTTP threads — dict swap is
        # atomic under the GIL, so readers never see a half-updated dict and never take a lock)
        self._status = {"alive": False, "error": None}
        self._config_fn = None

        self.error = None
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------ public, non-blocking
    def push_sample(self, row):
        """Append one full-rate sample. Called from the 200 Hz control thread ONLY.

        `row` is a flat tuple of ROW_LEN floats laid out as ROW_HEAD then one N-block per
        ROW_BLOCKS entry (see make_row_head/ROW_* above). Never blocks, never raises, never takes a
        lock. Returns False if the sample was dropped because the queue is full.
        """
        if self._stop.is_set() or self.error is not None:
            return False
        if len(self._sq) >= self.sample_queue_max:
            self._dropped += 1                    # drop and COUNT — the gap must be visible
            return False
        self._sq.append((self._dropped, row))
        return True

    def log_event(self, kind, /, **fields):
        """Queue one Tier C event. Safe from any thread, including the control loop."""
        if len(self._eq) >= EVENT_QUEUE_MAX:
            self._dropped_events += 1
            return False
        t_mono = time.monotonic()
        # `kind`/`t_mono`/... win over anything in **fields: the schema a reader relies on must not
        # be overwritable by a caller that happened to pass a field of the same name.
        self._eq.append({**{k: _jsonable(v) for k, v in fields.items()},
                         "kind": str(kind), "t_mono": round(t_mono, 6), "t_wall": time.time(),
                         "uptime_s": round(t_mono - self.t0_mono, 3),
                         "boot_id": self.boot_id, "session_id": self.session_id,
                         "wall_trusted": _wall_trusted()})
        return True

    def trigger_dump(self, reason, /, cooldown_s=None, **fields):
        """Ask for a Tier B dump of the whole 200 Hz ring (pre-trigger history included).

        Returns the filename it will be written to, or None if it was coalesced into a dump already
        in flight / suppressed by the cooldown. The file appears POST_TRIGGER_S later, once the
        post-trigger tail has been recorded.
        """
        if self._stop.is_set() or self.error is not None:
            return None
        self._dump_idx += 1
        name = f"B_{self.short}_{self._dump_idx:04d}_{_slug(reason)}{DUMP_EXT}"
        self._tq.append({"name": name, "reason": str(reason), "t_trig": time.monotonic(),
                         "fields": {k: _jsonable(v) for k, v in fields.items()},
                         "cooldown": DUMP_COOLDOWN_S if cooldown_s is None else float(cooldown_s),
                         "manual": bool(fields.get("manual"))})
        self.log_event("dump.trigger", reason=str(reason), file=name, **fields)
        return name

    def mark(self, text, **fields):
        """Operator annotation — the thing you type while the robot is still on the bench."""
        self.log_event("mark", text=str(text), **fields)
        return self.trigger_dump("operator_mark", manual=True, text=str(text))

    def status(self):
        """Counters as of the writer's last cycle, but liveness/error read live — a dead writer
        must show up in the UI immediately, not one publish interval later."""
        s = dict(self._status)
        t = self._thread
        s["error"] = self.error
        s["alive"] = bool(s.get("alive") and self.error is None and t is not None and t.is_alive())
        return s

    def set_config_provider(self, fn):
        """fn() -> {"calibration": ..., "dynamics": ...}; hashed into every Tier B dump header so a
        dump is self-describing without external context."""
        self._config_fn = fn

    def note_config_change(self, what):
        """Log the config as it now stands (called on every calibration/dynstore write)."""
        h, snap = self._config()
        self.log_event("config.snapshot", what=str(what), config_hash=h, config=snap)

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        os.makedirs(self.dir, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="BlackBoxWriter", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout=3.0):
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout)

    # ------------------------------------------------------------------ writer thread
    def _run(self):
        try:
            self._open_events()
            h, snap = self._config()
            self.log_event(
                "daemon.start", version=VERSION, pid=os.getpid(),
                uptime_s=self.uptime_at_start,
                boot_t_wall=(None if self.uptime_at_start is None
                             else self.t0_wall - self.uptime_at_start),
                record_bytes=RECORD_BYTES, motor_names=list(paths.MOTOR_NAMES),
                ring_seconds=round(len(self._ring) / RING_HZ, 1),
                budget_bytes=self.budget_bytes, config_hash=h, config=snap,
                note="boot_t_wall = t_wall - uptime; trust it only if wall_trusted")
            self._enforce_space(force=True)
            while not self._stop.is_set():
                self._cycle()
                time.sleep(WRITER_PERIOD_S)
            self._cycle(final=True)
            self.log_event("daemon.stop", samples=self._n_seen, written=self._n_written,
                           dropped=self._dropped, dropped_events=self._dropped_events,
                           note="a clean stop — an ABSENT daemon.stop before a gap means the "
                                "process died with the power")
            self._drain_events()
        except Exception:
            self.error = traceback.format_exc()
            print(f"!! BlackBox writer died (the robot is unaffected):\n{self.error}")
        finally:
            self._publish_status(alive=False)
            for f in (self._seg_f, self._ev_f):
                try:
                    if f is not None:
                        f.flush()
                        os.fsync(f.fileno())
                        f.close()
                except OSError:
                    pass
            self._seg_f = self._ev_f = None

    def _cycle(self, final=False):
        now = time.monotonic()
        self._drain_samples()
        self._drain_events()
        self._service_dumps(now, final=final)
        if now - self._last_hb >= self.heartbeat_s or final:
            self._last_hb = now
            self._heartbeat(now)
        if now - self._last_space >= self.space_check_s or final:
            self._last_space = now
            self._enforce_space()
        self._publish_status(alive=True)

    # ---- samples -----------------------------------------------------------
    def _drain_samples(self):
        sq = self._sq
        n = len(sq)
        if not n:
            return
        drops, rows = [], []
        for _ in range(n):
            try:
                d, r = sq.popleft()
            except IndexError:
                break
            if len(r) == ROW_LEN:
                drops.append(d)
                rows.append(r)
        if not rows:
            return
        rec = self._pack(rows, drops)
        self._n_seen += len(rec)
        self._ring_write(rec)
        if self.tier_a_enabled and not self._degraded:
            self._tier_a_write(rec)

    @staticmethod
    def _pack(rows, drops):
        flat = np.asarray(rows, dtype=np.float64)
        rec = np.zeros(len(rows), RECORD_DTYPE)
        rec["t_mono"] = flat[:, 0]
        rec["t_wall"] = flat[:, 1]
        rec["dt"] = flat[:, 2]
        rec["mode"] = np.nan_to_num(flat[:, 3]).astype(np.uint8)
        rec["estop"] = np.nan_to_num(flat[:, 4]).astype(np.uint8)
        rec["slip"] = np.nan_to_num(flat[:, 5]).astype(np.uint32)
        rec["drop"] = np.asarray(drops, dtype=np.uint32)
        o = len(ROW_HEAD)
        for f in MOTOR_FIELDS:
            rec[f] = flat[:, o:o + N]
            o += N
        rec["err"] = np.nan_to_num(flat[:, o:o + N]).astype(np.uint16)
        return rec

    def _ring_write(self, rec):
        ring, cap = self._ring, len(self._ring)
        if len(rec) >= cap:
            ring[:] = rec[-cap:]
            self._ring_seq += len(rec)
            return
        i = self._ring_seq % cap
        first = min(len(rec), cap - i)
        ring[i:i + first] = rec[:first]
        if first < len(rec):
            ring[:len(rec) - first] = rec[first:]
        self._ring_seq += len(rec)

    def _ring_snapshot(self):
        """The ring, oldest first — this is the pre-trigger history."""
        cap = len(self._ring)
        if self._ring_seq <= cap:
            return self._ring[:self._ring_seq].copy()
        i = self._ring_seq % cap
        return np.concatenate([self._ring[i:], self._ring[:i]])

    # ---- Tier A ------------------------------------------------------------
    def _tier_a_write(self, rec):
        take = []
        for k in range(len(rec)):
            if self._decim % TIER_A_DIV == 0:
                take.append(k)
            self._decim += 1
        if not take:
            return
        sub = rec[take]
        try:
            if self._seg_f is None:
                self._open_segment()
            self._seg_f.write(sub.tobytes())
            self._seg_bytes += sub.nbytes
            self._n_written += len(sub)
            self._seg_f.flush()                 # to the OS; fsync is Tier C's job, not Tier A's
            if self._seg_bytes >= self.segment_max_bytes:
                self._close_segment()
        except OSError as e:
            self._space_trouble(e)

    def _open_segment(self):
        self._seg_idx += 1
        stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        self._seg_name = f"A_{self.short}_{self._seg_idx:04d}_{stamp}{SEG_EXT}"
        p = os.path.join(self.dir, self._seg_name)
        self._seg_f = open(p, "ab")
        self._seg_bytes = 0
        if self._seg_f.tell() == 0:
            self._seg_f.write(self._header_bytes("A", {}))
            self._seg_bytes = self._seg_f.tell()
        self.log_event("segment.open", file=self._seg_name)

    def _close_segment(self):
        if self._seg_f is None:
            return
        try:
            self._seg_f.flush()
            os.fsync(self._seg_f.fileno())
            self._seg_f.close()
        except OSError:
            pass
        self.log_event("segment.close", file=self._seg_name, bytes=self._seg_bytes)
        self._seg_f, self._seg_name, self._seg_bytes = None, None, 0

    def _header_bytes(self, tier, extra):
        h = {"version": VERSION, "tier": tier, "created_wall": time.time(),
             "created_mono": time.monotonic(), "wall_trusted": _wall_trusted(),
             "boot_id": self.boot_id, "session_id": self.session_id,
             "t0_mono": self.t0_mono, "t0_wall": self.t0_wall,
             "uptime_at_start_s": self.uptime_at_start,
             "record_bytes": RECORD_BYTES,
             "dtype": [[n, str(RECORD_DTYPE.fields[n][0].base.str),
                        list(RECORD_DTYPE.fields[n][0].shape)]
                       for n in RECORD_DTYPE.names],
             "motor_names": list(paths.MOTOR_NAMES),
             "motor_fields": list(MOTOR_FIELDS) + ["err"],
             "mode_names": list(_MODE_NAMES),
             "rate_hz": (RING_HZ / TIER_A_DIV) if tier == "A" else RING_HZ,
             **extra}
        return MAGIC + b"\n" + json.dumps(h, default=str).encode() + b"\n"

    # ---- Tier B ------------------------------------------------------------
    def _service_dumps(self, now, final=False):
        while self._tq:
            t = self._tq.popleft()
            if self._pending is not None:
                self._pending["also"].append(t["reason"])       # coalesce into the one in flight
                continue
            # The cooldown is PER REASON. A repeating trigger (a warn re-arming, a jittering
            # origin) must not write the same 40 s window over and over — but a NEW kind of
            # trigger always gets its own file, because that is the one nobody has evidence for.
            if not t["manual"] and (now - self._last_dump_end.get(t["reason"], -1e9)) < t["cooldown"]:
                self.log_event("dump.suppressed", reason=t["reason"], file=t["name"],
                               cooldown_s=t["cooldown"],
                               note=f"a dump for this same reason closed <{t['cooldown']:.0f}s "
                                    f"ago and already contains this window")
                continue
            t["also"] = []
            t["deadline"] = t["t_trig"] + self.post_trigger_s
            self._pending = t
        p = self._pending
        if p is not None and (final or now >= p["deadline"]):
            self._write_dump(p)
            self._pending = None
            for r in [p["reason"]] + p["also"]:
                self._last_dump_end[r] = now

    def _write_dump(self, p):
        rec = self._ring_snapshot()
        pre = int(np.count_nonzero(rec["t_mono"] <= p["t_trig"])) if len(rec) else 0
        h, snap = self._config()
        span = (float(rec["t_mono"][-1] - rec["t_mono"][0]) if len(rec) > 1 else 0.0)
        extra = {"trigger": {"reason": p["reason"], "also": p["also"], "t_trig_mono": p["t_trig"],
                             "t_trig_wall": self.t0_wall + (p["t_trig"] - self.t0_mono),
                             **p["fields"]},
                 "n_samples": int(len(rec)), "n_pre_trigger": pre,
                 "n_post_trigger": int(len(rec)) - pre,
                 "pre_trigger_s": round(float(p["t_trig"] - rec["t_mono"][0]), 3) if len(rec) else 0,
                 "span_s": round(span, 3),
                 "config_hash": h, "config": snap,
                 "dropped_samples_total": self._dropped}
        tmp = os.path.join(self.dir, p["name"] + ".part")
        try:
            with open(tmp, "wb") as f:                    # never appended to by anyone else
                f.write(self._header_bytes("B", extra))
                f.write(rec.tobytes())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, os.path.join(self.dir, p["name"]))   # atomic publish
        except OSError as e:
            self._space_trouble(e)
            try:
                os.remove(tmp)
            except OSError:
                pass
            return
        self.log_event("dump.written", file=p["name"], reason=p["reason"], also=p["also"],
                       n_samples=int(len(rec)), n_pre_trigger=pre,
                       pre_trigger_s=extra["pre_trigger_s"], config_hash=h)

    # ---- Tier C ------------------------------------------------------------
    def _open_events(self):
        p = os.path.join(self.dir, EVENTS_NAME)
        self._ev_f = open(p, "a", encoding="utf-8")
        self._ev_bytes = self._ev_f.tell()

    def _drain_events(self):
        if not self._eq:
            return
        if self._ev_f is None:
            try:
                self._open_events()
            except OSError as e:
                self._space_trouble(e)
                self._eq.clear()
                return
        lines = []
        while self._eq:
            try:
                lines.append(json.dumps(self._eq.popleft(), default=str))
            except (TypeError, ValueError):
                pass
        if not lines:
            return
        blob = "\n".join(lines) + "\n"
        try:
            self._ev_f.write(blob)
            self._ev_f.flush()
            os.fsync(self._ev_f.fileno())       # Tier C is fsync'd: it must survive the power cut
            self._ev_bytes += len(blob.encode())
            if self._ev_bytes >= EVENTS_MAX_BYTES:
                self._archive_events()
        except OSError as e:
            self._space_trouble(e)

    def _archive_events(self):
        """Never delete the event log — move it aside and start a fresh one."""
        try:
            self._ev_f.close()
        except OSError:
            pass
        self._ev_f = None
        stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        src = os.path.join(self.dir, EVENTS_NAME)
        try:
            os.replace(src, os.path.join(self.dir, f"events_{stamp}_{self.short}.jsonl"))
        except OSError:
            pass
        try:
            self._open_events()
        except OSError:
            pass

    def _heartbeat(self, now):
        """The record that dates a power cut. The process that dies with the power cannot log its
        own death, so the LAST heartbeat before a gap is the power-off time, to heartbeat
        resolution. blackbox_read.py states that inference explicitly."""
        wall = time.time()
        off = wall - now
        stepped = abs(off - self._wall_offset0)
        if stepped > 1.0:
            self.log_event("clock.step", old_offset=self._wall_offset0, new_offset=off,
                           step_s=off - self._wall_offset0, wall_trusted=_wall_trusted(),
                           note="the wall clock jumped (NTP sync or a manual set) — every earlier "
                                "t_wall in this session is off by roughly -step_s")
            self._wall_offset0 = off
        try:
            du = shutil.disk_usage(self.dir)
            free, total = du.free, du.total
        except OSError:
            free = total = None
        self.log_event("heartbeat", uptime_s=round(now - self.t0_mono, 2),
                       sys_uptime_s=_uptime_s(), samples=self._n_seen, tier_a_written=self._n_written,
                       dropped=self._dropped, dropped_events=self._dropped_events,
                       bytes_used=self._dir_bytes(), disk_free=free, disk_total=total,
                       degraded=self._degraded, segment=self._seg_name)

    # ---- space -------------------------------------------------------------
    def _files(self):
        out = []
        try:
            with os.scandir(self.dir) as it:
                for e in it:
                    try:
                        if e.is_file():
                            out.append((e.name, e.stat().st_size, e.stat().st_mtime))
                    except OSError:
                        pass
        except OSError:
            pass
        return out

    def _dir_bytes(self):
        return sum(s for _, s, _ in self._files())

    def _space_trouble(self, e):
        if getattr(e, "errno", None) in (errno.ENOSPC, errno.EDQUOT):
            if not self._degraded:
                self._degraded = True
                self.log_event("space.degraded", error=str(e),
                               note="filesystem full — Tier A recording stopped. Events and "
                                    "triggered dumps continue; the ROBOT IS UNAFFECTED.")
            self._close_segment()
        else:
            self.log_event("io.error", error=str(e))

    def _enforce_space(self, force=False):
        """Global byte budget + a free-space floor. Degrade Tier A history FIRST; events and
        triggered dumps are the evidence and go last. A full SD card must never stop the robot."""
        files = self._files()
        total = sum(s for _, s, _ in files)
        self._bytes_used = total
        try:
            free = shutil.disk_usage(self.dir).free
        except OSError:
            free = None
        over = total > self.budget_bytes or (free is not None and free < self.min_free_bytes)
        if not over:
            if self._degraded and (free is None or free > self.min_free_bytes * 1.2):
                self._degraded = False
                self.log_event("space.recovered", bytes_used=total, disk_free=free)
            return

        def _cands(ext):
            return sorted([f for f in files if f[0].endswith(ext) and f[0] != self._seg_name],
                          key=lambda f: f[2])
        removed = []
        for ext in (SEG_EXT, DUMP_EXT):                # Tier A history dies before Tier B evidence
            for name, size, _mt in _cands(ext):
                if total <= self.budget_bytes and (free is None or free >= self.min_free_bytes):
                    break
                try:
                    os.remove(os.path.join(self.dir, name))
                except OSError:
                    continue
                removed.append(name)
                total -= size
                if free is not None:
                    free += size
            files = [f for f in files if f[0] not in removed]
        if removed:
            self._bytes_used = total
            self.log_event("space.dropped", removed=removed, n=len(removed),
                           bytes_used=total, disk_free=free, budget=self.budget_bytes,
                           note="oldest recordings deleted to stay inside the budget — these "
                                "windows are gone and no longer reconstructible")
        still_over = total > self.budget_bytes or (free is not None and free < self.min_free_bytes)
        if still_over and not self._degraded:
            self._degraded = True
            self._close_segment()
            self.log_event("space.degraded", bytes_used=total, disk_free=free,
                           note="nothing left to delete — Tier A stopped, events/dumps continue")

    # ---- misc --------------------------------------------------------------
    def _config(self):
        """Snapshot + content hash of the live calibration/dynamics, taken FRESH every time.

        Deliberately not cached: the requirement is that a Tier B dump references the config in
        force at that instant, and a cache is only ever as correct as its invalidation. The
        callers (boot, a config write, a dump) are rare; a dict copy and a sha256 of a few hundred
        bytes is not worth being subtly wrong about.
        """
        snap = {}
        if self._config_fn is not None:
            try:
                snap = _jsonable(self._config_fn() or {})
            except Exception as e:                      # a broken provider must not kill the writer
                snap = {"error": str(e)}
        return _sha(snap), snap

    def _publish_status(self, alive):
        self._status = {
            "alive": bool(alive and self.error is None),
            "error": self.error,
            "boot_id": self.boot_id, "session_id": self.session_id,
            "wall_trusted": _wall_trusted(),
            "uptime_s": round(time.monotonic() - self.t0_mono, 1),
            "samples": self._n_seen, "tier_a_written": self._n_written,
            "dropped": self._dropped, "dropped_events": self._dropped_events,
            "queued": len(self._sq), "queue_max": self.sample_queue_max,
            "ring_s": round(min(self._ring_seq, len(self._ring)) / RING_HZ, 1),
            "bytes_used": self._bytes_used, "budget_bytes": self.budget_bytes,
            "degraded": self._degraded, "segment": self._seg_name,
            "dumps": self._dump_idx, "dir": self.dir,
        }

    # ------------------------------------------------------------------ file access (HTTP)
    def list_files(self):
        """Newest first. Reads only the directory + each file's header — never the bulk data, so
        this is cheap enough to poll and can never block the control loop."""
        out = []
        for name, size, mt in sorted(self._files(), key=lambda f: -f[2]):
            tier = ("A" if name.endswith(SEG_EXT) else
                    "B" if name.endswith(DUMP_EXT) else
                    "C" if name.endswith(".jsonl") else "?")
            item = {"name": name, "bytes": size, "mtime": mt, "tier": tier,
                    "current": name == self._seg_name}
            if tier in ("A", "B"):
                try:
                    h = read_header(os.path.join(self.dir, name))
                    item["n_samples"] = h.get("n_samples")
                    item["session_id"] = h.get("session_id")
                    item["reason"] = (h.get("trigger") or {}).get("reason")
                    item["pre_trigger_s"] = h.get("pre_trigger_s")
                    if item["n_samples"] is None:
                        item["n_samples"] = max(0, (size - h["_data_offset"]) // RECORD_BYTES)
                except (OSError, ValueError):
                    item["error"] = "unreadable header"
            out.append(item)
        return out

    def read_bytes(self, name):
        """Bytes of one recording, for download. Files are append-only and never rewritten, so a
        concurrent read is always consistent up to whatever was flushed."""
        base = os.path.basename(name)
        if base != name or not base:
            raise FileNotFoundError(f"bad name {name!r}")
        p = os.path.join(self.dir, base)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"no such recording: {base}")
        if base == EVENTS_NAME:
            self._drain_events()                      # hand out everything logged so far
        with open(p, "rb") as f:
            return f.read(), base


# ===================================================================== reading (shared w/ reader)
def read_header(path):
    """Parse a segment/dump header. Adds _data_offset (first byte of record data)."""
    with open(path, "rb") as f:
        magic = f.readline()
        if magic.strip() != MAGIC:
            raise ValueError(f"{os.path.basename(path)}: not a black-box file")
        line = f.readline()
        h = json.loads(line.decode())
        h["_data_offset"] = f.tell()
    return h


def read_segment(path):
    """(header, records) — tolerant of a file that was killed mid-write: a torn tail record is
    floored away rather than raising, which is the whole point of the append-only format."""
    h = read_header(path)
    with open(path, "rb") as f:
        f.seek(h["_data_offset"])
        blob = f.read()
    n = len(blob) // RECORD_BYTES
    h["_torn_bytes"] = len(blob) - n * RECORD_BYTES
    return h, np.frombuffer(blob[:n * RECORD_BYTES], dtype=RECORD_DTYPE)


def read_events(path):
    """Every parseable JSONL line; a truncated last line is skipped, not fatal."""
    out = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


# ===================================================================== module-level hook
# calibration.py / dynstore.py / measurestore.py are plain stores with no daemon reference; rather
# than thread one through every call site they log through here. Before install() (and in tests and
# any offline import) these are silent no-ops, so importing those modules never requires a recorder.
_BB = None


def install(bb):
    global _BB
    _BB = bb
    return bb


def get():
    return _BB


def log_event(kind, /, **fields):
    bb = _BB
    if bb is not None:
        bb.log_event(kind, **fields)


def trigger_dump(reason, /, **fields):
    bb = _BB
    return bb.trigger_dump(reason, **fields) if bb is not None else None


def note_config_change(what):
    bb = _BB
    if bb is not None:
        bb.note_config_change(what)
