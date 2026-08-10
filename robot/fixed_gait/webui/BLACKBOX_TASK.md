# Task: a black box for DASH-01

Implement persistent, always-on recording on the Raspberry Pi so that any future failure can be
reconstructed after the fact. Today a joint destroyed itself and **the entire investigation was
impossible** — the only telemetry that existed was a 4096-sample RAM ring (`ringbuffer.TelemetryRing`,
~3.4 min at 20 Hz), of which the HTTP API can only ever return the last 512 samples (25.6 s), and
nothing was on disk. By the time anyone looked, the window had rolled past the event.

Note on wording: this is **not** CubeMars firmware — we cannot change what runs on the driver
boards. Everything here lives in the Pi-side daemon (`robot/fixed_gait/webui/`).

## The bar to clear

The design is right if it can answer these, unprompted, from disk alone:

1. When did the Pi last boot, and when did it last lose power?
2. What was `pos_raw` for all six motors at the last successful zero capture — and does it match
   `pos_raw` now, with the robot untouched? (A stale calibration is invisible until the first
   absolute move, and that move is a full-authority slew. This is the check that would have
   prevented today's failure.)
3. At the moment of a trip: commanded vs reported position, per motor, at full 200 Hz rate, for at
   least 10 s *before* the trigger.
4. Which calibration, which drive PID gains, and which dynamics config were live at that instant.
5. Did `pos_raw` ever change while the robot was LIMP and physically untouched? (i.e. did a driver
   board's multi-turn origin move on its own — the open question from 2026-08-10.)

## What exists today

| module | role |
|---|---|
| `daemon.py` | `RobotDaemon`, 200 Hz tick (`TICK_HZ`), modes LIMP/MANUAL/PLAYBACK/MEASURE/ESTOPPED, `_trip()`, `self._slip_count` |
| `ringbuffer.py` | `TelemetryRing(capacity=4096)`, pushed every `TELEMETRY_DIV`=10 ticks → 20 Hz, RAM only |
| `calibration.py` | `offsets[]`, `signs[]`, `stage`, `created`; `set_zero()` refuses partial capture (all six or none) |
| `dynstore.py` | weighed masses, per-motor position PID, Kt, and `DRIVE_GAINS` (board config, recorded not pushed) |
| `measurestore.py` | explicit MEASURE runs saved as `.npz` + metadata — only on an explicit finish |
| `canio.py` | servo mode: `set_pos` (packet 4), `set_current` (packet 1); `MockBus` for hardware-free tests |
| `paths.py` | `DATA`, `MEASURE_DIR`, `CALIB_FILE`, `DYN_CONFIG_FILE`; `MOTOR_NAMES` is **right leg first** |

## Architecture: three tiers

Do not try to stream 200 Hz to disk continuously. Six motors × 7 fields × float32 ≈ 176 B/sample →
**35 kB/s, 3 GB/day**. That destroys the SD card and fills the disk. Use a flight-recorder split:

### Tier A — continuous, low rate (always on)
20 Hz, every field, binary append, rotated. ~3.5 kB/s ≈ 300 MB/day before rotation.
Fields per motor: `pos_raw, pos_norm, cmd_raw, cmd_norm, spd, cur, temp, err`.
Plus per-sample scalars: `mode`, `estop_latched`, `slip_count`, loop period actual.

### Tier B — full rate, event-triggered (the actual black box)
Keep a 200 Hz RAM ring of **at least 30 s**. On any trigger, atomically dump the whole ring —
pre-trigger history included — to its own file. This is the tier that would have solved today.

Triggers: any `_trip()`, e-stop latch, mode transition, tracking error above a *warn* threshold
(below the trip threshold — catch the approach, not just the fall), `|spd|` above a warn threshold,
any calibration write, any `dynstore` write, motor `err != 0`, temperature threshold, and a manual
"dump now" endpoint.

Include **post-trigger** samples too (a few seconds), then close the file.

### Tier C — event log (append-only JSONL, `fsync` on write)
One line per event, never rotated away without archiving. Every event carries `boot_id`,
`t_mono`, `t_wall`, `wall_trusted`.

Log at minimum: daemon start/stop, mode changes with reason, every `_trip` with its full message,
e-stop set/clear, calibration `set_zero`/`set_sign`/save (with the **full raw pose captured**, all
six, before and after), every `dynstore` mutation (old → new value), MEASURE start/stop/finish/
delete, workspace and feasibility rejections, CAN bus errors/timeouts, motor error codes,
temperature warnings, disk-space warnings, and the periodic heartbeat.

## Hard requirements

**Do not perturb the 200 Hz loop.** The tick already slips ~12% (`slip_count` 33519 of ~274k).
The daemon thread must only push into a lock-free/bounded queue; a separate writer thread does all
file I/O. If the queue is full, **drop and count** — never block the control loop. Log the drop
count; a black box that lies about its own gaps is worse than none.

**Power-off cannot be logged by the process that dies with it.** Write a heartbeat record (Tier C,
`fsync`) every 5–10 s containing `t_mono`, `t_wall`, `uptime`. The last heartbeat before a gap *is*
the power-off time, to heartbeat resolution. State that inference explicitly in the reader.

**The clock is not trustworthy.** The Pi has no RTC and serves an AP at 192.168.4.1 with no
guaranteed internet, so `t_wall` after boot may be years wrong until/unless NTP syncs. Therefore:
- every record carries `t_mono` (`time.monotonic()`) as the primary time base,
- plus `t_wall` and a `wall_trusted` boolean (check whether the clock has stepped / NTP synced),
- plus a `boot_id` (random UUID per daemon start, or `/proc/sys/kernel/random/boot_id`) so records
  from different sessions can never be silently interleaved.
Record the wall-clock step when NTP does sync, so earlier records can be retro-corrected.

**Never fill the disk.** Global byte budget with rotation and deletion of the oldest Tier A
segments. Reserve headroom and refuse to grow past it. Tier B dumps and Tier C events are
higher-priority than Tier A history — degrade Tier A first. Log when data is dropped for space.
A full SD card must never be able to stop the robot from running.

**Config provenance.** On every boot, and on every change, snapshot `calibration.snapshot()`,
`dynstore.as_dict()` (including `DRIVE_GAINS`) into the log with a content hash. Every Tier B dump
references the hash live at that moment. A run must be self-describing without external context.

**Raw-at-rest fingerprint.** On every successful zero capture, and on every LIMP→active transition,
record `pos_raw` for all six. Expose a comparison so the pre-move guard (below) and any postmortem
can use it directly.

## Deliverables

1. `blackbox.py` — writer thread, tiering, rotation, budget, boot/heartbeat, event API. Public
   surface small: `log_event(kind, **fields)`, `push_sample(...)`, `trigger_dump(reason)`.
2. Wiring in `daemon.py` — sample push in the tick (after the existing `ring.push`), `_trip()` and
   every mode transition raising events and dumps. Must not change control behaviour.
3. Wiring in `calibration.py` / `dynstore.py` / `measurestore.py` — mutations raise events with
   old → new values.
4. **Pre-move safety guard** (the point of the exercise): on any transition out of LIMP, compare
   each motor's `pos_raw` against the raw recorded at the last successful zero. If any joint differs
   by more than a configurable threshold (default a few degrees), refuse the transition, latch a
   clear reason, and dump. Also make the first move after a re-zero or reboot a small bounded one —
   `center()`/`home()` must not be the first absolute command after either.
5. HTTP: `GET /api/blackbox/list`, `GET /api/blackbox/download?name=`, `POST /api/blackbox/mark`
   (operator annotation), `POST /api/blackbox/dump`. Same `send_file` pattern as
   `/api/measure/export`. Downloads must work while the daemon is running and must never block it.
6. `blackbox_read.py` — offline reader: segment → numpy/pandas, event timeline, gap detection
   (power-off inference), and a `--postmortem` mode that answers the five questions above.
7. Run the daemon under **systemd** so it starts at boot, restarts on crash, and its stdout reaches
   the journal. Today the only surviving record was a terminal that happened to still be open.
   Ship the unit file; the repo currently has none.
8. Tests using `canio.MockBus` (no hardware): rotation and budget enforcement, queue-full drop
   accounting, trigger dump contains genuine pre-trigger history, boot/heartbeat/gap detection,
   reader round-trip, and the pre-move guard both firing and not false-firing.

## Format

Prefer a self-describing binary segment (fixed-width records + a JSON header naming the columns,
`MOTOR_NAMES` order explicit) over CSV: ~5× smaller, cheaper to write, no float formatting cost.
JSONL is right for Tier C, where readability and append-safety beat size. Every file must be
readable even if the process was killed mid-write — append-only, and never rewrite in place.

## Explicit non-goals

Don't stream 200 Hz continuously. Don't change control behaviour, gains, or the CAN protocol.
Don't touch driver-board firmware. Don't make the robot depend on the black box to run — if the
writer thread dies, the robot keeps working and the failure is surfaced in the UI.

## Acceptance

Reproduce today's incident shape on the bench with `MockBus`: capture a zero, mutate the raw origin
underneath the daemon, command an absolute move. The guard must refuse it, an event must record the
mismatch with both raw poses, a Tier B dump must contain ≥10 s of pre-trigger 200 Hz data, and
`blackbox_read.py --postmortem` must state what happened without a human reading raw files.
