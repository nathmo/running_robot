# DASH-01 web control interface

One Flask app, served **on the robot's Raspberry Pi**, reachable from any phone/laptop on the
robot's WiFi hotspot. No internet needed at runtime — every asset is served locally.

```
python robot/fixed_gait/webui/server.py          # real robot (socketcan can0/can1)
python robot/fixed_gait/webui/server.py --mock   # simulated motors — try the UI on any machine
python robot/fixed_gait/webui/server.py --port 8080 --host 0.0.0.0
```

Then browse to `http://<robot-ip>:8080/` (on the hotspot, the Pi's own address).

What it does (one page):

| Panel | Function |
|---|---|
| Telemetry | live raw/normalized position, current, temperature per motor + strip charts |
| Sense HAT (B) | live 9-DOF IMU (attitude horizon, accel/gyro/mag), air temperature/humidity/pressure, ambient light/colour, 4 analog inputs, and the **IMU mount calibration** (upright + tilt captures, lever arm, 3D frame view) — see below |
| Calibration wizard | **blocks all motion after boot** until zero pose + direction check are done |
| Manual control | per-actuator slider that **tracks the live motor position** (grab to jog) + exact-angle box; **🏠 Home** slowly returns every joint to the zero pose and **⌖ Centre** parks both legs where the safe workspace leaves the most room in every direction (the largest inscribed box — the pose to excite from); per-actuator sine (start↔stop **preset to 70% of the safe range**, frequency); workspace-override checkbox |
| Safe workspace | abduction bar + (cam, thigh) pixel editor — draw/erase/flood-fill, undo, save/export/import |
| Gait trajectory | hand-**draw** a path, or **teach** by backdriving (record takes), smooth + save |
| EE animation | live linkage + workspace + gait + zero in foot (end-effector) space, per leg |
| Playback | position or torque-capped current mode, speed (period) + left/right dephasing live |
| E-STOP | header button (hotkey: Space) — streams zero current, latches until cleared |

## Why the calibration wizard exists

The CubeMars encoders report a **random angle (and possibly inverted direction) after every power
cycle**. The app therefore refuses any motion until you:

1. pose the robot in its **URDF zero pose** and press *Set zero* (motors are limp), then
2. backdrive each joint in its positive direction and *Confirm* (or *Flip*) its sign.

Everything the app stores (workspaces, gaits, zero) is in this **normalized frame**
(`norm = sign · (raw − offset)`), so recordings survive reboots. Legacy files from
`calibrate_workspace.py` are converted on import using the zero captured in their session
(`z` key) — files without a stored zero cannot be normalized and are refused.

After a **server** restart the calibration is restored from disk with a warning banner: it is only
valid if the motors were NOT power-cycled in between. Re-zero when unsure — it never invalidates
saved data.

## The FK lookup table (EE animation)

`fk_lut.npz` is generated **on the desktop** (needs mujoco, the Pi doesn't have it):

```
python mujoco/dash01/gen_fk_lut.py --check
```

It Newton-solves the closed 4-bar at every (cam, thigh) grid cell via
`mujoco/dash01/plot_reachability.py` and stores all linkage node positions. The Pi only
bilinearly interpolates it. After loading/importing a workspace, press **“verify FK map
(sign + offset)”** in the EE panel once: per side and per sign combination it FITS the offset
that lands the workspace band on the LUT assembly band (thin diagonal ⇒ sharp fit) and enables
the EE display per leg only when the winner is decisive.

> **Two zero poses — resolved (2026-07-11).** The robot's captured zero pose is NOT the model's
> qpos-0 pose, so the map is `model° = sign·norm° + offset°` per side. Two facts, both verified
> against recorded data:
> 1. **The cam is a crank.** The 4-bar assembles at EVERY cam angle (checked by sweeping loop
>    closure over the full circle); the MJCF's ±86° cam range is a CAD guess. The LUT is now
>    generated over cam ∈ [−90°, +270°] (`gen_fk_lut.py` default), which is why the old sign-only
>    verify failed (17% coverage): most of the real ~245° cam travel fell off the old grid.
> 2. **The captured zero sits far from model zero.** The zero pose is taken with the leg
>    near-extended — exactly the 4-bar dead-center, where big cam rotations barely move the leg —
>    so the captured cam zero is unrepeatable and lands far from model cam 0 (fitted ≈ +135° on
>    `joint_limits.npz`, both legs consistent, coverage 1.00 vs ≤0.95 for wrong signs). The EE
>    panel shows both: ○ = calibrated zero, ◆ = URDF/model zero.
>
> All recordings map onto the same model window (cam ≈ [−64°, +183°]) — that window is the real
> hard-stop travel expressed in model coordinates. `data/model_map.json` stores signs + offsets
> and can still be hand-edited / force-enabled in the UI.

## Drive control loops (read off the motor driver boards, 2026-08-04)

This app closes **no** control loop. `canio` speaks CubeMars servo mode — `SET_POS` and
`SET_CURRENT` only — so the drives close everything with their own three cascaded loops. These are
their configured gains, recorded in `dynstore.DRIVE_GAINS` and shipped with every measurement run so
a capture is self-describing. They are **not settable from here** (no gain-write packet; use the
CubeMars tool), and editing the PID table in the UI changes nothing on the robot.

| group | motors | current kp / ki | speed kp / ki | position kp / ki / kd |
|---|---|---|---|---|
| left sagittal | `left.cam`, `left.thigh` | 0.1255 / 1704.8199 | 0.002 / 0.1 | 0.003 / 0 / 0 |
| right sagittal | `right.cam`, `right.thigh` | 0.2066 / 2544.6150 | 0.002 / 0.1 | 0.003 / 0 / 0 |
| abduction | `left.abd`, `right.abd` | 0.1190 / 2290.1199 | 0.002 / 0.06 | 0.009 / 0 / 0 |

"Thigh+Knee" on the driver board is a leg's **sagittal pair**: the knee is driven by the cam through
the pushrod, so cam and thigh share a board and a tune. Left and right current-loop gains differ
despite identical AKE90-8 motors — that is per-board autotuning against the real winding R/L, not an
asymmetry in the robot.

Two consequences worth remembering when reading identification data:

* **The position loop is P-only** (ki = kd = 0 on all six). A pure proportional loop droops under a
  constant load, so a joint holding against gravity settles *short* of its target by roughly
  (gravity torque / gain). That is a systematic bias in a quasi-static run, not noise.
* **Abduction is ~3x stiffer** than the sagittal axes (kp 0.009 vs 0.003), so it droops ~3x less.

## Sense HAT (B) — the sensor panel

The Waveshare **Sense HAT (B)** sits on the Pi's 40-pin header and answers on **I2C bus 1**. The
chips are read by background threads of their own (`sensehat.py`), never by the Flask handlers and
never by the CAN daemon — an I2C stall must not be able to delay a motor command.

**Two threads, split by timescale.** The IMU polls at `--imu-hz` (200 by default, matching the
control loop); the air/pressure/light/ADC chips run at 1-5 Hz on a second thread. That split is not
tidiness: those drivers spend nearly all their time *asleep* waiting for a conversion (the SHTC3
alone sleeps 15 ms), which inline cost the fast loop ~43 ms every second — about 8 missed ticks —
and capped it at ~192 Hz. Out of the way, the IMU thread holds **199.5 Hz**.

| Address | Chip | Published |
|---|---|---|
| 0x68 | ICM-20948 | accel (g), gyro (deg/s), die temperature; magnetometer via the AK09916 at 0x0C |
| 0x70 | SHTC3 | air temperature, relative humidity |
| 0x5C | LPS22HB | barometric pressure + its own die temperature |
| 0x29 | TCS34725 | ambient light (lux), correlated colour temperature, raw RGBC |
| 0x48 | ADS1015 | AIN0..AIN3, volts (one channel per tick) |

Measured on the robot (100 kHz I2C, at rest, DLPF cfg 3):

| | accel | gyro |
|---|---|---|
| range / resolution | ±4 g, 0.1221 mg/LSB | ±1000 dps, 0.0305 dps/LSB |
| 3 dB bandwidth | 50.4 Hz | 51.2 Hz |
| noise (RMS) | 1.8 mg (~218 µg/√Hz) | 0.08 dps (~0.0096 dps/√Hz) |
| bias | scale error −0.03% | turn-on ~1 dps; 0.011 dps/s in-run |

Quantisation sits ~15x below the noise floor on both, so widening the ranges costs nothing real if
footfall impacts start clipping ±4 g. Sensor ODR is 225 Hz (`SMPLRT_DIV` derived from `ODR_HZ`); an
accel+gyro burst costs 1.68 ms and the magnetometer ~1 ms, hence the mag's decimation to 25 Hz.
The panel publishes the rate the IMU thread actually achieves, not the one it was asked for.

```
python robot/fixed_gait/webui/sensehat.py          # CLI self-test: one live line per second
python robot/fixed_gait/webui/sensehat.py --mock   # synthetic values, no hardware
python robot/fixed_gait/webui/server.py --no-sensors   # do not touch I2C at all
```

Things worth knowing before trusting a number here:

* **Roll and pitch are absolute** (a 6-axis Madgwick filter references them to gravity); **yaw is
  gyro-integrated and drifts.** The magnetometer is deliberately *not* fused into the attitude —
  it sits centimetres from two brushless motors and the battery, so its heading is published as an
  advisory number only.
* **Zero the gyro** (button under the horizon) with the robot standing still: it averages the
  zero-rate offset for 1.5 s and refuses if the robot moved (it says so rather than storing a bad
  bias).
* **Values are in the IMU chip's own frame until the mount is calibrated** (below), and in robot
  body axes after. The HAT is bolted UNDER the robot — it reads gravity on chip −Z — so the two
  frames are nowhere near each other and the difference is not cosmetic.
* Missing HAT, missing `smbus2`, or I2C disabled → the panel shows the reason and the rest of the
  UI is unaffected. Per-chip failures are counted and shown, not silent.

### Mount calibration — where the IMU is and which way it faces

`mountcal.py`, persisted to `data/sensehat_mount.json`, panel block "IMU mount & frame".

**Step 1 — upright reference.** Hang the robot on the test rig in its upright pose and capture.
This measures the direction of gravity in chip axes, which fixes the tilt. It does *not* separate
accelerometer bias from mount misalignment — a robot tilted 1° and a sensor with a 17 mg cross-axis
bias read identically — and it does not need to: both are absorbed into the frame in which the
reference pose reads roll = pitch = 0. (A true per-axis bias+scale calibration needs a
6-orientation tumble; not happening with a 15 kg robot, and pointless for a sensor already within
1% of 1 g.) **The repeatability of that pose is the accuracy ceiling of everything downstream.**

**Step 2 — fore-aft axis.** Gravity fixes only *two* of the three rotation DOF: rotation about the
vertical is invisible to an accelerometer at rest, and that is exactly the DOF separating **pitch
from roll**. Fix it by tipping the robot nose-down 10–20° and capturing (only the direction is
used, never the angle), and/or by declaring which chip axis points forward. With both, the panel
reports the angle between them — a large disagreement means the HAT is not bolted on square. The
*measured* axis is the one used.

**Step 3 — lever arm.** The IMU sits below the base, so while the body rotates it also measures
`α×r + ω×(ω×r)`. Zero at rest, but a bias on roll/pitch exactly while running. Type the CAD vector
(base centre → IMU, metres) and/or fit one by rocking the robot by hand.

> **What the fit is relative to.** A single IMU *cannot* observe its position relative to the base
> centre — `a_base` in the rigid-body relation is unknown, so `r` is not separable. It becomes
> identifiable only for rotation about a fixed pivot, where the fit returns `r_pivot→IMU`. Hung on
> the rig, that pivot is the hang point: the fit matches the CAD vector only insofar as the base
> centre sits at the pivot, and otherwise differs by exactly the pivot offset. The panel labels the
> fit with the point it is about, and reports the CAD-vs-fit gap rather than implying they must
> agree.

Rock about **two clearly different axes** — a single-axis rock leaves the fit unconstrained along
that axis and returns a number that looks fine and means nothing. The panel reports second-axis
coverage and flags a weak excitation.

The 3D view shows the base mesh, the body triad at the base centre, and the IMU's own axes at the
lever arm, plus the live measured up-vector — the visual check that the calibration says what you
think it does.

Two things this machinery gets right that are easy to get wrong, both verified in `--mock` against
a simulated HAT with a known mount and lever (mock-tools panel poses it: upright / nose-down /
rocking):

* **Changing the mount resets the attitude filter.** A filter's quaternion is expressed in the
  frame it was integrated in and is meaningless the moment that frame moves — and a near-antipodal
  Madgwick error sits on a saddle where the correction vanishes, so it never converges out on its
  own. Without the reset, finishing the wizard left the attitude stuck ~165° wrong.
* **The ω in the lever terms is the same low-passed ω that α is differentiated from.** Pairing a
  raw ω with a filtered-ω derivative mismatches their phase and biases the fit.

End-to-end in mock: mount rotation recovered to <0.01, lever arm to ~3 mm, and enabling the
compensation cuts attitude error during hard rocking by ~77% (2.0° → 0.5° mean).

## Black box (flight recorder) — `blackbox.py` / `blackbox_read.py`

On 2026-08-10 a joint destroyed itself and the investigation was impossible: the only telemetry
that existed was a 4096-sample RAM ring at 20 Hz, of which the HTTP API can only ever hand out the
last 512 (25.6 s), and nothing on disk. By the time anyone looked the window had rolled past.

Everything now lands in `data/blackbox/`, always on, in three tiers — because 200 Hz x 6 motors
streamed continuously is 35 kB/s, 3 GB/day, and that kills the SD card:

| tier | what | rate | file |
|---|---|---|---|
| A | continuous history, every field | 20 Hz | `A_<session>_<n>_<stamp>.bbseg` (rotated, budgeted) |
| B | the actual black box: the full-rate RAM ring dumped on a trigger, **pre-trigger history included** | 200 Hz, 40 s ring + 3 s tail | `B_<session>_<n>_<reason>.bbdump` |
| C | one JSON line per event, `fsync`ed, never deleted without archiving | — | `events.jsonl` |

Tier B triggers: every `_trip()`, e-stop, mode transition, a *warn*-level tracking error / speed /
temperature (below the trip thresholds, so the dump catches the approach and not only the fall),
any motor error code, any calibration or `dynstore` write, an encoder-origin jump, and the manual
"dump now" endpoint. A dump carries the calibration + drive gains + dynamics live at that instant,
with a content hash, so it is self-describing without external context.

**The control loop is not touched.** The 200 Hz thread only appends a tuple to a bounded deque (no
lock, ~2 us); a separate writer thread owns every file. A full queue **drops and counts**, and the
count is stamped into the next accepted record — a black box that lies about its own gaps is worse
than none. If the writer thread dies the robot keeps running and `state.blackbox.alive` goes false
in the UI.

**Time.** The Pi has no RTC and serves an AP with no guaranteed internet, so `t_wall` can be years
wrong. Every record carries `t_mono` (the primary base), `t_wall`, `wall_trusted`, a per-boot
`boot_id` and a per-run `session_id`. A `clock.step` event records the jump when NTP finally syncs,
so earlier stamps can be retro-corrected. **Power-off is inferred from the last heartbeat** (every
8 s) before a gap — the process that dies with the power cannot log its own death.

**Disk.** A global byte budget (default 1.2 GB, `--blackbox-budget-mb`) plus a free-space floor.
Oldest Tier A segments are deleted first; Tier B dumps and the event log are the evidence and go
last. Every deletion is logged. A full SD card can never stop the robot.

### Reading it back

```
python robot/fixed_gait/webui/blackbox_read.py --postmortem      # the five questions, answered
python robot/fixed_gait/webui/blackbox_read.py --list
python robot/fixed_gait/webui/blackbox_read.py --timeline 80
python robot/fixed_gait/webui/blackbox_read.py --dump B_xxx.bbdump --csv out.csv
```

`--postmortem` answers, from disk alone: when the Pi booted and when it last lost power; what
`pos_raw` was at the last zero capture versus now; commanded vs reported position per motor at
200 Hz for >= 10 s *before* a trip; which calibration / drive PID gains / dynamics were live at
that instant; and whether `pos_raw` ever moved while the robot was LIMP and untouched.

Over HTTP: `GET /api/blackbox/list`, `GET /api/blackbox/download?name=`,
`POST /api/blackbox/mark` (operator annotation — it lands on the same timeline) and
`POST /api/blackbox/dump`. Downloads are plain reads of append-only files, so they never block the
daemon.

### The pre-move guard

The 2026-08-10 damage was a full-authority absolute slew against a calibration whose raw origin had
moved underneath it. Four mechanisms now bound that, and **none of them adds a step to the normal
zero -> Home workflow** — homing straight after a capture is the correct operator action, because
the legs are limp and sagging and homing is what catches them:

a. **Hold before moving.** The first CAN command after leaving LIMP is `set_pos(pos_raw measured
   right now)`, held for 0.35 s. It never consults the offsets, so it is safe even against a
   completely wrong zero, and it stops the sag immediately.
b. **Raw-at-rest.** `pos_raw` is compared against the fingerprint captured at the last zero.
   Decisive when we cannot vouch for continuity ourselves — a calibration restored from disk and
   never re-captured in this process, i.e. after a power cycle, where the drives re-randomise their
   raw origin. Within a live session the continuity watchdog is used instead, because gravity sag
   legitimately moves the joints and must not be mistaken for an origin jump. Plus a range-sanity
   check that is decisive in every case: 678 deg on a +-88 deg axis is refused.
c. **A tighter tracking threshold while homing** (8 deg, not the 25 deg playback value): a slow
   guided slew should track well, so a stale zero shows up within the first few degrees.
d. **A per-joint travel budget** as the backstop: a guided move that has swept more than 1.3x the
   joint's own range without arriving is stopped. `left.cam` reached ~1.9 output turns on
   2026-08-10; this cuts it at ~0.6 of a turn whatever the calibration claims.

A refusal is latched, surfaced in `state.premove` with both raw poses, logged, and dumped.

### Tests

```
python -m pytest robot/fixed_gait/webui/tests -v
```

MockBus only, no hardware. The last test reproduces the 2026-08-10 incident shape end to end:
capture a zero, move the raw origin underneath the daemon (`MockBus.shift_origin`, which shifts the
reported *and* commanded frame while the shaft stays put and reports zero speed), command an
absolute move — and asserts the guard refuses it, the event carries both raw poses, the dump holds
>= 10 s of pre-trigger 200 Hz data, and `--postmortem` states what happened.

## Install on the Pi (offline)

```
# on the desktop (internet):
pip download flask smbus2 -d wheels/
scp -r wheels nemo@<pi>:running_robot/

# on the Pi:
pip install -r requirements-rpi.txt          # if not already there (numpy, python-can)
pip install --no-index --find-links wheels/ flask smbus2
# smbus2 is pure Python, so if Raspberry Pi OS already ships it system-wide you can instead just
# drop it into the venv:  cp -r /usr/lib/python3/dist-packages/smbus2 .venv/lib/python3*/site-packages/
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
python robot/fixed_gait/webui/server.py
```

### Run at boot (two systemd units)

Bringing CAN up needs root (`CAP_NET_ADMIN`); the web UI does not. Splitting them keeps the
app running as the unprivileged `nemo` user, and keeps the interfaces up even if the app
restarts. **One** unit brings CAN up (root, oneshot), the **other** starts the UI (as `nemo`)
and waits for the first.

**1. CAN bring-up — `/etc/systemd/system/can-up.service`** (runs as root; no `User=`):

```
[Unit]
Description=Bring up CAN interfaces (can0/can1)
After=sys-subsystem-net-devices-can0.device

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/sbin/ip link set can0 up type can bitrate 1000000
ExecStart=/sbin/ip link set can1 up type can bitrate 1000000

[Install]
WantedBy=multi-user.target
```

**2. Web UI — ship the unit that is in the repo**, `systemd/runningrobot-webui.service`. It is
version-controlled precisely because the only surviving record of 2026-08-10 was a terminal that
happened to still be open; a hand-started process has no journal and no restart.

```
sudo cp robot/fixed_gait/webui/systemd/runningrobot-webui.service /etc/systemd/system/
```

It runs as `nemo`, waits for `can-up.service`, sets `PYTHONUNBUFFERED=1` so the journal never loses
the last lines before a crash, restarts always (bounded to 5 restarts / 60 s so a boot loop stays
down visibly instead of power-cycling the motors), and stops with `SIGINT` so `atexit` runs — that
is what streams 0 A to every motor and lets the black box write its clean `daemon.stop` record,
whose ABSENCE is how a postmortem infers a power cut.

Note `ExecStart` uses the venv's Python by absolute path — systemd runs no shell, so it won't
find `python` on `PATH` or activate a venv. Enable both:

```
sudo systemctl daemon-reload
sudo systemctl enable --now can-up.service runningrobot-webui.service
ip -details link show can0                     # confirm state UP, bitrate 1000000
systemctl status runningrobot-webui         # confirm active (running)
journalctl -u runningrobot-webui -f         # live logs
```

> Don't put `ip link set` in the webui unit's `ExecStartPre`: that step would run as `nemo` and
> fail with `RTNETLINK answers: Operation not permitted`. If you must keep it in one unit, run
> just that line elevated with the `+` prefix (`ExecStartPre=+-/sbin/ip link set can0 up …`).

## First hardware bring-up (in this order)

1. Legs held by hand: press **E-STOP**, verify motors are limp; clear it.
2. Unplug one motor: the app must flag it silent and refuse all motion.
3. Run the calibration wizard for real; sanity-check a normalized angle with a protractor.
4. Import yesterday's workspace (`fixed_gait/joint_limits_right_2.npz`) in the workspace panel.
5. Press *verify FK map (sign + offset)* — both legs should be decisive; the EE panels light up.
6. Small manual moves near zero with safety checks ON (override OFF). The sliders track the live
   position, so watch a joint follow your drag; then press **🏠 Home** and confirm all six slew
   back to zero at the (slow) home speed. Homing trusts the CAD zero: it slews under the physical
   feasibility net rather than the eroded gait polygon, so it still reaches 0 when 0 sits a degree
   or so outside the hand-drawn safe region (a plain manual hold at zero is refused there).
7. Record a workspace sweep + a gait via the UI; play it back slow in **position** mode
   (period ≥ 10 s), then try **current** mode with the 3 A default.

## Architecture (for maintenance)

- `daemon.py` — the only thread touching CAN (200 Hz; mirrors `play_trajectory.py`'s loop).
  Limp/e-stop = **streaming** `SET_CURRENT 0` (never just "stop sending"). All targets are
  workspace-checked as a full pose *before anything is sent*.
- `calibration.py` — normalized↔raw conversion at the daemon boundary only; legacy converters.
- `workspace.py` / `gaitstore.py` — normalized-frame stores wrapping `calibrate_workspace.py`,
  `joint_limits.py`, `trajectory.py` (files stay compatible with the CLI tools).
- `fklut.py` — pure-numpy LUT runtime + the sign-map verifier.
- `canio.py` — real socketcan or `--mock` simulated motors (random boot offsets included).
- `sensehat.py` — Sense HAT (B) I2C drivers + its own poll thread (100 Hz IMU / 20 Hz logged);
  `--mock` is a rigid-body IMU simulator with a known mount and lever arm. Publishes a snapshot +
  a `ScalarRing`, same read-by-sequence contract as motor telemetry.
- `mountcal.py` — IMU mount rotation + lever arm: the capture maths, the lever least-squares and
  its diagnostics, persisted to `data/sensehat_mount.json`. No I2C, no threads of its own.
- `static/` — vanilla JS, no CDN. `server.py` — thin Flask routes.
- Runtime data lives in `webui/data/` (git-ignored, machine-local).
