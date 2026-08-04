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

## Install on the Pi (offline)

```
# on the desktop (internet):
pip download flask -d wheels/
scp -r wheels nemo@<pi>:running_robot/

# on the Pi:
pip install -r requirements-rpi.txt          # if not already there (numpy, python-can)
pip install --no-index --find-links wheels/ flask
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

**2. Web UI — `/etc/systemd/system/dash01-webui.service`** (runs as `nemo`, waits for CAN):

```
[Unit]
Description=DASH-01 web control UI
After=network.target can-up.service
Wants=can-up.service

[Service]
User=nemo
WorkingDirectory=/home/nemo/running_robot
ExecStart=/home/nemo/running_robot/.venv/bin/python robot/fixed_gait/webui/server.py --port 8080
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Note `ExecStart` uses the venv's Python by absolute path — systemd runs no shell, so it won't
find `python` on `PATH` or activate a venv. Enable both:

```
sudo systemctl daemon-reload
sudo systemctl enable --now can-up.service dash01-webui.service
ip -details link show can0                     # confirm state UP, bitrate 1000000
systemctl status dash01-webui               # confirm active (running)
journalctl -u dash01-webui -f               # live logs
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
- `static/` — vanilla JS, no CDN. `server.py` — thin Flask routes.
- Runtime data lives in `webui/data/` (git-ignored, machine-local).
