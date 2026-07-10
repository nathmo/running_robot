# SpiderBot web control interface

One Flask app, served **on the robot's Raspberry Pi**, reachable from any phone/laptop on the
robot's WiFi hotspot. No internet needed at runtime — every asset is served locally.

```
python fixed_gait/webui/server.py                # real robot (socketcan can0/can1)
python fixed_gait/webui/server.py --mock         # simulated motors — try the UI on any machine
python fixed_gait/webui/server.py --port 8080 --host 0.0.0.0
```

Then browse to `http://<robot-ip>:8080/` (on the hotspot, the Pi's own address).

What it does (one page):

| Panel | Function |
|---|---|
| Telemetry | live raw/normalized position, current, temperature per motor + strip charts |
| Calibration wizard | **blocks all motion after boot** until zero pose + direction check are done |
| Manual control | per-actuator slider + number, per-actuator sine (A↔B, frequency), workspace-override checkbox |
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
python mujoco/spiderbot/gen_fk_lut.py --check
```

It Newton-solves the closed 4-bar at every (cam, thigh) grid cell via
`mujoco/spiderbot/plot_reachability.py` and stores all linkage node positions. The Pi only
bilinearly interpolates it. After loading/importing a workspace, press **“verify FK sign map”**
in the EE panel once: it tries all sign combinations of the normalized-degrees→model-radians map
against the workspace band (the thin 4-bar assembly band makes the right combo obvious) and
enables the EE display per leg only when the winner is decisive.

> **Known issue (found against `joint_limits_right_2.npz`)**: the recorded cam travel spans
> ~249° normalized while the model's cam joint range is only ±86° (`spidebot.xml`, a CAD guess —
> the real cam may be a full crank). Most workspace cells therefore fall outside the LUT grid for
> EVERY sign combo and the verdict is *not decisive* (best 17% coverage for `-1,-1`). Until the
> model's cam range / gearing vs the real motor is reconciled (or a workspace is recorded around
> the zero pose within ±86° cam), the EE panels stay off. `data/model_map.json` can be hand-edited
> to force signs + `verified` if you know them.

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
python fixed_gait/webui/server.py
```

Optional systemd unit (`/etc/systemd/system/spiderbot-webui.service`):

```
[Unit]
Description=SpiderBot web control UI
After=network.target

[Service]
User=nemo
WorkingDirectory=/home/nemo/running_robot
ExecStartPre=-/sbin/ip link set can0 up type can bitrate 1000000
ExecStartPre=-/sbin/ip link set can1 up type can bitrate 1000000
ExecStart=/usr/bin/python3 fixed_gait/webui/server.py --port 8080
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

(`sudo systemctl enable --now spiderbot-webui`; bringing CAN up needs root — either the
ExecStartPre above, or do it once at boot elsewhere.)

## First hardware bring-up (in this order)

1. Legs held by hand: press **E-STOP**, verify motors are limp; clear it.
2. Unplug one motor: the app must flag it silent and refuse all motion.
3. Run the calibration wizard for real; sanity-check a normalized angle with a protractor.
4. Import yesterday's workspace (`fixed_gait/joint_limits_right_2.npz`) in the workspace panel.
5. Press *verify FK sign map* — both legs should be decisive; the EE panels light up.
6. Small manual moves near zero with safety checks ON (override OFF).
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
