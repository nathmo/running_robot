# fixed_gait — hard-coded straight-walking gait (base fixed in the air)

A hand-authored (no RL) walking gait for SpiderBot, meant to run with the **body clamped in a rig /
fixed in space** so the legs cycle in the air. It exists to prove we can command all six motors in
a coordinated walking pattern. There is no balance loop — nothing here keeps the robot upright.

The **same gait** (`gait.py`) drives two front-ends:

| you want to… | run |
| --- | --- |
| watch it in 3D on your computer | `sim_fixed_base.py` (MuJoCo, base welded in the air) |
| run it on the real motors (Pi) | `run_hardware.py` (CAN, servo-mode position) |

`gait.py` is **pure numpy** so it runs unchanged on the Raspberry Pi (whose runtime is
onnxruntime / numpy / python-can only). Only the sim and the validator import MuJoCo.

---

## Motor map

`can0` = **RIGHT** leg, `can1` = **LEFT** leg. CubeMars IDs on each bus:

| joint | role | motor ID | left bus | right bus | sim actuator |
| --- | --- | --- | --- | --- | --- |
| abduction | hip roll (lateral) — held at home for straight walking | **104** | can1 | can0 | `hip_roll_{L,R}` |
| cam | drives the knee through the parallel pushrod loop (leg extend/retract) | **105** | can1 | can0 | `cam_{L,R}` |
| hip | fore/aft thigh swing (the visible stepping motion) | **106** | can1 | can0 | `thigh_{L,R}` |

The knee and ankle are **passive** (parallel linkage + spring); they follow the cam and are not
commanded. The right leg is the geometric mirror of the left, so its joint signs are negated — the
gait handles this internally.

---

## Run it in simulation

```bash
# interactive 3D viewer (base fixed in the air, mesh self-collision on, gravity off)
.venv/Scripts/python.exe fixed_gait/sim_fixed_base.py

# tweak the gait live
.venv/Scripts/python.exe fixed_gait/sim_fixed_base.py --period 2.0 --thigh-amp 0.3 --cam-amp 0.15

# render a video instead of the live window
.venv/Scripts/python.exe fixed_gait/sim_fixed_base.py --video walk.mp4 --duration 6

# options: --gravity (legs load), --floor (keep ground), --no-collide (faster, less faithful),
#          --speed 0.5 (slow-mo)
```

Mouse orbits/zooms; space pauses. The base is welded to the world, so the torso stays put and you
watch the legs step.

## Validate against the reachable workspace

```bash
.venv/Scripts/python.exe fixed_gait/validate_gait.py --out fixed_gait/_gait_reachability.png
```

Drives the gait in a fixed-base sim **with the leg meshes made collidable** (the normal model only
collides the foot spheres) and checks: every joint stays in range, no *deep* link self-collision,
and the foot path is smooth. It overlays that foot path on the reachability map from
`mujoco/spiderbot/plot_reachability.py`. The gait's foot path is a small closed loop sitting at the
forward edge of the reachable band — see `_gait_reachability.png`.

> The pushrod grazes its adjacent hip link by ~2 mm at the top of the cam swing. That is the
> parallel linkage running right alongside the hip *by design* (it is the least-grazing operating
> point — lower cam centers graze the thigh more), so it is reported as benign, not a jam.

---

## Run it on the robot (Raspberry Pi)

Install the runtime and bring up both CAN buses at 1 Mbps:

```bash
pip install -r requirements-rpi.txt
sudo ip link set can0 up type can bitrate 1000000     # RIGHT leg
sudo ip link set can1 up type can bitrate 1000000     # LEFT leg
```

**Home = captured at startup.** At launch each motor's current position is read and taken as that
joint's home (the gait center pose), and the gait adds offsets on top — so the first command equals
where the motor already is (no jump). **Pose the legs in the nominal standing pose before you
start.** Direct 1:1 coupling is assumed (joint rad → motor deg × 180/π).

### Bring-up order (do this the first time / after any rewiring)

1. **Confirm every motor answers** and see the captured home positions:
   ```bash
   python fixed_gait/run_hardware.py --settle-only --duration 3
   ```
   It reads all six, prints their home angles, then just holds the center pose. If any motor is
   silent it aborts before commanding anything.

2. **Check each joint's direction** (once per joint) and set its sign in the `CALIB` table:
   ```bash
   python fixed_gait/run_hardware.py --test-joint thigh_L --deg 8
   ```
   The joint jogs +8° from home and back. If it moves the **wrong way**, flip that joint's `sign`
   (`+1.0` ↔ `-1.0`) in `CALIB` near the top of `run_hardware.py`. Repeat for all six
   (`hip_roll_L/R`, `cam_L/R`, `thigh_L/R`). For straight walking the abduction joints don't move,
   but check them anyway.

3. **First full run at reduced amplitude**, ramping up as you gain confidence:
   ```bash
   python fixed_gait/run_hardware.py --amp-scale 0.3      # 30%
   python fixed_gait/run_hardware.py --amp-scale 0.6      # 60%
   python fixed_gait/run_hardware.py                      # full
   ```
   Watch the `cam` motors' current near the top of their swing (the pushrod-graze zone). If it
   climbs, the parts are binding harder than the sim's 2 mm — lower `cam_amp` / `cam_center` in
   `gait.py`.

Ctrl+C at any time — motors are released (`SET_CURRENT 0`, limp) on exit.

### Safety built in

- **No blind commands**: aborts if any motor doesn't report a start position.
- **No startup jump**: the first command equals the captured home (gait starts at center, offset 0)
  and amplitudes soft-start over `ramp_s`.
- **Offset clamp**: never commands more than `MAX_OFFSET_DEG` (30°) from home.
- **Tracking cut**: if a motor's actual position strays > `MAX_TRACK_ERR_DEG` (25°) from its
  command, it cuts (a motor hitting a mechanical stop trips this).
- **Error / over-temp cut**: any motor error flag or temp ≥ 80 °C stops and releases.

The CAN protocol (servo-mode `SET_POS`, big-endian int32 × 10000; `SET_CURRENT 0` to release) is
copied verbatim from the tested `tools/ak_servo_sweep.py`.

---

---

## Record & replay your own gait (teach mode)

Instead of the analytic gait above, you can **move the leg by hand** to teach a trajectory, then
play it back — starting slow and gentle, then faster/stronger. This path is safer for first
motion: the recorder never drives the motors, and the player has a hard torque (current) cap.

### 1. Record (motors stay limp — you backdrive them)

```bash
python fixed_gait/record_trajectory.py --leg right     # RIGHT leg (can0), a few takes
python fixed_gait/record_trajectory.py --leg left      # LEFT  leg (can1), a few takes
```

The three motors are held **limp** (`SET_CURRENT 0`) so you can move them by hand while positions
are logged. In the terminal:

- **SPACE** — start / stop a take. Move the leg through **one full cycle** (start pose → step →
  back to start). Do a few takes; they're averaged.
- **u** — undo the last take, **q** — finish.

You don't need to be precise — being off by a few cm/deg on the return is fine, the loop is
**closed for you**. The **abduction** motor (id 104) doesn't need to move; it's held fixed and set
separately at playback. Record the **right leg first, then the left**.

When both legs are recorded it **auto-smooths and exports**:
`trajectories/gait_recorded.npz` (+ a `gait_recorded.png` preview). Re-smooth anytime without
recording: `python fixed_gait/record_trajectory.py --process-only`.

**What the smoothing does** (`trajectory.py`): resamples each take to one cycle, auto-aligns their
phase, detects the left↔right **mirror sign**, averages them into **one** canonical shape, closes
the loop (FFT low-pass → exactly periodic), and stores each side's **offset (its own 0°), sign, and
travel range** separately. Playback then runs the *same* shape on both legs, 180° apart, each
mapped into its own motor frame and clipped to its recorded range.

### 2. Play it back (tunable PID, torque-limited, at your speed)

```bash
python fixed_gait/play_trajectory.py --dry-run                                  # print targets, 0 A
python fixed_gait/play_trajectory.py --period 8 --current-limit 3 --kp 0.8 --ki 0.4 --log
python fixed_gait/play_trajectory.py --period 4 --current-limit 6 --kp 1.5 --ki 0.8 --kd 0.03
python fixed_gait/play_trajectory.py --abduction-right 5 --abduction-left -3    # set abduction hold
```

Each motor runs a software **PID** loop whose output current is **hard-clamped to
`--current-limit`** (Amps — Kt/Nm isn't known exactly), so torque can never exceed it:

```
current = kp·err + ki·∫err + kd·(target_vel − actual_vel),   clamped to ±limit
```

- **`--kp`** — stiffness. *This is the "stricter tracking" knob.* Too low and the leg lags, then
  friction makes it stick-slip (jagged/twitchy) — which is what you saw (only ~2 A pulled even at a
  20 A limit). Raise `kp` until tracking is crisp.
- **`--ki`** — removes the steady lag from gravity/friction (integral, with anti-windup). No gravity
  feedforward yet; the integral does that job.
- **`--kd`** — damping; uses `target_vel − actual_vel` so it tracks the *moving* trajectory instead
  of braking against it. Add a little only if it oscillates.
- **`--period`** seconds per cycle (bigger = slower). **`--log`** saves a target-vs-actual +
  current plot (`trajectories/last_run.png/.npz`) — run it, look at the lag, adjust gains, repeat.

**Speed governor (no more runaway crashes).** Instead of hard-cutting at 5000 ERPM (which tripped
on a normal fast move), the controller now tapers the *accelerating* current as a motor nears
`--speed-limit` (default 9000 ERPM), so speed **saturates** there smoothly — braking current is
never limited. `--max-speed` (default 16000) is only a last-resort runaway net, well above the
governor. Raise `--speed-limit` to allow faster moves, lower it to keep things gentle; `--speed-limit 0`
disables the governor. The live readout shows `maxSpd` so you can see where you're running.

**Tuning recipe:** start `--kp 0.4 --ki 0 --kd 0 --period 10 --current-limit 3`; raise `kp` until
tracking is tight without buzzing; add `ki` to kill the remaining lag; add a touch of `kd` only if
it oscillates; then shorten `--period` and raise `--current-limit` for speed. Both legs run dephased
180°, soft-start from the current pose, release on Ctrl+C; guards cut on runaway/over-temp/error.

### Inspect a trajectory (PNG + live animation)

```bash
python fixed_gait/view_trajectory.py                 # save trajectories/trajectory.png (headless)
python fixed_gait/view_trajectory.py --live          # animated window (needs a display / X-forwarding)
python fixed_gait/view_trajectory.py --anim walk.gif # save an animated GIF (works headless)
```

The PNG shows cam & hip vs phase for both legs plus the cam–hip loop; the animation adds a moving
phase marker and a schematic stick figure (angles are real, leg geometry is approximate).

---

## Files

| file | what |
| --- | --- |
| `gait.py` | the analytic gait — pure numpy, no MuJoCo. `GaitGenerator.targets(t)` → 6 joint targets (rad). |
| `sim_fixed_base.py` | MuJoCo viewer / video; welds the base, removes floor, drives the gait. Also exports `build_fixed_base_model()`. |
| `run_hardware.py` | CAN streamer (position/servo mode) for the analytic gait; `CALIB` table + safety. |
| `validate_gait.py` | sim-based safety check + reachability-map overlay. |
| `record_trajectory.py` | **teach recorder** — backdrive a leg by hand, SPACE-toggled takes; auto-smooths + exports. |
| `trajectory.py` | pure-numpy smoothing/fusion (align, mirror, close-loop) + per-side calibration; shared by record & play. |
| `play_trajectory.py` | **replay** a recorded trajectory, both legs dephased 180°; tunable current-**PID** with an Amp torque cap, `--log` tracking plot. |
| `view_trajectory.py` | inspect a trajectory: static PNG + live/animated visualization (no hardware). |
| `trajectories/` | recorded takes + exported `gait_recorded.npz` (+ preview png, tracking logs). |
| `_gait_reachability.png` | generated: gait foot path over the reachable workspace. |

## Tuning the gait

Edit `GaitParams` in `gait.py` (or pass `--period/--thigh-amp/--cam-amp/...` to the sim):

- `period_s` — seconds per stride (bigger = slower).
- `thigh_amp` — fore/aft step size (the obvious motion).
- `cam_amp` — knee lift/plant over the cycle.
- `thigh_center`, `cam_center` — the mid-stance pose. `cam_center` is kept high (~0.6) on purpose:
  below ~0.3 with the thigh forward the 4-bar crosses its dead-center and the foot folds. Keep the
  cam swing clear of that.

The reachability study found the foot workspace is a **thin diagonal band** (the 2-DOF Jacobian is
near-singular everywhere — a fat foot ellipse is mechanically impossible), so the gait sweeps the
foot back and forth along that band; running thigh and cam a quarter-cycle apart opens the small
swing/stance loop the band allows.
```
