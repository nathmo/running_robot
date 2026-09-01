# `robot/deploy` — running a trained policy on DASH-01

Everything needed to take a `walk_mit` checkpoint and run it on the robot in force-control mode,
under a safety governor and a winding-temperature observer.

Nothing here has been run on hardware yet. All of it has been exercised offline: 42 unit tests, an
end-to-end dress rehearsal on the mock CAN bus, and a MuJoCo verification that the deployed control
law is bit-identical to the trained policy.

---

## Read this first

**`imp_m2_long` is an m2 policy.** `base_lock = [0, 1, 0, 1, 1, 1]` — in training, the base's Y,
roll, pitch and yaw were **railed**. The policy has never experienced those degrees of freedom, so
there is nothing in it that stabilises them. On a free-standing robot it is open-loop in pitch and
roll and it will fall over. It is a *gait* policy, not a *balance* policy.

Run it with the torso supported — a gantry, a boom, or a hoist. A fall after removing that support
is the expected consequence of removing it, not a surprise.

**The checkpoint is mid-training.** 204 M of a 400 M-step run, policy `std` still 0.999 (pinned at
its clamp). The mean action — the only thing the robot ever runs — evaluates fine in a fresh sim
env (`ep_len 6504 ± 4779` of 12000, peak vx 1.16 m/s over 5 greedy episodes), but a later
checkpoint from the same run should be better. Re-export when one lands; nothing else changes.

---

## Layout

| file | side | what |
|---|---|---|
| `export_policy.py` | desktop | trained run → one self-contained `.npz` bundle |
| `verify_export.py` | desktop | proves the numpy control law **is** the trained policy |
| `thermal_fit.py` | desktop | fits the thermal model; grades the experiment |
| `bundle.py` | both | the bundle format |
| `policy_net.py` | robot | the actor, in numpy — no torch |
| `fourier_gait.py` | robot | gait reconstruction, **byte-for-byte** from `walk_mit/fourier_gait.py` |
| `controller.py` | robot | the 200 Hz control law: obs → action → targets + impedance |
| `safety.py` | robot | clamp ladder and kill conditions |
| `thermal.py` | robot | two-node winding-temperature observer + torque budget |
| `mit.py` | robot | CubeMars force-control frames |
| `jointmap.py` | robot | MuJoCo joint frame ↔ motor frame |
| `thermal_calibrate.py` | robot | blocked-rotor thermal experiment |
| `run_policy.py` | robot | the runner |

Nothing in the *robot* column imports torch, mujoco or `walk_mit`. The Pi has none of them, and a
deployed control law should be reviewable without a training stack.

---

## Bring-up order

Each step is either read-only or has its own abort path. Do not skip ahead — every step exists
because the one after it is unsafe without it.

### 1. Verify the export (desktop, no robot)

```bash
python robot/deploy/export_policy.py --run walk_mit/runs/imp_m2_long \
    --out robot/deploy/bundles/imp_m2_long_204M.npz
python robot/deploy/verify_export.py --bundle robot/deploy/bundles/imp_m2_long_204M.npz \
    --run walk_mit/runs/imp_m2_long --steps 6000
python -m pytest robot/deploy/tests -q
```

`verify_export` must print PASS. Current result: observation, joint target, `kp` and `kd` all
**exactly 0.0** different over 6000 steps and 3 episode restarts; action within float32 noise
(3e-6).

### 2. Zero + direction calibration (robot, existing web UI)

The drives re-randomise their raw encoder origin on **every power cycle**, so this is not optional
and it is not durable. Finish the wizard; do not trust a calibration restored from disk.

### 3. Verify the joint map (robot, **limp**, moved by hand)

`jointmap.py` maps normalized motor degrees to MuJoCo joint radians. A sign error here drives a
balance correction the *wrong way* at 200 N·m/rad and is invisible in telemetry — the robot just
falls, and it looks like a bad policy.

The project has already paid for this twice: `webui/fklut.py` records that the captured zero pose
is ~135° of cam away from model zero, and the dynamic-ID work found the right leg had never been
mirrored at all. `fklut` verifies cam and thigh per side; **abduction has never been mapped by
anything.**

Two read-only checks, robot limp throughout:
- `check_stance` — hold the robot in the model's stance pose, confirm every joint reads what
  `default_motor_pos` says (tolerance 8°).
- `check_direction` — move each joint by hand in a named physical direction, confirm the model
  angle moves the way the model says.

`run_policy.py` **refuses to start** until every one of the six is marked verified.

### 4. Thermal calibration

Two experiments feed the same model, the same priors and the same derate. Do **either**; doing both
is better, because they pin different things.

Either way, measure two things first — they turn a poorly-conditioned fit into a well-conditioned
one, and neither needs the robot:
- **motor mass** (a scale) → pins `C_w` via copper mass × 385 J/kg/K
- **phase resistance** (a milliohm meter across two phases) → pins `k_cu` = 1.5 × R_phase

#### 4a. Burst campaign — the web UI panel (interactive, ~1 h)

The **Thermal identification** panel in the web UI. Pick a motor, declare how the rotor is held
(clamped joint → saturated unidirectional current; motor off the robot with nothing on the shaft
→ a position-mode sine that heats by fighting its own rotor inertia), set current and duration,
press start; the daemon excites that motor while holding the other five limp, then you read the
temperature peak with an external probe and type it in. A few bursts at different durations plus
one hand-recorded cooldown curve.

**Identify the joint before you heat it.** The panel's *Wiggle 5° for 2 s* button moves the
selected joint a few degrees while **holding the other five where they are**, then reports how far
every one of the six actually moved. Exactly one number should be large. This is the only check in
the repo that covers abduction, and it works in raw encoder degrees, so its verdict does not depend
on the calibration being right — which is what lets it be used to check one. It also catches an
inverted calibration sign, because a joint driven the wrong way tracks at twice the amplitude while
a joint that simply cannot move never exceeds one.

**The panel predicts the rise before you run it, and will refuse a burst it knows is pointless.**
This matters more than it sounds. The intuitive burst — 12 A for 10 s — deposits 216 J and moves
the case **0.2 °C**, which no handheld probe resolves. A readable 3 °C needs thousands of joules:
about 25 A for a minute. The panel also refuses the opposite mistake, because the same energy that
moves the case 3 °C moves the *copper* about 36 °C, in seconds, with no sensor watching — so bursts
are bounded by predicted **winding** rise, not by anything you can read.

Two things about the reading itself:
- **Do not read the peak early.** The burst heats copper; a case probe only sees it as that heat
  diffuses out, a minute or more later. The panel counts down and tells you when the earliest
  plausible peak is.
- **Record *when* you read it** (`peak read at … s`). The peak value alone constrains only the
  deposited energy; the *delay* to the peak is what carries the winding time constant, and that is
  the parameter no other part of this experiment can see.
- **Start each burst from rest.** The fit assumes both nodes begin at the temperature you typed. A
  second burst on a warm motor starts with hot copper the model believes is cold; the adequacy
  check flags it.

```bash
python robot/deploy/thermal_fit.py --campaign robot/fixed_gait/webui/data/thermal_runs.json \
    --motor left.thigh --motor-type AKE90-8 --motor-mass 0.85 --r-phase 0.10 \
    --out robot/deploy/thermal_params.json
```

Run `--campaign-self-test` first: it synthesises a campaign from known parameters, quantises the
readings the way a person with an IR gun would (0.5 °C, peak read ~30 s late, sometimes early), and
checks the fit recovers the model and deploys a conservative rating.

#### 4b. Blocked-rotor hold — the offline script (~2.5 h per motor type)

```bash
python robot/deploy/thermal_calibrate.py --motor left.thigh --mock --steps 6:20,0:60   # rehearse
sudo systemctl stop runningrobot-webui.service
python robot/deploy/thermal_calibrate.py --motor left.thigh --joint-is-blocked \
    --steps 7:4000,0:4600
python robot/deploy/thermal_fit.py --logs .../thermal_left_thigh_*.npz \
    --motor-type AKE90-8 --motor-mass 0.85 --r-phase 0.10
```

**The 2.5 hours is not padding.** The self-test measures what a short experiment costs: a 3-minute
hold fits the data to 0.09 °C rms *and still gets the continuous current rating 31 % low*, because
the case time constant is ~25 min and a short hold does not measure the plateau, it extrapolates to
one.

#### Which pins what

| | blocked-rotor hold | burst campaign |
|---|---|---|
| steady-state gain → continuous rating | ✅ (needs ≥2×τ_case) | ✗ (extrapolated) |
| case time constant τ_c | ✅ from the cooldown | ✅ from the cooldown curve |
| winding time constant τ_w | weakly | ✅ from the timed peak delay |
| operator effort | start it and leave | present for every burst |
| commutation | one phase heats | all three, like the model assumes |

Both fitters **grade the experiment, not the residual**, and mark the parameters uncalibrated when
the experiment could not have pinned them. `MotorThermalModel` then refuses to arm a policy run —
that refusal is the point.

### 5. Dress rehearsal (desktop or Pi, mock bus)

```bash
python robot/deploy/run_policy.py --bundle .../imp_m2_long_204M.npz --mock --no-imu \
    --allow-uncalibrated-thermal --skip-jointmap-check --max-seconds 4 --v-cmd 0.5
```

### 6. First real run — **stand still, supported**

```bash
sudo systemctl stop runningrobot-webui.service
python robot/deploy/run_policy.py --bundle .../imp_m2_long_204M.npz \
    --jointmap robot/deploy/deploy_map.json --thermal robot/deploy/thermal_params.json \
    --drive-amp-limit 12 --v-cmd 0.0 --max-seconds 10 --deadman-file /tmp/dash_deadman
```

`--v-cmd 0.0` is the stand-still command, which the policy was explicitly trained on (25 % of
training draws were exactly zero). Walk forward only after standing is boring.

Keep `/tmp/dash_deadman` fresh from another shell (`while true; do touch /tmp/dash_deadman; sleep
0.2; done`); stop touching it and the robot soft-stops within 0.5 s.

---

## What the safety layer actually does

Two different answers to two different problems:

- **Momentary** overshoot — a target a few degrees past a limit, a torque spike on a footfall — is
  **clamped**, and counted. Killing the run for this drops a standing robot on the floor.
- **Persistent** demand — the same limit saturated for 40 consecutive ticks (0.2 s) — **kills**.
  That is not a transient; it is the policy asking for a machine it does not have, and continuing
  means running a control law that is no longer the one that was verified.

The ladder, in order: sanity (non-finite → hard kill) → position → rate → torque → gains →
watchdogs.

The torque cap bounds the **position error**, not the gains: `kp` and `kd` are what the policy
chose and what the simulator ran, so scaling them changes the closed-loop dynamics into something
untested. Bounding the error only saturates the command, exactly as a torque-limited actuator does
— which the simulator also did, via `forcerange`.

| kill | flavour | why |
|---|---|---|
| non-finite command or measurement | hard | a NaN on the wire means *something*, and nobody chose it |
| fallen (`grav_z` past `term_gravity_z`) | hard | holding a stance target now just drives the legs into the floor |
| body rate > 12 rad/s | hard | |
| telemetry stale > 50 ms (per motor) | hard | never command a joint you cannot see |
| drive error flag / case ≥ 80 °C | hard | |
| any limit clamped 0.2 s | soft | put it down under control |
| estimated winding ≥ trip | soft | |
| dead-man not refreshed | soft | |

Soft = freeze the target, bleed `kp`/`kd` to zero over 0.3 s, then limp. Hard = zero gains
immediately.

---

## Force control, and the three rules `mit.py` enforces

The AK drives speak force control **today**, no reflash — but not the classic MIT protocol. It is
an extended-id VESC-style command 8, byte order **kp first**, and every `mit_*.py` in `robot/tools`
implements the wrong one.

1. **DLC is always 8.** A 3-byte frame to `0x86A` drew **62.5 A** into a stalled rotor on
   2026-08-26. The missing bytes are not unset — they read as zero, and zero in the position field
   is the range *minimum* (−12.56 rad). `pack()` has no code path to a short frame.
2. **Velocity and torque are zero** unless their ranges are identified. The V/T spans are
   per-model firmware constants and the AKE90-8 and AK60-39 are not in the manual's table. Zero is
   the one value immune to a wrong span — it encodes to the mid-code and decodes to exactly 0.0
   for any span. With `v_des = 0` and `tau_ff = 0` the drive runs `tau = kp·(p_des − p) − kd·v`,
   which is **precisely** the MuJoCo position actuator the policy trained against.
3. **Position, kp and kd are clamped and the clamp is reported.**

A nice consequence, checked by a test: the impedance channel spans `kp` 40–500 N·m/rad and `kd`
1.0–5.0 N·m·s/rad, which fits the drive's 0–500 / 0–5 wire ranges *exactly*. Nothing has to be
rescaled to be commandable.

---

## Known gaps

- **ERPM → joint rad/s is unmeasured.** Nothing in this repo converts it (the only number that
  exists is a made-up constant inside `canio`'s mock). The runner uses differentiated position
  instead — noisy but unambiguous, and inside the noise band the policy trained against
  (0.15 rad/s + 0.05 bias). `jointmap.fit_erpm_scale` will fit the real scale from a hand-driven
  sweep when you want it.
- **`Kt` is the datasheet value derated by the measured 0.85 gearbox efficiency**, not a direct
  measurement. It sets the torque channel of the observation; 15 % here is roughly twice the
  `noise_torque_gain` the policy trained against.
- **Not integrated into the webui daemon.** The runner is standalone and refuses to start while
  the daemon is running. Integrating it as a daemon mode would put the policy behind the web
  E-stop and the black box — worth doing after the standalone has been exercised, not before.
- **Thermal `t_trip` = 120 °C is a placeholder** until the insulation class is known.
