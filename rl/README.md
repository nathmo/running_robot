# DASH-01 RL — train standing + walking from scratch

Teach the DASH-01 biped to balance and walk under joystick control, in MuJoCo, with
**Stable-Baselines3 PPO**. CPU-only — **no GPU required** (just a multi-core machine).

> **The one command to train walking** (run from the repo root, venv active):
> ```
> python -m rl.train --preset m2_walk --steps 8000000 --n-envs 6 --subproc --no-progress
> ```
> `m2_walk` learns to **balance and walk forward from scratch** (it also trains standing). Set
> `--n-envs` to about (CPU cores − 2). On a headless server keep `--no-progress` and use tmux.
> Full setup + all milestones below.

---

## 0. Requirements
- Python **3.10+** (tested 3.11/3.12), Git, a multi-core CPU. No GPU needed.
- The repo includes the robot CAD meshes (`robotCADdescription/`), so a normal clone is enough.
- ssh nemo@100.119.154.43

## 1. Get the code + create the environment

**Linux / macOS (typical training server):**
```bash
git clone https://github.com/nathmo/running_robot running_robot && cd running_robot
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build of torch
pip install -r rl/requirements.txt
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/nathmo/running_robot running_robot; cd running_robot
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r rl/requirements.txt
```

**All commands below assume: the venv is active and your working directory is the repo root.**
(The model and run paths are relative to the repo root.) If you prefer not to activate the venv,
prefix commands with the interpreter: `.venv/bin/python` (Linux) or `.venv\Scripts\python.exe`
(Windows).

## 2. Build + sanity-check the robot model
The simulatable model is generated from the CAD export (it is not committed pre-built):
```bash
python mujoco/dash01/build_model.py       # -> mujoco/dash01/dash01.xml
python mujoco/dash01/validate_model.py     # gate: closed loop holds, cam drives the knee, no blow-up
python -m rl.smoke_test                        # env sanity: obs/action shapes, a few steps, no NaN
```

## 3. Train — standing, then walking (each trains from scratch)
Pick `N` ≈ (CPU cores − 2). Times below are for an 8-core CPU at ~2.3k env-steps/s.

> **Tune `--n-envs` per machine — don't trust the `cores-2` rule blindly.** On a 20-core box
> (Ultra 7 265K, measured 2026-07), CPU throughput actually PEAKS around `n_envs=64-96`
> (~4.4-4.7k env-steps/s), not at `cores-2=18` (~2k env-steps/s) — more than 2x left on the table
> by following the old heuristic. Throughput keeps climbing well past the core count (subprocess/
> IPC overhead amortizes better with a bigger rollout batch) until it eventually regresses past
> ~100-128 envs. If retuning for new hardware, sweep a few `--n-envs` values with a short
> `--steps` run and watch `time_elapsed`/`total_timesteps` rather than assuming.

```bash
# M1 — stand / balance on the toe points  (sanity milestone; ~45-60 min)
python -m rl.train --preset m1_stand --steps 8000000 --n-envs 6 --subproc --no-progress

# M2 — stand + walk forward  (THE walking policy; also stands on command (0,0); ~60-90 min)
python -m rl.train --preset m2_walk  --steps 8000000 --n-envs 6 --subproc --no-progress

# M3 — full joystick: forward + turn  (optional next step; ~90 min)
python -m rl.train --preset m3_turn  --steps 10000000 --n-envs 6 --subproc --no-progress
```
- You do **not** need M1 before M2 — `m2_walk` learns balancing and walking together (20% of
  commands are "stand still"). M1 is just the simplest checkpoint to confirm the setup learns.
- Outputs go to `rl/runs/<preset>/`: periodic checkpoints `ppo_<steps>_steps.zip`, the matching
  `ppo_vecnormalize_<steps>_steps.pkl`, the final `final_model.zip` + `vecnormalize.pkl`, and
  TensorBoard logs.
- Drop `--subproc` to run envs serially (one process); drop `--no-progress` for a live progress bar.

**Resume an interrupted / extend a finished run:**
```bash
python -m rl.train --preset m2_walk --resume rl/runs/m2_walk/final_model.zip \
    --steps 4000000 --n-envs 6 --subproc --no-progress
```

**Headless Linux server (tmux):**

```
ssh nemo@100.119.154.43
```
```bash
tmux new -s train
source .venv/bin/activate
git reset --hard HEAD && git clean -fd && git pull
python -m rl.train --preset m2_walk --steps 64000000 --n-envs $(($(nproc)-2)) --subproc --no-progress
# detach: Ctrl-b then d   |   reattach: tmux attach -t train

scp -r nemo@100.119.154.43:running_robot/rl/runs/* "C:\Users\Nathann\Downloads\running_robot\rl\runs\"                                      



```

## 4. Monitor training

**Plots (no server needed) — written automatically.** Each run writes `progress.csv` and a
`training_plots.png` to `rl/runs/<name>/`, refreshed every ~500k steps and once more at the end.
The figure has six panels: episode reward (score), episode length, losses, `approx_kl` &
`clip_fraction`, policy `std` & `explained_variance`, and the **per-reward-term breakdown**
(`reward_terms/*`) — the last one shows *which* terms the policy is actually earning, so you can
spot reward gaming. On a headless server just copy the PNG down (or open it in place):
```bash
# regenerate the PNG from a run's progress.csv at any time (e.g. mid-run)
python -m rl.plot_training --run rl/runs/m2_walk
```

**TensorBoard (live, interactive).** Same scalars, including `reward_terms/*`:
```bash
python -m tensorboard.main --logdir rl/runs        # open http://localhost:6006
ssh -L 6006:localhost:6006 nemo@128.178.96.208 # on my host not the server

```
Watch **`rollout/ep_len_mean`** (climbs toward the 1000-step cap as it stops falling) and
**`rollout/ep_rew_mean`**. Expect a long flat early phase, then a sharp take-off. If
`train/clip_fraction` runs away above ~0.2 while reward stays flat, the policy is stuck — stop and
adjust the reward rather than training longer.

**Anti-gaming dashboard** (the terms that distinguish walking from skating):
`reward_terms/air_time` must climb **positive** (real strides), `reward_terms/foot_slip` and
`reward_terms/stance_time` must shrink toward 0, `train/std` should settle in 0.3–0.8 (2.1 = the
old entropy-farming failure; 0.14 = collapsed), and `curriculum/ent_coef` starts annealing only
once stepping has emerged. Mid-run, checkpoint-probe the gait:
`python -m rl.gait_probe --run rl/runs/m2_walk --checkpoint rl/runs/m2_walk/ppo_XXXXXX_steps`

## 5. Evaluate / watch / record
```bash
# GAIT PROBE — the acceptance gate. Reward curves can't tell walking from skating; this can.
# Measures duty factor, both-feet fraction, swing air times, in-contact toe slip, vx tracking,
# stand drift, RMS torque — and prints PASS/FAIL against the hardware gates.
python -m rl.gait_probe --run rl/runs/m2_walk

# metrics over several episodes (ep_len, command-tracking error)
python -m rl.evaluate --run rl/runs/m2_walk --episodes 5

# drive a fixed joystick command and record an mp4  (vx,yaw are normalized -1..1)
python -m rl.evaluate --run rl/runs/m2_walk --vx 0.5 --yaw 0.0 --video rl/runs/m2_walk/walk.mp4

# live 3D viewer (run locally, not over headless SSH)
python -m rl.evaluate --run rl/runs/m2_walk --viewer --vx 0.5
```
Evaluation auto-loads the matching VecNormalize stats; pass `--checkpoint rl/runs/.../ppo_XXXX_steps`
to evaluate a specific checkpoint instead of the final model. `--stochastic` samples actions the
way training does — check that gait metrics match the deterministic run before any hardware test.

---

## How it works (brief)
- **Model** (`mujoco/dash01/`, generated by `build_model.py`): free-floating base, the **6
  motors** (cam+thigh = CubeMars AKE90-8, hip-roll = AK60-39, real torque/inertia, and their
  **masses welded onto the stator bodies** — the CAD inertials omit them; total ~12.8 kg), the
  **closed parallel knee loop**, a passive **toe-down** ankle spring, an IMU + per-motor sensors,
  and a **point-toe contact** with realistic rubber-pad friction (condim=6: sliding + torsional +
  rolling all active; impratio=10 kills sub-friction-cone creep). The standing keyframe is settled
  UNDER GRAVITY, so it is the true loaded stance (knee ~157°, ~9 cm of leg-length authority).
  Link masses/inertias are still CAD placeholders until measured.
- **Observation** (`rl/env.py`, hardware-measurable, history-stacked): per-motor pos/vel/torque,
  IMU gravity direction + angular velocity, previous action, joystick command. No foot-contact
  sensor. Actions are applied with a fixed 1-step delay (Pi + CAN latency, plant truth).
- **Action**: 6 PD position targets (knee follows the linkage; ankle follows its spring).
- **Reward**: track commanded body-frame speed + integrated pose, stay upright, smoothness, alive
  — plus **gait shaping** (all sim-side, reward-only): **foot-slip penalty** (translation while
  grounded pays — the anti-skating term), **per-foot stance-time cap** (every foot must cycle
  when moving: no planted-feet drifting, no one-foot 'flamingo' perch), **swing clearance**
  (fresh swings above 2 cm pay immediately — the gradient that pulls the first steps out of
  standing), and a **one-sided air-time credit** (real strides pay, chatter earns zero instead of
  teaching 'never lift'). Reward normalization is OFF — raw scales are what PPO sees, and every
  penalty is capped so falling is never cheaper than living.
- **Hard rules (termination)**: fall (low/tilted), and **floor violation** — deep toe-sphere
  penetration or any foot/shin point below the floor kills the episode.
- **Config / all knobs**: `rl/config.py` (presets `m1_stand`, `m2_walk`, `m3_turn`).

## Base-DOF curriculum (simplified problem: `m1`..`m6`)

A separate, simpler track that **constrains the floating base** and asks the robot to run forward
**as fast as possible**, freeing one base DOF at a time. One shared model serves all of them — the
base has 6 explicit scalar joints `[X, Y, Z, roll, pitch, yaw]` (`build_model.py`) and 6
`<equality><joint>` locks; a preset's `base_lock` mask activates a subset at reset (`data.eq_active`),
a **rigid** rail (correct reaction forces → faithful leg torques for the treadmill/sim2real test).
`speed_mode` swaps command-tracking for a monotone `clip(vx, 0, v_ceiling)` reward while keeping every
anti-skating gait term. Set any mask you like; the ladder is just the intended path.

| preset | free base DOFs | notes |
|---|---|---|
| `m1` | X | rail; **Z locked at a per-episode RANDOM ride-height** (`z_rail_range`, seated from `ride_height_lut.npz`) so the policy adapts to any height. Treadmill/torque-validation milestone. |
| `m2` | X, Z | robot maintains its own height; straight-line max speed |
| `m3` | X, Z, pitch | |
| `m4` | X, Y, Z, pitch | Y free → `lat_vel` penalty keeps it straight |
| `m5` | X, Y, Z, roll, pitch | |
| `m6` | all 6 | fully free plant, max forward speed |

```bash
python -m rl.smoke_test --preset m1                 # locks hold, ride height varies
python -m rl.train --preset m1 --n-envs 6 --subproc --steps 8000000 --no-progress
python -m rl.gait_probe --run rl/runs/m1            # alternating gait + cam/thigh torque gate
```
Reachable ride-height band (measured, `measure_ride_band.py`): ~[0.81, 1.04] m, but deep crouches
need large fore/aft lunges and up to ~100 Nm holding torque, so the M1 default `z_rail_range` is the
moderate `(0.90, 1.03)`. **GPU/MJX height randomization is deferred** (CPU path gets exact
per-episode; MJX pins Z at the stand height for now).

### Fourier cyclic-gait policy (`m1_fourier`, `m2_fourier`)

Alternative policy **action representation** for the speed milestones (`action_mode="fourier"`): a
locomotion gait is cyclic, so instead of 6 per-step PD targets the policy emits, **once per gait
cycle**, a compact spec — cam+thigh **Fourier coefficients** (N=3 harmonics, symmetric mirror+antiphase),
a **learnable cadence**, and a **learned abduction (hip_roll) balance reflex** on roll/roll-rate that
runs at 50 Hz inside the cycle (feedforward CPG for propulsion + feedback for balance). `env.step`
becomes a macro-step = one cycle (~30/episode instead of 1000). Reconstruction math is in the
numpy-only `rl/fourier_gait.py` (Pi-shareable, like `fixed_gait/gait.py`). CPU-only for now (MJX raises
on fourier mode; parity gate still runs PD). The abduction reflex is inert while roll/Y are locked
(M1–M3) and activates at M4+.
```bash
python -m rl.smoke_test --preset m1_fourier         # obs 220, action 18, ~30 cycles/episode
python -m rl.train --preset m1_fourier --n-envs 8 --steps 800000 --no-progress
```
`evaluate.py` / `gait_probe.py` / `joystick.py` work unchanged on fourier runs: they sample frames,
gait metrics, viewer syncs and real-time pacing through the env's per-**control-step** hook
(`Dash01Env.on_control_step`), not per `env.step()` — a step() is a whole cycle in fourier mode, so
per-step() sampling strobed at cycle rate (one video frame per cycle, cycle-boundary-only metrics).
Any new rollout script must do the same.

### 100 m dash (`m1_sprint`, `m1_sprint_fourier`)

`sprint_mode`: one episode = one dash — start standing, run to a line `sprint_dist_m` ahead, STOP.
Reward = dense speed income while running (the speed_mode term) **plus a constant per-step clock
cost `w_time`**. The clock is what prices time: per-step vx alone integrates to w·distance no
matter the pace, while the clock integrates to −w_time·T (w_alive is 0 for the same reason —
a sprinter must not be paid per second of existence). At the line, the command channel the policy
observes flips [1,0]→[0,0] (its stop signal; [1,0] is exactly the speed_mode training distribution,
so a speed policy warm-starts cleanly), income switches to *be stationary* with a 5 m free braking
zone + overrun penalty, and coming to rest (|vx| ≤ 0.15 m/s held 1 s) ends the episode with
`finish_bonus`. The line starts at 25 m and ramps to 100 m (`sprint_curriculum_steps`) so slow
early policies still reach it and learn the stop; a `--resume` from a NON-sprint checkpoint
(e.g. `rl/runs/m1_fourier`) also gets the ramp.
```bash
python -m rl.train --preset m1_sprint_fourier --n-envs 8 --no-progress
# or warm-start from the walking policy:
python -m rl.train --preset m1_sprint_fourier --name m1_sprint_ft --resume rl/runs/m1_fourier/final_model.zip
python -m rl.evaluate --run rl/runs/m1_sprint_fourier --episodes 3   # prints per-dash line/stop times
```
`m2_sprint_fourier` is the same dash with the **height DOF free** (`base_lock=(0,1,0,1,1,1)`): no
ride-height rail, the height/vz reward terms come back on, and `term_height` is live — the Z-free
plant collapses in ~1.4 s under a passive hold (true of the existing `m2`/`m2_fourier` too), so the
policy must actively support its own height while sprinting. Warm-start it (base_lock doesn't change
the obs/action dims, so any m1 fourier checkpoint loads as-is):
```bash
python -m rl.train --preset m2_sprint_fourier --name m2_sprint_ft \
    --resume rl/runs/m1_sprint_fourier/final_model.zip --n-envs 8 --subproc --no-progress
```

## Milestones (legacy free-floating track: `m1_stand`/`m2_walk`/`m3_turn`)
| # | preset | goal | status |
|---|---|---|---|
| M0 | — | model sim-ready & validated | ✅ (rev2: +7.1 kg motor mass, loaded keyframe, condim 6) |
| M1 | `m1_stand` | balance on the toe points | ✅ on the rev1 model (retrain cheap if needed) |
| M2 | `m2_walk` | track forward velocity (+ stand), REAL steps (gait probe PASS) | v3 reward: ready to train |
| M3 | `m3_turn` | full joystick (forward + yaw) | next |
| M4 | — | domain randomization → ONNX export → Raspberry Pi (moteus) | later |

> **Why the v3 reward:** the v2 policy learned to **skate** — it scored ~1700 reward while both
> feet stayed grounded 57-66% of the time and all translation came from loaded-foot slip
> (~0.2 m/s); median "swing" was one control step, and under a +0.5 m/s command it actually
> moved at −0.09 m/s. Root causes fixed in v3: no term priced foot motion in contact (added:
> slip penalty), the air-time term could only PUNISH stepping (made one-sided + capped), the toe
> spheres were frictionless ball casters at condim=3 (fixed physics), the model was missing the
> 7.1 kg of motors, reward normalization (÷ return-std ≈ 110) had shrunk the fall penalty to
> −1.8, and ent_coef=0.015 on a clipped Gaussian farmed std up to 2.1 (bang-bang actions).
> Trust `rl/gait_probe.py`, not the reward curve.

## GPU training (MJX) — optional, server-only

An alternate training backend runs physics vectorized on GPU via **MuJoCo MJX**, with a
hand-rolled PPO loop built on Brax's low-level PPO primitives (not
`brax.training.agents.ppo.train.train()` — see `rl/mjx_train.py`'s docstring for why). It's a
**separate stack from everything above**: `rl/env.py`/`rl/train.py`/`rl/evaluate.py`/
`rl/gait_probe.py`/`rl/joystick.py` are all unchanged and remain the CPU/deployment source of
truth. MJX only replaces how the checkpoint gets *trained*; the exported checkpoint drops into
the same `rl/runs/<name>/final_model.zip` + `vecnormalize.pkl` shape and every eval/hardware tool
above works on it unmodified.

**Measured speedup on an RTX 4070 Ti Super (16 GB) / 20-core box: ~1.9x** against a *properly-
tuned* CPU baseline (9,024 env-steps/s at `n_envs=2048` on GPU vs. ~4,745 env-steps/s for CPU SB3
at `n_envs=80` on this same box — CPU throughput on this hardware peaks around `n_envs=64-96`, NOT
at the `cores-2` heuristic below, which under-uses this box badly: only ~1,981 steps/s at
`n_envs=18`. Sweep before trusting either number if the hardware changes again.) Given the extra
complexity of the GPU stack (a whole second training pipeline, hand-rolled PPO with documented
deviations from SB3's exact recipe), a ~1.9x gain is a real but modest case for reaching for it —
worth it for long runs where the difference compounds over hours, not obviously worth it for quick
experiments. `n_envs=4096` OOMs at 16 GB with the current `n_steps=1024` rollout length; 2048 is
near the practical ceiling without further memory tuning. There's also a one-time JIT-compile cost
(~5 min) paid once per training run start, not per iteration.

```bash
# one-time setup on the GPU server (separate from the CPU venv above -- same repo, extra packages):
pip install -U 'jax[cuda12]' mujoco-mjx brax     # CUDA 13.0 driver is backward-compatible w/ cuda12 wheels

# 0. physics parity gate -- confirms MJX handles this model's closed-loop knee + condim=6 contact
#    correctly on THIS machine before trusting any training run built on it. Read its docstring.
python mujoco/dash01/validate_mjx.py

# (optional) reward-math parity regression test -- rerun after touching rl/env.py or rl/mjx_env.py
python -m rl.mjx_env_test

# train (same presets/config as the CPU path -- rl/config.py is the single shared source of truth)
python -m rl.mjx_train --preset m2_walk --n-envs 2048 --steps 20000000

# export to the SB3 checkpoint format evaluate.py / gait_probe.py / joystick.py already expect
python -m rl.mjx_export --checkpoint rl/runs/m2_walk/final.pkl

# same acceptance gate as any CPU-trained run, completely unmodified:
python -m rl.gait_probe --run rl/runs/m2_walk
```

Known simplifications vs. the CPU/SB3 trainer (both deliberate scope cuts, documented in
`rl/mjx_train.py`'s docstring): minibatches are subsets of parallel envs rather than a flat
shuffle of (env, time) pairs (Brax's native `compute_ppo_loss` batching, not SB3's precompute-
advantages-then-SGD recipe), and there's no `target_kl` early-stopping within an epoch.
