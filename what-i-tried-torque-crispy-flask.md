# Parameter definition for a unified RL robot-control framework

## Context

Why this exists: the experiments have sprawled across **two disconnected codebases** and a pile of one-off feature flags, so it is hard to see what was tried, what actually worked, and hard to add a new robot.

- The **pendulum** sim2real study lives only on the `pendulum_sim2real` branch (old deleted `RL/` layout): 1-DOF inverted pendulum, direct-torque control, `[pos, vel, torque]` hardware-style obs, SB3 PPO → ONNX → moteus at 50 Hz. It reached sim2real (runs on the Pi).
- The **DASH-01 "running robot"** lives in the current `main` tree under `rl/`: 6 actuated joints + parallel linkage, PD-position control, big stacked obs, SB3 PPO (+ optional MJX GPU) → ONNX → moteus. It has reached balance (M1) but **not** honest walking yet (M2 skated / floor-clipped).
- Between them, capability was bolted on as mutually-aware flags: `action_mode` (pd/fourier), `speed_mode`, `sprint_mode`, `base_lock` masks, and ~15 preset factories in `rl/config.py`. Adding the next idea means touching all of them.

**Goal of the eventual framework:** one configuration space that expresses *everything from the pendulum to the untethered runner*, where control mode, DOF restriction, reward, curriculum, network, backend and deploy target are all per-experiment parameters — and adding a new robot is filling in a config, not forking the code.

**Goal of THIS task (only):** (1) remind us of every variant already tried, and (2) **define the parameter space** — the axes we are actually trying to sweep. No framework is built here; no schema/YAML is committed. This is a prose taxonomy to agree on before any refactor.

---

## Part 1 — Inventory: everything we have tried

Status legend: ✅ works · ⚠️ tried, failed/incomplete · 🔬 experimental · ⛔ attempted & reverted.

### A. Robots / plants
| Plant | DOF | Where | Status |
|---|---|---|---|
| Inverted pendulum | 1 hinge, base fixed | `pendulum_sim2real` branch, `RL/assets/inverted_pendulum.xml` | ✅ sim2real (Pi) |
| DASH-01 biped | 6 actuated + passive knee/pushrod + spring ankle | `mujoco/dash01/dash01.xml` (gen by `build_model.py`) | ✅ model, M1 balance |

### B. Controllers / actuation (how the 6/1 targets are produced)
| Variant | Where | Status |
|---|---|---|
| **Torque / current direct** (policy → Nm, clipped) | pendulum env; HW `--mode current` (`fixed_gait/play_trajectory.py`) | ✅ pendulum; ✅ HW (no gravity FF) |
| **Position PD** (policy → target angle, motor kp/kv tracks) | `rl/env.py:385`, `dash01.xml:141-147`; HW `--mode position` | ✅ primary DASH-01 path |
| **Software PID → current** (kp·e+ki·∫+kd·ė, torque-capped) | `fixed_gait/play_trajectory.py:11-15`, `webui/daemon.py:669` | ✅ HW |
| **Pattern generator (Fourier CPG)** — per-cycle cam+thigh Fourier coeffs + learnable cadence + hip reflex | `rl/fourier_gait.py`, `rl/env.py:432-484` (`action_mode="fourier"`) | 🔬 CPU-only, walks slowly |
| **Passive spring+damper+friction** (ankle) | `build_model.py:100-107`; stiffness 28.65, springref ±0.7 | ✅ (2.27 Nm breakaway preload NOT modeled) |
| **Fixed sinusoidal open-loop gait** | `fixed_gait/gait.py` | ✅ demonstrator (base in air) |
| Analytical spring-damper foot contact | `rl/mjx_env.py:30-35` | ⛔ reverted (unstable vs closed loop) |

### C. DOF restriction
- Base `<freejoint>` replaced by **6 lockable scalar joints** (`base_x/y/z` slide, `base_roll/pitch/yaw` hinge) + 6 `<equality><joint>` locks, activated per-episode via `data.eq_active[...] = cfg.base_lock` (`build_model.py:208-277`, `rl/env.py:287-288`). ✅
- **M1 ride-height rail**: Z locked at a per-episode random height from `ride_height_lut.npz` (`rl/env.py:289-301`). ✅
- Pendulum: base fully fixed (degenerate all-locked case). ✅

### D. Reward objectives (each is "a reward mode")
| Objective | Where | Status |
|---|---|---|
| Upright-hold (angle/vel/effort/alive/stable/success terms) | pendulum env `_reward_breakdown` | ✅ |
| Command-tracking v3 (track_vx/yaw, progress, pose/heading, + gait shaping) | `rl/env.py:606-649`, gait `:495-553` | ✅ balance; ⚠️ walk skated |
| `speed_mode` (monotone clip(vx,0,ceiling), tracking off) | `rl/env.py:597-605` | 🔬 |
| `sprint_mode` (speed income − per-step clock cost + stop phase + finish bonus) | `rl/env.py:575-596` | 🔬 not trained for real |
| Anti-skate gait shaping (foot-slip, air-time, stance-cap, clearance) | `rl/env.py:495-553` | ✅ added post-skating post-mortem |

### E. Curriculum types tried
- **Init-state span widening** (pendulum: start angle ramps to ±45°).
- **DOF-unlock ladder** (m1→m6 free one base DOF each, `rl/config.py:276-304`).
- **Command-magnitude ramp** (`CmdVxFracCurriculum`, `rl/train.py:58-79`).
- **Sprint-distance ramp** 25→100 m (`SprintDistCurriculum`, `rl/train.py:91-113`).
- **Competence-gated entropy anneal** (`EntropyCallback`, `rl/train.py:121-178`).
- **Ride-height randomization range** (M1).

### F. Observation designs
- Pendulum: raw HW telemetry `[pos_turns, vel_turns_s, torque_nm]`.
- DASH-01: motor pos/vel/torque + IMU accel/gyro + gravity dir + prev action + command, **history_len=5** (`rl/config.py:51-54`); true base lin-vel deliberately **excluded** (privileged/critic-only intent); no foot-contact sensor.

### G. Network / RL algorithm
- **SB3 PPO** `MlpPolicy`, `net_arch=[256,256]` tanh, state-independent log_std, CPU (`rl/config.py:215-238`, `rl/train.py`).
- **Hand-rolled Brax PPO** on MJX, networks built to map 1:1 onto SB3 weights (`rl/mjx_train.py:55-68`). 🔬 GPU.

### H. Backends / performance
- CPU SB3 (`SubprocVecEnv`, `n_envs≈cores−2`) — primary/deploy. ✅
- MJX/JAX GPU (`n_envs` 2048–4096, ~1.9× on RTX 4070TiS) — **pd-only**, raises `NotImplementedError` on fourier/sprint and defers per-episode height randomization (`rl/mjx_env.py:302-393`). 🔬

### I. Evaluation / visualization
- `rl/evaluate.py` (mp4 + live viewer + fixed `--vx/--yaw` command), `rl/joystick.py` (live WASD steering), `rl/gait_probe.py` (PASS/FAIL skating gate), `rl/plot_training.py` (6-panel PNG from `progress.csv`). ✅ All route through the per-control-step hook so they work in both pd and fourier modes.

### J. Sim2real / deploy
- ONNX export with VecNormalize baked in (pendulum `export_pendulum_onnx.py`; DASH-01 via `mjx_export.py` path). moteus/CAN at 50 Hz. Pendulum ✅ real; DASH-01 deploy path built, gait not ready.

### K. Sim2sim / workspace safety
- Termination/safety constraints as implicit "second plant": floor-penetration tolerance, tilt limit, torque-above-baseline, toe-only contact (`rl/env.py` `_floor_violation`).
- HW **current-mode torque model** (`fixed_gait/webui/canio.py:195`) — a crude second simulator.
- No explicit sim2sim harness yet (running a trained policy through a higher-fidelity/safety-constrained sim as a gate).

**Honest status line:** pendulum = full loop closed (sim→train→ONNX→real). DASH-01 = balance solved honestly; walking not yet (skating + foot-clip failures diagnosed, v3 reward + fourier + sprint are the open bets). Two codebases, no shared config.

---

## Part 2 — The parameter space (the actual deliverable)

These are the **axes an experiment configures**. Every past variant above is one point in this space; the pendulum and the untethered runner are the two extreme corners. Prose definition, grouped into 11 parameter families.

### 1. Robot / plant
Which model, and its structural facts the framework must read (not choose): number of joints, which are **actuated** vs **passive-linkage** (knee driven by the loop) vs **passive-spring** (ankle), closed-loop equality constraints, contact geometry. Span: pendulum = 1 actuated joint, no loops; DASH-01 = 6 actuated + 2 loop knees + 2 spring ankles. *A new robot enters the framework by supplying a model + this structural descriptor.*

### 2. Per-joint controller (the heart of the request)
Each **actuated** joint independently selects one controller type. This generalizes today's single global `action_mode` to a per-joint choice. The controller types are:
- **Locked** — joint welded; no policy output (a joint you choose not to drive).
- **Passive spring+damper+friction** — no policy output; parameters: stiffness, damping, Coulomb/viscous friction, spring rest angle, breakaway preload.
- **Position (PD)** — policy emits a target angle; motor loop tracks it; parameters: kp, kv, action_scale about nominal, target filter, actuation delay, ctrl range.
- **PID** — software PID → current/torque; parameters: kp/ki/kd, integral clamp, torque/current cap, optional gravity feedforward.
- **Torque / current (direct)** — policy emits Nm; parameters: torque scale, force/current cap.
- **Pattern generator (Fourier / CPG)** — policy emits, per gait cycle, a set of **Fourier weights ("number of weights" = n_harmonics)** + cadence; reconstructed to a per-step target; parameters: n_harmonics, amplitude bound, frequency range, and *which signal it drives* (position or torque).
- **Reflex / feedback law** — a small (optionally learned) linear controller on a chosen sensor (e.g. hip_roll on roll & roll-rate); parameters: input signals, gains (fixed or learned), bias.

**Mixing rule:** controllers are chosen per joint and may differ within one experiment (torque on cam_L, pattern-gen on thigh_L, PID on hip). The action vector the policy emits is the concatenation of whatever the chosen controllers need.

### 3. Symmetry / coupling groups
A separate layer on top of §2 so we don't pay for symmetry we know exists. Joints may be tied into a **group** where one policy output drives several joints through a **transform**: sign flip (mirror), constant offset, and **phase offset** (the antiphase gait: `right = mirror(left, φ+π)` as in `fourier_gait.py`). Parameters per group: member joints, and per-member transform (sign, offset, phase). Default = every joint independent (no coupling); the pendulum has one joint so this is inert.

### 4. Per-DOF restriction (segment mobility)
Independent of actuation. Each DOF (base's 6, and in principle any joint) selects: **free**, **locked-at-value** (rigid equality), **rail-randomized** (locked at a per-episode sampled value, e.g. ride height), or **soft-limited** (range). Parameters: the lock target(s) and, for rail-randomized, the sampling range. Span: pendulum = base all 6 locked; DASH-01 M1..M6 = the unlock ladder; treadmill = only forward+vertical free.

### 5. Observation
Which features the policy sees, from a menu: motor pos/vel/torque, IMU accel/gyro, gravity direction, previous action, task command, (privileged, critic-only) true base velocity. Parameters: feature set, **history length**, per-feature noise/scale model, and the privileged/asymmetric split. Span: pendulum `[pos,vel,torque]`; DASH-01 stacked history of the full set. *Deliberately matches hardware telemetry to keep sim2real honest.*

### 6. Task / command input
The command the policy is conditioned on (and, downstream, how you steer it in the viewer). Parameters: command dimensionality and ranges, injection convention. Span: pendulum = none; DASH-01 = `[forward, yaw-rate]` joystick with `(0,0)`=stand; sprint = a latched command flip as the "stop" signal.

### 7. Reward (defined per experiment)
A reward is a **named module = a set of terms + weights**, selected per experiment (the user explicitly wants an editable reward script per experiment). Parameters: which term set (upright-hold / command-tracking / speed / sprint / gait-shaping), per-term weights, penalty floors, and gating predicates (e.g. gait terms only above a speed). The framework's job is to make swapping the reward module a config choice, not an env edit, and to always log the per-term breakdown (both codebases already track this).

### 8. Curriculum (defined per experiment)
An ordered list of **stages**, each = a *trigger* + a *change*. Trigger types seen: elapsed steps, or a competence metric (entropy anneal keyed off emergent stepping). Change types seen: widen init-state span, unlock a DOF, ramp command magnitude, ramp sprint distance, widen ride-height range. Parameters: the stage list with per-stage trigger and target. Default = no curriculum (single stage).

### 9. Network / RL algorithm
Per-experiment policy + optimizer config. Parameters: hidden sizes & activation, log_std handling/clamp, and PPO hyperparameters (n_steps, batch, epochs, gamma, gae_lambda, clip range, entropy schedule, target_kl, lr schedule, seed). Today shared as one dataclass; the framework should let an experiment override any of them.

### 10. Backend / performance
Parameters: backend (CPU-SB3 vs GPU-MJX), number of parallel envs, vectorization (subproc vs in-process), sim rate & control decimation, and a **capability declaration** (which controller/reward modes a backend supports — MJX currently excludes fourier/sprint). This is where "optimise for max training speed (n_envs, …)" lives: the framework picks/validates n_envs per backend and refuses unsupported mode+backend combos up front.

### 11. Run / deploy / sim2sim-safety
- **Run:** total epochs/steps (with "nothing to train" short-circuit for pure passive/pattern configs), checkpoint interval, resume, and outputs (**CSV + training-curve plot** are required outputs; TensorBoard optional).
- **Deploy:** ONNX export (VecNormalize baked), moteus target semantics per joint (SET_POS vs SET_CURRENT vs torque — follows §2 per-joint controller), control rate, torque/current caps, gravity FF.
- **Sim2sim / workspace safety:** the safety constraints (floor-penetration, tilt, torque caps, self-collision, workspace limits) declared as parameters, and a **validation pass** that replays a trained policy through a second, safety-constrained or higher-fidelity plant (e.g. the moteus current-mode model) as a gate before hardware. This is the concrete meaning of "use workspace safety as a kind of sim2sim."

---

## Part 3 — Adapting to a new case (the generality test)

A new robot (or the pendulum, or the runner) is defined by filling in, in order: **§1** model + structural descriptor → **§4** which DOFs are free → **§2/§3** a controller per actuated joint (+ symmetry groups) → **§5/§6** obs features + command → **§7** a reward module → **§8** a curriculum (optional) → **§9** network/PPO overrides → **§10** backend+n_envs → **§11** run/deploy/safety. If a case cannot be expressed by choosing values on these 11 axes, the taxonomy is incomplete — that is the acceptance criterion below.

---

## Out of scope for this task (captured, not decided)
- **Strategy question** (versatile "just run" policy vs specific "run-100 m-then-stop"): this is an *objective/curriculum configuration choice* (§7/§8), not a new parameter axis — decide it when we design experiments, not now.
- **Any implementation / refactor / config-schema code** — deferred by request; this document only defines the axes.

## Verification (of the definition itself — no code changes)
Because the deliverable is a taxonomy, "testing" it means proving it can express every experiment we already ran, with no leftover parameter:
1. **Encode each existing experiment as a point** in the 11 axes: pendulum; DASH-01 `m1_stand`, `m1..m6`, `m1_fourier/m2_fourier`, `m1_sprint/m2_sprint_fourier`; and the treadmill (forward+vertical only) case.
2. **Flag any axis that has no value** for one of them → that is a gap to fix in the taxonomy before we build.
3. **Confirm the two extremes fit the same schema**: pendulum (1 joint, all base locked, torque, no command, upright reward, CPU) and untethered runner (6 joints mixed controllers, all DOF free, joystick, gait reward, GPU) must both be expressible with only the axes above.
4. Review the controller menu (§2) against every entry in Part-1 §B and the DOF menu (§4) against §C — each must map to one named parameter value.
