# SpiderBot RL

Teach the SpiderBot biped to walk under joystick control, in simulation, then transfer to the
real robot. Built from scratch around **MuJoCo** (physics) + **Stable-Baselines3 PPO** (learning).
Start simple — low-speed walking and a solid sim-to-real bridge first; chase top speed later.

## The pipeline

```
 CAD export ──build_model.py──► mujoco/spiderbot/spiderbot.xml ──► rl/env.py ──train.py──► policy
   (OPEN_B)     (adds motors,        (the simulated robot)        (what the      (PPO)        │
                loop, sensors…)                                    agent sees)                 │
                                                                                  evaluate.py ◄┘
                                                                                  (watch / measure)
                                                                                       │
                                                                                  export → Pi (later)
```

## How reinforcement learning is used here

Every 1/50 s the **policy** (a small neural net) reads an **observation** (what the robot can
sense), outputs an **action** (6 motor angle targets), the **simulator** advances, and a
**reward** scores how well it did. PPO adjusts the network to maximize long-term reward. The
agent only ever sees signals the *real* robot can measure, so the learned policy can transfer.

- **Observation** (`rl/env.py`): per motor — position, velocity, torque (6 each); plus the
  IMU-derived gravity direction (3) and angular velocity (3); plus the previous action (6) and
  the joystick command (2). 32 numbers, stacked over the last 5 steps → 160. No foot-contact
  sensor (the real robot has none); the true base velocity is *not* shown (not measurable) — the
  policy infers motion from the stacked history.
- **Action**: 6 PD position targets (hip-roll, cam, thigh × 2 legs). The knee follows the
  parallel linkage; the ankle follows its spring. `target = standing_pose + 0.5·action`.
- **Reward**: track the commanded body-frame forward speed and yaw rate (exp kernels), stay
  upright and at standing height, move smoothly, stay alive; big penalty for falling.
- **Command** = joystick `[forward, yaw]` in `[-1,1]`, mapped to ±1.5 m/s and ±2 rad/s.
  `(0,0)` = stand still.

## The robot model (`mujoco/spiderbot/`)

`build_model.py` turns the CAD export into a simulatable model: free-floating base, world
options, ground, **6 motors** (cam+thigh = CubeMars AKE90-8, hip-roll = AK60-39 — real specs,
incl. reflected `armature`), the **closed parallel knee loop** (`<equality><connect>` from the
pushrod tip to the shin), a **passive preloaded ankle spring**, an IMU + per-motor sensors, box
foot collision, and a flat-footed standing keyframe. `geometry.py` derives the loop/foot numbers
from the mesh; `validate_model.py` is the sanity gate; `find_stance.py`/`tune_ankle.py` are
design tools. Masses/inertias are **placeholders** until the real values are measured (all motor
and inertia numbers live in one table in `build_model.py`).

```
.venv/Scripts/python.exe mujoco/spiderbot/build_model.py      # (re)generate the model
.venv/Scripts/python.exe mujoco/spiderbot/validate_model.py   # gate: loop holds, cam drives knee, no blow-up
```

## Train & evaluate

```
.venv/Scripts/python.exe -m rl.smoke_test                                   # env sanity
.venv/Scripts/python.exe -m rl.train --preset m1_stand --n-envs 6 --subproc # train (CPU)
.venv/Scripts/python.exe -m tensorboard.main --logdir rl/runs               # watch curves
.venv/Scripts/python.exe -m rl.evaluate --run rl/runs/m1_stand --viewer     # watch the policy (local)
.venv/Scripts/python.exe -m rl.evaluate --run rl/runs/m1_stand --video rl/runs/m1_stand/rollout.mp4
```
Key training signals in TensorBoard / logs: `rollout/ep_len_mean` (climbs as it learns not to
fall) and `rollout/ep_rew_mean`.

## Milestones

| # | preset | goal |
|---|---|---|
| M0 | — | model sim-ready & validated ✅ |
| M1 | `m1_stand` | stand / balance at `(0,0)` |
| M2 | `m2_walk` | track forward velocity (curriculum raises the command range) |
| M3 | `m3_turn` | full joystick (forward + yaw) |
| M4 | — | domain randomization + observation history → robust, then ONNX export |
| M5 | — | sim-to-real on the Raspberry Pi (moteus/CAN) |
| M6 | — | push speed; design exploration |

Config and all knobs: `rl/config.py` (presets `m1_stand`, `m2_walk`, `m3_turn`).
