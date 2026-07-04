# SpiderBot RL — train standing + walking from scratch

Teach the SpiderBot biped to balance and walk under joystick control, in MuJoCo, with
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
python mujoco/spiderbot/build_model.py       # -> mujoco/spiderbot/spiderbot.xml
python mujoco/spiderbot/validate_model.py     # gate: closed loop holds, cam drives the knee, no blow-up
python -m rl.smoke_test                        # env sanity: obs/action shapes, a few steps, no NaN
```

## 3. Train — standing, then walking (each trains from scratch)
Pick `N` ≈ (CPU cores − 2). Times below are for an 8-core CPU at ~2.3k env-steps/s.

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
python -m rl.train --preset m2_walk --steps 8000000 --n-envs $(($(nproc)-2)) --subproc --no-progress
# detach: Ctrl-b then d   |   reattach: tmux attach -t train
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
```
Watch **`rollout/ep_len_mean`** (climbs toward the 1000-step cap as it stops falling) and
**`rollout/ep_rew_mean`**. Expect a long flat early phase, then a sharp take-off. If
`train/clip_fraction` runs away above ~0.2 while reward stays flat, the policy is stuck — stop and
adjust the reward rather than training longer.

## 5. Evaluate / watch / record
```bash
# metrics over several episodes (ep_len, command-tracking error)
python -m rl.evaluate --run rl/runs/m2_walk --episodes 5

# drive a fixed joystick command and record an mp4  (vx,yaw are normalized -1..1)
python -m rl.evaluate --run rl/runs/m2_walk --vx 0.5 --yaw 0.0 --video rl/runs/m2_walk/walk.mp4

# live 3D viewer (run locally, not over headless SSH)
python -m rl.evaluate --run rl/runs/m2_walk --viewer --vx 0.5
```
Evaluation auto-loads the matching VecNormalize stats; pass `--checkpoint rl/runs/.../ppo_XXXX_steps`
to evaluate a specific checkpoint instead of the final model.

---

## How it works (brief)
- **Model** (`mujoco/spiderbot/`, generated by `build_model.py`): free-floating base, the **6
  motors** (cam+thigh = CubeMars AKE90-8, hip-roll = AK60-39, real torque/inertia), the **closed
  parallel knee loop**, a passive **toe-down** ankle spring, an IMU + per-motor sensors, and a
  **point-toe contact**: only a small sphere at each foot tip touches the floor. Masses/inertias
  are placeholders until the real values are measured (all in one table in `build_model.py`).
- **Observation** (`rl/env.py`, hardware-measurable, history-stacked): per-motor pos/vel/torque,
  IMU gravity direction + angular velocity, previous action, joystick command. No foot-contact
  sensor.
- **Action**: 6 PD position targets (knee follows the linkage; ankle follows its spring).
- **Reward**: track commanded body-frame forward speed + yaw, stay upright/at height, smoothness,
  alive; **feet air-time** reward (deliberate strides, punishes tiny vibration steps).
- **Hard rules (termination)**: fall (low/ tilted), and **floor violation** — only the lower half
  of a toe sphere may touch; if a toe sphere sinks (upper-hemisphere contact) or any foot/shin
  point dips below the floor, the episode is killed (the robot can't cheat by clipping the floor).
- **Config / all knobs**: `rl/config.py` (presets `m1_stand`, `m2_walk`, `m3_turn`).

## Milestones
| # | preset | goal | status |
|---|---|---|---|
| M0 | — | model sim-ready & validated | ✅ |
| M1 | `m1_stand` | balance on the toe points | ✅ (953/1000, 0 floor violations) |
| M2 | `m2_walk` | track forward velocity (+ stand) | in progress |
| M3 | `m3_turn` | full joystick (forward + yaw) | next |
| M4 | — | domain randomization → ONNX export → Raspberry Pi (moteus) | later |
