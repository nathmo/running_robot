# walk_v3 — a forward joystick for DASH-01, trained from scratch

One command trains the policy. No warm start, no checkpoint borrowed from another run, no stage that
depends on something a previous project produced:

```bash
python walk_v3/train.py --preset v3 --name v3_s0 --seed 0 --devices 2 --n-envs 4096 --n-steps 9
```

and one command tells you whether the result is usable:

```bash
python walk_v3/verify.py --run walk_v3/runs/v3_s0
```

`verify.py` prints a PASS/FAIL table and exits non-zero if anything failed. **Do not ship a policy on
the strength of a training curve** — this lineage has produced a run that reported healthy episode
lengths for 215 M steps while training on a plant it was never meant to be tested on.

---

## What it is supposed to do

| | contract |
|---|---|
| **Command** | one axis, forward speed. Stick = fraction of top speed: 50% asks for ~50% of top speed, 100% sprints. Accepted error ~15% of top speed. |
| **Zero command** | step in place. Not a stand — this plant has no passive stance and topples in 0.7–1.0 s without a gait. |
| **Straight** | the robot holds the heading it was released at. Turning is out of scope: there is no yaw command. |
| **Bring-up** | you can hold it, start the policy, and let go — dropped from 5–10 cm, or released at up to ±20° pitch / ±8° roll. |
| **Plant** | trained across the randomisation ranges in `config.py` (mass, CoM, friction, gains, torque, actuator delay, IMU bias/noise, control jitter and dropped ticks). |
| **Inputs** | encoders, IMU, its own commanded history, and the operator's stick. **No foot contact, no odometry, no measured base velocity.** |

## Setup

```bash
pip install -r walk_v3/requirements.txt
python walk_v3/model/make_v2_model.py          # writes model/dash01_v2_*.xml
python walk_v3/tools/make_touch_table.py       # writes model/touch_height.npz (bring-up needs it)
python walk_v3/smoke_test.py                   # ~40 invariants, no GPU, run it before any GPU hour
```

`make_touch_table.py` precomputes the base height at which the feet touch, per (pitch, roll). Dropping
the robot "from 8 cm" means 8 cm above *touching*, which depends on how it is tilted; solving that
inside the reset is not jit-able, so it is a table. Without it `bringup_enable` refuses to start.

## Running it

```bash
# the recipe, 2 GPUs, ~3 h
python walk_v3/train.py --preset v3 --name v3_s0 --seed 0 --devices 2 --n-envs 4096 --n-steps 9
# cluster, requeue-safe (the same line is correct for the first start and every restart)
sbatch --export=ALL,PRESET=v3,NAME=v3_s0,SEED=0,DEVICES=2,NENVS=4096,NSTEPS=9 \
       walk_v3/slurm/izar_train.sbatch
```

`--n-envs` is the **total across devices**, not per device. `--n-envs 4096 --devices 2` runs 2048 per
GPU. Getting this wrong halves the batch silently.

**Always train at least two seeds.** Outcomes in this project are bimodal; a single seed is an
anecdote.

## Driving it

```bash
python walk_v3/export.py --run walk_v3/runs/v3_s0 --out results/v3_s0.npz
python walk_v3/tools/play_joystick.py --bundle walk_v3/results/v3_s0.npz
```

`W` / `S` move the stick ±10%, `SPACE` zeroes it, `F` is full, `R` resets, `Q` quits. This runs the
**shipping** control law (`robot/deploy/controller_v2.py`, torch-free numpy, the same code the Pi
executes) against CPU MuJoCo, so what you feel is what the robot would run.

---

## What v3 changes, and why each change is here

Every item below is a fix for something that was **measured** to be wrong, not a guess. The
corresponding evidence is in the code comment at each site.

### 1. The curricula can no longer silently never happen

v2 advanced every curriculum only while the exploring policy's episodes exceeded a fixed tick count,
and `_V2B` set that count to 1200 on a task whose episodes run 300–900. The result: `dr_scale`
finished at **0.000** in three separate runs — 55 M, 135 M and 215 M steps — including the one this
project's own notes called "robust-trained". Control jitter and dropped ticks were zero for the same
reason. Nothing warned; `dr_enable=True` was in every config.

v3 gates on a *fraction of what this policy has actually reached* (`curriculum_gate_mode="relative"`,
`_eff_gate` in `ppo.py`). A gate set relative to the run's own best episode cannot be set above what
the task can do, so it cannot deadlock — and it still retreats when the policy degrades, so a
curriculum never runs ahead of competence. The configured gates keep their relative order.

Belt and braces: the curricula print on **every** progress line, and `verify.py` check 1 fails the run
outright if `dr_scale` did not reach 0.9. The v2 failure was invisible because the only record was a
CSV column nobody read.

### 2. It runs straight, because it can finally see that it is not

v2 gave the actor a low-passed yaw **rate** and billed the same rate in the reward. That is a
controller with no set point: heading was free to random-walk, and on the other arm it did — 171 m of
path for 15–38 m of net progress.

v3 adds the **integrated** heading on both sides:

* the actor reads `clip(wrap_pi(∫ω_z dt), ±π/2)` — dead-reckoned from the *noisy* gyro the policy
  actually gets, after bias, drift, IMU misalignment and the staleness window. This is honest on real
  hardware: the ICM-20948 measures 0.0097 dps/√Hz with a 0.0024 dps bias floor, i.e. well under a
  degree of drift over a 30 s run;
* the reward bills the true angle (`w_heading`, `env.py`). A reward may be privileged; an observation
  may not.

`w_yaw_rate` stays, at a third of its old weight — it damps gait wobble, which is what it was measured
to bill (~95% wobble, ~5% drift). Heading is what bills drift.

Zero heading means "the direction you were pointing when you let go". On the robot,
`controller.zero_heading()` moves that origin; nothing resets it implicitly, because a heading that
silently re-zeroed itself would make the robot veer.

### 3. A cold start has something to climb

Four cold seeds in v2 parked on the 1.5 Hz clock floor and never left, which is why every later run
warm-started from someone else's checkpoint. The reason is visible in the income arithmetic: the
tracking income is `(w_track + w_fwd·v_cmd)·exp(−|v−v_cmd|/σ)`, and with the shipped σ = 0.6 a robot
that cannot walk yet earns `exp(−3.2/0.6)` = **0.5%** of its income at full stick. That is flat.
Meanwhile `w_alive` pays the same whether it obeys or not, so standing still wins.

Two curricula fix it, and the probe presets measure whether each is load-bearing:

* **the command band** starts at 15–45% of top speed and widens to the full 0–100% (`cmd_range_start`);
* **the tracking tolerance** starts at σ = 1.5 m/s and tightens to 0.6 (`track_sigma_start`,
  `track_sigma_steps`), so the first metre per second of speed is worth something.

### 4. The keeper scores the thing we actually want

v2 kept "most survivors, then best tracking", scored on whatever commands the env happened to draw
that episode — re-drawn mid-episode — and on **world-x** speed. Two checkpoints were therefore never
asked the same question, and a tuple keeper lets a policy buy one more survivor with any amount of
tracking error. The run that produced the last shipped policy saved a 37%-error checkpoint as `best`.

v3 (`train.run_eval`, `train.keeper_score`) scores:

* a **fixed command ladder** — one stick position per env, held for the whole episode, same ladder
  every time;
* **body-frame** forward speed, which is what the command means and what the reward bills (world-x
  scores an obedient robot that has turned as disobedient);
* over the **settled tail** only, excluding the bring-up transient;
* plus a second block started the dirty way, so being droppable counts;
* folded into **one scalar** where a fall rate and a heading error have explicit prices, so the two
  cannot be traded without saying at what rate.

---

## Verification

`verify.py` runs seven checks in the order in which they can invalidate one another:

| # | check | bar |
|---|---|---|
| 1 | the curricula actually ran (read from the checkpoint sidecar) | `dr_scale` ≥ 0.9 |
| 2 | command tracking across the ladder, nominal plant | worst error ≤ 15% of top speed |
| 3 | straight: heading and lateral drift at full stick | ≤ 15°, ≤ 1.0 m |
| 4 | bring-up: every episode dropped or released misaligned | ≥ 80% upright |
| 5 | the same ladder on a **randomised plant** | ≥ 80% upright, ≤ 25% error |
| 6 | no privilege: the actor obs is a pure function of measurable state | at the noise floor |
| 7 | deploy parity: the exported numpy law reproduces the JAX policy | match |

### Two traps this suite exists to avoid

**"Randomised plant" is not "every disturbance on".** `evaluate.load_run(dr=True)` builds the env from
the *training* config, which re-enables pushes, wind gusts, trips, a hot thermal start and observation
noise on top of the plant draw. Down that path a policy that completes 100 m dashes dies in 0.21 s. It
faked a v3 collapse once already and the finding had to be retracted. Check 5 draws the plant and
leaves the weather alone.

**Run a known-good control before believing any result.** Six of the bugs found in this project were in
the evaluation code, not the training code. A negative result that is uniform across every condition
(everything dies in 0.3 s, at every command) is a harness signature — real fragility varies with the
condition. A *clean* result needs a positive control too: check 6's leak detector is only meaningful
because the same test reports a 1.0 delta on a v2 checkpoint against a 3e-05 noise floor.

## Layout

```
train.py        the entry point; run_eval + keeper_score live here
verify.py       the acceptance suite -- one table, one exit code
config.py       every knob, with the measurement that set it; presets at the bottom
env.py          MJX env: observation, reward, terminations, resets, bring-up
ppo.py          PPO, the curricula (_gated / _eff_gate), the greedy eval
gait.py         the latched Fourier gait spec and its mirror
networks.py     actor / critic / estimator, the observation mirror
plant.py        the model, the plant draw, overrides
drive.py        the torque law: delay, torque-speed envelope, thermal node
export.py       run -> .npz bundle for robot/deploy
smoke_test.py   invariants that hold with no GPU and no trained policy
tools/          diagnostics; play_joystick.py is the keyboard driver
slurm/          Izar (V100) batch scripts
```

## Observation and action layout

```
actor 387 = 10 history frames x 34 (stride 2)  +  47 once-block
frame  34 = motor_pos 6, motor_vel 6, motor_torque 6, gravity 3, gyro 3,
            lp_yaw 1, phase [cos, sin] 2, prev_residual 6, heading 1
once   47 = latched spec 44, task 2, commit flag 1
task    2 = [v_cmd / v_max, 1.0]        <- task[1] is RESERVED and pinned at 1.0
critic 412 = actor 387 + privileged tail 25
action  50 = 44 latched spec + 6 per-tick residual
```

**`task[1]` is 1.0, not 0.** Under the v2 semantics this channel is a distance-to-go ramp where 1.0
means "nothing to brake for" and 0 means "brake now". Shipping a 0 here holds the policy in a
permanent stop request: measured, a 3.27 m/s runner made 0.1 m/s under a 3.2 m/s command and fell in
every episode. The trainer and `robot/deploy/controller_v2.py` must agree on this bit for bit.

**The privileged tail is critic-only.** It is the critic's input and the estimator's regression target,
never an actor input, and `export.py` never writes it (`networks.py`, `ppo.py`, `export.py`). Check 6
is what keeps that true.

## Known limits

* **Turning is not implemented.** `task[1]` is reserved for a yaw command; nothing drives it yet.
* **Backwards is not trained.** `v_min = 0`.
* The `library` gait-spec source (`gait_lib/`) is inherited from v2 and is **not** part of this
  recipe: it builds specs from true body velocity, which the deploy controller refuses outright.
  `spec_source="policy"` is what ships.
* `walk_mit/V2_CONTRACT.md` is the other arm's contract and describes **v2** — a contact-resynced
  clock and a 33-wide frame. v3 deliberately breaks parity with it: the clock free-runs and the frame
  is 34 wide.
