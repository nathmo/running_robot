# walk_v3 — a forward joystick for DASH-01, trained from scratch

Two commands train the policy, from random weights. No warm start from another project, no
checkpoint you have to be given -- stage 2 warm-starts from stage 1's own output and nothing else:

```bash
# stage 1 -- learn the gait on the planar plant (~1 h on 2 V100s)
python walk_v3/train.py --preset v3_stage1 --name v3_stage1_s0 --seed 0 \
       --devices 2 --n-envs 4096 --n-steps 9
# stage 2 -- the free plant, with heading, bring-up and DR (~3 h)
python walk_v3/train.py --preset v3 --name v3_s0 --seed 0 \
       --warm-start walk_v3/runs/v3_stage1_s0/best.msgpack \
       --devices 2 --n-envs 4096 --n-steps 9
```

and one command tells you whether the result is usable:

```bash
python walk_v3/verify.py --run walk_v3/runs/v3_s0
```

**Why two stages.** Measured with a seven-arm probe fleet (40 M steps each, one variable per arm):
cold on the **free** plant -- where roll and yaw exist -- every arm went backwards, while the same
recipe on the **planar** plant reached ep_len 473 and a positive return. The blocker is the plant,
not the command: learning a gait and learning to stay upright sideways at the same time is too much.
Both stages use the same objective, so the task channel never changes meaning under the warm start --
which is its own class of bug here (see `task[1]` below).

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
# cluster, requeue-safe (the same line is correct for the first start and every restart)
sbatch --export=ALL,PRESET=v3_stage1,NAME=v3_stage1_s0,SEED=0,DEVICES=2,NENVS=4096,NSTEPS=9 \
       walk_v3/slurm/izar_train.sbatch
sbatch --export=ALL,PRESET=v3,NAME=v3_s0,SEED=0,DEVICES=2,NENVS=4096,NSTEPS=9,\
WARM=$HOME/running_robot/walk_v3/runs/v3_stage1_s0/best.msgpack walk_v3/slurm/izar_train.sbatch
```

Warm-start stage 2 from **`best.msgpack`**, not `final.msgpack`. The keeper exists because runs
degrade: stage 1's own last checkpoint sits past its last evaluation, and this lineage has lost usable
policies to a late collapse more than once.

`STEPS` is deliberately unset by default: the preset owns the budget. Setting it in the sbatch
environment overrides the preset silently, which once turned a set of 40 M probes into 300 M runs.

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

### 0. One difficulty at a time

This is the change the rest of the folder was rebuilt around, and it came last because it took a
dozen runs to see. Every run followed the same arc -- climb to a peak, then decline from the point
where the curricula started biting. Six of them advance off one competence gate: the command band
widens, the gait-quality penalties come on, the starts get dirty, the plant randomises, the
controller gets jittery, and the base assist fades. The task hardens in six directions at once and
the policy never consolidates any of them.

v2 had the opposite failure -- absolute gates set so high that nothing ever advanced, which is how
`dr_scale` finished at **0.000** in three separate runs. The answer is neither extreme.

`curriculum_order` names a sequence of groups. A curriculum may advance only once every group before
it has reached 1.0 (`ppo._queued`):

```
("cmd_lo", "cmd_hi", "cmd_zero_p")   be able to do the job      25 M
"pitch_assist"                       stand on your own          30 M
"shape_scale"                        do it well                 25 M
"bringup_scale"                      do it from a bad start     30 M
"dr_scale"                           do it on a different robot 40 M
("ctrl_jitter_ms", "ctrl_drop_prob") with a worse controller    25 M
```

Sequential ramps do not overlap, so a run needs their **sum**, not their maximum: 175 M of stage 2's
200 M. Stage 1's budget reaches the first three groups and stops, which is why a stage-1 checkpoint
honestly reports that DR never ramped.

Putting the assist fade second is deliberate. Removing the crutch is the single step that has killed
every free-plant run, and it had always been happening while five other things also got harder.

Measured, with only the command band advancing and everything else verifiably frozen: the three
planar seeds reached **ep_len 2503-2798 of a 3000 cap** with returns of 4491-5186, against 1013 and
1197 for the same stage under parallel curricula.

**Three bugs lived in this mechanism before it worked**, all the same shape -- a curriculum whose real
behaviour did not match its config, which is exactly what this folder exists to prevent:

* clock-based ramps read the global step and never saw the queue, so `shape_scale` climbed to 0.45
  while the log said only `cmd_lo` was advancing;
* a name *omitted* from the order is not frozen, it advances unqueued, so DR ramped to 0.146 during a
  stage 1 whose order had been shortened for readability;
* `cmd_lo`, `cmd_hi` and `cmd_zero_p` are one curriculum wearing three names; queued separately they
  cost three ramps and would have eaten 75 M of a 100 M budget, leaving the assist fade unreached.

All three were caught because the queue prints which curriculum is live and the progress line prints
the values beside it. None would have been visible otherwise.

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

### 4. The objective no longer pays a policy to die

This is why no cold run in this lineage ever learned. Measured with `tools/reward_budget.py
--curriculum start` on two cold runs, one per plant:

```
INCOME 1.87   (alive 1.50 = 80%,  track 0.32 = 17%,  clearance 0.05)
COST  -2.34
LIVING        -0.236 / tick
```

Living is **negative**, so against a one-time `fall_penalty` of 100 over a 3000-tick episode, dying
immediately is worth seven times staying alive. The optimiser was correctly hunting for the shortest
episode -- which on this plant means railing the gait clock and falling early, which is what every
cold arm did.

The cost is not efficiency: `torque`, `motor_vel` and `energy` already ramp through `eff_scale` and
were **0.0%** of it. It is *gait quality*, billed at full weight to a policy that has no gait --
`residual` 27.6%, `phase_contact` 15.5%, `foot_slip` 12.9%, 56% between them. "Don't fight your own
clock" and "don't scuff your feet" are corrections to a walk; charged before there is a walk they are
a tax on trying.

Those terms now scale with `shape_scale`, ramped from 0.15 through the same retreating gate as the
other curricula. **Not** ramped: `upright`, `height`, `alive`, `track`, `clearance`, `air_time`,
`heading`, `lane` -- the income and the safety terms, which mean the same thing on day one as at the
end. Also not ramped: `duty_sym`, `swing_floor` and `stance_time`, which cost 0.0% and are the guards
against the one-legged hop and the dragged stance.

Same preset, same seed, same 5.5 M steps, with and without the ramp: return **-11 -> +89** on the free
plant and **-8 -> +137** on planar, and the clock left its floor.

> **Read returns carefully.** Return is not comparable across `shape_scale` values -- raising the
> penalties lowers it by construction -- any more than it is comparable across different rewards.
> Judge on the greedy ladder eval, not the training curve. This project's own rule is "always eval
> greedy", and it applies to this curriculum too: a run whose training return fell from 1197 to -18
> as its ramp completed was at the same time improving from 0% to 60% upright on the greedy ladder.

### 5. The keeper scores the thing we actually want

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

## Where this stands (2026-09-13)

**Stage 1 is solved and reproducible.** Four seeds, cold from random weights, 80 M steps each. Best
checkpoint, per stick position (command error m/s / fraction upright):

```
v3_stage1_s1     0%: 0.32/100%   25%: 0.25/100%   50%: 0.80/100%   75%: 1.36/100%   100%: 1.88/62%
v3_stage1g_s3    0%: 0.51/ 62%   25%: 0.14/ 88%   50%: 0.79/100%   75%: 1.47/100%   100%: 2.12/100%
```

**Stage 2 is not.** The transfer to the free plant is the open problem, and it is worth reading before
you spend GPU hours on it, because six things have been tried and measured:

| attempt | result |
|---|---|
| no assist at all | 0% upright through 44 M; **44 of 64 deaths to the workspace check**, none to falling |
| roomier workspace box | peaks at ep_len ~250 by 6 M, back to 145 by 14.7 M |
| roll+yaw assist, 30 M fade | 4/4 seeds collapse at assist 0; `dr_scale` retreats 0.25 → 0.000 |
| roll+yaw assist, 100 M fade | 3/3 degrade from ep_len 600–1150 to 102–168 by assist 0.30 |
| roll+yaw assist, 150 M fade | same shape, slower |
| per-episode assist | best at matched assist (ep_len 736 vs 276–542 at 0.73) — but greedy still 0% |

One stage-2 checkpoint did reach the target behaviour before collapsing, which is why this is a
handover problem and not an objective problem: **0.87 m/s on a 0.90 command, 1.60 on 1.80, 3.19 m/s at
full stick, heading drift 4–11°**. The command channel, the reward and the heading term all work.

The diagnosis that explains all six rows: the workspace check measures foot travel in the **base
frame**, and roll spends most of its ±0.14 m budget geometrically before the legs move — a foot 0.15 m
off the centreline sits `h(1−cos φ) + y·sin φ` lower, about 0.11 m at 20°. A planar-trained policy
rolls the instant it can and is killed by a limit it cannot attribute to anything it chose to do. An
assist prevents that, and then cannot be removed: at any scale it still corrects that share of every
error, so the policy never meets its own mistakes.

**What is under test:** `v3_stage1b`, a third rung on `model/dash01_v2_noyaw.xml` (the free plant minus
heading, nq 17). It introduces roll *without* yaw, so the policy meets one new degree of freedom at a
time and may need no assist at all. If that works, the recipe is planar → no-yaw → free and the
`v3` preset's assist should be switched off.

## Known limits

* **Turning is not implemented.** `task[1]` is reserved for a yaw command; nothing drives it yet.
* **Backwards is not trained.** `v_min = 0`.
* The `library` gait-spec source (`gait_lib/`) is inherited from v2 and is **not** part of this
  recipe: it builds specs from true body velocity, which the deploy controller refuses outright.
  `spec_source="policy"` is what ships.
* `walk_mit/V2_CONTRACT.md` is the other arm's contract and describes **v2** — a contact-resynced
  clock and a 33-wide frame. v3 deliberately breaks parity with it: the clock free-runs and the frame
  is 34 wide.
