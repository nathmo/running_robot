# walk_v3 — a forward joystick for DASH-01, trained from scratch

Three commands train the policy, from random weights. No warm start from another project, no
checkpoint you have to be given -- each stage warm-starts from the previous stage's own output:

```bash
# 1  the gait and the stick, on the planar plant where balance is free (~1 h on 2 V100s)
python walk_v3/train.py --preset v3_stage1 --name v3_stage1_s0 --seed 0 \
       --devices 2 --n-envs 4096 --n-steps 9
# 2  the free plant: roll, yaw and heading, on training wheels that fade (~3 h)
python walk_v3/train.py --preset v3 --name v3_q_s0 --seed 0 \
       --warm-start walk_v3/runs/v3_stage1_s0/best.msgpack \
       --devices 2 --n-envs 4096 --n-steps 9
# 3  being droppable, and running on a plant that is not the nominal one (~2 h)
python walk_v3/train.py --preset v3_stage3_v24 --name v3_s3_s0 --seed 0 \
       --warm-start walk_v3/runs/v3_q_s0/best.msgpack \
       --devices 2 --n-envs 4096 --n-steps 9
```

and one command tells you whether the result is usable:

```bash
python walk_v3/verify.py --run walk_v3/runs/v3_s3_s0
```

**Why stage 1 is separate.** Measured with a seven-arm probe fleet (40 M steps each, one variable per
arm): cold on the **free** plant -- where roll and yaw exist -- every arm went backwards, while the
same recipe on the **planar** plant reached ep_len 473 and a positive return. The blocker is the
plant, not the command: learning a gait and learning to stay upright sideways at the same time is too
much. Every stage uses the same objective, so the task channel never changes meaning under a warm
start -- which is its own class of bug here (see `task[1]` below).

**Why stage 3 is separate.** The curricula run one at a time (see *One difficulty at a time*), and
sequential curricula cost the SUM of their ramps. Stage 2's 200 M reaches the command band, the
assist fade and the gait-quality penalties, and stops -- every seed finishes with `bringup_scale`
near zero and `dr_scale` at **0.000**. Being droppable and running on a randomised plant are the last
two groups in the queue and they need their own budget.

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

Three stages. Each one warm-starts from the previous stage's **`best.msgpack`**, never
`final.msgpack` -- the keeper exists because runs degrade, and this lineage has lost usable policies
to a late collapse more than once.

```bash
# 1  planar plant, from random weights -- learn the joystick where balance is free
sbatch --export=ALL,PRESET=v3_stage1,NAME=v3_stage1_s0,SEED=0,DEVICES=2,NENVS=4096,NSTEPS=9 \
       walk_v3/slurm/izar_train.sbatch

# 2  free plant -- roll, yaw and heading, on training wheels that fade
sbatch --export=ALL,PRESET=v3,NAME=v3_q_s0,SEED=0,DEVICES=2,NENVS=4096,NSTEPS=9,\
WARM=$HOME/running_robot/walk_v3/runs/v3_stage1_s0/best.msgpack \
       walk_v3/slurm/izar_train.sbatch

# 3  the rest of the queue -- bring-up, then DR, then a jittery controller
sbatch --export=ALL,PRESET=v3_stage3_v24,NAME=v3_s3_s0,SEED=0,DEVICES=2,NENVS=4096,NSTEPS=9,\
WARM=$HOME/running_robot/walk_v3/runs/v3_q_s0/best.msgpack \
       walk_v3/slurm/izar_train.sbatch
```

Stage 3 is not optional. Stage 2 spends its whole 200 M budget on the first three curriculum groups
and finishes with `bringup_scale` near zero and **`dr_scale` at 0.000** -- without the two properties
the deliverable is specified on. Stage 3 buys them with the first three groups pinned at final.
`v3_stage3` is the same thing at the inherited `v_max` of 3.6; `v3_stage3_v24` puts full stick at
2.4, which is where the closed-loop frontier actually is (see *Where this stands*).

### Four ways to waste a run

* **Check the parent THROUGH the warm start, not as it was saved.** `warmstart_var_floor` rewrites
  the obs statistics on load, and measured it takes a good policy from 100% upright to 0% at every
  command. `python walk_v3/tools/speed_frontier.py --run <parent> --warm-start` loads through the
  same surgery training applies, so it shows what the next stage will actually inherit.
* **`STEPS` is deliberately unset.** The preset owns the budget; setting it in the sbatch
  environment overrides the preset silently, which once turned a set of 40 M probes into 300 M runs.
* **`--n-envs` is the total across devices**, not per device. `--n-envs 4096 --devices 2` runs 2048
  per GPU. Getting this wrong halves the batch with nothing in the log to say so.
* **Always train at least three seeds.** Outcomes here are bimodal: of five stage-2 seeds, two
  finished at 80% upright and one at 0%. Rank them with `tools/compare_runs.py`, which compares at
  matched steps and refuses to print training return.

Judge a run on the greedy ladder, never on rollout `ep_len`. The two are different numbers: on the
stage-2 keeper over 20 s, the same weights hold 100% upright at 1.80 m/s greedy on a quiet plant and
30% when sampling at their own std with the weather on. Short rollout episodes right after a handover
are the normal regime, not evidence of a collapse.


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

**Stage 2 is solved too**, by the curriculum queue (§0). Five seeds, 200 M each, warm from stage 1.
The greedy ladder over the full 0–100% stick, on the free plant, with the assist at zero:

```
step          v3_q_s1        v3_q_s2        v3_q_s5     err m/s / upright / dirty
 88,473,600  0.58/65%/28%   0.57/30%/ 5%   0.54/78%/ 5%
147,456,000  1.10/100%/25%  0.44/60%/20%   0.34/80%/ 8%
191,692,800  1.44/100%/25%  0.51/80%/28%   1.37/ 0%/ 0%
```

The full acceptance suite on the best of them (`v3_q_s5`, 176.9 M) passes command tracking, step-in-
place, straightness, the privilege audit and deploy parity (numpy actor vs JAX actor, 4.5e-07). It
is what the two open items below are measured against.

Six earlier attempts at this handover all failed, and the table is worth keeping because each one
looked plausible:

| attempt | result |
|---|---|
| no assist at all | 0% upright through 44 M; **44 of 64 deaths to the workspace check**, none to falling |
| roomier workspace box | peaks at ep_len ~250 by 6 M, back to 145 by 14.7 M |
| roll+yaw assist, 30 M fade | 4/4 seeds collapse at assist 0; `dr_scale` retreats 0.25 → 0.000 |
| roll+yaw assist, 100 M fade | 3/3 degrade from ep_len 600–1150 to 102–168 by assist 0.30 |
| roll+yaw assist, 150 M fade | same shape, slower |
| per-episode assist | best at matched assist (ep_len 736 vs 276–542 at 0.73) — but greedy still 0% |

The diagnosis that explains all six: the workspace check measures foot travel in the **base frame**,
and roll spends most of its ±0.14 m budget geometrically before the legs move — a foot 0.15 m off the
centreline sits `h(1−cos φ) + y·sin φ` lower, about 0.11 m at 20°. A planar-trained policy rolls the
instant it can and is killed by a limit it cannot attribute to anything it chose to do. Sequencing
the curricula is what got a policy through; nothing else did.

### What stage 2 does NOT deliver, and why stage 3 exists

Every one of the five seeds finished with `bringup_scale` between 0.00 and 0.31 and **`dr_scale` at
0.000**. Sequential curricula cost what they cost and 200 M does not buy six of them, so the queue
reached the command band, the assist fade and the shaping, and stopped. The two groups it never
reached are the two the deliverable is actually specified on. `v3_stage3` buys them from the stage-2
keeper with the first three groups pinned at final.

### Two measurements that changed the recipe

**1. The warm start was destroying the policy, at every handover.** Four stage-3 seeds warm-started
from a policy holding 100% upright at 2.70 m/s and fell to ep_len 82 within 3 M steps — on an env
strictly easier than the one the checkpoint came from, at the parent's own action noise, with every
`EnvParams` field matching. `tools/speed_frontier.py --warm-start` applies the obs-stat surgery a
warm start applies and re-runs the ladder on the same weights:

```
                       floor 0.01     no floor
 1.80 m/s commanded     0% upright   100% upright
 achieved                     2.08           1.64
 heading at rest           23.6 deg        2.0 deg
```

`warmstart_var_floor` raises every channel's variance to 0.01 before normalising, which shrinks the
normalised magnitude of every channel that varies less than that — **59 of 412 dims** here. The count
cap is innocent (1e5, 1e7 and uncapped give an identical ladder). The floor is not wrong in general:
it exists because v2's `task[0]` had variance 6.5e-5, so a command of 0.89 normalised to −13.6 σ, and
because a channel that is identically zero on the planar plant divides by ~0 on the free one. Neither
applies to a same-plant continuation, so it is **off for stage 3 and left alone elsewhere**. This is
very likely the real cost of every handover in this lineage, including the 59 M steps stage 2 spent at
0% upright, and nothing in the logs says so — the run just looks like it is learning slowly.

Run `speed_frontier.py --warm-start` on any parent checkpoint before spending a stage on it. It
measures the policy the next stage actually *inherits*, not the one that was saved.

**2. The top of the stick was asking for a fall.** `v_max` came from `tools/speed_lib.py`, a CEM
search over open-loop gait specs — which answers what the action space can express, not what a policy
can hold. Closed loop, `v3_q_s5` greedy, 8 envs per rung:

```
commanded  1.80  2.10  2.40  2.70  3.00  3.30  3.60
achieved   1.64  1.82  1.98  2.16  2.50  2.64  2.79
upright    100%  100%  100%  100%   25%    0%    0%
heading     8.0   6.4   6.7   5.8  10.5  13.7  17.5  deg
```

The cliff is at 3.0, and `cmd_hi` reaches 1.0 — so about a fifth of every stage-2 episode was spent
asking the policy for a speed it falls over at, and paying the 100-point fall penalty for the answer.
`v3_stage3_v24` puts full stick at 2.4 (2.7 is upright but it is the last rung before the cliff, and
full stick should be comfortable rather than marginal). It deliberately does **not** chase the ~0.15
m/s undershoot that runs through the whole band: that is the policy's honest risk-adjusted optimum
against a 100-point fall penalty, and sharpening the tracking income to close it would buy speed with
survival — the wrong trade for a machine an operator is holding.

### The keeper was picking the wrong checkpoint

`best.msgpack` is chosen during training by `keeper_score`, off an eval that ran **16 envs** over a
five-rung command ladder -- about three per rung, so `upright` could only take the values 0 / 33 /
67 / 100%. Re-scored at 12 envs per rung (`tools/rank_checkpoints.py`), the two stage-2 runs both
had a better checkpoint than the one the keeper kept:

```
v3_q_s2   ckpt_150405120   err 0.61   93% upright   5.5 deg      <- best
          best.msgpack     err 0.51   80% upright             (0% at full stick)
v3_q_s5   ckpt_160432128   err 0.39   90% upright   8.6 deg      <- best
          best.msgpack     err 0.36   80% upright
```

The default is now 64 envs, and `rank_checkpoints.py --write-best` fixes runs already on disk.

### Speed and robustness are a frontier, not a ranking

The two checkpoints are not "better" and "worse", they are different points on a trade-off, and the
difference is large enough that it changes what `v_max` should be:

```
                       ckpt_150405120        best.msgpack (176 M)
 top speed achieved       2.14 m/s              2.81 m/s
 upright at 3.30 cmd        100%                   0%
 upright at 3.60 cmd         83%                   0%
 error at 0.90 cmd          0.07 (8%)             0.14 (16%)
 error at 3.60 cmd          1.46 (41%)            0.86 (24%)
```

`ckpt_150405120` refuses to go faster than ~2.1 m/s and never falls; the keeper's pick runs 30%
faster and falls above 3.0. Neither is wrong. It does mean a frontier measured on one checkpoint is
not a statement about the run -- the `v_max` 2.4 in `v3_stage3_v24` was chosen from the keeper's pick
and the better checkpoint supports the same number for a different reason (it saturates at 2.14, so
2.4 is a stretch it can be trained into rather than a fall it cannot avoid).

### What DR costs today

The same checkpoint, plant drawn from the DR ranges, disturbances off: **0% upright at every
command, 0.00 m/s achieved everywhere.** Not degraded -- dead. `dr_scale` never left 0.000 in any
stage-2 run, so this is out-of-distribution rather than fragility, and it is the single clearest
statement of why stage 3 exists.


### Still open

1. **The queue oscillates.** The gates retreat, so when episode length drops after a fade the command
   curricula fall back below 0.99 and the queue returns to group 1 — visible in the log as
   `now advancing cmd_lo+cmd_hi+cmd_zero_p (0/6 complete)` appearing a second and third time. That is
   the servo behaving as designed, but it means a run can spend its budget re-doing early groups and
   never reach DR. Check the queue log before trusting a finished run, and read `dr_scale` out of the
   sidecar — `verify.py` check 1 does exactly this.
2. **Seed variance is large.** Of five stage-2 seeds, two finished at 80% upright and one at 0%. Run
   at least three and rank them with `tools/compare_runs.py`, never on training return.

`v3_stage1b` (roll without yaw, `model/dash01_v2_noyaw.xml`, nq 17) was the alternative to the queue
and is **not** the recipe: its seeds read 0% upright after the fade where the planar rung read 22–60%.
The model and preset stay for anyone who wants to retry that route.

## Known limits

* **Turning is not implemented.** `task[1]` is reserved for a yaw command; nothing drives it yet.
* **Backwards is not trained.** `v_min = 0`.
* The `library` gait-spec source (`gait_lib/`) is inherited from v2 and is **not** part of this
  recipe: it builds specs from true body velocity, which the deploy controller refuses outright.
  `spec_source="policy"` is what ships.
* `walk_mit/V2_CONTRACT.md` is the other arm's contract and describes **v2** — a contact-resynced
  clock and a 33-wide frame. v3 deliberately breaks parity with it: the clock free-runs and the frame
  is 34 wide.
