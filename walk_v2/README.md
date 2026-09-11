# walk_v2 — DASH-01 Walker v2 on MJX (GPU-parallel)

The training pipeline for the **DASH-01 Walker v2** design (artifact rev 2026-09-09, the ground
truth for everything here), rebuilt from `walk_mit/` so that the simulation runs as one batched
JAX program on a GPU: thousands of MuJoCo-XLA (MJX) plants stepped in one `jax.vmap`, PPO in
JAX, no subprocess workers. `walk_mit/` is the CPU stack it succeeds and is not touched.

```
walk_v2/
  config.py          Config dataclass + presets (v2_s1_planar, v2_s2_free, *_easy, v2_lib_*, v2_smoke)
  model/             dash01_base.xml (copy of walk_mit's plant) -> make_v2_model.py -> dash01_v2_{free,planar}.xml
                     calibrate_plant.py: leg spring 5 mm/BW + armature Bode fit -> plant_fit.json
  gait.py            the v2 gait law (50-dim action, latch semantics, knobs, kp(phi)/kd(phi), reflexes,
                     the mirror). numpy AND jax from one source (xp=...)
  drive.py           PD + torque-speed clamp + substep-granular delay + thermal node
  plant.py           Plant (MJX model, indices) + draw_plant (per-episode DR as batched model fields)
  env.py             DashEnvV2: reset/step for N envs, obs 377/402 (policy) or 353/378 (library)
  networks.py        estimator / actor / critic (flax), masked Gaussian, observation mirror
  ppo.py             PPO in JAX: masked log-prob, symmetry loss, supervised estimator, schedules, curricula
  train.py           CLI (--preset --steps --n-envs --resume auto --warm-start)
  evaluate.py        greedy batched eval + mp4 (classic MuJoCo render of the recorded qpos)
  export.py          deployment bundle v2 (.npz) for the Pi runtime
  bench.py           env steps/s vs batch, PPO iteration time (the GPU benchmark; --iterations caps)
  smoke_test.py      the invariants (gait mirror, thermal anchors, delay, plant fit, obs layout, mask)
  tools/eval_envelope.py   the §08 margin gate (wind, tilt, friction, delay, mass, thermal, pushes, lane)
  tools/trace.py, compare_traces.py   cross-implementation agreement protocol (see below)
  gait_lib/          §09: return map + Newton + Floquet (solver.py), CMA-ES (cmaes.py), Stage 4/5 (search.py)
  slurm/             Izar (and Lyra) job scripts + cluster README
```

## What the artifact specified and where it lives

| artifact | implementation |
|---|---|
| 100 Hz control, 1 kHz physics | `control_decimation=10`; every step-denominated schedule halved vs walk_mit |
| 50-dim action, 44 latched at φ-wrap + 6 residual | `gait.py` layout; latch in `env._step_one` (`state.commit`) |
| commit flag in the once-block; log-prob mask reads the same flag | obs index `env.wrap_index`; `info["commit"]` → `networks.dim_mask` in the rollout |
| one series per family, Δ/s/o knobs, (+,+) offsets and roll reflex, (+,−) pitch reflex | `gait.feedforward / reflexes`; sign rule tested in `smoke_test.py` |
| kp(φ), kd(φ) exp map ×2.5/÷3, soften-only ÷4 | `gait.impedance` (a0 is the level, harmonics modulate) |
| contact-triggered resync κ 0.5 ± DR, W 0.15 cycle, N_ema 5, 3-cycle warm-up, +1 tick commit | `env._step_one` after physics; a backward pull never uncrosses the wrap |
| obs 33-frame × 10 × stride 2, once-block 44+2(+1), priv tail 25 | `env._frame / _once / _priv` |
| estimator 376→128→64→3 supervised, detached; actor 256×256; critic 401 | `networks.py`, `ppo._est_update` (Adam on the estimator subtree only) |
| symmetry loss w_sym ‖k(M_o s)+k(s)‖² | `networks.ObsMirror` + `ppo.loss_fn`; stats mirror-symmetrized each rollout |
| spec change billed once at commit, standing knob price, w_residual 0.10 / rate 0.02 | reward terms `spec_cycle`, `knob`, `residual`, `residual_rate` |
| LP yaw (τ 0.7 s) in obs and reward, lane term, one-sided height floor 0.81 m | `lp_yaw_*`, `lane`, `height` |
| thermal ΔT ODE τ_th 45 s, penalty > 0.85, hot start U[0,0.7], scale U[0.8,1.2] | `drive.thermal_update`, `plant.draw_plant`; anchors checked in `smoke_test.py` |
| delay 12 ms nominal, U[6,18] ms per episode, substep granular | `drive.live_command` on a 3-tick command buffer |
| armature-fit drive, no EMA target filter | `model/calibrate_plant.py` → `plant_fit.json`, `<motor>` actuators + PD in `drive.py` |
| rigid loop closure + 30 kN/m series spring, 5 mm at 1 BW, ±50 % DR | see *Plant deviations* |
| DR on: mass/CoM/friction/gains/damping/tilt/zero/IMU + measured IMU noise | `plant.draw_plant`, `env._frame` |
| wind ±30 N + 30 N gusts, pushes, trips, timing jitter/drop | `env._step_one` |
| two-stage curriculum, frozen interface | presets `v2_s1_planar` (no y/roll/yaw joints) and `v2_s2_free`, same widths |
| gait library pipeline (§09) | `gait_lib/` + `spec_source="library"` env variant (action 6+3, once-block 23) |
| §08 margin gate | `tools/eval_envelope.py` |

## Plant deviations from the artifact text (measured while building, all in `model/plant_fit.json`)

* **Series spring location.** The artifact puts the 30 kN/m spring at the pushrod tip. Measured on
  the welded rig, a vertical stance load barely loads the loop closure at all (the leg is at
  99.5 % reach, the load path is axial): a rod spring cannot produce a 5 mm sink at any stiffness,
  and at 1 kHz nothing stiffer than the constraint's own 2 ms time constant integrates. The spring
  is therefore a prismatic joint along the shin axis at each foot (`leg_spring_L/R`), fitted to
  **5.00 mm vertical at 1 BW on one leg** (23.4 kN/m axial = 29.7 kN/m foot-referred; 10.0 / 3.3 mm
  at the ±50 % DR corners). The loop closure itself carries the ankle-lock values (solref 0.002).
* **Armature fit is a compromise.** The thigh's in-air corner is set by the link inertia; armature
  can only lower it. Best fit: **6.96 Hz at kp 200/kd 5 (target 6.3, −0.2 dB peaking) and
  16.1 Hz at 500/5 (target 19, +1.1 dB)**. Hip: 4.4 Hz at 120/4 (kp/kd rule 4.8).
* **Rigid ankle folded.** The ankle joint is removed and its keyframe angle folded into the foot
  body pose (exact to 1e-11 m); 0.209 kg of spring hardware leaves each shin.
* **Obs widths 377/402** (policy variant) instead of 376/401: the commit flag is an explicit
  once-block entry (§05 rev 2026-09-09), since a resync can move the wrap.
* **Δ_max defaults to 0.6 rad** (the CPG arm's value); `v2_s2_free_wide` is the ±π variant (§13).

* **Constants the artifact leaves open follow the CPU reference's `walk_mit/V2_CONTRACT.md`**
  (2026-09-10): roll_amp 0.20, offset scales (0.06, 0.06, 0.15), pitch reflex 1.0 / 0.1 with a
  0.9 EMA on the rate, w_knob 0.02, pushes U[0.3, 0.6], trips 0.0008, IMU 2°, gate 600, the
  residual symmetry term, spec_cycle not billed on the first commit. One deliberate difference
  remains: the phase channel is `(cos φ, sin φ)` as the artifact's §03 table states; the contract
  lists `(sin φ, cos φ)`.

## v2b: the readout-1 deltas (2026-09-10)

Both arms' artifact-literal `v2` runs fell into a clock-rail exploit inside the latched design: the
CPU arm's clock sat at 5.00 Hz on 100 % of commits with a bang-bang spec (regressing from ep_len
~700 at 19 M to ~150 by 33 M), this side's at 0.5 Hz on 94 % of commits with the residual at its
bound half the time. `walk_mit/V2_CONTRACT.md` defines `v2b`: frequency range (1.5, 4.0) Hz and
residual 0.10 rad at w_residual 0.20 (the two rows the GPU port must mirror), plus training-side
gates (pitch-assist fade opened by competence, entropy gate needing ep_len > 600 with the deadline
at 40 M, DR/jitter gates 1200). Presets `v2b_s1_planar` / `v2b_s2_free`; `v2_*` stay the
artifact-literal reference. Fixed at the same time: the stance-ratio and efficiency ramps are
clock-driven from step 0 as on the CPU arm (they were gated at ep_len 600 here, so the CPU arm
paid the efficiency terms from 1 M steps while this side paid nothing).

## Readout 2 (2026-09-10 evening): v2b collapses on both arms when the wheel goes

Matched at steps, `v2b_s1` on both arms (CPU: `walk_mit/runs/v2b_s1_s*`; GPU: `runs/v2b_s1_planar_s*`):

| | CPU s0 / s1 | GPU s0 / s1 |
|---|---|---|
| peak episode length | 1893 (22 M) / 920 (22 M) | 676 (40 M) / never above 115 |
| at 80 M | 23 / 24 | 19 / 108 |
| clock | 4.0 Hz on 100 % of commits | 4.0 Hz on 100 % / 1.5 Hz on 67 % |
| greedy eval without assist (GPU) | | 0/16, falls within 1 s at every checkpoint |

Both collapses coincide with the pitch-assist fade running out (v2: clock fade 0→30 M, collapse
~30 M on both arms; v2b: fade opened at ep_len 600, collapse 20–30 M later on both arms), and the
greedy eval with the assist removed never stood at all. The policy balances on the 100 N·m/rad
pitch spring and has no per-tick authority to replace it once the spec is latched (residual 0.10 rad
in v2b). Experiments running: `v2b_s1_planar_noassist` (seeds 0/1, no wheel from step 0: does the
latched design learn balance at all?) and `v2b_s1_planar_stiff` (seed 0, 10× leg spring = the CPU
arm's effectively rigid rod, isolating the plant compliance in the per-step lag).

## Readout 3 (2026-09-10, late): the wheel is the cause; the plant is not

Episode length / assist level / policy std at matched steps (GPU, `v2b_s1`):

| steps | soft leg, wheel (s0) | stiff leg, wheel (s0) | no wheel (s0) | no wheel (s1) |
|---|---|---|---|---|
| 20 M | 187 / 1.00 / 0.97 | 231 / 1.00 / 0.95 | 45 / 0 / 1.00 | 36 / 0 / 1.00 |
| 30 M | 436 / 1.00 / 0.95 | 602 / 1.00 / 0.94 | 44 / 0 / 0.99 | 38 / 0 / 1.00 |
| 40 M | 676 / 0.85 / 0.82 | 799 / 0.69 / 0.69 | 96 / 0 / 0.95 | 38 / 0 / 1.00 |
| 50 M | 630 / 0.51 / 0.59 | 392 / 0.35 / 0.49 | 114 / 0 / 0.68 | 73 / 0 / 0.71 |
| 55 M | 453 / 0.35 / 0.49 | 188 / 0.19 / 0.41 | 120 / 0 / 0.57 | 87 / 0 / 0.59 |

The stiff-leg run tracks the soft-leg run and collapses the same way as its assist fades (17 at
57 M with the assist at 0.13), so the leg compliance is not behind the per-step lag or the collapse.
Without the wheel the latched design learns balance, but slowly: episode length 120 / 87 at 55 M
with the greedy policy creeping 1.1 m at 0.8 m/s (seed 0), and the 40 M entropy deadline is now
annealing the std toward 0.25, which will freeze whatever exists by ~80 M. The wheel runs' episode
lengths were never the policy's own: everything above ~100 ticks was the 100 N·m/rad pitch spring.

## v2c: the readout-2 deltas (2026-09-10) — the first working policy

The CPU arm's readout of v2b at 23 M found 68–79 % of the spec means and 53–60 % of the residual
means OUTSIDE [−1, 1]: the Gaussian is sampled unbounded and clipped by the env, so every mean past a
rail earns the same clipped sample (bang-bang spec, parked clock, saturated residual are all
`clip(μ)` of drifted means, and greedy `clip(μ)` no longer matches the trained `E[clip(μ+ε)]`).
`v2c` = v2b + the action-mean bounds loss `w·mean(relu(|μ|−1)²)` (w_bound 1.0, logged as
`train/bound_loss` / `train/mu_out_frac`), a retreating curriculum (0.7), a 0.7 std cap before the
anneal, and the anti-crutch assist-torque bill (w_assist_penalty 0.005, the env already billed
−w·τ²). On the CPU arm `v2c_s1` seed 0 at 30 M finishes 3/3 greedy dashes at 2.3–2.5 m/s with the
wheel at 0.47 (`walk_mit/monitor/v2c_s1_s0_30M_greedy.mp4`, rendered here from the pulled
checkpoint: 138.7 m in 60 s, line at 38.9 s, peak 3.17 m/s); wheel-dependence at 30 M was the open
risk; at 42 M (assist 0.09) the pulled checkpoint runs the full 60 s greedy (146 m, line 33.5 s),
and **with the assist forced to 0 it still runs all three episodes: 182 / 146 / 189 m, lines at
25.0 / 33.6 / 25.9 s, 2.4–3.1 m/s, peak 3.8 m/s** (`walk_mit/runs/v2c_s1_s0_42M_noassist`,
curriculum.json with pitch_assist 0) — the first policy that stands without the wheel. Presets `v2c_s1_planar` / `v2c_s2_free`; the GPU-side training
of `v2c_s1_planar` seeds 0/1 is the parity run (Izar 3145269/70).

## Cross-load (contract parity step 3) and the fast GPU sizing (2026-09-10, afternoon)

`tools/import_sb3.py` loads a walk_mit SB3 checkpoint into a walk_v2 bundle (two steps: the torch
venv dumps `policy.pth` + VecNormalize to npz, the JAX venv builds the bundle; the graphs are
identical, the one layout difference — frames carry (cos φ, sin φ) here, (sin φ, cos φ) there —
is folded into the first layers). The 42 M `v2c_s1_s0` policy that runs 60 s wheel-free in classic
MuJoCo **falls within 0.1–1.8 s in MJX**, on the soft leg and on the 10× stiff leg alike. The
interface is not the cause: `replay_golden.py` now prints the newest frame per channel over the
first ticks (phase swapped) and shows no channel beyond the 0.05–0.13 trajectory divergence (the
torque channel differs by 0.04 on the reset frame, where this side reports zero). What remains is
the plant build itself (their literal rod spring and near-zero armature vs the measured fits here,
the folded ankle) — the open-loop traces already diverge to 0.1 rad within 0.7 s. So "the same
policy" is a matched-training statistics comparison, not a checkpoint swap, until the two plant
builds are reconciled.

Fast sizing for the 3-hour goal: `v2c_s1_planar_fast` = v2c + 2048 × 9 (the contract's 18 432-sample
rollout kept) + the 8 × 8 solver cap: **11.0–11.2k steps/s on one V100** (1.45× the parity sizing),
early learning unchanged. `--devices N` (train.py) splits the envs over N GPUs with pmapped rollouts
and lax.pmean gradient averaging, verified on two emulated CPU devices
(`XLA_FLAGS=--xla_force_host_platform_device_count=2`); the 4 × V100 runs are `runs/v2c_fast_dp4_s*`
(Izar `gpu-xl`, jobs 3145282/83). Kuma's H100 nodes need the same QOS enablement as Lyra.

## Multi-GPU scaling for the 3-hour goal (2026-09-10, measured on Izar)

| run | devices × envs | rollout samples | update | steps/s |
|---|---|---|---|---|
| `v2c_s1_planar` (parity sizing) | 1 × 1024 × 18 | 18 432 | per-minibatch | 7 600–7 900 |
| `v2c_s1_planar_fast` | 1 × 2048 × 9, cap 8 × 8 | 18 432 | per-minibatch | 11 000–11 200 |
| `v2c_fast_dp4` | 4 × 512 × 9 | 18 432 | per-minibatch pmap | 15 700–18 500 |
| `v2c_fast_dp2x` | 2 × 2048 × 9 | 36 864 | per-minibatch pmap | 18 000 |
| `v2c_fast_dp4x` | 4 × 2048 × 9 | 73 728 | fused per-epoch scan | **42 300** (rollout 1.38 s, update 0.16 s) |

The batched step is latency-bound, so splitting a fixed 18 432-sample rollout over more devices
barely helps (73 ms/step at 512 envs vs 155 ms at 2048): the rollout must grow with the devices
(2048 envs per device), keeping the minibatch and epochs — the same gradient updates per sample
as the contract, a larger batch per policy iteration. The per-minibatch pmap dispatch (~18 ms a
call) then dominates the update, hence the fused per-epoch `lax.scan` inside one pmap call
(`_update_epoch_p`, target-KL early stop carried inside). Izar's second 4-GPU node is held by a
30-hour job, so the 4 × 2048 run replaced the 4 × 512 one on `ixl01`.

**Sizing verdict (2026-09-10, 13:30):** PPO here is iteration-bound, not sample-bound. At 4× the
contract rollout (8192 × 9) the 4-GPU run sat at ep_len ~150 from 30 M to 70 M steps (28 min), no
better per wall-clock than the single-GPU contract run at 20 min, with or without the √4 learning-rate
and ×2 schedule scaling. At 2× the rollout (2 × 2048) learning per step held (ep_len 776 at 35 M in
27 min — faster per wall-clock than one GPU). So the 4-GPU configuration is 4 × 1024 envs × 9 (36 864
samples, contract schedules): `v2c_fast_dp4z_s2`. Protections added on the way: the best greedy
checkpoint is kept as `best.msgpack` (the end-of-fade cliff at 47 M erased a working policy from the
training curve, not from disk), and `lr_kl_adaptive` (rl_games' schedule) is under test on
`v2c_fastkl_dp2x_s3` against the target-KL early stop, which fired every iteration through that cliff
while the CPU arm's run sailed through the same phase (its KL stayed at 0.01–0.03).

Two lessons from the first 4 × 2048 run (`v2c_fast_dp4x_s0`, 42–44k steps/s): (1) the recipe's
step-based schedules are really iteration counts — the CPU arm's 40 M entropy deadline is 2170 policy
updates at 18 432 samples per rollout, and at 4× the rollout it arrived after a quarter of the updates
and froze a policy that had not yet left the low rail (ep_len 80–150 at 45 M); `v2c_s1_planar_dp4`
scales those schedules ×2, the learning rate by √4, and runs 450 M steps (~3 h). (2) An unconverged
8 × 8 solver step can produce a non-finite plant state; the episode ended correctly but the reward of
that tick was NaN and poisoned the update (`v2c_s1_planar_fast_s0` died at 36.5 M): the env now bills
a non-finite state as a fall and `grad_guard` (optax.apply_if_finite) skips a non-finite update.

## Run

```bash
# local CPU (tests only)
python -m venv .venv_v2 && .venv_v2/Scripts/activate && pip install -r walk_v2/requirements.txt
python walk_v2/model/make_v2_model.py            # only if the XMLs are missing; calibrate_plant.py refits
python walk_v2/smoke_test.py --quick
python walk_v2/train.py --preset v2_smoke --steps 2000 --n-envs 8

# GPU (Izar V100; Lyra is blocked, see slurm/README.md)
sbatch walk_v2/slurm/izar_bench.sbatch
sbatch --export=ALL,PRESET=v2_s1_planar,NAME=v2_s1_planar_s0,SEED=0 walk_v2/slurm/izar_train.sbatch
```

Checkpoints: `runs/<name>/ckpt_<steps>.msgpack` (+ `.json` with the curriculum state), `final.msgpack`;
`--resume auto` continues, `--warm-start <file>` carries weights + obs stats into a new stage
(count capped, variance floored, log_std re-inflated — the walk_mit warm-start rules).

PPO sizing follows `walk_mit/V2_CONTRACT.md` §PPO so both arms take the same gradient updates per
sample: rollout 18 432 samples (1024 envs × 18 steps here = 64 × 288 there), minibatch 4096, 4 epochs.
The first Izar runs used 2048 × 64 / 16 384 (4× fewer updates per sample) and learned visibly slower
per step (ep_len 105 vs ~400 at 13 M steps); they were restarted.

## Throughput on the V100 (2026-09-10, `results/`)

Two things had to be fixed before the GPU port was faster than the CPU arm at all:

1. **The bench recompiled inside the timed call.** `bench_env` warmed up with a 3-step scan and timed
   a 64-step scan with the length static; the ~40 s compile showed up as a batch-independent
   ~600 ms/step floor on CPU and GPU alike (the first numbers, 1.7–2.5k env steps/s, were ~15× low).
   The PPO-iteration and training-rollout numbers never had this problem.
2. **Under `jax.vmap` the Newton solver runs to the slowest env.** The XML carries classic MuJoCo's
   `iterations=100 ls_iterations=50`. Healthy states converge in one iteration (bare physics: 37k env
   ticks/s at 2048–4096 envs), but one flailing env makes the whole batch iterate: the training rollout
   went from 826 to 250 ms/step at 2048 envs as the policy stopped falling. `Config.mjx_iterations /
   mjx_ls_iterations` (default 16 × 8) cap this at model load. The batched-MJX recipe of 1–2 iterations
   does NOT work on this plant: the stiff loop closure (solref 0.002) needs a converged solve — the
   held stance explodes and the golden replay ends at tick 13 instead of 64. 8×8 to 100×8 replay the
   CPU fixture identically to the XML settings and hold a stance to 3e-3 rad over 3 s.

3. **The greedy eval ran the full 60 s episode cap.** A fixed 6000-step scan cost ~200 s on the V100
   every 50 rollouts (~120 s of training) even when all 16 envs fell within a second: 63% of the wall
   clock early in training (2.5k steps/s wall-clock against 7.7k in the loop). `PPO.evaluate` now
   exits when every env has ended, evals run every 400 rollouts (7.4 M steps), and a resumed run may
   change these two bookkeeping settings without being refused.

Tools: `tools/profile_step.py` (bare physics per solver variant + held-stance drift),
`tools/profile_env.py` (env step with the physics / auto-reset stubbed out), `bench.py --iterations N
--ls-iterations M --ppo`, `tools/replay_golden.py --iterations N --ls-iterations M` (the accuracy gate).

Measured on the V100 (`results/bench3_izar_solver_*`; env = 2048 envs under random actions with the
fixed bench, the flailing worst case; PPO iteration = rollout 32 steps + 4 epochs):

| solver cap | env steps/s | PPO steps/s | golden replay / stance drift |
|---|---|---|---|
| 100 × 50 (XML) | 3 940 | 2 990 | reference |
| 100 × 8 | 5 016 | 3 308 | identical / 3e-3 rad |
| 32 × 8 | 8 389 | 2 186* | identical / 3e-3 rad |
| 16 × 8 (default) | 9 783 | 2 155* | identical / 3e-3 rad |
| 8 × 8 | 12 648 | | identical / 3e-3 rad (CPU), 2.6e-2 transient (V100) |
| 1 × 4 | 56 090 | | plant explodes (NaN) |

\* `bench_ppo` carries ~13 s per iteration that the training runs do not show (their update is
0.16 s for the same minibatch count); the training runs' own phase timers below are the reference.

8 × 8 is the faster choice for the flailing regime and passes the same gates; 16 × 8 is kept as the
default for margin on contact-rich running (set `mjx_iterations=8` in a preset to switch).

Training regime (`runs/v2_s1_planar_s*`, 1024 × 18, cap 16 × 8, first 0.3 M steps): rollout 2.17 s
(120 ms/step), update 0.16 s, GAE + estimator 0.02 s, i.e. **7 600–7 700 env steps/s per V100**, and the
rollout gets cheaper as the policy stops falling (the archived 2048 × 64 runs went from 2 200 to
7 800 steps/s over their first 13 M steps). Because the batched rollout is latency-bound (solver
loops), more envs per rollout cost little: 2048 × 9 would roughly double this at the same rollout size.

CPU arm (JED, 64 envs × 288 steps on 72 cores): 4 660–4 760 env steps/s (`walk_mit/runs/v2_s1_s*`).

## Cross-checking against the CPU implementation

Another agent is porting the same design onto `walk_mit/` (classic MuJoCo, CPU). Two things
make the two implementations comparable:

1. `walk_v2/gait.py` runs on numpy too (`xp=np`), so the CPU arm can call the SAME control law
   and reward-term formulas, or diff its own against them state by state.
2. `tools/trace.py` writes a deterministic trace (fixed spec, sinusoidal residual, nominal plant,
   no noise, 12 ms delay, 300 ticks) with every per-tick number; `tools/compare_traces.py` reports
   functional agreement (targets/gains recomputed on the other's states) and the trajectory
   divergence time. `results/trace_mjx.json` is this side's trace. Agreement on the control law
   should be ~1e-6; trajectories diverge slowly (float32 vs float64 on a chaotic biped).

Throughput comparison: `bench.py --json` here vs the CPU stack's steps/s from its progress.csv.

S2 (free base) parity, 2026-09-10: the CPU arm had a golden fixture for S1 only, so before trusting
the S2 runs an S2 fixture was recorded with its `golden_v2.py --preset v2_s2_clean` (kept as
`results/golden_v2_s2_clean_seed0.npz`) and replayed through `tools/replay_golden.py --preset
v2_s2_free_easy`: 65 ticks on both arms (same fall tick), commit flags exact, per-tick reward
identical, per-channel newest-frame differences at the S1 level (torque 0.04, base rate 0.06-0.13
normalised units by tick 5, solver-level drift), phase channels in the known swapped order.

## Deploying a v2 policy on the robot

`walk_v2/export.py` writes a **version 2** bundle (`robot/deploy/bundle.py`), and the Pi runtime
for it is `robot/deploy/controller_v2.py` + `gait_v2.py` -- a numpy copy of `gait.py` with jax
removed, under the same safety governor, winding observer and joint map as the v1 (walk_mit) path.
`robot/deploy/README.md` has the full comparison; the three things specific to v2:

* **100 Hz out of the webui daemon's 200 Hz CAN loop.** The control law gets every second tick at
  its own `control_dt` and the force-control frame it produced is re-streamed on the one in
  between. This is what makes v2 deployable on the Pi 3B at all: the tick budget doubles to 10 ms
  while the nets get *smaller* than v1's (232k MACs against 305k), because the once-block replaced
  213 dims of stacked history. Only integer ratios are accepted.
* **The command is the task channel's run flag, not a velocity.** The panel's RUN / STOP pair
  writes `task[0]`, which is exactly the stoplight signal: RUN is the green light, STOP is a red
  one, and the policy keeps running at full gains and is asked to bring itself to a halt. A run
  always comes up STOPPED -- the approach crawls to the stance, the policy holds it, and nothing
  moves off until somebody presses RUN. Ending the run is a separate pair of buttons; the two are
  never the same thing. Headless: `run_policy.py --command-file /tmp/dash_command`.
* **The clock free-runs.** DASH-01 has no foot contact sensor, so the touchdown resync
  (`resync_kappa`) cannot happen on the robot. `controller_v2.note_contact()` is the hook if one
  ever exists.

Verification without torch, MuJoCo or jax: `robot/deploy/tests/test_v2_deploy.py` replays
`results/trace_mjx.json` -- the same fixture the two arms cross-check on -- through the deployed
runtime and diffs its targets, gains, clock and 33-dim observation frames against what the trainer
produced. Agreement is at the float32 rounding floor (~1e-5). It also pins the two things a port
gets wrong silently: the frame's phase channels are `[cos, sin]` on this arm (the CPU arm's
V2_CONTRACT.md documents the opposite order for its own implementation -- the known swapped-order
difference), and `pitch_reflex_rate_lp` is read from the bundle rather than assumed, because the
lineage has shipped both 0.9 and 0.0 and the difference is 0.4 rad of thigh target.

## Bring-up: holding the robot, starting the policy, letting go

The deployment question (2026-09-10): the operator holds the base on its stand, feet on the floor,
trunk roughly vertical, starts the policy, and releases a few seconds later. Training starts every
episode from one point -- the settled keyframe, at rest, feet flat, clock phase 0 -- so both the
held seconds and the release are off distribution.

`cfg.hold_enable` (opt-in, off in every preset; the contract presets are byte-identical) clamps the
six base DOFs to (key x/y/yaw, `hold_z`, `hold_roll`, `hold_pitch`) at every 1 kHz substep for
`EnvParams.hold_s` seconds -- an infinitely stiff hand, no fall scored while held -- then releases
with zero base velocity; `start_red_s` brings the episode up on a RED light. A kinematic clamp, not
a spring: the 1000 N m/rad three-wheel probe went NaN in 0.1 s. `tools/bringup_probe.py` sweeps the
release conditions and reports which termination fired. All numbers below: the best S2 runner
(`v2c_s2_free_dp2x_s0` `best_88473600.msgpack`), greedy, 16 envs, 8 s after the release.

* **The nominal case works, if the hold is long enough.** Held upright with both feet at the
  touching height: **16/16 for every hold >= 1 s** (1, 2, 3, 5, 10 s), running off at 3.2 m/s.
* **A sub-second release is a lottery**: 0.00 s 16/16, 0.02 s 16/16, 0.05 s 2/16, 0.10 s 16/16,
  0.15 s 8/16, 0.20 s 0/16, 0.30 s 14/16, 0.40 s 4/16, 0.60 s 0/16, 0.80 s 0/16. Release in the
  same tick the policy starts, or hold a full second -- never in between.
* **Attitude is the binding constraint and it is asymmetric.** Leaning BACK is fine (-5 deg 16/16,
  -10 deg 11/16); leaning FORWARD is fatal (+5 deg 0/16, *all 16 floor violations* -- the foot
  drives through the ground; +10 deg 0/16). Every roll is fatal (+-5, +-10 deg: 0/16) because a
  roll lifts one foot: 35 mm at 5 deg, 70 mm at 10 deg. The asymmetry, not the angle, is what
  kills: **both** feet 20 mm high is 16/16, **one** foot 35 mm high is 0/16, and -7 pitch / +7 roll
  (one foot up but leaning back) recovers to 12/16. The +-10 deg the operator can hold by hand is
  NOT inside the envelope -- it is roughly -5..0 deg of pitch and a couple of degrees of roll.
* **Height is forgiving**: -5 to +20 mm about the touching height is 16/16; +40 mm and +100 mm are
  0/16 (a real drop), -10 mm 11/16 and -20 mm 10/16 (feet pressed into the floor while held).
* **While held it marches in place** -- thighs 22 deg peak-to-peak, cams 13-16, hip rolls 11, toes
  lifting to 9 cm at 4 Hz, 22 % of peak torque, feet nearly stationary horizontally (0.12 m/s).
  It is not quiet in the operator's hands, and a human hold is compliant where this probe is rigid.
* **There is no passive stance to release into.** Zero action (the drive PD on `nominal_ctrl` plus
  the reflexes) falls in 1.0-1.3 s, 0/16, so "let go, let it stand, then start the policy" is not
  available on this plant: every hand-over variant (policy at +0.0, +0.2, +0.5, +1, +3 s) is 0/16.
* **The STOP flag does not stop it -- it makes it faster.** Brought up with the run flag DOWN
  (`task[0]` = 0, exactly what the runtime does -- "a run always comes up STOPPED"), the 88.5 M
  policy sprints anyway: never pressing RUN still gives 7.2 m at up to **4.2 m/s** before it tips
  (0/16), and pressing RUN at +2.0 / +0.5 / +0.0 s gives 8/16, 4/16, 3/16 at 4.1-5.0 m/s -- faster
  than the 3.2 m/s it runs on a green light, because `task[0]` = 0 with distance still to go is a
  state it only ever met past the finish line, where it never survived. That checkpoint predates
  the stop curriculum, so the runtime's STOPPED bring-up is fictional here **and the STOP button is
  not a brake**: a bench start on this policy is a runaway. Re-run this probe on the
  `v2c_s2_free_dp2x_stop_s*` seeds before trusting the RUN / STOP pair with a robot in hand.
* **None of it survives the randomization.** Paired on the same plants, no-hold vs held 3 s:
  `dr_scale` 0 (nominal plant, sensor noise only) **15/16 vs 9/16** -- the hold costs real margin as
  soon as the observation is noisy -- then 0.25: 0/16 vs 0/16, 0.50: 0/16, 0.75: 0/16, and at 1.0 a
  plain start dies 0.17 s in. For THIS checkpoint the binding constraint is its own robustness, not
  the bring-up (the 147 M robust runner, the one trained with the full DR / jitter / drop
  curricula, was lost in the quota outage between the 140 M and 145 M checkpoints).

Verdict: **no hand bring-up with the current policy.** The cheapest fix is a reset distribution
rather than a new curriculum -- initial base height, +-10 deg of tilt, a small base velocity, a
random clock phase and a red-light start, so that every release condition an operator can produce
is in distribution. Until then a release jig that drops both feet together, upright to slightly
back, is the only repeatable option.

## Status (2026-09-10)

* Local CPU: `smoke_test.py` passes; `train.py --preset v2_smoke` runs end to end (rollout,
  masked PPO update, estimator, symmetry loss, entropy/std anneal, curricula, eval, checkpoint).
* **S2 (free base, roll + yaw) started 2026-09-10 14:10**: `v2c_s2_free_fast` warm-started from the best S1
  checkpoint below, seeds 0/1/2 on 2 GPUs each (`runs/v2c_s2_free_dp2x_s0/s1/s2`, ~20.7k steps/s) plus one
  hedge `v2c_s2_free_dp2x_keepstd_s0` (preset `v2c_s2_free_fast_keepstd`: the warm start keeps the S1
  policy's std instead of re-inflating log sigma as the contract prescribes). The warm-started policy
  starts at stochastic ep_len ~80 on the free plant at std 0.70 (the S1 runner trained at 0.21) and
  climbs slowly (s0 166 at 6.6 M). Everything above this line in time was S1 (planar) only.
  The 4-GPU seed `v2c_fast_dp4z_s4` (4 x 1024) never stood (ep_len ~60 at 46 M, one KL 2.1 spike) and
  was cancelled.
* **First S2 runner on this arm (2026-09-10, 16:00)**: contract seed 0 (`runs/v2c_s2_free_dp2x_s0`,
  `v2c_s2_free_fast` warm-started from the 73.7 M S1 runner, 2 x 2048 envs x 9, ~21k steps/s) at 73.7 M
  steps: wheel-free greedy on the FREE plant (roll + yaw free) covers 94.6 m mean at 3.02 m/s over 16
  envs, 15 falls (one survives), no finish. The standalone greedy eval of that checkpoint
  (`results/v2c_s2_free_dp2x_s0_greedy_best.mp4`): all 16 envs reach 99.2-100.2 m in 30.8-31.8 s at
  3.15-3.25 m/s and fall at the line -- faster than the best S1 runner (2.88 m/s), as the free base
  allowed in RUN 8. That is ~1 h of S2 training on two V100s, ~2 h 05 for S1 + S2 end to end. The keeper
  then took 88.5 M (102.4 m at 2.99 m/s, 16/16 to the line; `best_88473600.msgpack`, local copy in
  `runs/v2c_s2_free_dp2x_s0/`). Seed 1 stood wheel-free at 73.7 M (0 falls, walking backwards) and then
  collapsed to the 1.5 Hz rail at 98 M (ep_len 44), as did the roll-wheel seed 0 at 72 M (ep_len 25):
  both cancelled, replaced by contract seed 3 and by seed 4 warm-started from the S2 88.5 M runner
  (`v2c_s2_free_dp2x_s2warm_s4`). Seed 2 (resumed) and the roll-wheel seed 1 are at ep_len ~370 / ~780.
  The rigid-hold probe (all three wheels at 1000 N m/rad) is numerically unstable (NaN within 0.1 s), so the
  transfer question stays at: the S1 gait does not survive on the free plant under 100 N m/rad holds.
* **Cold S2 loses to the slow-cadence rail (2026-09-10, 23:00)** -- the finding that governs the recipe.
  Four cold S2 seeds (three with the stop curriculum, one plain-contract control) all parked the gait
  clock on its 1.5 Hz lower rail and stopped learning (ep_len 42-62 at 14-32 M, greedy 0/16). The control
  rules the stop curriculum out, and an old-vs-new env parity test on a deterministic plant is exact
  (max |d reward| and |d obs| = 0 over 40 steps, both presets), so the patch is inert with the lights off.
  Clock history (Hz, median): cold S1 PLANAR 3.6 at 2 M then 4.0 for ever; warm S2 free 4.0, dips to 1.5
  at 18-22 M, recovers to 3.8 by 30 M; cold S2 free 4.0 at 6 M then 1.5 from 10 M on (or straight to 1.5).
  So **the rail is a free-plant attractor**: the planar stage is where a 4 Hz gait is discoverable, and a
  warm start (from S1 *or* from an S2 runner) is what keeps a free-plant seed off the rail. This corrects
  the earlier "S1 carries nothing over": the S1 policy transfers no *balance* (it falls in ~1 s on the free
  plant) but its *cadence prior* is load-bearing. No S1 stage needs re-training -- the runners are on disk.
* **Frequency-floor curriculum** (the structural fix for the rail, opt-in): `gait_freq_lo_start` (3.0) ramps
  down to `gait_freq_hz[0]` (1.5) over `gait_freq_floor_steps`, so a slow gait is not selectable while the
  running gait forms. Verified locally: the minimum action gives 3.00 / 2.25 / 1.50 Hz at the three
  curriculum points and the contract preset still gives 1.50. Preset `v2c_s2_free_fast_floorstop` = floor
  (60 M) + the stop curriculum, i.e. the cold-S2 "no S1 stage" recipe; seeds `runs/v2c_s2_floorstop_s8/s9`.
* **Why the robot will not stop, measured (2026-09-11, 02:30)** -- `evaluate.py` now logs the gait clock
  through the red phase and the state at the fall (`f_red_min/max`, `v_end`, `f_end`). Probing the banked
  2.95 m/s runner at 8 s AND 20 s ramps (0.36 and 0.15 m/s^2): it falls every episode, spends **0%** of the
  red phase slow, and **at the moment of the fall it is doing 2.1-3.35 m/s -- faster than its own 2.65 m/s
  average -- with the clock dropped from 4.0 to 2.4-3.8 Hz**. Asked to slow it lowers the cadence and
  ACCELERATES: longer strides, more push per stride, overstride, topple. Lowering the gait clock does not
  slow this plant down, and the policy has no braking behaviour in its repertoire. Survival time scales with
  the ramp (1.2 s at 2 s, 2.9 s at 8 s, 5.8 s at 20 s), so it is the command it cannot tolerate, not the
  braking effort. The reason it never learned one is the REWARD SHAPE: the Gaussian tracking term pays
  0.006 at the 1.8 m/s error the policy actually operates at -- a flat region carrying no direction
  information. `stop_track_laplace` uses exp(-|e|/sigma), which pays 0.105 there (17x) and has gradient
  everywhere, so slowing a little now beats not slowing. Seeds `runs/v2c_s2_stoplap_s22/23/24`.
* **The gait shaping was switched off for the whole deceleration (2026-09-11, 01:45)** -- a structural
  mismatch, not a tuning problem, and the likeliest reason braking plateaued at `reward_terms/stop` ~0.033
  (perfect would be ~0.38). `cmd_speed` was binary -- `v_ceiling` while running, **0** the instant the light
  turned -- and `gait_on = cmd_speed >= gait_cmd_gate` (0.25) gates the entire gait block: air-time credit,
  swing floor, stance time, clearance, phase contact. So through the whole 8 s ramp, 2.5 -> 1.0 -> 0 m/s,
  the robot had NO gait shaping at all -- exactly the intermediate-speed regime this morphology has never
  been shaped in and historically cannot balance through (the command-objective runs all converged to
  stand/creep; the sprint objective is what produced runners). Under `stop_cmd_continuous` `cmd_speed` now
  follows the ramp, so the shaping tracks the command down and releases only at a genuine standstill.
  Verified: at 2.5 / 1.87 / 1.25 / 0.62 m/s of commanded speed the gait terms are live, and they switch off
  at 0.06 m/s.
* **The deceleration demand was 7x the requirement (2026-09-11, 01:00)** -- the reason the stop still
  would not train. With the continuous command AND 5x the stop reward, `v2c_s2_stophard_s14` still fell on
  red 16/16, spending 0% of the red phase slow; it only survived longer (1.2 s vs 0.8 s). The arithmetic
  from 2.9 m/s: `stop_decel_s` 2.0 asks 1.45 m/s^2 and stops in 2.9 m -- a hard stop, which needs a capture
  step (foot planted AHEAD of the CoM). No policy in this project has ever learned one; walk_mit m3 measured
  feet landing ~8 cm BEHIND the CoM while falling. The stated requirement -- stop within ~20 m of the line --
  is 0.21 m/s^2 over ~14 s. `v2c_s2_free_fast_stopgentle`: `stop_decel_s` 8.0 (0.36 m/s^2, ~11.6 m, inside
  the budget), red phases 10-14 s so the stop can be completed AND held (`stop_hold_s` 1.0 at
  `stop_speed_eps` 0.25 = a finish), green 6-12 s, `sprint_brake_m` 20.0 so coasting past the line is not
  billed. Seeds `runs/v2c_s2_stopgentle_s16/17/18`, warm-started from the 2.95 m/s runner below.
  Runners banked from the 2 s-decel round (all `best_speed_44M.msgpack`, 44 M steps ~= 40 min on two V100s):
  s14 103.4 m at **2.95 m/s** (the fastest policy this project has produced on the free base), s15 103.3 m
  at 2.65, s13 97.9 m at 2.84.
* **The binary run flag is a step disturbance (2026-09-11, 00:10) -- measured, and the reason the first
  stop curriculum taught nothing.** `v2c_s2_stopwarm_s6` warm-started from the 88.5 M S2 runner reached a
  wheel-free greedy runner FASTER than any run so far -- 99.5 m at 2.65 m/s at 59 M (~55 min on two V100s),
  100.5 m at 2.84 m/s at 73.7 M (`best_speed_73728000.msgpack`) -- then railed at 84 M. But with the lights
  on (`--stoplight 1.0`) that same policy **falls 0.7-0.8 s after every red light, 16/16 episodes, spending
  0% of the red phase slow**: it never brakes, and `reward_terms/stop` stayed ~0.004 against ~4.7 of running
  income for 45 M steps. Two causes, both now addressed in `v2c_s2_free_fast_stoplight_hard`:
  1. **Incentive**: at the contract `w_stop_vel` 0.4 a red phase pays at most 0.4/step against ~5 for
     running, so braking was worth almost nothing and hard to find. Now 2.0, `decel_sigma` 0.8,
     `stop_decel_s` 2.0, and only 35% of episodes lit so the running signal stays strong.
  2. **Interface**: `task[0]` flipped 1 -> 0 in a single tick, a step change on an input the whole gait is
     conditioned on. With `stop_cmd_continuous` the channel carries the TARGET SPEED instead (normalised by
     `v_ceiling`): 1 while running, then the same ramp the stop reward tracks, down to 0. Continuous to
     learn, and it is exactly the speed command a deployment panel would drive.
* **Deployment note for the stop curriculum (for whoever owns `robot/deploy/controller_v2.py`)**: a
  policy trained with the lights obeys the *observation*, not the distance -- `task[0]` (actor obs, the once-block) is 1 while running and, under
  `stop_cmd_continuous`, the normalised target speed while it should brake (0 = stand), and `task[1]` is the
  clipped distance-to-go. So the runtime gets a live STOP command for free: drive `task[0]` to 0 and the
  policy decelerates to a standstill wherever it is, then hold it at 0 to keep it standing; set it back
  to 1 to run again. That is exactly what the red/green phases train. A policy trained WITHOUT the
  lights (every checkpoint before 2026-09-10 23:00, including the 88.5 M runner being exported now) has
  only ever seen `task[0]` drop at the 100 m line and will not brake on command -- do not expose a stop
  button backed by it. `stop_speed_eps` / `stop_hold_s` (0.15 m/s for 1.0 s, 0.25 m/s in the hard preset)
  are the environment's own "stopped" test and are the sensible defaults for the panel's readout.
* **Stop curriculum (2026-09-10, 22:30) -- red light / green light**: no runner on either arm ever
  stops: the post-line phase is a cliff it meets once per episode at 3 m/s and never survives, so the
  finish bonus is unreachable. New opt-in preset `v2c_s2_free_fast_stoplight` (all fields default off;
  the contract presets are byte-identical in behaviour): in a competence-gated fraction of episodes
  (0 -> 0.5 over 20 M once ep_len > 600) the run flag drops at random times before the line for 2-4 s
  (obs task[0] -> 0 while task[1], the distance countdown, stays > 0), the speed income stops and the
  stop term pays for tracking a target speed that ramps from the speed at the switch to 0 over
  `stop_decel_s` = 1.5 s, then for standing still; after 3-8 s the light turns green and the income
  resumes. The line is one more red light with the same deceleration target. `evaluate.py --stoplight P` puts lights into an eval.
  puts lights into an eval. Decision (user): S1 is dropped -- the S1 -> S2 warm start carries nothing --
  and S2 is trained cold at 220 M steps (~3 h on two V100s); seeds `runs/v2c_s2_free_dp2x_stop_s{0,1,2}`.
  `train.py --resume auto` now falls back to the previous checkpoint when the newest is truncated (four
  were, by the outage).
* **Outage 2026-09-10 16:52-21:50**: every training job died with `OSError: [Errno 122] Disk quota exceeded`
  (the Izar home is shared with the CPU arm's history: `walk_mit/runs` 71 GB, old `dash-mit-*.out` logs
  5 GB, pip + uv caches 34 GB, venvs 28 GB; `walk_v2/runs` was 5.3 GB). The VPN was down at the same time,
  so the monitor could not see it. Freed: the pip/uv caches and every intermediate `ckpt_*` of finished
  runs (kept: `best*`, the newest one or two per run, the 130 M S1 runner) -> `walk_v2/runs` 1.6 GB; the
  four S2 runs resumed from their last checkpoints (s0 175 M, s2 135 M, s3 70 M, s2warm_s4 80 M).
  Lost in the gap: the S2 seed-0 policy at 147 M that ran 16/16 to the line at 2.48 m/s **with the full
  randomization, jitter and drop curricula active** (the first robust runner on either arm) sat between the
  140 M and 145 M checkpoints (which stand / walk at 0.25 m/s) and the distance-first keeper preferred the
  88.5 M policy by 0.4 m; the resumed run now has the `best_speed.msgpack` keeper. S2 seed 0 collapsed at
  117 M when its randomization ramp reached full strength (all the late collapses follow the DR / jitter /
  drop gates opening at ep_len 1200, then the gait clock slides to the 1.5 Hz rail) and recovered by 147 M.
  Seed 4, warm-started from the S2 88.5 M runner, reached ep_len 483 at 15 M and 99.7 m at 2.29 m/s greedy
  at 73.7 M: warm starts transfer within the same plant, only the planar-to-free jump carried nothing.
* **S2 findings (2026-09-10, 15:00)**. (1) The S1 runner (73.7 M) on the free plant, greedy, falls
  sideways in 0.7-1.4 s with the pitch wheel at full and in 0.7-1.2 s without it
  (`runs/v2c_s2_free_dp2x_s0/greedy_s1best_on_free_assist{1.0,0.0}.json`): the S1 policy carries no lateral
  balance, so the contract's warm start gives S2 nothing to stand on and the three contract seeds start at
  ep_len ~80 (the CPU arm has not run S2 yet; no S2 golden fixture existed either, see the cross-check
  section). (2) The contract seed 1 nevertheless climbed to stochastic ep_len 618 at 31 M and its
  pitch-wheel fade opened there; seed 0 sits at ~160; seed 2 and the keep-std hedge were stopped.
  (3) New opt-in preset `v2c_s2_free_fast_rollassist` (`roll_assist_kp/kd` 100/10 on the base roll DOF,
  same fade scalar as the pitch wheel, billed in `assist_pen`, ignored on the planar plant): seeds
  `runs/v2c_s2_free_dp2x_roll_s0/s1` warm-started from the 73.7 M runner. (4) With pitch AND roll held
  at full the greedy S1 runner still falls in 1.4-1.8 s (`greedy_s1best_bothwheels`): the frames show the
  base yawing ~90 deg within the first second, i.e. the gait's yaw impulse, absorbed by the planar tree,
  spins it on the free plant. Added a yaw wheel (`yaw_assist_kp/kd`, preset `v2c_s2_free_fast_basewheels`
  = all three base wheels on the same fade) and a three-wheels probe of the S1 runner
  (`runs/v2c_s2_free_probe_basewheels`). The roll-wheel seed 0 became a long-lived stander early
  (ep_len 1785 at 32 M, return ~-100: standing, not running).
* **S1 keeps improving past the fade**: `v2c_fast_dp2x_s2` at 130 M (greedy, wheel-free) runs 16 of 16
  to ~103 m in ~36 s at 2.88 m/s (`results/v2c_fast_dp2x_s2_greedy_130M.mp4`, checkpoint
  `ckpt_130351104.msgpack`), after a 103 M collapse to 1.8 m and a 118 M policy at 2.06 m/s that drifted
  to 112 m; 135 M is back at 2.43 m/s. Still no stop phase (falls at the line). `train.py` now also keeps
  `best_speed.msgpack` (fastest eval covering the full 100 m) next to the distance-first `best.msgpack`.
* **Best GPU-arm policy so far:** `v2c_fast_dp2x_s2` (2 × 2048 envs × 9, contract schedules) at 73.7 M
  steps, ~63 min of training on two V100s: the wheel-free greedy eval runs 16 of 16 envs to 102–105 m
  in ~38 s at 2.66–2.76 m/s, then falls at the line (no stop phase learned yet — the CPU arm's
  checkpoints also "never stopped"); `runs/v2c_fast_dp2x_s2/best.msgpack`, video
  `results/v2c_fast_dp2x_s2_greedy_best73M.mp4` — in the CPU arm's speed band (2.4–3.1 m/s at 42 M).
* Izar (V100): full smoke test passes on the GPU. **First wheel-free runner on this arm:**
  `v2c_s1_planar` seed 1 (contract sizing, one V100, 1 h 45 min) — the 45 M checkpoint, greedy with the
  assist at 0, runs 15 of 16 envs to ~100–108 m in 60 s at 1.66–1.81 m/s (`slurm/izar_eval.sbatch`,
  video `results/v2c_s1_planar_s1_greedy_45121536.mp4`; the CPU arm's 42 M checkpoint: 146–189 m
  at 2.4–3.1 m/s). Same design, same stage, same wheel-free behaviour; the GPU policy is slower. Seed 0 (low rail) is at ep_len 932 / return +356 at 47 M.
  Speed runs: `v2c_fast_dp4y_s1` (4 × 2048, scaled recipe, 42k steps/s) and the 2-GPU hedge seeds
  `v2c_fast_dp2x_s1/s2` (~21k steps/s each). Stopped (checkpoints kept): `v2_*`, `v2b_*`, the
  stiff/no-assist experiments, `v2c_s1_planar_fast_s0` (NaN at 36.5 M, fixed) / `_s1`,
  `v2c_fast_dp4_s0`, `v2c_fast_dp4x_s0`, `v2c_fast_dp2x_s0`.
* Cross-check with the CPU arm: golden fixture `walk_mit/golden/v2_s1_clean_seed0.npz` replays with
  exact commit flags and rewards identical over the first 20 ticks; control-law agreement 5e-6 on the
  traces. Policy-level comparison (same preset, seed, budget; greedy dash eval) pending the runs.
* Lyra: account QOS `disable`; scripts ready (`slurm/lyra_*`).
