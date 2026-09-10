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

## Status (2026-09-10)

* Local CPU: `smoke_test.py` passes; `train.py --preset v2_smoke` runs end to end (rollout,
  masked PPO update, estimator, symmetry loss, entropy/std anneal, curricula, eval, checkpoint).
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
