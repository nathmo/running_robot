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
| 16 × 8 (default) | 9 783 | | identical / 3e-3 rad |
| 1 × 4 | 56 090 | | plant explodes (NaN) |

\* `bench_ppo` carries ~13 s per iteration that the training runs do not show (their update is
0.16 s for the same minibatch count); the training runs' own phase timers below are the reference.

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
* Izar (V100): full smoke test passes on the GPU. `v2_s1_planar` seeds 0/1 are training (jobs
  3144881/82, contract sizing 1024 × 18, cap 16 × 8, 300 M steps, ~7.7k steps/s each at the start).
  The first attempt (2048 × 64 / minibatch 16 384, 4× fewer updates per sample) reached 13 M steps
  at 7.8k steps/s but learned slower per step than the CPU arm (ep_len 105 vs 430–540 at 13 M);
  archived on Izar as `runs/*_2048x64`. Compare curves with `tools/compare_cpu_gpu.py`.
* Cross-check with the CPU arm: golden fixture `walk_mit/golden/v2_s1_clean_seed0.npz` replays with
  exact commit flags and rewards identical over the first 20 ticks; control-law agreement 5e-6 on the
  traces. Policy-level comparison (same preset, seed, budget; greedy dash eval) pending the runs.
* Lyra: account QOS `disable`; scripts ready (`slurm/lyra_*`).
