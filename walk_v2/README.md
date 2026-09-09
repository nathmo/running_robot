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
  bench.py           env steps/s vs batch, PPO iteration time (the Lyra benchmark)
  smoke_test.py      the invariants (gait mirror, thermal anchors, delay, plant fit, obs layout, mask)
  tools/eval_envelope.py   the §08 margin gate (wind, tilt, friction, delay, mass, thermal, pushes, lane)
  tools/trace.py, compare_traces.py   cross-implementation agreement protocol (see below)
  gait_lib/          §09: return map + Newton + Floquet (solver.py), CMA-ES (cmaes.py), Stage 4/5 (search.py)
  slurm/             Lyra job scripts + cluster README
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

# GPU (Lyra): see slurm/README.md
sbatch walk_v2/slurm/lyra_bench.sbatch
sbatch --export=ALL,PRESET=v2_s1_planar,NAME=v2_s1_planar_s0,SEED=0 walk_v2/slurm/lyra_train.sbatch
```

Checkpoints: `runs/<name>/ckpt_<steps>.msgpack` (+ `.json` with the curriculum state), `final.msgpack`;
`--resume auto` continues, `--warm-start <file>` carries weights + obs stats into a new stage
(count capped, variance floored, log_std re-inflated — the walk_mit warm-start rules).

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

## Status (2026-09-09)

* Local CPU: `smoke_test.py` passes; `train.py --preset v2_smoke` runs end to end (rollout,
  masked PPO update, estimator, symmetry loss, entropy/std anneal, curricula, eval, checkpoint).
* Lyra: venv built by `~/venvs/make_dash_v2.sh`, code copied to `~/running_robot/walk_v2`, but
  the user's Lyra association is QOS `disable` (no GPUs schedulable) — see `slurm/README.md`.
  `lyra_bench.sbatch` is the first job to submit once SCITAS enables the account.
* Nothing is trained yet; no policy claims are made.
