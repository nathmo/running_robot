# walk_mit — the MIT-drive walking policy

Self-contained working set for ONE question: **does the m2/m3 walker get better when the sim drive
is the measured MIT-mode (force-control) drive instead of the servo-mode planner?** Created
2026-08-26 as a clean copy of `training/` core (at commit d1723c1 + the `_mit` presets); none of
the study/diagnostic scripts came along. `training/` stays the archive; new work on this line
happens here.

## What is different from training/

- Presets `walk_fwd_{m2,m3,m4}_mit` — the ladder rung on the MIT-mode drive: `drive_bandwidth_hz`
  3.0 → 12.0 (stand-in; the servo planner the 3 Hz modelled is bypassed in MIT mode, measured
  ~6–7 ms command→response), `drive_delay_ms` 25 → 7, command start box 0.25 → 0.50 m/s.
  Everything else is bit-identical to `walk_fwd_m2..m4`.
- Presets `walk_fwd_{m2,m3}_mit_easy` — same, minus plant DR, sensor noise, pushes/trips, and
  control-timing jitter. The authorized fallback: a policy for an idealized robot beats no policy.
- `slurm/jed_train.sbatch` — identical to training's except it runs `walk_mit/*.py`.

## Layout

- core: `config.py env.py train.py evaluate.py teleop.py smoke_test.py fourier_gait.py
  cpg_gait.py asym_policy.py domain_rand.py plot_training.py` + `model/` (XMLs + LUTs)
- `runs/` — training outputs (cluster-side: `~/running_robot/walk_mit/runs/`)
- `monitor/` — overnight status log + 2-hourly eval videos (laptop-side)
- `MONITOR.md` — the overnight watch playbook

## Warm-start lineage

`training/runs/ladder3_m2_s0` (100 M steps, ep_len 2978, 3 Hz drive, obs 556) → `mit_m2_s*` →
`mit_m3_s*`. Obs width 556 (`obs_privileged_critic=True`) throughout; checkpoints do NOT load
into the obs-550 walk_fwd/teleop family.

## Deliberately not copied

Foot/ankle/crouch study scripts, montage tooling, CPG chain launchers, the runs_dl archive.
They live on in `training/` and git history.

## DASH-01 Walker v2 (2026-09-09) — the latched-spec lineage

Implements the "DASH-01 Walker v2" artifact as the next lineage on this folder. Interface
contract for a second implementation (the GPU port): `V2_CONTRACT.md`; parity fixture:
`golden_v2.py` + `golden/*.npz`.

- `gait_v2.py` — the 44-dim latched spec (one Fourier series per joint family, kp/kd profiles,
  frequency, roll-reflex gains, the five relationship knobs Δ/s/o) + 6 per-tick residuals;
  offset and roll reflex enter (+,+) (the FK sign rule), pitch reflex stays (+,−).
- `env.py` (`action_mode="latched"`) — latch register + commit flag (obs[376]), contact-triggered
  clock resync, substep-granular (target, kp, kd) delay ring drawn 6–18 ms, phase-scheduled
  impedance, winding thermal node (τ_th 45 s), wind/gusts/lateral pushes, 33-dim frame × 10 at
  stride 2 + once-block + 25-dim privileged tail, LP-yaw / one-sided-height / lane / thermal /
  per-cycle-spec / knob terms, `mirror_perm_sign()` for the symmetry loss, the library variant
  (`spec_source="library"`, `raibert.py`), and the return-map hooks the solver uses.
- `masked_policy.py` — log-prob/entropy of the latched dims scored on commit ticks only.
- `sym_ppo.py` — PPO + the mirror-equivariance loss on the knobs and the residual.
- `model/dash01_v2.xml` (`make_v2_plant.py`) — rigid loop closure + a series spring along the
  foot strut calibrated to 5 mm at 1 BW (13.06 kN/m, DR ±50 %), 20 DOF; `fit_drive.py`
  (armature → 0 from the sim Bode); `calibrate_loop_spring.py` (the bisection, and why the 5 mm
  cannot come from the artifact's pushrod-tip location, kept as `--spring rod`).
- `library/` — Stage 0 return map + Newton fixed points + Floquet multipliers, Stage 1 CMA-ES
  library solve, Stage 4 closed-loop search, Stage 5 residual absorption.
- presets `v2_s1` (planar sandbox), `v2_s2` (free), `v2_lib_s1/s2` (library variant),
  `v2_returnmap`, `v2_*_clean`; cluster chain `slurm/launch_v2.sh`, gate job
  `slurm/jed_v2_bench.sbatch`; throughput `bench_v2.py`; margin axes in `tools/eval_envelope.py`.

    python walk_mit/train.py --preset v2_s1 --steps 300000000 --n-envs 64 --n-steps 288 --subproc
    python walk_mit/train.py --preset v2_s2 --warm-start walk_mit/runs/v2_s1_s0 ...
