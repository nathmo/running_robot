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
