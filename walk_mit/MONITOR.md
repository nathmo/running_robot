# Overnight watch playbook — walk_mit MIT-drive runs (2026-08-26/27)

The user is asleep. Deliverable by morning: **a policy that walks in sim** (m2, ideally m3) on the
MIT-mode drive, with 2-hourly eval videos in `walk_mit/monitor/` for review. Priority order set by
the user: a working policy on a relaxed (idealized) robot BEATS no policy on the real-constraint
robot. Realism may be relaxed ONLY via the pre-baked `_easy` presets — never edit code or configs
mid-night, never touch `training/`, never invent new relaxations.

## What is running

Cluster: `ssh ncmorand@jed.hpc.epfl.ch` (VPN must stay up). Repo there: `~/running_robot`,
this arm's code+runs under `~/running_robot/walk_mit/`. Job name `dash-mit`, logs
`~/running_robot/dash-mit-<jobid>.out`.

Chain (job IDs in `walk_mit/monitor/status.log` header, or `squeue -n dash-mit`):
- `mit_m2_s0`, `mit_m2_s1` — preset `walk_fwd_m2_mit`, 100 M steps (~4.7 h at ~5,900 fps),
  warm from `training/runs/ladder3_m2_s0` (parent walker: ep_len 2978 at 3 Hz drive).
- `mit_m3_s0/s1` — preset `walk_fwd_m3_mit`, 80 M, `afterany` on its seed's m2 job,
  warm from that m2 run's newest checkpoint.
- **`imp_m2_s0/s1`** — preset `walk_fwd_m2_mit_imp`, 100 M, **COLD** (from scratch): +4 per-leg
  kp/kd impedance action dims (action 30, obs 596 — its own lineage, nothing loads into it).
  This is the user's priority arm. `imp_m3_s0/s1` — 80 M, `afterany`, warm from its seed's imp m2.

Healthy references: fps ≥ ~4,000; **warm** m2 ep_len_mean should recover toward >2,000 within
~20 M steps; m3 anything >600 is progress (ladder3_m3 got ~601), >1,500 is a win. **Cold** imp
arms are slow starters — judge them like a cold ladder3_m2: nothing alarming about ep_len ~250-400
for the first 30-40 M; what matters is a rising trend after that. Episode cap is 12,000 steps
(60 s at 200 Hz); fall ≈ 240.

## Every tick (30 min)

1. `ssh ncmorand@jed.hpc.epfl.ch 'bash running_robot/walk_mit/slurm/status_remote.sh'`
2. From each active run's progress.csv (header + last row printed): extract
   `time/total_timesteps`, `rollout/ep_len_mean`, `rollout/ep_rew_mean`, `time/fps`.
3. Append ONE line per active run to `walk_mit/monitor/status.log`:
   `HH:MM  run  steps  ep_len  ep_rew  fps  note`. Keep a short trend memory: the log itself is
   the memory — read its tail before judging a trend.
4. Act per the rules below. If nothing needs action, that's a healthy tick — no action IS the
   correct action most ticks.

## Health rules (in priority order)

- **VPN/ssh dead** → retry once after 60 s. Still dead: nothing can be done remotely; log it,
  keep ticking (jobs run fine without us; only monitoring is blind).
- **Job gone from queue early + Traceback in its .out** → resubmit its exact sbatch line from
  `walk_mit/slurm/launch_overnight.sh` (safe: `--resume auto` continues from newest checkpoint;
  WARM only applies to a fresh start). ONE resubmit per job per failure signature; a second
  identical crash = stop that seed, note for morning.
- **Job killed by wall/node (no Traceback)** → resubmit same line (resume). No retry cap.
- **m3 job FATALed at the WARM check** (its m2 died with no checkpoint) → fix m2 first (rule
  above), then resubmit the m3 line with `--dependency=afterany:<new m2 jobid>`.
- **NaN anywhere in progress.csv last row** → scancel that job; launch its `_easy` twin (below).
- **Quality gate, WARM arms (mit_*), checked only after a rung has ≥ 20 M steps of its own**: if
  ep_len_mean < 300 AND the last 3 ticks show no upward trend → that seed's MIT arm is failing;
  LAUNCH the `_easy` twin (do NOT cancel the original — the partition was empty, let it keep trying):
  `sbatch --export=ALL,PRESET=walk_fwd_m2_mit_easy,STEPS=100000000,SEED=0,NAME=mit_m2_easy_s0,WARM=training/runs/ladder3_m2_s0 walk_mit/slurm/jed_train.sbatch`
  (adjust SEED/NAME; for m3: PRESET=walk_fwd_m3_mit_easy, WARM=walk_mit/runs/mit_m2_s<seed> or
  the easy m2 if that's the one that worked, STEPS=80000000.)
- **Quality gate, COLD imp arms — more patient: threshold is ≥ 40 M steps.** If ep_len_mean < 300
  AND flat over the last 3 ticks after 40 M → launch the imp easy twin, COLD (no WARM):
  `sbatch --export=ALL,PRESET=walk_fwd_m2_mit_imp_easy,STEPS=100000000,SEED=<s>,NAME=imp_m2_easy_s<s> walk_mit/slurm/jed_train.sbatch`
  Do not cancel the original. Note in the log which relaxation was invoked and why.
- **Both m2 seeds AND both easy arms flat at ~07:00** → stop launching, write the honest summary.

## Every 4th tick (2 h): video

Film the user's priority first: the best **imp** arm once any imp run shows ep_len_mean > 400;
otherwise the furthest-along rung, best seed by ep_len_mean (m3 > m2 once m3 has >5 M steps).
When time allows, alternate: imp arm this video tick, best warm arm the next — the morning
comparison needs both on film:

```
# R = chosen run name, e.g. mit_m2_s0
ssh ncmorand@jed.hpc.epfl.ch "ls -1t running_robot/walk_mit/runs/$R/ppo_*_steps.zip | head -1"
# -> note step count N; then (PowerShell, laptop, from repo root):
mkdir -Force walk_mit\runs\$R
scp ncmorand@jed.hpc.epfl.ch:running_robot/walk_mit/runs/$R/{resolved_config.json,curriculum.json,progress.csv,ppo_<N>_steps.zip,ppo_vecnormalize_<N>_steps.pkl} walk_mit\runs\$R\
python walk_mit\evaluate.py --run walk_mit\runs\$R --video walk_mit\monitor\<HHMM>_$R.mp4
```

(`evaluate.py --help` if flags differ; `--episodes 1` if available, to keep it quick. If brace
expansion trips PowerShell's scp, copy the five files one per scp call.) Log the video filename in
status.log. If local rendering fails once, fall back to copying `training_plots.png` (if present)
and note it — do not burn a tick debugging the renderer.

## Stop conditions

- Both m3 rungs (or their easy twins) reached their step budget and a final video is rendered →
  write a 5-line summary at the top of status.log, render one last video per surviving arm, stop
  the loop.
- The user shows up in the conversation → hand over and stop.
- Otherwise keep ticking until ~09:00.

## Hard don'ts

No code/config edits (cluster or laptop). No new presets. No scancel except the NaN rule. No
training/ modifications. No commits. No second video attempt in the same tick if rendering fails.
