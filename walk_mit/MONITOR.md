# RUN 7 (2026-09-01): sprint_m4/m5/m6_mit — THE MILESTONE LADDER — ACTIVE playbook, supersedes below
#
# CONTEXT: RUN 6 closed with the project's first real runner (sprint_m3_mit: 16/16 greedy 100 m
# dashes, s0 32.5 s at 3.07 m/s). RUN 7 climbs the base-DOF ladder on the SAME sprint objective:
# m4 frees Y (lateral), m5 frees ROLL (the historical wall — but every "m5 is a wall" verdict came
# from command-objective runs never asked to move; sprint m5_stiff@600M did locomote), m6 = fully
# free. No joystick, speed only, per the user.
#
# CHAINS (afterany, warm from the previous rung's newest checkpoint; 300 M each, ~12 h at ~7k fps):
#   s0: m4=66327760 -> m5=66327761 -> m6=66327762   (m4 warm from sprint_m3_mit_s0)
#   s1: m4=66327763 -> m5=66327764 -> m6=66327765   (m4 warm from sprint_m3_mit_s1)
# Presets sprint_m{4,5,6}_mit = sprint_m3_mit kwargs + the rung's base_lock +
# warmstart_reset_log_std=True (parent std 0.25 -> re-inflated 1.0; anneal re-runs per rung).
# All ramps restart per rung: assist 60 M fade, DR, jitter 80 M, sprint line 25->100 m.
#
# Every tick: squeue; for the RUNNING rung(s) pull progress.csv last row
# (total_timesteps, ep_len_mean, ep_rew_mean, reward_terms/fwd_speed, curriculum/swing_frac_min,
# reward_terms/height, reward_terms/hip_roll [m5+: the roll-authority readout],
# curriculum/pitch_assist, train/std, fps); grep the active job logs for
# 'log_std reset|warm-started|Traceback|fatal|srun: error'. One line per active run to
# walk_mit/monitor/status.log.
#
# SIGNALS per rung (warm hop, so judge RECOVERY speed, not cold-start patience):
#   - the log MUST show "warm-start log_std reset ... (std 1.00)" at rung start — if absent, the
#     hop is running on std 0.25 and will not explore the new DoF: flag it, do not kill it.
#   - ep_len: expect an early collapse vs the parent (new DoF + re-eased ramps make it look GOOD
#     first, then the fades bite ~60-80 M). Healthy = recovery trend after the fades.
#   - fwd_speed income ~5.5 (2.75 m/s) is the m3 reference; m4/m5 will run lower early. A rung
#     that ends its 300 M with fwd_speed < 2 (1 m/s) has NOT learned to run at that milestone.
#   - m5 specifically: hip_roll penalty magnitude and ep_len vs the ~2-3 s historical roll wall.
#   - crouch pen ~0.001 (instant adoption is the norm).
# HEALTH: crash -> resubmit ONCE per signature with the same sbatch line (the chain's later rungs
# survive, afterany fires anyway — but a resubmitted rung loses its slot order: note it).
# NaN or second identical crash -> stop that chain, leave the other running, note for the user.
# NEVER early-stop a rung for looking bad mid-anneal. No code/config edits mid-campaign.
#
# FILM: when a rung finishes (final_model.zip appears), pull final + vecnormalize + curriculum,
# greedy eval N=8 + 40 s film locally, frames at 0.5 s spacing before ANY gait claim, one line
# verdict to status.log. m6_s* finals get the full treatment + memory update.
#
# ---------------------------------------------------------------------------------------------
# RUN 6 (2026-08-31): sprint_m3_mit — THE OBJECTIVE PIVOT — CLOSED (see FINAL VERDICT in status.log)
#
# CONTEXT: three command-objective fix rounds (imp_m3b/c/d) each cured their diagnosed defect;
# none walked. imp_m3d final: pinned-command endurance REGRESSED (0/5 episodes >= 20 s at both
# speeds). The only genuine locomotion in project history is sprint-objective (m5_stiff@600M:
# 100 m at 2.84 m/s roll-free). RUN 6 = sprint objective on the MIT plant.
#
# RUNS: sprint_m3_mit_s0 = jobs 66282757 -> 66282758 (afterany resume chain, 24 h wall each);
#       sprint_m3_mit_s1 = jobs 66282759 -> 66282760. 600 M each, COLD (own obs/action lineage:
#       295/28). Preset sprint_m3_mit: _sprint200('m3') + MIT drive 12 Hz/7 ms + crouch 0.05 +
#       imp dims + std anneal 0.25 + pitch assist (60 M fade) + jitter hardening.
#       ~28 h per seed at ~6,000 fps -> finishes ~2026-09-01 late.
# resubmit: sbatch --export=ALL,PRESET=sprint_m3_mit,STEPS=600000000,SEED=<0|1>,NAME=sprint_m3_mit_s<0|1> walk_mit/slurm/jed_train.sbatch
#
# Every tick (30 min): status pull for sprint_m3_mit_s0/s1 (progress.csv), error grep on all four
# job logs, one line per run to status.log. NOTE: sprint runs have NO cmd_scale — the signals are:
#   1. rollout/ep_len_mean climbing past the ~200 passive-fall floor (the July m3_reactive wall);
#      >600 = beating ladder3_m3; >1500 = real balance-while-running
#   2. reward_terms/fwd_speed (dense speed income) and sprint distance covered (info/sprint)
#   3. curriculum/swing_frac_min — sprint cannot be farmed by standing, so swing should rise
#      WITHOUT a floor penalty; watch it against the 0.13 line
#   4. reward_terms/height pen ~0 (crouch adopted), assist fade at 60 M, jitter ramps ~80 M+,
#      std anneal in the final third (log_std_clamp -> -1.386)
# Cold-start patience: ep_len 200-400 for the first 30-40 M is NORMAL (cold m3). Judge trends
# after 50 M. The m5_stiff reference: still climbing at 593 M — do NOT early-stop this run for
# looking slow at 300 M.
# Health: crash -> resubmit ONCE per signature (--resume auto continues; the B job in each chain
# already handles the 24 h wall). Second identical crash or NaN -> stop, note for user.
# FINAL: greedy eval N=12 + video; a sprint policy is judged on the 100 m dash (distance, time,
# falls) — frame-sample any gait claim at <= 0.5 s spacing (the RUN 4 lesson).
#
# RUN 5 (2026-08-30): imp_m3d crouch run — CONCLUDED. Crouch adopted (height pen ~0.001 all run),
# swing_min peaked 0.118 @37M then fell back to 0.04-0.05 through hardening; cmd converged 0.20;
# pinned-command endurance 0/5 >= 20 s (REGRESSION vs imp_m3c). See status.log verdict.
#
# CONTEXT: the user reviewed the imp_m3c finals and is right — it is a splayed shuffle-creep in
# ~2 s bouts, not a gait. RUN 5 = preset `walk_fwd_m3_mit_imp4` = imp3 + the two measured levers:
# crouch (height_target_offset_m=0.05: backward toe reach 2.7 -> 20.2 cm, the geometric unlock
# for an alternating cycle) + sustained command holds (cmd_resample_s 4 -> 8 s).
#
# LAUNCH (pending VPN): code is deployed-ready locally (config.py + env.py; smoke ALL OK).
# When jed resolves again:
#   1. deploy: tar cf - walk_mit/config.py walk_mit/env.py | ssh ncmorand@jed.hpc.epfl.ch
#      "cd ~/running_robot && tar xf -" ; verify get_config('walk_fwd_m3_mit_imp4') remotely.
#   2. launch BOTH seeds, 300 M each, warm from imp_m3c_s1 (the best tracker):
#      sbatch --export=ALL,PRESET=walk_fwd_m3_mit_imp4,STEPS=300000000,SEED=<0|1>,NAME=imp_m3d_s<0|1>,WARM=walk_mit/runs/imp_m3c_s1 walk_mit/slurm/jed_train.sbatch
#      (300 M ~ 14 h at ~5,900 fps; sbatch time limit is 1-00:00:00 — fits.)
#   3. verify "log_std reset" + "warm-started" lines in both .out logs, record job IDs in
#      status.log under a RUN 5 header, then tick as below.
#
# Every tick (30 min): same ssh status pull as RUN 4 (progress.csv keys + widen/narrow lines +
# error grep), runs imp_m3d_s0/s1. Signals, in order:
#   1. curriculum/height error shrinking: reward_terms/height pen small AND base z near the
#      CROUCHED target (settled 1.0105 - 0.05 = 0.9605) — is it actually adopting the crouch?
#   2. swing_frac_min: THE gait-quality number this run exists to move. imp_m3c never cleared
#      ~0.08; the crouch makes 0.13+ geometrically possible. If it clears 0.13 the ent gate
#      opens on competence (not deadline) for the first time ever.
#   3. cmd_scale: imp_m3c converged at 0.30 — match or beat at full DR.
#   4. ep_len through assist fade (60M) + full DR (~80M), then the anneal in the final third.
# Health rules: crash -> resubmit ONCE per signature (--resume auto). Second identical crash or
# NaN -> stop, note for the user. No code edits mid-watch.
# Film: past ~150 M, if swing_frac_min > 0.10, pull newest ckpt + film stochastic AND greedy.
# FINAL: greedy eval N=12 + PINNED-COMMAND test (+0.2/+0.4, the real bar) + video; frame-sample
# at <=0.5 s spacing before making ANY gait claim (the RUN 4 lesson). Success = alternating steps
# visible on film + pinned +0.2 tracked >=20 s without falling.
#
# RUN 4 (2026-08-29): imp_m3c watch — CONCLUDED (see status.log verdict + 2026-08-30 correction)
#
# GOAL (user's words): "a walker that works." TWO runs: `imp_m3c_s0` (66275303) and `imp_m3c_s1`
# (66275304), preset `walk_fwd_m3_mit_imp3`, 200 M each (~9.5 h at ~5,700 fps once RUNNING; they
# started PENDING), both warm from imp_m3b_s0 @200M. The preset carries the three walk-not-stand
# levers: achievable command box (0.20/0.10 start -> top-range 0.14 m/s = plant-demonstrated),
# gait_cmd_gate 0.10 + track_sigma_min 0.10 (anti-shuffle billed on ~all move commands, refusal
# pays 14% not 42%), track_yaw_couple (standing under command forfeits the 2.0/step yaw subsidy),
# warmstart_reset_log_std (parent's std 0.25 re-inflated to 1.0).
#
# Every tick (30 min): same ssh + status pull as below, runs imp_m3c_s0/s1. Append one line per
# run to walk_mit/monitor/status.log under the RUN 4 header. Also grep BOTH .out logs once for
# "log_std reset" (must appear — proves the warm-start re-inflation fired) and every tick for
# "command box widened" — cmd_scale > 0 for the first time in any m3 run is THE success signal;
# note the step count when it happens.
#
# Signals, in order of importance:
#   1. cmd_scale climbing off 0.0 (grep widen lines / curriculum/cmd_scale in progress.csv)
#   2. curriculum/cmd_track_err falling toward <= 0.22 at top-of-range
#   3. swing_frac_min rising toward 0.13+ (real stepping, both feet)
#   4. ep_len: expect ~1,000+ early (assist on, easy ramps) then a DIP when assist fades ~60 M —
#      a dip there is NOT a failure; watch whether it recovers WHILE cmd_scale stays > 0
#   5. late: curriculum/log_std_clamp descending toward ln(0.25) = -1.386
# Health rules: crash -> resubmit ONCE per signature with --resume auto (same sbatch line as the
# launch, in status.log header). Second identical crash or NaN -> stop, note for the user. No
# code/config edits mid-watch. Film: if ep_len_mean > 1500 AND cmd_scale > 0 at any tick past
# ~100 M, pull newest ckpt + film one best-of-2 video. At final: greedy eval N=12 LOCAL (login
# node kills long evals) + video + verdict block at top of status.log; success = mean vx > 0.10
# in fresh-env greedy eval with commands active.
#
# NIGHT 2 (2026-08-27/28): imp_m3_long watch — CONCLUDED (see status.log verdict)
#
# ONE run to watch: `imp_m3_long` (job 66248743, 300 M, warm from imp_m2_long final; finishes
# ~09:30-10:00). The m3 budget answer. User asked for the watch until 10:00.
#
# Every tick (30 min): ssh ncmorand@jed.hpc.epfl.ch, pull imp_m3_long's progress.csv head+tail
# (fields: total_timesteps, ep_len_mean, ep_rew_mean, track_lin, dr_scale, swing_frac_min,
# ent_coef, train/std, fps), grep dash-mit-*.out for traceback|fatal|srun: error. Append ONE line
# to walk_mit/monitor/status.log under the NIGHT-2 header. Baseline at 42.8 M: ep_len ~430,
# rew ~140, fall floor ~240.
#
# Health rules: crash -> resubmit ONCE per failure signature
#   sbatch --export=ALL,PRESET=walk_fwd_m3_mit_imp,STEPS=300000000,SEED=2,NAME=imp_m3_long,WARM=walk_mit/runs/imp_m2_long walk_mit/slurm/jed_train.sbatch
#   (--resume auto continues losslessly). Second identical crash or NaN -> stop, note for morning.
#   NO escalation presets for this arm, NO code/config edits, do NOT cancel it for low numbers —
#   it is supposed to look bad until late.
#
# Milestones to annotate: ~80 M = DR/hardening complete (does it break out of the 400s after?);
# ~120 M+ = if ep_len_mean > 1500, fetch newest ckpt + film ONE best-of-2 video (else skip — a
# mid-anneal m3 checkpoint films falls); ~300 M final = full eval (5 eps) + film + 5-line verdict
# at the top of status.log.
#
# Stop: at ~10:00, or earlier if the final is evaled+filmed+summarized. CronDelete the loop job.
#
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
- **`imp_m2_long`** (66248742, 400 M, warm from imp_m2_s1) → `imp_m3_long` (66248743, 300 M) —
  the literature-parity arm (≈23 simulated robot-days, matches Li et al. Cassie): budget > the
  240 M ramp pin, so the efficiency curriculum COMPLETES with ~160 M at full hardening. Runs into
  Thursday; judge it by trend only tonight, never escalate it (it is supposed to look unfinished).

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

**03:00 amendment (imp-eval anomaly):** imp checkpoints currently collapse in any fresh eval env
(see status.log 02:40/03:00 and the imp-eval-anomaly memory) — filming them produces 0.1 s clips.
Until resolved: film the best WARM (mit) arm each video tick, and once per 2 ticks re-try a single
short imp eval (1 episode, no video) to check whether the anomaly persists at newer checkpoints.
Do NOT escalate or cancel imp runs over this — their in-distribution curves remain the A/B signal:

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
