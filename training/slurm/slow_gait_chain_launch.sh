#!/bin/bash -l
# Launch the FULL m1..m6_slow_gait milestone chain, unattended.
#
#   bash training/slurm/slow_gait_chain_launch.sh
#
# The slow_gait lineage RETRAINS the whole base-DOF ladder from scratch with the two cadence root-
# cause fixes baked into ONE identical plant from step 0 (see the mX_slow_gait block in config.py):
#   gait_freq_hz=(0.5, 4.0)  — neutral action ~2 Hz, not the 25 Hz the 200 Hz (0.5,50) rescale gave
#   ankle_stiffness=350.0    — the preload-preserving stiff spring that solved m3 pitch balance
#   ankle_damping=10.0       — ~55% critical; kills the 6.3 Hz ankle-spring ring
#   w_contact_switch=0.05    — gentle from-scratch nudge to fewer, longer steps
# A warm-start CANNOT unlearn the fast-stepping chatter, which is why this is a from-scratch chain.
#
# Structure mirrors cpg_chain_launch.sh (the proven pattern): cold at m1, unlock one base DOF per
# stage to m6, each stage `afterany` on the previous (hands over on timeout/crash instead of stalling
# the chain) and warm-starting from the previous run DIRECTORY (resolves to its newest checkpoint,
# because a wall-clock-stopped stage never writes final_model.zip). Every stage gets its own afterany
# post job (plot + greedy eval + cadence diagnostic + mp4) via the generic cpg_stage_post.sbatch, so
# the video and training plot exist for each rung — not just the last.
#
# Obs/action widths are constant across the ladder (fourier generator, no steering, privileged vel),
# so every checkpoint loads into the next stage. Nothing here needs babysitting: the whole ladder is
# submitted up front and SLURM sequences it. Runs in parallel with the CPG chain (different presets,
# run names, and nodes — no conflict).
set -euo pipefail

SB=training/slurm/izar_train.sbatch
POST=training/slurm/cpg_stage_post.sbatch      # generic (takes RUN=), reused as-is
NENVS="${NENVS:-18}"
SEED="${SEED:-0}"

# Per-stage step budget + wall-time cap. Throughput is ~15 M steps/h on one V100 (measured: m7_freq
# did 300 M in 19.6 h), so each stage EXITS at its STEPS well under --time; --time only bounds a hang.
# m3 (the pitch release) is the hard rung and gets the most; m1 (X-only, Z railed) the least.
steps_for() { case "$1" in
    m1) echo 100000000;; m3) echo 250000000;; m6) echo 180000000;; *) echo 150000000;; esac; }
time_for()  { case "$1" in
    m1) echo 12:00:00;;  m3) echo 24:00:00;;  m6) echo 18:00:00;;  *) echo 16:00:00;;  esac; }

echo "=== m1..m6_slow_gait chain (fourier + cadence fixes), seed $SEED, $NENVS envs ==="
prev_train=""
prev_run=""
for m in m1 m2 m3 m4 m5 m6; do
    run="${m}_slow_gait"
    st=$(steps_for "$m"); tl=$(time_for "$m")
    args=(--parsable --time="$tl" --job-name="$run")
    exp="ALL,PRESET=${m}_slow_gait,STEPS=$st,NENVS=$NENVS,NAME=$run,SEED=$SEED"
    if [ -n "$prev_train" ]; then
        args+=(--dependency="afterany:$prev_train")
        exp="$exp,WARM=training/runs/$prev_run"
    fi
    jid=$(sbatch "${args[@]}" --export="$exp" "$SB")
    pid=$(sbatch --parsable --dependency="afterany:$jid" --job-name="post_$run" \
                 --export="ALL,RUN=$run" "$POST")
    if [ -n "$prev_run" ]; then
        echo "  $run  train $jid (warm <- $prev_run)   post $pid   $st steps / $tl"
    else
        echo "  $run  train $jid (COLD START)          post $pid   $st steps / $tl"
    fi
    prev_train=$jid
    prev_run=$run
done

echo
squeue -u "$USER" -o "%.10i %.16j %.9T %.11l %R"
