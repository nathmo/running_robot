#!/bin/bash -l
# Launch the FULL m1..m6_sym_gait milestone chain, unattended.
#
#   bash training/slurm/sym_gait_chain_launch.sh
#
# The sym_gait lineage is the ANTI-ONE-LEGGED fix on top of slow_gait: identical plant (freq (0.5,4.0)
# + k=350 ankle + damping 10) plus the duty-symmetry penalty (w_duty_sym=8.0, duty_floor=0.30 — a foot
# whose grounded-fraction EMA sinks below the floor is expensive, so one leg cannot be a passenger) and
# a 4x stronger contact-switch (0.05 -> 0.20) to keep the now-symmetric gait slow. slow_gait solved
# balance/performance but converged one-legged (m3: left foot duty 0%, right patters 6.4 Hz, asymmetry
# 0.99); this is the A/B fix for it. Same obs/action dims as slow_gait, so it warm-chains end to end.
#
# Structure mirrors slow_gait_chain_launch.sh / cpg_chain_launch.sh: cold m1, warm-chained to m6,
# afterany deps + WARM=<dir> hand-off, one afterany post job (plot + eval + cadence + mp4) per stage.
# Runs in parallel with the slow_gait (one-legged baseline) and CPG chains — disjoint presets/nodes.
set -euo pipefail

SB=training/slurm/izar_train.sbatch
POST=training/slurm/cpg_stage_post.sbatch      # generic (takes RUN=), reused as-is
NENVS="${NENVS:-18}"
SEED="${SEED:-0}"

# Same per-stage budget/time as slow_gait, for a clean A/B (m3 the pitch-release rung gets the most).
steps_for() { case "$1" in
    m1) echo 100000000;; m3) echo 250000000;; m6) echo 180000000;; *) echo 150000000;; esac; }
time_for()  { case "$1" in
    m1) echo 12:00:00;;  m3) echo 24:00:00;;  m6) echo 18:00:00;;  *) echo 16:00:00;;  esac; }

echo "=== m1..m6_sym_gait chain (fourier + cadence fixes + duty-symmetry), seed $SEED, $NENVS envs ==="
prev_train=""
prev_run=""
for m in m1 m2 m3 m4 m5 m6; do
    run="${m}_sym_gait"
    st=$(steps_for "$m"); tl=$(time_for "$m")
    args=(--parsable --time="$tl" --job-name="$run")
    exp="ALL,PRESET=${m}_sym_gait,STEPS=$st,NENVS=$NENVS,NAME=$run,SEED=$SEED"
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
