#!/bin/bash -l
# Launch the FULL m1..m6_CPG milestone chain, unattended.
#
#   bash training/slurm/cpg_chain_launch.sh [HOURS_PER_STAGE]      # default 4.5 -> ~27 h total
#
# Six training stages on the CPG gait generator, cold at m1 (only X free) and unlocking one base DOF
# per stage to m6 (fully free). Each stage:
#   * depends on the previous one with `afterany`, so a stage that times out or crashes still hands
#     over instead of stalling the chain on DependencyNeverSatisfied;
#   * warm-starts from the previous stage's run DIRECTORY, which resolves to its newest checkpoint —
#     required, because a wall-clock-stopped stage never writes final_model.zip;
#   * gets its own `afterany` post job (plot + eval + cadence + mp4), so the artefacts exist even
#     for a stage that died. See cpg_stage_post.sbatch for why that must be a separate job.
#
# Obs/action widths are constant across the ladder (same generator, same steer/residual settings),
# so every checkpoint loads into the next stage. Milestones differ only in base_lock + reward extras.
#
# Nothing here needs babysitting: the whole ladder is submitted up front and SLURM sequences it.
set -euo pipefail

H="${1:-4.5}"
SB=training/slurm/izar_train.sbatch
POST=training/slurm/cpg_stage_post.sbatch
STEPS="${STEPS:-60000000}"     # matches _CPG_STAGE_STEPS; the stage completes rather than being cut
NENVS="${NENVS:-18}"
SEED="${SEED:-0}"

# HH:MM:SS from a possibly fractional hour count
hms=$(python3 - "$H" <<'PY'
import sys
h = float(sys.argv[1]); t = int(round(h * 3600))
print("%d:%02d:%02d" % (t // 3600, (t % 3600) // 60, t % 60))
PY
)
echo "=== m1..m6_CPG chain: ${H} h/stage ($hms), ${STEPS} steps/stage, seed $SEED ==="

prev_train=""
prev_run=""
for m in m1 m2 m3 m4 m5 m6; do
    run="${m}_CPG"
    args=(--parsable --time="$hms" --job-name="$run")
    exp="ALL,PRESET=${m}_CPG,STEPS=$STEPS,NENVS=$NENVS,NAME=$run,SEED=$SEED"
    if [ -n "$prev_train" ]; then
        args+=(--dependency="afterany:$prev_train")
        exp="$exp,WARM=training/runs/$prev_run"
    fi
    jid=$(sbatch "${args[@]}" --export="$exp" "$SB")
    pid=$(sbatch --parsable --dependency="afterany:$jid" --job-name="post_$run" \
                 --export="ALL,RUN=$run" "$POST")
    if [ -n "$prev_run" ]; then
        echo "  $run  train $jid (warm <- $prev_run)   post $pid"
    else
        echo "  $run  train $jid (COLD START)          post $pid"
    fi
    prev_train=$jid
    prev_run=$run
done

echo
squeue -u "$USER" -o "%.10i %.14j %.9T %.11l %R"
