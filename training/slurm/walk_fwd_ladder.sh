#!/bin/bash -l
# THE walk_fwd BASE-DOF LADDER (2026-08-09).
#
#   bash training/slurm/walk_fwd_ladder.sh                 # m3 -> m4 -> m5 -> m6, seed 0
#   DRY=1 bash training/slurm/walk_fwd_ladder.sh           # print, submit nothing
#   SEED=1 STAGES="m5 m6" bash training/slurm/walk_fwd_ladder.sh
#
# WHY. Every run in the teleop -> walk_fwd lineage trains at m6 -- all six base DOFs free -- from
# step 0. Checked against every resolved_config in the archive: teleop, teleop_easy, v2, v3, v5,
# walk_fwd, walk_fwd_easy, walk_fwd2 are ALL base_lock (0,0,0,0,0,0). It is the only lineage here
# that skipped the ladder and the only one that will not converge: three plateaus, ~1.4 G steps,
# dr_scale self-limiting at 0.27 because the retreating ramp cannot find competence to ramp on.
#
# Everything that ever worked on this robot used the ladder. 97 archived runs sit at m3, and the
# only controller to reach 2.55 m/s on the MEASURED plant (ankle2_m3_rigid) was m3 with roll and
# yaw locked -- 15.5 s survival in 73 M steps, against 300 M spent going nowhere at m6.
#
# SHAPE. Each rung differs from the next by base_lock and nothing else, so obs (550) and action
# (26) are identical the whole way up and each stage warm-starts the one below by DIRECTORY
# (izar_train.sbatch resolves a dir to its newest checkpoint, so a stage killed by its wall clock
# still hands off). Stage k+1 depends on `afterany` of stage k's LAST link.
#
#   m3  y/roll/yaw locked - sagittal, the configuration that has actually worked
#   m4  + lateral translation
#   m5  + ROLL          <- THE RUNG TO WATCH. The CPG A/B walled here at matched budget.
#   m6  + yaw           == walk_fwd3 exactly, so it is a clean comparison against the control run
#
# One seed: this is a STRUCTURAL probe (does it clear m3? where does it wall?), not an A/B. Add
# seeds once the ladder tells us which rung is the problem.
set -euo pipefail

SB=training/slurm/izar_train.sbatch
STAGES="${STAGES:-m3 m4 m5 m6}"
SEED="${SEED:-0}"
NENVS="${NENVS:-18}"
STEPS="${STEPS:-80000000}"          # per stage; ankle2_m3_rigid reached 15.5 s in 73 M
LINKS="${LINKS:-3}"                 # 4 h each -> 12 h of container per stage; a stage that hits
                                    # STEPS early just exits and its spare links exit immediately
TIME="${TIME:-4:00:00}"
# the 0.8 Hz walker: 340 M steps, completed the drive curriculum, ep_len 4938 / best 6833
WARM0="${WARM0:-training/runs/walk_fwd_easy_warm}"
DRY="${DRY:-}"

echo "=== walk_fwd ladder, seed $SEED ==="
echo "    stages: $STAGES"
echo "    ${LINKS}x${TIME} per stage, ${STEPS} steps, ${NENVS} envs"
echo "    rung 1 warm-starts from $WARM0"
echo

prev_last=""        # last job id of the previous stage
prev_run=""         # previous stage's run dir, the warm start for this one
for m in $STAGES; do
    preset="walk_fwd_${m}"
    name="ladder_${m}_s${SEED}"
    warm="${prev_run:-$WARM0}"
    export_str="ALL,PRESET=${preset},STEPS=${STEPS},NENVS=${NENVS},SEED=${SEED},NAME=${name},WARM=${warm}"

    dep=""
    [ -n "$prev_last" ] && dep="--dependency=afterany:${prev_last}"

    echo "--- $name   (preset $preset, warm <- $warm)"
    if [ -n "$DRY" ]; then
        echo "    sbatch --job-name=$name --time=$TIME $dep --export=$export_str $SB   x$LINKS"
        prev_last="<dry:${name}>"
        prev_run="training/runs/${name}"
        continue
    fi

    jid=$(sbatch --parsable --job-name="$name" --time="$TIME" $dep \
                 --export="$export_str" "$SB")
    echo "    link 1: $jid${dep:+  ($dep)}"
    for i in $(seq 2 "$LINKS"); do
        jid=$(sbatch --parsable --job-name="$name" --time="$TIME" \
                     --dependency=afterany:"$jid" --export="$export_str" "$SB")
        echo "    link $i: $jid (afterany)"
    done
    prev_last="$jid"
    prev_run="training/runs/${name}"
done

echo
echo "submitted. watch:  squeue -u \$USER -o '%.10i %.18j %.2t %.11M'"
echo "the rung to watch is m5 (roll)."
