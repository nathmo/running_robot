#!/bin/bash -l
# ANKLE-2 SCREEN (2026-08-05): can the REAL ankle, or no ankle spring at all, be stabilised?
#
#   bash training/slurm/ankle2_launch.sh                    # the m3 screen
#   DRY=1 bash training/slurm/ankle2_launch.sh              # print, submit nothing
#   PHASE=m6 ARMS="bar rigid k350" bash training/slurm/ankle2_launch.sh    # phase B on the winners
#
# Two questions, both MECHANICAL DESIGN decisions rather than control questions:
#   Q1  k = 41.4 N*m/rad (the real spring) with NO preload      -> arm k41_4_np
#   Q2  no spring, a rigid tension-only strut + a 249 g lighter shin -> arm bar
# with controls (k350 = the known-good unbuildable spring, k28_65 = today's robot as modelled,
# rigid = welded) and two attribution arms (k41_4 separates k from preload, bar_heavy separates
# the strut from the mass saving).
#
# JOB SHAPE. Fairshare is ~0.0017 from heavy recent usage, and a long job backfills terribly at that
# priority -- teleop_v4 sat PENDING for 25 h and never got a node. So each run is a chain of SHORT
# jobs that each resume the last (`--resume auto` resolves the newest checkpoint, so a stage killed
# by its own wall clock still hands off). Short links get scheduled in gaps; long ones do not.
set -euo pipefail

SB=training/slurm/izar_train.sbatch
POST=training/slurm/cpg_stage_post.sbatch
NENVS="${NENVS:-18}"
STEPS="${STEPS:-400000000}"
PHASE="${PHASE:-m3}"
PREFIX="${PREFIX:-ankle2}"   # ankle2 | ankle2drv (measured 0.8 Hz drive) | ankle2drv50
SEEDS="${SEEDS:-0 1}"
DRY="${DRY:-}"
LINKS="${LINKS:-4}"
TIME="${TIME:-4:00:00}"

# Ordered so the CONTROLS land first: if k350 does not train on the corrected plant (144.5 N*m
# torque, measured masses, workspace_kill, re-settled stance) then nothing else here is readable,
# and that is the first curve to look at in the morning.
ALL_ARMS="${ALL_ARMS:-k350 bar k28_65 k41_4_np rigid k41_4 bar_heavy}"
ARMS="${ARMS:-$ALL_ARMS}"

nj=0
echo "=== ankle2 screen, phase $PHASE ==="
echo "    arms:  $ARMS"
echo "    seeds: $SEEDS   steps: $STEPS   envs: $NENVS   links: ${LINKS}x${TIME}"
echo
for arm in $ARMS; do
    preset="${PREFIX}_${PHASE}_${arm}"
    for seed in $SEEDS; do
        run="${preset}_s${seed}"
        exp="ALL,PRESET=$preset,STEPS=$STEPS,NENVS=$NENVS,NAME=$run,SEED=$seed"
        if [ -n "$DRY" ]; then
            echo "  [dry] $run   (${LINKS} links x $TIME)"
            nj=$((nj + LINKS + 1))
            continue
        fi
        dep=""
        for _ in $(seq 1 "$LINKS"); do
            jid=$(sbatch --parsable --time="$TIME" --job-name="$run" \
                         ${dep:+--dependency="afterany:$dep"} --export="$exp" "$SB")
            dep=$jid
            nj=$((nj + 1))
        done
        pid=$(sbatch --parsable --dependency="afterany:$dep" --job-name="post_$run" \
                     --export="ALL,RUN=$run" "$POST")
        nj=$((nj + 1))
        echo "  $run  train ...$dep  post $pid"
    done
done
echo
echo "submitted $nj jobs"
