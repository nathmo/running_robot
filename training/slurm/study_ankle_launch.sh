#!/bin/bash -l
# ANKLE-SPRING STUDY: is the passive foot spring useful, must it be actuated, is there an optimum?
#
#   bash training/slurm/study_ankle_launch.sh              # phase A: the m3 sweep (33 jobs)
#   DRY=1 bash training/slurm/study_ankle_launch.sh        # print what it would submit, submit nothing
#   PHASE=m6 ARMS="k350 active free" bash training/slurm/study_ankle_launch.sh   # phase B
#
# Phase A sweeps all 11 arms x 3 seeds at m3, COLD (no warm start anywhere). Cold is the point: the
# 2026-07-24 sweep warm-started every arm from a policy trained on the SOFT spring, so a stiff arm
# was being asked to inherit a gait tuned for a different plant, and "k350 wins" could partly have
# been "k350 transfers". Every arm here starts from scratch on its own plant.
#
# Phase B is NOT chained to phase A on purpose. Which arms deserve the expensive m6 confirmation is
# a judgement call that needs the phase-A curves in front of a human -- run study_analyze.py, look
# at the ranking, then launch phase B with the arms you actually want. study_analyze.py prints the
# exact command.
set -euo pipefail

SB=training/slurm/izar_train.sbatch
POST=training/slurm/cpg_stage_post.sbatch        # generic (takes RUN=), reused as-is
NENVS="${NENVS:-18}"
STEPS="${STEPS:-400000000}"
PHASE="${PHASE:-m3}"
SEEDS="${SEEDS:-0 1 2}"
DRY="${DRY:-}"
# 400 M at m3 does not fit the 3-day `gpu` QOS in one job (m3_sym_gait needed ~24 h for 250 M), so
# each run gets a second job depending on afterany with the SAME NAME -- train.py's `--resume auto`
# picks up the checkpoint and continues. Two links is enough for 400 M with margin.
LINKS="${LINKS:-2}"
TIME="${TIME:-2-23:30:00}"

ALL_ARMS="rigid free k28_65 k90 k200 k350 k550 k750 k1100 active active_k350"
ARMS="${ARMS:-$ALL_ARMS}"

nj=0
echo "=== ankle-spring study, phase $PHASE ==="
echo "    arms:  $ARMS"
echo "    seeds: $SEEDS      steps: $STEPS      envs: $NENVS      links: $LINKS"
echo
for arm in $ARMS; do
    preset="study_${PHASE}_${arm}"
    for seed in $SEEDS; do
        run="${preset}_s${seed}"
        exp="ALL,PRESET=$preset,STEPS=$STEPS,NENVS=$NENVS,NAME=$run,SEED=$seed"
        if [ -n "$DRY" ]; then
            echo "  [dry] $run   ($LINKS links x $TIME)"
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
[ -n "$DRY" ] && echo "(DRY run -- nothing was actually submitted)"
echo
echo "next:  python training/study_analyze.py --runs training/runs"
[ -n "$DRY" ] || squeue -u "$USER" -o "%.10i %.28j %.9T %.11l %R" | head -20
