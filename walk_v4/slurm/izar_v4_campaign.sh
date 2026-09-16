#!/bin/bash
# The v4 campaign on Izar: does the policy need the reflexes, or only their authority?
#
# Two arms x three seeds. Stage 1 (planar: the pitch-reflex question) from random weights; stage 2
# (free plant: the roll-reflex question) chained on each stage-1 job with afterok and warm-started
# from THAT seed's best.msgpack (never final -- see walk_v4/README.md). Stage 3 is submitted by
# hand after tools/compare_runs.py has ranked the stage-2 seeds.
#
#   walk_v4/slurm/izar_v4_campaign.sh            # from the repo root on the login node
#
# --gres=gpu:2 on the command line overrides the sbatch file's gpu:1, which is how every v3 job was
# sized (sacct: gres/gpu=2, cpu=10, mem=90G). STEPS stays unset: the preset owns the budget.
set -euo pipefail
cd "$HOME/running_robot"
COMMON="DEVICES=2,NENVS=4096,NSTEPS=9"
for arm in "" _reflex; do
    for s in 0 1 2; do
        n1="v4_stage1${arm}_s$s"
        j1=$(sbatch --parsable --gres=gpu:2 \
             --export=ALL,PRESET=v4_stage1${arm},NAME=$n1,SEED=$s,$COMMON \
             walk_v4/slurm/izar_train.sbatch)
        n2="v4${arm}_s$s"
        j2=$(sbatch --parsable --gres=gpu:2 --dependency=afterok:"$j1" \
             --export=ALL,PRESET=v4${arm},NAME=$n2,SEED=$s,$COMMON,WARM=$HOME/running_robot/walk_v4/runs/$n1/best.msgpack \
             walk_v4/slurm/izar_train.sbatch)
        echo "stage 1 $n1 = $j1   ->   stage 2 $n2 = $j2 (afterok)"
    done
done
squeue -u "$USER" -o "%.9i %.12j %.4t %.10M %.6D %R"
