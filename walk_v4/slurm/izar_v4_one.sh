#!/bin/bash
# v4_one on Izar: the whole curriculum in ONE run from random weights. The 6-DOF base spring stands
# in for the planar stage 1 and its checkpoint; lane is off; heading is billed on the robot's own
# averaged estimate. Three seeds, no chaining, no warm start -- judge them against the two-stage
# no-lane runs (v4_nolane_s*) at matched steps with tools/compare_runs.py.
#
#   walk_v4/slurm/izar_v4_one.sh                     # from the repo root on the login node
#   PRESET=v4_one_runstop walk_v4/slurm/izar_v4_one.sh
#
# --gres=gpu:2 overrides the sbatch file's gpu:1, as for every v3/v4 job. STEPS stays unset: the
# preset owns the budget (450 M = the three stages it replaces).
set -euo pipefail
cd "$HOME/running_robot"
COMMON="DEVICES=2,NENVS=4096,NSTEPS=9"
PRESET=${PRESET:-v4_one}
for s in 0 1 2; do
    n="${PRESET}_s$s"
    j=$(sbatch --parsable --gres=gpu:2 \
         --export=ALL,PRESET=$PRESET,NAME=$n,SEED=$s,$COMMON \
         walk_v4/slurm/izar_train.sbatch)
    echo "$n = $j"
done
squeue -u "$USER" -o "%.9i %.16j %.4t %.10M %.6D %R"
