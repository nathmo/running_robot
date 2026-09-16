#!/bin/bash
# The recipe on Izar: the whole curriculum in ONE run from random weights, three seeds.
#
# There is no planar stage and no checkpoint hand-off. The previous generation needed a 6-DOF base
# spring to hold the torso up while the policy learned to balance, because that robot was a
# point-foot machine that could not stand; this one stands on its own feet, so every episode starts
# from a standing robot on the free plant and the scaffolding is gone.
#
#   RLframework/slurm/izar_dash.sh                     # all three seeds
#   SEEDS=0 RLframework/slurm/izar_dash.sh             # just one, to watch it start
#   PRESET=dash_planar RLframework/slurm/izar_dash.sh
#
# --gres=gpu:2 overrides the sbatch file's gpu:1. STEPS stays unset: the preset owns the budget
# (450 M steps).
set -euo pipefail

# Self-locating: the repo is two levels above this script, so the launcher works from whichever
# directory the clone lives in rather than from one hard-coded path.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
mkdir -p RLframework/logs

PRESET="${PRESET:-dash}"
COMMON="DEVICES=2,NENVS=4096,NSTEPS=9,REPO=$REPO"

for s in ${SEEDS:-0 1 2}; do
    n="${PRESET}_s$s"
    j=$(sbatch --parsable --gres=gpu:2 \
        --export=ALL,PRESET="$PRESET",NAME="$n",SEED="$s","$COMMON" \
        RLframework/slurm/izar_train.sbatch)
    echo "$n = $j"
done
squeue -u "$USER" -o "%.9i %.16j %.4t %.10M %.6D %R"
