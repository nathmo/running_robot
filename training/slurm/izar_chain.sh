#!/bin/bash
# Submit a chain of N dependent Izar training jobs. Each job runs the same sbatch script with
# --resume auto, so job k+1 continues exactly where job k stopped (checkpoints every ~1M steps).
# afterany (not afterok): a job killed by the 3-day wall clock must still trigger its successor.
#
#   training/slurm/izar_chain.sh 2
#   training/slurm/izar_chain.sh 3 --export=ALL,PRESET=m3_sprint,STEPS=240000000
set -euo pipefail
N=${1:?usage: izar_chain.sh N [extra sbatch args, e.g. --export=ALL,PRESET=m2_sprint]}
shift || true
DIR="$(cd "$(dirname "$0")" && pwd)"
jid=$(sbatch --parsable "$@" "$DIR/izar_train.sbatch")
echo "job 1: $jid"
for i in $(seq 2 "$N"); do
    jid=$(sbatch --parsable --dependency=afterany:"$jid" "$@" "$DIR/izar_train.sbatch")
    echo "job $i: $jid (afterany chain)"
done
echo "monitor: squeue -u \$USER    logs: tail -f dash-sprint-<jobid>.out"
