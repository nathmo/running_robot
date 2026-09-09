#!/bin/bash
# Submit N dependent Izar jobs of izar_train.sbatch; each resumes the newest checkpoint.
#   walk_v2/slurm/izar_chain.sh 3 --export=ALL,PRESET=v2_s1_planar,NAME=v2_s1_planar_s0,SEED=0
set -euo pipefail
N=${1:?usage: izar_chain.sh N [sbatch args...]}
shift || true
DIR="$(cd "$(dirname "$0")" && pwd)"
jid=$(sbatch --parsable "$@" "$DIR/izar_train.sbatch")
echo "job 1: $jid"
for i in $(seq 2 "$N"); do
    jid=$(sbatch --parsable --dependency=afterany:"$jid" "$@" "$DIR/izar_train.sbatch")
    echo "job $i: $jid (afterany chain)"
done
