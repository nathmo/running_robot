#!/bin/bash
# Submit N dependent Lyra jobs of lyra_train.sbatch; each resumes the previous one's newest
# checkpoint (--resume auto). afterany, not afterok: a link killed by the wall clock or by a
# beta-phase interruption must still trigger its successor.
#
#   walk_v2/slurm/lyra_chain.sh 3 --export=ALL,PRESET=v2_s2_free,NAME=v2_s2_free_s0
set -euo pipefail
N=${1:?usage: lyra_chain.sh N [sbatch args...]}
shift || true
DIR="$(cd "$(dirname "$0")" && pwd)"
jid=$(sbatch --parsable "$@" "$DIR/lyra_train.sbatch")
echo "job 1: $jid"
for i in $(seq 2 "$N"); do
    jid=$(sbatch --parsable --dependency=afterany:"$jid" "$@" "$DIR/lyra_train.sbatch")
    echo "job $i: $jid (afterany chain)"
done
echo "monitor: squeue -u \$USER    logs: tail -f dash-v2-<jobid>.out"
