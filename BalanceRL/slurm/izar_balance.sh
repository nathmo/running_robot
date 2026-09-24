#!/bin/bash
# Launch the balance campaign on Izar: SEEDS x LINKS chained jobs (each link resumes the last).
#
#   BalanceRL/slurm/izar_balance.sh                  # seeds 0 1 2, 4 links of 4.5 h each
#   SEEDS=0 LINKS=1 BalanceRL/slurm/izar_balance.sh  # one link of one seed, to watch it start
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
mkdir -p BalanceRL/logs
PREFIX="${PREFIX:-bal}"
for s in ${SEEDS:-0 1 2}; do
    n="${PREFIX}_s$s"
    dep=""
    for l in $(seq 1 "${LINKS:-4}"); do
        j=$(sbatch --parsable $dep --job-name="$n" \
            --export=ALL,REPO="$REPO",NAME="$n",SEED="$s",PRESET="${PRESET:-balance}",EXTRA="${EXTRA:-}" \
            BalanceRL/slurm/izar_balance.sbatch)
        echo "$n link $l = $j"
        dep="--dependency=afterany:$j"
    done
done
squeue -u "$USER" -o "%.9i %.16j %.4t %.10M %.6D %R"
