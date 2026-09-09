#!/bin/bash
# DASH-01 Walker v2 on JED: the two-stage curriculum (artifact §11), two seeds, chained per seed.
#   S1 "planar" (v2_s1: x, z, pitch free) 300 M cold  ->  S2 "free" (v2_s2) 300 M warm from S1.
# Submit from the repo root on the cluster:  bash walk_mit/slurm/launch_v2.sh
# A short throughput/parity job first (BENCH=1) runs smoke_test + golden check + bench_v2 on one
# node and exits -- the "does the CPU version run on the cluster" gate.
set -euo pipefail
cd "${REPO:-$HOME/running_robot}"
STEPS="${STEPS:-300000000}"
if [ "${BENCH:-0}" = "1" ]; then
    sbatch --parsable --job-name=dash-v2-bench --time=00:40:00 \
        --export=ALL,PRESET=v2_s1,STEPS=2000000,SEED=0,NAME=v2_bench walk_mit/slurm/jed_v2_bench.sbatch
    exit 0
fi
S1S0=$(sbatch --parsable --export=ALL,PRESET=v2_s1,STEPS=$STEPS,SEED=0,NAME=v2_s1_s0 walk_mit/slurm/jed_train.sbatch)
S1S1=$(sbatch --parsable --export=ALL,PRESET=v2_s1,STEPS=$STEPS,SEED=1,NAME=v2_s1_s1 walk_mit/slurm/jed_train.sbatch)
S2S0=$(sbatch --parsable --dependency=afterany:$S1S0 --export=ALL,PRESET=v2_s2,STEPS=$STEPS,SEED=0,NAME=v2_s2_s0,WARM=walk_mit/runs/v2_s1_s0 walk_mit/slurm/jed_train.sbatch)
S2S1=$(sbatch --parsable --dependency=afterany:$S1S1 --export=ALL,PRESET=v2_s2,STEPS=$STEPS,SEED=1,NAME=v2_s2_s1,WARM=walk_mit/runs/v2_s1_s1 walk_mit/slurm/jed_train.sbatch)
echo "v2 chains: s0 S1=$S1S0 -> S2=$S2S0   s1 S1=$S1S1 -> S2=$S2S1"
